"""
Smoke test for ExperimentRunner.

Runs a minimal experiment end-to-end to verify the pipeline works.
"""

import pytest
import tempfile
from pathlib import Path

from src.exp.config import ExperimentCfg
from src.exp.runner import ExperimentRunner


@pytest.fixture
def minimal_config():
    """Create minimal valid config for testing."""
    return ExperimentCfg.model_validate({
        "experiment_id": "test_smoke",
        "seed": 12345,
        "model": {
            "name": "BS",
            "params": {"S0": 1.0, "r": 0.02, "sigma": 0.2}
        },
        # Grid must be large enough that the non-extrapolation interior
        # (K in [0.85, 1.15]*S0, T <= 0.8) still yields val >= 1 after the
        # 70/10/10 split; smaller grids crash the val DataLoader with val=0.
        "grid": {"K": 20, "T": 6, "spacing": "uniform"},
        "net": {"L": 2, "W": 16, "activation": "softplus"},
        "optim": {
            "epochs": 3,  # Very short for smoke test
            "batch_data": 5,
            "batch_collocation": 10,
            "amp": False,  # Disable AMP for CPU
            "early_stop_patience": None,  # Disable early stopping
        },
        "eval": {
            "metrics": ["rmse_price"],
            "test_grid": "heldout"
        }
    })


def test_runner_smoke(minimal_config):
    """Run minimal experiment end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        
        # Create runner
        runner = ExperimentRunner(minimal_config, run_dir=run_dir, device="cpu")
        
        # Prepare
        runner.prepare()
        
        assert runner.model is not None
        assert runner.train_data is not None
        assert len(runner.train_data) > 0
        
        # Train (very short)
        runner.train_and_eval()
        
        assert runner.history is not None
        assert len(runner.history) > 0
        
        # Report
        metrics = runner.report()
        
        assert 'rmse_price' in metrics
        assert metrics['rmse_price'] >= 0

        # Timing metrics
        assert metrics['train_time_s'] > 0
        assert metrics['inference_latency_ms'] > 0
        assert metrics['n_epochs'] == len(runner.history)
        assert metrics['epoch_time_ms'] > 0

        # The restored checkpoint comes from the final curriculum stage
        assert metrics['best_epoch'] >= metrics['final_stage_start']

        # Check artifacts exist
        assert (run_dir / "cfg.json").exists()
        assert (run_dir / "model.pt").exists()
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "summary.json").exists()


def test_early_stopping_restricted_to_final_stage(minimal_config):
    """Early stopping must not end training before the final curriculum stage.

    With a three-stage curriculum and 6 epochs the final stage starts at
    epoch 4. Even with patience 1 the run has to reach that stage, and the
    restored checkpoint must come from it (otherwise a model reported as
    Greek/PDE-trained could be a price-only network).
    """
    cfg = minimal_config.model_copy(deep=True)
    cfg.optim.epochs = 6
    cfg.optim.early_stop_patience = 1
    cfg.loss.curriculum = ["price", "price+greeks", "price+greeks+pde"]
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(cfg, run_dir=Path(tmpdir) / "run", device="cpu")
        runner.prepare()
        runner.train_and_eval()
        metrics = runner.report()

        assert metrics['final_stage_start'] == 4
        assert len(runner.history) >= 5
        assert metrics['best_epoch'] >= 4


def test_runner_reproducibility(minimal_config):
    """Test that same config produces same results (deterministic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run
        run_dir1 = Path(tmpdir) / "run1"
        runner1 = ExperimentRunner(minimal_config, run_dir=run_dir1, device="cpu")
        runner1.prepare()
        runner1.train_and_eval()
        metrics1 = runner1.report()
        
        # Second run (same config)
        run_dir2 = Path(tmpdir) / "run2"
        runner2 = ExperimentRunner(minimal_config, run_dir=run_dir2, device="cpu")
        runner2.prepare()
        runner2.train_and_eval()
        metrics2 = runner2.report()
        
        # Metrics should be close (allowing for minor numerical differences)
        assert abs(metrics1['rmse_price'] - metrics2['rmse_price']) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


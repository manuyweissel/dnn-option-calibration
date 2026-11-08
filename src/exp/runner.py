"""
Experiment runner: orchestrates training, evaluation, and reporting.

The ExperimentRunner is the main entry point for running experiments.
It handles data generation, model training, evaluation, and artifact saving.

Supports multi-seed execution for statistical confidence.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentCfg
from .io_utils import save_config, compute_config_hash, save_artifact
from .logging_utils import setup_logging, log_system_info
from .metrics import compute_all_metrics
from .registry import get_model
from .seeds import seed_everything

from ..data.grids import create_grid, split_data, create_extrapolation_grid
from ..data.synth import generate_synthetic_data
from ..data.noise import apply_noise_config
from ..models.nets import MLP
from ..train.engine import TrainingEngine


class ExperimentRunner:
    """
    Orchestrates a complete option pricing experiment.
    
    Workflow:
    1. prepare(): Load config, set seeds, generate data, create model
    2. train_and_eval(): Train model, evaluate on test set
    3. report(): Generate summary, figures, and tables
    
    Attributes:
        cfg: Experiment configuration.
        run_dir: Run output directory.
        logger: Logger instance.
        device: Torch device.
        
    Example:
        >>> from .io_utils import load_config
        >>> cfg = load_config("configs/bs_softplus_full.json")
        >>> runner = ExperimentRunner(cfg)
        >>> runner.prepare()
        >>> runner.train_and_eval()
        >>> runner.report()
    """
    
    def __init__(
        self,
        cfg: ExperimentCfg,
        run_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize experiment runner.
        
        Args:
            cfg: Experiment configuration.
            run_dir: Custom run directory (None = auto-generate).
            device: Device ('cpu', 'cuda', or None for auto-detect).
        """
        self.cfg = cfg
        
        # Create run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path("runs") / f"{timestamp}__{cfg.experiment_id}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Set up logging
        log_file = self.run_dir / "logs" / "run.log"
        self.logger = setup_logging(log_file, log_name=cfg.experiment_id)
        self.logger.info(f"Initializing experiment: {cfg.experiment_id}")
        self.logger.info(f"Run directory: {self.run_dir}")
        log_system_info(self.logger)
        
        # Placeholders
        self.model: Optional[torch.nn.Module] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.extrap_data: Optional[pd.DataFrame] = None
        self.history: Optional[pd.DataFrame] = None
        self.pricer = None
        self.train_time_s: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.final_stage_start: Optional[int] = None

    def prepare(self) -> None:
        """
        Prepare experiment: seeds, data, model.
        
        Steps:
        1. Set random seeds
        2. Generate synthetic data
        3. Add noise (if configured)
        4. Split into train/val/test
        5. Create neural network
        """
        self.logger.info("=== Preparation Phase ===")

        if self.cfg.model.name == "Heston" and self.cfg.loss.lambda_pde > 0:
            raise ValueError(
                "Heston model does not support an exact scalar (K,T)-only PDE "
                "residual. Set lambda_pde=0 for Heston experiments (Ch8 sec 8.2)."
            )

        try:
            from src.train.collocation import reset_sobol_cache
            reset_sobol_cache()
        except ImportError:
            pass

        seed_everything(self.cfg.seed, deterministic=True)
        self.logger.info(f"Random seed set to {self.cfg.seed}")
        
        # Create pricer
        pricer_class = get_model(self.cfg.model.name)
        self.pricer = pricer_class(**self.cfg.model.params)
        self.logger.info(f"Pricer: {self.cfg.model.name} with params {self.cfg.model.params}")
        
        # Generate grid
        grid = create_grid(
            K=self.cfg.grid.K,
            T=self.cfg.grid.T,
            spacing=self.cfg.grid.spacing,
            K_min=self.cfg.grid.K_min,
            K_max=self.cfg.grid.K_max,
            T_min=self.cfg.grid.T_min,
            T_max=self.cfg.grid.T_max,
            S0=self.cfg.model.params.get("S0", 1.0),
        )
        self.logger.info(f"Grid created: {len(grid)} points ({self.cfg.grid.K} × {self.cfg.grid.T})")
        
        # Generate synthetic data
        data = generate_synthetic_data(
            grid, self.cfg.model.name, self.cfg.model.params
        )
        self.logger.info("Synthetic data generated")
        
        # Add noise using noise config
        noise_cfg = {
            'price_rel': self.cfg.noise.price_rel,
            'greek_rel': self.cfg.noise.greek_rel,
        }
        if self.cfg.noise.price_rel > 0 or self.cfg.noise.greek_rel > 0:
            data = apply_noise_config(
                data, noise_cfg,
                seed=self.cfg.noise.seed or self.cfg.seed,
            )
            self.logger.info(
                f"Noise added: price={self.cfg.noise.price_rel}, greek={self.cfg.noise.greek_rel}"
            )
        
        S0 = self.cfg.model.params.get("S0", 1.0)
        train_df, val_df, test_df, extrap_df = split_data(
            data, seed=self.cfg.seed, S0=S0,
        )
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        self.extrap_data = extrap_df if len(extrap_df) > 0 else None
        self.logger.info(
            f"Data split: train={len(train_df)}, val={len(val_df)}, "
            f"test_interp={len(test_df)}, test_extrap={len(extrap_df)}"
        )
        
        # Create model
        self.model = MLP(
            L=self.cfg.net.L,
            W=self.cfg.net.W,
            activation=self.cfg.net.activation,
            head=self.cfg.net.head,
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"Model created: {self.cfg.net.L} layers × {self.cfg.net.W} units, "
            f"{n_params} parameters, activation={self.cfg.net.activation}"
        )
        
        # Save config
        config_path = self.run_dir / "cfg.json"
        save_config(self.cfg, config_path)
        cfg_hash = compute_config_hash(self.cfg)
        self.logger.info(f"Config saved: {config_path} (hash={cfg_hash})")
    
    def train_and_eval(self) -> None:
        """
        Train model and evaluate on test set.
        
        Steps:
        1. Create DataLoaders
        2. Initialize TrainingEngine with model-specific PDE config
        3. Train with curriculum and early stopping
        4. Evaluate on test/extrap sets
        """
        self.logger.info("=== Training Phase ===")
        
        # Create DataLoaders
        train_loader = self._create_dataloader(self.train_data, batch_size=self.cfg.optim.batch_data)
        val_loader = self._create_dataloader(self.val_data, batch_size=self.cfg.optim.batch_data)
        
        # Get grid ranges for collocation
        K_min = self.train_data['K'].min()
        K_max = self.train_data['K'].max()
        T_min = self.train_data['T'].min()
        T_max = self.train_data['T'].max()
        
        # Get model params for PDE
        r = self.cfg.model.params.get("r", 0.02)
        S0 = self.cfg.model.params.get("S0", 1.0)
        model_type = self.cfg.model.name

        import torch as _torch

        local_vol_fn = None
        heston_params = None

        if model_type == "LocalVol":
            sigma0 = self.cfg.model.params.get("sigma0", 0.2)
            alpha = self.cfg.model.params.get("alpha", 0.0)
            beta = self.cfg.model.params.get("beta", 0.0)
            _S0 = float(S0)

            def local_vol_fn(K: _torch.Tensor, T: _torch.Tensor) -> _torch.Tensor:
                return sigma0 * (K / _S0) ** alpha * (1.0 + beta * T)

            sigma = sigma0
            self.logger.info(
                f"LocalVol PDE: sigma0={sigma0}, alpha={alpha}, beta={beta}"
            )
        elif model_type == "Heston":
            heston_params = {
                k: self.cfg.model.params[k]
                for k in ("kappa", "theta", "xi", "rho", "v0")
                if k in self.cfg.model.params
            }
            sigma = self.cfg.model.params.get("sigma", 0.2)
            self.logger.info(f"Heston PDE params: {heston_params}")
        else:
            sigma = self.cfg.model.params.get("sigma", 0.2)

        # Training engine with model-specific PDE dispatch
        engine = TrainingEngine(
            model=self.model,
            optim_cfg=self.cfg.optim,
            loss_cfg=self.cfg.loss,
            device=self.device,
            logger=self.logger,
            model_type=model_type,
            local_vol_fn=local_vol_fn,
            heston_params=heston_params,
        )
        
        # Train
        t0 = time.perf_counter()
        self.history = engine.train(
            train_loader=train_loader,
            val_loader=val_loader,
            K_range=(K_min, K_max),
            T_range=(T_min, T_max),
            r=r,
            sigma=sigma,
            S0=S0,
        )
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.train_time_s = time.perf_counter() - t0
        self.best_epoch = engine.best_epoch
        self.final_stage_start = engine.final_stage_start

        self.logger.info(f"Training completed in {self.train_time_s:.1f}s")
        
        # Save checkpoint
        checkpoint_path = self.run_dir / "model.pt"
        engine.save_checkpoint(checkpoint_path)
        self.logger.info(f"Model saved: {checkpoint_path}")
        
        # Save training history
        history_path = self.run_dir / "metrics.csv"
        self.history.to_csv(history_path, index=False)
        self.logger.info(f"Training history saved: {history_path}")
    
    def report(self) -> dict:
        """
        Generate evaluation report with metrics and figures.
        
        Returns:
            Dict with final metrics.
        """
        self.logger.info("=== Reporting Phase ===")
        
        # Evaluate on test set (before any device moves)
        test_pred = self._predict(self.test_data)
        
        extrap_pred = None
        if self.extrap_data is not None and len(self.extrap_data) > 0:
            extrap_pred = self._predict(self.extrap_data)

        # Parameter recovery (moves model to CPU internally, then restores)
        params_pred = self._recover_params()
        params_true = self._get_true_params()

        metrics = compute_all_metrics(
            df_pred=test_pred,
            df_true=self.test_data,
            params_pred=params_pred,
            params_true=params_true,
        )

        if extrap_pred is not None:
            metrics['rmse_extrap'] = compute_all_metrics(
                extrap_pred, self.extrap_data
            )['rmse_price']

        # PDE/arb diagnostics on CPU with gradients explicitly enabled
        try:
            from src.exp.metrics import pde_residual_metric, arbitrage_diagnostics
            K_test = self.test_data['K'].values
            T_test = self.test_data['T'].values
            r = self.cfg.model.params.get("r", 0.02)
            if self.cfg.model.name == "LocalVol":
                sigma = self.cfg.model.params.get("sigma0", 0.2)
            else:
                sigma = self.cfg.model.params.get("sigma", 0.2)
            self.model.cpu()
            self.model.eval()
            with torch.enable_grad():
                pde_m = pde_residual_metric(
                    self.model, K_test, T_test, r, sigma,
                    model_type=self.cfg.model.name,
                )
                metrics.update(pde_m)
                arb_m = arbitrage_diagnostics(self.model, K_test, T_test)
                metrics.update(arb_m)
        except Exception as e:
            self.logger.warning(f"PDE/arb diagnostics failed: {e}")
        finally:
            self.model.to(self.device)

        # Timing metrics: training wall-clock and inference latency
        if self.train_time_s is not None:
            n_epochs = len(self.history) if self.history is not None else 0
            metrics['train_time_s'] = float(self.train_time_s)
            metrics['n_epochs'] = float(n_epochs)
            if n_epochs > 0:
                metrics['epoch_time_ms'] = float(self.train_time_s * 1000.0 / n_epochs)
        if self.final_stage_start is not None:
            metrics['final_stage_start'] = float(self.final_stage_start)
        if self.best_epoch is not None:
            metrics['best_epoch'] = float(self.best_epoch)
        try:
            metrics['inference_latency_ms'] = self._benchmark_inference()
        except Exception as e:
            self.logger.warning(f"Inference benchmark failed: {e}")

        self.logger.info(f"Test metrics: {metrics}")
        
        # Save summary
        summary = {
            "experiment_id": self.cfg.experiment_id,
            "config_hash": compute_config_hash(self.cfg),
            "seed": self.cfg.seed,
            "model": self.cfg.model.name,
            "metrics": metrics,
            "best_epoch": self.best_epoch,
            "final_stage_start": self.final_stage_start,
            "last_epoch": int(self.history['epoch'].iloc[-1]) if self.history is not None else 0,
        }
        
        summary_path = self.run_dir / "summary.json"
        save_artifact(summary, summary_path, format="json")
        self.logger.info(f"Summary saved: {summary_path}")
        
        # Generate figures
        self._generate_figures(test_pred, self.test_data)
        
        self.logger.info("=== Experiment Complete ===")
        
        # Close logger handlers to release file locks (important for Windows)
        self.cleanup_logger()
        
        return metrics
    
    def cleanup_logger(self) -> None:
        """Close and remove all logger handlers to release file locks."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def _create_dataloader(
        self, df: pd.DataFrame, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader from DataFrame."""
        K = torch.tensor(df['K'].values, dtype=torch.float32).unsqueeze(1)
        T = torch.tensor(df['T'].values, dtype=torch.float32).unsqueeze(1)
        price = torch.tensor(df['price'].values, dtype=torch.float32).unsqueeze(1)
        
        tensors = {'K': K, 'T': T, 'price': price}

        if 'delta' in df.columns:
            tensors['delta'] = torch.tensor(df['delta'].values, dtype=torch.float32).unsqueeze(1)
        if 'gamma' in df.columns:
            tensors['gamma'] = torch.tensor(df['gamma'].values, dtype=torch.float32).unsqueeze(1)
        if 'theta' in df.columns:
            tensors['theta'] = torch.tensor(df['theta'].values, dtype=torch.float32).unsqueeze(1)
        if 'vega' in df.columns:
            tensors['vega'] = torch.tensor(df['vega'].values, dtype=torch.float32).unsqueeze(1)
        
        # Create TensorDataset and DataLoader
        dataset = TensorDataset(*tensors.values())
        
        # Custom collate to return dict
        def collate_fn(batch):
            batch_dict = {}
            for i, key in enumerate(tensors.keys()):
                batch_dict[key] = torch.stack([item[i] for item in batch])
            return batch_dict
        
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
        
        return loader
    
    def _get_true_params(self) -> dict[str, float] | None:
        """Return the ground-truth calibratable parameters for the model."""
        p = self.cfg.model.params
        name = self.cfg.model.name
        if name == "BS":
            return {"sigma": p.get("sigma", 0.2)}
        elif name == "LocalVol":
            return {"sigma0": p.get("sigma0", 0.2), "alpha": p.get("alpha", 0.5)}
        return None

    def _recover_params(self) -> dict[str, float] | None:
        """Recover model parameters via Dupire extraction on the trained net."""
        from src.models.dupire import recover_bs_sigma, recover_lv_params

        name = self.cfg.model.name
        K = self.test_data['K'].values
        T = self.test_data['T'].values
        r = self.cfg.model.params.get("r", 0.02)
        S0 = self.cfg.model.params.get("S0", 1.0)

        self.model.cpu()
        try:
            if name == "BS":
                sigma_hat = recover_bs_sigma(self.model, K, T, r=r)
                return {"sigma": sigma_hat}
            elif name == "LocalVol":
                return recover_lv_params(self.model, K, T, S0=S0, r=r)
        except Exception as e:
            self.logger.warning(f"Parameter recovery failed: {e}")
        finally:
            self.model.to(self.device)
        return None

    def _benchmark_inference(
        self, n_warmup: int = 10, n_repeats: int = 100
    ) -> float:
        """Median forward-pass latency (ms) on the full test grid.

        Consumes no RNG (deterministic MLP forward under no_grad), so
        downstream reproducibility is unaffected.
        """
        self.model.eval()
        K = torch.tensor(
            self.test_data['K'].values, dtype=torch.float32
        ).unsqueeze(1).to(self.device)
        T = torch.tensor(
            self.test_data['T'].values, dtype=torch.float32
        ).unsqueeze(1).to(self.device)

        use_cuda = str(self.device).startswith("cuda")
        timings_ms = []
        with torch.no_grad():
            for _ in range(n_warmup):
                self.model(K, T)
            if use_cuda:
                torch.cuda.synchronize()
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                self.model(K, T)
                if use_cuda:
                    torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - t0) * 1000.0)

        latency = float(np.median(timings_ms))
        self.logger.info(
            f"Inference latency: {latency:.3f} ms (median of {n_repeats} runs, "
            f"{len(self.test_data)} grid points)"
        )
        return latency

    def _predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on DataFrame."""
        self.model.eval()
        
        K = torch.tensor(df['K'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        T = torch.tensor(df['T'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            price_pred = self.model(K, T).cpu().numpy().flatten()
        
        df_pred = df[['K', 'T']].copy()
        df_pred['price'] = price_pred
        
        try:
            K_grad = torch.tensor(
                df['K'].values, dtype=torch.float32
            ).unsqueeze(1).to(self.device).requires_grad_(True)
            T_grad = torch.tensor(
                df['T'].values, dtype=torch.float32
            ).unsqueeze(1).to(self.device).requires_grad_(True)

            with torch.enable_grad():
                greeks = self.model.compute_greeks_for_loss(K_grad, T_grad)
            df_pred['delta'] = greeks['delta'].detach().cpu().numpy().flatten()
            df_pred['gamma'] = greeks['gamma'].detach().cpu().numpy().flatten()
            df_pred['theta'] = greeks['theta'].detach().cpu().numpy().flatten()
        except Exception as e:
            self.logger.warning(f"Could not compute Greeks: {e}")
            df_pred['delta'] = np.nan
            df_pred['gamma'] = np.nan
            df_pred['theta'] = np.nan
        
        return df_pred
    
    def _generate_figures(self, df_pred: pd.DataFrame, df_true: pd.DataFrame) -> None:
        """Generate and save figures."""
        from .figures import plot_price_scatter, plot_loss_curve
        
        fig_dir = self.run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # Price scatter
        plot_price_scatter(
            df_pred, df_true,
            save_path=fig_dir / f"{self.cfg.experiment_id}_price_scatter.png",
            title=f"{self.cfg.experiment_id}: Predicted vs True",
        )
        
        # Loss curve
        if self.history is not None:
            plot_loss_curve(
                self.history,
                save_path=fig_dir / f"{self.cfg.experiment_id}_loss_curve.png",
                title=f"{self.cfg.experiment_id}: Training Loss",
            )
        
        self.logger.info(f"Figures saved to {fig_dir}")


def run_multi_seed(
    cfg: ExperimentCfg,
    seeds: List[int] = None,
    device: Optional[str] = None,
    base_run_dir: Optional[str | Path] = None,
) -> dict:
    """
    Run experiment with multiple seeds and aggregate results.
    
    Args:
        cfg: Base experiment configuration.
        seeds: List of seeds to use (default: [20251, 20252, 20253, 20254, 20255]).
        device: Torch device.
        base_run_dir: Base directory for all runs.
        
    Returns:
        Dict with aggregated statistics (mean, std, all_metrics).
        
    Example:
        >>> cfg = load_config("configs/bs_softplus_full.json")
        >>> results = run_multi_seed(cfg, seeds=[20251, 20252, 20253])
        >>> print(f"RMSE: {results['mean']['rmse_price']:.4f} ± {results['std']['rmse_price']:.4f}")
    """
    if seeds is None:
        seeds = [20251, 20252, 20253, 20254, 20255]
    
    if base_run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_dir = Path("runs") / f"{timestamp}__{cfg.experiment_id}__multiseed"
    base_run_dir = Path(base_run_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for seed in seeds:
        # Create modified config with new seed
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.seed = seed
        
        # Create run directory for this seed
        run_dir = base_run_dir / f"seed_{seed}"
        
        print(f"\n{'='*60}")
        print(f"Running seed {seed} ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*60}")
        
        try:
            # Run experiment
            runner = ExperimentRunner(cfg_seed, run_dir=run_dir, device=device)
            runner.prepare()
            runner.train_and_eval()
            metrics = runner.report()
            
            metrics['seed'] = seed
            all_metrics.append(metrics)
            
        except Exception as e:
            import traceback
            print(f"Error with seed {seed}: {e}")
            traceback.print_exc()
            continue
    
    # Aggregate results
    if not all_metrics:
        return {"error": "No successful runs"}
    
    # Extract metric keys
    metric_keys = [k for k in all_metrics[0].keys() if k != 'seed' and isinstance(all_metrics[0][k], (int, float))]
    
    # Compute statistics
    mean_metrics = {}
    std_metrics = {}
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
    
    # Save aggregated results
    results = {
        "experiment_id": cfg.experiment_id,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean": mean_metrics,
        "std": std_metrics,
        "all_metrics": all_metrics,
    }
    
    stats_path = base_run_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    for key in metric_keys:
        if key in mean_metrics:
            print(f"  {key}: {mean_metrics[key]:.6f} ± {std_metrics[key]:.6f}")
    print(f"\nResults saved to: {stats_path}")
    
    return results


def run_param_sweep(
    base_cfg: ExperimentCfg,
    sweep_settings: list[dict],
    seeds: List[int] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Run an experiment across multiple diffusion-model parameter settings
    and aggregate results over both settings and seeds (Ch8 sec 8.3).

    Each entry in *sweep_settings* is a dict that is merged into
    ``base_cfg.model.params`` before training.

    Returns:
        Dict with per-setting results and grand-aggregate statistics.
    """
    if seeds is None:
        seeds = [20251, 20252, 20253, 20254, 20255]

    per_setting = []
    all_flat_metrics: list[dict] = []

    for idx, setting in enumerate(sweep_settings):
        cfg_s = copy.deepcopy(base_cfg)
        for k, v in setting.items():
            cfg_s.model.params[k] = v

        tag = "_".join(f"{k}{v}" for k, v in setting.items())
        cfg_s.experiment_id = f"{base_cfg.experiment_id}__ps{idx}_{tag}"

        print(f"\n{'='*60}")
        print(f"Param-sweep setting {idx+1}/{len(sweep_settings)}: {setting}")
        print(f"{'='*60}")

        result = run_multi_seed(cfg_s, seeds=seeds, device=device)
        per_setting.append({"setting": setting, "result": result})
        for m in result.get("all_metrics", []):
            all_flat_metrics.append(m)

    if not all_flat_metrics:
        return {"error": "No successful runs across any parameter setting"}

    metric_keys = [
        k for k in all_flat_metrics[0]
        if k != "seed" and isinstance(all_flat_metrics[0][k], (int, float))
    ]
    grand_mean = {}
    grand_std = {}
    for key in metric_keys:
        vals = [m[key] for m in all_flat_metrics if key in m and m[key] is not None]
        if vals:
            grand_mean[key] = float(np.mean(vals))
            grand_std[key] = float(np.std(vals))

    return {
        "experiment_id": base_cfg.experiment_id,
        "n_settings": len(sweep_settings),
        "n_seeds": len(seeds),
        "grand_mean": grand_mean,
        "grand_std": grand_std,
        "per_setting": per_setting,
    }


def run_ablation_grid(
    base_cfg: ExperimentCfg,
    ablation_params: dict,
    seeds: List[int] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run grid of ablation experiments.
    
    Args:
        base_cfg: Base experiment configuration.
        ablation_params: Dict of {param_path: [values]} for ablation.
        seeds: Seeds for each experiment.
        device: Torch device.
        
    Returns:
        DataFrame with all results.
        
    Example:
        >>> cfg = load_config("configs/bs_softplus_full.json")
        >>> ablation = {
        ...     "net.activation": ["softplus", "tanh", "relu"],
        ...     "loss.lambda_pde": [0.05, 0.1, 0.2],
        ... }
        >>> results_df = run_ablation_grid(cfg, ablation)
    """
    import itertools
    
    if seeds is None:
        seeds = [20251]
    
    # Generate all combinations
    param_names = list(ablation_params.keys())
    param_values = list(ablation_params.values())
    
    all_results = []
    
    for values in itertools.product(*param_values):
        # Create modified config
        cfg_ablation = copy.deepcopy(base_cfg)
        
        # Set parameters
        for name, val in zip(param_names, values):
            parts = name.split('.')
            obj = cfg_ablation
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], val)
        
        # Generate unique experiment ID
        param_str = "__".join(f"{n.split('.')[-1]}_{v}" for n, v in zip(param_names, values))
        cfg_ablation.experiment_id = f"{base_cfg.experiment_id}__{param_str}"
        
        print(f"\n{'='*60}")
        print(f"Ablation: {param_str}")
        print(f"{'='*60}")
        
        # Run with multiple seeds
        results = run_multi_seed(cfg_ablation, seeds=seeds, device=device)
        
        # Add ablation parameters to results
        row = {
            'experiment_id': cfg_ablation.experiment_id,
            **dict(zip(param_names, values)),
            **{f'{k}_mean': v for k, v in results.get('mean', {}).items()},
            **{f'{k}_std': v for k, v in results.get('std', {}).items()},
        }
        all_results.append(row)
    
    return pd.DataFrame(all_results)

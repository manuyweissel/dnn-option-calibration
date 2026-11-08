"""
Tests for configuration validation.

Verifies that all configs in configs/ directory are valid and can be loaded.
"""

import json
from pathlib import Path

import pytest

from src.exp.config import ExperimentCfg
from src.exp.io_utils import load_config


def test_config_schema_basic():
    """Test basic config schema validation."""
    config_dict = {
        "experiment_id": "test_exp",
        "seed": 42,
        "model": {"name": "BS", "params": {"S0": 1.0, "r": 0.02, "sigma": 0.2}},
        "grid": {"K": 10, "T": 5, "spacing": "uniform"},
        "net": {"L": 3, "W": 32, "activation": "softplus"},
    }
    
    cfg = ExperimentCfg.model_validate(config_dict)
    
    assert cfg.experiment_id == "test_exp"
    assert cfg.seed == 42
    assert cfg.model.name == "BS"
    assert cfg.grid.K == 10
    assert cfg.net.L == 3


def test_all_provided_configs():
    """Load and validate all configs in configs/ directory."""
    config_dir = Path("configs")
    
    if not config_dir.exists():
        pytest.skip("Config directory not found")
    
    config_files = list(config_dir.glob("*.json"))
    
    assert len(config_files) > 0, "No config files found"
    
    for config_path in config_files:
        print(f"Testing: {config_path.name}")
        cfg = load_config(config_path)
        
        # Basic checks
        assert cfg.experiment_id is not None
        assert cfg.seed > 0
        assert cfg.model.name in ["BS", "LocalVol", "Heston"]
        assert cfg.grid.K > 0
        assert cfg.grid.T > 0
        assert cfg.net.L >= 1
        assert cfg.net.W >= 8


def test_invalid_config_raises():
    """Test that invalid configs raise validation errors."""
    # Missing required field
    with pytest.raises(Exception):
        ExperimentCfg.model_validate({"experiment_id": "test"})
    
    # Invalid model name
    with pytest.raises(Exception):
        ExperimentCfg.model_validate({
            "experiment_id": "test",
            "model": {"name": "INVALID"},
            "grid": {"K": 10, "T": 5},
            "net": {"L": 3, "W": 32},
        })


def test_default_values():
    """Test that default values are correctly applied."""
    config_dict = {
        "experiment_id": "test_defaults",
        "model": {"name": "BS", "params": {}},
        "grid": {"K": 10, "T": 5},
        "net": {"L": 3, "W": 32},
    }
    
    cfg = ExperimentCfg.model_validate(config_dict)
    
    # Check defaults
    assert cfg.seed == 20251  # Default seed from thesis
    assert cfg.noise.price_rel == 0.0
    assert cfg.loss.lambda_price == 1.0
    assert cfg.optim.name == "adamw"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


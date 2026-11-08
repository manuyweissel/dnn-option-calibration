"""
I/O utilities for configuration loading, saving, and content hashing.

Functions handle loading/validating JSON configs, computing reproducible hashes,
and saving resolved configurations to run directories.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ExperimentCfg


def load_config(config_path: str | Path) -> ExperimentCfg:
    """
    Load and validate experiment configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file.
        
    Returns:
        Validated ExperimentCfg object.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        ValidationError: If config does not match schema.
        
    Example:
        >>> cfg = load_config("configs/bs_softplus_full.json")
        >>> assert cfg.seed == 20251
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    return ExperimentCfg.model_validate(config_dict)


def save_config(cfg: ExperimentCfg, save_path: str | Path) -> None:
    """
    Save resolved configuration to JSON file.
    
    Args:
        cfg: Experiment configuration.
        save_path: Path to save JSON file.
        
    Example:
        >>> save_config(cfg, "runs/20251010_120000__bs_full/cfg.json")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(), f, indent=2)


def compute_config_hash(cfg: ExperimentCfg) -> str:
    """
    Compute deterministic SHA256 hash of configuration.
    
    The hash is computed from the canonical JSON representation (sorted keys)
    to ensure reproducibility across runs.
    
    Args:
        cfg: Experiment configuration.
        
    Returns:
        Hexadecimal hash string (first 16 characters).
        
    Example:
        >>> hash1 = compute_config_hash(cfg)
        >>> hash2 = compute_config_hash(cfg)
        >>> assert hash1 == hash2
    """
    # Convert to dict and serialize with sorted keys for determinism
    config_str = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"))
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def save_artifact(data: Any, path: str | Path, format: str = "json") -> None:
    """
    Save arbitrary artifact to disk.
    
    Args:
        data: Data to save (dict, list, etc.).
        path: Save path.
        format: Format ('json', 'csv', 'txt').
        
    Raises:
        ValueError: If format is unsupported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif format == "txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_artifact(path: str | Path, format: str = "json") -> Any:
    """
    Load artifact from disk.
    
    Args:
        path: Path to artifact.
        format: Format ('json', 'txt').
        
    Returns:
        Loaded data.
        
    Raises:
        FileNotFoundError: If artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    if format == "json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif format == "txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported format: {format}")


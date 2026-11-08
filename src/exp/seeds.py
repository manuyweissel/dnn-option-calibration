"""
Deterministic seeding utilities for reproducibility.

Ensures all random number generators (Python, NumPy, PyTorch) use the same seed
and that CUDA operations are deterministic. Follows the reproducibility protocol
documented in Chapter 10, Section 10.6 of the thesis.
"""

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Set global random seed for all libraries.
    
    This function seeds:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA generators
    - CuDNN (made deterministic if requested)
    
    Args:
        seed: Integer seed value (recommended: 20251 per thesis protocol).
        deterministic: If True, enforce deterministic CuDNN operations (slower).
        
    Example:
        >>> seed_everything(20251)
        >>> # All subsequent random operations are now reproducible
        
    Note:
        Deterministic mode may reduce training speed but ensures exact
        reproducibility across runs with the same seed.
    """
    # Python built-in
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # CuDNN deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, tag: str) -> int:
    """
    Derive a deterministic subseed from a base seed and string tag.
    
    This is useful for creating different seeds for data generation,
    noise injection, etc., while maintaining reproducibility.
    
    Args:
        base_seed: Base seed (e.g., from experiment config).
        tag: String identifier (e.g., "noise", "data_split").
        
    Returns:
        Derived integer seed.
        
    Example:
        >>> data_seed = derive_seed(20251, "data_generation")
        >>> noise_seed = derive_seed(20251, "noise_injection")
        >>> assert data_seed != noise_seed  # Different tags -> different seeds
    """
    # Use hash of base_seed + tag for deterministic derivation
    import hashlib
    
    combined = f"{base_seed}_{tag}".encode("utf-8")
    hash_int = int(hashlib.sha256(combined).hexdigest()[:8], 16)
    return hash_int % (2**31)  # Keep in int32 range


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random number generator with optional seed.
    
    Args:
        seed: Optional seed (None = use current global state).
        
    Returns:
        NumPy Generator object.
        
    Example:
        >>> rng = get_rng(12345)
        >>> samples = rng.normal(0, 1, size=100)
    """
    return np.random.default_rng(seed)


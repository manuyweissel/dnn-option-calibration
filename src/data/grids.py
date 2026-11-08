"""
Grid generation and data splitting utilities.

Creates strike-maturity grids for training, validation, and test sets.
Supports both uniform and log-moneyness spacing.
"""

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def create_grid(
    K: int,
    T: int,
    spacing: Literal["uniform", "logm"] = "logm",
    K_min: float | None = None,
    K_max: float | None = None,
    T_min: float | None = None,
    T_max: float | None = None,
    S0: float = 1.0,
) -> pd.DataFrame:
    """
    Create strike-maturity grid.
    
    Args:
        K: Number of strike points.
        T: Number of maturity points.
        spacing: Spacing type ('uniform' or 'logm' for log-moneyness).
        K_min: Minimum strike (defaults based on spacing).
        K_max: Maximum strike (defaults based on spacing).
        T_min: Minimum maturity (default: 0.01).
        T_max: Maximum maturity (default: 1.0).
        S0: Initial underlying price (for log-moneyness).
        
    Returns:
        DataFrame with columns ['K', 'T'] of shape (K * T, 2).
        
    Example:
        >>> grid = create_grid(K=50, T=20, spacing="logm")
        >>> assert grid.shape == (1000, 2)
        >>> assert (grid['K'] > 0).all()
    """
    # Set defaults
    if T_min is None:
        T_min = 0.01
    if T_max is None:
        T_max = 1.0
    
    # Create maturity grid (always linear)
    T_grid = np.linspace(T_min, T_max, T)
    
    # Create strike grid
    if spacing == "uniform":
        if K_min is None:
            K_min = 0.5 * S0
        if K_max is None:
            K_max = 1.5 * S0
        K_grid = np.linspace(K_min, K_max, K)
    
    elif spacing == "logm":
        # Log-moneyness: m = log(K / S0)
        if K_min is None:
            m_min = -0.3  # Default log-moneyness range
        else:
            m_min = np.log(K_min / S0)
        
        if K_max is None:
            m_max = 0.3
        else:
            m_max = np.log(K_max / S0)
        
        m_grid = np.linspace(m_min, m_max, K)
        K_grid = S0 * np.exp(m_grid)
    
    else:
        raise ValueError(f"Unknown spacing: {spacing}")
    
    # Create meshgrid
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid, indexing='ij')
    
    # Flatten and create DataFrame
    df = pd.DataFrame({
        'K': K_mesh.flatten(),
        'T': T_mesh.flatten(),
    })
    
    return df


def split_data(
    df: pd.DataFrame,
    seed: int = 42,
    S0: float = 1.0,
    extrap_T_threshold: float = 0.8,
    extrap_K_low: float = 0.85,
    extrap_K_high: float = 1.15,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data with geometric extrapolation carving (Ch8 sec 8.3).

    1. Carve extrapolation test set: T > extrap_T_threshold
       or K < extrap_K_low*S0 or K > extrap_K_high*S0.
    2. From remaining points, randomly split into train/val/interp-test.

    Args:
        df: DataFrame with columns ['K', 'T', ...].
        seed: Random seed.
        S0: Initial underlying price.
        extrap_T_threshold: Maturity threshold for extrapolation region.
        extrap_K_low: Lower moneyness boundary (fraction of S0).
        extrap_K_high: Upper moneyness boundary (fraction of S0).
        train_frac: Training fraction of non-extrapolation points.
        val_frac: Validation fraction.
        test_frac: Interpolation test fraction.

    Returns:
        Tuple of (train_df, val_df, test_interp_df, test_extrap_df).
    """
    extrap_mask = (
        (df['T'] > extrap_T_threshold)
        | (df['K'] < extrap_K_low * S0)
        | (df['K'] > extrap_K_high * S0)
    )

    extrap_df = df[extrap_mask].reset_index(drop=True)
    interior_df = df[~extrap_mask].reset_index(drop=True)

    total = train_frac + val_frac + test_frac
    t_frac = train_frac / total
    v_frac = val_frac / total

    interior_shuffled = interior_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(interior_shuffled)
    n_train = int(n * t_frac)
    n_val = int(n * v_frac)

    train_df = interior_shuffled.iloc[:n_train]
    val_df = interior_shuffled.iloc[n_train:n_train + n_val]
    test_interp_df = interior_shuffled.iloc[n_train + n_val:]

    return train_df, val_df, test_interp_df, extrap_df


def create_extrapolation_grid(
    S0: float = 1.0,
    K_extrap_min: float = 0.5,
    K_extrap_max: float = 1.6,
    T_extrap_min: float = 1.0,
    T_extrap_max: float = 1.5,
    nK: int = 20,
    nT: int = 10,
) -> pd.DataFrame:
    """
    Create extrapolation grid outside training support.
    
    Args:
        S0: Initial underlying price.
        K_extrap_min: Minimum extrapolation strike.
        K_extrap_max: Maximum extrapolation strike.
        T_extrap_min: Minimum extrapolation maturity.
        T_extrap_max: Maximum extrapolation maturity.
        nK: Number of strike points.
        nT: Number of maturity points.
        
    Returns:
        DataFrame with columns ['K', 'T'].
        
    Example:
        >>> extrap_grid = create_extrapolation_grid()
        >>> assert (extrap_grid['T'] >= 1.0).all()
    """
    K_grid = np.linspace(K_extrap_min, K_extrap_max, nK)
    T_grid = np.linspace(T_extrap_min, T_extrap_max, nT)
    
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid, indexing='ij')
    
    df = pd.DataFrame({
        'K': K_mesh.flatten(),
        'T': T_mesh.flatten(),
    })
    
    return df


def normalize_inputs(
    K: np.ndarray,
    T: np.ndarray,
    K_min: float,
    K_max: float,
    T_min: float,
    T_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize inputs to [-1, 1] range.
    
    Args:
        K: Strike prices.
        T: Maturities.
        K_min: Minimum strike for normalization.
        K_max: Maximum strike for normalization.
        T_min: Minimum maturity for normalization.
        T_max: Maximum maturity for normalization.
        
    Returns:
        Tuple of (K_normalized, T_normalized).
        
    Example:
        >>> K_norm, T_norm = normalize_inputs(
        ...     np.array([0.8, 1.0, 1.2]),
        ...     np.array([0.25, 0.5, 0.75]),
        ...     K_min=0.7, K_max=1.3,
        ...     T_min=0.0, T_max=1.0,
        ... )
        >>> assert K_norm.min() >= -1 and K_norm.max() <= 1
    """
    K_norm = 2 * (K - K_min) / (K_max - K_min) - 1
    T_norm = 2 * (T - T_min) / (T_max - T_min) - 1
    
    return K_norm, T_norm


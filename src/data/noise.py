"""
Noise models for option pricing experiments.

Implements realistic market noise models:
- Relative price noise (bid-ask spread emulation)
- Heteroscedastic noise (larger noise for OTM options)
- Missing data (random masking)
- Greek noise (derivative uncertainty)
"""

from typing import Optional
import numpy as np
import pandas as pd


def add_relative_noise(
    data: pd.DataFrame,
    price_noise_std: float = 0.01,
    greek_noise_std: float = 0.005,
    seed: int = 20251,
) -> pd.DataFrame:
    """
    Add relative (multiplicative) noise to prices and Greeks.
    
    Models bid-ask spread: price_noisy = price * (1 + ε), ε ~ N(0, σ²)
    
    Args:
        data: DataFrame with 'price' and Greek columns.
        price_noise_std: Relative noise std for prices (e.g., 0.01 = 1%).
        greek_noise_std: Relative noise std for Greeks.
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with noisy values (original columns preserved with '_clean' suffix).
    """
    np.random.seed(seed)
    
    df = data.copy()
    n = len(df)
    
    # Price noise
    if 'price' in df.columns and price_noise_std > 0:
        df['price_clean'] = df['price'].copy()
        noise = np.random.normal(0, price_noise_std, n)
        df['price'] = df['price'] * (1 + noise)
        df['price'] = df['price'].clip(lower=0)  # Prices must be non-negative
    
    # Greek noise
    greek_cols = ['delta', 'gamma', 'vega', 'theta']
    for col in greek_cols:
        if col in df.columns and greek_noise_std > 0:
            df[f'{col}_clean'] = df[col].copy()
            noise = np.random.normal(0, greek_noise_std, n)
            df[col] = df[col] * (1 + noise)
    
    return df


def add_absolute_noise(
    data: pd.DataFrame,
    price_noise_std: float = 0.001,
    greek_noise_std: float = 0.0005,
    seed: int = 20251,
) -> pd.DataFrame:
    """
    Add absolute (additive) noise to prices and Greeks.
    
    Args:
        data: DataFrame with 'price' and Greek columns.
        price_noise_std: Absolute noise std for prices.
        greek_noise_std: Absolute noise std for Greeks.
        seed: Random seed.
        
    Returns:
        DataFrame with noisy values.
    """
    np.random.seed(seed)
    
    df = data.copy()
    n = len(df)
    
    if 'price' in df.columns and price_noise_std > 0:
        df['price_clean'] = df['price'].copy()
        noise = np.random.normal(0, price_noise_std, n)
        df['price'] = df['price'] + noise
        df['price'] = df['price'].clip(lower=0)
    
    greek_cols = ['delta', 'gamma', 'vega', 'theta']
    for col in greek_cols:
        if col in df.columns and greek_noise_std > 0:
            df[f'{col}_clean'] = df[col].copy()
            noise = np.random.normal(0, greek_noise_std, n)
            df[col] = df[col] + noise
    
    return df


def add_heteroscedastic_noise(
    data: pd.DataFrame,
    base_std: float = 0.005,
    otm_multiplier: float = 3.0,
    S0: float = 1.0,
    seed: int = 20251,
) -> pd.DataFrame:
    """
    Add heteroscedastic noise that increases for OTM options.
    
    Models real market behavior where OTM options have wider bid-ask spreads.
    
    Args:
        data: DataFrame with 'K', 'T', 'price' columns.
        base_std: Base noise std for ATM options.
        otm_multiplier: Multiplier for deep OTM options.
        S0: Spot price for moneyness calculation.
        seed: Random seed.
        
    Returns:
        DataFrame with heteroscedastic noise applied.
    """
    np.random.seed(seed)
    
    df = data.copy()
    n = len(df)
    
    # Compute moneyness
    K = df['K'].values
    moneyness = np.abs(np.log(K / S0))  # 0 for ATM, larger for OTM
    
    # Noise std increases with OTM-ness
    # std(ε) = base_std * (1 + (otm_multiplier - 1) * min(moneyness / 0.3, 1))
    noise_mult = 1 + (otm_multiplier - 1) * np.minimum(moneyness / 0.3, 1.0)
    std_vec = base_std * noise_mult
    
    if 'price' in df.columns:
        df['price_clean'] = df['price'].copy()
        noise = np.random.normal(0, 1, n) * std_vec
        df['price'] = df['price'] * (1 + noise)
        df['price'] = df['price'].clip(lower=0)
    
    return df


def add_missing_data(
    data: pd.DataFrame,
    missing_fraction: float = 0.1,
    seed: int = 20251,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Randomly mask data points (simulate missing observations).
    
    Args:
        data: DataFrame with price/Greek data.
        missing_fraction: Fraction of points to mask (0 to 1).
        seed: Random seed.
        
    Returns:
        Tuple of (masked DataFrame, boolean mask array where True = observed).
    """
    np.random.seed(seed)
    
    df = data.copy()
    n = len(df)
    
    # Generate mask
    mask = np.random.random(n) > missing_fraction
    
    # Apply mask (set missing to NaN)
    for col in ['price', 'delta', 'gamma', 'vega', 'theta']:
        if col in df.columns:
            df.loc[~mask, col] = np.nan
    
    return df, mask


def add_bid_ask_spread(
    data: pd.DataFrame,
    spread_pct: float = 0.02,
    seed: int = 20251,
) -> pd.DataFrame:
    """
    Add bid-ask spread simulation.
    
    Creates bid/ask prices around mid price.
    
    Args:
        data: DataFrame with 'price' column (treated as mid price).
        spread_pct: Spread as percentage of mid (e.g., 0.02 = 2%).
        seed: Random seed.
        
    Returns:
        DataFrame with 'price_bid', 'price_ask', 'price_mid' columns.
    """
    np.random.seed(seed)
    
    df = data.copy()
    
    if 'price' not in df.columns:
        return df
    
    mid = df['price'].values
    half_spread = mid * spread_pct / 2
    
    # Add random variation to spread
    spread_noise = np.random.uniform(0.8, 1.2, len(df))
    half_spread = half_spread * spread_noise
    
    df['price_mid'] = mid
    df['price_bid'] = mid - half_spread
    df['price_ask'] = mid + half_spread
    
    # Ensure bid >= 0
    df['price_bid'] = df['price_bid'].clip(lower=0)
    
    return df


def apply_noise_config(
    data: pd.DataFrame,
    noise_cfg: dict,
    seed: int = 20251,
) -> pd.DataFrame:
    """
    Apply noise configuration from experiment config.
    
    Args:
        data: Clean data DataFrame.
        noise_cfg: Dict with keys like 'price_rel', 'greek_rel', 'missing_frac'.
        seed: Random seed.
        
    Returns:
        Noisy DataFrame.
    """
    df = data.copy()
    
    # Get noise parameters
    price_rel = noise_cfg.get('price_rel', 0.0)
    greek_rel = noise_cfg.get('greek_rel', 0.0)
    price_abs = noise_cfg.get('price_abs', 0.0)
    greek_abs = noise_cfg.get('greek_abs', 0.0)
    missing_frac = noise_cfg.get('missing_frac', 0.0)
    heteroscedastic = noise_cfg.get('heteroscedastic', False)
    
    # Apply relative noise
    if price_rel > 0 or greek_rel > 0:
        df = add_relative_noise(df, price_rel, greek_rel, seed)
    
    # Apply absolute noise
    if price_abs > 0 or greek_abs > 0:
        df = add_absolute_noise(df, price_abs, greek_abs, seed + 1)
    
    # Apply heteroscedastic noise
    if heteroscedastic and 'K' in df.columns:
        df = add_heteroscedastic_noise(df, seed=seed + 2)
    
    # Apply missing data
    if missing_frac > 0:
        df, _ = add_missing_data(df, missing_frac, seed + 3)
    
    return df


def estimate_noise_level(
    noisy_prices: np.ndarray,
    clean_prices: np.ndarray,
) -> dict:
    """
    Estimate noise statistics from noisy vs clean data.
    
    Args:
        noisy_prices: Observed prices with noise.
        clean_prices: True (noiseless) prices.
        
    Returns:
        Dict with noise statistics.
    """
    diff = noisy_prices - clean_prices
    rel_diff = diff / (clean_prices + 1e-10)
    
    return {
        'abs_mean': np.mean(np.abs(diff)),
        'abs_std': np.std(diff),
        'rel_mean': np.mean(np.abs(rel_diff)),
        'rel_std': np.std(rel_diff),
        'max_abs_error': np.max(np.abs(diff)),
        'max_rel_error': np.max(np.abs(rel_diff)),
    }

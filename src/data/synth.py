"""
Synthetic data generation for option pricing experiments.

Generates price and Greek surfaces using analytic pricers (BS, LocalVol, Heston, MC).
Supports parameter sweeps for testing identifiability.
"""

from typing import Literal, Optional
import itertools
import numpy as np
import pandas as pd

from ..models.pricing import (
    BSPricer, LocalVolPricer, HestonPricer, MonteCarloPricer, create_pricer
)


def generate_synthetic_data(
    grid: pd.DataFrame,
    model: Literal["BS", "LocalVol", "Heston", "MC"],
    params: dict,
    compute_greeks: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic option prices and Greeks.
    
    Args:
        grid: DataFrame with columns ['K', 'T'].
        model: Model name ('BS', 'LocalVol', 'Heston', 'MC').
        params: Model parameters dict.
        compute_greeks: Whether to compute Greeks (slower for some models).
        
    Returns:
        DataFrame with columns: ['K', 'T', 'price', 'delta', 'gamma', 'vega', 'theta'].
        
    Example:
        >>> from .grids import create_grid
        >>> grid = create_grid(10, 5)
        >>> params = {"S0": 1.0, "r": 0.02, "sigma": 0.2}
        >>> data = generate_synthetic_data(grid, "BS", params)
        >>> assert 'price' in data.columns
        >>> assert (data['price'] >= 0).all()
    """
    K = grid['K'].values
    T = grid['T'].values
    
    # Create pricer using factory
    pricer = create_pricer(model, params)
    
    # Compute prices
    price = pricer.price(K, T)
    
    # Create output DataFrame
    df_out = grid.copy()
    df_out['price'] = price
    
    if compute_greeks:
        df_out['delta'] = pricer.strike_delta(K, T)
        df_out['gamma'] = pricer.strike_gamma(K, T)
        df_out['theta'] = pricer.strike_theta(K, T)
        df_out['vega'] = pricer.vega(K, T)
    else:
        for col in ['delta', 'gamma', 'theta', 'vega']:
            df_out[col] = np.nan
    
    return df_out


def generate_parameter_sweep(
    grid: pd.DataFrame,
    model: Literal["BS", "LocalVol", "Heston", "MC"],
    base_params: dict,
    sweep_params: dict,
    compute_greeks: bool = True,
) -> pd.DataFrame:
    """
    Generate data for multiple parameter values (parameter sweep).
    
    Args:
        grid: Base grid.
        model: Model name.
        base_params: Fixed parameters.
        sweep_params: Dict of {param_name: [values]}.
        compute_greeks: Whether to compute Greeks.
        
    Returns:
        DataFrame with additional columns 'param_id' and 'param_*' for swept params.
        
    Example:
        >>> grid = create_grid(10, 5)
        >>> base_params = {"S0": 1.0, "r": 0.02}
        >>> sweep_params = {"sigma": [0.1, 0.2, 0.3]}
        >>> data = generate_parameter_sweep(grid, "BS", base_params, sweep_params)
        >>> assert data['param_id'].nunique() == 3
    """
    all_data = []
    param_id = 0
    
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    for values in itertools.product(*param_values):
        # Merge base + sweep params
        params = base_params.copy()
        for name, val in zip(param_names, values):
            params[name] = val
        
        # Generate data
        df_sweep = generate_synthetic_data(grid, model, params, compute_greeks)
        df_sweep['param_id'] = param_id
        
        # Add parameter columns
        for name, val in zip(param_names, values):
            df_sweep[f'param_{name}'] = val
        
        all_data.append(df_sweep)
        param_id += 1
    
    return pd.concat(all_data, ignore_index=True)


def generate_heldout_test_data(
    grid: pd.DataFrame,
    model: str,
    params: dict,
    extrapolation_factor: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate both interpolation and extrapolation test sets.
    
    Args:
        grid: Training grid.
        model: Model name.
        params: Model parameters.
        extrapolation_factor: Factor for extending extrapolation range.
        
    Returns:
        Tuple of (interpolation_df, extrapolation_df).
    """
    K_train = grid['K'].values
    T_train = grid['T'].values
    
    K_min, K_max = K_train.min(), K_train.max()
    T_min, T_max = T_train.min(), T_train.max()
    
    # Interpolation grid (midpoints)
    n_interp = 20
    K_interp = np.linspace(K_min, K_max, n_interp)
    T_interp = np.linspace(T_min, T_max, n_interp // 2)
    K_mesh, T_mesh = np.meshgrid(K_interp, T_interp)
    
    interp_grid = pd.DataFrame({
        'K': K_mesh.flatten(),
        'T': T_mesh.flatten(),
    })
    interp_df = generate_synthetic_data(interp_grid, model, params)
    interp_df['split'] = 'interpolation'
    
    # Extrapolation grid
    K_extrap_low = np.linspace(K_min / extrapolation_factor, K_min * 0.9, 10)
    K_extrap_high = np.linspace(K_max * 1.1, K_max * extrapolation_factor, 10)
    T_extrap = np.linspace(T_max * 1.1, T_max * extrapolation_factor, 5)
    
    # Combine extrapolation regions
    extrap_data = []
    
    # Low K extrapolation
    for k in K_extrap_low:
        for t in T_interp:
            extrap_data.append({'K': k, 'T': t})
    
    # High K extrapolation
    for k in K_extrap_high:
        for t in T_interp:
            extrap_data.append({'K': k, 'T': t})
    
    # High T extrapolation
    for k in K_interp:
        for t in T_extrap:
            extrap_data.append({'K': k, 'T': t})
    
    extrap_grid = pd.DataFrame(extrap_data)
    extrap_df = generate_synthetic_data(extrap_grid, model, params)
    extrap_df['split'] = 'extrapolation'
    
    return interp_df, extrap_df


def generate_mc_baseline(
    grid: pd.DataFrame,
    params: dict,
    n_paths: int = 100000,
    n_seeds: int = 5,
) -> pd.DataFrame:
    """
    Generate Monte Carlo baseline with uncertainty estimates.
    
    Args:
        grid: Grid of (K, T) points.
        params: Model parameters.
        n_paths: Number of MC paths per estimate.
        n_seeds: Number of seeds for variance estimation.
        
    Returns:
        DataFrame with price mean, std, and 95% CI.
    """
    K = grid['K'].values
    T = grid['T'].values
    
    prices_all = []
    
    for seed in range(n_seeds):
        mc_params = params.copy()
        mc_params['seed'] = 20251 + seed
        mc_params['n_paths'] = n_paths
        
        pricer = MonteCarloPricer(**mc_params)
        prices = pricer.price(K, T)
        prices_all.append(prices)
    
    prices_array = np.array(prices_all)
    
    df_out = grid.copy()
    df_out['price_mean'] = prices_array.mean(axis=0)
    df_out['price_std'] = prices_array.std(axis=0)
    df_out['price_ci_low'] = df_out['price_mean'] - 1.96 * df_out['price_std']
    df_out['price_ci_high'] = df_out['price_mean'] + 1.96 * df_out['price_std']
    
    # Compute Greeks via central MC estimate
    mc_params['seed'] = 20251
    mc_params['n_paths'] = n_paths * 2  # More paths for Greeks
    pricer = MonteCarloPricer(**mc_params)
    
    df_out['delta'] = pricer.delta(K, T)
    df_out['gamma'] = pricer.gamma(K, T)
    df_out['vega'] = pricer.vega(K, T)
    
    return df_out

"""
Evaluation metrics for option pricing models.

Implements RMSE, MAE, parameter errors, and extrapolation metrics as defined
in Chapter 10 (Table tab:metrics) of the thesis.
"""

import numpy as np
import pandas as pd
import torch


def rmse_price(df_pred: pd.DataFrame, df_true: pd.DataFrame, column: str = 'price') -> float:
    """
    Compute root mean squared error for prices.
    
    Formula: RMSE = sqrt(mean((pred - true)²))
    
    Args:
        df_pred: DataFrame with predicted values.
        df_true: DataFrame with true values.
        column: Column name to compute RMSE for.
        
    Returns:
        RMSE value (scalar).
        
    Example:
        >>> df_pred = pd.DataFrame({'price': [0.10, 0.20, 0.30]})
        >>> df_true = pd.DataFrame({'price': [0.11, 0.19, 0.31]})
        >>> error = rmse_price(df_pred, df_true)
        >>> assert error > 0
    """
    pred = df_pred[column].values
    true = df_true[column].values
    
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def mae_greeks(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    greeks: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute mean absolute error for Greeks.
    
    Formula: MAE_G = mean(|pred_G - true_G|)
    
    Args:
        df_pred: DataFrame with predicted Greeks.
        df_true: DataFrame with true Greeks.
        greeks: List of Greek names (None = ['delta', 'gamma', 'theta', 'vega']).
        
    Returns:
        Dict with MAE for each Greek and average MAE.
        
    Example:
        >>> df_pred = pd.DataFrame({'delta': [0.5, 0.6], 'gamma': [0.1, 0.2], 'theta': [-0.01, -0.02]})
        >>> df_true = pd.DataFrame({'delta': [0.52, 0.58], 'gamma': [0.11, 0.19], 'theta': [-0.012, -0.018]})
        >>> errors = mae_greeks(df_pred, df_true, greeks=['delta', 'gamma', 'theta'])
        >>> assert 'mae_delta' in errors
        >>> assert 'mae_avg' in errors
    """
    if greeks is None:
        greeks = ['delta', 'gamma', 'theta', 'vega']
    
    errors = {}
    valid_greeks = []
    norm_scores = []

    for greek in greeks:
        if greek in df_pred.columns and greek in df_true.columns:
            pred = df_pred[greek].values
            true = df_true[greek].values
            mae = np.mean(np.abs(pred - true))
            errors[f'mae_{greek}'] = float(mae)
            valid_greeks.append(mae)

            med_abs = float(np.median(np.abs(true)))
            if med_abs > 1e-10:
                norm_scores.append(mae / med_abs)

    if valid_greeks:
        errors['mae_avg'] = float(np.mean(valid_greeks))
    else:
        errors['mae_avg'] = 0.0

    if norm_scores:
        errors['greek_norm_score'] = float(np.mean(norm_scores))
    else:
        errors['greek_norm_score'] = 0.0

    return errors


def rpe(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    column: str = 'price',
    min_abs_value: float = 0.02,
    cap: float = 10.0,
) -> float:
    """
    Compute mean relative price error, excluding near-zero true values.

    Points where |true| < min_abs_value are excluded to avoid
    astronomical relative errors from deep-OTM options.
    Individual relative errors are capped at ``cap`` to prevent
    outlier domination (winsorization).
    """
    pred = df_pred[column].values
    true = df_true[column].values

    mask = np.abs(true) >= min_abs_value
    if mask.sum() == 0:
        return 0.0

    rel_error = np.abs(pred[mask] - true[mask]) / np.abs(true[mask])
    rel_error = np.minimum(rel_error, cap)
    return float(np.mean(rel_error))


def param_rel_err(params_pred: dict[str, float], params_true: dict[str, float]) -> float:
    """
    Compute relative parameter error (L2 norm).
    
    Formula: RPE_φ = ||φ_pred - φ_true||_2 / ||φ_true||_2
    
    Args:
        params_pred: Dict of predicted parameters.
        params_true: Dict of true parameters.
        
    Returns:
        Relative parameter error (scalar).
        
    Example:
        >>> pred = {'sigma': 0.21, 'r': 0.022}
        >>> true = {'sigma': 0.20, 'r': 0.020}
        >>> error = param_rel_err(pred, true)
        >>> assert error > 0
    """
    # Extract common keys
    keys = set(params_pred.keys()) & set(params_true.keys())
    
    if not keys:
        return 0.0
    
    # Compute L2 norms
    pred_vec = np.array([params_pred[k] for k in keys])
    true_vec = np.array([params_true[k] for k in keys])
    
    norm_diff = np.linalg.norm(pred_vec - true_vec)
    norm_true = np.linalg.norm(true_vec)
    
    if norm_true < 1e-10:
        return 0.0
    
    return float(norm_diff / norm_true)


def rmse_iv(df_pred: pd.DataFrame, df_true: pd.DataFrame) -> float:
    """
    Compute RMSE of implied volatility.
    
    Assumes DataFrames have 'iv' column.
    
    Args:
        df_pred: DataFrame with predicted implied vol.
        df_true: DataFrame with true implied vol.
        
    Returns:
        IV RMSE.
        
    Example:
        >>> df_pred = pd.DataFrame({'iv': [0.19, 0.21]})
        >>> df_true = pd.DataFrame({'iv': [0.20, 0.20]})
        >>> error = rmse_iv(df_pred, df_true)
        >>> assert error >= 0
    """
    if 'iv' not in df_pred.columns or 'iv' not in df_true.columns:
        return 0.0
    
    return rmse_price(df_pred, df_true, column='iv')


def rmse_extrap(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    extrap_mask: np.ndarray | None = None,
) -> float:
    """
    Compute RMSE on extrapolation region.
    
    Args:
        df_pred: Predictions.
        df_true: True values.
        extrap_mask: Boolean mask indicating extrapolation points (None = all).
        
    Returns:
        Extrapolation RMSE.
        
    Example:
        >>> df_pred = pd.DataFrame({'price': [0.1, 0.2, 0.3, 0.4]})
        >>> df_true = pd.DataFrame({'price': [0.11, 0.19, 0.31, 0.39]})
        >>> mask = np.array([False, False, True, True])
        >>> error = rmse_extrap(df_pred, df_true, mask)
        >>> assert error >= 0
    """
    if extrap_mask is None:
        extrap_mask = np.ones(len(df_pred), dtype=bool)
    
    df_pred_ex = df_pred[extrap_mask]
    df_true_ex = df_true[extrap_mask]
    
    if len(df_pred_ex) == 0:
        return 0.0
    
    return rmse_price(df_pred_ex, df_true_ex)


def pde_residual_metric(
    model: torch.nn.Module,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: float,
    model_type: str = "BS",
) -> dict[str, float]:
    """Compute PDE residual RMSE on a test grid (Ch8 sec 8.5)."""
    from src.models.operators import pde_residual

    K_t = torch.tensor(K, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    T_t = torch.tensor(T, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    res = pde_residual(model, K_t, T_t, r, sigma, model_type=model_type)
    res_np = res.detach().cpu().numpy().flatten()

    with torch.no_grad():
        K_t2 = torch.tensor(K, dtype=torch.float32).unsqueeze(1)
        T_t2 = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
        prices = model(K_t2, T_t2).cpu().numpy().flatten()
    price_scale = max(np.abs(prices).mean(), 1e-8)

    return {
        "pde_residual_rmse": float(np.sqrt(np.mean(res_np ** 2))),
        "pde_residual_norm": float(np.sqrt(np.mean(res_np ** 2)) / price_scale),
    }


def arbitrage_diagnostics(
    model: torch.nn.Module,
    K: np.ndarray,
    T: np.ndarray,
) -> dict[str, float]:
    """
    Compute no-arbitrage violation diagnostics (Ch8 sec 8.5).

    Checks strike monotonicity (dC/dK <= 0), strike convexity (d2C/dK2 >= 0),
    and calendar monotonicity (dC/dT >= 0).
    """
    K_t = torch.tensor(K, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    T_t = torch.tensor(T, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    C = model(K_t, T_t)

    dC_dK = torch.autograd.grad(C.sum(), K_t, create_graph=True, retain_graph=True)[0]
    d2C_dK2 = torch.autograd.grad(dC_dK.sum(), K_t, retain_graph=True)[0]
    dC_dT = torch.autograd.grad(C.sum(), T_t, retain_graph=False)[0]

    dCdK_np = dC_dK.detach().cpu().numpy().flatten()
    gamma_np = d2C_dK2.detach().cpu().numpy().flatten()
    dCdT_np = dC_dT.detach().cpu().numpy().flatten()

    mono_violations = dCdK_np > 1e-8
    convex_violations = gamma_np < -1e-8
    calendar_violations = dCdT_np < -1e-8

    return {
        "arb_mono_frac": float(mono_violations.mean()),
        "arb_mono_severity": float(np.abs(dCdK_np[mono_violations]).mean()) if mono_violations.any() else 0.0,
        "arb_convex_frac": float(convex_violations.mean()),
        "arb_convex_severity": float(np.abs(gamma_np[convex_violations]).mean()) if convex_violations.any() else 0.0,
        "arb_calendar_frac": float(calendar_violations.mean()),
        "arb_calendar_severity": float(np.abs(dCdT_np[calendar_violations]).mean()) if calendar_violations.any() else 0.0,
    }


def paired_t_test(
    metrics_a: list[float],
    metrics_b: list[float],
) -> dict[str, float]:
    """
    Paired t-test for comparing two methods across seeds (Ch8 sec 8.5).

    Returns t-statistic and two-sided p-value.
    """
    from scipy.stats import ttest_rel

    a = np.array(metrics_a)
    b = np.array(metrics_b)
    if len(a) < 2 or len(a) != len(b):
        return {"t_stat": float("nan"), "p_value": float("nan")}
    stat, pval = ttest_rel(a, b)
    return {"t_stat": float(stat), "p_value": float(pval)}


def compute_all_metrics(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    params_pred: dict[str, float] | None = None,
    params_true: dict[str, float] | None = None,
    extrap_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all standard metrics.
    
    Args:
        df_pred: Predictions DataFrame.
        df_true: True values DataFrame.
        params_pred: Predicted parameters (optional).
        params_true: True parameters (optional).
        extrap_mask: Extrapolation mask (optional).
        
    Returns:
        Dict with all metrics.
        
    Example:
        >>> df_pred = pd.DataFrame({'price': [0.1], 'delta': [0.5], 'gamma': [0.1]})
        >>> df_true = pd.DataFrame({'price': [0.11], 'delta': [0.52], 'gamma': [0.09]})
        >>> metrics = compute_all_metrics(df_pred, df_true)
        >>> assert 'rmse_price' in metrics
    """
    metrics = {}

    if 'price' in df_pred.columns and 'price' in df_true.columns:
        metrics['rmse_price'] = rmse_price(df_pred, df_true)
        metrics['rpe'] = rpe(df_pred, df_true)

    greek_metrics = mae_greeks(df_pred, df_true)
    metrics.update(greek_metrics)

    if 'iv' in df_pred.columns and 'iv' in df_true.columns:
        metrics['rmse_iv'] = rmse_iv(df_pred, df_true)

    if params_pred is not None and params_true is not None:
        metrics['param_rel_err'] = param_rel_err(params_pred, params_true)

    if extrap_mask is not None:
        metrics['rmse_extrap'] = rmse_extrap(df_pred, df_true, extrap_mask)

    return metrics


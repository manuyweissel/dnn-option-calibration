"""
Dupire local-volatility extraction from a trained neural network surface.

Given a network C(K, T), the Dupire formula recovers the local variance:

    sigma_loc^2(K, T) = (dC/dT + r K dC/dK) / (0.5 K^2 d2C/dK2)

For BS, the extracted sigma_loc should be constant = sigma.
For LocalVol, it should match sigma(S, t) evaluated on the (K, T) grid
via the symmetry K <-> S for flat local-vol models.
"""

import numpy as np
import torch


def dupire_extract_localvol(
    model: torch.nn.Module,
    K: np.ndarray,
    T: np.ndarray,
    r: float = 0.02,
) -> np.ndarray:
    """
    Extract Dupire local volatility from a trained price surface.

    Args:
        model: Trained network C(K, T).
        K: Strike array, shape (N,).
        T: Maturity array, shape (N,).
        r: Risk-free rate.

    Returns:
        sigma_loc array, shape (N,). NaN where denominator is non-positive.
    """
    K_t = torch.tensor(K, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    T_t = torch.tensor(T, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    C = model(K_t, T_t)

    dC_dK = torch.autograd.grad(C.sum(), K_t, create_graph=True)[0]
    d2C_dK2 = torch.autograd.grad(dC_dK.sum(), K_t, create_graph=True)[0]
    dC_dT = torch.autograd.grad(C.sum(), T_t, create_graph=True)[0]

    numerator = dC_dT + r * K_t * dC_dK
    denominator = 0.5 * K_t ** 2 * d2C_dK2

    num_np = numerator.detach().cpu().numpy().flatten()
    den_np = denominator.detach().cpu().numpy().flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = np.where(den_np > 1e-12, num_np / den_np, np.nan)
        sigma_loc = np.where(local_var > 0, np.sqrt(local_var), np.nan)

    return sigma_loc


def recover_bs_sigma(
    model: torch.nn.Module,
    K: np.ndarray,
    T: np.ndarray,
    r: float = 0.02,
) -> float:
    """
    Recover the constant BS sigma by taking the median of the Dupire surface.

    Returns scalar sigma estimate (NaN-robust).
    """
    sigma_loc = dupire_extract_localvol(model, K, T, r=r)
    valid = sigma_loc[np.isfinite(sigma_loc)]
    if len(valid) == 0:
        return float("nan")
    return float(np.median(valid))


def recover_lv_params(
    model: torch.nn.Module,
    K: np.ndarray,
    T: np.ndarray,
    S0: float = 1.0,
    r: float = 0.02,
) -> dict[str, float]:
    """
    Recover LocalVol (sigma0, alpha) by least-squares fit of
    sigma_loc(K) ~ sigma0 * exp(alpha * (K - S0)) to the Dupire surface.

    Returns dict with sigma0 and alpha estimates.
    """
    from scipy.optimize import curve_fit

    sigma_loc = dupire_extract_localvol(model, K, T, r=r)
    mask = np.isfinite(sigma_loc)
    if mask.sum() < 3:
        return {"sigma0": float("nan"), "alpha": float("nan")}

    K_valid = K[mask]
    sig_valid = sigma_loc[mask]

    def lv_model(k, sigma0, alpha):
        return sigma0 * (k / S0) ** alpha

    try:
        popt, _ = curve_fit(lv_model, K_valid, sig_valid, p0=[0.2, 0.5], maxfev=5000)
        return {"sigma0": float(popt[0]), "alpha": float(popt[1])}
    except RuntimeError:
        return {"sigma0": float("nan"), "alpha": float("nan")}

"""
Monte Carlo path simulation for the classical MC calibration baseline
(Ch.8, sec. 8.4.2).

Motivation
----------
The MC calibration baseline must price by *simulation* of the model SDE, not by
the analytic / semi-analytic pricer.  Previously ``MCCalibrator`` evaluated its
objective with ``create_pricer(model).price(...)`` (the analytic BS formula, the
Dupire PDE solver, or the Lewis-integral Heston pricer), so the "MC" baseline was
byte-identical to the deterministic L-BFGS-B baseline.  This module provides
genuine path-based pricing so the two baselines are distinct.

Design
------
* European call prices are estimated as ``e^{-rT} E[(S_T - K)^+]``.
* **Common random numbers**: the normal draws are keyed off a fixed ``seed`` that
  the caller holds constant across optimiser iterations, so the calibration
  objective is a smooth (deterministic) function of the parameters within a phase.
* **Antithetic variates** for variance reduction.
* Draws are shared across all strikes that share a maturity.

Models
------
* ``BS``       : exact log-normal terminal sampling (no time-stepping).
* ``LocalVol`` : log-Euler with spot/time-dependent vol
                 ``sigma_loc(S,t) = sigma0 (S/S0)^alpha (1 + beta t)``.
* ``Heston``   : full-truncation Euler for the (S, v) system.

Degenerate-limit checks (used in tests/test_mc_pricing.py):
  LocalVol(alpha=0, beta=0) == BS(sigma=sigma0);
  Heston(xi=0, v0=theta=sigma^2) == BS(sigma).
"""

from __future__ import annotations

import numpy as np


def _draw_normals(rng: np.random.Generator, n_paths: int, n_cols: int,
                  antithetic: bool) -> np.ndarray:
    """Return an (n_paths, n_cols) normal array, antithetic in the path axis."""
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_cols))
        z = np.concatenate([z, -z], axis=0)[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_cols))
    return z


def _terminal_bs(S0, r, sigma, T, n_paths, rng, antithetic):
    """Exact GBM terminal sample S_T for a single maturity T."""
    z = _draw_normals(rng, n_paths, 1, antithetic)[:, 0]
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)


def _terminal_localvol(S0, r, sigma0, alpha, beta, T, n_paths, n_steps, rng, antithetic):
    """Log-Euler terminal sample for a local-volatility model."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    z = _draw_normals(rng, n_paths, n_steps, antithetic)
    logS = np.full(n_paths, np.log(S0))
    for j in range(n_steps):
        t_cur = j * dt
        S = np.exp(logS)
        sig = sigma0 * (S / S0) ** alpha * (1.0 + beta * t_cur)
        sig = np.clip(sig, 1e-6, 5.0)
        logS += (r - 0.5 * sig ** 2) * dt + sig * sqrt_dt * z[:, j]
    return np.exp(logS)


def _terminal_heston(S0, r, kappa, theta, xi, rho, v0, T, n_paths, n_steps, rng, antithetic):
    """Full-truncation Euler terminal sample for the Heston model."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    z1 = _draw_normals(rng, n_paths, n_steps, antithetic)
    z2 = _draw_normals(rng, n_paths, n_steps, antithetic)
    w2 = rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z2  # correlated with z1
    logS = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, v0)
    for j in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        logS += (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1[:, j]
        v += kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * w2[:, j]
    return np.exp(logS)


def mc_price(
    model_name: str,
    params: dict,
    K: np.ndarray,
    T: np.ndarray,
    n_paths: int = 100_000,
    n_steps: int = 200,
    seed: int = 20251,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Monte Carlo European-call prices for ``model_name`` at the given (K, T).

    Parameters
    ----------
    model_name : {'BS', 'LocalVol', 'Heston'}
    params     : model parameter dict (S0, r, and model-specific params).
    K, T       : strike and maturity arrays (broadcast to a common 1-D shape).
    n_paths    : number of simulated paths (antithetic counts toward this total).
    n_steps    : Euler steps (ignored for BS, which is sampled exactly).
    seed       : RNG seed; hold constant across optimiser iterations for CRN.
    antithetic : use antithetic variates.

    Returns
    -------
    np.ndarray of prices, same length as the flattened (K, T).
    """
    K = np.asarray(K, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    if K.shape != T.shape:
        K, T = np.broadcast_arrays(K, T)
        K = K.ravel().copy()
        T = T.ravel().copy()

    S0 = float(params.get("S0", 1.0))
    r = float(params.get("r", 0.02))
    out = np.zeros_like(K)

    for t in np.unique(T):
        idx = np.where(T == t)[0]
        if t < 1e-8:
            out[idx] = np.maximum(S0 - K[idx], 0.0)
            continue
        # Fresh generator seeded identically per call -> common random numbers.
        rng = np.random.default_rng(seed)
        if model_name == "BS":
            sigma = float(params.get("sigma", 0.2))
            S_T = _terminal_bs(S0, r, sigma, t, n_paths, rng, antithetic)
        elif model_name == "LocalVol":
            S_T = _terminal_localvol(
                S0, r,
                float(params.get("sigma0", params.get("sigma", 0.2))),
                float(params.get("alpha", 0.0)),
                float(params.get("beta", 0.0)),
                t, n_paths, n_steps, rng, antithetic,
            )
        elif model_name == "Heston":
            S_T = _terminal_heston(
                S0, r,
                float(params.get("kappa", 2.0)),
                float(params.get("theta", 0.04)),
                float(params.get("xi", 0.5)),
                float(params.get("rho", -0.5)),
                float(params.get("v0", 0.04)),
                t, n_paths, n_steps, rng, antithetic,
            )
        else:
            raise ValueError(f"Unknown model for MC pricing: {model_name}")

        disc = np.exp(-r * t)
        payoff_all = np.maximum(S_T[:, None] - K[idx][None, :], 0.0)
        out[idx] = disc * payoff_all.mean(axis=0)

    return out

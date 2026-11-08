"""
Tests for the Monte Carlo pricing used by the MC calibration baseline.

These guard against the regression where the "MC" baseline silently reduced to
the deterministic analytic pricer (making ``classical_bs`` and ``classical_mc_bs``
identical). The key checks:

* BS Monte Carlo matches the analytic BS price within MC tolerance.
* LocalVol(alpha=0, beta=0) reduces to BS (GBM).
* Heston(xi=0, v0=theta=sigma^2) reduces to BS (constant variance).
* Prices are arbitrage-sane: decreasing in strike and bounded by [0, S0].
"""

import numpy as np
import pytest

from src.models.mc_pricing import mc_price
from src.models.pricing import BSPricer


S0, R, SIGMA = 1.0, 0.02, 0.2
K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
T = np.full_like(K, 1.0)


@pytest.fixture(scope="module")
def bs_reference():
    return BSPricer(S0=S0, r=R, sigma=SIGMA).price(K, T)


def test_bs_mc_matches_analytic(bs_reference):
    mc = mc_price("BS", {"S0": S0, "r": R, "sigma": SIGMA}, K, T,
                  n_paths=200_000, seed=1)
    assert np.max(np.abs(mc - bs_reference)) < 3e-3


def test_localvol_reduces_to_bs(bs_reference):
    mc = mc_price("LocalVol",
                  {"S0": S0, "r": R, "sigma0": SIGMA, "alpha": 0.0, "beta": 0.0},
                  K, T, n_paths=200_000, n_steps=100, seed=1)
    assert np.max(np.abs(mc - bs_reference)) < 5e-3


def test_heston_reduces_to_bs(bs_reference):
    mc = mc_price("Heston",
                  {"S0": S0, "r": R, "kappa": 2.0, "theta": SIGMA**2,
                   "xi": 0.0, "rho": -0.5, "v0": SIGMA**2},
                  K, T, n_paths=200_000, n_steps=200, seed=1)
    assert np.max(np.abs(mc - bs_reference)) < 5e-3


@pytest.mark.parametrize("model,params", [
    ("BS", {"S0": S0, "r": R, "sigma": SIGMA}),
    ("LocalVol", {"S0": S0, "r": R, "sigma0": SIGMA, "alpha": 0.5, "beta": 0.0}),
    ("Heston", {"S0": S0, "r": R, "kappa": 2.0, "theta": 0.04,
                "xi": 0.5, "rho": -0.5, "v0": 0.04}),
])
def test_price_is_arbitrage_sane(model, params):
    p = mc_price(model, params, K, T, n_paths=100_000, n_steps=100, seed=7)
    assert np.all(np.diff(p) < 0), "call price must decrease in strike"
    assert np.all(p <= S0 + 1e-9) and np.all(p >= -1e-9)


def test_mc_is_not_the_deterministic_pricer(bs_reference):
    """The MC estimate must carry genuine sampling error (the bug was zero error)."""
    mc = mc_price("BS", {"S0": S0, "r": R, "sigma": SIGMA}, K, T,
                  n_paths=50_000, seed=3)
    rmse = np.sqrt(np.mean((mc - bs_reference) ** 2))
    assert rmse > 0.0


def test_mc_calibration_wrappers_importable():
    """E1 plumbing: all three MC calibration wrappers must exist and be callable."""
    from src.models.calibration import (
        calibrate_mc_bs, calibrate_mc_localvol, calibrate_mc_heston,
    )
    assert callable(calibrate_mc_bs)
    assert callable(calibrate_mc_localvol)
    assert callable(calibrate_mc_heston)

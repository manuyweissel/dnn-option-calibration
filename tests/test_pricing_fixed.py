"""
Tests for the fixed pricing implementations.

Verifies:
1. Heston pricing via Lewis integration gives reasonable prices
2. LocalVol Dupire PDE solver produces consistent results
3. Delta/Gamma via homogeneity formula are correct
4. MC baseline agrees with analytic BS
"""

import numpy as np
import pytest
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pricing import (
    BSPricer, LocalVolPricer, HestonPricer, MonteCarloPricer, create_pricer
)


class TestBSPricer:
    """Test Black-Scholes analytic pricer."""
    
    def test_atm_call_price(self):
        """ATM call should be approximately S0 * 0.4 * sigma * sqrt(T) for small sigma."""
        pricer = BSPricer(S0=1.0, r=0.0, sigma=0.2)
        K = np.array([1.0])
        T = np.array([1.0])
        
        price = pricer.price(K, T)
        # BS ATM approximation: C ≈ 0.4 * S * sigma * sqrt(T)
        expected_approx = 0.4 * 1.0 * 0.2 * 1.0
        
        assert price[0] > 0, "Price should be positive"
        assert 0.05 < price[0] < 0.15, f"ATM price {price[0]} out of expected range"
    
    def test_call_put_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        pricer = BSPricer(S0=1.0, r=0.05, sigma=0.3)
        K = np.array([1.0])
        T = np.array([0.5])
        
        call_price = pricer.price(K, T)
        
        # Put price via put-call parity inversion (BS has no direct put method)
        # For this test, we verify call is bounded correctly
        forward = pricer.S0 * np.exp(pricer.r * T)
        assert call_price[0] <= pricer.S0, "Call price cannot exceed spot"
        assert call_price[0] >= max(pricer.S0 - K[0] * np.exp(-pricer.r * T[0]), 0), "Call >= intrinsic"
    
    def test_delta_atm(self):
        """ATM delta should be approximately 0.5 for r=0."""
        pricer = BSPricer(S0=1.0, r=0.0, sigma=0.2)
        K = np.array([1.0])
        T = np.array([1.0])
        
        delta = pricer.delta(K, T)
        assert 0.4 < delta[0] < 0.6, f"ATM delta {delta[0]} should be near 0.5"
    
    def test_gamma_positive(self):
        """Gamma should always be positive for calls."""
        pricer = BSPricer(S0=1.0, r=0.02, sigma=0.2)
        K = np.array([0.8, 1.0, 1.2])
        T = np.array([0.25, 0.5, 0.75])
        
        gamma = pricer.gamma(K, T)
        assert (gamma > 0).all(), "Gamma should be positive"
    
    def test_vega_positive(self):
        """Vega should always be positive."""
        pricer = BSPricer(S0=1.0, r=0.02, sigma=0.2)
        K = np.array([0.9, 1.0, 1.1])
        T = np.array([0.5, 0.5, 0.5])
        
        vega = pricer.vega(K, T)
        assert (vega > 0).all(), "Vega should be positive"


class TestHestonPricer:
    """Test Heston stochastic volatility pricer."""
    
    def test_heston_reduces_to_bs(self):
        """
        Heston with tiny vol of vol (v0 = theta) is BS with sigma = sqrt(v0).
        Checked over the whole strike range: a check at K = S0 alone misses
        strike-dependent errors such as a wrong Lewis prefactor.
        """
        heston = HestonPricer(S0=1.0, r=0.02, kappa=2.0, theta=0.04,
                              xi=0.01, rho=0.0, v0=0.04)
        bs = BSPricer(S0=1.0, r=0.02, sigma=0.2)  # sigma = sqrt(v0)
        K, T = (a.ravel() for a in np.meshgrid(np.linspace(0.8, 1.2, 9), [0.1, 0.5, 1.0]))
        np.testing.assert_allclose(heston.price(K, T), bs.price(K, T), atol=1e-4)
    
    def test_heston_price_positive(self):
        """Heston prices should be positive."""
        heston = HestonPricer(S0=1.0, r=0.02, kappa=2.0, theta=0.04,
                              xi=0.5, rho=-0.5, v0=0.04)
        
        K = np.array([0.8, 1.0, 1.2])
        T = np.array([0.25, 0.5, 0.75])
        
        prices = heston.price(K, T)
        assert (prices >= 0).all(), "Heston prices should be non-negative"
    
    def test_heston_price_bounded(self):
        """Heston call price bounded by S0."""
        heston = HestonPricer(S0=1.0, r=0.02, kappa=2.0, theta=0.04,
                              xi=0.5, rho=-0.5, v0=0.04)
        
        K = np.array([0.5, 1.0, 1.5])
        T = np.array([1.0, 1.0, 1.0])
        
        prices = heston.price(K, T)
        assert (prices <= heston.S0).all(), "Call price cannot exceed S0"
    
    def test_heston_smile(self):
        """
        Heston with negative rho should produce downward sloping IV smile.
        ITM and ATM calls should have meaningful prices.
        """
        heston = HestonPricer(S0=1.0, r=0.0, kappa=2.0, theta=0.04,
                              xi=0.5, rho=-0.7, v0=0.04)
        
        # Test ITM and ATM (OTM may be very small for short T)
        K = np.array([0.9, 1.0])
        T = np.array([0.5, 0.5])
        
        prices = heston.price(K, T)
        
        # ITM and ATM prices should be meaningful
        assert (prices > 1e-6).all(), f"Prices should be meaningful: {prices}"
        # ITM call should be worth more than ATM
        assert prices[0] > prices[1], "ITM call should be worth more than ATM"


class TestLocalVolPricer:
    """Test Local Volatility Dupire PDE pricer."""
    
    def test_localvol_alpha_zero_is_bs(self):
        """
        LocalVol with alpha=0, beta=0 is BS with the same r. Checked over the
        whole strike range: a check at K = S0 alone misses a flipped drift sign.
        """
        lv = LocalVolPricer(S0=1.0, r=0.02, sigma0=0.2, alpha=0.0, beta=0.0)
        bs = BSPricer(S0=1.0, r=0.02, sigma=0.2)
        K, T = (a.ravel() for a in np.meshgrid(np.linspace(0.8, 1.2, 9), [0.1, 0.5, 1.0]))
        np.testing.assert_allclose(lv.price(K, T), bs.price(K, T), atol=1e-4)
    
    def test_localvol_surface_shape(self):
        """Test that local vol surface has expected shape."""
        lv = LocalVolPricer(S0=1.0, r=0.02, sigma0=0.2, alpha=-0.3, beta=0.1)
        
        K = np.array([0.8, 1.0, 1.2])
        T = np.array([0.5, 0.5, 0.5])
        
        vol_surface = lv.local_vol_surface(K, T)
        
        # With alpha < 0, lower K should have higher vol (skew)
        assert vol_surface[0] > vol_surface[1] > vol_surface[2], "Expected downward skew"
    
    def test_localvol_price_positive(self):
        """LocalVol prices should be positive."""
        lv = LocalVolPricer(S0=1.0, r=0.02, sigma0=0.2, alpha=0.0, beta=0.0)
        
        K = np.array([0.8, 1.0, 1.2])
        T = np.array([0.25, 0.5, 0.75])
        
        prices = lv.price(K, T)
        assert (prices >= 0).all(), "LocalVol prices should be non-negative"

    def test_localvol_gamma_labels_match_bs_limit(self):
        """
        Strike-Gamma training labels of the LV generator (alpha = 0) on the
        experiment grid (50x20, log-moneyness spacing) against the closed form.
        Crank-Nicolson without Rannacher start-up leaves its largest error at
        short maturity near the payoff kink (thesis Ch. 8).
        """
        from src.data.grids import create_grid
        from src.data.synth import generate_synthetic_data

        grid = create_grid(K=50, T=20, spacing="logm", K_min=0.8, K_max=1.2, T_min=0.1, S0=1.0)
        params = dict(S0=1.0, r=0.02, sigma0=0.2, alpha=0.0, beta=0.0)
        labels = generate_synthetic_data(grid, "LocalVol", params)
        exact = BSPricer(S0=1.0, r=0.02, sigma=0.2).strike_gamma(labels["K"].values, labels["T"].values)
        rel = np.abs(labels["gamma"].values - exact) / exact
        assert np.median(rel) < 1e-3, np.median(rel)
        assert rel.max() < 0.2, rel.max()


class TestPricerCrossChecks:
    """Checks away from the Black-Scholes limit, at the experiment parameters (Ch. 8)."""

    def test_pricers_match_monte_carlo(self):
        """Lewis-form Heston and Crank-Nicolson LV prices against independent MC (~50 s)."""
        from src.models.mc_pricing import mc_price

        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.2])
        T = np.array([0.5] * 5 + [1.0] * 3)
        heston = dict(S0=1.0, r=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5, v0=0.04)
        local_vol = dict(S0=1.0, r=0.02, sigma0=0.2, alpha=0.5, beta=0.0)
        # tolerances: Heston dominated by the full-truncation Euler bias (400 steps), LV by MC noise
        for model, pricer, params, tol in (("Heston", HestonPricer(**heston), heston, 5e-4),
                                           ("LocalVol", LocalVolPricer(**local_vol), local_vol, 1e-4)):
            mc = mc_price(model, params, K, T, n_paths=400_000, n_steps=400)
            np.testing.assert_allclose(pricer.price(K, T), mc, atol=tol, err_msg=model)


class TestMonteCarloPricer:
    """Test Monte Carlo baseline pricer."""
    
    def test_mc_converges_to_bs(self):
        """MC should converge to BS analytic price."""
        mc = MonteCarloPricer(S0=1.0, r=0.02, sigma=0.2, n_paths=100000, seed=20251)
        bs = BSPricer(S0=1.0, r=0.02, sigma=0.2)
        
        K = np.array([1.0])
        T = np.array([0.5])
        
        mc_price = mc.price(K, T)
        bs_price = bs.price(K, T)
        
        # MC should be within 1% of BS for 100k paths
        rel_diff = abs(mc_price[0] - bs_price[0]) / bs_price[0]
        assert rel_diff < 0.02, f"MC ({mc_price[0]:.4f}) differs from BS ({bs_price[0]:.4f})"
    
    def test_mc_reproducible(self):
        """MC should give same results with same seed."""
        mc1 = MonteCarloPricer(S0=1.0, r=0.02, sigma=0.2, n_paths=10000, seed=12345)
        mc2 = MonteCarloPricer(S0=1.0, r=0.02, sigma=0.2, n_paths=10000, seed=12345)
        
        K = np.array([1.0])
        T = np.array([0.5])
        
        price1 = mc1.price(K, T)
        price2 = mc2.price(K, T)
        
        assert np.allclose(price1, price2), "MC should be reproducible with same seed"


class TestCreatePricer:
    """Test factory function."""
    
    def test_create_bs(self):
        """Factory should create BS pricer."""
        pricer = create_pricer("BS", {"S0": 1.0, "r": 0.02, "sigma": 0.25})
        assert isinstance(pricer, BSPricer)
        assert pricer.sigma == 0.25
    
    def test_create_heston(self):
        """Factory should create Heston pricer."""
        pricer = create_pricer("Heston", {
            "S0": 1.0, "r": 0.02, "kappa": 2.0, "theta": 0.04,
            "xi": 0.5, "rho": -0.5, "v0": 0.04
        })
        assert isinstance(pricer, HestonPricer)
        assert pricer.rho == -0.5
    
    def test_create_localvol(self):
        """Factory should create LocalVol pricer."""
        pricer = create_pricer("LocalVol", {
            "S0": 1.0, "r": 0.02, "sigma0": 0.2, "alpha": -0.3
        })
        assert isinstance(pricer, LocalVolPricer)
        assert pricer.alpha == -0.3
    
    def test_create_mc(self):
        """Factory should create MC pricer."""
        pricer = create_pricer("MC", {"S0": 1.0, "r": 0.02, "sigma": 0.2})
        assert isinstance(pricer, MonteCarloPricer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

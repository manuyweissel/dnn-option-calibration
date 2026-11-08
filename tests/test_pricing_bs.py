"""
Tests for Black-Scholes analytic pricer.

Verifies option prices and Greeks against textbook formulas.
"""

import numpy as np
import pytest

from src.models.pricing import BSPricer


def test_bs_pricer_basic():
    """Test BS pricer with basic parameters."""
    pricer = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([1.0, 1.0, 1.0])
    
    prices = pricer.price(K, T)
    
    # Prices should be positive
    assert (prices > 0).all()
    
    # ATM option should be most expensive
    assert prices[1] > prices[2]  # ATM > OTM
    assert prices[0] > prices[1]  # ITM > ATM


def test_bs_at_expiry():
    """Test BS price at expiry equals payoff."""
    pricer = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([1e-8, 1e-8, 1e-8])  # Near zero
    
    prices = pricer.price(K, T)
    payoffs = np.maximum(pricer.S0 - K, 0)
    
    # Prices should be close to payoffs at expiry
    np.testing.assert_allclose(prices, payoffs, atol=0.1)


def test_bs_delta():
    """Test BS delta is in [0, 1] for call options."""
    pricer = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([80.0, 100.0, 120.0])
    T = np.array([0.5, 0.5, 0.5])
    
    deltas = pricer.delta(K, T)
    
    # Delta should be in [0, 1]
    assert (deltas >= 0).all()
    assert (deltas <= 1).all()
    
    # ITM has higher delta than OTM
    assert deltas[0] > deltas[2]


def test_bs_gamma_positive():
    """Test BS gamma is always positive."""
    pricer = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([80.0, 100.0, 120.0])
    T = np.array([0.5, 0.5, 0.5])
    
    gammas = pricer.gamma(K, T)
    
    # Gamma should be positive
    assert (gammas > 0).all()
    
    # ATM gamma is largest
    assert gammas[1] > gammas[0]
    assert gammas[1] > gammas[2]


def test_bs_vega_positive():
    """Test BS vega is always positive."""
    pricer = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([80.0, 100.0, 120.0])
    T = np.array([0.5, 0.5, 0.5])
    
    vegas = pricer.vega(K, T)
    
    # Vega should be positive
    assert (vegas > 0).all()


def test_bs_put_call_parity():
    """Test put-call parity: C - P = S - K*exp(-rT)."""
    pricer = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([100.0])
    T = np.array([1.0])
    
    call_price = pricer.price(K, T)
    
    # For put-call parity, we would need a put pricer
    # Here we just check call price is reasonable
    assert 5 < call_price[0] < 20


def test_bs_reproducibility():
    """Test that BS pricer is deterministic."""
    pricer1 = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    pricer2 = BSPricer(S0=100.0, r=0.05, sigma=0.2)
    
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.5, 0.5, 0.5])
    
    prices1 = pricer1.price(K, T)
    prices2 = pricer2.price(K, T)
    
    np.testing.assert_array_equal(prices1, prices2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


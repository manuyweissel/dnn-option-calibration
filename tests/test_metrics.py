"""
Tests for evaluation metrics.

Verifies RMSE, MAE, and other metrics compute correctly.
"""

import numpy as np
import pandas as pd
import pytest

from src.exp.metrics import (
    rmse_price,
    mae_greeks,
    rpe,
    param_rel_err,
    compute_all_metrics,
)


def test_rmse_price_perfect():
    """Test RMSE is zero for perfect predictions."""
    df_pred = pd.DataFrame({'price': [1.0, 2.0, 3.0]})
    df_true = pd.DataFrame({'price': [1.0, 2.0, 3.0]})
    
    error = rmse_price(df_pred, df_true)
    
    assert error == 0.0


def test_rmse_price_nonzero():
    """Test RMSE is positive for imperfect predictions."""
    df_pred = pd.DataFrame({'price': [1.0, 2.0, 3.0]})
    df_true = pd.DataFrame({'price': [1.1, 1.9, 3.1]})
    
    error = rmse_price(df_pred, df_true)
    
    assert error > 0
    assert error < 0.2  # Should be small


def test_mae_greeks():
    """Test Greek MAE computation."""
    df_pred = pd.DataFrame({
        'delta': [0.5, 0.6, 0.7],
        'gamma': [0.1, 0.2, 0.3],
    })
    df_true = pd.DataFrame({
        'delta': [0.52, 0.58, 0.71],
        'gamma': [0.11, 0.19, 0.31],
    })
    
    errors = mae_greeks(df_pred, df_true, greeks=['delta', 'gamma'])
    
    assert 'mae_delta' in errors
    assert 'mae_gamma' in errors
    assert 'mae_avg' in errors
    
    assert errors['mae_delta'] > 0
    assert errors['mae_gamma'] > 0
    assert errors['mae_avg'] > 0


def test_rpe():
    """Test relative percentage error."""
    df_pred = pd.DataFrame({'price': [1.0, 2.0, 3.0]})
    df_true = pd.DataFrame({'price': [1.1, 2.2, 3.3]})
    
    error = rpe(df_pred, df_true)
    
    # Relative error should be around 10%
    assert 0.08 < error < 0.12


def test_param_rel_err_zero():
    """Test parameter error is zero for perfect match."""
    pred = {'sigma': 0.2, 'r': 0.02}
    true = {'sigma': 0.2, 'r': 0.02}
    
    error = param_rel_err(pred, true)
    
    assert error == 0.0


def test_param_rel_err_nonzero():
    """Test parameter error is positive for mismatch."""
    pred = {'sigma': 0.21, 'r': 0.022}
    true = {'sigma': 0.20, 'r': 0.020}
    
    error = param_rel_err(pred, true)
    
    assert error > 0
    assert error < 0.2


def test_compute_all_metrics():
    """Test computing all metrics at once."""
    df_pred = pd.DataFrame({
        'price': [1.0, 2.0, 3.0],
        'delta': [0.5, 0.6, 0.7],
        'gamma': [0.1, 0.2, 0.3],
    })
    df_true = pd.DataFrame({
        'price': [1.05, 2.05, 3.05],
        'delta': [0.52, 0.62, 0.72],
        'gamma': [0.11, 0.21, 0.31],
    })
    
    metrics = compute_all_metrics(df_pred, df_true)
    
    assert 'rmse_price' in metrics
    assert 'mae_avg' in metrics
    assert all(v >= 0 for v in metrics.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


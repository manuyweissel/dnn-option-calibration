"""
Tests for loss function shapes and outputs.

Verifies that loss functions return scalars without NaNs.
"""

import torch
import pytest

from src.models.nets import MLP
from src.train.losses import (
    price_loss,
    greeks_loss,
    pde_loss,
    bc_loss,
    noarb_loss,
)


def test_price_loss_shape():
    """Test price loss returns scalar."""
    pred = torch.randn(10, 1)
    target = torch.randn(10, 1)
    
    loss = price_loss(pred, target)
    
    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0  # Non-negative
    assert not torch.isnan(loss)


def test_greeks_loss_shape():
    """Test Greeks loss returns scalar."""
    pred_greeks = {
        'delta': torch.randn(10, 1),
        'gamma': torch.randn(10, 1),
    }
    target_greeks = {
        'delta': torch.randn(10, 1),
        'gamma': torch.randn(10, 1),
    }
    
    loss = greeks_loss(pred_greeks, target_greeks, keys=['delta', 'gamma'])
    
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_pde_loss_shape():
    """Test PDE loss returns scalar."""
    model = MLP(L=2, W=16, activation='softplus')
    
    K = torch.randn(20, 1, requires_grad=True)
    T = torch.rand(20, 1, requires_grad=True)
    
    loss = pde_loss(model, K, T, r=0.02, sigma=0.2)
    
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_bc_loss_shape():
    """Test boundary condition loss returns scalar."""
    model = MLP(L=2, W=16, activation='softplus')
    
    K = torch.tensor([[0.9], [1.0], [1.1]])
    T = torch.tensor([[0.01], [0.01], [0.01]])
    
    loss = bc_loss(model, K, T, S0=1.0)
    
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_noarb_loss_shape():
    """Test no-arbitrage loss returns scalar."""
    model = MLP(L=2, W=16, activation='softplus')
    
    K = torch.randn(15, 1, requires_grad=True)
    T = torch.rand(15, 1)
    
    loss = noarb_loss(model, K, T)
    
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_loss_backward():
    """Test that losses support backpropagation."""
    model = MLP(L=2, W=16, activation='softplus')
    
    K = torch.randn(10, 1)
    T = torch.rand(10, 1)
    
    pred = model(K, T)
    target = torch.randn(10, 1)
    
    loss = price_loss(pred, target)
    
    # Backward should not raise
    loss.backward()
    
    # Gradients should exist
    assert any(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


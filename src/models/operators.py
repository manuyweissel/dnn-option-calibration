"""
PDE operators for option pricing.

Implements model-specific PDE operators:
- Black-Scholes operator B[u]
- Dupire forward operator for local volatility
- Heston 2D PDE operator (simplified)

And utilities for computing PDE residuals via automatic differentiation.
"""

from typing import Literal, Optional, Callable
import torch
import torch.nn as nn


def bs_operator(
    u: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    sigma: float,
) -> torch.Tensor:
    """
    Compute Black-Scholes forward PDE operator B[u] in strike space.
    
    The BS forward PDE (Dupire form with constant vol) is:
        ∂C/∂T = (1/2)σ²K² ∂²C/∂K² − rK ∂C/∂K
    
    Residual: ∂C/∂T − (1/2)σ²K² ∂²C/∂K² + rK ∂C/∂K  (= 0 for exact solution).
    
    Args:
        u: Option values, shape (batch_size, 1).
        K: Strike prices, shape (batch_size, 1), requires_grad=True.
        T: Times to maturity, shape (batch_size, 1), requires_grad=True.
        r: Risk-free rate.
        sigma: Volatility.
        
    Returns:
        PDE residual, shape (batch_size, 1).
        
    Example:
        >>> K = torch.tensor([[1.0]], requires_grad=True)
        >>> T = torch.tensor([[0.5]], requires_grad=True)
        >>> u = model(K, T)
        >>> residual = bs_operator(u, K, T, r=0.02, sigma=0.2)
    """
    # Compute first derivatives
    du_dK = torch.autograd.grad(
        u.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    du_dT = torch.autograd.grad(
        u.sum(), T, create_graph=True, retain_graph=True
    )[0]
    
    # Compute second derivative w.r.t. K
    d2u_dK2 = torch.autograd.grad(
        du_dK.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # BS operator in forward form (Dupire-like):
    # ∂C/∂T = (1/2)σ²K² ∂²C/∂K² - rK ∂C/∂K
    # Residual = ∂C/∂T - (1/2)σ²K² ∂²C/∂K² + rK ∂C/∂K
    residual = du_dT - 0.5 * sigma**2 * K**2 * d2u_dK2 + r * K * du_dK
    
    return residual


def dupire_operator(
    u: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    local_vol_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute Dupire forward PDE operator for local volatility models.
    
    The Dupire PDE is:
        ∂C/∂T = (1/2)σ²(K,T) K² ∂²C/∂K² - rK ∂C/∂K
    
    Args:
        u: Option values, shape (batch_size, 1).
        K: Strike prices, shape (batch_size, 1), requires_grad=True.
        T: Times to maturity, shape (batch_size, 1), requires_grad=True.
        r: Risk-free rate.
        local_vol_fn: Function that returns σ(K, T).
        
    Returns:
        PDE residual, shape (batch_size, 1).
    """
    # Compute first derivatives
    du_dK = torch.autograd.grad(
        u.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    du_dT = torch.autograd.grad(
        u.sum(), T, create_graph=True, retain_graph=True
    )[0]
    
    # Compute second derivative w.r.t. K
    d2u_dK2 = torch.autograd.grad(
        du_dK.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # Get local volatility at (K, T)
    sigma_local = local_vol_fn(K, T)
    sigma_sq = sigma_local ** 2
    
    # Dupire forward PDE residual
    residual = du_dT - 0.5 * sigma_sq * K**2 * d2u_dK2 + r * K * du_dK
    
    return residual


def heston_operator_simplified(  # NOTE: unused — Heston experiments use lambda_pde=0
    u: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    v: torch.Tensor,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> torch.Tensor:
    """
    Simplified Heston PDE operator (1D projection).
    
    The full Heston PDE is 2D in (S, V). This implements a simplified
    1D version by assuming V follows its expected path.
    
    For the full 2D PDE:
        ∂u/∂t + (1/2)vS²∂²u/∂S² + ρξvS∂²u/∂S∂v + (1/2)ξ²v∂²u/∂v²
        + rS∂u/∂S + κ(θ-v)∂u/∂v - ru = 0
    
    Args:
        u: Option values.
        K: Strike prices.
        T: Times to maturity.
        r: Risk-free rate.
        v: Variance values (could be learned or from trajectory).
        kappa: Mean reversion rate.
        theta: Long-term variance.
        xi: Vol of vol.
        rho: Correlation.
        
    Returns:
        PDE residual.
    """
    # Compute derivatives in K direction
    du_dK = torch.autograd.grad(
        u.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    du_dT = torch.autograd.grad(
        u.sum(), T, create_graph=True, retain_graph=True
    )[0]
    
    d2u_dK2 = torch.autograd.grad(
        du_dK.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # Simplified 1D residual using average variance
    # This is essentially BS with time-varying vol
    residual = du_dT - 0.5 * v * K**2 * d2u_dK2 + r * K * du_dK
    
    return residual


def pde_residual(
    model: nn.Module,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    sigma: float,
    model_type: Literal["BS", "LocalVol", "Heston"] = "BS",
    local_vol_fn: Optional[Callable] = None,
    heston_params: Optional[dict] = None,
) -> torch.Tensor:
    """
    Compute PDE residual for collocation points.
    
    Dispatches to appropriate PDE operator based on model type.
    
    Args:
        model: Neural network model.
        K: Collocation strike prices, shape (N, 1).
        T: Collocation maturities, shape (N, 1).
        r: Risk-free rate.
        sigma: Volatility (for BS) or base vol (for LocalVol).
        model_type: Type of pricing model.
        local_vol_fn: Local vol function (for LocalVol model).
        heston_params: Heston parameters dict (for Heston model).
        
    Returns:
        PDE residual values, shape (N, 1).
    """
    # Ensure gradients are enabled
    K = K.requires_grad_(True)
    T = T.requires_grad_(True)
    
    # Forward pass
    u = model(K, T)
    
    if model_type == "BS":
        return bs_operator(u, K, T, r, sigma)
    
    elif model_type == "LocalVol":
        if local_vol_fn is None:
            # Default local vol: constant
            local_vol_fn = lambda k, t: torch.full_like(k, sigma)
        return dupire_operator(u, K, T, r, local_vol_fn)
    
    elif model_type == "Heston":
        if heston_params is None:
            heston_params = {
                "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.5, "v0": 0.04
            }
        # Use initial variance as proxy
        v = torch.full_like(K, heston_params.get("v0", 0.04))
        return heston_operator_simplified(
            u, K, T, r, v,
            heston_params["kappa"],
            heston_params["theta"],
            heston_params["xi"],
            heston_params["rho"],
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_derivatives(
    u: torch.Tensor,
    x: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Compute derivatives of u with respect to x via AD.
    
    Args:
        u: Function values, shape (batch_size, 1).
        x: Input variable, shape (batch_size, 1), requires_grad=True.
        order: Derivative order (1 or 2).
        
    Returns:
        Derivative values, shape (batch_size, 1).
    """
    if order == 1:
        du_dx = torch.autograd.grad(
            u.sum(), x, create_graph=True, retain_graph=True
        )[0]
        return du_dx
    elif order == 2:
        du_dx = torch.autograd.grad(
            u.sum(), x, create_graph=True, retain_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx.sum(), x, create_graph=True, retain_graph=True
        )[0]
        return d2u_dx2
    else:
        raise ValueError(f"Unsupported order: {order}")


def no_arbitrage_penalty(
    model: nn.Module,
    K: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """
    Compute no-arbitrage penalty.
    
    Enforces:
    - Monotonicity: ∂C/∂K < 0 (call price decreases with strike)
    - Convexity: ∂²C/∂K² > 0 (butterfly spread non-negative)
    - Calendar: ∂C/∂T > 0 (call price increases with maturity for non-div)
    
    Args:
        model: Neural network.
        K: Strike prices.
        T: Maturities.
        
    Returns:
        Scalar penalty loss.
    """
    K = K.requires_grad_(True)
    T = T.requires_grad_(True)
    
    price = model(K, T)
    
    # First derivative w.r.t. K (should be negative)
    dC_dK = torch.autograd.grad(
        price.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # Second derivative w.r.t. K (should be positive)
    d2C_dK2 = torch.autograd.grad(
        dC_dK.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # First derivative w.r.t. T (should be positive for non-dividend)
    dC_dT = torch.autograd.grad(
        price.sum(), T, create_graph=True, retain_graph=True
    )[0]
    
    # Squared ReLU penalties (Ch6 eq 6.6)
    mono_penalty = torch.relu(dC_dK).pow(2).mean()
    convex_penalty = torch.relu(-d2C_dK2).pow(2).mean()
    calendar_penalty = torch.relu(-dC_dT).pow(2).mean()

    return mono_penalty + convex_penalty + calendar_penalty

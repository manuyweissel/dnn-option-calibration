"""
Loss functions for training differential neural networks.

Implements price loss, Greek loss, PDE residual, boundary conditions,
no-arbitrage penalties, and composite loss with curriculum scheduling.

Supports model-specific PDE operators (BS, LocalVol, Heston).
"""

from typing import Literal, Optional, Callable
import torch
import torch.nn as nn

from ..models.operators import pde_residual, no_arbitrage_penalty


def price_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    loss_type: Literal["mse", "mae", "huber"] = "mse",
) -> torch.Tensor:
    """
    Loss for option prices.
    
    Args:
        pred: Predicted prices, shape (N, 1).
        target: Target prices, shape (N, 1).
        mask: Optional mask for missing values, shape (N, 1).
        loss_type: Type of loss function.
        
    Returns:
        Scalar loss.
    """
    if loss_type == "mse":
        diff = (pred - target) ** 2
    elif loss_type == "mae":
        diff = (pred - target).abs()
    elif loss_type == "huber":
        diff = torch.nn.functional.huber_loss(pred, target, reduction='none')
    else:
        diff = (pred - target) ** 2
    
    if mask is not None:
        loss = (diff * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = diff.mean()
    
    return loss


def greeks_loss(
    pred_greeks: dict[str, torch.Tensor],
    target_greeks: dict[str, torch.Tensor],
    keys: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Mean squared error loss for strike-space Greeks (Ch6 eq 6.2).

    When external ``weights`` are supplied (e.g. inverse-median weights from
    Ch8), the function returns the weighted sum of per-Greek MSEs without
    additional normalization, so the caller's weight semantics are preserved.
    When no weights are supplied, each Greek contributes equally (weight 1).

    Args:
        pred_greeks: Dict with keys ['delta', 'gamma', 'theta', ...].
        target_greeks: Dict with target Greek values.
        keys: List of keys to include (None = all available).
        weights: Optional per-Greek weights for scale normalisation.

    Returns:
        Scalar loss (weighted sum of per-Greek MSE).
    """
    if keys is None:
        keys = [k for k in pred_greeks.keys() if k in target_greeks]

    external_weights = weights is not None
    if weights is None:
        weights = {k: 1.0 for k in keys}

    total_loss = 0.0
    n_terms = 0

    for key in keys:
        if key in pred_greeks and key in target_greeks:
            pred = pred_greeks[key]
            target = target_greeks[key]

            if pred.abs().max() < 1e-10 and target.abs().max() < 1e-10:
                continue

            loss_k = (pred - target).pow(2).mean()
            w = weights.get(key, 1.0)
            total_loss += w * loss_k
            n_terms += 1

    if n_terms == 0:
        device = next(iter(pred_greeks.values())).device
        return torch.tensor(0.0, device=device)

    if external_weights:
        return total_loss
    return total_loss / n_terms


def pde_loss(
    model: nn.Module,
    K_colloc: torch.Tensor,
    T_colloc: torch.Tensor,
    r: float,
    sigma: float,
    model_type: Literal["BS", "LocalVol", "Heston"] = "BS",
    local_vol_fn: Optional[Callable] = None,
    heston_params: Optional[dict] = None,
) -> torch.Tensor:
    """
    PDE residual loss at collocation points.
    
    Uses model-specific PDE operators.
    
    Args:
        model: Neural network.
        K_colloc: Collocation strike prices, shape (N, 1).
        T_colloc: Collocation maturities, shape (N, 1).
        r: Risk-free rate.
        sigma: Volatility.
        model_type: Type of pricing model ('BS', 'LocalVol', 'Heston').
        local_vol_fn: Local volatility function (for LocalVol).
        heston_params: Heston parameters (for Heston).
        
    Returns:
        Scalar loss (MSE of PDE residual).
    """
    residual = pde_residual(
        model, K_colloc, T_colloc, r, sigma,
        model_type=model_type,
        local_vol_fn=local_vol_fn,
        heston_params=heston_params,
    )
    loss = (residual ** 2).mean()
    return loss


def bc_loss(
    model: nn.Module,
    K_bc: torch.Tensor,
    T_bc: torch.Tensor,
    S0: float = 1.0,
    r: float = 0.02,
) -> torch.Tensor:
    """
    Boundary condition loss.
    
    Enforces:
    1. Terminal payoff: C(K, T=0) = max(S0 - K, 0) for call options.
    2. Deep ITM limit: C(K→0, T) → S0
    3. Deep OTM limit: C(K→∞, T) → 0
    
    Args:
        model: Neural network.
        K_bc: Boundary strike prices, shape (N, 1).
        T_bc: Boundary maturities (should be close to 0), shape (N, 1).
        S0: Initial underlying price.
        r: Risk-free rate.
        
    Returns:
        Scalar loss.
    """
    # Terminal payoff condition (T ≈ 0)
    pred_terminal = model(K_bc, T_bc)
    payoff = torch.relu(S0 - K_bc)
    loss_terminal = ((pred_terminal - payoff) ** 2).mean()
    
    # Deep ITM condition (K → 0): C ≈ S0 - K*exp(-rT) ≈ S0
    K_itm = torch.full_like(K_bc, 0.01 * S0)  # Very low strike
    T_itm = T_bc
    pred_itm = model(K_itm, T_itm)
    target_itm = S0 - K_itm * torch.exp(-r * T_itm)
    loss_itm = ((pred_itm - target_itm) ** 2).mean()
    
    # Deep OTM condition (K → ∞): C → 0
    K_otm = torch.full_like(K_bc, 5.0 * S0)  # Very high strike
    T_otm = T_bc
    pred_otm = model(K_otm, T_otm)
    loss_otm = (pred_otm ** 2).mean()
    
    return loss_terminal + 0.1 * loss_itm + 0.1 * loss_otm


def noarb_loss(
    model: nn.Module,
    K: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """
    No-arbitrage penalty loss.
    
    Penalizes violations of:
    - Monotonicity: ∂C/∂K < 0 (call price decreases with strike)
    - Convexity: ∂²C/∂K² > 0 (call price is convex in strike)
    - Calendar: ∂C/∂T > 0 (for non-dividend paying stocks)
    
    Args:
        model: Neural network.
        K: Strike prices, shape (N, 1).
        T: Maturities, shape (N, 1).
        
    Returns:
        Scalar loss (penalty for violations).
    """
    return no_arbitrage_penalty(model, K, T)


def composite_loss(
    model: nn.Module,
    K_data: torch.Tensor,
    T_data: torch.Tensor,
    price_target: torch.Tensor,
    greeks_target: dict[str, torch.Tensor] | None,
    K_colloc: torch.Tensor,
    T_colloc: torch.Tensor,
    K_bc: torch.Tensor,
    T_bc: torch.Tensor,
    lambda_price: float,
    lambda_greeks: float,
    lambda_pde: float,
    lambda_bc: float,
    lambda_noarb: float,
    r: float,
    sigma: float,
    S0: float = 1.0,
    model_type: Literal["BS", "LocalVol", "Heston"] = "BS",
    local_vol_fn: Optional[Callable] = None,
    heston_params: Optional[dict] = None,
    greek_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Composite loss combining all loss terms.
    
    Total loss:
        L = λ_price * L_price + λ_greeks * L_greeks + λ_pde * L_pde 
            + λ_bc * L_bc + λ_noarb * L_noarb
    
    Args:
        model: Neural network.
        K_data, T_data: Data points (prices/Greeks).
        price_target: Target prices.
        greeks_target: Target Greeks (dict or None).
        K_colloc, T_colloc: Collocation points for PDE.
        K_bc, T_bc: Boundary condition points.
        lambda_*: Loss weights.
        r: Risk-free rate.
        sigma: Volatility.
        S0: Initial underlying price.
        model_type: Pricing model type.
        local_vol_fn: Local vol function (optional).
        heston_params: Heston params (optional).
        
    Returns:
        Tuple of (total_loss, loss_dict with individual components).
    """
    loss_dict = {}
    
    # Price loss
    price_pred = model(K_data, T_data)
    loss_price = price_loss(price_pred, price_target)
    loss_dict['price'] = loss_price.item()
    
    # Greeks loss
    loss_greeks_val = torch.tensor(0.0, device=K_data.device)
    if greeks_target is not None and lambda_greeks > 0:
        # Compute predicted Greeks using the model's method
        if hasattr(model, 'compute_greeks_for_loss'):
            pred_greeks = model.compute_greeks_for_loss(K_data, T_data, S0=S0)
        else:
            K_grad = K_data.requires_grad_(True)
            T_grad = T_data.requires_grad_(True)
            price_for_greeks = model(K_grad, T_grad)

            dC_dK = torch.autograd.grad(
                price_for_greeks.sum(), K_grad,
                create_graph=True, retain_graph=True
            )[0]
            d2C_dK2 = torch.autograd.grad(
                dC_dK.sum(), K_grad,
                create_graph=True, retain_graph=True
            )[0]
            dC_dT = torch.autograd.grad(
                price_for_greeks.sum(), T_grad,
                create_graph=True, retain_graph=True
            )[0]

            pred_greeks = {
                'delta': dC_dK,
                'gamma': d2C_dK2,
                'theta': -dC_dT,
            }
        
        if pred_greeks:
            loss_greeks_val = greeks_loss(pred_greeks, greeks_target, weights=greek_weights)
            loss_dict['greeks'] = loss_greeks_val.item()
    
    # PDE loss
    loss_pde_val = torch.tensor(0.0, device=K_data.device)
    if lambda_pde > 0:
        loss_pde_val = pde_loss(
            model, K_colloc, T_colloc, r, sigma,
            model_type=model_type,
            local_vol_fn=local_vol_fn,
            heston_params=heston_params,
        )
        loss_dict['pde'] = loss_pde_val.item()
    
    # Boundary condition loss
    loss_bc_val = torch.tensor(0.0, device=K_data.device)
    if lambda_bc > 0:
        loss_bc_val = bc_loss(model, K_bc, T_bc, S0=S0, r=r)
        loss_dict['bc'] = loss_bc_val.item()
    
    # No-arbitrage loss
    loss_noarb_val = torch.tensor(0.0, device=K_data.device)
    if lambda_noarb > 0:
        loss_noarb_val = noarb_loss(model, K_data, T_data)
        loss_dict['noarb'] = loss_noarb_val.item()
    
    # Total loss
    total = (
        lambda_price * loss_price
        + lambda_greeks * loss_greeks_val
        + lambda_pde * loss_pde_val
        + lambda_bc * loss_bc_val
        + lambda_noarb * loss_noarb_val
    )
    
    loss_dict['total'] = total.item()
    
    return total, loss_dict


def relative_price_loss(  # NOTE: unused — composite_loss uses MSE price_loss
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Relative price loss: |pred - target| / |target|.
    
    More appropriate when prices vary over several orders of magnitude.
    
    Args:
        pred: Predicted prices.
        target: Target prices.
        eps: Small constant to avoid division by zero.
        
    Returns:
        Scalar loss.
    """
    rel_error = (pred - target).abs() / (target.abs() + eps)
    return rel_error.mean()


def log_price_loss(  # NOTE: unused — composite_loss uses MSE price_loss
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Log price loss: (log(pred) - log(target))^2.
    
    Scale-invariant loss that emphasizes relative errors.
    
    Args:
        pred: Predicted prices (must be positive).
        target: Target prices (must be positive).
        eps: Small constant for numerical stability.
        
    Returns:
        Scalar loss.
    """
    log_pred = torch.log(pred + eps)
    log_target = torch.log(target + eps)
    return ((log_pred - log_target) ** 2).mean()

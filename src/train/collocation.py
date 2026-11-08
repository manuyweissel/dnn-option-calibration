"""
Collocation point sampling for PDE residual computation.

Uses Sobol quasi-random sequences (Ch8 sec 8.2) for low-discrepancy
coverage of the (K, T) domain, with optional residual-based adaptive
resampling after a warm-up period.
"""

import torch


_sobol_engine_cache: dict[str, torch.quasirandom.SobolEngine] = {}


def sample_collocation_points(
    n_points: int,
    K_min: float,
    K_max: float,
    T_min: float,
    T_max: float,
    device: str = "cpu",
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample Sobol quasi-random collocation points in the (K, T) domain.

    A persistent SobolEngine is used so successive calls produce
    non-overlapping low-discrepancy sequences within the same run.
    """
    key = f"{device}_{K_min}_{K_max}_{T_min}_{T_max}"
    if key not in _sobol_engine_cache:
        _sobol_engine_cache[key] = torch.quasirandom.SobolEngine(
            dimension=2, scramble=True,
        )

    sobol = _sobol_engine_cache[key]
    pts = sobol.draw(n_points).to(device)

    K = pts[:, 0:1] * (K_max - K_min) + K_min
    T = pts[:, 1:2] * (T_max - T_min) + T_min

    if requires_grad:
        K = K.requires_grad_(True)
        T = T.requires_grad_(True)

    return K, T


def reset_sobol_cache() -> None:
    """Reset cached Sobol engines (call between experiments)."""
    _sobol_engine_cache.clear()


def sample_boundary_points(
    n_points: int,
    K_min: float,
    K_max: float,
    T_boundary: float = 0.01,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample boundary points at small maturity (terminal condition).
    
    Args:
        n_points: Number of boundary points.
        K_min: Minimum strike.
        K_max: Maximum strike.
        T_boundary: Maturity value for boundary (close to 0).
        device: Device.
        
    Returns:
        Tuple of (K_bc, T_bc), each of shape (n_points, 1).
        
    Example:
        >>> K, T = sample_boundary_points(100, 0.7, 1.3)
        >>> assert (T == 0.01).all()
    """
    K = torch.rand(n_points, 1, device=device) * (K_max - K_min) + K_min
    T = torch.full((n_points, 1), T_boundary, device=device)
    
    return K, T


def sample_adaptive_collocation(
    model: torch.nn.Module,
    n_points: int,
    K_min: float,
    K_max: float,
    T_min: float,
    T_max: float,
    r: float,
    sigma: float,
    device: str = "cpu",
    top_k: int = 500,
    model_type: str = "BS",
    local_vol_fn=None,
    heston_params: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample collocation points adaptively based on PDE residual magnitude.
    
    First samples a large pool of candidates, then selects the top_k points
    with highest residuals for refinement.
    
    Args:
        model: Neural network.
        n_points: Initial candidate pool size.
        K_min, K_max: Strike range.
        T_min, T_max: Maturity range.
        r: Risk-free rate.
        sigma: Volatility.
        device: Device.
        top_k: Number of points to return.
        model_type: Pricing model type for PDE operator dispatch.
        local_vol_fn: Torch-compatible sigma(K, T) callable for LocalVol PDE.
        heston_params: Dict of Heston parameters for Heston PDE.
        
    Returns:
        Tuple of (K_colloc, T_colloc) with highest residuals.
    """
    from ..models.operators import pde_residual
    
    K_cand, T_cand = sample_collocation_points(
        n_points, K_min, K_max, T_min, T_max, device, requires_grad=True
    )
    
    residuals = pde_residual(
        model, K_cand, T_cand, r, sigma,
        model_type=model_type,
        local_vol_fn=local_vol_fn,
        heston_params=heston_params,
    )
    residual_magnitudes = residuals.detach().abs().squeeze()
    
    _, top_idx = torch.topk(residual_magnitudes, min(top_k, len(residual_magnitudes)))
    
    K_selected = K_cand[top_idx].requires_grad_(True)
    T_selected = T_cand[top_idx].requires_grad_(True)
    
    return K_selected, T_selected


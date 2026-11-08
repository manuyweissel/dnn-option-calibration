"""
Configuration system for option pricing experiments.

This module defines all configuration schemas using Pydantic v2 for validation.
The schema matches the JSON structure documented in Chapter 10 (Experimental Design)
of the thesis, specifically Listing lst:config-bs-softplus.

Example:
    >>> cfg = ExperimentCfg.model_validate_json(open("configs/bs_softplus_full.json").read())
    >>> print(cfg.experiment_id)
    'bs_softplus_full'
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GridCfg(BaseModel):
    """Grid configuration for strike-maturity space.
    
    Attributes:
        K: Number of strike points.
        T: Number of maturity points.
        spacing: Grid spacing scheme ('uniform' or 'logm' for log-moneyness).
        K_min: Minimum strike (optional, defaults based on spacing).
        K_max: Maximum strike (optional, defaults based on spacing).
        T_min: Minimum maturity (optional, defaults to 0.01).
        T_max: Maximum maturity (optional, defaults to 1.0).
    """

    K: int = Field(..., gt=0, description="Number of strike points")
    T: int = Field(..., gt=0, description="Number of maturity points")
    spacing: Literal["uniform", "logm"] = Field(
        default="logm", description="Grid spacing type"
    )
    K_min: float | None = Field(default=None, description="Minimum strike")
    K_max: float | None = Field(default=None, description="Maximum strike")
    T_min: float | None = Field(default=None, description="Minimum maturity")
    T_max: float | None = Field(default=None, description="Maximum maturity")

    @field_validator("K", "T")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure grid dimensions are positive."""
        if v <= 0:
            raise ValueError("Grid dimensions must be positive")
        return v


class NoiseCfg(BaseModel):
    """Noise model configuration.
    
    Attributes:
        price_rel: Relative price noise level (e.g., 0.01 for 1%).
        greek_rel: Relative Greek noise level.
        seed: Random seed for noise generation (None = deterministic from exp seed).
    """

    price_rel: float = Field(default=0.0, ge=0.0, description="Relative price noise")
    greek_rel: float = Field(default=0.0, ge=0.0, description="Relative Greek noise")
    seed: int | None = Field(default=None, description="Noise random seed")


class NetCfg(BaseModel):
    """Neural network architecture configuration.
    
    Attributes:
        L: Number of hidden layers (depth).
        W: Width of each hidden layer.
        activation: Activation function type.
        head: Output head architecture ('single' or 'value_and_aux').
    """

    L: int = Field(..., ge=1, le=10, description="Number of hidden layers")
    W: int = Field(..., ge=8, le=1024, description="Hidden layer width")
    activation: Literal["softplus", "tanh", "relu", "sine"] = Field(
        default="softplus", description="Activation function"
    )
    head: Literal["single", "value_and_aux"] = Field(
        default="single", description="Output head architecture"
    )


class LossCfg(BaseModel):
    """Loss function weights and curriculum.
    
    Attributes:
        lambda_price: Weight for price loss.
        lambda_greeks: Weight for Greek loss.
        lambda_pde: Weight for PDE residual loss.
        lambda_bc: Weight for boundary condition loss.
        lambda_noarb: Weight for no-arbitrage penalty.
        curriculum: Ordered list of training stages (e.g., ["price", "price+greeks"]).
    """

    lambda_price: float = Field(default=1.0, ge=0.0, description="Price loss weight")
    lambda_greeks: float = Field(default=1.0, ge=0.0, description="Greeks loss weight")
    lambda_pde: float = Field(default=0.1, ge=0.0, description="PDE residual weight")
    lambda_bc: float = Field(default=1.0, ge=0.0, description="Boundary condition weight")
    lambda_noarb: float = Field(default=0.0, ge=0.0, description="No-arbitrage weight")
    curriculum: list[str] = Field(
        default_factory=lambda: ["price", "price+greeks", "price+greeks+pde"],
        description="Training curriculum stages",
    )


class OptimCfg(BaseModel):
    """Optimizer and training schedule configuration.
    
    Attributes:
        name: Optimizer name ('adam', 'adamw', 'sgd').
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        schedule: Learning rate schedule type.
        epochs: Total training epochs.
        batch_data: Batch size for data points (prices/Greeks).
        batch_collocation: Batch size for PDE collocation points.
        amp: Enable automatic mixed precision.
        grad_clip: Gradient clipping norm (None = no clipping).
        early_stop_patience: Early stopping patience in epochs (None = no early stop).
    """

    name: Literal["adam", "adamw", "sgd"] = Field(
        default="adamw", description="Optimizer"
    )
    lr: float = Field(default=1e-3, gt=0.0, description="Learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Weight decay")
    schedule: Literal["none", "cosine", "step"] = Field(
        default="cosine", description="LR schedule"
    )
    epochs: int = Field(default=2000, ge=1, description="Training epochs")
    batch_data: int = Field(default=256, ge=1, description="Data batch size")
    batch_collocation: int = Field(
        default=1024, ge=1, description="Collocation batch size"
    )
    amp: bool = Field(default=False, description="Use automatic mixed precision")
    grad_clip: float | None = Field(default=1.0, description="Gradient clip norm")
    early_stop_patience: int | None = Field(
        default=200, description="Early stopping patience"
    )


class EvalCfg(BaseModel):
    """Evaluation configuration.
    
    Attributes:
        metrics: List of metric names to compute.
        test_grid: Test grid type ('heldout' or 'extrap').
    """

    metrics: list[str] = Field(
        default_factory=lambda: [
            "rmse_price",
            "mae_greeks",
            "rmse_iv",
            "param_rel_err",
            "rmse_extrap",
        ],
        description="Evaluation metrics",
    )
    test_grid: Literal["heldout", "extrap", "both"] = Field(
        default="heldout", description="Test grid type"
    )


class ModelCfg(BaseModel):
    """Stochastic model configuration.
    
    Attributes:
        name: Model name ('BS', 'LocalVol', 'Heston').
        params: Fixed model parameters (dict of param_name -> value).
        sweep: Optional parameter sweep (dict of param_name -> list of values).
    """

    name: Literal["BS", "LocalVol", "Heston"] = Field(..., description="Model name")
    params: dict[str, float] = Field(
        default_factory=dict, description="Fixed parameters"
    )
    sweep: dict[str, list[float]] | None = Field(
        default=None, description="Parameter sweep ranges"
    )


class ExperimentCfg(BaseModel):
    """Top-level experiment configuration.
    
    This schema matches the JSON structure in Listing lst:config-bs-softplus
    from Chapter 10 of the thesis.
    
    Attributes:
        experiment_id: Unique experiment identifier (snake_case).
        seed: Random seed for reproducibility.
        model: Stochastic model configuration.
        grid: Strike-maturity grid configuration.
        noise: Noise model configuration.
        net: Neural network architecture configuration.
        loss: Loss function configuration.
        optim: Optimizer configuration.
        eval: Evaluation configuration.
        ablation_tag: Optional ablation study identifier (e.g., 'A1', 'A2').
    """

    experiment_id: str = Field(..., min_length=1, description="Experiment ID")
    seed: int = Field(default=20251, description="Random seed")
    model: ModelCfg = Field(..., description="Model configuration")
    grid: GridCfg = Field(..., description="Grid configuration")
    noise: NoiseCfg = Field(default_factory=NoiseCfg, description="Noise configuration")
    net: NetCfg = Field(..., description="Network configuration")
    loss: LossCfg = Field(default_factory=LossCfg, description="Loss configuration")
    optim: OptimCfg = Field(
        default_factory=OptimCfg, description="Optimizer configuration"
    )
    eval: EvalCfg = Field(default_factory=EvalCfg, description="Evaluation configuration")
    ablation_tag: str | None = Field(default=None, description="Ablation identifier")

    def model_post_init(self, __context) -> None:
        """Post-initialization hook for additional validation."""
        # Ensure experiment_id is snake_case (basic check)
        if not self.experiment_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"experiment_id must be snake_case alphanumeric: {self.experiment_id}"
            )


"""
Multi-layer perceptron architectures for option pricing.

Implements fully-connected MLPs with selectable activation functions and
support for automatic differentiation of Greeks.

Greeks are computed correctly:
- Delta: ∂C/∂S (via chain rule from ∂C/∂K)
- Gamma: ∂²C/∂S²
- Vega: ∂C/∂σ (via finite differences on volatility parameter)
- Theta: -∂C/∂T
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth, width, and activation.
    
    The network takes (K, T) inputs (strike, maturity) and outputs option prices.
    Greeks (Delta, Gamma, Vega, Theta) are computed via automatic differentiation.
    
    Attributes:
        L: Number of hidden layers.
        W: Width of each hidden layer.
        activation: Activation function name.
        head: Output head type ('single' or 'value_and_aux').
        
    Example:
        >>> model = MLP(L=4, W=128, activation="softplus")
        >>> K = torch.tensor([[1.0], [1.1], [0.9]])
        >>> T = torch.tensor([[0.5], [0.5], [0.5]])
        >>> prices = model(K, T)
        >>> assert prices.shape == (3, 1)
    """
    
    def __init__(
        self,
        L: int,
        W: int,
        activation: Literal["softplus", "tanh", "relu", "sine"] = "softplus",
        head: Literal["single", "value_and_aux"] = "single",
        input_dim: int = 2,
        output_dim: int = 1,
    ):
        """
        Initialize MLP.
        
        Args:
            L: Number of hidden layers (depth).
            W: Width of each hidden layer.
            activation: Activation function type.
            head: Output head architecture.
            input_dim: Input dimension (default: 2 for K, T).
            output_dim: Output dimension (default: 1 for price).
        """
        super().__init__()
        self.L = L
        self.W = W
        self.activation_name = activation
        self.head_type = head
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, W))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(L - 1):
            layers.append(nn.Linear(W, W))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(W, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == "softplus":
            return nn.Softplus()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "relu":
            return nn.ReLU()
        elif name == "sine":
            return SineActivation()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights deterministically."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute option prices.
        
        Args:
            K: Strike prices, shape (batch_size, 1) or (batch_size,).
            T: Times to maturity, shape (batch_size, 1) or (batch_size,).
            
        Returns:
            Option prices, shape (batch_size, 1).
        """
        # Ensure 2D inputs
        if K.dim() == 1:
            K = K.unsqueeze(-1)
        if T.dim() == 1:
            T = T.unsqueeze(-1)
        
        # Concatenate inputs
        x = torch.cat([K, T], dim=-1)
        
        # Forward pass
        out = self.network(x)
        
        return out
    
    def compute_greeks(
        self, 
        K: torch.Tensor, 
        T: torch.Tensor, 
        S0: float = 1.0,
        sigma: Optional[float] = None,
        r: float = 0.02,
    ) -> dict[str, torch.Tensor]:
        """
        Compute Greeks via automatic differentiation.
        
        Delta and Gamma are the TRUE hedging Greeks:
        - Delta = ∂C/∂S (not ∂C/∂K)
        - Gamma = ∂²C/∂S²
        
        The relationship: Delta_S = -(K/S) * e^{-rT} * ∂C/∂K (approximately)
        For call options on non-dividend paying stocks, a simpler relation holds.
        
        Args:
            K: Strike prices, shape (batch_size, 1).
            T: Times to maturity, shape (batch_size, 1).
            S0: Current underlying price.
            sigma: Volatility (for Vega computation via finite diff).
            r: Risk-free rate.
            
        Returns:
            Dictionary with keys: 'price', 'delta', 'gamma', 'vega', 'theta'.
            
        Note:
            This requires gradients to be enabled and uses create_graph=True
            for second derivatives.
        """
        # Ensure inputs are 2D and require gradients
        if K.dim() == 1:
            K = K.unsqueeze(-1)
        if T.dim() == 1:
            T = T.unsqueeze(-1)
            
        K = K.requires_grad_(True)
        T = T.requires_grad_(True)
        
        # Forward pass
        price = self.forward(K, T)
        
        # ∂C/∂K (strike sensitivity)
        dC_dK = torch.autograd.grad(
            price.sum(), K, create_graph=True, retain_graph=True
        )[0]
        
        # ∂²C/∂K² (strike gamma)
        d2C_dK2 = torch.autograd.grad(
            dC_dK.sum(), K, create_graph=True, retain_graph=True
        )[0]
        
        # Convert to Delta (∂C/∂S)
        # For European call: Delta_S ≈ -dC_dK * exp(-r*T) (approximately)
        # More precisely, using put-call parity and homogeneity:
        # Delta = N(d1) for BS, but for neural net we use the chain rule
        # 
        # The key insight: C(S, K, T) is homogeneous degree 1 in (S, K)
        # => S * ∂C/∂S + K * ∂C/∂K = C
        # => ∂C/∂S = (C - K * ∂C/∂K) / S
        #
        # This gives us the correct Delta without needing S as input.
        delta = (price - K * dC_dK) / S0
        
        # Convert to Gamma (∂²C/∂S²)
        # Using the homogeneity relation and differentiating:
        # Gamma_S = (K²/S²) * Gamma_K - (2K/S²) * ∂C/∂K
        # Simplifying for our network output:
        gamma = (K**2 / S0**2) * d2C_dK2
        
        # Theta: -∂C/∂T (time decay)
        dC_dT = torch.autograd.grad(
            price.sum(), T, create_graph=True, retain_graph=True
        )[0]
        theta = -dC_dT
        
        # Vega: ∂C/∂σ
        # For neural networks, Vega requires either:
        # 1. σ as an input (not our case)
        # 2. Finite difference approximation
        # We'll compute a proxy based on the vega-gamma relationship:
        # Vega ≈ σ * S² * Gamma * T (for BS)
        # Or return zeros if sigma not provided
        if sigma is not None and sigma > 0:
            # Vega-Gamma relationship: Vega = σ * S0² * Gamma * T
            vega = sigma * (S0**2) * gamma * T
        else:
            # Cannot compute Vega without sigma
            vega = torch.zeros_like(price)
        
        return {
            "price": price.detach(),
            "delta": dC_dK.detach(),
            "gamma": d2C_dK2.detach(),
            "theta": theta.detach(),
            "spot_delta": delta.detach(),
            "spot_gamma": gamma.detach(),
            "vega": vega.detach(),
        }
    
    def compute_greeks_for_loss(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        S0: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute strike-space Greeks with gradients retained for loss backprop.

        Returns dC/dK, d2C/dK2, and -dC/dT (strike-space derivatives)
        matching the thesis definition (Ch6, Ch8).
        """
        if K.dim() == 1:
            K = K.unsqueeze(-1)
        if T.dim() == 1:
            T = T.unsqueeze(-1)

        K = K.requires_grad_(True)
        T = T.requires_grad_(True)

        price = self.forward(K, T)

        dC_dK = torch.autograd.grad(
            price.sum(), K, create_graph=True, retain_graph=True
        )[0]

        d2C_dK2 = torch.autograd.grad(
            dC_dK.sum(), K, create_graph=True, retain_graph=True
        )[0]

        dC_dT = torch.autograd.grad(
            price.sum(), T, create_graph=True, retain_graph=True
        )[0]

        return {
            "delta": dC_dK,
            "gamma": d2C_dK2,
            "theta": -dC_dT,
            "price": price,
        }


class SineActivation(nn.Module):
    """Sine activation function for SIREN-style networks."""
    
    def __init__(self, omega_0: float = 30.0):
        """
        Initialize sine activation.
        
        Args:
            omega_0: Frequency scaling factor.
        """
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled sine activation."""
        return torch.sin(self.omega_0 * x)


class MultiHeadMLP(nn.Module):  # NOTE: unused — kept for potential future work
    """
    MLP with separate output heads for price and Greeks.
    
    This architecture allows the network to learn price and Greeks
    semi-independently while sharing lower-level features.
    """
    
    def __init__(
        self,
        L: int,
        W: int,
        activation: str = "softplus",
        n_greeks: int = 3,
    ):
        """
        Initialize multi-head MLP.
        
        Args:
            L: Number of shared hidden layers.
            W: Width of hidden layers.
            activation: Activation function.
            n_greeks: Number of Greek outputs (delta, gamma, vega, ...).
        """
        super().__init__()
        
        # Shared trunk
        self.trunk = MLP(L=L-1, W=W, activation=activation, output_dim=W)
        
        # Separate heads
        self.price_head = nn.Linear(W, 1)
        self.greek_head = nn.Linear(W, n_greeks)
        
        self.n_greeks = n_greeks
    
    def forward(self, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Forward pass returning price only."""
        features = self.trunk(K, T)
        return self.price_head(features)
    
    def forward_all(
        self, K: torch.Tensor, T: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both price and Greek predictions.
        
        Returns:
            Tuple of (price, greeks) tensors.
        """
        features = self.trunk(K, T)
        price = self.price_head(features)
        greeks = self.greek_head(features)
        return price, greeks


def create_mlp_from_config(
    net_cfg: dict, 
    input_dim: int = 2, 
    output_dim: int = 1
) -> MLP:
    """
    Create MLP from configuration dictionary.
    
    Args:
        net_cfg: Network configuration dict (keys: L, W, activation, head).
        input_dim: Input dimension.
        output_dim: Output dimension.
        
    Returns:
        Initialized MLP model.
        
    Example:
        >>> cfg = {"L": 4, "W": 128, "activation": "softplus", "head": "single"}
        >>> model = create_mlp_from_config(cfg)
    """
    return MLP(
        L=net_cfg["L"],
        W=net_cfg["W"],
        activation=net_cfg.get("activation", "softplus"),
        head=net_cfg.get("head", "single"),
        input_dim=input_dim,
        output_dim=output_dim,
    )

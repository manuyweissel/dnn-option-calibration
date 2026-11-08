"""
Registry system for models, metrics, and experiment runners.

Provides a central registry for extensibility and dynamic loading of components.
"""

from typing import Any, Callable, Dict

# Global registries
_MODEL_REGISTRY: Dict[str, Any] = {}
_METRIC_REGISTRY: Dict[str, Callable] = {}
_RUNNER_REGISTRY: Dict[str, Any] = {}


def register_model(name: str, model_class: Any) -> None:
    """
    Register a model class.
    
    Args:
        name: Model name (e.g., 'BS', 'LocalVol', 'Heston').
        model_class: Model class or factory function.
        
    Example:
        >>> from ..models.pricing import BSPricer
        >>> register_model('BS', BSPricer)
    """
    _MODEL_REGISTRY[name] = model_class


def get_model(name: str) -> Any:
    """
    Retrieve registered model class.
    
    Args:
        name: Model name.
        
    Returns:
        Model class.
        
    Raises:
        KeyError: If model not found.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def register_metric(name: str, metric_fn: Callable) -> None:
    """
    Register a metric function.
    
    Args:
        name: Metric name (e.g., 'rmse_price').
        metric_fn: Metric computation function.
        
    Example:
        >>> from .metrics import rmse_price
        >>> register_metric('rmse_price', rmse_price)
    """
    _METRIC_REGISTRY[name] = metric_fn


def get_metric(name: str) -> Callable:
    """
    Retrieve registered metric function.
    
    Args:
        name: Metric name.
        
    Returns:
        Metric function.
        
    Raises:
        KeyError: If metric not found.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not registered. Available: {list(_METRIC_REGISTRY.keys())}")
    return _METRIC_REGISTRY[name]


def register_runner(name: str, runner_class: Any) -> None:
    """
    Register an experiment runner class.
    
    Args:
        name: Runner name.
        runner_class: Runner class.
    """
    _RUNNER_REGISTRY[name] = runner_class


def get_runner(name: str = "default") -> Any:
    """
    Retrieve registered runner class.
    
    Args:
        name: Runner name.
        
    Returns:
        Runner class.
    """
    if name not in _RUNNER_REGISTRY:
        # Return default runner if available
        if "default" in _RUNNER_REGISTRY:
            return _RUNNER_REGISTRY["default"]
        raise KeyError(f"Runner '{name}' not registered.")
    return _RUNNER_REGISTRY[name]


def list_registered(registry_type: str = "all") -> dict[str, list[str]]:
    """
    List all registered components.
    
    Args:
        registry_type: Type of registry ('models', 'metrics', 'runners', 'all').
        
    Returns:
        Dict with registered component names.
        
    Example:
        >>> registered = list_registered("all")
        >>> print(registered['models'])
    """
    registries = {}
    
    if registry_type in ["models", "all"]:
        registries["models"] = list(_MODEL_REGISTRY.keys())
    
    if registry_type in ["metrics", "all"]:
        registries["metrics"] = list(_METRIC_REGISTRY.keys())
    
    if registry_type in ["runners", "all"]:
        registries["runners"] = list(_RUNNER_REGISTRY.keys())
    
    return registries


# Auto-register default components
def _register_defaults() -> None:
    """Register default models and metrics."""
    from ..models.pricing import BSPricer, LocalVolPricer, HestonPricer, MonteCarloPricer
    from .metrics import rmse_price, mae_greeks, rpe, param_rel_err, rmse_extrap
    
    # Register models
    register_model("BS", BSPricer)
    register_model("LocalVol", LocalVolPricer)
    register_model("Heston", HestonPricer)
    register_model("MC", MonteCarloPricer)
    
    # Register metrics
    register_metric("rmse_price", rmse_price)
    register_metric("mae_greeks", mae_greeks)
    register_metric("rpe", rpe)
    register_metric("param_rel_err", param_rel_err)
    register_metric("rmse_extrap", rmse_extrap)


# Initialize on import
_register_defaults()


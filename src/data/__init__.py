"""Data generation and grid utilities."""

from .grids import create_grid, split_data, create_extrapolation_grid
from .synth import generate_synthetic_data
from .noise import (
    add_relative_noise,
    add_absolute_noise,
    add_heteroscedastic_noise,
    add_missing_data,
    apply_noise_config,
)

# Alias for backward compatibility
add_noise = add_relative_noise

__all__ = [
    "create_grid",
    "split_data",
    "create_extrapolation_grid",
    "generate_synthetic_data",
    "add_noise",
    "add_relative_noise",
    "add_absolute_noise",
    "add_heteroscedastic_noise",
    "add_missing_data",
    "apply_noise_config",
]

"""
Plotting utilities for option pricing experiments.

Provides functions to create consistent, publication-quality figures matching
the thesis style (Chapter 11, figures in assets/figures/experiments/).
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def plot_price_scatter(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    title: str = "Predicted vs True Prices",
) -> plt.Figure:
    """
    Create scatter plot of predicted vs true prices.
    
    Corresponds to fig:bs-scatter in the thesis.
    
    Args:
        df_pred: DataFrame with 'price' column.
        df_true: DataFrame with 'price' column.
        save_path: Path to save figure (None = don't save).
        title: Plot title.
        
    Returns:
        Matplotlib Figure object.
        
    Example:
        >>> df_pred = pd.DataFrame({'price': np.random.rand(100)})
        >>> df_true = pd.DataFrame({'price': np.random.rand(100)})
        >>> fig = plot_price_scatter(df_pred, df_true, "output/scatter.png")
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    pred = df_pred['price'].values
    true = df_true['price'].values
    
    # Scatter plot
    ax.scatter(true, pred, alpha=0.5, s=10, c='steelblue', edgecolors='none')
    
    # Diagonal line
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='Ideal')
    
    ax.set_xlabel('True Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_greek_mae(
    greek_dict: dict[str, float],
    save_path: Optional[str | Path] = None,
    title: str = "Greek MAE Comparison",
) -> plt.Figure:
    """
    Create bar chart of Greek MAE values.
    
    Corresponds to fig:bs-greek-mae in the thesis.
    
    Args:
        greek_dict: Dict with keys like 'mae_delta', 'mae_gamma', etc.
        save_path: Path to save figure.
        title: Plot title.
        
    Returns:
        Figure object.
        
    Example:
        >>> greeks = {'mae_delta': 0.02, 'mae_gamma': 0.01, 'mae_vega': 0.03}
        >>> fig = plot_greek_mae(greeks, "output/greek_mae.png")
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Extract Greek names and values
    greek_names = []
    values = []
    for key, val in greek_dict.items():
        if key.startswith('mae_') and key != 'mae_avg':
            greek_name = key.replace('mae_', '').capitalize()
            greek_names.append(greek_name)
            values.append(val)
    
    # Bar plot
    x_pos = np.arange(len(greek_names))
    ax.bar(x_pos, values, color='steelblue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(greek_names)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_heatmap(
    df: pd.DataFrame,
    error_column: str = 'error',
    K_column: str = 'K',
    T_column: str = 'T',
    save_path: Optional[str | Path] = None,
    title: str = "Error Heatmap",
) -> plt.Figure:
    """
    Create heatmap of errors over (K, T) grid.
    
    Corresponds to fig:lv-error-heatmap in the thesis.
    
    Args:
        df: DataFrame with columns [K, T, error].
        error_column: Name of error column.
        K_column: Name of strike column.
        T_column: Name of maturity column.
        save_path: Path to save figure.
        title: Plot title.
        
    Returns:
        Figure object.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'K': np.tile(np.linspace(0.8, 1.2, 10), 10),
        ...     'T': np.repeat(np.linspace(0.1, 1.0, 10), 10),
        ...     'error': np.random.rand(100) * 0.05,
        ... })
        >>> fig = plot_error_heatmap(df, save_path="output/heatmap.png")
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Pivot for heatmap
    pivot = df.pivot_table(
        values=error_column, index=T_column, columns=K_column, aggfunc='mean'
    )
    
    # Heatmap
    sns.heatmap(
        pivot, ax=ax, cmap='RdYlBu_r', cbar_kws={'label': 'Error'},
        linewidths=0, square=False
    )
    
    ax.set_xlabel('Strike (K)')
    ax.set_ylabel('Maturity (T)')
    ax.set_title(title)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_curve(
    history: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    title: str = "Training Loss Curve",
) -> plt.Figure:
    """
    Plot training loss curves over epochs.
    
    Corresponds to fig:bs-loss-curve in the thesis.
    
    Args:
        history: DataFrame with columns ['epoch', 'total', 'price', 'greeks', ...].
        save_path: Path to save figure.
        title: Plot title.
        
    Returns:
        Figure object.
        
    Example:
        >>> history = pd.DataFrame({
        ...     'epoch': range(100),
        ...     'total': np.exp(-np.linspace(0, 5, 100)),
        ...     'price': np.exp(-np.linspace(0, 4, 100)),
        ... })
        >>> fig = plot_loss_curve(history, "output/loss_curve.png")
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot loss components
    val_keys = {'val_loss', 'val_price_loss', 'val_greeks_loss'}
    loss_cols = [c for c in history.columns if c not in (['epoch'] + list(val_keys))]
    
    for col in loss_cols:
        if col in history.columns:
            ax.plot(history['epoch'], history[col], label=col, alpha=0.8)
    
    for vc in ('val_loss', 'val_price_loss'):
        if vc in history.columns:
            ax.plot(history['epoch'], history[vc],
                    label='val_loss', linestyle='--', linewidth=2)
            break
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_activation_ablation(
    results: dict[str, float],
    save_path: Optional[str | Path] = None,
    title: str = "Activation Function Comparison",
    metric_name: str = "Error",
) -> plt.Figure:
    """
    Plot ablation study for activation functions.
    
    Corresponds to fig:bs-activation / fig:ablationA2 in the thesis.
    
    Args:
        results: Dict with {activation_name: error_value}.
        save_path: Path to save figure.
        title: Plot title.
        metric_name: Y-axis label.
        
    Returns:
        Figure object.
        
    Example:
        >>> results = {'softplus': 0.015, 'tanh': 0.018, 'relu': 0.035}
        >>> fig = plot_activation_ablation(results, "output/activation_bar.png")
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    activations = list(results.keys())
    values = list(results.values())
    
    x_pos = np.arange(len(activations))
    colors = ['steelblue' if act in ['softplus', 'tanh'] else 'coral' 
              for act in activations]
    
    ax.bar(x_pos, values, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activations)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_scatter(
    params_pred: dict[str, np.ndarray],
    params_true: dict[str, np.ndarray],
    save_path: Optional[str | Path] = None,
    title: str = "Parameter Recovery",
) -> plt.Figure:
    """
    Plot scatter of predicted vs true parameters (Heston model).
    
    Corresponds to fig:heston-scatter in the thesis.
    
    Args:
        params_pred: Dict with {param_name: array_of_predictions}.
        params_true: Dict with {param_name: array_of_true_values}.
        save_path: Path to save figure.
        title: Plot title.
        
    Returns:
        Figure object.
        
    Example:
        >>> pred = {'rho': np.array([- 0.48, -0.52]), 'v0': np.array([0.039, 0.041])}
        >>> true = {'rho': np.array([-0.5, -0.5]), 'v0': np.array([0.04, 0.04])}
        >>> fig = plot_parameter_scatter(pred, true, "output/param_scatter.png")
    """
    n_params = len(params_pred)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, param_name in zip(axes, params_pred.keys()):
        pred = params_pred[param_name]
        true = params_true[param_name]
        
        ax.scatter(true, pred, alpha=0.6, s=30, c='steelblue')
        
        # Diagonal
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        
        ax.set_xlabel(f'True {param_name}')
        ax.set_ylabel(f'Predicted {param_name}')
        ax.set_title(param_name)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_all_figures(
    output_dir: str | Path,
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    greek_dict: dict,
    history: pd.DataFrame,
) -> None:
    """
    Save all standard figures for an experiment.
    
    Args:
        output_dir: Directory to save figures.
        df_pred: Predictions DataFrame.
        df_true: True values DataFrame.
        greek_dict: Greek MAE dict.
        history: Training history.
        
    Example:
        >>> save_all_figures("runs/exp001/figures", df_pred, df_true, greeks, history)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Price scatter
    plot_price_scatter(
        df_pred, df_true, output_dir / "price_scatter.png"
    )
    plt.close()
    
    # Greek MAE
    if greek_dict:
        plot_greek_mae(
            greek_dict, output_dir / "greek_mae.png"
        )
        plt.close()
    
    # Loss curve
    if not history.empty:
        plot_loss_curve(
            history, output_dir / "loss_curve.png"
        )
        plt.close()


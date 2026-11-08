#!/usr/bin/env python3
"""
Generate all thesis figures from experimental JSON results and run directories.

Usage:
    python scripts/gen_all_figures.py [--results-file PATH]

Output:
    All figures saved to: ../thesis_latex_framework/thesis/assets/figures/

Figures generated:
    Chapter 3 - Differential Neural Networks:
    - learning_rate_schedule.png (fig:lr_schedule)

    Chapter 7 - Methodology:
    - method_comparison.png     (fig:method-pipeline)

    Chapter 10 - Experimental Results:
    - bs_price_scatter.png      (fig:bs-scatter)
    - bs_greek_mae.png          (fig:bs-greek-mae)
    - bs_loss_curve.png         (fig:bs-loss-curve)
    - activation_bar.png        (fig:bs-activation)
    - lv_price_true.png         (fig:lv-surfaces)
    - lv_price_pinn.png         (fig:lv-surfaces)
    - lv_price_error.png        (fig:lv-surfaces)
    - lv_dupire_true.png        (fig:lv-dupire-surfaces)
    - lv_dupire_extracted.png   (fig:lv-dupire-surfaces)
    - lv_dupire_error.png       (fig:lv-dupire-surfaces)
    - lv_error_heatmap.png      (fig:lv-error-heatmap)
    - lv_smoothness_plot.png    (fig:lv-smoothness)
    - lv_extrap_dense.png       (fig:lv-extrap)
    - lv_extrap_sparse.png      (fig:lv-extrap)
    - heston_param_scatter.png  (fig:heston-scatter)
    - correlation_error_plot.png (fig:corr-error)
    - grid_density_error.png    (fig:grid-density)
    - ablation_A1.png           (fig:ablationA1)
    - ablation_A2.png           (fig:ablationA2)
    - ablation_A3.png           (fig:ablationA3)
    - ablation_A4.png           (fig:ablationA4)
    - failure_map_ablation.png  (fig:failure-map-ablation)

Note: 6 figures are TikZ diagrams (inline LaTeX) and cannot be auto-generated:
    - fig:concept-map (Ch1), fig:ad-graph (Ch3), fig:obs-operator-map (Ch4)
    - fig:composite-loss-structure (Ch6), fig:stability-plot (Ch8), fig:failure-map-theory (Ch8)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"
# Inside the thesis repository the figures go straight into the LaTeX assets;
# in the stand-alone code repository they are written to outputs/figures.
_THESIS_ASSETS = PROJECT_ROOT.parent / "thesis_latex_framework" / "thesis" / "assets"
OUTPUT_DIR = (_THESIS_ASSETS / "figures") if _THESIS_ASSETS.is_dir() else (PROJECT_ROOT / "outputs" / "figures")

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

NO_DATA_MSG = "No experiment data available"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_results() -> Path:
    # Timestamped files only: all_results_rolling.json sorts after any
    # all_results_<YYYYMMDD_HHMMSS>.json name and is a mid-run snapshot.
    results_files = sorted(RESULTS_DIR.glob("all_results_[0-9]*.json"), reverse=True)
    if not results_files:
        raise FileNotFoundError(f"No results files found in {RESULTS_DIR}")
    return results_files[0]


def _find_latest_run(experiment_prefix: str) -> Optional[Path]:
    """Find the latest multiseed run directory for a given experiment prefix."""
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(
        RUNS_DIR.glob(f"*__{experiment_prefix}__multiseed"), reverse=True
    )
    if candidates:
        seed_dirs = sorted(candidates[0].glob("seed_*"))
        return seed_dirs[0] if seed_dirs else None
    candidates = sorted(RUNS_DIR.glob(f"*__{experiment_prefix}"), reverse=True)
    if candidates:
        seed_dirs = sorted(candidates[0].glob("seed_*"))
        return seed_dirs[0] if seed_dirs else candidates[0]
    return None


def _load_model_and_data(experiment_prefix: str, split: Optional[str] = None):
    """Load a trained model and generate predictions on the training grid.

    For BS models, also generates analytic true prices for comparison.
    For LocalVol/Heston, only neural-net predictions are returned (true
    prices require expensive PDE/MC solves that are too slow for figure
    generation).
    """
    import torch
    run_dir = _find_latest_run(experiment_prefix)
    if run_dir is None:
        return None, None, None

    from src.exp.io_utils import load_config
    from src.models.nets import MLP
    from src.data.grids import create_grid, split_data

    cfg_path = run_dir / "cfg.json"
    ckpt_path = run_dir / "model.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        return None, None, None

    cfg = load_config(cfg_path)
    model = MLP(L=cfg.net.L, W=cfg.net.W, activation=cfg.net.activation, head=cfg.net.head)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    grid = create_grid(
        K=cfg.grid.K, T=cfg.grid.T, spacing=cfg.grid.spacing,
        K_min=cfg.grid.K_min, K_max=cfg.grid.K_max,
        T_min=cfg.grid.T_min, T_max=cfg.grid.T_max,
        S0=cfg.model.params.get("S0", 1.0),
    )

    if split is not None:
        from src.data.synth import generate_synthetic_data

        data = generate_synthetic_data(grid, cfg.model.name, cfg.model.params)
        train_df, val_df, test_df, extrap_df = split_data(
            data, seed=cfg.seed, S0=cfg.model.params.get("S0", 1.0)
        )
        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "extrap": extrap_df,
        }
        if split not in splits:
            raise ValueError(f"Unknown split: {split}")
        grid = splits[split].reset_index(drop=True)

    K_t = torch.tensor(grid["K"].values, dtype=torch.float32).unsqueeze(1)
    T_t = torch.tensor(grid["T"].values, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred_prices = model(K_t, T_t).numpy().flatten()

    df = grid.copy()
    df["pred_price"] = pred_prices

    if cfg.model.name == "BS":
        from src.models.pricing import BSPricer
        p = BSPricer(**{k: v for k, v in cfg.model.params.items()
                        if k in ("S0", "r", "sigma")})
        df["price"] = p.price(grid["K"].values, grid["T"].values)
    else:
        df["price"] = np.nan

    return model, cfg, df


def _load_metrics_csv(experiment_prefix: str) -> Optional[pd.DataFrame]:
    run_dir = _find_latest_run(experiment_prefix)
    if run_dir is None:
        return None
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def _get(data: dict, key: str, metric: str, kind: str = "mean") -> Optional[float]:
    r = data.get(key, {})
    return r.get(kind, {}).get(metric)


def _nodata(ax: plt.Axes, msg: str = NO_DATA_MSG):
    ax.text(
        0.5, 0.5, msg, transform=ax.transAxes,
        ha="center", va="center", fontsize=12, color="gray", style="italic",
    )


# ===========================================================================
# Chapter 3 - Differential Neural Networks
# ===========================================================================

def gen_learning_rate_schedule(data: dict, output_dir: Path) -> None:
    print("  [learning_rate_schedule.png]")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    epochs = np.arange(0, 2001)
    initial_lr = 1e-3
    step_lr = initial_lr * (0.5 ** (epochs // 500))
    cosine_lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epochs / 2000))

    ax1.plot(epochs, step_lr, color="steelblue", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning Rate")
    ax1.set_title("Step Decay Schedule")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-5, 2e-3)

    ax2.plot(epochs, cosine_lr, color="coral", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Cosine Annealing Schedule")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-5, 2e-3)

    plt.tight_layout()
    fig.savefig(output_dir / "learning_rate_schedule.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Chapter 7 - Methodology
# ===========================================================================

def gen_method_comparison(data: dict, output_dir: Path) -> None:
    print("  [method_comparison.png]")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(1.5, 5, "Synthetic Data\nGeneration", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4e8", edgecolor="green", linewidth=2))
    ax.text(1.5, 3, "True Model\n(BS/LV/Heston)", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor="orange", linewidth=1.5))
    ax.annotate("", xy=(1.5, 4.3), xytext=(1.5, 3.7),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.text(5, 5, "PINN\n(Prices + Greeks + PDE)", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="blue", linewidth=2))
    ax.text(5, 3.5, "NN\n(Prices only)", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fce4ec", edgecolor="red", linewidth=2))
    ax.text(5, 2, "Classical (MC)", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3e5f5", edgecolor="purple", linewidth=2))

    for tgt_y, src_y in [(5, 5), (3.5, 4.5), (2, 4)]:
        ax.annotate("", xy=(3.5, tgt_y), xytext=(2.5, src_y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.text(8.5, 3.5, "Evaluation\n(Test Grid)", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="goldenrod", linewidth=2))
    for src_y, tgt_y in [(5, 4), (3.5, 3.5), (2, 3)]:
        ax.annotate("", xy=(7, tgt_y), xytext=(6.5, src_y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    for label, y in [("Price RMSE", 4.5), ("Greek MAE", 3.5), ("Param. Error", 2.5)]:
        ax.text(11, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", linewidth=1))
    for tgt_y, src_y in [(4.5, 3.8), (3.5, 3.5), (2.5, 3.2)]:
        ax.annotate("", xy=(10, tgt_y), xytext=(9.5, src_y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.set_title("Comparative Calibration Pipeline", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Black-Scholes Figures
# ===========================================================================

def gen_bs_price_scatter(data: dict, output_dir: Path) -> None:
    """Generate fig:bs-scatter from actual trained model predictions."""
    print("  [bs_price_scatter.png]")

    model, cfg, df = _load_model_and_data("bs_softplus_full", split="test")
    if df is None:
        model, cfg, df = _load_model_and_data("bs_nn_baseline", split="test")

    fig, ax = plt.subplots(figsize=(6, 5))
    if df is not None:
        true_prices = df["price"].values
        pred_prices = df["pred_price"].values
        ax.scatter(true_prices, pred_prices, alpha=0.5, s=10, c="steelblue", edgecolors="none")
        lo = min(true_prices.min(), pred_prices.min())
        hi = max(true_prices.max(), pred_prices.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Ideal")
        rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
        ax.text(0.05, 0.92, f"Test RMSE = {rmse:.4f}", transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8))
    else:
        _nodata(ax)

    ax.set_xlabel("True Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Black-Scholes: Predicted vs True Prices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "bs_price_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_bs_greek_mae(data: dict, output_dir: Path) -> None:
    """Generate fig:bs-greek-mae from aggregated results."""
    print("  [bs_greek_mae.png]")

    fig, ax = plt.subplots(figsize=(9, 5))

    method_keys = [
        ("bs_nn_baseline.json", "NN"),
        ("bs_nnpde.json", "NN-PDE"),
        ("bs_nnm2.json", "NN-M2"),
        ("bs_softplus_full.json", "PINN"),
    ]

    present = [(k, l) for k, l in method_keys if k in data]
    if not present:
        _nodata(ax)
    else:
        methods = [l for _, l in present]
        delta_mae = [_get(data, k, "mae_delta") or 0 for k, _ in present]
        gamma_mae = [_get(data, k, "mae_gamma") or 0 for k, _ in present]
        theta_mae = [_get(data, k, "mae_theta") or 0 for k, _ in present]
        x = np.arange(len(methods))
        width = 0.25
        ax.bar(x - width, delta_mae, width, label=r"$\mathrm{MAE}_\Delta$", color="steelblue", alpha=0.8)
        ax.bar(x, gamma_mae, width, label=r"$\mathrm{MAE}_\Gamma$", color="coral", alpha=0.8)
        ax.bar(x + width, theta_mae, width, label=r"$\mathrm{MAE}_\Theta$", color="seagreen", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)

    ax.set_xlabel("Method")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Greek MAE Comparison (Black-Scholes)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "bs_greek_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_bs_loss_curve(data: dict, output_dir: Path) -> None:
    """Generate fig:bs-loss-curve from the metrics.csv of one BS PINN run.

    The loss curriculum of the PINN switches on the Greek term at epoch
    E/3 and the PDE term at 2E/3 (E = 2000); both switches are marked, since
    the training and validation losses jump there.
    """
    print("  [bs_loss_curve.png]")

    fig, ax = plt.subplots(figsize=(7, 4))
    hist = _load_metrics_csv("bs_softplus_full")

    if hist is not None and "epoch" in hist.columns:
        if "total" in hist.columns:
            ax.plot(hist["epoch"], hist["total"], label="Training loss (total)", linewidth=1.8)
        if "price" in hist.columns:
            ax.plot(hist["epoch"], hist["price"], label="Training price loss", linewidth=1.2)
        if "greeks" in hist.columns:
            valid = hist["greeks"].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                ax.plot(hist.loc[valid.index, "epoch"], valid, label="Training Greek loss",
                        linewidth=1.2, linestyle=":")
        if "val_loss" in hist.columns:
            ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss", linewidth=1.2, linestyle="--")
        n_total = 2000
        stage = n_total // 3
        for x, lab in ((stage, "Greeks on"), (2 * stage, "PDE on")):
            ax.axvline(x, color="grey", linewidth=0.8, linestyle="-.")
            ax.text(x, 1.0, " " + lab, transform=ax.get_xaxis_transform(), va="top", ha="left",
                    fontsize=8, color="grey")
    else:
        _nodata(ax)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "bs_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fmt_pct(v: float) -> str:
    """Readable percentage label for values spanning many decades."""
    if v >= 10:
        return f"{v:.1f}%"
    if v >= 0.01:
        return f"{v:.2f}%"
    exp = int(np.floor(np.log10(v)))
    return rf"${v / 10**exp:.1f}\times10^{{{exp}}}$%"


def gen_bs_sigma_recovery(data: dict, output_dir: Path) -> None:
    """Generate fig:bs-recovery: relative sigma-recovery error by variant and under noise.

    Left: post-hoc volatility recovery from the learned BS surfaces (NN, NN-M2,
    NN-PDE, PINN) and the two classical calibrations, mean +/- SD over seeds
    (log scale). Right: the PINN under relative price noise of 0, 0.5 and 1 %
    (the Greek noise is half the price noise).
    """
    print("  [bs_sigma_recovery.png]")
    variants = [
        ("NN", "bs_nn_baseline.json"),
        ("NN-M2", "bs_nnm2.json"),
        ("NN-PDE", "bs_nnpde.json"),
        ("PINN", "bs_softplus_full.json"),
        ("Classical\n(det.)", "classical_bs"),
        ("Classical\n(MC)", "classical_mc_bs"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})

    names, means, stds = [], [], []
    for name, key in variants:
        m = _get(data, key, "param_rel_err")
        if m is None:
            continue
        s = _get(data, key, "param_rel_err", kind="std") or 0.0
        names.append(name)
        means.append(100.0 * m)
        stds.append(100.0 * s)
    if names:
        x = np.arange(len(names))
        colors = ["#c0392b", "#27ae60", "#e67e22", "#2471a3", "#7f8c8d", "#95a5a6"][: len(names)]
        lower = [min(s, 0.95 * m) for m, s in zip(means, stds)]  # keep log-scale bars positive
        ax1.bar(x, means, color=colors, edgecolor="black", linewidth=0.5,
                yerr=[lower, stds], capsize=3)
        ax1.set_yscale("log")
        for xi, m, s in zip(x, means, stds):
            ax1.text(xi, (m + s) * 1.25, _fmt_pct(m), ha="center", va="bottom", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=8)
        ax1.set_ylim(min(means) / 5, max(m + s for m, s in zip(means, stds)) * 12)
    else:
        _nodata(ax1)
    ax1.set_ylabel(r"Relative error in $\hat\sigma$ (%)")
    ax1.set_title("(a) Recovery error by method", fontsize=10)
    ax1.grid(True, axis="y", which="both", alpha=0.25)

    levels = [("0", "noise_0.0"), ("0.5", "noise_0.005"), ("1", "noise_0.01")]
    xs, ms, ss = [], [], []
    for lab, key in levels:
        m = _get(data, key, "param_rel_err")
        if m is None:
            continue
        xs.append(float(lab))
        ms.append(100.0 * m)
        ss.append(100.0 * (_get(data, key, "param_rel_err", kind="std") or 0.0))
    if xs:
        ax2.errorbar(xs, ms, yerr=ss, fmt="o-", capsize=4, color="#2471a3", linewidth=1.5)
        for x_, m in zip(xs, ms):
            ax2.annotate(_fmt_pct(m), (x_, m), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([f"{x_:g}%" for x_ in xs])
        ax2.set_ylim(0, max(m + s for m, s in zip(ms, ss)) * 1.2)
    else:
        _nodata(ax2)
    ax2.set_xlabel("Relative price noise (Greek noise: half)")
    ax2.set_ylabel(r"Relative error in $\hat\sigma$ (%)")
    ax2.set_title("(b) PINN under observation noise", fontsize=10)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_dir / "bs_sigma_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_activation_bar(data: dict, output_dir: Path) -> None:
    """Generate fig:bs-activation from ablation results."""
    print("  [activation_bar.png]")

    fig, ax = plt.subplots(figsize=(6, 4))

    act_keys = [
        ("act_ablation_bs.json", "Softplus"),
        ("act_ablation_tanh.json", "Tanh"),
        ("act_ablation_relu.json", "ReLU"),
    ]
    present = [(k, l) for k, l in act_keys if k in data]

    if not present:
        _nodata(ax)
    else:
        labels = [l for _, l in present]
        rmse = [(_get(data, k, "rmse_price") or 0) * 1000 for k, _ in present]
        stds = [(_get(data, k, "rmse_price", "std") or 0) * 1000 for k, _ in present]
        colors = {"Softplus": "#2ecc71", "Tanh": "#3498db", "ReLU": "#e74c3c"}
        bar_colors = [colors.get(l, "gray") for l in labels]
        x = np.arange(len(labels))
        ax.bar(x, rmse, yerr=stds, capsize=5, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    ax.set_ylabel(r"Price RMSE ($\times 10^{-3}$)")
    ax.set_title("Activation Function Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "activation_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Local Volatility Figures
# ===========================================================================

def _save_lv_price_panel(output_path, Kg, Tg, Z, vmin, vmax, cbar_label,
                         cmap="viridis"):
    """Filled-contour price-surface panel with a fixed colour range.

    A fixed ``vmin``/``vmax`` is used so that the reference and predicted price
    panels share one colour scale and can be compared pointwise.
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    levels = np.linspace(vmin, vmax, 21)
    cs = ax.contourf(Kg, Tg, Z, levels=levels, cmap=cmap, extend="both")
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(cbar_label)
    ax.set_xlabel(r"Strike $K$")
    ax.set_ylabel(r"Maturity $T$")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_lv_vol_panel(output_path, Kg, Tg, Z, vmin, vmax, cbar_label,
                       cmap="viridis"):
    """pcolormesh local-volatility panel.

    NaN cells (where the Dupire denominator is non-positive, so no real
    volatility can be extracted) are masked and shown in grey rather than
    silently interpolated away.
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.set_facecolor("0.85")
    cs = ax.pcolormesh(
        Kg, Tg, np.ma.masked_invalid(Z),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
    )
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(cbar_label)
    ax.set_xlabel(r"Strike $K$")
    ax.set_ylabel(r"Maturity $T$")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_lv_surfaces(experiment_prefix="lv_dnn", fallback="lv_nn_baseline"):
    """Load the trained local-volatility PINN and reconstruct the reference and
    predicted PRICE surfaces on the experiment's own (K, T) grid.

    The reference surface ``u*(K,T)`` is regenerated with the SAME synthetic
    data generator, grid, parameters and Dupire pricer that produced the
    training labels and the metrics in ``tab:lv-metrics`` (namely
    ``src/data/synth.generate_synthetic_data`` with the LocalVol pricer).  It
    is therefore the experiment's own ground truth, not fabricated data.

    Returns ``(model, cfg, Kg, Tg, U_true, U_pinn)`` or all-``None`` if no run
    directory is found.
    """
    model, cfg, df = _load_model_and_data(experiment_prefix)
    if df is None:
        model, cfg, df = _load_model_and_data(fallback)
    if df is None:
        return None, None, None, None, None, None

    from src.data.synth import generate_synthetic_data
    ref = generate_synthetic_data(
        df[["K", "T"]].copy(), cfg.model.name, cfg.model.params,
        compute_greeks=False,
    )
    df = df.copy()
    df["true_price"] = ref["price"].values

    K_u = np.sort(df["K"].unique())
    T_u = np.sort(df["T"].unique())
    Kg, Tg = np.meshgrid(K_u, T_u)
    U_true = (
        df.pivot_table(values="true_price", index="T", columns="K", aggfunc="first")
        .reindex(index=T_u, columns=K_u).values
    )
    U_pinn = (
        df.pivot_table(values="pred_price", index="T", columns="K", aggfunc="first")
        .reindex(index=T_u, columns=K_u).values
    )
    return model, cfg, Kg, Tg, U_true, U_pinn


def _lv_nodata_panels(output_dir, panels):
    """Write 'no data' placeholder panels when no LV run is available."""
    for fname in panels:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        _nodata(ax, "No LV run data found")
        ax.set_xlabel(r"Strike $K$")
        ax.set_ylabel(r"Maturity $T$")
        plt.tight_layout()
        fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def gen_lv_surfaces(data: dict, output_dir: Path) -> None:
    """Generate fig:lv-surfaces -- local-volatility PRICE-surface recovery.

    Three directly comparable panels on the same (K, T) grid:
      (a) reference price surface  u*(K,T)       -> lv_price_true.png
      (b) PINN-predicted surface   u_theta(K,T)  -> lv_price_pinn.png
      (c) absolute error           |u_theta-u*|  -> lv_price_error.png

    Panels (a) and (b) share a single colour scale, so they show the SAME
    quantity in the same units and can be compared pointwise.  This replaces
    the earlier figure that juxtaposed the local-volatility surface (a) with
    the price surface (b), which were on different scales and not comparable.
    The local-volatility surface itself is not learned by the network; it is
    recovered only post hoc (see ``gen_lv_dupire_surfaces``).
    """
    print("  [lv_price_true.png, lv_price_pinn.png, lv_price_error.png]")

    model, cfg, Kg, Tg, U_true, U_pinn = _load_lv_surfaces()
    if U_true is None:
        _lv_nodata_panels(
            output_dir,
            ["lv_price_true.png", "lv_price_pinn.png", "lv_price_error.png"],
        )
        return

    U_err = np.abs(U_pinn - U_true)
    vmin = float(min(np.nanmin(U_true), np.nanmin(U_pinn)))
    vmax = float(max(np.nanmax(U_true), np.nanmax(U_pinn)))

    _save_lv_price_panel(output_dir / "lv_price_true.png", Kg, Tg, U_true,
                         vmin, vmax, r"Call price $u^*(K,T)$")
    _save_lv_price_panel(output_dir / "lv_price_pinn.png", Kg, Tg, U_pinn,
                         vmin, vmax, r"Call price $u_\theta(K,T)$")
    _save_lv_price_panel(output_dir / "lv_price_error.png", Kg, Tg, U_err,
                         0.0, float(np.nanmax(U_err)),
                         r"$|u_\theta-u^*|$", cmap="magma")


def gen_lv_dupire_surfaces(data: dict, output_dir: Path) -> None:
    """Generate fig:lv-dupire-surfaces -- post-hoc Dupire LV-surface recovery.

    Three panels on the same (K, T) grid:
      (a) true local-volatility surface  sigma_loc(K,T)       -> lv_dupire_true.png
      (b) Dupire-extracted surface        hat_sigma_loc(K,T)  -> lv_dupire_extracted.png
      (c) absolute error  |hat_sigma_loc - sigma_loc|         -> lv_dupire_error.png

    The extracted surface is computed directly from the trained PINN via
    ``src/models/dupire.dupire_extract_localvol`` (the same Dupire operator that
    feeds the parameter-error column of ``tab:lv-metrics``), so it is
    reproducible from the stored network output rather than only the fitted
    parametric error.  Cells where the Dupire denominator (1/2) K^2 d_KK C is
    non-positive yield no valid volatility and are shown in grey, which is part
    of the diagnostic: it is exactly this ill-conditioning that inflates the
    post-hoc parameter error.
    """
    print("  [lv_dupire_true.png, lv_dupire_extracted.png, lv_dupire_error.png]")

    model, cfg, Kg, Tg, U_true, U_pinn = _load_lv_surfaces()
    if model is None:
        _lv_nodata_panels(
            output_dir,
            ["lv_dupire_true.png", "lv_dupire_extracted.png", "lv_dupire_error.png"],
        )
        return

    from src.models.dupire import dupire_extract_localvol

    params = cfg.model.params
    S0 = params.get("S0", 1.0)
    r = params.get("r", 0.02)
    sigma0 = params.get("sigma0", 0.2)
    alpha = params.get("alpha", 0.5)
    beta = params.get("beta", 0.0)

    K_flat = Kg.flatten().astype(float)
    T_flat = Tg.flatten().astype(float)
    sig_true = sigma0 * (K_flat / S0) ** alpha * (1.0 + beta * T_flat)
    sig_hat = dupire_extract_localvol(model, K_flat, T_flat, r=r)

    S_true = sig_true.reshape(Kg.shape)
    S_hat = sig_hat.reshape(Kg.shape)
    S_err = np.abs(S_hat - S_true)

    finite_hat = np.isfinite(S_hat)
    lo_hat = float(np.nanpercentile(S_hat, 2)) if finite_hat.any() else float(np.nanmin(S_true))
    hi_hat = float(np.nanpercentile(S_hat, 98)) if finite_hat.any() else float(np.nanmax(S_true))
    vmin = float(min(np.nanmin(S_true), lo_hat))
    vmax = float(max(np.nanmax(S_true), hi_hat))

    _save_lv_vol_panel(output_dir / "lv_dupire_true.png", Kg, Tg, S_true,
                       vmin, vmax, r"$\sigma_{\mathrm{loc}}(K,T)$")
    _save_lv_vol_panel(output_dir / "lv_dupire_extracted.png", Kg, Tg, S_hat,
                       vmin, vmax, r"$\hat\sigma_{\mathrm{loc}}(K,T)$")
    err_top = float(np.nanpercentile(S_err, 98)) if np.isfinite(S_err).any() else 1.0
    if not err_top > 0:
        err_top = 1.0
    _save_lv_vol_panel(output_dir / "lv_dupire_error.png", Kg, Tg, S_err,
                       0.0, err_top,
                       r"$|\hat\sigma_{\mathrm{loc}}-\sigma_{\mathrm{loc}}|$",
                       cmap="magma")


def gen_lv_error_heatmap(data: dict, output_dir: Path) -> None:
    """Generate fig:lv-error-heatmap.

    Since true LV prices require an expensive PDE solve, this figure is
    generated only when actual experiment run data is available (with true
    prices stored alongside predictions).  Otherwise shows aggregate RMSE
    from the results JSON.
    """
    print("  [lv_error_heatmap.png]")

    fig, ax = plt.subplots(figsize=(8, 5))

    lv_rmse = _get(data, "lv_dnn.json", "rmse_price")
    if lv_rmse is None:
        lv_rmse = _get(data, "lv_nn_baseline.json", "rmse_price")

    if lv_rmse is not None:
        ax.text(0.5, 0.5, f"LV Price RMSE = {lv_rmse:.4f}\n(heatmap requires full run data)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
    else:
        _nodata(ax)

    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (T)")
    ax.set_title("Local Volatility Error Heatmap")
    plt.tight_layout()
    fig.savefig(output_dir / "lv_error_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_lv_smoothness_plot(data: dict, output_dir: Path) -> None:
    """Generate fig:lv-smoothness from lambda-sweep run data."""
    print("  [lv_smoothness_plot.png]")

    fig, ax = plt.subplots(figsize=(7, 5))

    lambdas = []
    errors = []
    for run in sorted(RUNS_DIR.glob("*__lv_lambda_sweep_*")) if RUNS_DIR.exists() else []:
        tag = run.name.split("lv_lambda_sweep_")[-1]
        try:
            lam = float(tag)
        except ValueError:
            continue
        stats = run / "stats.json"
        if not stats.exists():
            seed_dirs = sorted(run.glob("seed_*"))
            if seed_dirs:
                s = seed_dirs[0] / "summary.json"
                if s.exists():
                    m = json.load(open(s))
                    lambdas.append(lam)
                    errors.append(m.get("metrics", {}).get("rmse_extrap", m.get("metrics", {}).get("rmse_price", 0)))
            continue
        s = json.load(open(stats))
        lambdas.append(lam)
        errors.append(s.get("mean", {}).get("rmse_extrap", s.get("mean", {}).get("rmse_price", 0)))

    if lambdas:
        order = np.argsort(lambdas)
        lambdas = np.array(lambdas)[order]
        errors = np.array(errors)[order]
        ax.plot(lambdas, errors, "o-", color="steelblue", linewidth=2, markersize=8)
        best_idx = np.argmin(errors)
        ax.axvline(x=lambdas[best_idx], color="red", linestyle="--", alpha=0.5,
                    label=rf"Optimal $\lambda={lambdas[best_idx]:.2f}$")
        ax.set_xscale("log")
        ax.legend()
    else:
        _nodata(ax, "No lambda-sweep runs found")

    ax.set_xlabel(r"PDE Weight $\lambda_{\mathrm{PDE}}$")
    ax.set_ylabel("Extrapolation RMSE")
    ax.set_title("Effect of PDE Regularisation on LV Recovery")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "lv_smoothness_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_lv_extrap(data: dict, output_dir: Path) -> None:
    """Generate fig:lv-extrap from density experiment results (aggregated metrics)."""
    print("  [lv_extrap_dense.png, lv_extrap_sparse.png]")

    for tag, title, fname in [
        ("lv_density_50x20.json", "Dense Training Grid (50x20)", "lv_extrap_dense.png"),
        ("lv_density_10x5.json", "Sparse Training Grid (10x5)", "lv_extrap_sparse.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))

        rmse = _get(data, tag, "rmse_price")
        extrap = _get(data, tag, "rmse_extrap")
        if rmse is not None:
            rmse_std = _get(data, tag, "rmse_price", "std") or 0
            extrap_std = _get(data, tag, "rmse_extrap", "std") or 0
            labels = ["Price RMSE", "Extrap RMSE"]
            vals = [rmse * 1000, (extrap or 0) * 1000]
            stds = [rmse_std * 1000, extrap_std * 1000]
            ax.bar(labels, vals, yerr=stds, capsize=5, color=["steelblue", "coral"], alpha=0.8)
            ax.set_ylabel(r"RMSE ($\times 10^{-3}$)")
        else:
            _nodata(ax)

        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ===========================================================================
# Heston Figures
# ===========================================================================

def gen_heston_param_scatter(data: dict, output_dir: Path) -> None:
    """Generate fig:heston-scatter from Heston sweep results.

    Per the corrected thesis (Ch9), neural methods are evaluated only on
    surface-learning metrics, not on parameter recovery.  This figure shows
    Price RMSE across different Heston parameter settings for the classical
    baseline and NN-M2.
    """
    print("  [heston_param_scatter.png]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    heston_keys = [k for k in data if "heston" in k.lower() and "sweep" in k.lower()]
    if not heston_keys:
        heston_keys = [k for k in data if "heston" in k.lower()]

    sweep_rmse = {}
    for k in sorted(data.keys()):
        if k.startswith("heston_sweep_") or (k.startswith("sweep_heston") and "mean" in data.get(k, {})):
            rmse = _get(data, k, "rmse_price")
            if rmse is not None:
                sweep_rmse[k] = rmse

    method_specs = [
        ("NN", ["heston_nn_baseline.json"]),
        ("NN-M2", ["heston_nnm2.json", "heston_full_vs_priceonly.json"]),
        ("Classical (L-BFGS-B)", ["classical_heston"]),
    ]

    has_data = any(
        data.get(key, {}).get("mean", {}).get("rmse_price") is not None
        for _, keys in method_specs
        for key in keys
    )
    if has_data:
        labels = []
        rmse_vals = []
        std_vals = []
        for label, keys in method_specs:
            r = next(
                (
                    data.get(key, {})
                    for key in keys
                    if data.get(key, {}).get("mean", {}).get("rmse_price") is not None
                ),
                {},
            )
            m = r.get("mean", {}).get("rmse_price")
            if m is not None:
                labels.append(label)
                rmse_vals.append(m * 1000)
                std_vals.append((r.get("std", {}).get("rmse_price", 0)) * 1000)
        if labels:
            x = np.arange(len(labels))
            ax1.bar(x, rmse_vals, yerr=std_vals, capsize=5, color="steelblue", alpha=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.set_ylabel(r"Price RMSE ($\times 10^{-3}$)")
            ax1.set_title("Heston: Method Comparison")
            ax1.grid(True, axis="y", alpha=0.3)
        else:
            _nodata(ax1)
    else:
        _nodata(ax1)

    if sweep_rmse:
        keys = list(sweep_rmse.keys())
        vals = [sweep_rmse[k] * 1000 for k in keys]
        short_labels = [k.replace("heston_sweep_", "").replace("sweep_heston_", "") for k in keys]
        x = np.arange(len(keys))
        ax2.bar(x, vals, color="coral", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel(r"Price RMSE ($\times 10^{-3}$)")
        ax2.set_title("Heston: Parameter Sensitivity")
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        _nodata(ax2, "No Heston sweep data")

    plt.tight_layout()
    fig.savefig(output_dir / "heston_param_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_correlation_error_plot(data: dict, output_dir: Path) -> None:
    """Generate fig:corr-error from Heston sweep (rho variation)."""
    print("  [correlation_error_plot.png]")

    fig, ax = plt.subplots(figsize=(7, 5))

    rho_vals = []
    rmse_vals = []
    for key in sorted(data.keys()):
        if "heston" in key and "sweep" in key and "rho" in key:
            r = data[key]
            rmse = r.get("mean", {}).get("rmse_price")
            if rmse is None:
                continue
            parts = key.split("rho")
            if len(parts) > 1:
                try:
                    rho = float(parts[-1].split("_")[0])
                except ValueError:
                    continue
                rho_vals.append(rho)
                rmse_vals.append(rmse * 1000)

    if rho_vals:
        order = np.argsort(rho_vals)
        rho_vals = np.array(rho_vals)[order]
        rmse_vals = np.array(rmse_vals)[order]
        ax.plot(rho_vals, rmse_vals, "o-", color="steelblue", linewidth=2, markersize=8)
    else:
        _nodata(ax, "No Heston rho-sweep data")

    ax.set_xlabel(r"Correlation $\rho$")
    ax.set_ylabel(r"Price RMSE ($\times 10^{-3}$)")
    ax.set_title("Price Error vs Heston Correlation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_error_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_grid_density_error(data: dict, output_dir: Path) -> None:
    """Generate fig:grid-density from density experiment pair."""
    print("  [grid_density_error.png]")

    fig, ax = plt.subplots(figsize=(7, 5))

    sparse = data.get("lv_density_10x5.json", {})
    dense = data.get("lv_density_50x20.json", {})

    has_data = bool(sparse.get("mean")) or bool(dense.get("mean"))
    if has_data:
        labels = ["10x5 (Sparse)", "50x20 (Dense)"]
        rmse = [
            (sparse.get("mean", {}).get("rmse_price", 0)) * 1000,
            (dense.get("mean", {}).get("rmse_price", 0)) * 1000,
        ]
        stds = [
            (sparse.get("std", {}).get("rmse_price", 0)) * 1000,
            (dense.get("std", {}).get("rmse_price", 0)) * 1000,
        ]
        extrap_rmse = [
            (sparse.get("mean", {}).get("rmse_extrap", 0)) * 1000,
            (dense.get("mean", {}).get("rmse_extrap", 0)) * 1000,
        ]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, rmse, width, yerr=stds, capsize=5, label="Price RMSE",
               color="#e74c3c", alpha=0.8)
        ax.bar(x + width / 2, extrap_rmse, width, label="Extrap RMSE",
               color="#2ecc71", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    else:
        _nodata(ax)

    ax.set_ylabel(r"RMSE ($\times 10^{-3}$)")
    ax.set_title("Grid Density Impact on Local Vol Recovery")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "grid_density_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Ablation Figures
# ===========================================================================

def gen_ablation_A1(data: dict, output_dir: Path) -> None:
    """Generate fig:ablationA1 - marginal effect of Greek supervision.

    Plots all four Black-Scholes neural variants so that both factorial
    pairs are visible: NN vs NN-M2 (effect of adding Greek labels without
    PDE regularisation) and NN-PDE vs PINN (effect of adding Greek labels
    with PDE regularisation). Colour encodes whether Greek supervision is
    present so the marginal effect is legible within each pair.
    """
    from matplotlib.patches import Patch

    print("  [ablation_A1.png]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4),
                                   constrained_layout=True)

    nn = data.get("bs_nn_baseline.json", data.get("bs_softplus_price_only.json", {}))
    nnm2 = data.get("bs_nnm2.json", {})
    nnpde = data.get("bs_nnpde.json", {})
    full = data.get("bs_softplus_full.json", data.get("act_ablation_bs.json", {}))

    NO_GREEK = "#e07b54"    # coral : prices only / prices + PDE
    WITH_GREEK = "#3a6ea5"  # blue  : prices + Greeks / prices + Greeks + PDE

    # Fixed order: NN, NN-M2, NN-PDE, PINN.
    spec = [
        ("NN", nn, NO_GREEK),
        ("NN-M2", nnm2, WITH_GREEK),
        ("NN-PDE", nnpde, NO_GREEK),
        ("PINN", full, WITH_GREEK),
    ]
    method_data = [(label, d, c) for label, d, c in spec if d.get("mean")]

    if not method_data:
        _nodata(ax1)
        _nodata(ax2)
    else:
        methods = [l for l, _, _ in method_data]
        colors = [c for _, _, c in method_data]
        x = np.arange(len(methods))

        def _panel(ax, metric, scale, ylabel, title, fmt):
            vals = [d.get("mean", {}).get(metric, 0) * scale for _, d, _ in method_data]
            errs = [d.get("std", {}).get(metric, 0) * scale for _, d, _ in method_data]
            ax.bar(x, vals, yerr=errs, capsize=5, color=colors,
                   alpha=0.9, edgecolor="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_title(title, fontsize=13)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=11)
            top = max((v + e) for v, e in zip(vals, errs))
            ax.set_ylim(0, top * 1.20)
            for xi, v, e in zip(x, vals, errs):
                ax.text(xi, v + e + top * 0.025, fmt.format(v),
                        ha="center", va="bottom", fontsize=10.5)

        _panel(ax1, "rmse_price", 1000.0,
               r"Price RMSE ($\times 10^{-3}$)", "Price Error", "{:.1f}")
        _panel(ax2, "mae_gamma", 1.0,
               "Gamma MAE", "Greek Error", "{:.2f}")

        legend_handles = [
            Patch(facecolor=NO_GREEK, edgecolor="black",
                  label="Without Greek supervision"),
            Patch(facecolor=WITH_GREEK, edgecolor="black",
                  label="With Greek supervision"),
        ]
        fig.legend(handles=legend_handles, loc="outside lower center",
                   ncol=2, fontsize=10.5, frameon=False)

    fig.suptitle("Ablation A1: Effect of Greek Supervision", fontsize=15)
    fig.savefig(output_dir / "ablation_A1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_ablation_A2(data: dict, output_dir: Path) -> None:
    """Generate fig:ablationA2 - Activation function comparison."""
    print("  [ablation_A2.png]")

    fig, ax = plt.subplots(figsize=(8, 5))

    act_keys = [
        ("act_ablation_bs.json", "Softplus"),
        ("act_ablation_tanh.json", "Tanh"),
        ("act_ablation_relu.json", "ReLU"),
    ]
    present = [(k, l) for k, l in act_keys if data.get(k, {}).get("mean")]

    if not present:
        _nodata(ax)
    else:
        labels = [l for _, l in present]
        delta_mae = [_get(data, k, "mae_delta") or 0 for k, _ in present]
        delta_std = [_get(data, k, "mae_delta", "std") or 0 for k, _ in present]
        gamma_mae = [_get(data, k, "mae_gamma") or 0 for k, _ in present]
        gamma_std = [_get(data, k, "mae_gamma", "std") or 0 for k, _ in present]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, delta_mae, width, yerr=delta_std,
               label=r"$\mathrm{MAE}_\Delta$", color="steelblue", alpha=0.8, capsize=4)
        ax.bar(x + width / 2, gamma_mae, width, yerr=gamma_std,
               label=r"$\mathrm{MAE}_\Gamma$", color="coral", alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Ablation A2: Activation Function Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_A2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_ablation_A3(data: dict, output_dir: Path) -> None:
    """Generate fig:ablationA3 - Collocation density / grid density effect."""
    print("  [ablation_A3.png]")

    fig, ax = plt.subplots(figsize=(6, 4))

    sparse = data.get("lv_density_10x5.json", {})
    dense = data.get("lv_density_50x20.json", {})

    # The 10x5 grid (50 nodes) leaves 21 training nodes and the 50x20 grid
    # (1000 nodes) 420 training nodes after the extrapolation carve-out and the
    # 7:1:1 split of the interior nodes (Chapter 8).
    has_data = bool(sparse.get("mean")) or bool(dense.get("mean"))
    if has_data:
        labels = []
        extrap_rmse = []
        extrap_std = []
        for tag, lab in [("lv_density_10x5.json", "10 x 5 grid\n(21 training nodes)"),
                         ("lv_density_50x20.json", "50 x 20 grid\n(420 training nodes)")]:
            r = data.get(tag, {})
            e = r.get("mean", {}).get("rmse_extrap")
            if e is not None:
                labels.append(lab)
                extrap_rmse.append(1e3 * e)
                extrap_std.append(1e3 * r.get("std", {}).get("rmse_extrap", 0))

        if labels:
            x = np.arange(len(labels))
            ax.errorbar(x, extrap_rmse, yerr=extrap_std, fmt="o", capsize=5,
                        color="steelblue", markersize=8, linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.6, len(labels) - 0.4)
    else:
        _nodata(ax)

    ax.set_ylabel(r"Extrapolation price RMSE ($\times 10^{-3}$)")
    ax.set_title("Ablation A3: grid density (local-volatility PINN)", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_A3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_ablation_A4(data: dict, output_dir: Path) -> None:
    """Generate fig:ablationA4 - Noise robustness."""
    print("  [ablation_A4.png]")

    fig, ax = plt.subplots(figsize=(7, 5))

    noise_labels = []
    noise_rmse = []
    noise_std = []
    for key in sorted(data.keys()):
        if key.startswith("noise_"):
            r = data[key]
            try:
                lvl = float(key.replace("noise_", "")) * 100
            except ValueError:
                continue
            m = r.get("mean", {}).get("rmse_price")
            if m is not None:
                noise_labels.append(lvl)
                noise_rmse.append(m)
                noise_std.append(r.get("std", {}).get("rmse_price", 0))

    if noise_labels:
        ax.errorbar(noise_labels, noise_rmse, yerr=noise_std, fmt="o-",
                    capsize=4, color="steelblue", linewidth=2, markersize=8, label="PINN")
        ax.legend()
    else:
        _nodata(ax, "No noise ablation data available")

    ax.set_xlabel("Noise Level (%)")
    ax.set_ylabel("Price RMSE")
    ax.set_title("Ablation A4: Noise Robustness")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_A4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gen_failure_map_ablation(data: dict, output_dir: Path) -> None:
    """Generate fig:failure-map-ablation from activation x PDE-weight grid."""
    print("  [failure_map_ablation.png]")

    fig, ax = plt.subplots(figsize=(8, 6))

    act_keys = {
        "Softplus": "act_ablation_bs.json",
        "Tanh": "act_ablation_tanh.json",
        "ReLU": "act_ablation_relu.json",
    }

    has_any = any(data.get(v, {}).get("mean") for v in act_keys.values())
    if has_any:
        activations = list(act_keys.keys())
        rmse_vals = []
        for act in activations:
            r = data.get(act_keys[act], {})
            rmse_vals.append(r.get("mean", {}).get("rmse_price", float("nan")))

        rmse_arr = np.array(rmse_vals).reshape(-1, 1)
        im = ax.imshow(rmse_arr, cmap="RdYlGn_r", aspect="auto")
        ax.set_yticks(np.arange(len(activations)))
        ax.set_yticklabels(activations)
        ax.set_xticks([0])
        ax.set_xticklabels(["Default"])
        ax.set_xlabel("Configuration")
        fig.colorbar(im, ax=ax, label="Price RMSE")

        for i, v in enumerate(rmse_vals):
            if not np.isnan(v):
                ax.text(0, i, f"{v:.4f}", ha="center", va="center", fontsize=10)
    else:
        _nodata(ax, "No activation ablation data for failure map")

    ax.set_title("Failure Map: Activation vs Error")
    plt.tight_layout()
    fig.savefig(output_dir / "failure_map_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None):
    """Generate all thesis figures.

    ``argv`` lets callers pass an explicit argument list. When ``None``
    (standalone CLI use) argparse falls back to ``sys.argv``. Importers MUST
    pass an explicit list (e.g. ``main([])`` or ``main(["--results-file", p])``)
    so the parent process's CLI arguments are never inspected.
    """
    parser = argparse.ArgumentParser(description="Generate all thesis figures")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Path to results JSON file (default: latest in results/)")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("GENERATING ALL THESIS FIGURES")
    print("=" * 60)

    if args.results_file:
        results_file = Path(args.results_file)
    else:
        try:
            results_file = find_latest_results()
        except FileNotFoundError:
            print("WARNING: No results file found. Figures will show 'no data'.")
            results_file = None

    if results_file and results_file.exists():
        print(f"\nLoading: {results_file}")
        with open(results_file, "r") as f:
            data = json.load(f)
        print(f"  Keys: {list(data.keys())}")
    else:
        print("\nNo results file found. Figures will show 'no data' for metric-based plots.")
        data = {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    print("Chapter 3 - Differential NN:")
    gen_learning_rate_schedule(data, OUTPUT_DIR)

    print("\nChapter 7 - Methodology:")
    gen_method_comparison(data, OUTPUT_DIR)

    print("\nChapter 9 - Black-Scholes:")
    gen_bs_sigma_recovery(data, OUTPUT_DIR)
    gen_bs_price_scatter(data, OUTPUT_DIR)
    gen_bs_greek_mae(data, OUTPUT_DIR)
    gen_bs_loss_curve(data, OUTPUT_DIR)
    gen_activation_bar(data, OUTPUT_DIR)

    print("\nChapter 9 - Local Volatility:")
    gen_lv_surfaces(data, OUTPUT_DIR)
    gen_lv_dupire_surfaces(data, OUTPUT_DIR)
    gen_lv_error_heatmap(data, OUTPUT_DIR)
    gen_lv_smoothness_plot(data, OUTPUT_DIR)
    gen_lv_extrap(data, OUTPUT_DIR)

    print("\nChapter 10 - Heston:")
    gen_heston_param_scatter(data, OUTPUT_DIR)
    gen_correlation_error_plot(data, OUTPUT_DIR)
    gen_grid_density_error(data, OUTPUT_DIR)

    print("\nChapter 10 - Ablation studies:")
    gen_ablation_A1(data, OUTPUT_DIR)
    gen_ablation_A2(data, OUTPUT_DIR)
    gen_ablation_A3(data, OUTPUT_DIR)
    gen_ablation_A4(data, OUTPUT_DIR)
    gen_failure_map_ablation(data, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"DONE! Generated 24 PNG figures in: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nNote: 6 figures are TikZ diagrams (inline LaTeX) - not generated here.")
    print("Next: Recompile thesis to see updated figures.")


if __name__ == "__main__":
    main()

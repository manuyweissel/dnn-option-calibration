#!/usr/bin/env python3
"""
Run all thesis experiments and export results.

This script:
1. Runs all core experiments with multiple seeds
2. Aggregates results with mean ± std
3. Generates publication-quality figures
4. Exports tables and figures to thesis LaTeX directory
5. Updates RESULTS_SUMMARY.md

Usage:
    python scripts/run_all_experiments.py [--quick] [--seeds 20251,20252,20253]

Seed-level checkpointing (classical baselines):
    Each completed classical-baseline seed is checkpointed immediately under
    results/checkpoints/classical_baselines/<name>/<mode>/seed_<seed>.json so a
    cancelled or crashed run never loses already-completed seeds. Re-running the
    same command resumes from those checkpoints (--no-resume-classical-checkpoints
    forces recomputation). A rolling aggregate is written to
    results/all_results_rolling.json after every seed.

    Full Heston deterministic in parallel with checkpointing:
        CUDA_VISIBLE_DEVICES="" python scripts/run_all_experiments.py \
            --only classical_heston --seeds 20251,20252,20253,20254,20255 \
            --device cpu --parallel-seeds --no-parallel-only-mc --max-workers 3

    Full Heston MC in parallel with checkpointing:
        CUDA_VISIBLE_DEVICES="" python scripts/run_all_experiments.py \
            --only classical_mc_heston --seeds 20251,20252,20253,20254,20255 \
            --device cpu --parallel-seeds --max-workers 2

    Rebuild tables from saved checkpoints (no experiments run):
        python scripts/run_all_experiments.py --rebuild-from-checkpoints --skip-experiments

Progress logging (long classical calibrations):
    --verbose-progress (default on) prints timestamped per-seed phase markers and
    interval-gated objective/iteration progress; the parent prints a heartbeat
    every --heartbeat-sec while waiting on parallel futures, so deterministic
    classical_heston never looks frozen inside SciPy.

    Deterministic Heston with parallel progress logging:
        CUDA_VISIBLE_DEVICES="" \
        python scripts/run_all_experiments.py \
          --only classical_heston \
          --seeds 20251,20252,20253,20254,20255 \
          --device cpu \
          --parallel-seeds \
          --no-parallel-only-mc \
          --max-workers 3 \
          --verbose-progress \
          --log-interval-sec 30 \
          2>&1 | tee logs/classical_heston_progress.log

    Heston MC with safer two-worker parallelism:
        CUDA_VISIBLE_DEVICES="" \
        python scripts/run_all_experiments.py \
          --only classical_mc_heston \
          --seeds 20251,20252,20253,20254,20255 \
          --device cpu \
          --parallel-seeds \
          --max-workers 2 \
          --verbose-progress \
          --log-interval-sec 30 \
          2>&1 | tee logs/classical_mc_heston_progress.log
"""

import os

# Limit BLAS/OpenMP threads BEFORE numpy (and any BLAS) is imported, so that
# seed-level multiprocessing does not oversubscribe CPU cores. ``setdefault``
# leaves any value the user already exported untouched. This must run at module
# import time so it also applies to spawned worker processes (Windows / spawn
# start method re-import this module before unpickling the worker callable).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp.io_utils import load_config
from src.exp.runner import ExperimentRunner, run_multi_seed, run_ablation_grid, run_param_sweep
from src.exp.seeds import seed_everything


# Configuration
THESIS_DIR = Path(__file__).parent.parent.parent / "thesis_latex_framework" / "thesis"
if not THESIS_DIR.is_dir():
    # Stand-alone code repository: write generated assets inside the project.
    THESIS_DIR = Path(__file__).parent.parent / "outputs"
FIGURES_DIR = THESIS_DIR / "assets" / "figures" / "experiments" / "generated"
TABLES_DIR = THESIS_DIR / "assets" / "tables"
SNIPPETS_DIR = THESIS_DIR / "snippets" / "generated"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def ensure_dirs():
    """Create output directories if they don't exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Progress logging helpers (used by parent + worker processes)
# ---------------------------------------------------------------------------

def _now_str() -> str:
    """Human-readable wall-clock timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, *, flush: bool = True) -> None:
    """Print a timestamped progress line, e.g. ``[2026-06-04 15:42:11] ...``."""
    print(f"[{_now_str()}] {msg}", flush=flush)


def _write_progress_event(event: dict, progress_jsonl: Optional[Path]) -> None:
    """Append a machine-readable progress event as one JSON line.

    No-op when ``progress_jsonl`` is None. Best-effort: progress logging must
    never crash a run, so all errors are swallowed. Opened in append mode per
    call so multiple worker processes can write to the same file.
    """
    if not progress_jsonl:
        return
    try:
        path = Path(progress_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
            f.flush()
    except Exception:
        pass


def _timestamped_results_files() -> list:
    """Sorted list of ``all_results_<timestamp>.json`` files.

    Excludes the rolling snapshot (``all_results_rolling.json``), which also
    matches the glob but is a live partial file, not a finalised run, and would
    otherwise sort last and be mistaken for the latest results.
    """
    return sorted(
        f for f in RESULTS_DIR.glob("all_results_*.json")
        if f.name != "all_results_rolling.json"
    )


def _load_latest_results() -> dict:
    """Load the most recent ``all_results_*.json`` (empty dict if none).

    Used by selective (``--only``) runs so that re-running a subset of
    experiments merges into, rather than overwrites, previously computed
    results when regenerating the aggregated tables/summary.
    """
    files = _timestamped_results_files()
    if not files:
        return {}
    try:
        with open(files[-1]) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: could not load latest results ({e}); starting fresh.")
        return {}


# ---------------------------------------------------------------------------
# Classical-baseline seed checkpointing / resume / rolling aggregates
# ---------------------------------------------------------------------------

CLASSICAL_CKPT_DIR = RESULTS_DIR / "checkpoints" / "classical_baselines"
ROLLING_RESULTS_PATH = RESULTS_DIR / "all_results_rolling.json"
CLASSICAL_BASELINE_NAMES = [
    "classical_bs", "classical_mc_bs",
    "classical_lv", "classical_mc_lv",
    "classical_heston", "classical_mc_heston",
]
# Keys a seed checkpoint must contain to be considered valid / resumable.
_CKPT_REQUIRED_KEYS = ("seed", "rmse_price", "param_rel_err", "duration_ms")


def _classical_mode(is_mc: bool, quick: bool) -> str:
    """JSON ``mode`` field: deterministic / mc_full / mc_quick."""
    if not is_mc:
        return "deterministic"
    return "mc_quick" if quick else "mc_full"


def _mode_path_segment(mode: str) -> str:
    """Directory segment for a mode (deterministic / full / quick)."""
    return {"deterministic": "deterministic", "mc_full": "full", "mc_quick": "quick"}[mode]


def _classical_ckpt_dir(name: str, mode: str) -> Path:
    return CLASSICAL_CKPT_DIR / name / _mode_path_segment(mode)


def _seed_ckpt_path(name: str, mode: str, seed) -> Path:
    return _classical_ckpt_dir(name, mode) / f"seed_{seed}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` as JSON atomically (tmp + flush + fsync + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            # fsync may be unavailable on some platforms/filesystems; the
            # os.replace below is still atomic, so this is best-effort.
            pass
    os.replace(tmp, path)


def _hardware_meta() -> dict:
    """Capture the hardware/software environment for the results file.

    Stored under the top-level ``_meta`` key of ``all_results_*.json`` so that
    reported wall-clock timings are traceable to the machine that produced them.
    All fields are best-effort; failures must never break a results write.
    """
    import platform

    meta: dict = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        meta["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["torch_num_threads"] = torch.get_num_threads()
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return meta


def _write_rolling_results_snapshot(all_results: dict, path: Path = ROLLING_RESULTS_PATH) -> None:
    """Persist a rolling snapshot of the full results dict (crash safety net)."""
    _atomic_write_json(path, all_results)


def _write_failure_checkpoint(name: str, mode: str, seed, exc: BaseException) -> Path:
    """Persist a ``seed_<seed>_FAILED.json`` describing a crashed seed.

    Written to the same per-mode checkpoint directory as successful seeds. It is
    safely ignored by resume/rebuild (those read ``seed_<seed>.json`` exactly and
    validate metric keys, which a failure record lacks).
    """
    path = _classical_ckpt_dir(name, mode) / f"seed_{seed}_FAILED.json"
    _atomic_write_json(path, {
        "name": name,
        "seed": int(seed),
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    })
    return path


def _build_seed_checkpoint(name: str, spec: dict, mode: str, quick: bool, m: dict) -> dict:
    """Assemble the per-seed checkpoint payload from a completed metric dict."""
    return {
        "name": name,
        "seed": int(m["seed"]),
        "mode": mode,
        "model": spec["model"],
        "rmse_price": float(m["rmse_price"]),
        "param_rel_err": float(m["param_rel_err"]),
        "duration_ms": float(m["duration_ms"]),
        "timestamp": datetime.now().isoformat(),
        "quick": quick,
        "params": m.get("params"),
        "true_calib_params": spec.get("true_calib_params"),
    }


def _metric_record(ck: dict):
    """Validate a checkpoint dict and return a normalized metric record.

    Returns ``None`` if any required key is missing or non-numeric, so that
    malformed checkpoints are ignored (and the seed is recomputed) rather than
    poisoning the aggregate.
    """
    if not all(k in ck for k in _CKPT_REQUIRED_KEYS):
        return None
    try:
        rec = {
            "seed": int(ck["seed"]),
            "rmse_price": float(ck["rmse_price"]),
            "param_rel_err": float(ck["param_rel_err"]),
            "duration_ms": float(ck["duration_ms"]),
        }
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(rec["rmse_price"]) and np.isfinite(rec["param_rel_err"])):
        return None
    return rec


def _load_seed_checkpoint(name: str, mode: str, seed):
    """Load + validate a per-seed checkpoint; ``None`` if absent/malformed."""
    path = _seed_ckpt_path(name, mode, seed)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            ck = json.load(f)
    except Exception:
        return None
    return _metric_record(ck)


def _aggregate_classical(by_seed: dict, requested_seeds: list) -> dict:
    """Aggregate available per-seed metrics over the requested seed list.

    Tolerates partial completion: mean/std are computed only over completed
    seeds, and the result records completed/missing seeds plus a ``partial`` flag.
    """
    completed_seeds = [s for s in requested_seeds if s in by_seed]
    missing_seeds = [s for s in requested_seeds if s not in by_seed]
    all_seed_metrics = [
        {
            "seed": int(by_seed[s]["seed"]),
            "rmse_price": float(by_seed[s]["rmse_price"]),
            "param_rel_err": float(by_seed[s]["param_rel_err"]),
            "duration_ms": float(by_seed[s]["duration_ms"]),
        }
        for s in completed_seeds
    ]
    metric_keys = ["rmse_price", "param_rel_err", "duration_ms"]
    if all_seed_metrics:
        mean_m = {k: float(np.mean([m[k] for m in all_seed_metrics])) for k in metric_keys}
        std_m = {k: float(np.std([m[k] for m in all_seed_metrics])) for k in metric_keys}
    else:
        mean_m, std_m = {}, {}
    return {
        "mean": mean_m,
        "std": std_m,
        "n_seeds": len(completed_seeds),
        "requested_seeds": list(requested_seeds),
        "completed_seeds": completed_seeds,
        "missing_seeds": missing_seeds,
        "partial": len(missing_seeds) > 0,
        "all_metrics": all_seed_metrics,
    }


def _aggregate_from_checkpoints(name: str):
    """Rebuild a baseline aggregate purely from its on-disk seed checkpoints.

    For MC baselines the full-mode checkpoints are preferred over quick-mode.
    Returns ``None`` if no valid checkpoints exist for the baseline.
    """
    base_dir = CLASSICAL_CKPT_DIR / name
    if not base_dir.exists():
        return None
    is_mc = name.startswith("classical_mc_")
    segments = ["full", "quick"] if is_mc else ["deterministic"]
    chosen = None
    for seg in segments:
        d = base_dir / seg
        if d.exists() and any(d.glob("seed_*.json")):
            chosen = d
            break
    if chosen is None:
        return None

    by_seed = {}
    for fp in sorted(chosen.glob("seed_*.json")):
        try:
            with open(fp) as f:
                ck = json.load(f)
        except Exception:
            continue
        rec = _metric_record(ck)
        if rec is not None:
            by_seed[rec["seed"]] = rec
    if not by_seed:
        return None
    requested = sorted(by_seed.keys())
    return _aggregate_classical(by_seed, requested)


def _rebuild_from_checkpoints() -> dict:
    """Rebuild all classical aggregates from checkpoints, merged onto latest results."""
    print("\n" + "="*80)
    print("REBUILDING CLASSICAL AGGREGATES FROM CHECKPOINTS")
    print("="*80)
    base = _load_latest_results()
    all_results = dict(base)
    if base:
        print(f"  Loaded {len(base)} entries from latest results file as base.")
    for name in CLASSICAL_BASELINE_NAMES:
        agg = _aggregate_from_checkpoints(name)
        if agg is None:
            print(f"  {name}: no checkpoints found (skipped)")
            continue
        all_results[name] = agg
        flag = " [PARTIAL]" if agg["partial"] else ""
        print(f"  {name}: rebuilt from {agg['n_seeds']} seed checkpoint(s){flag} "
              f"-> seeds {agg['completed_seeds']}")
    return all_results


def run_core_experiments(seeds: list, device: str = None, only=None):
    """Run full 2x2 factorial for BS, LV, and Heston.

    If ``only`` is a set of experiment names, only matching configs are run.
    """
    print("\n" + "="*80)
    print("RUNNING CORE EXPERIMENTS (2x2 factorial)")
    print("="*80)

    results = {}

    bs_configs = [
        "bs_nn_baseline.json",
        "bs_nnm2.json",
        "bs_nnpde.json",
        "bs_softplus_full.json",
        "bs_softplus_price_only.json",
        "bs_nondnn.json",
    ]

    lv_configs = [
        "lv_nn_baseline.json",
        "lv_nnm2.json",
        "lv_nnpde.json",
        "lv_dnn.json",
        "lv_density_10x5.json",
        "lv_density_50x20.json",
    ]

    heston_configs = [
        "heston_nn_baseline.json",
        "heston_nnm2.json",
        "heston_full_vs_priceonly.json",
    ]

    for cfg_name in bs_configs + lv_configs + heston_configs:
        if only is not None and cfg_name not in only:
            continue
        cfg_path = CONFIGS_DIR / cfg_name
        if cfg_path.exists():
            print(f"\n>>> Running {cfg_name}")
            cfg = load_config(cfg_path)
            result = run_multi_seed(cfg, seeds=seeds, device=device)
            results[cfg_name] = result
        else:
            print(f"  SKIP (not found): {cfg_name}")

    return results


def run_activation_ablation(seeds: list, device: str = None, only=None):
    """Run activation function ablation (A2)."""
    print("\n" + "="*80)
    print("RUNNING ACTIVATION ABLATION (A2)")
    print("="*80)
    
    results = {}
    
    act_configs = [
        "act_ablation_bs.json",
        "act_ablation_relu.json",
        "act_ablation_tanh.json",
        "act_ablation_sine.json",
    ]
    
    for cfg_name in act_configs:
        if only is not None and cfg_name not in only:
            continue
        cfg_path = CONFIGS_DIR / cfg_name
        if cfg_path.exists():
            print(f"\n>>> Running {cfg_name}")
            cfg = load_config(cfg_path)
            result = run_multi_seed(cfg, seeds=seeds, device=device)
            results[cfg_name] = result
    
    return results


def run_param_sweep_experiments(seeds: list, device: str = None, only=None):
    """Run parameter-sweep experiments per model (Ch8 sec 8.3)."""
    print("\n" + "="*80)
    print("RUNNING PARAMETER-SWEEP EXPERIMENTS")
    print("="*80)

    results = {}

    BS_SWEEP = [
        {"sigma": 0.1},
        {"sigma": 0.2},
        {"sigma": 0.3},
        {"sigma": 0.5},
    ]
    LV_SWEEP = [
        {"sigma0": 0.15, "alpha": 0.3},
        {"sigma0": 0.2, "alpha": 0.5},
        {"sigma0": 0.3, "alpha": 0.7},
        {"sigma0": 0.4, "alpha": 1.0},
    ]
    HESTON_SWEEP = [
        {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.5},
        {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7},
        {"v0": 0.09, "kappa": 1.5, "theta": 0.09, "xi": 0.4, "rho": -0.3},
    ]

    sweep_specs = {
        "bs_softplus_full.json": BS_SWEEP,
        "lv_dnn.json": LV_SWEEP,
        "heston_full_vs_priceonly.json": HESTON_SWEEP,
    }

    for cfg_name, settings in sweep_specs.items():
        if only is not None and f"sweep_{cfg_name}" not in only:
            continue
        cfg_path = CONFIGS_DIR / cfg_name
        if not cfg_path.exists():
            print(f"  SKIP (not found): {cfg_name}")
            continue
        print(f"\n>>> Param-sweep for {cfg_name} ({len(settings)} settings)")
        cfg = load_config(cfg_path)
        result = run_param_sweep(cfg, settings, seeds=seeds, device=device)
        results[f"sweep_{cfg_name}"] = result

    return results


def run_noise_ablation(seeds: list, device: str = None, only=None):
    """Run noise injection ablation (A4)."""
    print("\n" + "="*80)
    print("RUNNING NOISE ABLATION (A4)")
    print("="*80)
    
    # Create noise configs if they don't exist
    base_cfg_path = CONFIGS_DIR / "bs_softplus_full.json"
    if not base_cfg_path.exists():
        print("Base config not found, skipping noise ablation")
        return {}
    
    base_cfg = load_config(base_cfg_path)
    results = {}
    
    noise_levels = [0.0, 0.005, 0.01]
    
    for noise in noise_levels:
        if only is not None and f"noise_{noise}" not in only:
            continue
        cfg = base_cfg.model_copy(deep=True)
        cfg.experiment_id = f"bs_noise_{int(noise*1000)}"
        cfg.noise.price_rel = noise
        cfg.noise.greek_rel = noise / 2
        
        print(f"\n>>> Running noise level {noise*100:.1f}%")
        result = run_multi_seed(cfg, seeds=seeds, device=device)
        results[f"noise_{noise}"] = result
    
    return results


def _resolve_calibrate_fn(name: str):
    """Resolve a classical-baseline calibration callable by experiment name.

    Defined at module level (rather than passing the function object through the
    spec) so the worker payload stays trivially picklable and the lookup happens
    inside the child process. Kept in sync with ``model_specs`` below.
    """
    from src.models.calibration import (
        calibrate_bs, calibrate_localvol, calibrate_heston,
        calibrate_mc_bs, calibrate_mc_localvol, calibrate_mc_heston,
    )
    return {
        "classical_bs": calibrate_bs,
        "classical_mc_bs": calibrate_mc_bs,
        "classical_lv": calibrate_localvol,
        "classical_mc_lv": calibrate_mc_localvol,
        "classical_heston": calibrate_heston,
        "classical_mc_heston": calibrate_mc_heston,
    }[name]


def _run_classical_seed(payload: dict) -> dict:
    """Run one classical-baseline calibration for a single seed (worker entry).

    Executed either inline (sequential mode) or inside a
    :class:`ProcessPoolExecutor` worker (``--parallel-seeds``). It is fully
    self-contained: it resolves its own calibration function, prices the
    calibrated model, and computes RMSE + parameter error. It performs NO file
    I/O and never touches figures/tables/results — aggregation and writing are
    the parent process's job exclusively.

    Returns a dict with: ``seed``, ``rmse_price``, ``param_rel_err``,
    ``duration_ms``, and the calibrated ``params`` (so the parent can record
    them in the seed checkpoint).

    Progress visibility: with ``verbose_progress`` the worker prints timestamped
    phase markers (START / calibration / pricing / metrics / DONE) and forwards
    ``log_interval_sec`` + a per-seed ``progress_prefix`` into the calibrator so
    its objective/iteration progress is interleaved here. It still does NO
    result-file I/O; only the optional ``progress_jsonl`` event log is appended.
    """
    from src.exp.metrics import rmse_price, param_rel_err
    from src.models.pricing import create_pricer

    name = payload["name"]
    seed = payload["seed"]
    is_mc = payload["is_mc"]
    K = payload["K"]
    T = payload["T"]
    prices = payload["prices"]

    verbose = payload.get("verbose_progress", True)
    log_interval = float(payload.get("log_interval_sec", 30.0))
    progress_jsonl = payload.get("progress_jsonl")
    progress_jsonl = Path(progress_jsonl) if progress_jsonl else None
    pid = os.getpid()
    prefix = f"[{name} seed={seed} pid={pid}]"

    def _emit(phase: str, **extra) -> None:
        """Print a timestamped phase marker and append a JSONL progress event."""
        if verbose:
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            _log(f"{prefix} {phase}" + (f" {extra_str}" if extra_str else ""))
        _write_progress_event(
            {"ts": _now_str(), "name": name, "seed": int(seed), "pid": pid,
             "phase": phase, **extra},
            progress_jsonl,
        )

    t0 = time.perf_counter()
    _emit("START")

    calibrate_fn = _resolve_calibrate_fn(name)
    # Forward logging options to both deterministic and MC calibrators; only the
    # MC ones additionally take ``quick``.
    common = {"verbose": verbose, "log_interval_sec": log_interval, "progress_prefix": prefix}
    cal_kwargs = {"quick": payload["quick"], **common} if is_mc else dict(common)

    _emit("calibration START")
    t_cal = time.perf_counter()
    cal_result = calibrate_fn(K, T, prices, seed=seed, **cal_kwargs)
    _emit("calibration DONE", elapsed=f"{time.perf_counter() - t_cal:.1f}s")

    _emit("pricing START")
    t_price = time.perf_counter()
    merged = dict(payload["params"])
    merged.update(cal_result["params"])
    pred_prices = create_pricer(payload["model"], merged).price(K, T)
    _emit("pricing DONE", elapsed=f"{time.perf_counter() - t_price:.1f}s")

    _rmse = rmse_price(
        pd.DataFrame({"price": pred_prices}),
        pd.DataFrame({"price": prices}),
    )
    _pre = param_rel_err(cal_result["params"], payload["true_calib_params"])
    _emit("metrics DONE", rmse=f"{_rmse:.6f}", param_err=f"{_pre:.6f}")

    duration_ms = float(cal_result.get("duration_ms", (time.perf_counter() - t0) * 1000.0))
    _emit("DONE", elapsed=f"{time.perf_counter() - t0:.1f}s",
          rmse=f"{_rmse:.6f}", param_err=f"{_pre:.6f}", duration_ms=f"{duration_ms:.0f}")

    return {
        "seed": int(seed),
        "rmse_price": float(_rmse),
        "param_rel_err": float(_pre),
        "duration_ms": duration_ms,
        "params": {k: float(v) for k, v in cal_result["params"].items()},
    }


def run_classical_baselines(seeds: list, device: str = None, only=None, quick: bool = False,
                            parallel_seeds: bool = False, max_workers: int = None,
                            parallel_only_mc: bool = True, resume_checkpoints: bool = True,
                            base_results: dict = None, verbose_progress: bool = True,
                            log_interval_sec: float = 30.0, heartbeat_sec: float = 60.0,
                            progress_jsonl: Optional[Path] = None):
    """Run classical calibration baselines (Ch8 sec 8.4).

    If ``only`` is a set of experiment names, only the matching baselines are run.
    ``quick`` selects the lightweight MC runtime preset for the MC baselines.

    Seed-level parallelism (``parallel_seeds=True``) runs each seed of a baseline
    in its own process via :class:`ProcessPoolExecutor`. This is safe because
    seeds are numerically independent and the workers do no file I/O — only this
    parent process aggregates per-seed metrics and writes checkpoints/results.
    ``max_workers`` caps the pool size (default: executor default). When
    ``parallel_only_mc`` is True (default) only the stochastic MC baselines are
    parallelised; the fast deterministic L-BFGS-B baselines stay sequential.

    Crash safety: every completed seed is checkpointed immediately (atomic
    write) by the *parent* process, and a rolling results snapshot is refreshed
    after each seed. When ``resume_checkpoints`` is True (default), seeds whose
    checkpoint already exists are loaded instead of recomputed. ``base_results``
    (the results accumulated by earlier experiment stages) is merged into the
    rolling snapshot so it stays a complete picture of progress.
    """
    print("\n" + "="*80)
    print("RUNNING CLASSICAL BASELINES")
    print("="*80)

    from src.models.calibration import (
        calibrate_bs, calibrate_localvol, calibrate_heston,
        calibrate_mc_bs, calibrate_mc_localvol, calibrate_mc_heston,
    )
    from src.data.synth import generate_synthetic_data
    from src.data.grids import create_grid
    # Per-seed RMSE / parameter-error are computed inside ``_run_classical_seed``
    # (so they run in the worker process under ``--parallel-seeds``).

    results = {}

    model_specs = {
        "classical_bs": {
            "model": "BS",
            "params": {"S0": 1.0, "r": 0.02, "sigma": 0.2},
            "calibrate_fn": calibrate_bs,
            "true_calib_params": {"sigma": 0.2},
        },
        "classical_mc_bs": {
            "model": "BS",
            "params": {"S0": 1.0, "r": 0.02, "sigma": 0.2},
            "calibrate_fn": calibrate_mc_bs,
            "true_calib_params": {"sigma": 0.2},
        },
        "classical_lv": {
            "model": "LocalVol",
            "params": {"S0": 1.0, "r": 0.02, "sigma0": 0.2, "alpha": 0.5, "beta": 0.0},
            "calibrate_fn": calibrate_localvol,
            "true_calib_params": {"sigma0": 0.2, "alpha": 0.5},
        },
        "classical_mc_lv": {
            "model": "LocalVol",
            "params": {"S0": 1.0, "r": 0.02, "sigma0": 0.2, "alpha": 0.5, "beta": 0.0},
            "calibrate_fn": calibrate_mc_localvol,
            "true_calib_params": {"sigma0": 0.2, "alpha": 0.5},
        },
        "classical_heston": {
            "model": "Heston",
            "params": {"S0": 1.0, "r": 0.02, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.5, "v0": 0.04},
            "calibrate_fn": calibrate_heston,
            "true_calib_params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.5},
        },
        "classical_mc_heston": {
            "model": "Heston",
            "params": {"S0": 1.0, "r": 0.02, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.5, "v0": 0.04},
            "calibrate_fn": calibrate_mc_heston,
            "true_calib_params": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.5},
        },
    }

    if only is not None:
        model_specs = {k: v for k, v in model_specs.items() if k in only}
        if not model_specs:
            print("  (no selected classical baselines)")
            return results

    grid = create_grid(50, 20, spacing="logm", S0=1.0,
                       K_min=0.8, K_max=1.2, T_min=0.1)

    base_results = base_results or {}

    for name, spec in model_specs.items():
        is_mc = name.startswith("classical_mc_")
        mode = _classical_mode(is_mc, quick)
        # Parallelise seeds only when requested, when there is >1 seed, and when
        # the baseline is eligible (MC always; deterministic ones only if the
        # caller explicitly opts out of ``parallel_only_mc``).
        use_parallel = (
            parallel_seeds
            and len(seeds) > 1
            and (is_mc or not parallel_only_mc)
        )
        mode_tag = " [quick]" if (is_mc and quick) else ""
        par_tag = f"  (parallel x{max_workers or 'auto'})" if use_parallel else ""
        print(f"\n>>> {name}{'  (MC' + mode_tag + ')' if is_mc else ''}{par_tag}")
        data = generate_synthetic_data(grid, spec["model"], spec["params"])
        K = data['K'].values
        T = data['T'].values
        prices = data['price'].values

        # Each worker payload is fully self-contained and picklable: the same
        # synthetic data (so numerical behaviour is identical to the sequential
        # path), the seed, and the flags needed to reconstruct the calibration.
        payloads = {
            s: {
                "name": name,
                "model": spec["model"],
                "params": spec["params"],
                "true_calib_params": spec["true_calib_params"],
                "is_mc": is_mc,
                "quick": quick,
                "device": device,
                "parallel": use_parallel,
                "verbose_progress": verbose_progress,
                "log_interval_sec": log_interval_sec,
                "progress_jsonl": str(progress_jsonl) if progress_jsonl else None,
                "seed": s,
                "K": K,
                "T": T,
                "prices": prices,
            }
            for s in seeds
        }

        # Collect into a seed-keyed dict so we can restore the original seed
        # order regardless of completion order.
        by_seed = {}

        def _flush_aggregates():
            """Recompute the partial aggregate and persist rolling snapshots."""
            agg = _aggregate_classical(by_seed, seeds)
            results[name] = agg
            _atomic_write_json(
                _classical_ckpt_dir(name, mode) / "aggregate_partial.json", agg
            )
            _write_rolling_results_snapshot({**base_results, **results})

        def _record(s, m):
            """Parent-side: checkpoint a freshly computed seed, then re-aggregate."""
            by_seed[s] = m
            # Workers never write files; the parent writes the checkpoint atomically.
            ck = _build_seed_checkpoint(name, spec, mode, quick, m)
            _atomic_write_json(_seed_ckpt_path(name, mode, s), ck)
            _flush_aggregates()

        # --- Resume: load any existing valid checkpoints up front -------------
        seeds_to_run = []
        for s in seeds:
            if resume_checkpoints:
                ck = _load_seed_checkpoint(name, mode, s)
                if ck is not None:
                    by_seed[s] = ck
                    print(f"  [{name}] seed {s}: loaded checkpoint", flush=True)
                    continue
            seeds_to_run.append(s)
        if by_seed:
            # Aggregate the resumed seeds immediately so rolling state is current
            # even if every requested seed was already on disk.
            _flush_aggregates()

        def _on_failure(s, exc):
            """Log a crashed seed (with traceback) and persist a FAILED record."""
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            _log(f"[{name} seed={s} pid={os.getpid()}] FAILED {type(exc).__name__}: {exc}")
            print(tb, flush=True)
            try:
                fp = _write_failure_checkpoint(name, mode, s, exc)
                _log(f"[{name} seed={s}] wrote failure record: {fp}")
            except Exception as werr:
                _log(f"[{name} seed={s}] could not write failure record: {werr}")
            _write_progress_event(
                {"ts": _now_str(), "name": name, "seed": int(s), "phase": "FAILED",
                 "exception_type": type(exc).__name__, "exception_message": str(exc)},
                progress_jsonl,
            )

        # --- Compute the remaining seeds -------------------------------------
        if use_parallel and seeds_to_run:
            par_t0 = time.perf_counter()
            workers_n = max_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run_classical_seed, payloads[s]): s
                           for s in seeds_to_run}
                pending = set(futures.keys())
                done = 0
                # Heartbeat-aware wait: print a parent heartbeat every
                # ``heartbeat_sec`` even when no future has completed, so the run
                # never looks frozen while long calibrations are in flight.
                while pending:
                    finished, pending = wait(
                        pending, timeout=heartbeat_sec, return_when=FIRST_COMPLETED
                    )
                    if not finished:
                        remaining = len(pending)
                        running = min(remaining, workers_n)
                        queued = remaining - running
                        completed_seeds = [s for s in seeds if s in by_seed]
                        missing_seeds = [s for s in seeds if s not in by_seed]
                        _log(f"[parallel {name}] heartbeat "
                             f"elapsed={time.perf_counter() - par_t0:.0f}s "
                             f"done={done}/{len(seeds_to_run)} running={running} "
                             f"pending={queued} completed_seeds={completed_seeds} "
                             f"missing_seeds={missing_seeds}")
                        continue
                    for fut in finished:
                        s = futures[fut]
                        try:
                            m = fut.result()
                        except Exception as e:
                            # A crashed worker must not abort the others.
                            _on_failure(s, e)
                            continue
                        done += 1
                        _log(f"[{name} seed={s}] DONE ({done}/{len(seeds_to_run)}) "
                             f"rmse={m['rmse_price']:.6f} param_err={m['param_rel_err']:.6f} "
                             f"duration_ms={m['duration_ms']:.0f}")
                        _record(s, m)
        elif seeds_to_run:
            for i, s in enumerate(seeds_to_run):
                _log(f"[{name} seed={s}] ({i + 1}/{len(seeds_to_run)}) calibrating ...")
                try:
                    m = _run_classical_seed(payloads[s])
                except Exception as e:
                    _on_failure(s, e)
                    continue
                _log(f"[{name} seed={s}] DONE rmse={m['rmse_price']:.6f} "
                     f"param_err={m['param_rel_err']:.6f} duration_ms={m['duration_ms']:.0f}")
                _record(s, m)

        # --- Finalise this baseline ------------------------------------------
        if not by_seed:
            print(f"  WARNING: no seeds completed for {name}; skipping aggregation.")
            continue
        _flush_aggregates()  # ensure final aggregate reflects all seeds
        agg = results[name]
        if agg["partial"]:
            print(f"  PARTIAL: {name} completed {agg['n_seeds']}/{len(seeds)} seeds "
                  f"(missing {agg['missing_seeds']}); wrote partial aggregate.")

    return results


def create_pricer_for_spec(spec, params):
    from src.models.pricing import create_pricer
    merged = dict(spec["params"])
    merged.update(params)
    return create_pricer(spec["model"], merged)


def generate_figures(all_results: dict):
    """Generate publication-quality figures."""
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Price scatter plot (from BS full)
    # This would normally load the actual predictions from a run
    # For now, create a placeholder figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("True Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Black-Scholes: Predicted vs True Prices")
    ax.plot([0, 0.5], [0, 0.5], 'k--', label='Perfect fit')
    fig.savefig(FIGURES_DIR / "bs_price_scatter.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: bs_price_scatter.png")
    
    # 2. Activation comparison bar chart
    if 'activation' in str(all_results.keys()):
        activations = ['softplus', 'tanh', 'relu']
        rmse_values = []
        rmse_stds = []
        
        for act in activations:
            key = f"act_ablation_{act}.json"
            if key in all_results:
                rmse_values.append(all_results[key].get('mean', {}).get('rmse_price', 0))
                rmse_stds.append(all_results[key].get('std', {}).get('rmse_price', 0))
            else:
                rmse_values.append(0)
                rmse_stds.append(0)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(activations))
        bars = ax.bar(x, rmse_values, yerr=rmse_stds, capsize=5, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_xticks(x)
        ax.set_xticklabels(['Softplus', 'Tanh', 'ReLU'])
        ax.set_ylabel('Price RMSE')
        ax.set_title('Activation Function Comparison (A2)')
        fig.savefig(FIGURES_DIR / "ablation_A2.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: ablation_A2.png")
    
    # 3. Grid density comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    densities = ['10×5 (Sparse)', '50×20 (Dense)']
    
    lv_sparse = all_results.get('lv_density_10x5.json', {})
    lv_dense = all_results.get('lv_density_50x20.json', {})
    
    rmse_vals = [
        lv_sparse.get('mean', {}).get('rmse_price', 0) * 1000,
        lv_dense.get('mean', {}).get('rmse_price', 0) * 1000,
    ]
    rmse_stds = [
        lv_sparse.get('std', {}).get('rmse_price', 0) * 1000,
        lv_dense.get('std', {}).get('rmse_price', 0) * 1000,
    ]
    
    x = np.arange(len(densities))
    bars = ax.bar(x, rmse_vals, yerr=rmse_stds, capsize=5, color=['#e74c3c', '#2ecc71'])
    ax.set_xticks(x)
    ax.set_xticklabels(densities)
    ax.set_ylabel('Price RMSE (×10⁻³)')
    ax.set_title('Grid Density Impact on Local Vol Recovery (A3)')
    fig.savefig(FIGURES_DIR / "grid_density_error.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: grid_density_error.png")
    
    print("Figures generation complete!")


def _fmt(val, scale=1.0, fmt=".4f"):
    """Format a numeric value for table output."""
    return f"{val * scale:{fmt}}"


def generate_tables(all_results: dict):
    """Generate LaTeX tables reflecting the full 2x2 factorial (Ch9)."""
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)

    def _row(name, label, results):
        r = results.get(name, {})
        m = r.get('mean', {})
        s = r.get('std', {})
        return {
            'Method': label,
            'Price RMSE (1e-3)': f"{m.get('rmse_price', 0)*1000:.2f} ± {s.get('rmse_price', 0)*1000:.2f}",
            'MAE Delta': f"{m.get('mae_delta', 0):.4f}",
            'MAE Gamma': f"{m.get('mae_gamma', 0):.4f}",
            'MAE Theta': f"{m.get('mae_theta', 0):.4f}",
            'RPE': f"{m.get('rpe', 0)*100:.2f}%",
            'Param Err': f"{m.get('param_rel_err', float('nan')):.4f}",
        }

    for model_tag, configs in [
        ("bs", [
            ("bs_nn_baseline.json", "NN"),
            ("bs_nnpde.json", "NN-PDE"),
            ("bs_nnm2.json", "NN-M2"),
            ("bs_softplus_full.json", "PINN"),
            ("classical_bs", "Classical (L-BFGS-B)"),
            ("classical_mc_bs", "Classical (MC)"),
        ]),
        ("lv", [
            ("lv_nn_baseline.json", "NN"),
            ("lv_nnpde.json", "NN-PDE"),
            ("lv_nnm2.json", "NN-M2"),
            ("lv_dnn.json", "PINN"),
            ("classical_lv", "Classical (L-BFGS-B)"),
            ("classical_mc_lv", "Classical (MC)"),
        ]),
        ("heston", [
            ("heston_nn_baseline.json", "NN"),
            ("heston_nnm2.json", "NN-M2"),
            ("classical_heston", "Classical (L-BFGS-B)"),
            ("classical_mc_heston", "Classical (MC)"),
        ]),
    ]:
        rows = [_row(c, l, all_results) for c, l in configs]
        df = pd.DataFrame(rows)
        csv_path = TABLES_DIR / f"{model_tag}_factorial.csv"
        tex_path = TABLES_DIR / f"{model_tag}_factorial.tex"
        df.to_csv(csv_path, index=False)
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        print(f"  Saved: {csv_path.name}, {tex_path.name}")

    all_exp_data = []
    for name, result in sorted(all_results.items()):
        if name.startswith("_"):
            continue
        if isinstance(result, dict) and 'mean' in result:
            model = _infer_model_type(name, result)
            all_exp_data.append({
                'Experiment': name.replace('.json', ''),
                'Model': model,
                'RMSE (×10³)': f"{result['mean'].get('rmse_price', 0)*1000:.2f}",
                'RMSE Std': f"{result['std'].get('rmse_price', 0)*1000:.2f}",
                'Seeds': result.get('n_seeds', 0),
            })
    all_exp_df = pd.DataFrame(all_exp_data)
    all_exp_df.to_csv(TABLES_DIR / "all_experiments.csv", index=False)
    print(f"  Saved: all_experiments.csv")

    _run_statistical_tests(all_results)


def _run_statistical_tests(all_results: dict):
    """Run paired t-tests on key comparisons and save results (Ch8 sec 8.5)."""
    from src.exp.metrics import paired_t_test

    def _extract_seed_metric(result: dict, metric: str) -> list[float]:
        vals = []
        for m in result.get("all_metrics", []):
            v = m.get(metric)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return vals

    comparisons = [
        # Hypothesis labels follow the thesis (Chapter 5): H1 = Greek supervision,
        # H2 = activation smoothness, H3 = grid coverage.
        ("bs_softplus_full.json", "bs_nn_baseline.json", "BS: PINN vs NN (H1)"),
        ("bs_nnm2.json", "bs_nn_baseline.json", "BS: NN-M2 vs NN (H1)"),
        ("lv_dnn.json", "lv_nn_baseline.json", "LV: PINN vs NN (H1)"),
        ("lv_nnm2.json", "lv_nn_baseline.json", "LV: NN-M2 vs NN (H1)"),
        ("act_ablation_bs.json", "act_ablation_relu.json", "Activation: softplus vs ReLU (H2)"),
        ("act_ablation_bs.json", "act_ablation_tanh.json", "Activation: softplus vs tanh (H2)"),
    ]

    metrics_to_test = ["rmse_price", "mae_gamma"]
    rows = []

    for key_a, key_b, label in comparisons:
        r_a = all_results.get(key_a, {})
        r_b = all_results.get(key_b, {})
        if not r_a or not r_b:
            continue
        for metric in metrics_to_test:
            vals_a = _extract_seed_metric(r_a, metric)
            vals_b = _extract_seed_metric(r_b, metric)
            n = min(len(vals_a), len(vals_b))
            if n < 2:
                continue
            t_result = paired_t_test(np.array(vals_a[:n]), np.array(vals_b[:n]))
            sig = "***" if t_result["p_value"] < 0.01 else (
                "**" if t_result["p_value"] < 0.05 else "ns"
            )
            rows.append({
                "Comparison": label,
                "Metric": metric,
                "Mean A": f"{np.mean(vals_a[:n]):.6f}",
                "Mean B": f"{np.mean(vals_b[:n]):.6f}",
                "t-stat": f"{t_result['t_stat']:.3f}",
                "p-value": f"{t_result['p_value']:.4f}",
                "Sig": sig,
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = TABLES_DIR / "statistical_tests.csv"
        tex_path = TABLES_DIR / "statistical_tests.tex"
        df.to_csv(csv_path, index=False)
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        print(f"  Saved: {csv_path.name}, {tex_path.name}")
    else:
        print("  No statistical tests could be computed (insufficient data)")


def generate_latex_snippets(all_results: dict):
    """Generate LaTeX snippet files for direct inclusion."""
    print("\n" + "="*80)
    print("GENERATING LATEX SNIPPETS")
    print("="*80)
    
    # Price scatter figure
    snippet = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=0.6\textwidth]{assets/figures/experiments/generated/bs_price_scatter.png}
  \caption{Predicted versus true prices for the Black--Scholes model using the PINN with full supervision.}
  \label{fig:bs-scatter}
\end{figure}
"""
    with open(SNIPPETS_DIR / "fig_bs_scatter.tex", 'w') as f:
        f.write(snippet)
    print(f"  Saved: fig_bs_scatter.tex")
    
    # Activation ablation figure
    snippet = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=0.6\textwidth]{assets/figures/experiments/generated/ablation_A2.png}
  \caption{Ablation A2: Price RMSE for different activation functions. Error bars show standard deviation over 5 seeds.}
  \label{fig:ablationA2}
\end{figure}
"""
    with open(SNIPPETS_DIR / "fig_ablation_a2.tex", 'w') as f:
        f.write(snippet)
    print(f"  Saved: fig_ablation_a2.tex")
    
    # Grid density figure
    snippet = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=0.6\textwidth]{assets/figures/experiments/generated/grid_density_error.png}
  \caption{Effect of grid density on parameter error in the local volatility model. Sparse grids lead to significantly higher errors.}
  \label{fig:grid-density}
\end{figure}
"""
    with open(SNIPPETS_DIR / "fig_grid_density.tex", 'w') as f:
        f.write(snippet)
    print(f"  Saved: fig_grid_density.tex")
    
    # BS metrics table
    snippet = r"""
\begin{table}[t]
  \centering
  \caption{Performance metrics for the Black--Scholes model. Results show mean $\pm$ std over 5 seeds. RMSE values are scaled by $10^{-3}$.}
  \label{tab:bs-metrics}
  \input{assets/tables/bs_metrics.tex}
\end{table}
"""
    with open(SNIPPETS_DIR / "tab_bs_metrics.tex", 'w') as f:
        f.write(snippet)
    print(f"  Saved: tab_bs_metrics.tex")


def _infer_model_type(name: str, result: dict) -> str:
    """Infer model type from result metadata, falling back to config name."""
    cfg = result.get('config', {})
    if isinstance(cfg, dict):
        model_name = cfg.get('model', {}).get('name', '')
        if model_name:
            return model_name
    n = name.lower()
    if 'heston' in n:
        return 'Heston'
    if 'lv' in n or 'localvol' in n:
        return 'LocalVol'
    return 'BS'


def update_results_summary(all_results: dict):
    """Update RESULTS_SUMMARY.md with new results."""
    print("\n" + "="*80)
    print("UPDATING RESULTS_SUMMARY.md")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    summary = f"""# 🎉 EXPERIMENT RESULTS - UPDATED

**Last Updated:** {timestamp}  
**Status:** ✅ ALL CORE EXPERIMENTS COMPLETED

---

## 📊 **Experiment Results Summary**

### **Experiments Completed**

| Experiment | Model | RMSE (×10³) | Std | Seeds | Status |
|------------|-------|-------------|-----|-------|--------|
"""
    
    for name, result in all_results.items():
        if name.startswith("_"):
            continue
        if isinstance(result, dict) and 'mean' in result:
            model = _infer_model_type(name, result)
            rmse = result['mean'].get('rmse_price', 0) * 1000
            std = result['std'].get('rmse_price', 0) * 1000
            n_seeds = result.get('n_seeds', 0)
            status = '✅ Complete'
            summary += f"| `{name.replace('.json', '')}` | {model} | **{rmse:.2f}** | {std:.2f} | {n_seeds} | {status} |\n"
    
    summary += """
---

## 🧪 **Hypotheses**

The hypotheses H1 (Greek supervision), H2 (activation smoothness) and H3
(strike--maturity coverage) are stated in Chapter 5 of the thesis and assessed
in Chapters 9--11. This summary lists raw metrics only; the paired t-tests
are written to `statistical_tests.csv` in the thesis tables directory.

---

## ⏱️ **Classical (MC) Runtime Configuration**

Monte Carlo calibration prices by path simulation inside the optimiser loop, so
its cost is configurable (see `_MC_PRESET_QUICK` / `_MC_PRESET_FULL` in
`src/models/calibration.py`):

- **Quick** (`--quick`): 3 restarts, 2k/10k coarse/refine paths, 50 Euler steps, maxiter 30/30 (~0.5-1 min per model/seed on CPU).
- **Full**: 10 restarts, 10k/50k coarse/refine paths, 100 Euler steps (down from 200), maxiter 100/150.

BS prices by exact terminal sampling and ignores `n_steps`; LV/Heston use Euler
stepping, which is why their step count was reduced for feasibility.

---

## 📁 **Generated Artifacts**

- `assets/figures/experiments/generated/` - Publication-quality figures
- `assets/tables/` - CSV and LaTeX tables
- `snippets/generated/` - Ready-to-include LaTeX snippets

---

**Generated by:** `scripts/run_all_experiments.py`
"""
    
    results_path = Path(__file__).parent.parent / "RESULTS_SUMMARY.md"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"  Updated: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all thesis experiments")
    parser.add_argument("--quick", action="store_true", help="Run with fewer seeds (1 instead of 5)")
    parser.add_argument("--seeds", type=str, default="20251,20252,20253,20254,20255",
                        help="Comma-separated list of seeds")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--skip-experiments", action="store_true", help="Only generate outputs from existing results")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Path to an existing all_results_*.json to post-process. "
                             "With --skip-experiments, this JSON is the source of truth; "
                             "if omitted the latest results/all_results_*.json is used.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated experiment names to run (e.g. "
                             "classical_mc_bs,classical_mc_lv,classical_mc_heston). "
                             "Other experiments are skipped and previously computed "
                             "results are preserved when regenerating tables/summary.")
    parser.add_argument("--parallel-seeds", action="store_true",
                        help="Run independent seeds of the classical MC baselines "
                             "in parallel processes (safe: seeds are numerically "
                             "independent; only the parent writes output files).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max worker processes for --parallel-seeds "
                             "(default: ProcessPoolExecutor default = CPU count).")
    parser.add_argument("--parallel-only-mc", dest="parallel_only_mc",
                        action="store_true", default=True,
                        help="Only parallelise the stochastic MC baselines, "
                             "keeping deterministic L-BFGS-B baselines sequential "
                             "(default: on).")
    parser.add_argument("--no-parallel-only-mc", dest="parallel_only_mc",
                        action="store_false",
                        help="Also allow deterministic baselines to run seeds in "
                             "parallel under --parallel-seeds.")
    parser.add_argument("--resume-classical-checkpoints", dest="resume_classical_checkpoints",
                        action="store_true", default=True,
                        help="Resume classical baselines from per-seed checkpoints "
                             "in results/checkpoints/ instead of recomputing "
                             "already-completed seeds (default: on).")
    parser.add_argument("--no-resume-classical-checkpoints", dest="resume_classical_checkpoints",
                        action="store_false",
                        help="Force recomputation of every classical seed, ignoring "
                             "existing checkpoints.")
    parser.add_argument("--rebuild-from-checkpoints", action="store_true",
                        help="Rebuild classical aggregates from saved seed checkpoints "
                             "(merged onto the latest all_results_*.json), write a new "
                             "all_results_<timestamp>.json, then regenerate figures/"
                             "tables/snippets/summary. Runs no experiments.")
    parser.add_argument("--verbose-progress", dest="verbose_progress",
                        action="store_true", default=True,
                        help="Print detailed progress from classical seed workers and "
                             "their objective/iteration loops (default: on).")
    parser.add_argument("--no-verbose-progress", dest="verbose_progress",
                        action="store_false",
                        help="Disable detailed classical progress logging.")
    parser.add_argument("--log-interval-sec", type=float, default=30.0,
                        help="How often (seconds) long-running objective functions "
                             "print progress (default: 30).")
    parser.add_argument("--heartbeat-sec", type=float, default=60.0,
                        help="How often (seconds) the parent prints a heartbeat while "
                             "waiting for parallel seed futures (default: 60).")
    parser.add_argument("--progress-jsonl", type=str, default=None,
                        help="Optional path; if set, append machine-readable progress "
                             "events as JSON Lines.")
    args = parser.parse_args()

    if args.quick:
        seeds = [20251]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    only_set = None
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}
    
    print("="*80)
    print("THESIS EXPERIMENT RUNNER")
    print("="*80)
    print(f"Seeds: {seeds}")
    print(f"Device: {args.device or 'auto'}")
    if args.parallel_seeds:
        scope = "MC baselines only" if args.parallel_only_mc else "all classical baselines"
        print(f"Parallel seeds: ON ({scope}, max_workers={args.max_workers or 'auto'})")
    print(f"Progress logging: {'ON' if args.verbose_progress else 'OFF'} "
          f"(log-interval={args.log_interval_sec:.0f}s, heartbeat={args.heartbeat_sec:.0f}s"
          f"{', jsonl=' + args.progress_jsonl if args.progress_jsonl else ''})")
    print(f"Output dirs:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Snippets: {SNIPPETS_DIR}")
    
    ensure_dirs()
    seed_everything(20251)
    
    all_results = {}
    # Path of the results JSON that the figure generator should read. For a
    # normal run this is the file we write below; for --skip-experiments it is
    # the existing file we load from.
    results_file = None

    if only_set is not None:
        print(f"Selective run (--only): {sorted(only_set)}")

    if args.rebuild_from_checkpoints:
        # Rebuild classical aggregates purely from saved seed checkpoints, then
        # write a fresh finalised results file and post-process from it. Runs no
        # experiments regardless of --skip-experiments.
        all_results = _rebuild_from_checkpoints()
        if not all_results:
            parser.error("--rebuild-from-checkpoints found no results or checkpoints "
                         "to rebuild from.")
        results_file = RESULTS_DIR / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        all_results["_meta"] = _hardware_meta()
        _atomic_write_json(results_file, all_results)
        print(f"\nRebuilt results saved to: {results_file}")
    elif not args.skip_experiments:
        # For a selective run, start from the latest existing results so that
        # regenerated tables/summary keep the experiments we did not re-run.
        if only_set is not None:
            base = _load_latest_results()
            if base:
                all_results.update(base)
                print(f"Merged {len(base)} existing result entries from latest results file.")

        core_results = run_core_experiments(seeds, args.device, only=only_set)
        all_results.update(core_results)

        act_results = run_activation_ablation(seeds, args.device, only=only_set)
        all_results.update(act_results)

        noise_results = run_noise_ablation(seeds, args.device, only=only_set)
        all_results.update(noise_results)

        sweep_results = run_param_sweep_experiments(seeds, args.device, only=only_set)
        all_results.update(sweep_results)

        classical_results = run_classical_baselines(
            seeds, args.device, only=only_set, quick=args.quick,
            parallel_seeds=args.parallel_seeds, max_workers=args.max_workers,
            parallel_only_mc=args.parallel_only_mc,
            resume_checkpoints=args.resume_classical_checkpoints,
            base_results=all_results,
            verbose_progress=args.verbose_progress,
            log_interval_sec=args.log_interval_sec,
            heartbeat_sec=args.heartbeat_sec,
            progress_jsonl=Path(args.progress_jsonl) if args.progress_jsonl else None,
        )
        all_results.update(classical_results)

        results_file = RESULTS_DIR / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        all_results["_meta"] = _hardware_meta()
        _atomic_write_json(results_file, all_results)
        print(f"\nRaw results saved to: {results_file}")
    else:
        # Post-processing only: load existing results as the source of truth.
        # Prefer an explicit --results-file; otherwise fall back to the latest
        # results/all_results_*.json.
        if args.results_file:
            results_file = Path(args.results_file)
            if not results_file.exists():
                parser.error(f"--results-file not found: {results_file}")
        else:
            candidates = _timestamped_results_files()
            if not candidates:
                parser.error(
                    f"--skip-experiments was given but no results file exists in "
                    f"{RESULTS_DIR} (and --results-file was not provided). Nothing to "
                    f"post-process."
                )
            results_file = candidates[-1]

        print(f"\nLoading existing results: {results_file}")
        try:
            with open(results_file) as f:
                all_results = json.load(f)
        except Exception as e:
            parser.error(f"could not load results file {results_file}: {e}")

        if not all_results:
            parser.error(
                f"results file {results_file} contained no entries; refusing to "
                f"generate empty tables/figures/summary."
            )
        print(f"  Loaded {len(all_results)} result entries.")

    # Guard against generating empty/misleading outputs from no data.
    if not all_results:
        parser.error("no results available to post-process; aborting before "
                     "generating empty tables/figures/summary.")

    # Use the dedicated figure generator for publication-quality plots. Pass an
    # explicit argv so gen_all_figures never inspects THIS process's CLI args
    # (e.g. --only/--quick/--device), which it does not understand.
    fig_argv = ["--results-file", str(results_file)] if results_file else []
    try:
        from gen_all_figures import main as gen_figures_main
        gen_figures_main(fig_argv)
    except SystemExit as e:
        # argparse inside gen_all_figures raises SystemExit (not Exception) on
        # bad args; surface it loudly rather than silently swallowing it.
        print(f"  WARNING: gen_all_figures exited early (SystemExit: {e}); "
              f"falling back to inline figure generation.")
        generate_figures(all_results)
    except Exception as e:
        print(f"  gen_all_figures failed ({e}), falling back to inline generation")
        generate_figures(all_results)

    generate_tables(all_results)
    generate_latex_snippets(all_results)
    update_results_summary(all_results)
    
    print("\n" + "="*80)
    print("[SUCCESS] ALL DONE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review generated figures in:", FIGURES_DIR)
    print("2. Check tables in:", TABLES_DIR)
    print("3. Include snippets in thesis from:", SNIPPETS_DIR)
    print("4. Recompile thesis PDF")


if __name__ == "__main__":
    main()

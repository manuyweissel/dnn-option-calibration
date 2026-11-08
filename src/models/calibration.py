"""
Classical calibration baselines (Ch8 sec 8.4).

Provides L-BFGS-B deterministic-pricer calibration and Monte Carlo
calibration for BS, LocalVol, and Heston models.
"""

import logging
import os
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.models.pricing import create_pricer
from src.models.mc_pricing import mc_price

logger = logging.getLogger(__name__)


class ClassicalCalibrator:
    """
    Calibrate model parameters by minimising price MSE via L-BFGS-B
    with multiple random restarts (Ch8 sec 8.4).
    """

    def __init__(
        self,
        model_name: str,
        fixed_params: dict[str, float],
        free_param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        n_restarts: int = 20,
        seed: int = 42,
        verbose: bool = False,
        log_interval_sec: float = 30.0,
        progress_prefix: str | None = None,
    ):
        self.model_name = model_name
        self.fixed_params = dict(fixed_params)
        self.free_param_names = list(free_param_names)
        self.bounds = bounds
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        # Progress logging: print an objective/iteration progress line at most
        # every ``log_interval_sec`` so long L-BFGS-B runs never look frozen.
        self.verbose = verbose
        self.log_interval_sec = log_interval_sec
        self.progress_prefix = progress_prefix
        # Live progress state (reset in ``calibrate``).
        self._t0 = 0.0
        self._last_log_t = 0.0
        self._last_cb_log_t = 0.0
        self._n_evals = 0
        self._best_loss = np.inf
        self._restart_idx = 0

    def _prefix(self) -> str:
        return self.progress_prefix or f"[{self.model_name} pid={os.getpid()}]"

    def _build_pricer(self, free_vals: np.ndarray):
        params = dict(self.fixed_params)
        for name, val in zip(self.free_param_names, free_vals):
            params[name] = float(val)
        return create_pricer(self.model_name, params)

    def _objective(self, free_vals: np.ndarray, K: np.ndarray, T: np.ndarray, prices: np.ndarray) -> float:
        try:
            pricer = self._build_pricer(free_vals)
            pred = pricer.price(K, T)
            loss = float(np.mean((pred - prices) ** 2))
        except Exception:
            loss = 1e12
        self._n_evals += 1
        if loss < self._best_loss:
            self._best_loss = loss
        # Interval-gated progress: log only every ``log_interval_sec``, not per eval.
        if self.verbose:
            now = time.perf_counter()
            if now - self._last_log_t >= self.log_interval_sec:
                self._last_log_t = now
                params_str = ", ".join(
                    f"{n}={float(v):.4g}" for n, v in zip(self.free_param_names, free_vals)
                )
                print(f"{self._prefix()} eval={self._n_evals} "
                      f"elapsed={now - self._t0:.1f}s loss={loss:.2e} "
                      f"best={self._best_loss:.2e} params={{{params_str}}}", flush=True)
        return loss

    def _make_callback(self):
        """L-BFGS-B iteration callback: log optimiser iterations on interval."""
        if not self.verbose:
            return None
        state = {"iter": 0}

        def _cb(xk):
            state["iter"] += 1
            now = time.perf_counter()
            # Independent timer from the objective logger so optimiser-iteration
            # lines still appear (the objective fires far more often and would
            # otherwise always win a shared interval gate).
            if now - self._last_cb_log_t >= self.log_interval_sec:
                self._last_cb_log_t = now
                print(f"{self._prefix()} iter={state['iter']} "
                      f"restart={self._restart_idx}/{self.n_restarts} "
                      f"elapsed={now - self._t0:.1f}s best_loss={self._best_loss:.2e}",
                      flush=True)

        return _cb

    def calibrate(
        self,
        K: np.ndarray,
        T: np.ndarray,
        prices: np.ndarray,
    ) -> dict[str, Any]:
        """
        Run L-BFGS-B with ``n_restarts`` random initial points.

        Returns dict with ``params`` (best-fit), ``loss``, and ``duration_ms``.
        """
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        prices = np.asarray(prices).flatten()

        lb = np.array([self.bounds[n][0] for n in self.free_param_names])
        ub = np.array([self.bounds[n][1] for n in self.free_param_names])
        scipy_bounds = list(zip(lb, ub))

        best_loss = np.inf
        best_x = None
        t0 = time.perf_counter()
        # Reset live progress state for this run.
        self._t0 = t0
        self._last_log_t = t0
        self._last_cb_log_t = t0
        self._n_evals = 0
        self._best_loss = np.inf
        if self.verbose:
            print(f"{self._prefix()} L-BFGS-B calibration START: "
                  f"{self.n_restarts} restarts, free={self.free_param_names}", flush=True)

        for restart in range(self.n_restarts):
            self._restart_idx = restart + 1
            x0 = self.rng.uniform(lb, ub)
            res = minimize(
                self._objective,
                x0,
                args=(K, T, prices),
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": 500, "ftol": 1e-12},
                callback=self._make_callback(),
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x

        duration_ms = (time.perf_counter() - t0) * 1000
        if self.verbose:
            print(f"{self._prefix()} L-BFGS-B calibration DONE: "
                  f"loss={best_loss:.3e} evals={self._n_evals} "
                  f"elapsed={duration_ms / 1000:.1f}s", flush=True)

        params_pred = dict(self.fixed_params)
        for name, val in zip(self.free_param_names, best_x):
            params_pred[name] = float(val)

        return {
            "params": params_pred,
            "loss": float(best_loss),
            "duration_ms": duration_ms,
        }


def calibrate_bs(K, T, prices, S0=1.0, r=0.02, seed=42,
                 verbose=False, log_interval_sec=30.0, progress_prefix=None):
    """Convenience: calibrate BS sigma."""
    cal = ClassicalCalibrator(
        model_name="BS",
        fixed_params={"S0": S0, "r": r},
        free_param_names=["sigma"],
        bounds={"sigma": (0.01, 2.0)},
        seed=seed,
        verbose=verbose,
        log_interval_sec=log_interval_sec,
        progress_prefix=progress_prefix,
    )
    return cal.calibrate(K, T, prices)


def calibrate_localvol(K, T, prices, S0=1.0, r=0.02, seed=42,
                       verbose=False, log_interval_sec=30.0, progress_prefix=None):
    """Convenience: calibrate LV (sigma0, alpha)."""
    cal = ClassicalCalibrator(
        model_name="LocalVol",
        fixed_params={"S0": S0, "r": r, "beta": 0.0},
        free_param_names=["sigma0", "alpha"],
        bounds={"sigma0": (0.01, 2.0), "alpha": (-5.0, 5.0)},
        seed=seed,
        verbose=verbose,
        log_interval_sec=log_interval_sec,
        progress_prefix=progress_prefix,
    )
    return cal.calibrate(K, T, prices)


def calibrate_heston(K, T, prices, S0=1.0, r=0.02, seed=42,
                     verbose=False, log_interval_sec=30.0, progress_prefix=None):
    """Convenience: calibrate Heston (v0, kappa, theta, xi, rho)."""
    cal = ClassicalCalibrator(
        model_name="Heston",
        fixed_params={"S0": S0, "r": r},
        free_param_names=["v0", "kappa", "theta", "xi", "rho"],
        bounds={
            "v0": (0.001, 1.0),
            "kappa": (0.1, 10.0),
            "theta": (0.001, 1.0),
            "xi": (0.01, 3.0),
            "rho": (-0.99, 0.99),
        },
        seed=seed,
        verbose=verbose,
        log_interval_sec=log_interval_sec,
        progress_prefix=progress_prefix,
    )
    return cal.calibrate(K, T, prices)


# ---------------------------------------------------------------------------
# Monte Carlo calibration baselines (Ch8 sec 8.4.2)
# ---------------------------------------------------------------------------

class MCCalibrator:
    """
    Two-phase MC calibration baseline (Ch8 sec 8.4.2).

    Phase 1 (coarse): L-BFGS-B with ``n_paths_coarse`` MC paths and
    ``n_restarts`` random starting points.
    Phase 2 (refine): Re-optimise from the coarse optimum with
    ``n_paths_refine`` paths and a fresh seed block for unbiased evaluation.
    """

    def __init__(
        self,
        model_name: str,
        fixed_params: dict[str, float],
        free_param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        n_restarts: int = 20,
        n_paths_coarse: int = 10_000,
        n_paths_refine: int = 100_000,
        n_steps: int = 200,
        maxiter_coarse: int = 200,
        maxiter_refine: int = 500,
        seed: int = 42,
        verbose: bool = False,
        log_interval_sec: float = 30.0,
        progress_prefix: str | None = None,
    ):
        self.model_name = model_name
        self.fixed_params = dict(fixed_params)
        self.free_param_names = list(free_param_names)
        self.bounds = bounds
        self.n_restarts = n_restarts
        self.n_paths_coarse = n_paths_coarse
        self.n_paths_refine = n_paths_refine
        self.n_steps = n_steps
        self.maxiter_coarse = maxiter_coarse
        self.maxiter_refine = maxiter_refine
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.verbose = verbose
        self.log_interval_sec = log_interval_sec
        self.progress_prefix = progress_prefix
        # Live progress state (set in ``calibrate``).
        self._t0 = 0.0
        self._last_log_t = 0.0
        self._n_evals = 0
        self._best_loss = np.inf
        self._phase = "coarse"

    def _prefix(self) -> str:
        return self.progress_prefix or f"[{self.model_name} MC pid={os.getpid()}]"

    def _mc_objective(self, free_vals, K, T, prices, n_paths, mc_seed):
        try:
            params = dict(self.fixed_params)
            for name, val in zip(self.free_param_names, free_vals):
                params[name] = float(val)
            # Genuine Monte Carlo pricing by path simulation of the model SDE.
            # ``mc_seed`` is held constant within a phase (common random numbers),
            # so the objective is a smooth function of the parameters; ``n_paths``
            # distinguishes the coarse and refine phases.
            pred = mc_price(
                self.model_name, params, K, T,
                n_paths=n_paths, n_steps=self.n_steps,
                seed=mc_seed, antithetic=True,
            )
            loss = float(np.mean((pred - prices) ** 2))
        except Exception:
            loss = 1e12
        self._n_evals += 1
        if loss < self._best_loss:
            self._best_loss = loss
        # Interval-gated progress so the long refine phase never looks frozen.
        if self.verbose:
            now = time.perf_counter()
            if now - self._last_log_t >= self.log_interval_sec:
                self._last_log_t = now
                print(f"{self._prefix()} {self._phase} progress eval={self._n_evals} "
                      f"elapsed={now - self._t0:.1f}s loss={loss:.3e} "
                      f"best={self._best_loss:.3e}", flush=True)
        return loss

    def calibrate(self, K, T, prices):
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        prices = np.asarray(prices).flatten()

        lb = np.array([self.bounds[n][0] for n in self.free_param_names])
        ub = np.array([self.bounds[n][1] for n in self.free_param_names])
        scipy_bounds = list(zip(lb, ub))

        t0 = time.perf_counter()
        # Reset live progress state for this run.
        self._t0 = t0
        self._last_log_t = t0
        self._n_evals = 0
        self._best_loss = np.inf

        # Phase 1: coarse search
        best_loss = np.inf
        best_x = None
        mc_seed_coarse = self.seed
        self._phase = "coarse"
        if self.verbose:
            print(f"{self._prefix()} coarse START restarts={self.n_restarts} "
                  f"paths={self.n_paths_coarse} steps={self.n_steps} "
                  f"maxiter={self.maxiter_coarse}", flush=True)
        for i in range(self.n_restarts):
            x0 = self.rng.uniform(lb, ub)
            res = minimize(
                self._mc_objective,
                x0,
                args=(K, T, prices, self.n_paths_coarse, mc_seed_coarse),
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": self.maxiter_coarse, "ftol": 1e-10},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
            if self.verbose:
                print(f"{self._prefix()} coarse restart {i + 1}/{self.n_restarts} done "
                      f"elapsed={time.perf_counter() - t0:.1f}s "
                      f"loss={res.fun:.3e} best={best_loss:.3e}", flush=True)

        # Phase 2: refine from coarse optimum with more paths + fresh seed
        mc_seed_refine = self.seed + 9999
        self._phase = "refine"
        if self.verbose:
            print(f"{self._prefix()} refine START paths={self.n_paths_refine} "
                  f"maxiter={self.maxiter_refine}", flush=True)
        res_fine = minimize(
            self._mc_objective,
            best_x,
            args=(K, T, prices, self.n_paths_refine, mc_seed_refine),
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": self.maxiter_refine, "ftol": 1e-12},
        )
        if res_fine.fun < best_loss:
            best_loss = res_fine.fun
            best_x = res_fine.x

        duration_ms = (time.perf_counter() - t0) * 1000
        if self.verbose:
            print(f"{self._prefix()} refine DONE elapsed={duration_ms / 1000:.1f}s "
                  f"loss={best_loss:.3e}", flush=True)

        params_pred = dict(self.fixed_params)
        for name, val in zip(self.free_param_names, best_x):
            params_pred[name] = float(val)

        return {
            "params": params_pred,
            "loss": float(best_loss),
            "duration_ms": duration_ms,
        }


# Runtime presets for the MC calibration baselines (Ch8 sec 8.4.2).
#
# Monte Carlo pricing inside the optimiser loop is expensive -- especially for
# LocalVol/Heston, which require Euler time-stepping -- so we expose a "quick"
# preset for ``--quick`` smoke runs and a heavier "full" preset for reporting.
# Full mode deliberately uses n_steps=100 (rather than 200) and n_restarts=10
# for LV/Heston to stay feasible; BS prices by exact terminal sampling and
# ignores n_steps. See the runtime note in RESULTS_SUMMARY.md.
_MC_PRESET_QUICK = dict(
    n_restarts=3,
    n_paths_coarse=2_000,
    n_paths_refine=10_000,
    n_steps=50,
    maxiter_coarse=30,
    maxiter_refine=30,
)
_MC_PRESET_FULL = dict(
    n_restarts=10,
    n_paths_coarse=10_000,
    n_paths_refine=50_000,
    n_steps=100,
    maxiter_coarse=100,
    maxiter_refine=150,
)


def _mc_preset(quick: bool) -> dict:
    """Return a copy of the MC runtime preset for quick or full mode."""
    return dict(_MC_PRESET_QUICK if quick else _MC_PRESET_FULL)


def calibrate_mc_bs(K, T, prices, S0=1.0, r=0.02, seed=42, quick=False, verbose=False,
                    log_interval_sec=30.0, progress_prefix=None):
    """MC calibration baseline for BS (Ch8 sec 8.4.2).

    ``quick=True`` selects the lightweight runtime preset for smoke runs.
    """
    cal = MCCalibrator(
        model_name="BS",
        fixed_params={"S0": S0, "r": r},
        free_param_names=["sigma"],
        bounds={"sigma": (0.01, 2.0)},
        seed=seed,
        verbose=verbose,
        log_interval_sec=log_interval_sec,
        progress_prefix=progress_prefix,
        **_mc_preset(quick),
    )
    return cal.calibrate(K, T, prices)


def calibrate_mc_localvol(K, T, prices, S0=1.0, r=0.02, seed=42, quick=False, verbose=False,
                          log_interval_sec=30.0, progress_prefix=None):
    """MC calibration baseline for LocalVol (Ch8 sec 8.4.2).

    ``quick=True`` selects the lightweight runtime preset for smoke runs.
    """
    cal = MCCalibrator(
        model_name="LocalVol",
        fixed_params={"S0": S0, "r": r, "beta": 0.0},
        free_param_names=["sigma0", "alpha"],
        bounds={"sigma0": (0.01, 2.0), "alpha": (-5.0, 5.0)},
        seed=seed,
        verbose=verbose,
        log_interval_sec=log_interval_sec,
        progress_prefix=progress_prefix,
        **_mc_preset(quick),
    )
    return cal.calibrate(K, T, prices)


def calibrate_mc_heston(K, T, prices, S0=1.0, r=0.02, seed=42, quick=False, verbose=False,
                        log_interval_sec=30.0, progress_prefix=None):
    """MC calibration baseline for Heston (Ch8 sec 8.4.2).

    Mirrors the deterministic :func:`calibrate_heston` baseline (same free
    parameters and bounds) but prices by Monte Carlo simulation via
    :class:`MCCalibrator`. ``quick=True`` selects the lightweight runtime
    preset for smoke runs.
    """
    cal = MCCalibrator(
        model_name="Heston",
        fixed_params={"S0": S0, "r": r},
        free_param_names=["v0", "kappa", "theta", "xi", "rho"],
        bounds={
            "v0": (0.001, 1.0),
            "kappa": (0.1, 10.0),
            "theta": (0.001, 1.0),
            "xi": (0.01, 3.0),
            "rho": (-0.99, 0.99),
        },
        seed=seed,
        verbose=verbose,
        log_interval_sec=log_interval_sec,
        progress_prefix=progress_prefix,
        **_mc_preset(quick),
    )
    return cal.calibrate(K, T, prices)

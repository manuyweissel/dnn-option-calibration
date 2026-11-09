"""Paired two-sided t-tests quoted in chapters 9-11 of the thesis, with Holm over all of them.

Usage (from the repository root; thesis_python_experiments/ inside the thesis repository):
    python scripts/paired_tests.py [--results-file results/all_results_20261108_213044.json]

Runs are paired by seed (``all_metrics[*].seed``); n = 5 per experiment. "better" counts the seeds in
which the second experiment has the smaller value. The Holm correction runs over every test in TESTS
(the family of all paired comparisons evaluated for chapters 9-11), so it does not depend on which
tests the text happens to quote. "*" marks the 14 primary tests (Greek supervision, activation,
Heston, A3 and the LV parameter contrast); they are a subset of the family, not a separate one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT = Path(__file__).resolve().parent.parent / "results" / "all_results_20261108_213044.json"

BS0, BSM, BSP, BSI = "bs_nn_baseline.json", "bs_nnm2.json", "bs_nnpde.json", "bs_softplus_full.json"
RELU, SOFT, TANH = "act_ablation_relu.json", "act_ablation_bs.json", "act_ablation_tanh.json"
LV0, LVM, LVP, LVI = "lv_nn_baseline.json", "lv_nnm2.json", "lv_nnpde.json", "lv_dnn.json"
HE0, HEM = "heston_nn_baseline.json", "heston_nnm2.json"
DENSE, SPARSE = "lv_density_50x20.json", "lv_density_10x5.json"

# (label, first experiment, second experiment, metric, primary test)
TESTS = [
    ("BS A1   Gamma   NN -> NN-M2", BS0, BSM, "mae_gamma", True),
    ("BS A1   Gamma   NN-PDE -> PINN", BSP, BSI, "mae_gamma", True),
    ("BS A1   price   NN -> NN-M2", BS0, BSM, "rmse_price", True),
    ("BS A1   price   NN-PDE -> PINN", BSP, BSI, "rmse_price", True),
    ("BS A2   Gamma   ReLU -> softplus", RELU, SOFT, "mae_gamma", True),
    ("BS A2   price   ReLU -> softplus", RELU, SOFT, "rmse_price", True),
    ("BS A2   price   ReLU -> tanh", RELU, TANH, "rmse_price", True),
    ("LV      e_psi   NN -> PINN", LV0, LVI, "param_rel_err", True),
    ("Heston  Gamma   NN -> NN-M2", HE0, HEM, "mae_gamma", True),
    ("Heston  Delta   NN -> NN-M2", HE0, HEM, "mae_delta", True),
    ("Heston  price   NN -> NN-M2", HE0, HEM, "rmse_price", True),
    ("A3      e_psi   50x20 -> 10x5", DENSE, SPARSE, "param_rel_err", True),
    ("A3      Delta   50x20 -> 10x5", DENSE, SPARSE, "mae_delta", True),
    ("A3      extrap  50x20 -> 10x5", DENSE, SPARSE, "rmse_extrap", True),
    ("BS      Gamma   NN -> NN-PDE", BS0, BSP, "mae_gamma", False),
    ("BS      Theta   NN -> NN-PDE", BS0, BSP, "mae_theta", False),
    ("BS      sigma   NN -> NN-PDE", BS0, BSP, "param_rel_err", False),
    ("BS      Gamma   NN-M2 -> PINN", BSM, BSI, "mae_gamma", False),
    ("BS      extrap  NN -> PINN", BS0, BSI, "rmse_extrap", False),
    ("BS A2   Theta   ReLU -> softplus", RELU, SOFT, "mae_theta", False),
    ("BS A2   Theta   ReLU -> tanh", RELU, TANH, "mae_theta", False),
    ("LV      e_psi   NN -> NN-M2", LV0, LVM, "param_rel_err", False),
    ("LV      e_psi   NN-PDE -> PINN", LVP, LVI, "param_rel_err", False),
    ("LV      e_psi   NN-M2 -> PINN", LVM, LVI, "param_rel_err", False),
    ("LV      Gamma   NN -> NN-M2", LV0, LVM, "mae_gamma", False),
    ("LV      Gamma   NN-PDE -> PINN", LVP, LVI, "mae_gamma", False),
    ("LV      Gamma   NN-M2 -> PINN", LVM, LVI, "mae_gamma", False),
    ("LV      price   NN -> NN-M2", LV0, LVM, "rmse_price", False),
    ("LV      extrap  NN -> PINN", LV0, LVI, "rmse_extrap", False),
    ("Heston  Theta   NN -> NN-M2", HE0, HEM, "mae_theta", False),
    ("Heston  extrap  NN -> NN-M2", HE0, HEM, "rmse_extrap", False),
    ("A3      price   50x20 -> 10x5", DENSE, SPARSE, "rmse_price", False),
    ("A3      Gamma   50x20 -> 10x5", DENSE, SPARSE, "mae_gamma", False),
]


def paired(results, a, b, metric):
    """Means, paired two-sided t-test p-value and #seeds where b < a."""
    va = {m["seed"]: m[metric] for m in results[a]["all_metrics"]}
    vb = {m["seed"]: m[metric] for m in results[b]["all_metrics"]}
    seeds = sorted(set(va) & set(vb))
    x, y = np.array([va[s] for s in seeds]), np.array([vb[s] for s in seeds])
    return x.mean(), y.mean(), stats.ttest_rel(x, y).pvalue, int(np.sum(y < x)), len(seeds)


def holm(pvalues):
    """Holm step-down adjusted p-values (monotone, capped at 1), in input order."""
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-file", type=Path, default=DEFAULT)
    results = json.load(open(ap.parse_args().results_file))

    rows = [(label, core, *paired(results, a, b, metric)) for label, a, b, metric, core in TESTS]
    adjusted = holm(np.array([r[4] for r in rows]))
    # Self-check against the thesis (canonical Black-Scholes value, Tab. 9.6 caption "p < 0.001").
    assert format(rows[0][4], ".1e") == "1.1e-05", rows[0]

    print(f"{'test':36s} {'mean 1':>10s} {'mean 2':>10s} {'change':>8s} {'p':>9s} {'p Holm':>8s} better")
    for (label, core, m1, m2, p, better, n), p_holm in zip(rows, adjusted):
        print(f"{label + (' *' if core else ''):36s} {m1:10.4g} {m2:10.4g} {100 * (m2 / m1 - 1):+7.1f}% {p:9.2g} {p_holm:8.3g} {better}/{n}")
    print(f"\nHolm over all {len(rows)} tests: {int(np.sum(adjusted < 0.05))} significant at 5 % "
          f"(uncorrected: {sum(r[4] < 0.05 for r in rows)}); * = primary tests")


if __name__ == "__main__":
    main()

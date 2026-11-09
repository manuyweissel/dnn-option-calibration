# Differential neural networks for option calibration: experiment code

Code and results accompanying the Master's thesis

> Manú Weissel, *Learning the Dynamics Behind Prices: Differential Neural Networks for Option Calibration*,
> Master's thesis, Department of Mathematics, Goethe-Universität Frankfurt, 2026.

The code trains multilayer perceptrons `u_θ(K, T)` on synthetic European call-price surfaces generated
under three models (Black–Scholes, power-law local volatility, Heston). It compares four neural training
variants with two classical calibration baselines and writes every table and figure of the thesis's
results chapter.

| Variant | Training signal |
|---|---|
| NN | prices |
| NN-M2 | prices + strike Greeks |
| NN-PDE | prices + PDE residual (Black–Scholes, local volatility) |
| PINN | prices + strike Greeks + PDE residual (Black–Scholes, local volatility) |
| Classical (det.) | multi-start L-BFGS-B with the model's pricer (closed form, Dupire Crank–Nicolson solver, Fourier inversion) |
| Classical (MC) | multi-start L-BFGS-B with Monte Carlo pricing under common random numbers |

The PDE residual of NN-PDE and PINN uses the volatility of the generating model (privileged
information), so these variants are surface-learning benchmarks. They are not run for Heston.

## Installation

Python 3.12 and PyTorch 2.x. The thesis results were produced with Python 3.12.3 and
PyTorch 2.10.0+cu128 on an NVIDIA GeForce RTX 5090 (Linux).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # unit and smoke tests, CPU
```

## Reproducing the thesis results

```bash
python scripts/run_all_experiments.py --device cuda
```

This runs every experiment of the thesis with the seeds 20251–20255 and writes

- `results/all_results_<timestamp>.json`: mean, standard deviation (population SD over seeds) and
  per-seed metrics of every experiment, plus hardware and software information under `_meta`,
- `runs/<timestamp>__<experiment>__multiseed/`: one folder per seed with `cfg.json`, `metrics.csv`
  (per-epoch history), `model.pt` and `summary.json`,
- the PNG figures and auxiliary tables (the thesis tables are written by `gen_all_tables.py`, see below).

The classical baselines run on the CPU and are checkpointed per seed, so an interrupted run resumes;
the Heston baselines take roughly 40 minutes per seed, deterministic and Monte Carlo alike.
Selected experiments can be re-run with `--only`, for example
`--only act_ablation_relu.json,act_ablation_sine.json`.

The thesis reports `results/all_results_20261108_213044.json`. It combines the Black–Scholes entries
(including ablations A1, A2 and A4 and the Black–Scholes classical calibrations) of
`results/all_results_20261031_162739.json` (run of 31 October 2026) with the local-volatility and Heston
entries (including ablation A3 and the classical calibrations) of
`results/pricer_fix/all_results_pricer_fix.json` (run of 8 November 2026, same seeds); `_meta` records
both sources. In the run of 8 November, neural and classical jobs ran concurrently, so its wall-clock
times are upper bounds. `runs/` holds the per-run folders of both runs.

To regenerate the thesis tables, figures and p-values from these results without training:

```bash
python scripts/gen_all_tables.py  --results-file results/all_results_20261108_213044.json
python scripts/gen_all_figures.py --results-file results/all_results_20261108_213044.json
python scripts/paired_tests.py    --results-file results/all_results_20261108_213044.json
```

`paired_tests.py` prints the paired two-sided t-tests (paired by seed) quoted in the thesis with their
Holm-corrected p-values. Inside the thesis repository the tables and figures are written to the LaTeX
assets; in this stand-alone repository they are written to `outputs/`.

The numerical illustration of the grid-density stability constant (thesis Appendix A,
"Numerical Illustration: Grid-Sparsity Scaling for Black–Scholes") is reproduced by

```bash
python scripts/stability_scaling.py
```

## Experiments

| Thesis experiment | Configuration files |
|---|---|
| Black–Scholes 2×2 design (price-only vs. Greeks, without vs. with PDE) | `bs_nn_baseline`, `bs_nnm2`, `bs_nnpde`, `bs_softplus_full` |
| Local volatility, same design | `lv_nn_baseline`, `lv_nnm2`, `lv_nnpde`, `lv_dnn` |
| Heston (no PDE variants) | `heston_nn_baseline`, `heston_nnm2` (`heston_full_vs_priceonly` is identical to `heston_nnm2`) |
| Ablation A1, Greek supervision | comparisons within the Black–Scholes design |
| Ablation A2, activation function | `act_ablation_bs` (softplus), `act_ablation_tanh`, `act_ablation_relu`, `act_ablation_sine` |
| Ablation A3, grid density | `lv_density_10x5`, `lv_density_50x20` |
| Ablation A4, observation noise | `bs_softplus_full` with 0, 0.5 and 1 % relative price noise (Greek noise half as large), created in `run_all_experiments.py` |

Further configurations (`bs_softplus_price_only`, `bs_nondnn`, `bs_kink_mollify_on/off`,
`lv_full_vs_priceonly`, `probe_jacobian_lv_price`) and the parameter sweeps are diagnostics that the
thesis does not report.

## Training protocol (summary)

- MLP with 4 hidden layers of width 64, softplus activation unless ablated, Xavier-uniform
  initialisation; the sine activation is `sin(30 x)`.
- AdamW (learning rate 1e-3, weight decay 1e-4), cosine schedule over 2000 epochs, mini-batches of 256,
  gradient clipping at norm 1.
- Loss curriculum: the epoch budget is split into equal stages that switch on the loss terms of the
  variant (PINN: prices, then Greeks from epoch 666, then the PDE residual from epoch 1332;
  NN-M2 and NN-PDE: second term from epoch 1000). Early stopping (patience 200 epochs on the validation
  loss) and the choice of the restored checkpoint are restricted to the final stage, so every model is
  trained with all of its loss terms.
- Data: 1000-node strike–maturity grid. The extrapolation region (T > 0.8 or K outside
  [0.85, 1.15]·S0) is held out; the 540 interior nodes are split 7:1:1 into training, validation and
  interpolation-test nodes.

## Repository structure

```
configs/    experiment configurations (JSON)
src/
  cli/      command-line interface (python -m src.cli.main run|grid|report)
  data/     grids, synthetic data generation, noise models
  exp/      configuration schema, experiment runner, metrics, seeding, logging
  models/   networks, pricers (closed form, Dupire, Fourier, Monte Carlo), PDE operators, calibration
  train/    training engine, losses, collocation sampling
scripts/    run_all_experiments.py, gen_all_tables.py, gen_all_figures.py, paired_tests.py,
            stability_scaling.py
results/    aggregated results (JSON) of the reported runs
runs/       per-run artefacts of the reported runs
tests/      unit and smoke tests, including pricer regression tests
```

## Citation

```bibtex
@misc{weissel2026code,
  author       = {Weissel, Man\'u},
  title        = {Learning the Dynamics Behind Prices: Differential Neural Networks
                  for Option Calibration --- Source Code},
  year         = {2026},
  howpublished = {\url{https://github.com/manuyweissel/dnn-option-calibration}}
}
```

## License

MIT, see `LICENSE`.

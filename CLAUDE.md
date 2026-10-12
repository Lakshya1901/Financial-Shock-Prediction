# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Financial shock propagation as node classification on stock graphs: after a shock at stock *s* on day *d*, predict which other stocks move > 3% within *h* days. Two experiments share one training/eval stack:

1. **Finance sector** (`data/finance_universe.py`, `experiments/run_static_comparison.py`): 10 US banks, 1-day horizon, 8 features, peer-link + shock-star graph, single snapshot. Its seed-0 GATv2 is the model served by `app.py`.
2. **Multi-market ablation** (`data/shock_dataset.py`, `experiments/run_ablation.py`): 194 stocks in 4 markets, 5-day horizon, 14 features, 12 weekly snapshots, Pearson vs DCC-GARCH top-k graphs, `TemporalGATv2`.

`ARCHITECTURE.md` explains the data flow and evaluation protocol; `docs/TEMPORAL_GATV2_DESIGN.md` covers the temporal model's math. `README.md` holds the results tables. Update the README whenever results change.

## Commands

Run everything from the repo root (paths are CWD-relative). The local venv is `.venv/` (`--system-site-packages`, which provides torch/PyG).

```bash
.venv/bin/python -m pytest                          # all tests
.venv/bin/python -m pytest tests/test_dcc.py::test_correlation_path_has_no_lookahead
.venv/bin/python -m experiments.run_static_comparison [--quick]
.venv/bin/python -m experiments.run_ablation [--quick] [--seeds N] [--workers N] [--rebuild]
.venv/bin/python -m experiments.generate_figures
.venv/bin/python -m streamlit run app.py
```

- Set `OMP_NUM_THREADS=1` when running tests alongside training jobs. Thread oversubscription makes everything crawl.
- The full ablation takes hours on CPU. `TemporalGATv2` on 194 nodes × 12 snapshots is about 2 s per batch of 16 events. MPS is much slower than CPU here.

## Invariants to preserve

- **No look-ahead.** Shock z-scores use trailing volatility (`shift(1)`). Features and Pearson edges use data ≤ d. DCC/GARCH parameters and feature scaling are fit on train days only, then applied forward via `DCCGARCHEstimator.correlation_path`. Splits are by event date, with an `horizon`-day purge gap. `tests/` checks these; keep them passing.
- **Evaluation protocol.** Early stopping and the F1 threshold use validation only; reported numbers are test-set mean ± std over seeds. Always compare against `VolatilityHeuristic` and the base rate (PR-AUC). Never report a best-of-many-configs number as a result.
- **Train/serve parity.** `app.py` must build inputs only via `node_features` / `build_graph` from `data/finance_universe.py`. Feature order is stored in the checkpoint (`feature_names`).
- **Figures are built only from saved results.** `generate_figures.py` exits if an input is missing; don't add synthetic fallbacks.
- **DCC with many assets.** Use the composite likelihood (`composite_above`). The full likelihood collapses to a ≈ 0 at N ≈ 200.

## Data and caching

- `data/.cache/` holds yfinance pickles keyed by the fixed window `DATA_START`/`DATA_END` in `data/multi_market.py`.
- `data/processed/shock_dataset.pkl` caches the built multi-market dataset (rebuild with `--rebuild`, which takes a few minutes).
- Both cache directories are gitignored. `trained_models/` and `experiments/results|figures/` are committed.

## Layout notes

- Every model returns probabilities, not logits. `experiments/training.py` applies BCE to clamped probabilities with a `pos_weight` sample weight.
- Static models share `forward(x, edge_index, edge_attr)` (`models/baselines.py`). `TemporalGATv2.forward` takes a list of T `Data`/`Batch` objects. `experiments/training.forward_batch` dispatches between the two.
- A dataset "item" is `{"snaps", "y", "train_mask", "eval_mask"}`. The multi-market experiment scores only S&P 500 nodes other than the source, for every config.

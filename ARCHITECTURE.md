# Architecture

How the code fits together. For the temporal model's math see
[docs/TEMPORAL_GATV2_DESIGN.md](docs/TEMPORAL_GATV2_DESIGN.md); for results see the README.

---

## The task

Given that stock *s* had a shock on day *d*, predict for every other stock *i* whether it
moves by more than 3% over the next *h* trading days:

```
y_i = 1  if  |P_i(d + h) / P_i(d) − 1| > 3%
```

This is node-level binary classification on a stock graph. The shock source is marked
with a node feature and excluded from scoring.

A **shock** is a daily return more than 3 standard deviations from zero, measured against the
stock's *trailing* 60-day volatility (only past data). The CascadeNet experiment also requires
|return| > 3%. At most 3 shocks per day are kept (the largest), so market-wide crash days
don't flood the dataset with near-duplicate samples.

---

## Two experiments

| | Finance sector | CascadeNet ablation |
|---|---|---|
| Universe | 10 US banks | 194 stocks: S&P 500 top 50, NIFTY 50, FTSE top 50, Nikkei top 50 |
| Data | `data/finance_universe.py` | `data/multi_market.py` → `data/shock_dataset.py` |
| Horizon | 1 trading day | 5 trading days |
| Graph | documented peer links + a star from the shock source | top-10 \|correlation\| neighbours per node (Pearson or DCC-GARCH) |
| Features | 8 per node | 14 per node, z-scored on the training period |
| Input | one snapshot on the shock date | 12 snapshots, 5 trading days apart, ending on the shock date |
| Runner | `experiments/run_static_comparison.py` | `experiments/run_ablation.py` |
| Scored nodes | all banks except the source | S&P 500 stocks except the source |

The finance-sector experiment re-evaluates the original capstone model ("OptimalGNN"). Its
trained GATv2 is the model served by `app.py`.

---

## Data pipeline (CascadeNet)

```
yfinance (2021-10-01 → 2026-09-29, cached in data/.cache/)
   │  data/multi_market.py: load_usd_panel()
   ▼
USD close + volume on the NYSE calendar
   (local close × FX rate; non-US markets forward-filled over local holidays)
   │  data/shock_dataset.py: build_dataset()
   ├─► feature panel X[day, stock, 14]   rolling stats, z-scored with train-period mean/std
   ├─► labels Y[day, stock]              |5-day forward return| > 3%
   ├─► events (day, source)              trailing-vol z-score > 3 and |r| > 3%
   │                                     split by date: 60% train / 20% val / 20% test,
   │                                     with a 5-day purge gap between splits
   ├─► Pearson graphs                    per event day, trailing 120-day correlation
   └─► DCC graphs                        DCC-GARCH fitted on train days only, then filtered
                                         forward (data/dynamic_correlation.py: correlation_path)
   ▼
data/processed/shock_dataset.pkl  (cached; `--rebuild` to regenerate)
```

**Look-ahead guards** (each one has a test in `tests/`):
- the shock z-score uses volatility up to day d−1
- features and Pearson edges use data up to day d
- DCC parameters, GARCH parameters and feature scaling are estimated on training days only;
  the DCC filter at day d only uses returns up to d
- split boundaries have a gap of `horizon` days so no training label overlaps evaluation

**DCC estimation.** With ~200 assets, the full DCC likelihood drives the news parameter
*a* toward 0, which makes correlations effectively constant (this happened here: a≈0.0006).
`DCCGARCHEstimator` therefore uses the pairwise composite likelihood (Pakel et al., 2021)
when N > 20. On the training window it gives a≈0.018, b≈0.928.

---

## Models

All single-snapshot models are in `models/baselines.py` and share the interface
`forward(x, edge_index, edge_attr) → probabilities (N,)`:

- `StaticGATv2`: 2 GATv2 layers (2 heads → 1 head, hidden 32) with edge weights in the
  attention score, BatchNorm + ELU. This is the "OptimalGNN" architecture.
- `GATv1Net`, `GCNNet`, `SAGENet`: the same depth and width with other message-passing layers
- `MLPNet`: a per-node MLP with no message passing

`models/temporal_gatv2.py: TemporalGATv2`:
1. a shared 2-layer GATv2 runs on each of the T snapshots
2. a transformer block runs over each node's T embeddings
3. the last position goes through an MLP head and a sigmoid

There are also two non-learned references: the volatility heuristic (score = 20-day volatility)
and logistic regression.

---

## Training and evaluation (`experiments/training.py`)

- **Loss and optimiser:** weighted BCE with `pos_weight = neg/pos`, Adam, gradient clipping
  at 1.0.
- **Early stopping:** on validation ROC-AUC; the best weights are restored.
- **F1 threshold:** chosen on validation and then applied unchanged to test.
- **Seeds:** each trained model runs over 5 seeds, and results report mean ± std.
- **Reported metrics:** test ROC-AUC, PR-AUC (compare against the positive base rate), F1,
  precision and recall.
- **Parallelism:** `run_ablation.py` runs the (config, seed) pairs in a process pool, one
  thread per worker.

---

## Interpretability (`analysis/attention_attribution.py`)

- `TemporalAttentionAnalyzer`: shows how much attention the shock-date position puts on each
  of the 12 weeks.
- `EdgeImportanceAnalyzer`: reads first-layer GATv2 attention on each edge of the shock-date
  snapshot and aggregates it into a 4×4 market-pair matrix.

Attention shows what the model weighted. It is not a causal explanation.

---

## App (`app.py`)

The app loads `trained_models/finance_gatv2.pth` and downloads the last 3 months of prices
for the 10 banks. It builds features and the graph with the **same functions used in
training**, then scores every bank. Optionally, it overrides the shock bank's latest return to
explore a hypothetical shock.

The flag threshold and the test metrics shown in the app come from the checkpoint's metadata.

---

## Interview questions you should be able to answer

1. **Why GATv2 rather than GAT?** In GAT every node ranks its neighbours in the same order
   ("static" attention), because the attention vector is applied before the nonlinearity.
   GATv2 applies it after, so the ranking can depend on the query node. It can also include
   edge weights (here, correlations) in the score.
2. **Why a chronological split with a purge gap?** A random split lets the model train on
   events from the same days it's tested on (crash days produce many correlated events). The
   gap stops a training label's 5-day window from overlapping test days.
3. **Why report PR-AUC and a base rate?** About 44% of test labels are positive in the
   ablation and about 9% in the finance set. An F1 of around 0.6 is roughly what predicting
   "everyone moves" gets at a 44% base rate, so F1 alone is misleading.
4. **Why include a volatility heuristic?** "Will this stock move more than 3%?" is mostly
   answered by how volatile the stock already is. Any graph model has to beat that to show
   that the graph adds information.
5. **Why composite likelihood for DCC?** The full likelihood with ~200 assets is biased toward
   constant correlations and costs O(N³) per day. Summing bivariate likelihoods over
   neighbouring pairs fixes the bias and costs O(N).
6. **What do the attention maps tell you?** They show where the model put its weight, not why
   prices moved, so they're a diagnostic and not evidence of contagion paths.

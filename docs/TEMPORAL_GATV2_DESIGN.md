# TemporalGATv2 — Architecture Design

> Design notes for `models/temporal_gatv2.py`.
> Decision: T=12 weekly snapshots, spatial_heads=2, hidden_dim=32, temporal_heads=4.
> Data pipeline: `data/shock_dataset.py`. Measured results: README → Results.

---

## Problem Statement (Refined)

The existing model sees one graph snapshot — a single correlation/supply-chain matrix computed over the full historical period. This means it cannot answer:

- *How did correlations between AAPL and NVDA evolve in the 3 months before this shock?*
- *Was there a contagion pathway that built up gradually, or did it appear suddenly?*
- *Which market (US/India/UK/Japan) was the transmission source?*

**CascadeNet's core claim:** temporal context — how the graph structure and node signals *evolve over time* — is predictive of which shocks cross-market boundaries and cascade. A temporal model can learn that gradual correlation increases precede contagion, while sudden spikes are often self-contained.

---

## Input/Output Specification

```
Input:
  T = 12           # weekly snapshots (≈3 months of history)
  N = ~200         # stocks across 4 markets
  F = 14           # features per node (see below)
  
  For each t in 1..T:
    X_t  ∈ R^(N × F)    # node feature matrix at week t
    A_t  ∈ R^(N × N)    # DCC-GARCH correlation matrix (top-10 |corr| per node → sparse)
    E_t  ∈ R^(|E_t|)    # edge weights (|DCC correlation| of kept edges)

Output:
  p ∈ R^N              # shock probability per stock at time T+1
  p_i = P(|return_i over next 5d| > 3%)
```

---

## Mathematical Formulation

### Stage 1: Spatial Attention Within Each Snapshot

For each snapshot t, run GATv2Conv over graph G_t = (X_t, E_t):

**Attention score** between node i and neighbor j at time t:

```
e_ij^t = a^T · LeakyReLU( W_spatial · [h_i^t ‖ h_j^t] + W_edge · ε_ij^t )
```

where:
- `h_i^t` = current node embedding for stock i at time t  (dim: F)
- `‖` = concatenation  
- `W_spatial` ∈ R^(2F → hidden_dim)  is a learned weight matrix
- `W_edge` ∈ R^(1 → hidden_dim)  lifts the scalar edge weight to hidden_dim
- `a` ∈ R^hidden_dim  is the learned attention vector
- `ε_ij^t` = DCC-GARCH correlation between stocks i and j at week t  (scalar)

**Why GATv2 formulation?**
Standard GAT scores `e_ij = LeakyReLU(a^T · [W·h_i ‖ W·h_j])`. Because `a` is applied
*before* the nonlinearity, the score splits into a query term plus a key term, and the
ranking of neighbours j is the same for every query node i ("static attention").
GATv2 applies `a` *after* the nonlinearity, so the ranking can depend on the query
("dynamic attention"), which is strictly more expressive (Brody et al., 2021).

**Normalized attention weights** (softmax over neighbors):

```
α_ij^t = exp(e_ij^t) / Σ_{k ∈ N(i)} exp(e_ik^t)
```

**Multi-head aggregation** (2 heads, concatenated):

```
z_i^t = ‖_{k=1}^{K=2} ELU( Σ_{j ∈ N(i)} α_ij^{t,k} · W_v^k · h_j^t )
```

After BatchNorm + ELU + Dropout(0.3), a second GATv2Conv layer (1 head, 64 → 64) with
BatchNorm + ELU produces the spatial embedding `z_i^t ∈ R^64` that feeds the temporal stage.
The same spatial weights are shared across all T snapshots.

---

### Stage 2: Temporal Attention Across T Snapshots

For each stock i independently, we have a sequence of T embeddings:

```
Z_i = [ z_i^1, z_i^2, ..., z_i^T ]  ∈ R^(T × 64)
```

This is exactly a sequence of length T with feature dimension 64 — the same
structure that transformers were designed for.

**Sinusoidal Positional Encoding** (Vaswani et al. 2017, Section 3.5):

```
PE(t, 2k)   = sin( t / 10000^(2k/d_model) )
PE(t, 2k+1) = cos( t / 10000^(2k/d_model) )
```

where t = week index (0 to T-1), d_model = 64.

Why sinusoidal? — It generalizes to sequence lengths not seen during training
(important: we might want to run inference on T=8 snapshots if early in the year).
A learned positional embedding would fail on unseen lengths.

Add positional encoding:
```
Z_i' = Z_i + PE   (elementwise, same PE matrix for all stocks)
```

**Multi-Head Self-Attention** (4 heads, d_k = 16 each):

```
Q = Z_i' · W_Q,   K = Z_i' · W_K,   V = Z_i' · W_V
           W_Q, W_K, W_V ∈ R^(64 × 64)

Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V
                     where d_k = 64/4 = 16  (per head)

head_k = Attention(Z_i'·W_Q^k,  Z_i'·W_K^k,  Z_i'·W_V^k)

MultiHead(Z_i') = Concat(head_1, ..., head_4) · W_O
                  W_O ∈ R^(64 × 64)
```

Why `1/√d_k` scaling? — Without scaling, dot products grow large for high-dimensional
keys, pushing softmax into extremely peaked regions where gradients vanish.
Dividing by √d_k keeps magnitudes reasonable regardless of d_k.

**Residual blocks** (Pre-LN formulation, more stable than Post-LN):

```
H_i = Z_i' + MultiHead( LayerNorm(Z_i') )
H_i = H_i  + FFN( LayerNorm(H_i) )

where FFN(x) = Linear(ReLU(Linear(x, dim_ff=256)), dim_out=64)
```

Result: `H_i ∈ R^(T × 64)` — temporally-contextualized embeddings for all T weeks.

---

### Stage 3: Readout and Classification

We only need to predict at T+1 — the week after the last snapshot. We use the
**last position** H_i[T-1] as our representation of the "current state":

```
h_i^final = H_i[T-1]   ∈ R^64
```

Why last position, not mean pooling?
- The shock prediction task is inherently about "what happens next" from the current
  state. The most recent snapshot (T) carries the most relevant signal.
- Mean pooling would dilute the most recent information with 11 older snapshots.
- An alternative: learned weighted sum (attention pooling over T). We'll try this in ablation.

**Classification head:**

```
p_i = σ( W_2 · ReLU( W_1 · h_i^final + b_1 ) + b_2 )
      W_1 ∈ R^(64 → 32),  W_2 ∈ R^(32 → 1)
```

Sigmoid output gives shock probability ∈ (0, 1).

---

## Full Architecture Diagram (ASCII)

```
For each week t = 1..12:
  ┌─────────────────────────────────────────┐
  │  Graph G_t: N nodes, DCC-GARCH edges    │
  │  X_t ∈ R^(N×14), E_t (correlations)    │
  │                                          │
  │  ┌──────────────────────────────────┐   │
  │  │  GATv2Conv (heads=2, edge_dim=1) │   │
  │  │  14 → 32 per head → 64 total     │   │
  │  └──────────────┬───────────────────┘   │
  │                 │ BatchNorm + ELU        │
  │                 │ Dropout(0.3)           │
  │                 ▼                        │
  │            z_i^t ∈ R^64                 │
  └─────────────────────────────────────────┘
            (repeated T=12 times)

  Collect per-stock temporal sequence:
  Z_i = [z_i^1 | z_i^2 | ... | z_i^12]  ∈ R^(12 × 64)

  ┌─────────────────────────────────────────────────┐
  │  Temporal Attention (per stock independently)   │
  │                                                  │
  │  Z_i' = Z_i + SinusoidalPE(T=12, d=64)         │
  │                                                  │
  │  ┌───────────────────────────────────────────┐  │
  │  │  Multi-Head Self-Attention (heads=4)      │  │
  │  │  Q, K, V projections in R^(12×64)         │  │
  │  │  Attention: softmax(QK^T / √16) · V       │  │
  │  └───────────────────────────────────────────┘  │
  │  Add & LayerNorm  →  Feed-Forward  →  Add & LN  │
  │                                                  │
  │  H_i ∈ R^(12 × 64)                             │
  └─────────────────────────────────────────────────┘

  Readout: h_i^final = H_i[T-1 = 11]  ∈ R^64

  ┌───────────────────────────────┐
  │  Classifier                   │
  │  Linear(64→32) + ReLU         │
  │  Linear(32→1)  + Sigmoid      │
  └───────────────────────────────┘

  p_i ∈ (0,1) = P(stock i affected in next 5 days)
```

---

## Design Decisions and Alternatives Rejected

### Why T=12 weekly snapshots?

Weekly (not daily) because:
- Daily would give T=60 snapshots for 3 months → attention is O(T²) in memory, 60² = 3600 vs 12² = 144
- DCC-GARCH is estimated on daily returns; snapshots sample the daily correlation path every 5 trading days
- 12 weeks = ~3 months, which covers most financial regime transitions
- Alternative: T=24 monthly over 2 years — rejected because DCC-GARCH needs enough observations per window (need ≥30 data points per weekly window)

### Why temporal attention vs LSTM?

Two alternatives considered:

| Approach | Pros | Cons |
|----------|------|------|
| LSTM | Proven for time series | Sequential → no parallelization, forgets distant past |
| **Transformer (chosen)** | Parallel, attends any distance, attention is interpretable | Needs positional encoding, more params |
| 1D-CNN over T | Very fast | Cannot model long-range temporal dependencies |

Transformer chosen because:
1. Attention weights are **interpretable** — we can ask "which past week was most important for this prediction?" — directly relevant for the paper
2. Parallel computation over T (LSTM is sequential)
3. The T=12 sequence is short enough that quadratic attention complexity is not a problem

### Why not use existing temporal GNN layers (TGN, A3TGCN)?

- **TGN (Temporal Graph Network)**: designed for continuous-time dynamic graphs with events. Our graphs are discrete weekly snapshots — different setting.
- **A3TGCN**: combines attention TGCN with temporal attention, but uses GCN (no edge weights) for the spatial layer. We need edge weights (DCC correlations).
- **STGCN**: spatial-temporal GCN, but no edge attributes, designed for traffic networks.
- **Our approach**: explicitly designed for weighted, attributed financial graphs with discrete weekly snapshots.

The combination used here — DCC-GARCH correlations as GATv2 edge weights, with transformer-style temporal attention over weekly snapshots — is a design choice for this project, not a claim of novelty.

### Why sinusoidal PE vs learned PE?

- Sinusoidal generalizes to T lengths not seen during training
- We might run inference at T=8 if the model is deployed mid-year
- Learned PE would fail silently on unseen T

---

## Node Features for CascadeNet (14 features)

Computed from USD prices on the NYSE calendar (`data/shock_dataset.py`); continuous
features are z-scored with training-period statistics.

| # | Feature |
|---|---------|
| 0 | 20-day mean return (%) |
| 1 | 20-day volatility (%) |
| 2 | Latest daily return (%) |
| 3 | 5-day mean return (%) |
| 4 | Price / 20-day mean price − 1 |
| 5 | log(volume / 20-day mean volume) |
| 6 | Is shock source (set only on the shock-date snapshot) |
| 7 | Is non-US stock |
| 8–11 | Market one-hot (S&P 500, NIFTY, FTSE, Nikkei) |
| 12 | Sector index / 11 (single scalar, not one-hot) |
| 13 | log(volume / 5-day mean volume), clipped to ±3 |

---

## Parameter Count

62,977 trainable parameters with the defaults (`count_parameters()`), most of them in
the temporal block's feed-forward layer (64 → 256 → 64). Small enough to train on CPU.

---

## Loss Function

**Weighted BCE** with class-imbalance correction (`experiments/training.py`):

```
pos_weight = num_negative / num_positive   (computed on the training events)
L = BCE(clamp(p, 1e-6, 1-1e-6), y, weight = pos_weight if y = 1 else 1)
```

The model returns probabilities (sigmoid inside `forward`), so the loss is applied to
clamped probabilities rather than logits.

---

## What We're Validating

Ablation study will quantify the contribution of each component (Days 12–14):

| Config | Temporal | Dynamic Edges | Cross-market |
|--------|----------|---------------|--------------|
| Static GATv2 (baseline) | ✗ | ✗ | ✗ |
| + Temporal attn | ✓ | ✗ | ✗ |
| + DCC-GARCH edges | ✓ | ✓ | ✗ |
| **Full CascadeNet** | ✓ | ✓ | ✓ |

**Hypothesis (before running):** each component adds measurable ROC-AUC, with cross-market
nodes giving the largest gain. See README → Results for what was actually measured.

---

## Interview Prep — Explain the Design

**Q: Why do you apply spatial attention first, then temporal?**

A: The alternative (temporal first, then spatial) would require sharing edge structure across time, which is tricky when edges change at every snapshot. Spatial-then-temporal is the natural decomposition: first aggregate neighborhood information *within* a snapshot (GATv2), then reason about *how that aggregated signal evolved* over time (transformer). This matches how financial analysts think: "what was this stock's position in its supply chain at week t?" before "how did that position change over 12 weeks?"

**Q: Why not just concatenate all 12 snapshots and run a big static GATv2?**

A: Three reasons. (1) Concatenating X_1..X_T into a single feature vector loses the sequential structure — a permutation of the snapshots would give the same output. The temporal attention preserves order via positional encoding. (2) The graph structure (edges) changes across snapshots (DCC-GARCH changes weekly) — concatenating can't represent this. (3) Concatenating T=12 snapshots with F=14 features each gives 168-dimensional input, which overparameterizes the spatial layer and overfits on small financial datasets.

**Q: What's the computational bottleneck?**

A: The temporal attention is O(N · T²) per forward pass (N stocks, each attending over T=12 snapshots). With N=200, T=12: 200 × 144 = 28,800 attention computations. Trivial. The DCC-GARCH estimation (Stage 0, before the model) is the actual bottleneck: O(N² · T_data) where T_data = number of trading days.

**Q: How does information cross markets?**

A: Through cross-market edges in the graph. When AAPL shocks, its DCC correlation with INFY (NIFTY) during that period may be high (USD/INR exposure). That correlation becomes an edge. The GATv2 spatial layer propagates AAPL's shock embedding to INFY's neighborhood, and the temporal layer learns whether cross-market propagation tends to precede or follow domestic propagation.

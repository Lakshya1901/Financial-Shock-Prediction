"""
attention_attribution.py — attention-based interpretability for the multi-market model.

    TemporalAttentionAnalyzer  — which past *weeks* did each node attend to?
    EdgeImportanceAnalyzer     — how much GATv2 attention flows along each edge,
                                 aggregated by market pair
    ShockAttributionReport     — both of the above for one shock event, as markdown

Caveat: attention weights describe what the model looked at, not what caused
the outcome. They are a diagnostic, not a causal attribution.

* ``TemporalAttentionAnalyzer`` uses ``TemporalGATv2.get_temporal_attention_weights()``,
  which re-runs the forward pass with ``need_weights=True``.
* ``EdgeImportanceAnalyzer`` calls ``model.spatial.conv1`` with
  ``return_attention_weights=True`` on the shock-date snapshot.

Markets index convention used throughout:
    0 → SP500   1 → NIFTY   2 → FTSE   3 → NIKKEI

Usage
-----
>>> from analysis.attention_attribution import ShockAttributionReport
>>> report_gen = ShockAttributionReport(model, snapshots, tickers, ticker_to_market)
>>> report = report_gen.generate(shock_ticker="AAPL", top_k=10)
>>> print(report_gen.to_markdown(report))
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Resolve sibling-package imports regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.temporal_gatv2 import TemporalGATv2  # noqa: E402


# ---------------------------------------------------------------------------
# Utility: pure-Python markdown table formatter (no tabulate dependency)
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a GitHub-flavoured markdown table.

    Pure-Python fallback for pandas .to_markdown() which requires tabulate.
    """
    cols = [""] + list(df.columns)
    rows = [[str(idx)] + [str(v) for v in row] for idx, row in df.iterrows()]
    col_widths = [max(len(str(c)), max((len(r[i]) for r in rows), default=0))
                  for i, c in enumerate(cols)]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(cols), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MARKETS: List[str] = ["SP500", "NIFTY", "FTSE", "NIKKEI"]
MARKET_INDEX: Dict[str, int] = {m: i for i, m in enumerate(MARKETS)}


# ===========================================================================
# 1. TemporalAttentionAnalyzer
# ===========================================================================


class TemporalAttentionAnalyzer:
    """
    Extracts and summarises the *temporal* attention maps produced by
    the multi-market model's transformer block.

    The transformer assigns each node a (T × T) soft-attention matrix
    ``A[i]``, where ``A[i, q, k]`` is the weight that query position *q*
    placed on key position *k* when building the contextualised embedding
    for node *i* at time *q*.

    For shock prediction we care almost exclusively about the **last query**
    row ``A[i, T-1, :]``, because the classifier reads the T-1 embedding.

    Args:
        model: A trained (or untrained, for testing) ``TemporalGATv2`` instance.
    """

    def __init__(self, model: TemporalGATv2) -> None:
        self.model = model
        self.model.eval()

    # ------------------------------------------------------------------
    def get_attention_maps(self, snapshots: List[Data]) -> Dict:
        """
        Run the model's interpretability forward pass and return structured
        attention information.

        Args:
            snapshots: List of T ``torch_geometric.data.Data`` objects.

        Returns:
            dict with keys:
                ``attention_weights`` (np.ndarray, shape N×T×T):
                    Per-node soft-attention matrices.  Entry [i, q, k]
                    is node i's attention from query week q to key week k.
                ``most_important_week`` (np.ndarray, shape N):
                    For each node, the key week index that received the
                    highest attention weight from the *last* query position
                    (i.e. ``argmax(A[i, -1, :])``, the week the model found
                    most informative for today's prediction).
                ``attention_entropy`` (np.ndarray, shape N):
                    Shannon entropy (nats) of each node's last-query
                    attention distribution.  Low entropy → model focused on
                    a single week.  High entropy → diffuse attention.
        """
        with torch.no_grad():
            probs, attn_weights = self.model.get_temporal_attention_weights(snapshots)

        # attn_weights: Tensor (N, T, T)
        attn_np: np.ndarray = attn_weights.cpu().numpy()  # (N, T, T)

        # Last-query row: which key week got most attention?
        last_query_attn = attn_np[:, -1, :]           # (N, T)
        most_important_week = np.argmax(last_query_attn, axis=1)  # (N,)

        # Shannon entropy of the last-query distribution
        # Clip to avoid log(0); add tiny epsilon for numerical safety
        eps = 1e-9
        p = np.clip(last_query_attn, eps, 1.0)
        attention_entropy = -np.sum(p * np.log(p), axis=1)  # (N,)

        return {
            "attention_weights": attn_np,           # (N, T, T)
            "most_important_week": most_important_week,  # (N,)
            "attention_entropy": attention_entropy,  # (N,)
        }

    # ------------------------------------------------------------------
    def get_node_attribution(
        self, snapshots: List[Data], node_idx: int
    ) -> Dict:
        """
        Detailed temporal attribution for a single node.

        Args:
            snapshots: List of T ``Data`` objects.
            node_idx:  0-based index of the target node (stock).

        Returns:
            dict with keys:
                ``node_idx``              – the queried node index.
                ``shock_probability``     – model's predicted shock probability (float).
                ``attention_by_week``     – dict {week_t: attention_weight} from
                                            the last query position (T-1 → each t).
                ``full_attention_matrix`` – np.ndarray (T, T) for this node.
                ``most_informative_weeks``– list of week indices sorted by
                                            descending attention (last query).
                ``attention_entropy``     – float, entropy of last-query distribution.
        """
        with torch.no_grad():
            probs, attn_weights = self.model.get_temporal_attention_weights(snapshots)

        attn_node: np.ndarray = attn_weights[node_idx].cpu().numpy()  # (T, T)
        last_query: np.ndarray = attn_node[-1, :]  # (T,)

        attention_by_week = {t: float(last_query[t]) for t in range(len(last_query))}
        sorted_weeks = sorted(attention_by_week, key=attention_by_week.get, reverse=True)

        eps = 1e-9
        p = np.clip(last_query, eps, 1.0)
        entropy = float(-np.sum(p * np.log(p)))

        return {
            "node_idx": node_idx,
            "shock_probability": float(probs[node_idx].cpu().item()),
            "attention_by_week": attention_by_week,
            "full_attention_matrix": attn_node,
            "most_informative_weeks": sorted_weeks,
            "attention_entropy": entropy,
        }

    # ------------------------------------------------------------------
# ===========================================================================
# 2. EdgeImportanceAnalyzer
# ===========================================================================


class EdgeImportanceAnalyzer:
    """
    Captures GATv2 spatial edge-attention weights and aggregates them into
    per-edge importance scores.

    PyTorch-Geometric's ``GATv2Conv`` exposes attention coefficients through
    the ``return_attention_weights=True`` keyword argument in its ``forward``
    call.  When this flag is set the layer returns a tuple
    ``(out, (edge_index_with_loops, alpha))`` where ``alpha`` has shape
    ``(E_with_loops, heads)``.

    We call ``model.spatial.conv1`` directly with this flag on the shock-date
    snapshot and average ``alpha`` across heads.

    Args:
        model: A trained (or untrained) ``TemporalGATv2`` instance.
    """

    def __init__(self, model: TemporalGATv2) -> None:
        self.model = model
        self.model.eval()

    # ------------------------------------------------------------------
    def compute_edge_importance(self, snapshots: List[Data]) -> np.ndarray:
        """
        Head-averaged first-layer GATv2 attention for each edge of the last
        (shock-date) snapshot.

        Snapshots can have different edge sets, so attention is not averaged
        across them. GATv2Conv appends self-loops after the original edges,
        so the first E entries of ``alpha`` line up with ``edge_index``.

        Returns:
            np.ndarray of shape (E_last,).
        """
        snap = snapshots[-1]
        with torch.no_grad():
            _, (_, alpha) = self.model.spatial.conv1(
                snap.x, snap.edge_index, edge_attr=snap.edge_attr,
                return_attention_weights=True,
            )
        return alpha.mean(dim=-1)[: snap.edge_index.shape[1]].cpu().numpy()

    # ------------------------------------------------------------------
    def cross_market_edge_heatmap(
        self,
        snapshots: List[Data],
        tickers: List[str],
        ticker_to_market: Dict[str, str],
    ) -> np.ndarray:
        """
        Build a (4, 4) market-to-market average edge-weight matrix.

        Entry [i, j] answers: "on average, how strongly do edges flowing
        from market i to market j contribute to the GATv2 attention?"

        Uses the last snapshot's edge topology.

        Args:
            snapshots:        List of T ``Data`` objects.
            tickers:          Ordered list of N ticker strings.
            ticker_to_market: Dict mapping ticker → market string.

        Returns:
            np.ndarray of shape (4, 4).
        """
        mean_alpha = self.compute_edge_importance(snapshots)

        last_snap = snapshots[-1]
        edge_index = last_snap.edge_index.cpu().numpy()  # (2, E)
        E = edge_index.shape[1]

        node_market = np.array(
            [MARKET_INDEX.get(ticker_to_market.get(tk, "SP500"), 0) for tk in tickers]
        )

        matrix = np.zeros((4, 4), dtype=np.float64)
        counts = np.zeros((4, 4), dtype=np.int64)

        # Truncate mean_alpha to E in case of size mismatch from self-loops
        alpha_use = mean_alpha[:E] if mean_alpha.shape[0] >= E else mean_alpha

        for e in range(alpha_use.shape[0]):
            src_node = edge_index[0, e]
            dst_node = edge_index[1, e]
            m_src = node_market[src_node]
            m_dst = node_market[dst_node]
            matrix[m_src, m_dst] += alpha_use[e]
            counts[m_src, m_dst] += 1

        # Avoid division by zero
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(counts > 0, matrix / counts, 0.0)

        return result  # (4, 4)


# ===========================================================================
# 3. ShockAttributionReport
# ===========================================================================


class ShockAttributionReport:
    """
    End-to-end attribution pipeline: runs both analyzers and formats a
    structured report suitable for the paper appendix.

    Args:
        model:            Trained ``TemporalGATv2`` instance.
        snapshots:        List of T ``Data`` objects (the inference window).
        tickers:          Ordered list of N ticker strings (node order).
        ticker_to_market: Dict mapping ticker → market string.
    """

    def __init__(
        self,
        model: TemporalGATv2,
        snapshots: List[Data],
        tickers: List[str],
        ticker_to_market: Dict[str, str],
    ) -> None:
        self.model = model
        self.snapshots = snapshots
        self.tickers = tickers
        self.ticker_to_market = ticker_to_market

        self._temporal_analyzer = TemporalAttentionAnalyzer(model)
        self._edge_analyzer = EdgeImportanceAnalyzer(model)

    # ------------------------------------------------------------------
    def generate(self, shock_ticker: str, top_k: int = 10) -> Dict:
        """
        Run the full attribution pipeline for a given shock-originating ticker.

        Steps
        -----
        1. Obtain model shock probabilities for all nodes.
        2. Identify the top-k stocks most likely to be affected.
        3. For each top-k stock, extract temporal attribution and identify
           the most informative past weeks.
        4. Compute the cross-market edge flow (4×4 matrix).

        Args:
            shock_ticker: Ticker symbol of the stock whose shock we are
                          attributing (used for labelling only; the model
                          predicts all nodes simultaneously).
            top_k:        Number of most-at-risk stocks to include in the report.

        Returns:
            dict with keys:
                ``shock_ticker``         – input ticker string.
                ``predictions``          – dict {ticker: probability (float)}.
                ``top_k_tickers``        – list of top-k ticker strings.
                ``temporal_attribution`` – dict {ticker: {week_t: weight}}.
                ``attention_entropy``    – dict {ticker: entropy (float)}.
                ``most_important_week``  – dict {ticker: week_idx (int)}.
                ``cross_market_flow``    – np.ndarray (4, 4), market edge weights.
                ``cross_market_df``      – pd.DataFrame version of cross_market_flow.
                ``temporal_summary_df``  – pd.DataFrame: rows=top-k tickers,
                                           cols=week indices, values=attention weights.
        """
        # ── 1. Run temporal analyzer ──────────────────────────────────
        with torch.no_grad():
            probs, _ = self.model.get_temporal_attention_weights(self.snapshots)
        probs_np = probs.detach().cpu().numpy()  # (N,)

        predictions = {tk: float(probs_np[i]) for i, tk in enumerate(self.tickers)}

        # ── 2. Top-k most affected stocks ────────────────────────────
        sorted_indices = np.argsort(probs_np)[::-1]
        top_k_capped = min(top_k, len(self.tickers))
        top_k_indices = sorted_indices[:top_k_capped].tolist()
        top_k_tickers = [self.tickers[i] for i in top_k_indices]

        # ── 3. Temporal attribution per top-k stock ──────────────────
        attn_maps = self._temporal_analyzer.get_attention_maps(self.snapshots)
        attn_weights: np.ndarray = attn_maps["attention_weights"]  # (N, T, T)
        entropy_all: np.ndarray = attn_maps["attention_entropy"]    # (N,)
        best_week_all: np.ndarray = attn_maps["most_important_week"]  # (N,)

        T = attn_weights.shape[1]

        temporal_attribution: Dict[str, Dict[int, float]] = {}
        attention_entropy: Dict[str, float] = {}
        most_important_week: Dict[str, int] = {}

        for i in top_k_indices:
            tk = self.tickers[i]
            last_q = attn_weights[i, -1, :]  # (T,)
            temporal_attribution[tk] = {t: float(last_q[t]) for t in range(T)}
            attention_entropy[tk] = float(entropy_all[i])
            most_important_week[tk] = int(best_week_all[i])

        # ── 4. Cross-market edge flow ─────────────────────────────────
        cross_market_flow = self._edge_analyzer.cross_market_edge_heatmap(
            self.snapshots, self.tickers, self.ticker_to_market
        )

        cross_market_df = pd.DataFrame(
            cross_market_flow, index=MARKETS, columns=MARKETS
        )

        # ── 5. Summary DataFrame (top-k × T) ─────────────────────────
        rows = []
        for tk in top_k_tickers:
            row = [temporal_attribution[tk].get(t, 0.0) for t in range(T)]
            rows.append(row)
        temporal_summary_df = pd.DataFrame(
            rows,
            index=top_k_tickers,
            columns=[f"week_{t}" for t in range(T)],
        )

        return {
            "shock_ticker": shock_ticker,
            "predictions": predictions,
            "top_k_tickers": top_k_tickers,
            "temporal_attribution": temporal_attribution,
            "attention_entropy": attention_entropy,
            "most_important_week": most_important_week,
            "cross_market_flow": cross_market_flow,
            "cross_market_df": cross_market_df,
            "temporal_summary_df": temporal_summary_df,
        }

    # ------------------------------------------------------------------
    def to_markdown(self, report: Dict) -> str:
        """
        Render the attribution report as a clean Markdown string suitable
        for pasting into the paper appendix or a Jupyter Notebook cell.

        Args:
            report: Dict returned by :meth:`generate`.

        Returns:
            Formatted Markdown string.
        """
        shock = report["shock_ticker"]
        top_k_tickers: List[str] = report["top_k_tickers"]
        predictions: Dict[str, float] = report["predictions"]
        entropy: Dict[str, float] = report["attention_entropy"]
        best_week: Dict[str, int] = report["most_important_week"]
        temporal_df: pd.DataFrame = report["temporal_summary_df"]
        cross_df: pd.DataFrame = report["cross_market_df"]

        lines: List[str] = []

        # ── Header ───────────────────────────────────────────────────
        lines.append("## Multi-Market Shock Attribution Report")
        lines.append("")
        lines.append(f"**Shock origin ticker:** `{shock}`  ")
        lines.append(f"**Top-{len(top_k_tickers)} most affected stocks:**")
        lines.append("")

        # ── Predictions table ─────────────────────────────────────────
        lines.append("### Predicted Shock Probabilities (Top-k)")
        lines.append("")
        lines.append(
            "| Rank | Ticker | Market | Shock Prob | Most Informative Week | Attn Entropy |"
        )
        lines.append(
            "|-----:|--------|--------|------------|----------------------|--------------|"
        )
        for rank, tk in enumerate(top_k_tickers, 1):
            mkt = self.ticker_to_market.get(tk, "?")
            prob = predictions.get(tk, 0.0)
            bw = best_week.get(tk, -1)
            ent = entropy.get(tk, 0.0)
            lines.append(
                f"| {rank} | `{tk}` | {mkt} | {prob:.4f} | {bw} | {ent:.4f} |"
            )
        lines.append("")

        # ── Temporal attention heatmap (text) ─────────────────────────
        lines.append("### Temporal Attention Weights (Last-Query Position)")
        lines.append("")
        lines.append(
            "_Each cell shows the attention weight that the model placed on that "
            "historical week when forming the final prediction embedding._"
        )
        lines.append("")
        lines.append(_df_to_markdown(temporal_df.round(4)))
        lines.append("")

        # ── Cross-market flow ─────────────────────────────────────────
        lines.append("### Cross-Market Edge Attention Flow")
        lines.append("")
        lines.append(
            "_Entry [src, tgt] = average GATv2 edge-attention weight for edges "
            "flowing from **src** market nodes to **tgt** market nodes "
            "(last snapshot topology)._"
        )
        lines.append("")
        lines.append(_df_to_markdown(cross_df.round(4)))
        lines.append("")

        # ── Interpretation notes ──────────────────────────────────────
        lines.append("### Interpretation Notes")
        lines.append("")
        lines.append(
            "- **Low entropy** → model focused on a narrow set of past weeks "
            "(high-confidence signal source)."
        )
        lines.append(
            "- **High entropy** → model spread attention evenly across weeks "
            "(diffuse, uncertain history)."
        )
        lines.append(
            "- **Diagonal dominance** in cross-market flow → intra-market "
            "correlations dominate; off-diagonal → cross-market contagion."
        )
        lines.append("")

        return "\n".join(lines)

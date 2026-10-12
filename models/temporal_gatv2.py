"""
TemporalGATv2 — Temporal Graph Attention Network for Financial Shock Propagation.

Architecture (multi-market extension):

    Stage 1 — Spatial Attention (per snapshot):
        For each of T weekly graph snapshots, run GATv2Conv to aggregate
        neighborhood information using DCC-GARCH correlations as edge weights.

    Stage 2 — Temporal Attention (per node):
        For each node independently, apply transformer-style multi-head
        self-attention across its T spatial embeddings to capture how the
        node's graph-context evolved over time.

    Stage 3 — Classification:
        Use the last temporal position (most recent week) as the node's
        final representation. Linear head → sigmoid → shock probability.

Reference:
    Brody et al., 2021 — "How Attentive are Graph Attention Networks?" (GATv2)
    Vaswani et al., 2017 — "Attention is All You Need" (temporal attention)

Usage:
    >>> model = TemporalGATv2(input_dim=14)
    >>> # snapshot_list: list of T PyG Data objects
    >>> probs = model(snapshot_list)   # shape (N,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from typing import List


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017, Section 3.5).

    For position t and dimension 2k / 2k+1:

        PE(t, 2k)   = sin( t / 10000^(2k / d_model) )
        PE(t, 2k+1) = cos( t / 10000^(2k / d_model) )

    Sinusoidal (not learned) because it generalizes to sequence lengths not
    seen during training. If we train on T=12 and later run inference at
    T=8 (mid-year deployment), learned embeddings would fail silently.

    Args:
        d_model: Embedding dimension. Must match spatial output dim.
        max_len: Maximum sequence length to pre-compute. Default 64 is safe.
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()

        # Pre-compute PE matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # Compute the frequency terms: 10000^(2k/d_model) for k = 0..d_model//2
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # shape: (d_model // 2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        # Register as a buffer (not a parameter — not updated by optimizer)
        self.register_buffer("pe", pe)  # shape: (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input sequence.

        Args:
            x: Tensor of shape (T, d_model) or (N, T, d_model).

        Returns:
            x + PE, same shape as input.
        """
        if x.dim() == 2:
            # x: (T, d_model)
            T = x.size(0)
            return x + self.pe[:T, :]
        else:
            # x: (N, T, d_model)
            T = x.size(1)
            return x + self.pe[:T, :].unsqueeze(0)  # broadcast over N


# ---------------------------------------------------------------------------
# Spatial GATv2 Block — one snapshot
# ---------------------------------------------------------------------------

class SpatialGATv2Block(nn.Module):
    """
    Two-layer GATv2Conv that maps one graph snapshot to per-node embeddings.

    For each node i and neighbor j in snapshot G_t:

        e_ij = a^T · LeakyReLU( W_s · [h_i ‖ h_j] + W_e · edge_weight_ij )
        α_ij = softmax_j( e_ij )
        z_i  = ‖_{k=1}^{heads} ELU( Σ_j α_ij^k · W_v^k · h_j )

    Concatenation across heads gives output dim = hidden_dim * heads.

    Args:
        input_dim:  Number of input node features F.
        hidden_dim: Per-head output dimension. Total output = hidden_dim * heads.
        heads:      Number of attention heads. Default 2.
        dropout:    Dropout probability applied after each conv layer.
        edge_dim:   Edge attribute dimension. 1 for scalar DCC correlations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        dropout: float = 0.3,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.dropout = dropout

        # Layer 1: input → hidden, multi-head, concat → hidden_dim * heads
        self.conv1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            concat=True,            # concatenate heads → dim: hidden_dim * heads
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        # Layer 2: hidden * heads → hidden, single head → hidden_dim
        # (single head for a clean fixed output dimension into the temporal stage)
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim * heads,   # keep dim stable for temporal input
            heads=1,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            concat=False,           # average head outputs → dim: hidden_dim * heads
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)

        self.out_dim = hidden_dim * heads   # 64 with defaults (32 * 2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            x:          Node features, shape (N, input_dim).
            edge_index: COO edge indices, shape (2, E).
            edge_attr:  Edge weights (DCC correlations), shape (E, 1).

        Returns:
            Node embeddings, shape (N, hidden_dim * heads).
        """
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr=edge_attr)   # (N, hidden_dim * heads)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index, edge_attr=edge_attr)   # (N, hidden_dim * heads)
        h = self.bn2(h)
        h = F.elu(h)

        return h   # (N, out_dim)


# ---------------------------------------------------------------------------
# Temporal Attention Block — transformer over T node embeddings
# ---------------------------------------------------------------------------

class TemporalAttentionBlock(nn.Module):
    """
    Transformer encoder block applied independently to each node's
    temporal sequence of spatial embeddings.

    For node i with T spatial embeddings Z_i ∈ R^(T × d_model):

        Z_i'  = Z_i + SinusoidalPE(T, d_model)

        # Multi-Head Self-Attention:
        Q = Z_i' W_Q,  K = Z_i' W_K,  V = Z_i' W_V
        Attn = softmax( Q K^T / √d_k ) V,   d_k = d_model / num_heads
        H_i  = LayerNorm( Z_i' + Concat(heads) W_O )   # Add & Norm

        # Feed-Forward:
        H_i  = LayerNorm( H_i  + FFN(H_i) )
        FFN(x) = Linear(ReLU(Linear(x, dim_ff)), d_model)

    The 1/√d_k scaling prevents vanishing gradients when dot products grow
    large in high-dimensional key spaces.

    Args:
        d_model:   Embedding dimension (= spatial block's out_dim).
        num_heads: Number of attention heads. d_model must be divisible by this.
        dim_ff:    Hidden dimension of the feed-forward sub-layer.
        dropout:   Dropout applied inside attention and FFN.
        max_len:   Maximum T for positional encoding pre-computation.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        # PyTorch's MultiheadAttention: batch_first=True means input is (N, T, d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        # Layer Norms (Pre-LN formulation: more stable than Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Temporal sequence of spatial embeddings.
               Shape (N, T, d_model) — N nodes, T time steps, d_model features.

        Returns:
            Temporally-contextualized embeddings, shape (N, T, d_model).
        """
        # Add positional encoding: z + PE broadcast over N
        z = self.pos_enc(z)   # (N, T, d_model)

        # Multi-Head Self-Attention with residual + LayerNorm (Pre-LN)
        z_norm = self.norm1(z)
        attn_out, _ = self.attn(z_norm, z_norm, z_norm)  # self-attention: Q=K=V
        z = z + self.dropout(attn_out)

        # Feed-Forward with residual + LayerNorm
        z_norm = self.norm2(z)
        ffn_out = self.ffn(z_norm)
        z = z + self.dropout(ffn_out)

        return z   # (N, T, d_model)


# ---------------------------------------------------------------------------
# TemporalGATv2 — Full Model
# ---------------------------------------------------------------------------

class TemporalGATv2(nn.Module):
    """
    Temporal Graph Attention Network for cross-market financial shock propagation.

    Takes a sequence of T weekly graph snapshots and predicts, for each node
    (stock), the probability that it will be significantly affected (|return| > 3%)
    in the 5 trading days following the last snapshot.

    Architecture:
        1. SpatialGATv2Block applied to each snapshot independently.
           Output: node embedding z_i^t ∈ R^(hidden_dim * spatial_heads) per node per week.

        2. Stack embeddings across T: Z_i ∈ R^(T × d_model) per node.

        3. TemporalAttentionBlock: transformer over Z_i per node.
           Output: H_i ∈ R^(T × d_model). Take last position: h_i = H_i[T-1].

        4. Classification head: Linear(d_model → hidden_dim) + ReLU
                                + Linear(hidden_dim → 1) + Sigmoid.

    Args:
        input_dim:      Number of node features F. Default 14 (see TEMPORAL_GATV2_DESIGN.md).
        hidden_dim:     Per-head spatial output dimension. Default 32.
        spatial_heads:  GATv2 attention heads. Default 2 → d_model = 64.
        temporal_heads: Transformer attention heads. Default 4.
        dim_ff:         Feed-forward hidden dim inside temporal block. Default 256.
        spatial_dropout:  Dropout in spatial GATv2 layers.
        temporal_dropout: Dropout in transformer layers.
        edge_dim:       Edge attribute dimension. 1 for scalar correlations.

    Example:
        >>> model = TemporalGATv2(input_dim=14)
        >>> snapshot_list = [pyg_data_week1, pyg_data_week2, ..., pyg_data_week12]
        >>> probs = model(snapshot_list)   # shape (N,)
        >>> print(probs.shape)
        torch.Size([200])
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 32,
        spatial_heads: int = 2,
        temporal_heads: int = 4,
        dim_ff: int = 256,
        spatial_dropout: float = 0.3,
        temporal_dropout: float = 0.1,
        edge_dim: int = 1,
    ):
        super().__init__()

        # d_model = hidden_dim * spatial_heads (spatial output dimension)
        self.d_model = hidden_dim * spatial_heads  # 64 with defaults

        # Stage 1: shared spatial block (same weights applied to each snapshot)
        self.spatial = SpatialGATv2Block(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heads=spatial_heads,
            dropout=spatial_dropout,
            edge_dim=edge_dim,
        )

        # Stage 2: temporal attention block
        self.temporal = TemporalAttentionBlock(
            d_model=self.d_model,
            num_heads=temporal_heads,
            dim_ff=dim_ff,
            dropout=temporal_dropout,
        )

        # Stage 3: classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(spatial_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, snapshots: List[Data]) -> Tensor:
        """
        Forward pass over T graph snapshots.

        Args:
            snapshots: List of T PyG Data objects, one per weekly snapshot.
                       Each Data must have:
                           .x          — node features,    shape (N, input_dim)
                           .edge_index — COO edge indices, shape (2, E_t)
                           .edge_attr  — DCC correlations, shape (E_t, 1)

                       All snapshots must have the same N nodes in the same order.

        Returns:
            Shock probabilities, shape (N,). Values in (0, 1).
            p_i = P(stock i has |return| > 3% in next 5 trading days).
        """
        T = len(snapshots)
        if T == 0:
            raise ValueError("snapshots list is empty")

        # -----------------------------------------------------------------
        # Stage 1: Spatial attention on each snapshot
        # spatial_embeds: list of T tensors, each shape (N, d_model)
        # -----------------------------------------------------------------
        spatial_embeds = []
        for snap in snapshots:
            z_t = self.spatial(snap.x, snap.edge_index, snap.edge_attr)  # (N, d_model)
            spatial_embeds.append(z_t)

        # Stack along time dimension: (N, T, d_model)
        Z = torch.stack(spatial_embeds, dim=1)   # (N, T, d_model)

        # -----------------------------------------------------------------
        # Stage 2: Temporal attention (transformer over T)
        # Each node attends to its own history of T embeddings
        # H: (N, T, d_model)
        # -----------------------------------------------------------------
        H = self.temporal(Z)   # (N, T, d_model)

        # -----------------------------------------------------------------
        # Readout: take the last temporal position (most recent week)
        # This represents the node's current state informed by all T weeks
        # Rationale: task is "predict T+1", most recent state dominates
        # -----------------------------------------------------------------
        h_final = H[:, -1, :]   # (N, d_model) — last time step

        # -----------------------------------------------------------------
        # Stage 3: Classification
        # -----------------------------------------------------------------
        logits = self.classifier(h_final)   # (N, 1)
        probs = torch.sigmoid(logits).squeeze(-1)   # (N,)

        return probs

    def get_temporal_attention_weights(self, snapshots: List[Data]):
        """
        Run forward pass and return temporal attention weights for interpretability.

        Use this in the GNNExplainer comparison study to see which weeks
        the model "focused on" when making each node's prediction.

        Returns:
            probs:    Shock probabilities, shape (N,).
            attn_map: Attention weights averaged across heads, shape (N, T, T).
                      attn_map[i, t_query, t_key] = how much node i's
                      query at position t_query attended to position t_key.
        """
        T = len(snapshots)

        # Spatial stage
        spatial_embeds = []
        for snap in snapshots:
            z_t = self.spatial(snap.x, snap.edge_index, snap.edge_attr)
            spatial_embeds.append(z_t)
        Z = torch.stack(spatial_embeds, dim=1)   # (N, T, d_model)

        # Positional encoding
        Z = self.temporal.pos_enc(Z)

        # Forward through attention, capturing weights
        Z_norm = self.temporal.norm1(Z)
        # need_weights=True returns attention weights averaged across heads
        attn_out, attn_weights = self.temporal.attn(
            Z_norm, Z_norm, Z_norm,
            need_weights=True,
            average_attn_weights=True,   # average across heads → (N, T, T)
        )
        Z = Z + self.temporal.dropout(attn_out)
        Z_norm = self.temporal.norm2(Z)
        ffn_out = self.temporal.ffn(Z_norm)
        H = Z + self.temporal.dropout(ffn_out)

        h_final = H[:, -1, :]
        logits = self.classifier(h_final)
        probs = torch.sigmoid(logits).squeeze(-1)

        return probs, attn_weights   # (N,), (N, T, T)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check — run this directly to verify shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("TemporalGATv2 — Shape Sanity Check")
    print("=" * 60)

    # Hyperparameters
    N = 50              # number of stocks (nodes)
    T = 12              # weekly snapshots
    num_features = 14   # node features (named to avoid shadowing F = nn.functional)
    E = 120             # edges per snapshot (approx)
    hidden_dim = 32
    spatial_heads = 2
    temporal_heads = 4

    # Build fake snapshot list
    snapshots = []
    for t in range(T):
        # Random node features
        x = torch.randn(N, num_features)
        # Random edges (COO format)
        edge_index = torch.randint(0, N, (2, E))
        # Random DCC correlation weights
        edge_attr = torch.rand(E, 1)
        snapshots.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    # Build model
    model = TemporalGATv2(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        spatial_heads=spatial_heads,
        temporal_heads=temporal_heads,
    )
    model.eval()

    print(f"\nModel config:")
    print(f"  input_dim     = {num_features}")
    print(f"  hidden_dim    = {hidden_dim}")
    print(f"  spatial_heads = {spatial_heads}  →  d_model = {hidden_dim * spatial_heads}")
    print(f"  temporal_heads= {temporal_heads}")
    print(f"  T (snapshots) = {T}")
    print(f"  N (stocks)    = {N}")
    print(f"  Trainable params: {model.count_parameters():,}")

    # Forward pass
    with torch.no_grad():
        probs = model(snapshots)

    print(f"\n✓ Output shape: {probs.shape}  (expected: torch.Size([{N}]))")
    assert probs.shape == (N,), f"SHAPE MISMATCH: {probs.shape} != ({N},)"
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of [0, 1] range"

    print(f"✓ Output range: [{probs.min():.4f}, {probs.max():.4f}]  (expected: in [0, 1])")

    # Test attention weight extraction
    probs2, attn = model.get_temporal_attention_weights(snapshots)
    print(f"✓ Attention weights shape: {attn.shape}  (expected: torch.Size([{N}, {T}, {T}]))")

    print("\n✓ All assertions passed — model is ready for Days 6-7 (DCC-GARCH data).")

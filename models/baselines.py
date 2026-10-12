"""
Single-snapshot models used as baselines (and as the deployed app model).

All models share one interface:
    forward(x, edge_index, edge_attr=None) -> probabilities, shape (N,)
so experiments can swap them freely. MLPNet ignores the graph.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv


class StaticGATv2(nn.Module):
    """
    2-layer GATv2 with edge weights in the attention score
    (the "OptimalGNN" architecture: 2 heads → 1 head, hidden 32, BN + ELU).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=2, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=1, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr: Optional[torch.Tensor] = None):
        h = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index, edge_attr=edge_attr)))
        return torch.sigmoid(self.head(h)).squeeze(-1)


class GATv1Net(nn.Module):
    """Original GAT (no edge features)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        return torch.sigmoid(self.head(h)).squeeze(-1)


class GCNNet(nn.Module):
    """2-layer GCN; edge weights used as GCN edge_weight."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        w = edge_attr.view(-1) if edge_attr is not None else None
        h = F.relu(self.bn1(self.conv1(x, edge_index, w)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index, w)))
        return torch.sigmoid(self.head(h)).squeeze(-1)


class SAGENet(nn.Module):
    """2-layer GraphSAGE (mean aggregation, unweighted)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return torch.sigmoid(self.head(h)).squeeze(-1)


class MLPNet(nn.Module):
    """Per-node MLP on the same features — no message passing."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

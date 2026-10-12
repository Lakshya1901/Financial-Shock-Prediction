import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from analysis.attention_attribution import EdgeImportanceAnalyzer, TemporalAttentionAnalyzer
from experiments.training import best_f1_threshold, compute_metrics, predict, train_model
from models.baselines import GATv1Net, GCNNet, MLPNet, SAGENet, StaticGATv2
from models.temporal_gatv2 import TemporalGATv2

N, F, T = 12, 14, 4


def _snap(seed):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, N, (2, 30), generator=g)
    ei = ei[:, ei[0] != ei[1]]
    return Data(x=torch.randn(N, F, generator=g), edge_index=ei, edge_attr=torch.rand(ei.shape[1], 1, generator=g))


def _item(seed):
    snaps = [_snap(seed * 10 + t) for t in range(T)]
    y = (snaps[-1].x[:, 1] > 0).float()  # learnable from features
    mask = torch.ones(N, dtype=torch.bool)
    return {"snaps": snaps, "y": y, "train_mask": mask, "eval_mask": mask}


@pytest.mark.parametrize("cls", [StaticGATv2, GATv1Net, GCNNet, SAGENet, MLPNet])
def test_static_models_output_probabilities(cls):
    s = _snap(0)
    p = cls(F).eval()(s.x, s.edge_index, s.edge_attr)
    assert p.shape == (N,) and (p >= 0).all() and (p <= 1).all()


def test_temporal_model_and_attention_shapes():
    model = TemporalGATv2(input_dim=F).eval()
    snaps = _item(0)["snaps"]
    assert model(snaps).shape == (N,)
    maps = TemporalAttentionAnalyzer(model).get_attention_maps(snaps)
    assert maps["attention_weights"].shape == (N, T, T)
    assert np.allclose(maps["attention_weights"].sum(-1), 1.0, atol=1e-5)
    edges = EdgeImportanceAnalyzer(model)
    assert edges.compute_edge_importance(snaps).shape == (snaps[-1].edge_index.shape[1],)
    tickers = [f"T{i}" for i in range(N)]
    heat = edges.cross_market_edge_heatmap(snaps, tickers, {t: "SP500" for t in tickers})
    assert heat.shape == (4, 4)


def test_training_learns_and_restores_best_weights():
    torch.manual_seed(0)
    train = [_item(i) for i in range(60)]
    val = [_item(100 + i) for i in range(10)]
    model = TemporalGATv2(input_dim=F)
    hist = train_model(model, train, val, epochs=20, lr=5e-3, batch_size=8, patience=20)
    y, p = predict(model, val)
    assert len(y) == len(val) * N
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(y, p) - max(hist["val_roc_auc"])) < 1e-6
    assert max(hist["val_roc_auc"]) > 0.7


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    t = best_f1_threshold(y, p)
    m = compute_metrics(y, p, t)
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["base_rate"] == 0.5

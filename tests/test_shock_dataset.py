import numpy as np
import pytest

from data.shock_dataset import SHOCK_FLAG, DatasetConfig, build_dataset

CFG = DatasetConfig(T=4, step=5, corr_window=60, top_k=3)


@pytest.fixture(scope="module")
def ds():
    from tests.conftest import make_panel
    return build_dataset(CFG, panel=make_panel(), verbose=False)


def test_splits_are_chronological_with_purge(ds):
    ev = ds.events
    last_train = ev[ev.split == "train"].date_idx.max()
    first_val, last_val = ev[ev.split == "val"].date_idx.min(), ev[ev.split == "val"].date_idx.max()
    first_test = ev[ev.split == "test"].date_idx.min()
    assert first_val > last_train + CFG.horizon
    assert first_test > last_val + CFG.horizon


def test_sample_shapes_and_masks(ds):
    row = next(ds.events.itertuples())
    item = ds.sample(row, "dcc", "all")
    n = len(ds.tickers)
    assert len(item["snaps"]) == CFG.T
    assert all(s.x.shape == (n, 14) for s in item["snaps"])
    # shock flag only on the shock-date snapshot, only on the source
    assert item["snaps"][-1].x[:, SHOCK_FLAG].sum() == 1
    assert all(s.x[:, SHOCK_FLAG].sum() == 0 for s in item["snaps"][:-1])
    src = row.src_idx
    assert not item["eval_mask"][src] and not item["train_mask"][src]
    # only US stocks are scored
    us = np.array([m == "SP500" for m in ds.markets])
    assert not item["eval_mask"].numpy()[~us].any()


def test_us_node_set_excludes_foreign_nodes(ds):
    row = next(ds.events.itertuples())
    item = ds.sample(row, "pearson", "us", temporal=False)
    assert len(item["snaps"]) == 1
    assert item["snaps"][0].x.shape[0] == len(ds.node_sets["us"])


def test_features_have_no_lookahead():
    """Altering prices after day d must not change features or edges at d."""
    from tests.conftest import make_panel
    p1 = make_panel()
    p2 = make_panel()
    d = 250
    p2["close"].iloc[d + 1:] *= 1.5
    a = build_dataset(CFG, panel=p1, verbose=False)
    b = build_dataset(CFG, panel=p2, verbose=False)
    assert np.allclose(a.raw_vol[d], b.raw_vol[d])

import numpy as np

from data.dynamic_correlation import DCCGARCHEstimator
from tests.conftest import make_panel


def _returns(n_days=500):
    return make_panel(n_days=n_days)["close"].pct_change().dropna()


def test_fit_gives_stationary_params():
    est = DCCGARCHEstimator(min_obs=60).fit(_returns())
    assert est.dcc_a_ > 0 and est.dcc_b_ > 0 and est.dcc_a_ + est.dcc_b_ < 1


def test_correlation_path_is_valid_correlation():
    r = _returns()
    est = DCCGARCHEstimator(min_obs=60).fit(r.iloc[:300])
    mats = [R for _, R in est.correlation_path(r)]
    assert len(mats) == len(r)
    R = mats[-1]
    assert np.allclose(R, R.T)
    assert np.allclose(np.diag(R), 1.0)
    assert np.linalg.eigvalsh(R).min() > -1e-8


def test_correlation_path_has_no_lookahead():
    """Changing returns after day t must not change the correlation known at day t."""
    r = _returns()
    est = DCCGARCHEstimator(min_obs=60).fit(r.iloc[:300])
    t = 350
    before = dict(est.correlation_path(r))[r.index[t]]
    r2 = r.copy()
    r2.iloc[t + 1:] *= -3.0
    after = dict(est.correlation_path(r2))[r.index[t]]
    assert np.allclose(before, after)


def test_composite_likelihood_used_for_many_assets():
    est = DCCGARCHEstimator(min_obs=60, composite_above=5).fit(_returns())
    assert 0 < est.dcc_a_ < 0.5

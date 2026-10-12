"""
DCC-GARCH Dynamic Correlation Estimator for the multi-market model.

Implements the two-step Dynamic Conditional Correlation estimator
(Engle, 2002) to produce time-varying correlation matrices from
raw return time series. These matrices become the edge weights in the
TemporalGATv2 graph snapshots.

Mathematical overview
---------------------
The DCC(1,1)-GARCH(1,1) model has two stages:

    Stage 1 — Univariate GARCH per stock (using `arch` library):
        ε_i,t   = σ_i,t · z_i,t,        z_i,t ~ N(0, 1)
        σ²_i,t  = ω_i + α_i · ε²_i,t-1 + β_i · σ²_i,t-1

        Yields standardized residuals z_i,t for each stock i at each day t.

    Stage 2 — DCC dynamics on the standardized residuals:
        Q̄       = (1/T) Σ_t  z_t z_t'      (unconditional cov of std residuals)
        Q_t     = (1 − a − b) · Q̄  +  a · z_{t-1} z'_{t-1}  +  b · Q_{t-1}
        R_t     = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

        R_t is the time-varying correlation matrix.
        Parameters a ≥ 0, b ≥ 0, a + b < 1 (stationarity constraint).

    DCC log-likelihood (Engle 2002, eq. 11):
        L_DCC = −½ Σ_t [log|R_t| + z_t' R_t^{-1} z_t − z_t' z_t]

    With many assets the full likelihood is biased toward a → 0, so for
    N > 20 (a, b) are estimated with the pairwise composite likelihood of
    Pakel, Shephard, Sheppard & Engle (2021).

Reference:
    Engle, R. (2002). Dynamic Conditional Correlations: A Simple Class of
    Multivariate Generalized Autoregressive Conditional Heteroskedasticity
    Models. Journal of Business & Economic Statistics, 20(3), 339–350.
    Pakel, C., Shephard, N., Sheppard, K., & Engle, R. (2021). Fitting vast
    dimensional time-varying covariance models. Journal of Business &
    Economic Statistics, 39(3), 652–668. (composite likelihood)

Usage:
    >>> from data.dynamic_correlation import DCCGARCHEstimator
    >>> est = DCCGARCHEstimator()
    >>> est.fit(returns_df)                         # returns_df: (T_days, N) DataFrame
    >>> corr_list = est.get_correlation_sequence(T=12, freq="W")  # 12 weekly matrices
    >>> adj = est.to_adjacency(corr_list[0], threshold=0.4)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

# scipy.optimize is preferred (L-BFGS-B) but we fall back to a pure-numpy
# Nelder-Mead if scipy is unavailable or broken.
try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    _scipy_minimize = None

logger = logging.getLogger(__name__)

# Suppress convergence warnings from arch during batch fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*convergence.*")


class DCCGARCHEstimator:
    """
    Two-step DCC(1,1)-GARCH(1,1) estimator for N-stock return panels.

    Workflow:
        1. Call fit(returns_df)         — estimates GARCH params per stock
                                          and DCC params (a, b).
        2. Call get_correlation_sequence — returns T weekly R_t matrices.
        3. Call to_adjacency            — thresholds R_t to a sparse graph.

    Args:
        garch_p: GARCH lag order for squared residuals (default 1).
        garch_q: GARCH lag order for variance           (default 1).
        dcc_max_iter: Max iterations for DCC likelihood optimization.
        min_obs: Minimum daily observations required per stock to attempt fitting.
        fallback_corr: If a stock's GARCH fails, use Pearson correlation instead
                       of dropping it. Default True — keeps graph connected.
        composite_above: Use the pairwise composite likelihood for DCC
                       estimation when there are more assets than this.

    Attributes after fit():
        std_residuals_:  pd.DataFrame of shape (T_days, N) — z_i,t values.
        Q_bar_:          np.ndarray (N, N) — unconditional cov of std residuals.
        dcc_a_:          float — estimated DCC alpha parameter.
        dcc_b_:          float — estimated DCC beta parameter.
        fitted_tickers_: list of tickers that were successfully fitted.
    """

    def __init__(
        self,
        garch_p: int = 1,
        garch_q: int = 1,
        dcc_max_iter: int = 200,
        min_obs: int = 60,
        fallback_corr: bool = True,
        composite_above: int = 20,
    ):
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.dcc_max_iter = dcc_max_iter
        self.min_obs = min_obs
        self.fallback_corr = fallback_corr
        self.composite_above = composite_above

        # Set after fit()
        self.std_residuals_: Optional[pd.DataFrame] = None
        self.Q_bar_: Optional[np.ndarray] = None
        self.dcc_a_: float = 0.05
        self.dcc_b_: float = 0.90
        self.fitted_tickers_: List[str] = []
        self.garch_params_: dict = {}   # ticker -> arch params, or (mu, sigma) for z-score fallback
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Stage 1: Univariate GARCH fitting
    # ------------------------------------------------------------------

    def _fit_garch_single(
        self, returns: pd.Series
    ) -> Tuple[pd.Series, bool]:
        """
        Fit GARCH(1,1) to a single return series and return standardized residuals.

        The GARCH(1,1) variance equation:
            σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

        Standardized residuals:  z_t = ε_t / σ_t

        Args:
            returns: Daily return series for one stock.

        Returns:
            (std_residuals, success_flag)
            If GARCH fails (too few obs, convergence failure), returns
            (zscore-normalized returns, False) as a fallback.
        """
        try:
            from arch import arch_model   # import here so module loads without arch

            # arch expects percentage returns (multiply by 100 for numerical stability)
            r = returns.dropna() * 100

            if len(r) < self.min_obs:
                raise ValueError(f"Only {len(r)} obs, need {self.min_obs}")

            # Fit GARCH(1,1) with Normal innovations
            am = arch_model(
                r,
                mean="Zero",      # assume zero-mean returns (demeaned externally)
                vol="GARCH",
                p=self.garch_p,
                q=self.garch_q,
                dist="Normal",
            )
            res = am.fit(disp="off", show_warning=False)
            self.garch_params_[returns.name] = res.params

            # Conditional volatility (σ_t) estimated by the model
            cond_vol = res.conditional_volatility  # in % units

            # Standardized residuals: z_t = ε_t / σ_t
            # res.resid is the raw residual (= return in "Zero mean" model)
            std_resid = res.resid / cond_vol

            # Align to original index
            std_resid = std_resid.reindex(returns.index).fillna(0.0)
            return std_resid, True

        except Exception as e:
            logger.debug(f"GARCH failed: {e}. Using z-score fallback.")

            if self.fallback_corr:
                # Fallback: z-score normalize the raw returns
                r = returns.dropna()
                mu, sigma = r.mean(), r.std()
                self.garch_params_[returns.name] = (mu, sigma)
                if sigma > 0:
                    z = (r - mu) / sigma
                else:
                    z = r * 0.0
                return z.reindex(returns.index).fillna(0.0), False
            else:
                return pd.Series(dtype=float), False

    def _fit_all_garch(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit GARCH(1,1) to each column (stock) in the returns DataFrame.

        Returns:
            std_residuals: DataFrame (T_days, N) of standardized residuals z_i,t.
        """
        std_residuals = {}
        failed = []

        for ticker in returns_df.columns:
            series = returns_df[ticker].dropna()
            z, success = self._fit_garch_single(series)

            if len(z) > 0:
                std_residuals[ticker] = z
                if not success:
                    failed.append(ticker)
            else:
                failed.append(ticker)
                logger.warning(f"Dropping {ticker}: GARCH and fallback both failed.")

        if failed:
            n_garch_fail = sum(1 for t in failed if t in returns_df.columns)
            logger.info(
                f"GARCH fitted successfully for "
                f"{len(std_residuals)}/{len(returns_df.columns)} stocks. "
                f"Fallback z-score used for: {[t for t in failed if t in std_residuals]}"
            )

        # Align all residuals to a common date index
        df = pd.DataFrame(std_residuals)
        df = df.reindex(returns_df.index).fillna(0.0)
        return df

    # ------------------------------------------------------------------
    # Stage 2: DCC parameter estimation
    # ------------------------------------------------------------------

    def _dcc_neg_loglik(
        self, params: np.ndarray, Z: np.ndarray, Q_bar: np.ndarray
    ) -> float:
        """
        DCC negative log-likelihood for optimization.

        The DCC log-likelihood (Engle 2002, eq. 11):
            L_DCC = -½ Σ_t [log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t]

        We maximize L_DCC (i.e., minimize -L_DCC) over (a, b).

        Args:
            params:  [a, b] — DCC parameters.
            Z:       np.ndarray (T, N) — standardized residuals.
            Q_bar:   np.ndarray (N, N) — unconditional correlation.

        Returns:
            Negative log-likelihood (scalar float).
        """
        a, b = params

        # Constraint: a >= 0, b >= 0, a + b < 1
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10   # infeasible

        T, N = Z.shape
        Q_t = Q_bar.copy()
        loglik = 0.0

        for t in range(1, T):
            z_t = Z[t]   # shape (N,)

            # DCC recursion:
            # Q_t = (1 - a - b) * Q̄ + a * z_{t-1} z'_{t-1} + b * Q_{t-1}
            Q_t = (1 - a - b) * Q_bar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q_t

            # Normalize Q_t to get R_t (correlation matrix):
            # R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}
            q_diag = np.diag(Q_t)
            q_diag_inv_sqrt = np.where(q_diag > 0, 1.0 / np.sqrt(q_diag), 0.0)
            R_t = q_diag_inv_sqrt[:, None] * Q_t * q_diag_inv_sqrt[None, :]

            # Ensure R_t is numerically symmetric and positive definite
            R_t = (R_t + R_t.T) / 2
            np.fill_diagonal(R_t, 1.0)

            try:
                # log|R_t| via Cholesky (faster and more stable than np.linalg.det)
                L = np.linalg.cholesky(R_t)
                log_det_R = 2 * np.sum(np.log(np.diag(L)))

                # z_t' R_t^{-1} z_t via solving: L x = z_t, then ||x||²
                x = np.linalg.solve(L, z_t)
                quadratic = np.dot(x, x)

                # Subtract z_t' z_t to get the DCC-specific part of the likelihood
                loglik -= 0.5 * (log_det_R + quadratic - np.dot(z_t, z_t))

            except np.linalg.LinAlgError:
                # Matrix not positive definite — add regularization and skip
                return 1e10

        return -loglik   # return negative for minimization

    def _dcc_neg_composite_loglik(
        self, params: np.ndarray, Z: np.ndarray, Q_bar: np.ndarray
    ) -> float:
        """
        Composite likelihood over contiguous pairs (i, i+1) — the "2MSCLE" of
        Pakel, Shephard, Sheppard & Engle (2021). With many assets the full DCC
        likelihood drives a → 0 (a known bias); summing bivariate likelihoods
        avoids it and costs O(N) per step instead of O(N^3).
        """
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        i = np.arange(Z.shape[1] - 1)
        j = i + 1
        qbar_ii, qbar_jj, qbar_ij = Q_bar[i, i], Q_bar[j, j], Q_bar[i, j]
        q_ii, q_jj, q_ij = qbar_ii.copy(), qbar_jj.copy(), qbar_ij.copy()
        loglik = 0.0
        for t in range(1, Z.shape[0]):
            zi, zj = Z[t - 1, i], Z[t - 1, j]
            q_ii = (1 - a - b) * qbar_ii + a * zi * zi + b * q_ii
            q_jj = (1 - a - b) * qbar_jj + a * zj * zj + b * q_jj
            q_ij = (1 - a - b) * qbar_ij + a * zi * zj + b * q_ij
            rho = np.clip(q_ij / np.sqrt(q_ii * q_jj), -0.999, 0.999)
            xi, xj = Z[t, i], Z[t, j]
            one_m = 1 - rho ** 2
            loglik -= 0.5 * np.sum(np.log(one_m) + (xi ** 2 + xj ** 2 - 2 * rho * xi * xj) / one_m
                                   - xi ** 2 - xj ** 2)
        return -loglik

    def _estimate_dcc_params(
        self, Z: np.ndarray, Q_bar: np.ndarray
    ) -> Tuple[float, float]:
        """
        Maximize DCC log-likelihood over (a, b).

        Coarse grid search over (a, b), then L-BFGS-B refinement from the best
        grid point when scipy is available. Uses the composite likelihood when
        there are more than `composite_above` assets.

        Returns:
            (a_hat, b_hat) — estimated DCC parameters.
        """
        # Coarse grid first: the likelihood is flat in places and returns a
        # 1e10 penalty outside the feasible region, so a local optimizer started
        # blindly can stall on the penalty plateau. The grid gives a safe warm start.
        objective = (self._dcc_neg_composite_loglik if Z.shape[1] > self.composite_above
                     else self._dcc_neg_loglik)
        best_ll, a_hat, b_hat = np.inf, 0.05, 0.90
        for a in (0.002, 0.005, 0.01, 0.02, 0.05):
            for b in (0.80, 0.90, 0.95, 0.97, 0.99):
                if a + b >= 1.0:
                    continue
                ll = objective(np.array([a, b]), Z, Q_bar)
                if ll < best_ll:
                    best_ll, a_hat, b_hat = ll, a, b

        if _SCIPY_AVAILABLE:
            # Refine with L-BFGS-B; keep it only if it actually improves on the grid
            result = _scipy_minimize(
                objective,
                np.array([a_hat, b_hat]),
                args=(Z, Q_bar),
                method="L-BFGS-B",
                bounds=[(1e-4, 0.49), (1e-4, 0.995)],
                options={"maxiter": self.dcc_max_iter, "ftol": 1e-9},
            )
            if result.fun < best_ll:
                a_hat, b_hat = result.x

        # Final stationarity check
        if a_hat + b_hat >= 1.0:
            logger.warning(
                f"DCC stationarity violated (a+b={a_hat+b_hat:.3f}). "
                f"Clamping b to ensure a+b < 1."
            )
            b_hat = max(0.01, 0.99 - a_hat)

        logger.info(f"DCC estimated: a={a_hat:.4f}, b={b_hat:.4f}")
        return float(a_hat), float(b_hat)

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------

    def fit(self, returns_df: pd.DataFrame) -> "DCCGARCHEstimator":
        """
        Fit the full DCC-GARCH model to historical return data.

        This is the one-time offline estimation step. After fitting, call
        get_correlation_sequence() to extract time-varying correlations.

        Args:
            returns_df: pd.DataFrame of shape (T_days, N) with daily
                        log-returns or simple returns. Index must be datetime.
                        Columns are stock tickers. NaNs are handled gracefully.

        Returns:
            self (for method chaining).

        Example:
            >>> est = DCCGARCHEstimator()
            >>> est.fit(returns_df)
            >>> corr_list = est.get_correlation_sequence(T=12)
        """
        if returns_df.empty:
            raise ValueError("returns_df is empty.")

        # Step 1: fit GARCH(1,1) per stock → standardized residuals z_i,t
        print(f"[DCC-GARCH] Stage 1: fitting GARCH(1,1) to {returns_df.shape[1]} stocks...")
        std_residuals = self._fit_all_garch(returns_df)

        # Drop columns with all zeros (failed stocks with no fallback)
        valid_cols = std_residuals.columns[std_residuals.std() > 0].tolist()
        if len(valid_cols) < 2:
            raise RuntimeError(
                f"Too few valid stocks ({len(valid_cols)}) after GARCH fitting. "
                f"Check your returns data."
            )
        std_residuals = std_residuals[valid_cols]
        self.fitted_tickers_ = valid_cols

        # Convert to numpy for the DCC stage
        Z = std_residuals.dropna().values   # (T_valid, N)

        # Step 2: compute Q̄ = unconditional covariance of standardized residuals
        # This is the "target" correlation that Q_t mean-reverts to
        # Note: we use np.cov (not np.corrcoef) because the DCC normalization
        # step (diag(Q_t)^{-1/2}) converts Q_t into a proper correlation matrix.
        self.Q_bar_ = np.cov(Z.T)   # (N, N)

        print(f"[DCC-GARCH] Stage 2: estimating DCC(1,1) parameters...")
        self.dcc_a_, self.dcc_b_ = self._estimate_dcc_params(Z, self.Q_bar_)

        self.std_residuals_ = std_residuals
        self._is_fitted = True
        print(f"[DCC-GARCH] Fitting complete. a={self.dcc_a_:.4f}, b={self.dcc_b_:.4f}")
        return self

    # ------------------------------------------------------------------
    # DCC filter: produce daily R_t sequence
    # ------------------------------------------------------------------

    def _filter_correlations(self) -> List[np.ndarray]:
        """
        Run the DCC filter on the fitted residuals to produce daily R_t matrices.

        Uses the estimated (a, b) and Q̄ to compute:
            Q_t = (1 - a - b) * Q̄ + a * z_{t-1} z'_{t-1} + b * Q_{t-1}
            R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

        Returns:
            List of T daily correlation matrices R_t, each shape (N, N).
        """
        assert self._is_fitted, "Call fit() first."

        Z = self.std_residuals_.values   # (T, N)
        T, N = Z.shape
        a, b = self.dcc_a_, self.dcc_b_
        Q_bar = self.Q_bar_

        R_sequence = []
        Q_t = Q_bar.copy()

        for t in range(T):
            if t > 0:
                # DCC recursion
                Q_t = (1 - a - b) * Q_bar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q_t

            # Convert Q_t → R_t (normalize to unit diagonal)
            q_diag = np.diag(Q_t)
            q_diag_inv_sqrt = np.where(q_diag > 0, 1.0 / np.sqrt(q_diag), 0.0)
            R_t = q_diag_inv_sqrt[:, None] * Q_t * q_diag_inv_sqrt[None, :]

            # Symmetrize and clip to valid correlation range
            R_t = (R_t + R_t.T) / 2
            np.fill_diagonal(R_t, 1.0)
            R_t = np.clip(R_t, -1.0, 1.0)

            R_sequence.append(R_t)

        return R_sequence

    # ------------------------------------------------------------------
    # Public API: get_correlation_sequence
    # ------------------------------------------------------------------

    def get_correlation_sequence(
        self,
        T: int = 12,
        freq: str = "W",
        agg: str = "mean",
    ) -> List[np.ndarray]:
        """
        Return T aggregated correlation matrices (one per week).

        Daily R_t matrices from the DCC filter are aggregated to weekly
        by taking the mean (or last) correlation within each calendar week.

        Args:
            T:    Number of snapshots to return. Returned list has length T.
                  If there are fewer calendar weeks in the data than T,
                  the available weeks are returned (may be < T).
            freq: Pandas frequency string for resampling. "W" = weekly
                  (week-ending Friday), "M" = monthly.
            agg:  How to aggregate daily correlations within a period.
                  "mean" averages the R_t matrices. "last" takes the final day.

        Returns:
            List of up to T np.ndarray matrices, each shape (N, N).
            Entry k is the correlation matrix for week k.

        Example:
            >>> corr_list = est.get_correlation_sequence(T=12, freq="W")
            >>> len(corr_list)   # ≤ 12
            12
            >>> corr_list[0].shape
            (N, N)
        """
        assert self._is_fitted, "Call fit() first."

        # Get daily R_t sequence
        daily_R = self._filter_correlations()   # list of (N, N) arrays
        daily_dates = self.std_residuals_.index

        if len(daily_R) != len(daily_dates):
            raise RuntimeError("Mismatch between R_t count and date index length.")

        # Build a 3D array: (T_days, N, N)
        N = self.Q_bar_.shape[0]
        R_3d = np.stack(daily_R, axis=0)   # (T_days, N, N)

        # Resample to weekly using pandas DatetimeIndex grouping
        # We flatten to (T_days, N*N), resample, then reshape back
        df_flat = pd.DataFrame(
            R_3d.reshape(len(daily_dates), N * N),
            index=daily_dates,
        )

        if agg == "mean":
            weekly_flat = df_flat.resample(freq).mean().dropna()
        elif agg == "last":
            weekly_flat = df_flat.resample(freq).last().dropna()
        else:
            raise ValueError(f"agg must be 'mean' or 'last', got '{agg}'")

        # Take the last T weeks (most recent)
        weekly_flat = weekly_flat.tail(T)

        # Reshape back to list of (N, N) matrices
        weekly_R = []
        for _, row in weekly_flat.iterrows():
            R_week = row.values.reshape(N, N)
            # Re-clip and symmetrize after averaging (mean can push out of range)
            R_week = (R_week + R_week.T) / 2
            np.fill_diagonal(R_week, 1.0)
            R_week = np.clip(R_week, -1.0, 1.0)
            weekly_R.append(R_week)

        return weekly_R

    # ------------------------------------------------------------------
    # Public API: out-of-sample filtering
    # ------------------------------------------------------------------

    def _apply_garch(self, ticker: str, returns: pd.Series) -> pd.Series:
        """Standardized residuals for `returns` using the GARCH params fitted in fit()."""
        params = self.garch_params_[ticker]
        if isinstance(params, tuple):
            mu, sigma = params
            return ((returns - mu) / sigma) if sigma > 0 else returns * 0.0
        from arch import arch_model
        am = arch_model(returns * 100, mean="Zero", vol="GARCH",
                        p=self.garch_p, q=self.garch_q, dist="Normal")
        res = am.fix(params)
        return res.resid / res.conditional_volatility

    def correlation_path(self, returns_df: pd.DataFrame):
        """
        Run the DCC filter over `returns_df` with the parameters estimated by
        fit() (GARCH params, Q̄, a, b are NOT re-estimated). This lets you fit on
        a training window and then filter through later data without look-ahead.

        Yields (date, R) where R is the correlation forecast for the next day
        given information up to and including `date`:
            Q_{t+1} = (1 - a - b) Q̄ + a z_t z_t' + b Q_t
        Columns must be a superset of fitted_tickers_; R is ordered like
        fitted_tickers_.
        """
        assert self._is_fitted, "Call fit() first."
        Z = pd.DataFrame({t: self._apply_garch(t, returns_df[t].fillna(0.0))
                          for t in self.fitted_tickers_}).fillna(0.0).values
        a, b, Q_bar = self.dcc_a_, self.dcc_b_, self.Q_bar_
        Q = Q_bar.copy()
        for t, date in enumerate(returns_df.index):
            Q = (1 - a - b) * Q_bar + a * np.outer(Z[t], Z[t]) + b * Q
            d = 1.0 / np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            R = np.clip(d[:, None] * Q * d[None, :], -1.0, 1.0)
            np.fill_diagonal(R, 1.0)
            yield date, R

    # ------------------------------------------------------------------
    # Public API: to_adjacency
    # ------------------------------------------------------------------

    def to_adjacency(
        self,
        corr_matrix: np.ndarray,
        threshold: float = 0.4,
        absolute: bool = True,
        tickers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        """
        Threshold a correlation matrix to produce an adjacency representation
        suitable for use as PyG edge_index + edge_attr.

        Edges are created between stocks i and j if |corr(i,j)| > threshold.
        The edge weight is the absolute correlation value (or signed if absolute=False).

        Args:
            corr_matrix: np.ndarray (N, N) — output from get_correlation_sequence.
            threshold:   Minimum |correlation| to create an edge. Default 0.4.
                         Lower = denser graph, higher = sparser graph.
                         Typical values: 0.3 (dense) to 0.5 (sparse).
            absolute:    If True, edge weight = |corr|. If False, signed.
            tickers:     Optional list of N ticker strings for logging.

        Returns:
            (adj_dense, edge_list) where:
                adj_dense:  np.ndarray (N, N) — dense adjacency matrix (thresholded).
                edge_list:  list of (i, j, weight) tuples — PyG-compatible format.
                            Use edge_list to build PyG edge_index + edge_attr directly.

        Example:
            >>> adj, edges = est.to_adjacency(corr_list[0], threshold=0.4)
            >>> edge_index = torch.tensor([[e[0] for e in edges],
            ...                            [e[1] for e in edges]], dtype=torch.long)
            >>> edge_attr  = torch.tensor([[e[2]] for e in edges], dtype=torch.float)
        """
        N = corr_matrix.shape[0]
        edge_list = []
        adj_dense = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue   # skip self-loops (GATv2Conv adds them internally)

                corr_val = corr_matrix[i, j]
                weight = abs(corr_val) if absolute else corr_val

                if abs(corr_val) > threshold:
                    edge_list.append((i, j, float(weight)))
                    adj_dense[i, j] = float(weight)

        if tickers:
            logger.debug(
                f"Adjacency: {len(edge_list)} edges for {N} stocks "
                f"(threshold={threshold}, density={len(edge_list)/(N*(N-1)):.3f})"
            )

        return adj_dense, edge_list

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        if not self._is_fitted:
            return "DCCGARCHEstimator: not fitted yet."

        N = len(self.fitted_tickers_)
        T = len(self.std_residuals_) if self.std_residuals_ is not None else 0
        return (
            f"DCCGARCHEstimator (fitted)\n"
            f"  Stocks (N):   {N}\n"
            f"  Daily obs:    {T}\n"
            f"  DCC a:        {self.dcc_a_:.4f}\n"
            f"  DCC b:        {self.dcc_b_:.4f}\n"
            f"  Persistence:  {self.dcc_a_ + self.dcc_b_:.4f}  (< 1 = stationary)\n"
            f"  Tickers:      {self.fitted_tickers_[:5]}{'...' if N > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    print("=" * 60)
    print("DCCGARCHEstimator — Sanity Check")
    print("=" * 60)

    # Simulate 10 stocks, 504 daily returns (~2 years)
    np.random.seed(42)
    n_stocks = 10
    n_days = 504

    # Simulate correlated returns (simple factor model)
    market_factor = np.random.randn(n_days) * 0.01
    stock_returns = pd.DataFrame(
        {
            f"STOCK_{i}": market_factor * (0.5 + 0.5 * np.random.rand())
            + np.random.randn(n_days) * 0.015
            for i in range(n_stocks)
        },
        index=pd.date_range("2023-01-01", periods=n_days, freq="B"),
    )

    print(f"\nInput returns shape: {stock_returns.shape}")

    # Fit DCC-GARCH
    est = DCCGARCHEstimator(min_obs=60, dcc_max_iter=100)
    est.fit(stock_returns)

    print(f"\n{est.summary()}")

    # Get 12 weekly correlation matrices
    corr_sequence = est.get_correlation_sequence(T=12, freq="W")
    print(f"\n✓ Correlation sequence: {len(corr_sequence)} matrices")
    print(f"  Each matrix shape: {corr_sequence[0].shape}  (expected: ({n_stocks}, {n_stocks}))")

    # Check correlation properties
    R0 = corr_sequence[0]
    assert R0.shape == (n_stocks, n_stocks), f"Shape mismatch: {R0.shape}"
    assert np.allclose(np.diag(R0), 1.0), "Diagonal should be 1.0"
    assert np.all(np.abs(R0) <= 1.0 + 1e-6), "Correlations must be in [-1, 1]"
    assert np.allclose(R0, R0.T, atol=1e-6), "R must be symmetric"

    print(f"  Correlation range: [{R0.min():.3f}, {R0.max():.3f}]")
    print(f"✓ All correlation properties verified.")

    # Convert to adjacency
    adj, edges = est.to_adjacency(R0, threshold=0.4)
    print(f"\n✓ Adjacency (threshold=0.4): {len(edges)} edges, density={len(edges)/(n_stocks*(n_stocks-1)):.2f}")

    # Build PyG-compatible tensors
    if edges:
        edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([[e[2]] for e in edges], dtype=torch.float)
        print(f"  edge_index shape: {edge_index.shape}")
        print(f"  edge_attr  shape: {edge_attr.shape}")

    # Verify the full pipeline: DCC → PyG Data → TemporalGATv2
    print(f"\n--- Full pipeline test: DCC → PyG Data objects ---")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models.temporal_gatv2 import TemporalGATv2

    snapshots = []
    for R_week in corr_sequence:
        adj_w, edges_w = est.to_adjacency(R_week, threshold=0.3)
        if edges_w:
            ei = torch.tensor([[e[0], e[1]] for e in edges_w], dtype=torch.long).t().contiguous()
            ea = torch.tensor([[e[2]] for e in edges_w], dtype=torch.float)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 1), dtype=torch.float)

        x = torch.randn(n_stocks, 14)   # dummy node features
        snapshots.append(Data(x=x, edge_index=ei, edge_attr=ea))

    model = TemporalGATv2(input_dim=14)
    model.eval()
    with torch.no_grad():
        probs = model(snapshots)

    print(f"✓ TemporalGATv2 output shape: {probs.shape}  (expected: torch.Size([{n_stocks}]))")
    assert probs.shape == (n_stocks,)
    print("\n✓ Full pipeline DCC-GARCH → TemporalGATv2 verified.")

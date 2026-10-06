"""
model.py
========
Dual-model architecture:

    * MarginModel  -> predicts scoring margin  (drives SPREAD bets)
    * TotalModel   -> predicts combined points (drives OVER/UNDER bets)

Each is an **ensemble** of:

    1. A gradient-boosted tree regressor (LightGBM if installed, otherwise
       sklearn's HistGradientBoostingRegressor). Trees capture the non-linear
       interactions – e.g. a tired team at altitude against a fast opponent –
       that a pure ratings projection misses.
    2. A Bayesian Ridge regressor. College box-score data is high-variance and
       low signal-to-noise; a heavily regularised linear model is hard to beat,
       barely over-fits, and stabilises the ensemble. Bayesian Ridge also yields
       a predictive variance we can blend into the sigma estimate.

Why ensemble + ratings prior, not a single big model? The ratings projection
(`proj_margin`, `proj_total`) is already a strong, low-variance baseline. The
trees learn the *residual* corrections, the linear model anchors them, and
averaging the two cuts variance — which is the whole game when the irreducible
noise (~10-11 pts on a margin) dwarfs any edge you can extract.

Loss functions
--------------
* Point head: Huber loss (robust to blowout outliers) rather than plain MSE.
* Uncertainty: separate quantile heads (q=0.5 +/- ) OR empirical residual std
  from the walk-forward backtest. `predict_with_sigma` returns both mean and the
  sigma the edge layer needs to turn a projection into P(cover).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# LightGBM if available, else a strong sklearn fallback (quantile-capable).
try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except Exception:                                    # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_LGB = False


def _make_gbm(objective: str = "huber", quantile: float | None = None):
    """Construct a GBM regressor with sensible high-noise hyper-parameters."""
    if _HAS_LGB:
        params = dict(n_estimators=600, learning_rate=0.02, num_leaves=31,
                      max_depth=-1, min_child_samples=40, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=2.0, reg_alpha=0.5, n_jobs=-1, verbosity=-1)
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        else:
            params.update(objective="huber", alpha=0.9)
        return LGBMRegressor(**params)
    # sklearn fallback
    loss = "quantile" if quantile is not None else "squared_error"
    kw = dict(learning_rate=0.03, max_iter=600, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=2.0, early_stopping=False)
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(loss="squared_error", **kw)


@dataclass
class EnsembleRegressor:
    """GBM + Bayesian Ridge mean predictor, plus a quantile GBM pair for sigma."""
    feature_cols: list
    gbm: object = None
    ridge: BayesianRidge = None
    scaler: StandardScaler = None
    q_lo: object = None
    q_hi: object = None
    w_gbm: float = 0.6          # ensemble weight on the trees
    resid_sigma: float = 11.0   # fallback sigma if quantile heads are absent

    def fit(self, df: pd.DataFrame, target: str):
        X = df[self.feature_cols].to_numpy(dtype=float)
        y = df[target].to_numpy(dtype=float)

        self.gbm = _make_gbm()
        self.gbm.fit(X, y)

        self.scaler = StandardScaler().fit(X)
        self.ridge = BayesianRidge().fit(self.scaler.transform(X), y)

        # 16th/84th percentile heads ~ +/-1 sigma of a Normal.
        self.q_lo = _make_gbm(quantile=0.16).fit(X, y)
        self.q_hi = _make_gbm(quantile=0.84).fit(X, y)

        resid = y - self._mean(X)
        self.resid_sigma = float(np.std(resid))
        return self

    def _mean(self, X: np.ndarray) -> np.ndarray:
        g = self.gbm.predict(X)
        r = self.ridge.predict(self.scaler.transform(X))
        return self.w_gbm * g + (1 - self.w_gbm) * r

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._mean(df[self.feature_cols].to_numpy(dtype=float))

    def predict_with_sigma(self, df: pd.DataFrame):
        """Return (mean, sigma). Sigma = blend of quantile spread + residual std."""
        X = df[self.feature_cols].to_numpy(dtype=float)
        mean = self._mean(X)
        lo, hi = self.q_lo.predict(X), self.q_hi.predict(X)
        q_sigma = np.clip((hi - lo) / 2.0, 1e-6, None)
        # robust blend: don't let a single weird quantile row drive the bet
        sigma = 0.5 * q_sigma + 0.5 * self.resid_sigma
        return mean, sigma


@dataclass
class DualModel:
    margin: EnsembleRegressor
    total: EnsembleRegressor

    @classmethod
    def train(cls, train_df: pd.DataFrame, feature_cols: list,
              margin_col: str = "actual_margin",
              total_col: str = "actual_total") -> "DualModel":
        m = EnsembleRegressor(feature_cols=list(feature_cols)).fit(train_df, margin_col)
        t = EnsembleRegressor(feature_cols=list(feature_cols)).fit(train_df, total_col)
        return cls(margin=m, total=t)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        mm, ms = self.margin.predict_with_sigma(df)
        tm, ts = self.total.predict_with_sigma(df)
        out = df.copy()
        out["pred_margin"], out["sigma_margin"] = mm, ms
        out["pred_total"], out["sigma_total"] = tm, ts
        return out


def feature_importance(model: EnsembleRegressor) -> pd.DataFrame:
    """Tree feature importances (gain) for diagnostics, if available."""
    imp = getattr(model.gbm, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return (pd.DataFrame({"feature": model.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))

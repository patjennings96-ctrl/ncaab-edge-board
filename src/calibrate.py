"""
calibrate.py
============
Self-calibration of the Edge Board's projection from this season's results.

Every run logs, for each untipped game, the pieces of the projection
(data/history/components_<season>.csv):

    hca_pts   home-court points
    own_s     strength gap from our own ratings (points, neutral floor)
    kp_s      strength gap from the KenPom paste alone (blank if no fresh paste)
    own_t     total from our own ratings
    kp_t      total from the KenPom paste alone

Margin.  Our ratings squeeze teams together early in the season (they need a
stretch of ~1.6-1.7x in November); KenPom's do not (~1.1x). Stretching the
50/50 blend, as the board did until Nov 14 2026, over-stretched the KenPom half
and overrated favorites by ~1.7 points. Each source now gets its own weight:

    margin = hca_pts + a * own_s + b * kp_s

(a, b) are re-fit every run on the last FIT_DAYS days of finished games, with a
light pull toward the old 50/50 recipe (a = 0.8, b = 0.5) so a short or noisy
sample cannot swing them. As our ratings absorb real games, `a` earns weight on
its own. Without a fresh KenPom paste: margin = hca_pts + a_solo * own_s.

Total.  total = average of own_t and kp_t (own_t alone without KenPom)
        + level, the recent average miss (actual - projected). November is
played ~2 possessions a game faster than the season as a whole, so the level
starts at LEVEL_PRIOR and follows the last LEVEL_DAYS days from there.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import live

COMP_LOG = os.path.join(live.ROOT, "data", "history", "components_{season}.csv")

FIT_DAYS = 21                 # margin weights: finished games from the last N days
PRIOR_AB = (0.8, 0.5)         # the old recipe: 50/50 blend with our half stretched 1.6x
PRIOR_GAMES = 30              # how many games the pull toward PRIOR_AB is worth
SOLO_PRIOR, SOLO_GAMES = 1.6, 30
A_RANGE, B_RANGE = (0.0, 2.0), (0.0, 1.5)

LEVEL_DAYS = 14
LEVEL_PRIOR, LEVEL_GAMES = 5.0, 30


def log_components(rows: pd.DataFrame, season: int) -> None:
    """Keep the last pre-tip row per game."""
    if rows is None or rows.empty:
        return
    path = COMP_LOG.format(season=season)
    new = rows.copy()
    new["game_id"] = new["game_id"].astype(str)
    new["logged_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"game_id": str, "home": str, "away": str})
        started = pd.to_datetime(old["start"], utc=True, errors="coerce") <= pd.Timestamp.now(tz="UTC")
        old = old[started | ~old["game_id"].isin(new["game_id"])]
        new = pd.concat([old, new[~new["game_id"].isin(old["game_id"])]], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False)


def finished(season: int, games_raw: pd.DataFrame, before=None) -> pd.DataFrame:
    """Logged components joined to final scores, oriented to the logged home team."""
    path = COMP_LOG.format(season=season)
    if not os.path.exists(path) or games_raw is None or games_raw.empty:
        return pd.DataFrame()
    c = pd.read_csv(path, dtype={"game_id": str, "home": str, "away": str})
    g = games_raw.dropna(subset=["home_score", "away_score"]).copy()
    g["game_id"] = g["game_id"].astype(str)
    g = g.drop_duplicates("game_id", keep="last")
    d = c.merge(g[["game_id", "home_id", "home_score", "away_score"]], on="game_id")
    if d.empty:
        return d
    flip = d["home"].astype(str) != d["home_id"].astype(str)
    d["margin"] = np.where(flip, d["away_score"] - d["home_score"], d["home_score"] - d["away_score"])
    d["total_pts"] = d["home_score"] + d["away_score"]
    d["t"] = pd.to_datetime(d["start"], utc=True, errors="coerce")
    if before is not None:
        d = d[d["t"] < before]
    return d.dropna(subset=["t", "own_s"])


def _recent(d: pd.DataFrame, days: int, now=None) -> pd.DataFrame:
    if d.empty:
        return d
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    return d[d["t"] >= now - pd.Timedelta(days=days)]


def margin_weights(d: pd.DataFrame, now=None) -> dict:
    """(a, b) for margin = hca + a*own_s + b*kp_s, and a_solo for own ratings alone."""
    out = {"a": PRIOR_AB[0], "b": PRIOR_AB[1], "a_solo": SOLO_PRIOR, "n": 0}
    d = _recent(d, FIT_DAYS, now)
    if d.empty:
        return out
    y = (d["margin"] - d["hca_pts"]).values
    x = d["own_s"].values
    sxx = float((x ** 2).sum())
    lam = SOLO_GAMES * float(np.var(x)) if len(x) > 1 else 0.0
    if sxx + lam > 0:
        out["a_solo"] = float(np.clip((float((x * y).sum()) + lam * SOLO_PRIOR) / (sxx + lam), 1.0, 2.0))
    k = d.dropna(subset=["kp_s"])
    if len(k) >= 30:
        X = k[["own_s", "kp_s"]].values
        yk = (k["margin"] - k["hca_pts"]).values
        L = np.diag(PRIOR_GAMES * X.var(axis=0))
        ab = np.linalg.solve(X.T @ X + L, X.T @ yk + L @ np.array(PRIOR_AB))
        a, b = float(np.clip(ab[0], *A_RANGE)), float(np.clip(ab[1], *B_RANGE))
        if a != ab[0] or b != ab[1]:
            # one weight hit its limit: re-fit the other with it held there
            if a != ab[0]:
                r = yk - a * X[:, 0]
                b = float(np.clip((float((X[:, 1] * r).sum()) + L[1, 1] * PRIOR_AB[1]) / (float((X[:, 1] ** 2).sum()) + L[1, 1]), *B_RANGE))
            else:
                r = yk - b * X[:, 1]
                a = float(np.clip((float((X[:, 0] * r).sum()) + L[0, 0] * PRIOR_AB[0]) / (float((X[:, 0] ** 2).sum()) + L[0, 0]), *A_RANGE))
        out.update(a=a, b=b, n=int(len(k)))
    return out


def raw_total(own_t, kp_t):
    own_t, kp_t = pd.to_numeric(own_t, errors="coerce"), pd.to_numeric(kp_t, errors="coerce")
    return np.where(pd.isna(kp_t), own_t, 0.5 * own_t + 0.5 * kp_t)


def total_level(d: pd.DataFrame, now=None) -> dict:
    d = _recent(d, LEVEL_DAYS, now)
    if d.empty:
        return {"level": LEVEL_PRIOR, "n": 0}
    miss = d["total_pts"].values - raw_total(d["own_t"], d["kp_t"])
    miss = miss[~np.isnan(miss)]
    return {"level": float((miss.sum() + LEVEL_GAMES * LEVEL_PRIOR) / (len(miss) + LEVEL_GAMES)), "n": int(len(miss))}


def predict_margin(hca_pts, own_s, kp_s, w: dict):
    kp = pd.to_numeric(kp_s, errors="coerce")
    return np.where(pd.isna(kp), hca_pts + w["a_solo"] * own_s, hca_pts + w["a"] * own_s + w["b"] * kp)

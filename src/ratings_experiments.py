"""
ratings_experiments.py
======================
Walk-forward test of team-rating variants on a past season, graded against
the actual results and DraftKings closing lines.

Variants
  base      current production ratings (alpha 3, equal weight within season)
  cap       blowout cap: a game's efficiency margin is capped at +/-CAP per 100
            possessions before fitting (garbage time stops inflating ratings)
  recency   current-season games weighted 0.5 ** (days_ago / HALF_LIFE)
  cap+rec   both
  +tempo    any of the above with opponent-adjusted pace
            (possessions = mu + t_home + t_away, fitted like efficiency)
            instead of the log5 tempo product

    python ratings_experiments.py --season 2026
Writes data/history/ratings_variants_<season>.csv (one row per game, every variant).
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

import live
import ratings as R

warnings.filterwarnings("ignore")


def cap_blowouts(g: pd.DataFrame, cap: float) -> pd.DataFrame:
    g = g.copy()
    em = g["raw_oe"] - g["raw_de"]
    mid = (g["raw_oe"] + g["raw_de"]) / 2
    capped = np.sign(em) * np.minimum(em.abs(), cap)
    g["raw_oe"] = mid + capped / 2
    g["raw_de"] = mid - capped / 2
    return g


def fit_tempo(g: pd.DataFrame, alpha: float = 3.0) -> tuple[float, dict]:
    """Opponent-adjusted pace: tempo_obs = mu + t[team] + t[opp]."""
    teams = sorted(set(g.team) | set(g.opp))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(g)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([g.team.map(idx).values, g.opp.map(idx).values])
    X = sparse.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, len(teams)))
    w = g["weight"].values if "weight" in g else np.ones(n)
    m = Ridge(alpha=alpha, solver="sparse_cg", max_iter=5000).fit(X, g["tempo"].values, sample_weight=w)
    return float(m.intercept_), {t: float(m.coef_[idx[t]]) for t in teams}


def project(r, h, a, neutral, tempo_adj=None):
    p = r.project(h, a, neutral=neutral)
    if tempo_adj is None:
        return p["proj_margin"], p["proj_total"]
    mu, t = tempo_adj
    poss = mu + t.get(h, 0.0) + t.get(a, 0.0)
    scale = poss / p["proj_poss"]
    return p["proj_margin"] * scale, p["proj_total"] * scale


def run(season: int, start: str, cap: float = 30.0, half_life: float = 45.0) -> pd.DataFrame:
    prev_raw, cur_raw = live.load_games(season - 1), live.load_games(season)
    prev = R.add_tempo_efficiency(live.to_long(prev_raw))
    cur = R.add_tempo_efficiency(live.to_long(cur_raw))
    cur["date"] = pd.to_datetime(cur["date"])
    games = cur_raw.copy(); games["date"] = pd.to_datetime(games["date"])
    out = []
    for cut in pd.date_range(pd.to_datetime(start), games.date.max(), freq="7D"):
        hc = cur[cur.date < cut].copy()
        n_avg = len(hc) / max(hc.team.nunique(), 1)
        pp = prev.copy(); pp["weight"] = 0.5 * 12 / (12 + n_avg)
        hc["weight"] = 1.0
        base_hist = pd.concat([pp, hc], ignore_index=True)
        n = base_hist.groupby("team").size(); d1 = set(n[n >= 10].index)
        base_hist = base_hist[base_hist.team.isin(d1) & base_hist.opp.isin(d1)]

        rec_hist = base_hist.copy()
        is_cur = rec_hist["date"].notna() & (pd.to_datetime(rec_hist["date"]) >= pd.Timestamp(f"{season-1}-10-01"))
        days = (cut - pd.to_datetime(rec_hist.loc[is_cur, "date"])).dt.days
        rec_hist.loc[is_cur, "weight"] = 0.5 ** (days / half_life)

        variants = {
            "base": base_hist,
            "cap": cap_blowouts(base_hist, cap),
            "rec": rec_hist,
            "caprec": cap_blowouts(rec_hist, cap),
        }
        fitted = {k: R.fit_adjusted_ratings(v, alpha=3.0) for k, v in variants.items()}
        tempos = {k: fit_tempo(v) for k, v in variants.items() if k in ("base", "caprec")}

        wk = games[(games.date >= cut) & (games.date < cut + pd.Timedelta(days=7))]
        wk = wk[wk.home_id.isin(d1) & wk.away_id.isin(d1)].dropna(subset=["home_score", "away_score"])
        for g in wk.itertuples(index=False):
            row = {"game_id": g.game_id, "date": g.date.date().isoformat(),
                   "margin": g.home_score - g.away_score, "total": g.home_score + g.away_score}
            for k, r in fitted.items():
                row[f"{k}_m"], row[f"{k}_t"] = project(r, g.home_id, g.away_id, bool(g.neutral))
            for k, t in tempos.items():
                row[f"{k}T_m"], row[f"{k}T_t"] = project(fitted[k], g.home_id, g.away_id, bool(g.neutral), t)
            out.append(row)
        print(f"  week {cut.date()}: {len(wk)} games")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--start", default="2025-11-24")
    ap.add_argument("--cap", type=float, default=30.0)
    ap.add_argument("--half-life", type=float, default=45.0)
    a = ap.parse_args()
    df = run(a.season, a.start, a.cap, a.half_life)
    df.to_csv(os.path.join(live.ROOT, "data", "history", f"ratings_variants_{a.season}.csv"), index=False)


if __name__ == "__main__":
    main()

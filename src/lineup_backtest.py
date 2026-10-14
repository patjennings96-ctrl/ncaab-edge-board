"""
lineup_backtest.py
==================
Does knowing who plays improve the projection? Walk forward through a season
week by week (no future data) and compare three versions of every game's
ratings projection:

  base      team ratings only (what the board did before)
  heuristic + lineup adjustment from recent absences only (no injury reports):
              a rotation player who missed the last k team games is assumed
              likely out at an empirically measured return rate
  informed  + lineup adjustment knowing exactly who dressed and played
              (the best case: a complete injury report)

Each is scored on margin and total error, and against DraftKings closing lines.

    python lineup_backtest.py --season 2026
Writes docs/data/lineup_backtest_<season>.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import live
import players as P
import ratings as R

warnings.filterwarnings("ignore")


def _d1_long(raw):
    g = R.add_tempo_efficiency(live.to_long(raw))
    g["date"] = pd.to_datetime(g["date"])
    return g


def return_rates(pg: pd.DataFrame) -> dict:
    """Empirical P(rotation player plays | missed team's last k games)."""
    rows = []
    for tid, t in pg.groupby("team_id"):
        m = t.pivot_table(index=["date", "game_id"], columns="player_id", values="min",
                          aggfunc="sum", fill_value=0).sort_index()
        if len(m) < 8:
            continue
        arr = m.values
        for j in range(arr.shape[1]):
            s = arr[:, j]
            played_mins = []
            streak = 0
            for i in range(len(s)):
                if i >= 5 and played_mins and np.mean(played_mins[-5:]) >= 10 and streak >= 1:
                    rows.append((min(streak, 3), s[i] > 0))
                if s[i] > 0:
                    played_mins.append(s[i]); streak = 0
                else:
                    streak += 1
    df = pd.DataFrame(rows, columns=["k", "played"])
    return {int(k): round(float(v), 3) for k, v in df.groupby("k")["played"].mean().items()}, \
        {int(k): int(v) for k, v in df.groupby("k").size().items()}


def run(season: int, start: str, step_days: int = 7) -> dict:
    prev_raw, cur_raw = live.load_games(season - 1), live.load_games(season)
    prev_pg, cur_pg = P.load_player_games(season - 1), P.load_player_games(season)
    prev_long, cur_long = _d1_long(prev_raw), _d1_long(cur_raw)
    lines = pd.read_csv(os.path.join(live.ROOT, "data", "lines", f"{season}.csv"), dtype={"game_id": str})
    games = cur_raw.merge(lines, on="game_id", how="left")
    games["date"] = pd.to_datetime(games["date"])
    cur_pg["date"] = pd.to_datetime(cur_pg["date"])
    prev_pg["date"] = pd.to_datetime(prev_pg["date"])

    # previous season, full: team ratings -> player ratings (the prior)
    n = prev_long.groupby("team").size(); d1p = set(n[n >= 10].index)
    pl = prev_long[prev_long.team.isin(d1p) & prev_long.opp.isin(d1p)]
    prev_tr = R.fit_adjusted_ratings(pl, alpha=3.0)
    prior = P.player_ratings(prev_pg[prev_pg.team_id.isin(d1p)], prev_raw, prev_tr)
    print(f"[prior] {len(prior)} players rated from {season - 1}")

    rr, rr_n = return_rates(prev_pg)
    print(f"[return rates from {season - 1}] {rr}  (n={rr_n})")

    cuts = pd.date_range(pd.to_datetime(start), games["date"].max(), freq=f"{step_days}D")
    out = []
    for cut in cuts:
        hc = cur_long[cur_long["date"] < cut].copy()
        n_avg = len(hc) / max(hc["team"].nunique(), 1)
        w_prev = 0.5 * 12 / (12 + n_avg)
        pp = prev_long.copy(); pp["weight"] = w_prev; hc["weight"] = 1.0
        hist = pd.concat([pp, hc], ignore_index=True)
        n = hist.groupby("team").size(); d1 = set(n[n >= 10].index)
        hist = hist[hist.team.isin(d1) & hist.opp.isin(d1)]
        tr = R.fit_adjusted_ratings(hist, alpha=3.0)

        pg_before = cur_pg[(cur_pg["date"] < cut) & cur_pg.team_id.isin(d1)]
        cur_r = P.player_ratings(pg_before, cur_raw, tr, prior=prior)
        allr = pd.concat([prior[~prior.index.isin(cur_r.index)], cur_r])

        weights = pd.concat([
            pd.Series(w_prev, index=prev_raw.game_id.astype(str)),
            pd.Series(1.0, index=cur_raw[pd.to_datetime(cur_raw.date) < cut].game_id.astype(str))])
        base = P.baseline_strength(pd.concat([prev_pg, pg_before]), weights, allr)

        wk = games[(games.date >= cut) & (games.date < cut + pd.Timedelta(days=step_days))]
        wk = wk[wk.home_id.isin(d1) & wk.away_id.isin(d1)].dropna(subset=["home_score", "away_score"])
        n_wk = 0
        for g in wk.itertuples(index=False):
            pr = tr.project(g.home_id, g.away_id, neutral=bool(g.neutral))
            pg_g = cur_pg[cur_pg.game_id == g.game_id]
            row = {"game_id": g.game_id, "date": g.date.date().isoformat(),
                   "margin": g.home_score - g.away_score, "total": g.home_score + g.away_score,
                   "close_spread_home": g.close_spread_home, "close_total": g.close_total,
                   "poss": pr["proj_poss"], "base_margin": pr["proj_margin"], "base_total": pr["proj_total"],
                   "prior_weight": w_prev}
            ok = True
            for side, tid in (("h", g.home_id), ("a", g.away_id)):
                hist_team = cur_pg[(cur_pg.team_id == tid) & (cur_pg.date < g.date)]
                if hist_team.game_id.nunique() < 3 or tid not in base.index:
                    ok = False
                    break
                b_o, b_d = base.loc[tid, "base_o"], base.loc[tid, "base_d"]
                so, sd, em = P.project_lineup(hist_team, tid, allr, return_prob=rr)
                played = set(pg_g[(pg_g.team_id == tid) & (pg_g["min"] > 0)].player_id)
                io, idd, _ = P.project_lineup(hist_team, tid, allr, played_override=played)
                row.update({f"{side}_heur_o": so - b_o, f"{side}_heur_d": sd - b_d,
                            f"{side}_inf_o": io - b_o, f"{side}_inf_d": idd - b_d})
            if ok:
                out.append(row); n_wk += 1
        print(f"  week {cut.date()}: {n_wk} games scored, prior w={w_prev:.2f}")
    return {"rows": pd.DataFrame(out), "return_rates": rr, "return_n": rr_n}


def evaluate(df: pd.DataFrame) -> dict:
    """Fit the adjustment weight on the first half, score on the second half."""
    df = df.copy()
    for v in ("heur", "inf"):
        df[f"{v}_dm"] = ((df[f"h_{v}_o"] + df[f"h_{v}_d"]) - (df[f"a_{v}_o"] + df[f"a_{v}_d"])) * df.poss / 100
        df[f"{v}_dt"] = ((df[f"h_{v}_o"] - df[f"a_{v}_d"]) + (df[f"a_{v}_o"] - df[f"h_{v}_d"])) * df.poss / 100
    df = df.sort_values("date")
    half = len(df) // 2
    fit, test = df.iloc[:half], df.iloc[half:]
    rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
    res = {"games": len(df), "fit_games": half, "test_games": len(test),
           "test_from": test.date.iloc[0]}

    def lam(col_adj, target, basecol):
        x, y = fit[col_adj].values, (fit[target] - fit[basecol]).values
        return float(np.dot(x, y) / max(np.dot(x, x), 1e-9))

    for v in ("heur", "inf"):
        lm = lam(f"{v}_dm", "margin", "base_margin")
        lt = lam(f"{v}_dt", "total", "base_total")
        m_adj = test.base_margin + lm * test[f"{v}_dm"]
        t_adj = test.base_total + lt * test[f"{v}_dt"]
        s = test.dropna(subset=["close_spread_home"])
        mk = -s.close_spread_home
        sel = s.index
        # does the adjusted projection add information beyond the closing line?
        X = np.column_stack([np.ones(len(s)), mk, m_adj.loc[sel]])
        b, *_ = np.linalg.lstsq(X, s.margin.values, rcond=None)
        # ATS when adjusted projection disagrees with the close by 3+ points
        d = m_adj.loc[sel] - mk
        cover = np.sign(s.margin - mk)
        big = d.abs() >= 3
        res[v] = {
            "weight_margin": round(lm, 3), "weight_total": round(lt, 3),
            "margin_rmse": round(rmse(test.margin - m_adj), 3),
            "total_rmse": round(rmse(test.total - t_adj), 3),
            "avg_abs_adj_margin": round(float((lm * test[f"{v}_dm"]).abs().mean()), 2),
            "share_games_adj_over_2pts": round(float(((lm * test[f"{v}_dm"]).abs() > 2).mean()), 3),
            "weight_vs_close": round(float(b[2]), 3),
            "ats_disagree_3plus": {"games": int(big.sum()),
                                   "cover": round(float((cover[big] == np.sign(d[big])).mean()), 4)},
        }
    s = test.dropna(subset=["close_spread_home"])
    res["base"] = {"margin_rmse": round(rmse(test.margin - test.base_margin), 3),
                   "total_rmse": round(rmse(test.total - test.base_total), 3)}
    res["market"] = {"margin_rmse": round(rmse(s.margin + s.close_spread_home), 3),
                     "total_rmse": round(rmse(test.dropna(subset=["close_total"]).pipe(lambda t: t.total - t.close_total)), 3)}
    # same-games comparison for base on the spread subset
    res["base_on_lined"] = round(rmse(s.margin - s.base_margin), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--start", default="2025-11-24")
    a = ap.parse_args()
    out = run(a.season, a.start)
    res = evaluate(out["rows"])
    res["return_rates"] = out["return_rates"]
    res["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    path = os.path.join(live.ROOT, "docs", "data", f"lineup_backtest_{a.season}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    out["rows"].to_csv(os.path.join(live.ROOT, "data", "history", f"lineup_rows_{a.season}.csv"), index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

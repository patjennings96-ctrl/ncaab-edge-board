"""
cal_backtest.py
===============
Walk-forward test of Cal's model on a past season, using the same play rules
as the sheet and DraftKings closing lines (from ESPN).

Week by week: Model 1 stats come only from games before that week; Model 2
uses the Edge Board's adjusted ratings fitted on games before that week (the
sheet used KenPom; historical KenPom snapshots aren't available). The daily
Model 2 total adjustment is computed per game date, as the sheet did each day.

    python cal_backtest.py --season 2026
Writes docs/cal/data/backtest_<season>.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import calmodel as C
import live
import ratings as R

warnings.filterwarnings("ignore")


def run(season: int, start: str, step_days: int = 7) -> pd.DataFrame:
    prev_raw, cur_raw = live.load_games(season - 1), live.load_games(season)
    lines = pd.read_csv(os.path.join(live.ROOT, "data", "lines", f"{season}.csv"), dtype={"game_id": str})
    games = cur_raw.merge(lines, on="game_id", how="left")
    games["date"] = pd.to_datetime(games["date"])
    prev_long = R.add_tempo_efficiency(live.to_long(prev_raw))
    cur_long = R.add_tempo_efficiency(live.to_long(cur_raw))
    cur_long["date"] = pd.to_datetime(cur_long["date"])

    out = []
    for cut in pd.date_range(pd.to_datetime(start), games.date.max(), freq=f"{step_days}D"):
        before = cur_raw[pd.to_datetime(cur_raw.date) < cut]
        m1 = C.model1_inputs(before, prev_raw)

        hc = cur_long[cur_long.date < cut].copy()
        n_avg = len(hc) / max(hc.team.nunique(), 1)
        pp = prev_long.copy(); pp["weight"] = 0.5 * 12 / (12 + n_avg); hc["weight"] = 1.0
        hist = pd.concat([pp, hc], ignore_index=True)
        n = hist.groupby("team").size(); d1 = set(n[n >= 10].index)
        hist = hist[hist.team.isin(d1) & hist.opp.isin(d1)]
        m2, _ = C.model2_inputs(R.fit_adjusted_ratings(hist, alpha=3.0))

        wk = games[(games.date >= cut) & (games.date < cut + pd.Timedelta(days=step_days))]
        wk = wk[wk.home_id.isin(d1) & wk.away_id.isin(d1)].dropna(subset=["home_score", "away_score"])
        wk = wk.dropna(subset=["close_spread_home", "close_total"], how="all")
        for day, dg in wk.groupby("date"):
            slate = pd.DataFrame({"game_id": dg.game_id, "start": day.date().isoformat(),
                                  "home": dg.home_id, "away": dg.away_id,
                                  "spread_home": dg.close_spread_home, "total": dg.close_total,
                                  "neutral": dg.neutral.astype(bool)})
            priced = C.price_slate(slate, m1, m2)
            if priced.empty:
                continue
            priced = priced.merge(dg[["game_id", "home_score", "away_score"]], on="game_id")
            out.append(priced)
        print(f"  week {cut.date()}: {len(wk)} games")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(df: pd.DataFrame, season: int) -> dict:
    df = df.copy()
    df["ats_u"] = [C.grade(p, l, t, h, a) for p, l, t, h, a in
                   zip(df.ats_play, df.line, df.total_line, df.home_score, df.away_score)]
    df["ou_u"] = [C.grade(p, l, t, h, a) for p, l, t, h, a in
                  zip(df.ou_play, df.line, df.total_line, df.home_score, df.away_score)]
    df["month"] = df.start.str[:7]

    def rec(u):
        u = pd.Series(u).dropna()
        n = len(u)
        return {"bets": n, "w": int((u > 0).sum()), "l": int((u < 0).sum()), "p": int((u == 0).sum()),
                "win_pct": round(float((u > 0).sum() / max((u != 0).sum(), 1)), 4),
                "units": round(float(u.sum()), 1), "roi": round(float(u.sum() / max(n, 1)), 4)}

    margin = df.home_score - df.away_score
    total = df.home_score + df.away_score
    s = df.dropna(subset=["line"])
    rmse = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 2)
    res = {
        "season": season, "games": int(len(df)),
        "ats": rec(df.ats_u), "totals": rec(df.ou_u),
        "ats_side": {k: rec(g.ats_u) for k, g in df[df.ats_play != ""].groupby("ats_play")},
        "totals_side": {k: rec(g.ou_u) for k, g in df[df.ou_play != ""].groupby("ou_play")},
        "by_month": {m: {"ats": rec(g.ats_u), "totals": rec(g.ou_u)} for m, g in df.groupby("month")},
        "accuracy": {
            "m1_margin_rmse": rmse(margin.loc[s.index] - (s.m1_home - s.m1_away)),
            "m2_margin_rmse": rmse(margin.loc[s.index] - (s.m2_home - s.m2_away)),
            "market_margin_rmse": rmse(margin.loc[s.index] + s.line),
            "m1_total_rmse": rmse(total - (df.m1_home + df.m1_away)),
            "m2_total_rmse": rmse(total - (df.m2_home + df.m2_away)),
            "market_total_rmse": rmse(total - df.total_line),
            "m1_margin_bias": round(float(((s.m1_home - s.m1_away) - margin.loc[s.index]).mean()), 2),
        },
        "notes": "DraftKings closing lines via ESPN; Model 2 uses Edge Board adjusted ratings "
                 "(historical KenPom not available); 3-point home-court edge added to both models.",
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--start", default="2025-11-24")
    a = ap.parse_args()
    df = run(a.season, a.start)
    res = summarize(df, a.season)
    path = os.path.join(live.ROOT, "docs", "cal", "data", f"backtest_{a.season}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    df.to_csv(os.path.join(live.ROOT, "data", "history", f"cal_backtest_rows_{a.season}.csv"), index=False)
    print(json.dumps({k: res[k] for k in ("games", "ats", "totals", "ats_side", "totals_side", "accuracy")}, indent=1))


if __name__ == "__main__":
    main()

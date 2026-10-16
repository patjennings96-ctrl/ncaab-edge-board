"""
model_compare.py
================
Head-to-head of every model on the same 2025-26 games, out of sample, graded
at DraftKings closing lines. Inputs are the walk-forward row files the other
backtests write:

    data/history/ratings_variants_<season>.csv   (ratings_experiments.py)
    data/history/lineup_rows_<season>.csv        (lineup_backtest.py)
    data/history/cal_backtest_rows_<season>.csv  (cal_backtest.py)

For each candidate we report
  rmse        error vs the actual result (the close is the benchmark)
  beyond      regression weight on (model - close) when predicting the result
              alongside the close: 0 = the model adds nothing the market
              didn't know, 1 = take the model's full disagreement
  thresholds  win % and units at -110 betting every game where the model
              disagrees with the close by at least N points
The "stack" candidate is a walk-forward ridge fit, using only earlier weeks,
of (result - close) on every model's (prediction - close).

    python model_compare.py --season 2026
Writes docs/data/model_compare_<season>.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import live

LINEUP_W_M, LINEUP_W_T = 0.8, 0.5
SPREAD_THRESH = (1, 2, 3, 4, 5)
TOTAL_THRESH = (2, 3, 5, 7, 10)


def load(season: int) -> pd.DataFrame:
    h = os.path.join(live.ROOT, "data", "history")
    v = pd.read_csv(f"{h}/ratings_variants_{season}.csv", dtype={"game_id": str})
    lu = pd.read_csv(f"{h}/lineup_rows_{season}.csv", dtype={"game_id": str})
    cal = pd.read_csv(f"{h}/cal_backtest_rows_{season}.csv", dtype={"game_id": str})
    lines = pd.read_csv(os.path.join(live.ROOT, "data", "lines", f"{season}.csv"), dtype={"game_id": str})

    for s in ("heur",):
        lu[f"{s}_dm"] = ((lu[f"h_{s}_o"] + lu[f"h_{s}_d"]) - (lu[f"a_{s}_o"] + lu[f"a_{s}_d"])) * lu.poss / 100
        lu[f"{s}_dt"] = ((lu[f"h_{s}_o"] - lu[f"a_{s}_d"]) + (lu[f"a_{s}_o"] - lu[f"h_{s}_d"])) * lu.poss / 100
    d = (v.merge(lines[["game_id", "close_spread_home", "close_total"]], on="game_id")
          .merge(lu[["game_id", "heur_dm", "heur_dt"]], on="game_id", how="left")
          .merge(cal[["game_id", "m1_home", "m1_away", "m2_home", "m2_away", "m2_tot_adj",
                      "ats_play", "ou_play"]], on="game_id", how="left"))
    d[["heur_dm", "heur_dt"]] = d[["heur_dm", "heur_dt"]].fillna(0.0)
    d = d.dropna(subset=["close_spread_home", "close_total", "m1_home", "m2_home"])
    d["close_m"] = -d.close_spread_home
    d["close_t"] = d.close_total

    # margin predictions (home minus away)
    d["edge_ratings_m"] = d.base_m
    d["edge_tempo_m"] = d.baseT_m
    d["edge_lineup_m"] = d.base_m + LINEUP_W_M * d.heur_dm
    d["edge_tempo_lineup_m"] = d.baseT_m + LINEUP_W_M * d.heur_dm
    d["cal_m1_m"] = d.m1_home - d.m1_away
    d["cal_m2_m"] = d.m2_home - d.m2_away
    d["cal_avg_m"] = (d.cal_m1_m + d.cal_m2_m) / 2
    d["blend_m"] = (d.edge_lineup_m + d.cal_m1_m) / 2
    # total predictions
    d["edge_ratings_t"] = d.base_t
    d["edge_tempo_t"] = d.baseT_t
    d["edge_lineup_t"] = d.base_t + LINEUP_W_T * d.heur_dt
    d["edge_tempo_lineup_t"] = d.baseT_t + LINEUP_W_T * d.heur_dt
    d["cal_m1_t"] = d.m1_home + d.m1_away
    d["cal_m2_t"] = d.m2_home + d.m2_away
    d["cal_m2adj_t"] = d.close_t + d.m2_tot_adj
    d["cal_avg_t"] = (d.cal_m1_t + d.cal_m2adj_t) / 2
    d["blend_t"] = (d.edge_tempo_lineup_t + d.cal_m1_t) / 2
    d["date"] = pd.to_datetime(d["date"])
    return d.reset_index(drop=True)


def stack(d: pd.DataFrame, cols: list[str], close: str, actual: str, alpha: float = 200.0,
          min_weeks: int = 3) -> pd.Series:
    """Walk-forward ridge on (model - close) features; returns close + predicted residual."""
    X = np.column_stack([d[c] - d[close] for c in cols])
    y = (d[actual] - d[close]).values
    wk = d.date.dt.to_period("W").astype(str).values
    out = np.full(len(d), np.nan)
    weeks = sorted(set(wk))
    for i, w in enumerate(weeks):
        if i < min_weeks:
            continue
        tr, te = np.isin(wk, weeks[:i]), wk == w
        Xt, yt = X[tr], y[tr]
        mu = Xt.mean(0)
        A = (Xt - mu).T @ (Xt - mu) + alpha * np.eye(len(cols))
        b = np.linalg.solve(A, (Xt - mu).T @ (yt - yt.mean()))
        out[te] = d[close].values[te] + yt.mean() + (X[te] - mu) @ b
    return pd.Series(out, index=d.index)


def score(d: pd.DataFrame, pred: str, actual: str, close: str, thresholds) -> dict:
    x = d.dropna(subset=[pred])
    err = x[actual] - x[pred]
    diff = x[pred] - x[close]
    beyond = np.linalg.lstsq(np.c_[np.ones(len(x)), x[close], diff], x[actual], rcond=None)[0][2]
    res = {"n": int(len(x)), "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
           "bias": round(float(err.mean()), 2), "beyond": round(float(beyond), 3), "at": {}}
    hit = np.sign(x[actual] - x[close]) * np.sign(diff)
    for t in thresholds:
        h = hit[diff.abs() >= t]
        w, l = int((h > 0).sum()), int((h < 0).sum())
        res["at"][str(t)] = {"w": w, "l": l, "pct": round(w / max(w + l, 1), 3),
                             "units": round(w - 1.1 * l, 1)}
    return res


def cal_rule(d: pd.DataFrame) -> dict:
    out = {}
    for col, actual, close in (("ats_play", "margin", "close_m"), ("ou_play", "total", "close_t")):
        x = d[d[col].fillna("") != ""]
        if col == "ats_play":
            pick_home = x[col].astype(str).str.upper().str.startswith("H")
            r = np.sign(x[actual] - x[close]) * np.where(pick_home, 1, -1)
        else:
            over = x[col].astype(str).str.upper().str.startswith("O")
            r = np.sign(x[actual] - x[close]) * np.where(over, 1, -1)
        w, l = int((r > 0).sum()), int((r < 0).sum())
        out[col] = {"w": w, "l": l, "pct": round(w / max(w + l, 1), 3), "units": round(w - 1.1 * l, 1)}
    return out


def agree(d: pd.DataFrame) -> dict:
    """Games where the Edge Board and Cal's sheet rule pick the same side,
    split by how far the Edge Board's number is from the close."""
    out = {"spreads": {}, "totals": {}}
    x = d[d.ats_play.fillna("") != ""]
    cal_home = np.where(x.ats_play.str.upper().str.startswith("H"), 1, -1)
    diff = x.edge_tempo_lineup_m - x.close_m
    res = np.sign(x.margin - x.close_m)
    for t in (0, 1, 2, 3, 4):
        for label, mask in (("agree", (np.sign(diff) == cal_home) & (diff.abs() >= t)),
                            ("disagree", (np.sign(diff) == -cal_home) & (diff.abs() >= t))):
            r = (res * cal_home)[mask]
            w, l = int((r > 0).sum()), int((r < 0).sum())
            out["spreads"][f"{label}_{t}"] = {"w": w, "l": l, "pct": round(w / max(w + l, 1), 3),
                                              "units": round(w - 1.1 * l, 1)}
    x = d[d.ou_play.fillna("") != ""]
    cal_over = np.where(x.ou_play.str.upper().str.startswith("O"), 1, -1)
    diff = x.edge_tempo_lineup_t - x.close_t
    res = np.sign(x.total - x.close_t)
    for t in (0, 2, 3, 5):
        for label, mask in (("agree", (np.sign(diff) == cal_over) & (diff.abs() >= t)),
                            ("disagree", (np.sign(diff) == -cal_over) & (diff.abs() >= t))):
            r = (res * cal_over)[mask]
            w, l = int((r > 0).sum()), int((r < 0).sum())
            out["totals"][f"{label}_{t}"] = {"w": w, "l": l, "pct": round(w / max(w + l, 1), 3),
                                             "units": round(w - 1.1 * l, 1)}
    return out


def run(season: int) -> dict:
    d = load(season)
    m_models = ["edge_ratings_m", "edge_tempo_m", "edge_lineup_m", "edge_tempo_lineup_m",
                "cal_m1_m", "cal_m2_m", "cal_avg_m", "blend_m"]
    t_models = ["edge_ratings_t", "edge_tempo_t", "edge_lineup_t", "edge_tempo_lineup_t",
                "cal_m1_t", "cal_m2_t", "cal_m2adj_t", "cal_avg_t", "blend_t"]
    d["stack_m"] = stack(d, ["edge_tempo_lineup_m", "cal_m1_m", "cal_m2_m"], "close_m", "margin")
    d["stack_t"] = stack(d, ["edge_tempo_lineup_t", "cal_m1_t", "cal_m2adj_t"], "close_t", "total")
    m_models.append("stack_m"); t_models.append("stack_t")

    halves = {"season": d, "from_jan15": d[d.date >= "2026-01-15"]}
    res = {"season": season, "games": int(len(d)),
           "market": {k: {"margin_rmse": round(float(np.sqrt(np.mean((h.margin - h.close_m) ** 2))), 2),
                          "total_rmse": round(float(np.sqrt(np.mean((h.total - h.close_t) ** 2))), 2)}
                      for k, h in halves.items()},
           "spreads": {k: {m: score(h, m, "margin", "close_m", SPREAD_THRESH) for m in m_models}
                       for k, h in halves.items()},
           "totals": {k: {m: score(h, m, "total", "close_t", TOTAL_THRESH) for m in t_models}
                      for k, h in halves.items()},
           "cal_rule": {k: cal_rule(h) for k, h in halves.items()},
           "agreement": {k: agree(h) for k, h in halves.items()}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    res = run(a.season)
    path = os.path.join(live.ROOT, "docs", "data", f"model_compare_{a.season}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    for mkt, th in (("spreads", "3"), ("totals", "5")):
        for half in ("season", "from_jan15"):
            print(f"\n== {mkt} / {half}  (market rmse {res['market'][half]})")
            for m, s in res[mkt][half].items():
                at = " ".join(f"{t}:{v['pct']:.3f}/{v['w']+v['l']}/{v['units']:+.0f}u" for t, v in s["at"].items())
                print(f"  {m:20s} rmse {s['rmse']:5.2f} bias {s['bias']:+.2f} beyond {s['beyond']:+.3f} | {at}")
    print("\ncal rule", json.dumps(res["cal_rule"]))
    for half, a in res["agreement"].items():
        for mkt, rows in a.items():
            print(f"\n== agreement {mkt} / {half}")
            for k, v in rows.items():
                print(f"  {k:12s} {v['w']}-{v['l']}  {v['pct']:.3f}  {v['units']:+.1f}u")


if __name__ == "__main__":
    main()

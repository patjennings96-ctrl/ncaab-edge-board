"""
backtest.py
===========
Leakage-safe, walk-forward validation – the part that decides whether the model
is real or a curve-fit illusion.

The cardinal rule: to predict games on date D, EVERYTHING the model sees
(adjusted ratings, features, fitted ML weights) must be derived only from games
before D. We march forward through the calendar in steps, refitting on the
expanding past and scoring the next slice.

Metrics reported
----------------
* RMSE / MAE        – raw projection accuracy (margin & total).
* ATS hit-rate      – % of graded bets that won (52.38% = break-even at -110).
* ROI               – profit / total staked.
* CLV               – did we beat the closing line? The single best leading
                      indicator of long-run edge; positive CLV with negative
                      short-run ROI is variance, negative CLV is a broken model.
* Kelly bankroll    – simulated fractional-Kelly bankroll trajectory + max DD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from edge import (recommend_spread, recommend_total, american_to_decimal,
                  clv_points)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    bets: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _grade_spread(side: str, line: float, actual_margin: float) -> int:
    """1 win, 0 loss, -1 push. line is the bet side's spread."""
    edge = actual_margin + line if side == "HOME" else -actual_margin + line
    return 1 if edge > 0 else (-1 if edge == 0 else 0)


def _grade_total(side: str, line: float, actual_total: float) -> int:
    if actual_total == line:
        return -1
    over = actual_total > line
    return 1 if (over == (side == "OVER")) else 0


def walk_forward(games: pd.DataFrame,
                 build_features_fn,
                 fit_ratings_fn,
                 train_model_fn,
                 feature_cols,
                 start_date: str,
                 step_days: int = 7,
                 min_train_games: int = 800,
                 kelly_mult: float = 0.25,
                 min_edge: float = 0.02,
                 odds_juice: float = -110) -> BacktestResult:
    """
    Parameters
    ----------
    games            : completed games with closing lines (see schema in run_daily).
    build_features_fn(history, upcoming, ratings) -> feature DataFrame for upcoming.
    fit_ratings_fn(history) -> AdjustedRatings.
    train_model_fn(feat_history, feature_cols) -> DualModel.
    start_date       : first date to start grading (need enough history before it).
    """
    games = games.sort_values("date").copy()
    games["date"] = pd.to_datetime(games["date"])
    dates = pd.date_range(pd.to_datetime(start_date), games["date"].max(),
                          freq=f"{step_days}D")

    all_preds, all_bets = [], []

    for cut in dates:
        history = games[games["date"] < cut]
        upcoming = games[(games["date"] >= cut) &
                         (games["date"] < cut + pd.Timedelta(days=step_days))]
        if len(history) < min_train_games or upcoming.empty:
            continue

        # 1) point-in-time ratings, 2) features, 3) model — all from `history` only
        ratings = fit_ratings_fn(history)
        feat_hist = build_features_fn(history, history, ratings)
        model = train_model_fn(feat_hist, feature_cols)

        feat_up = build_features_fn(history, upcoming, ratings)
        preds = model.predict(feat_up)
        # build_features_fn is required to carry a 'date' column (one row per
        # game, home perspective). Fall back to the cut date if it didn't.
        if "date" not in preds.columns:
            preds["date"] = cut
        all_preds.append(preds)

        # 4) grade bets against the recorded closing lines
        for _, row in preds.iterrows():
            sp = recommend_spread(
                row["pred_margin"], row["sigma_margin"],
                row["close_spread_home"], odds_juice, odds_juice,
                kelly_mult=kelly_mult, min_edge=min_edge)
            if sp is not None:
                res = _grade_spread(sp.side, sp.line, row["actual_margin"])
                all_bets.append(_bet_record(row, sp, res))

            tot = recommend_total(
                row["pred_total"], row["sigma_total"],
                row["close_total"], odds_juice, odds_juice,
                kelly_mult=kelly_mult, min_edge=min_edge)
            if tot is not None:
                res = _grade_total(tot.side, tot.line, row["actual_total"])
                all_bets.append(_bet_record(row, tot, res))

    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    bets_df = pd.DataFrame(all_bets)
    return BacktestResult(preds_df, bets_df, _metrics(preds_df, bets_df))


def _bet_record(row, rec, result) -> dict:
    """One graded bet with CLV vs the closing line."""
    if rec.market == "spread":
        bet_line = row.get("open_spread_home", row["close_spread_home"])
        close_line = row["close_spread_home"]
        cl = clv_points(bet_line if rec.side == "HOME" else -bet_line,
                        close_line if rec.side == "HOME" else -close_line, rec.side)
    else:
        bet_line = row.get("open_total", row["close_total"])
        cl = clv_points(bet_line, row["close_total"], rec.side)
    return {
        "date": row["date"], "market": rec.market, "side": rec.side,
        "line": rec.line, "stake": rec.stake_units, "edge": rec.edge,
        "result": result, "clv_points": cl,
    }


def _metrics(preds: pd.DataFrame, bets: pd.DataFrame) -> dict:
    m = {}
    if not preds.empty:
        em = preds["pred_margin"] - preds["actual_margin"]
        et = preds["pred_total"] - preds["actual_total"]
        m["margin_rmse"] = float(np.sqrt(np.mean(em ** 2)))
        m["margin_mae"] = float(np.mean(np.abs(em)))
        m["total_rmse"] = float(np.sqrt(np.mean(et ** 2)))
        m["total_mae"] = float(np.mean(np.abs(et)))

    if not bets.empty:
        graded = bets[bets["result"] >= 0]           # drop pushes from win%
        wins = (graded["result"] == 1).sum()
        n = len(graded)
        m["n_bets"] = int(len(bets))
        m["ats_pct"] = float(wins / n) if n else float("nan")

        # ROI on flat 1u and on Kelly stakes, at the bet's decimal odds (-110)
        dec = american_to_decimal(-110) - 1.0
        pnl_flat, staked_flat = 0.0, 0.0
        bankroll, peak, max_dd, kelly_pnl, kelly_staked = 100.0, 100.0, 0.0, 0.0, 0.0
        curve = []
        for _, b in bets.iterrows():
            if b["result"] == -1:                      # push
                curve.append(bankroll); continue
            won = b["result"] == 1
            # flat
            pnl_flat += dec if won else -1.0
            staked_flat += 1.0
            # kelly (stake is in units; treat 1u = 1% of starting bankroll)
            stake_cash = b["stake"] / 100.0 * bankroll
            delta = stake_cash * dec if won else -stake_cash
            bankroll += delta
            kelly_pnl += delta
            kelly_staked += stake_cash
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak)
            curve.append(bankroll)

        m["roi_flat"] = float(pnl_flat / staked_flat) if staked_flat else float("nan")
        m["roi_kelly"] = float(kelly_pnl / kelly_staked) if kelly_staked else float("nan")
        m["bankroll_final"] = float(bankroll)
        m["max_drawdown"] = float(max_dd)
        m["clv_points_avg"] = float(bets["clv_points"].mean())
        m["clv_positive_pct"] = float((bets["clv_points"] > 0).mean())
        m["bankroll_curve"] = curve
    return m


def summarize(result: BacktestResult) -> str:
    m = result.metrics
    L = ["=== Walk-forward backtest ==="]
    for k in ["margin_rmse", "margin_mae", "total_rmse", "total_mae"]:
        if k in m:
            L.append(f"{k:>16}: {m[k]:.3f}")
    for k in ["n_bets", "ats_pct", "roi_flat", "roi_kelly",
              "clv_points_avg", "clv_positive_pct", "max_drawdown", "bankroll_final"]:
        if k in m:
            v = m[k]
            L.append(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
    L.append("Break-even ATS at -110 = 0.5238. CLV>0 is the health check.")
    return "\n".join(L)

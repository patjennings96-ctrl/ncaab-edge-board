"""
run_daily.py
============
Daily orchestration entrypoint. Schedule this (GitHub Actions cron / Cloud
Scheduler / local cron) to refresh ratings, score today's slate, and publish the
betting card to Google Sheets.

Pipeline
--------
    1. Load season-to-date games (+ early-season prior blend if it's November).
    2. Fit point-in-time adjusted ratings on everything played SO FAR.
    3. Train the dual model on historical (game, features, result) rows.
    4. Pull today's schedule + current market lines.
    5. Build matchup features, predict margin/total + sigma.
    6. Compute Edge % and fractional-Kelly stake per market.
    7. Write three tabs to Sheets: Ratings, Today's Card, Backtest Summary.

Network calls (odds/schedule/ratings) are isolated in ingest.py. This file runs
its full logic on synthetic data when `--demo` is passed so you can see the
shape of the output with no credentials.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import ingest
import ratings as R
import model as M
from edge import recommend_spread, recommend_total


FEATURES = ["proj_margin", "proj_total", "proj_poss", "adj_em_diff",
            "tempo_sum", "custom_hca"]


def build_training_frame(games_long: pd.DataFrame, ratings) -> pd.DataFrame:
    """Home-perspective game rows with projection features + actual results."""
    rows = []
    for x in games_long[games_long.location == "H"].itertuples(index=False):
        pr = ratings.project(x.team, x.opp, neutral=False)
        rows.append({
            "date": x.date, "home": x.team, "away": x.opp,
            "proj_margin": pr["proj_margin"], "proj_total": pr["proj_total"],
            "proj_poss": pr["proj_poss"],
            "adj_em_diff": ratings.adj_em(x.team) - ratings.adj_em(x.opp),
            "tempo_sum": ratings.team_tempo(x.team) + ratings.team_tempo(x.opp),
            "custom_hca": ratings.hca,
            "actual_margin": x.team_score - x.opp_score,
            "actual_total": x.team_score + x.opp_score,
        })
    return pd.DataFrame(rows)


def build_slate_frame(slate: pd.DataFrame, ratings) -> pd.DataFrame:
    """`slate`: today's games [home, away, neutral, spread_home, total,
    over_odds, under_odds]. Adds projection features for prediction."""
    rows = []
    for x in slate.itertuples(index=False):
        pr = ratings.project(x.home, x.away, neutral=bool(getattr(x, "neutral", False)))
        rows.append({
            "home": x.home, "away": x.away,
            "proj_margin": pr["proj_margin"], "proj_total": pr["proj_total"],
            "proj_poss": pr["proj_poss"],
            "adj_em_diff": ratings.adj_em(x.home) - ratings.adj_em(x.away),
            "tempo_sum": ratings.team_tempo(x.home) + ratings.team_tempo(x.away),
            "custom_hca": 0.0 if getattr(x, "neutral", False) else ratings.hca,
            "spread_home": x.spread_home, "total": x.total,
            "over_odds": x.over_odds, "under_odds": x.under_odds,
        })
    return pd.DataFrame(rows)


def make_card(pred_df: pd.DataFrame, kelly_mult=0.25, min_edge=0.02) -> pd.DataFrame:
    """Turn predictions + lines into a recommended-bet card."""
    card = []
    for row in pred_df.itertuples(index=False):
        sp = recommend_spread(row.pred_margin, row.sigma_margin,
                              row.spread_home, -110, -110,
                              kelly_mult=kelly_mult, min_edge=min_edge)
        tot = recommend_total(row.pred_total, row.sigma_total,
                             row.total, -110, -110,
                             kelly_mult=kelly_mult, min_edge=min_edge)
        base = {"home": row.home, "away": row.away,
                "proj_margin": round(row.pred_margin, 1),
                "proj_total": round(row.pred_total, 1)}
        if sp:
            card.append({**base, "mkt": "SPREAD", **sp.as_row()})
        if tot:
            card.append({**base, "mkt": "TOTAL", **tot.as_row()})
    cols = ["home", "away", "mkt", "side", "line", "odds",
            "proj_margin", "proj_total", "model_prob", "edge_pct", "stake_units"]
    df = pd.DataFrame(card)
    return df[[c for c in cols if c in df.columns]] if not df.empty else df


def run(demo: bool = True, sheet_id: str | None = None):
    if demo:
        games = ingest.synthesize_demo_games()
        games = R.add_tempo_efficiency(games)
        games["date"] = pd.to_datetime(games["date"])
    else:
        raise NotImplementedError(
            "Wire ingest.fetch_* for live box scores, schedule, and odds.")

    ratings = R.fit_adjusted_ratings(games, alpha=50.0)
    train = build_training_frame(games, ratings)
    dual = M.DualModel.train(train, FEATURES)

    # --- today's slate ---
    if demo:
        teams = ratings_table_teams(ratings)
        rng = np.random.default_rng(0)
        slate = pd.DataFrame([{
            "home": teams[i], "away": teams[i + 1], "neutral": False,
            "spread_home": -round(ratings.project(teams[i], teams[i + 1])["proj_margin"]
                                  + rng.normal(0, 2.5), 1),
            "total": round(ratings.project(teams[i], teams[i + 1])["proj_total"]
                           + rng.normal(0, 4), 1),
            "over_odds": -110, "under_odds": -110,
        } for i in range(0, 12, 2)])
    else:
        slate = ingest.fetch_odds("today", api_key=os.environ["ODDS_API_KEY"])

    slate_feat = build_slate_frame(slate, ratings)
    preds = dual.predict(slate_feat)
    card = make_card(preds)

    print("\n=== TODAY'S CARD ===")
    print(card.to_string(index=False) if not card.empty else "No qualifying edges.")

    # --- publish for the website + history log ---
    import publish
    mode = "demo" if demo else "live"
    settings = {"kelly_mult": 0.25, "min_edge_pct": 2.5, "max_stake_units": 5.0}
    print("\nWrote", publish.write_json(card, mode, settings))
    log = publish.append_log(card, mode)
    if log:
        print("Appended", log)

    # --- Google Sheets (only when a sheet id and credentials are configured) ---
    creds = os.environ.get("GOOGLE_CREDS_PATH", "credentials.json")
    if sheet_id and os.path.exists(creds):
        from sheets import write_dataframe
        write_dataframe(R.ratings_table(ratings), sheet_id, "Ratings", creds)
        write_dataframe(card, sheet_id, "Today", creds)
        print(f"Pushed Ratings + Today tabs to sheet {sheet_id}")
    elif sheet_id:
        print("Sheet id set but no credentials file found; skipped Sheets export.")
    return card


def ratings_table_teams(ratings):
    return list(R.ratings_table(ratings)["team"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="use live data feeds (default: synthetic demo season)")
    ap.add_argument("--sheet-id", default=os.environ.get("SHEET_ID") or None)
    args = ap.parse_args()
    run(demo=not args.live, sheet_id=args.sheet_id)

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
import live
import ratings as R
import model as M
from edge import recommend_spread, recommend_total


FEATURES = ["proj_margin", "proj_total", "proj_poss", "adj_em_diff",
            "tempo_sum", "custom_hca"]


def build_training_frame(games_long: pd.DataFrame, ratings) -> pd.DataFrame:
    """Home-perspective game rows with projection features + actual results."""
    rows = []
    home_rows = games_long[games_long["team"] == games_long["venue_id"]]
    for x in home_rows.itertuples(index=False):
        neutral = x.location == "N"
        pr = ratings.project(x.team, x.opp, neutral=neutral)
        rows.append({
            "date": x.date, "home": x.team, "away": x.opp,
            "proj_margin": pr["proj_margin"], "proj_total": pr["proj_total"],
            "proj_poss": pr["proj_poss"],
            "adj_em_diff": ratings.adj_em(x.team) - ratings.adj_em(x.opp),
            "tempo_sum": ratings.team_tempo(x.team) + ratings.team_tempo(x.opp),
            "custom_hca": 0.0 if neutral else ratings.hca,
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
            "game_id": getattr(x, "game_id", ""), "start": getattr(x, "start", ""),
            "proj_margin": pr["proj_margin"], "proj_total": pr["proj_total"],
            "proj_poss": pr["proj_poss"],
            "adj_em_diff": ratings.adj_em(x.home) - ratings.adj_em(x.away),
            "tempo_sum": ratings.team_tempo(x.home) + ratings.team_tempo(x.away),
            "custom_hca": 0.0 if getattr(x, "neutral", False) else ratings.hca,
            "spread_home": x.spread_home, "total": x.total,
            "home_odds": getattr(x, "home_odds", -110), "away_odds": getattr(x, "away_odds", -110),
            "over_odds": x.over_odds, "under_odds": x.under_odds,
            **{f"best_{sd}_{k}": getattr(x, f"best_{sd}_{k}", np.nan)
               for sd in ("home", "away", "over", "under") for k in ("line", "odds", "book")},
        })
    return pd.DataFrame(rows)


# Error floors (points) = out-of-sample RMSE of the closing line itself in the
# 2025-26 walk-forward test (margin 11.2, total 16.8). The model's in-sample
# residuals understate uncertainty, so never price a bet tighter than this.
SIGMA_FLOOR_MARGIN = 11.5
SIGMA_FLOOR_TOTAL = 17.0

# Minimum disagreement with the consensus number before betting. From the
# model head-to-head (src/model_compare.py): spreads were break-even or worse
# under 2 points; totals needed 3+.
MIN_POINTS_SPREAD = 2.0
MIN_POINTS_TOTAL = 3.0

# Betting window: conference regular season. In 2025-26 the same rules lost
# heavily in December (spreads -40u, totals -75u at 3+) and in March
# tournaments, and won in January-February. Outside the window plays are still
# listed and CLV-tracked, at 0 units.
BET_WINDOW = ((1, 1), (3, 8))        # (month, day) inclusive, Jan 1 - Mar 8

# Totals where Cal's sheet rule picks the same side won 82-55 (59.9%) from
# mid-January in the backtest; they get this stake multiplier. Spread agreement
# showed no lift, so spreads are only labelled.
AGREE_TOTAL_MULT = 1.5

def in_bet_window(start) -> bool:
    try:
        d = pd.to_datetime(start)
        if pd.isna(d):
            return True
        if d.tzinfo is not None:
            d = d.tz_convert("America/New_York")
    except Exception:
        return True
    (m0, d0), (m1, d1) = BET_WINDOW
    return (m0, d0) <= (d.month, d.day) <= (m1, d1)


def _shop(row, side: str, p_fn, kelly_mult: float, max_stake: float = 5.0) -> dict:
    """Re-price a chosen side at the best line/price across books."""
    line = getattr(row, f"best_{side.lower()}_line", np.nan)
    odds = getattr(row, f"best_{side.lower()}_odds", np.nan)
    book = getattr(row, f"best_{side.lower()}_book", "")
    if pd.isna(line) or pd.isna(odds):
        return {}
    from edge import kelly_fraction, american_to_decimal
    p = p_fn(float(line))
    stake = min(kelly_mult * kelly_fraction(p, odds) * 100.0, max_stake)
    return {"best_line": float(line), "best_odds": int(odds), "best_book": book,
            "best_prob": round(p, 4),
            "best_ev": round(p * (american_to_decimal(odds) - 1) - (1 - p), 4),
            "stake_units": round(stake, 3)}


def make_card(pred_df: pd.DataFrame, kelly_mult=0.25, min_edge=0.025,
              cal_plays: dict | None = None) -> pd.DataFrame:
    """Turn predictions + lines into a recommended-bet card.

    The bet decision uses the consensus (median) line; the stake is then sized
    at the best number available across books, which is the line to bet.
    `cal_plays`: {game_id: {"ATS": "HOME"/"AWAY"/"", "TOTAL": "OVER"/"UNDER"/""}}.
    """
    from edge import prob_cover_spread, prob_over
    cal_plays = cal_plays or {}
    card = []
    for row in pred_df.itertuples(index=False):
        gid = str(getattr(row, "game_id", ""))
        start = getattr(row, "start", "")
        window = in_bet_window(start)
        cal = cal_plays.get(gid, {})
        base = {"home": row.home, "away": row.away, "game_id": gid, "start": start,
                "notes": getattr(row, "notes", ""),
                "proj_margin": round(row.pred_margin, 1),
                "proj_total": round(row.pred_total, 1)}
        if pd.notna(row.spread_home) and abs(row.pred_margin + row.spread_home) >= MIN_POINTS_SPREAD:
            sig = max(row.sigma_margin, SIGMA_FLOOR_MARGIN)
            sp = recommend_spread(row.pred_margin, sig, row.spread_home,
                                  getattr(row, "home_odds", -110), getattr(row, "away_odds", -110),
                                  kelly_mult=kelly_mult, min_edge=min_edge)
            if sp:
                r = {**base, "mkt": "SPREAD", **sp.as_row()}
                if sp.side == "HOME":
                    pf = lambda L: prob_cover_spread(row.pred_margin, L, sig)
                else:
                    pf = lambda L: 1.0 - prob_cover_spread(row.pred_margin, -L, sig)
                r.update(_shop(row, sp.side, pf, kelly_mult))
                r["agree"] = cal.get("ATS", "") == sp.side
                if not window:
                    r["stake_units"] = 0.0
                card.append(r)
        if pd.notna(row.total) and abs(row.pred_total - row.total) >= MIN_POINTS_TOTAL:
            sig = max(row.sigma_total, SIGMA_FLOOR_TOTAL)
            tot = recommend_total(row.pred_total, sig, row.total, row.over_odds, row.under_odds,
                                  kelly_mult=kelly_mult, min_edge=min_edge)
            if tot:
                r = {**base, "mkt": "TOTAL", **tot.as_row()}
                if tot.side == "OVER":
                    pf = lambda L: prob_over(row.pred_total, L, sig)
                else:
                    pf = lambda L: 1.0 - prob_over(row.pred_total, L, sig)
                r.update(_shop(row, tot.side, pf, kelly_mult))
                r["agree"] = cal.get("TOTAL", "") == tot.side
                if r["agree"]:
                    r["stake_units"] = round(min(r["stake_units"] * AGREE_TOTAL_MULT, 5.0), 3)
                if not window:
                    r["stake_units"] = 0.0
                card.append(r)
    cols = ["game_id", "start", "home", "away", "mkt", "side", "line", "odds",
            "best_line", "best_odds", "best_book", "proj_margin", "proj_total",
            "model_prob", "edge_pct", "stake_units", "agree", "notes"]
    df = pd.DataFrame(card)
    return df[[c for c in cols if c in df.columns]] if not df.empty else df


def run(demo: bool = True, sheet_id: str | None = None):
    if demo:
        games = ingest.synthesize_demo_games()
        games = R.add_tempo_efficiency(games)
        games["date"] = pd.to_datetime(games["date"])
        names, extra = {}, None
    else:
        games, names, extra = load_live_games()

    # alpha=3 minimized held-out margin error on 2025-26 D-I games
    # (RMSE 11.31 vs 13.3 at the old default); demo data keeps its old setting.
    ps = None
    if demo:
        ratings = R.fit_adjusted_ratings(games, alpha=50.0)
    else:
        # Roster-based starting ratings + this season's games (see preseason.py).
        import players as P
        import preseason as PS
        ps = PS.build(extra["season"], extra["prev_raw"], extra["cur_raw"],
                      P.load_player_games(extra["season"] - 1), P.load_player_games(extra["season"]),
                      extra["d1"])
        ratings = ps["ratings"]
    ctx = None
    if not demo:
        import external as X
        import players as P
        own_ratings = ratings
        ratings = X.blend_team_ratings(ratings, X.load_team_sources(names))
        roster = P.load_player_games(extra["season"])
        roster = roster.drop_duplicates("player_id", keep="last").set_index("player_id")[["name", "team_id"]]
        ext_players = X.load_player_sources(names, roster) if not roster.empty else None
        ctx = P.build_context(extra["season"], extra["prev_raw"], extra["cur_raw"],
                              extra["prev_long"], own_ratings, extra["weights"], ext_players,
                              roster_minutes=ps["roster_minutes"], prior_games=ps["prior_games"])
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
        key = os.environ.get("ODDS_API_KEY", "")
        if not key:
            raise RuntimeError("MODEL_MODE is live but the ODDS_API_KEY secret is not set.")
        slate = live.build_live_slate(key, names)
        if not slate.empty:
            # skip games involving teams with too few games to rate (non-D1 etc.)
            counts = games.groupby("team").size()
            ok = slate["home"].map(counts).fillna(0).ge(5) & slate["away"].map(counts).fillna(0).ge(5)
            slate = slate[ok]

    # --- Cal's model: same slate and lines, its own two-model rules ---
    cal_plays = {}
    if not demo:
        try:
            import calmodel
            priced = calmodel.run_daily(slate if not slate.empty else pd.DataFrame(),
                               extra["cur_raw"], extra["prev_raw"], own_ratings, names, "live")
            if priced is not None and not priced.empty:
                cal_plays = {str(r.game_id): {"ATS": r.ats_play or "", "TOTAL": r.ou_play or ""}
                             for r in priced.itertuples(index=False)}
        except Exception as exc:
            print(f"[cal] skipped: {exc}")

    if slate.empty:
        card = pd.DataFrame()
    else:
        slate_feat = build_slate_frame(slate, ratings)
        preds = dual.predict(slate_feat)
        if ctx is not None:
            preds = apply_lineups(preds, ctx, names)
        card = make_card(preds, cal_plays=cal_plays)
        if names:
            card["home"] = card["home"].map(names).fillna(card["home"])
            card["away"] = card["away"].map(names).fillna(card["away"])

    print("\n=== TODAY'S CARD ===")
    print(card.to_string(index=False) if not card.empty else "No qualifying edges.")

    # --- publish for the website + history log ---
    import publish
    mode = "demo" if demo else "live"
    settings = {"kelly_mult": 0.25, "min_edge_pct": 2.5, "max_stake_units": 5.0,
                "min_points_spread": MIN_POINTS_SPREAD, "min_points_total": MIN_POINTS_TOTAL,
                "bet_window": "Jan 1 - Mar 8", "agree_total_mult": AGREE_TOTAL_MULT}
    print("\nWrote", publish.write_json(card, mode, settings))
    log = publish.append_log(card, mode)
    if log:
        print("Appended", log)
    if not demo:
        publish.update_clv(live.closing_lines(), games_raw=names_and_results())

    # --- Google Sheets (only when a sheet id and credentials are configured) ---
    creds = os.environ.get("GOOGLE_CREDS_PATH", "credentials.json")
    if sheet_id and os.path.exists(creds):
        from sheets import write_dataframe
        table = R.ratings_table(ratings)
        if names:
            table.insert(2, "name", table["team"].map(names))
        write_dataframe(table, sheet_id, "Ratings", creds)
        write_dataframe(card, sheet_id, "Today", creds)
        if ctx is not None and not ctx["current"].empty:
            import players as P
            write_dataframe(P.ratings_table(ctx, names), sheet_id, "Players", creds)
        print(f"Pushed Ratings, Today and Players tabs to sheet {sheet_id}")
    elif sheet_id:
        print("Sheet id set but no credentials file found; skipped Sheets export.")
    return card


def apply_lineups(preds: pd.DataFrame, ctx: dict, names: dict) -> pd.DataFrame:
    """Shift each game's margin and total for who is expected to play tonight."""
    import players as P
    out = preds.copy()
    notes, dms, dts = [], [], []
    for r in out.itertuples(index=False):
        ho, hd, hn = P.team_adjustment(ctx, r.home)
        ao, ad, an = P.team_adjustment(ctx, r.away)
        dm = ((ho + hd) - (ao + ad)) * r.proj_poss / 100.0
        dt = ((ho - ad) + (ao - hd)) * r.proj_poss / 100.0
        dms.append(P.LINEUP_WEIGHT_MARGIN * dm)
        dts.append(P.LINEUP_WEIGHT_TOTAL * dt)
        parts = []
        if hn:
            parts.append(f"{names.get(r.home, r.home)}: {hn}")
        if an:
            parts.append(f"{names.get(r.away, r.away)}: {an}")
        notes.append("; ".join(parts))
    out["lineup_margin_adj"] = dms
    out["lineup_total_adj"] = dts
    out["pred_margin"] = out["pred_margin"] + out["lineup_margin_adj"]
    out["pred_total"] = out["pred_total"] + out["lineup_total_adj"]
    out["notes"] = notes
    big = out[out["lineup_margin_adj"].abs() >= 1.0]
    for r in big.itertuples(index=False):
        print(f"[lineups] {names.get(r.away, r.away)} @ {names.get(r.home, r.home)}: "
              f"margin {r.lineup_margin_adj:+.1f} ({r.notes})")
    return out


def load_live_games():
    """
    Current season + last season from ESPN (cached in data/games/).
    Last season's games are down-weighted and fade out as this season's sample
    grows: weight = 0.5 * 12 / (12 + games_played_per_team).
    """
    import players as P
    season = live.current_season()
    prev_raw = live.update_games(season - 1)
    cur_raw = live.update_games(season)
    P.update_player_games(season - 1)
    P.update_player_games(season)
    prev, cur = live.to_long(prev_raw), live.to_long(cur_raw)

    n_avg = len(cur) / max(cur["team"].nunique(), 1) if not cur.empty else 0.0
    w_prev = 0.5 * 12.0 / (12.0 + n_avg)
    prev["weight"], cur["weight"] = w_prev, 1.0
    print(f"[ratings] season {season}: {len(cur)//2} games this season, "
          f"{len(prev)//2} last season at weight {w_prev:.2f}")

    games = pd.concat([prev, cur], ignore_index=True)
    if games.empty:
        raise RuntimeError("No completed games available from ESPN yet.")
    # Drop games against non-Division I schools (one or two games a year each);
    # they distort the efficiency scale. D-I teams have 10+ games across the
    # two seasons on disk even on opening night.
    n = games.groupby("team").size()
    d1 = set(n[n >= 10].index)
    games = games[games["team"].isin(d1) & games["opp"].isin(d1)]
    games = R.add_tempo_efficiency(games)
    weights = pd.concat([pd.Series(w_prev, index=prev_raw["game_id"].astype(str)),
                         pd.Series(1.0, index=cur_raw["game_id"].astype(str))])
    import preseason as PS
    PS.ensure_rosters(season, sorted(d1))
    extra = {"season": season, "prev_raw": prev_raw, "cur_raw": cur_raw,
             "prev_long": R.add_tempo_efficiency(prev), "weights": weights, "d1": d1}
    return games, live.team_names(prev_raw, cur_raw), extra


def names_and_results():
    """Completed games (both seasons on disk) for grading logged plays."""
    season = live.current_season()
    frames = [live.load_games(s) for s in (season - 1, season)]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()


def ratings_table_teams(ratings):
    return list(R.ratings_table(ratings)["team"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="use live data feeds (default: synthetic demo season)")
    ap.add_argument("--sheet-id", default=os.environ.get("SHEET_ID") or None)
    args = ap.parse_args()
    run(demo=not args.live, sheet_id=args.sheet_id)

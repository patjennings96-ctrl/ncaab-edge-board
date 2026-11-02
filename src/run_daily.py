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


# Home court by season phase. Fit on 2024-25 and 2025-26, home court is worth
# about twice as much in Nov-Dec (4.0 per side per 100 possessions) as in
# conference play (1.9-2.2). The ratings carry last season's full-season value
# (~2.5-2.7), which fits Nov-Dec best in the backtest; from Jan 1 the
# conference-season value is used. Backtest, Jan 1 - Mar 8 2026: spread RMSE
# 11.20 -> 11.15, ATS 2+ 55.1% (+69u) -> 56.8% (+96u).
CONF_SEASON_START = (1, 1)


def game_hca(start, ratings, conf_hca: float | None) -> float:
    if conf_hca is None:
        return ratings.hca
    try:
        d = pd.to_datetime(start, utc=True).tz_convert("America/New_York")
    except Exception:
        return ratings.hca
    if pd.isna(d):
        return ratings.hca
    return conf_hca if 1 <= d.month <= 6 else ratings.hca


def build_slate_frame(slate: pd.DataFrame, ratings, conf_hca: float | None = None) -> pd.DataFrame:
    """`slate`: today's games [home, away, neutral, spread_home, total,
    over_odds, under_odds]. Adds projection features for prediction."""
    rows = []
    for x in slate.itertuples(index=False):
        neutral = bool(getattr(x, "neutral", False))
        pr = ratings.project(x.home, x.away, neutral=neutral)
        h = 0.0 if neutral else game_hca(getattr(x, "start", ""), ratings, conf_hca)
        if not neutral:
            # project() used ratings.hca on each side; swap in this game's value
            pr["proj_margin"] += 2.0 * (h - ratings.hca) * pr["proj_poss"] / 100.0
        rows.append({
            "home": x.home, "away": x.away,
            "game_id": getattr(x, "game_id", ""), "start": getattr(x, "start", ""),
            "proj_margin": pr["proj_margin"], "proj_total": pr["proj_total"],
            "proj_poss": pr["proj_poss"],
            "adj_em_diff": ratings.adj_em(x.home) - ratings.adj_em(x.away),
            "tempo_sum": ratings.team_tempo(x.home) + ratings.team_tempo(x.away),
            "custom_hca": h,
            "spread_home": x.spread_home, "total": x.total,
            "home_odds": getattr(x, "home_odds", -110), "away_odds": getattr(x, "away_odds", -110),
            "over_odds": x.over_odds, "under_odds": x.under_odds,
            **{f"best_{sd}_{k}": getattr(x, f"best_{sd}_{k}", np.nan)
               for sd in ("home", "away", "over", "under") for k in ("line", "odds", "book")},
            **{f"quotes_{sd}": getattr(x, f"quotes_{sd}", "[]") for sd in ("home", "away", "over", "under")},
        })
    return pd.DataFrame(rows)


# Error floors (points) = out-of-sample RMSE of the closing line itself in the
# 2025-26 walk-forward test (margin 11.2, total 16.8). The model's in-sample
# residuals understate uncertainty, so never price a bet tighter than this.
USE_ML_LAYER = False
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
    """
    Pick the book to bet: every quote for this side is scored by expected
    value with the model's win probability at that book's line, so a better
    number and a better price are weighed together (+4 at -105 can beat +4.5
    at -125). Returns the best quote, its stake, and all quotes ranked.
    """
    import json as _json
    from edge import kelly_fraction, american_to_decimal
    try:
        quotes = _json.loads(getattr(row, f"quotes_{side.lower()}", "[]") or "[]")
    except (TypeError, ValueError):
        quotes = []
    if not quotes:
        line = getattr(row, f"best_{side.lower()}_line", np.nan)
        odds = getattr(row, f"best_{side.lower()}_odds", np.nan)
        if pd.isna(line) or pd.isna(odds):
            return {}
        quotes = [(float(line), int(odds), getattr(row, f"best_{side.lower()}_book", ""))]
    ranked = []
    for line, odds, book in quotes:
        p = p_fn(float(line))
        ev = p * (american_to_decimal(odds) - 1) - (1 - p)
        ranked.append({"book": book, "line": float(line), "odds": int(odds),
                       "prob": round(p, 4), "ev": round(ev, 4)})
    ranked.sort(key=lambda q: -q["ev"])
    top = ranked[0]
    stake = min(kelly_mult * kelly_fraction(top["prob"], top["odds"]) * 100.0, max_stake)
    return {"best_line": top["line"], "best_odds": top["odds"], "best_book": top["book"],
            "best_prob": top["prob"], "best_ev": top["ev"],
            "stake_units": round(stake, 3), "quotes": _json.dumps(ranked[:12])}


def _half_up(x: float) -> float:
    return float(np.ceil(round(x * 2, 6)) / 2)


def _half_down(x: float) -> float:
    return float(np.floor(round(x * 2, 6)) / 2)


def play_to(mkt: str, side: str, proj: float, sigma: float, min_points: float,
            min_edge: float) -> float:
    """
    Worst line (the picked side's own number) that still meets the card's
    rules at -110: the gap to the model must be at least `min_points` and the
    win probability at least 50% + `min_edge`.
    """
    from scipy.stats import norm
    gap = max(min_points, sigma * float(norm.ppf(0.5 + min_edge)))
    if mkt == "SPREAD":
        # home line L: proj + L >= gap; away line A = -L: A - proj >= gap
        return _half_up(gap - proj) if side == "HOME" else _half_up(proj + gap)
    return _half_down(proj - gap) if side == "OVER" else _half_up(proj + gap)


# --- Staking (Oct 2026 review of the 2025-26 backtest) -------------------------
# The pricing sigmas above decide WHETHER to bet (with the point minimums). The
# win probabilities they imply were far too confident for SIZING: at a 3-point
# spread gap they say 60%, while backtested plays won ~55% in Jan-Mar and ~51%
# over the full season. Stakes now use calibrated probabilities:
#   spreads  P(win) = Phi(gap / CAL_SIGMA_SPREAD): 52.7% at 2 pts, 54.0% at 3,
#            56.6% at 5 (between the Jan-Mar fit, sigma 23, and the full season, 97)
#   totals   bigger gaps did not win more (5+ point gaps lost), so totals are
#            flat-staked, and gaps over TOTAL_MAX_GAP are tracked, not bet.
CAL_SIGMA_SPREAD = 30.0
CAL_SIGMA_TOTAL = 42.0          # display only (fit on 3-5 point totals)
MAX_STAKE_SPREAD = 3.0
TOTAL_FLAT_UNITS = 1.0
TOTAL_MAX_GAP = 5.0


# Early-season margin calibration. While the roster-based starting rating
# still carries a lot of weight, projected margins are too compressed: in the
# 2025-26 backtest actual margins ran 1.25-1.43x the projection in Nov-Dec
# (the market's margins matched results), and ~1.0x from January. Stretch the
# team-strength part of the margin (home court left as is) by a factor that
# fades with the starting rating's weight (in games):
#   k = min(1 + 0.10 * max(weight - 11.5, 0), 1.7)
# 18.5+ games (preseason, November) -> 1.7, 15 -> 1.35, 13 -> 1.15, under 11.5 -> 1.0.
# Backtest margin RMSE: Nov 13.59 -> 12.98, Dec 12.71 -> 12.35, Jan-Mar unchanged.
STRETCH_SLOPE, STRETCH_FROM, STRETCH_CAP = 0.10, 11.5, 1.7


def _stretch_weight(prior_games) -> np.ndarray:
    return np.maximum(np.asarray(prior_games, dtype=float) - STRETCH_FROM, 0.0)


def early_margin_stretch(prior_games: float, slope: float = STRETCH_SLOPE) -> float:
    return float(min(1.0 + slope * float(_stretch_weight(prior_games)), STRETCH_CAP))


# Self-calibration: every run logs each untipped game's projection (strength
# gap before the stretch, home-court points, starting-rating weight). Once
# games finish, the stretch slope is re-fit on this season's results and
# blended with last season's 0.10, weighted like STRETCH_PRIOR_GAMES games.
PROJ_LOG = os.path.join(live.ROOT, "data", "history", "projections_{season}.csv")
STRETCH_PRIOR_GAMES = 400


def log_projections(preds: pd.DataFrame, season: int, prior_games: float, k: float) -> None:
    if preds is None or preds.empty:
        return
    path = PROJ_LOG.format(season=season)
    hca_pts = 2.0 * preds["custom_hca"].fillna(0.0) * preds["proj_poss"] / 100.0
    new = pd.DataFrame({
        "game_id": preds["game_id"].astype(str), "start": preds["start"].astype(str),
        "home": preds["home"].astype(str), "away": preds["away"].astype(str),
        "hca_pts": hca_pts.round(3), "strength0": (preds["pred_margin_raw"] - hca_pts).round(3),
        "prior_games": round(prior_games, 3), "k": round(k, 4),
        "pred_margin": preds["pred_margin"].round(2), "pred_total": preds["pred_total"].round(2),
        "spread_home": preds["spread_home"], "total": preds["total"],
        "logged_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")})
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"game_id": str})
        # keep the last projection made before tip-off: replace only games that haven't started
        now = pd.Timestamp.now(tz="UTC")
        started = pd.to_datetime(old["start"], utc=True, errors="coerce") <= now
        old = old[started | ~old["game_id"].isin(new["game_id"])]
        new = pd.concat([old, new[~new["game_id"].isin(old["game_id"])]], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False)


def fitted_stretch_slope(season: int, games_raw: pd.DataFrame) -> tuple[float, int]:
    """Slope c in actual strength = (1 + c*w) * projected strength, fit on this
    season's finished games and shrunk toward last season's value."""
    path = PROJ_LOG.format(season=season)
    if not os.path.exists(path) or games_raw is None or games_raw.empty:
        return STRETCH_SLOPE, 0
    log = pd.read_csv(path, dtype={"game_id": str})
    res = games_raw.dropna(subset=["home_score", "away_score"]).copy()
    res["game_id"] = res["game_id"].astype(str)
    d = log.merge(res[["game_id", "home_score", "away_score"]], on="game_id")
    d["fw"] = _stretch_weight(d["prior_games"]) * d["strength0"]
    d = d[d["fw"].abs() > 0]
    if len(d) < 30:
        return STRETCH_SLOPE, len(d)
    actual = (d["home_score"] - d["away_score"]) - d["hca_pts"]
    sxx = float((d["fw"] ** 2).sum())
    c_obs = float(((actual - d["strength0"]) * d["fw"]).sum() / sxx)
    n = len(d)
    c = (STRETCH_PRIOR_GAMES * STRETCH_SLOPE + n * c_obs) / (STRETCH_PRIOR_GAMES + n)
    return float(np.clip(c, 0.0, 0.25)), n


def _cal_prob(gap_for_side: float, sigma: float) -> float:
    """Calibrated win probability for the picked side, given its gap in points."""
    from scipy.stats import norm
    return float(norm.cdf(gap_for_side / sigma))


def make_card(pred_df: pd.DataFrame, kelly_mult=0.25, min_edge=0.025,
              cal_plays: dict | None = None) -> pd.DataFrame:
    """Turn predictions + lines into a recommended-bet card.

    Which side, and whether to bet: the consensus line, the point minimums and
    the pricing sigmas. How much: calibrated probabilities (spreads, quarter
    Kelly at the best number across books) or a flat stake (totals).
    `cal_plays`: {game_id: {"ATS": "HOME"/"AWAY"/"", "TOTAL": "OVER"/"UNDER"/""}}.
    """
    from edge import kelly_fraction
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
                # picked side's own line L -> its gap to the model
                if sp.side == "HOME":
                    gap = lambda L: row.pred_margin + L
                else:
                    gap = lambda L: L - row.pred_margin
                pf = lambda L: _cal_prob(gap(L), CAL_SIGMA_SPREAD)
                p0 = pf(sp.line)
                r["gap"] = round(gap(sp.line), 2)
                r["model_prob"] = round(p0, 4)
                r["edge_pct"] = round(100 * (p0 - sp.fair_prob), 2)
                r["stake_units"] = round(min(kelly_mult * kelly_fraction(p0, sp.odds) * 100, MAX_STAKE_SPREAD), 3)
                r.update(_shop(row, sp.side, pf, kelly_mult, max_stake=MAX_STAKE_SPREAD))
                r["play_to"] = play_to("SPREAD", sp.side, row.pred_margin, sig, MIN_POINTS_SPREAD, min_edge)
                r["agree"] = cal.get("ATS", "") == sp.side
                r["reason"] = "" if window else "Outside betting window"
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
                    gap = lambda L: row.pred_total - L
                else:
                    gap = lambda L: L - row.pred_total
                pf = lambda L: _cal_prob(gap(L), CAL_SIGMA_TOTAL)
                p0 = pf(tot.line)
                r["gap"] = round(gap(tot.line), 2)
                r["model_prob"] = round(p0, 4)
                r["edge_pct"] = round(100 * (p0 - tot.fair_prob), 2)
                r.update(_shop(row, tot.side, pf, kelly_mult))
                r["play_to"] = play_to("TOTAL", tot.side, row.pred_total, sig, MIN_POINTS_TOTAL, min_edge)
                r["agree"] = cal.get("TOTAL", "") == tot.side
                r["stake_units"] = TOTAL_FLAT_UNITS * (AGREE_TOTAL_MULT if r["agree"] else 1.0)
                r["reason"] = ""
                if abs(row.pred_total - row.total) > TOTAL_MAX_GAP:
                    r["stake_units"], r["reason"] = 0.0, f"Over {TOTAL_MAX_GAP:g} pts off the market"
                elif not window:
                    r["stake_units"], r["reason"] = 0.0, "Outside betting window"
                card.append(r)
    cols = ["game_id", "start", "home", "away", "mkt", "side", "line", "odds",
            "best_line", "best_odds", "best_book", "proj_margin", "proj_total",
            "model_prob", "edge_pct", "stake_units", "play_to", "gap", "agree", "reason", "notes", "quotes"]
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
        X.refresh_stamps()          # KenPom paste -> teams.csv; date stamps for freshness
        own_ratings = ratings
        ratings = X.blend_team_ratings(ratings, X.load_team_sources(names))
        roster = P.load_player_games(extra["season"])
        roster = roster.drop_duplicates("player_id", keep="last").set_index("player_id")[["name", "team_id"]]
        ext_players = X.load_player_sources(names, roster) if not roster.empty else None
        ctx = P.build_context(extra["season"], extra["prev_raw"], extra["cur_raw"],
                              extra["prev_long"], own_ratings, extra["weights"], ext_players,
                              roster_minutes=ps["roster_minutes"], prior_games=ps["prior_games"])
    # The ML layer (DualModel) was never backtested; every calibration on the
    # card (early-season stretch, staking probabilities, point thresholds,
    # home court) was fit on the ratings + lineup projection, so that is what
    # gets priced. Set USE_ML_LAYER = True to go back.
    dual = None
    if USE_ML_LAYER or demo:
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
    priced = None
    preds = None
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
        conf_hca = ps.get("conf_hca") if ps else None
        slate_feat = build_slate_frame(slate, ratings, conf_hca)
        if dual is not None:
            preds = dual.predict(slate_feat)
        else:
            preds = slate_feat.copy()
            preds["pred_margin"], preds["pred_total"] = preds["proj_margin"], preds["proj_total"]
            preds["sigma_margin"], preds["sigma_total"] = SIGMA_FLOOR_MARGIN, SIGMA_FLOOR_TOTAL
        preds["pred_margin_raw"] = preds["pred_margin"]
        k = 1.0
        if ps is not None:
            slope, n_fit = fitted_stretch_slope(extra["season"], extra["cur_raw"])
            k = early_margin_stretch(ps["prior_games"], slope)
            print(f"[calibration] stretch slope {slope:.3f} ({n_fit} finished games this season)")
            if k != 1.0:
                # stretch the strength gap, not home court
                hca_pts = 2.0 * preds["custom_hca"].fillna(0.0) * preds["proj_poss"] / 100.0
                preds["pred_margin"] = hca_pts + k * (preds["pred_margin"] - hca_pts)
                print(f"[calibration] early-season margin stretch x{k:.2f} "
                      f"(starting rating worth {ps['prior_games']:.1f} games)")
        if ps is not None:
            log_projections(preds, extra["season"], ps["prior_games"], k)
        if ctx is not None:
            preds = apply_lineups(preds, ctx, names)
        card = make_card(preds, cal_plays=cal_plays)
        if names and not card.empty:
            card["home"] = card["home"].map(names).fillna(card["home"])
            card["away"] = card["away"].map(names).fillna(card["away"])

    print("\n=== TODAY'S CARD ===")
    print(card.to_string(index=False) if not card.empty else "No qualifying edges.")

    # --- publish for the website + history log ---
    import publish
    mode = "demo" if demo else "live"
    settings = {"kelly_mult": 0.25, "min_edge_pct": 2.5, "max_stake_units": MAX_STAKE_SPREAD,
                "total_flat_units": TOTAL_FLAT_UNITS, "total_max_gap": TOTAL_MAX_GAP,
                "cal_sigma_spread": CAL_SIGMA_SPREAD,
                "min_points_spread": MIN_POINTS_SPREAD, "min_points_total": MIN_POINTS_TOTAL,
                "bet_window": "Jan 1 - Mar 8", "agree_total_mult": AGREE_TOTAL_MULT}
    print("\nWrote", publish.write_json(card, mode, settings))
    print("Wrote", publish.write_slate(preds, priced, card, names))
    log = publish.append_log(card, mode)
    if log:
        print("Appended", log)
    if not demo:
        publish.update_clv(live.closing_lines(), games_raw=names_and_results())
        print("Wrote", publish.write_record())

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

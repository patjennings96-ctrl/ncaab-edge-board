"""
season_backtest.py
==================
Replay a past season exactly the way the live board runs, one week at a time,
and grade every play against real closing lines.

For each week starting on day D:
  1. Games before D only: last season (down-weighted, fading) + this season so far,
     Division I games only, ridge alpha 3.
  2. Train the dual model the same way run_daily does.
  3. Price that week's games against DraftKings' closing spread and total
     (from ESPN's pickcenter), with the same sigma floors, 2.5% edge threshold
     and quarter-Kelly stakes as the live card.
  4. Grade with the final score.

Also scores a "ratings only" variant (the raw efficiency projection, no ML layer)
and the closing line itself as a forecast, so the model is compared with the market.

Usage (needs internet; runs on GitHub Actions):
    python season_backtest.py --season 2026
Outputs:
    data/lines/<season>.csv               closing lines per game (cached)
    docs/data/backtest_<season>.json      summary + weekly + per-bet detail
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import live
import model as M
import ratings as R
from edge import recommend_spread, recommend_total, american_to_decimal
from run_daily import FEATURES, build_training_frame, SIGMA_FLOOR_MARGIN, SIGMA_FLOOR_TOTAL

warnings.filterwarnings("ignore")
ROOT = live.ROOT
LINES_DIR = os.path.join(ROOT, "data", "lines")


# -----------------------------------------------------------------------------
# Historical lines from ESPN pickcenter
# -----------------------------------------------------------------------------
def _num(v):
    """'+1.5' -> 1.5, 'o156.5' -> 156.5, 'EVEN' -> 100, '-110' -> -110."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("even", "ev"):
        return 100.0
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def espn_lines(game_id: str) -> dict:
    """Closing (and, when listed, opening) spread/total for one game."""
    data = live._get_json(f"{live.ESPN}/summary", {"event": game_id})
    picks = data.get("pickcenter") or []
    if not picks:
        return {"game_id": game_id}
    pc = picks[0]
    home, away = pc.get("homeTeamOdds") or {}, pc.get("awayTeamOdds") or {}

    def home_line(stage):
        v = _num(_dig(home, stage, "pointSpread", "american")) \
            or _num(_dig(home, stage, "pointSpread", "alternateDisplayValue"))
        return v

    spread_abs = _num(pc.get("spread"))
    close_sp = home_line("close")
    if close_sp is None and spread_abs is not None:
        # ESPN's 'spread' is unsigned on some feeds; sign it from the favorite flag
        if home.get("favorite") is True:
            close_sp = -abs(spread_abs)
        elif away.get("favorite") is True:
            close_sp = abs(spread_abs)
        else:
            close_sp = spread_abs
    total = _num(_dig(pc, "close", "over", "line")) or _num(pc.get("overUnder"))

    return {
        "game_id": game_id,
        "provider": _dig(pc, "provider", "name"),
        "close_spread_home": close_sp,
        "close_total": total,
        "open_spread_home": home_line("open"),
        "open_total": _num(_dig(pc, "open", "over", "line")),
        "home_spread_odds": _num(_dig(home, "close", "spread", "american")) or _num(home.get("spreadOdds")),
        "away_spread_odds": _num(_dig(away, "close", "spread", "american")) or _num(away.get("spreadOdds")),
        "over_odds": _num(_dig(pc, "close", "over", "american")) or _num(pc.get("overOdds")),
        "under_odds": _num(_dig(pc, "close", "under", "american")) or _num(pc.get("underOdds")),
    }


def load_lines(season: int, game_ids: list[str], workers: int = 8) -> pd.DataFrame:
    os.makedirs(LINES_DIR, exist_ok=True)
    path = os.path.join(LINES_DIR, f"{season}.csv")
    have = pd.read_csv(path, dtype={"game_id": str}) if os.path.exists(path) else pd.DataFrame()
    done = set(have["game_id"]) if not have.empty else set()
    todo = [g for g in game_ids if g not in done]
    print(f"[lines {season}] {len(done)} cached, fetching {len(todo)}")

    def one(gid):
        try:
            return espn_lines(gid)
        except Exception as exc:
            print(f"  lines failed {gid}: {exc}")
            return {"game_id": gid}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, todo)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)}")
    allr = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if rows else have
    allr.to_csv(path, index=False)
    return allr


# -----------------------------------------------------------------------------
# Walk-forward replay
# -----------------------------------------------------------------------------
def _d1(long: pd.DataFrame) -> pd.DataFrame:
    n = long.groupby("team").size()
    keep = set(n[n >= 10].index)
    return long[long["team"].isin(keep) & long["opp"].isin(keep)]


def _price(pm, sm, pt, st, r):
    """Card logic from run_daily.make_card, applied to one game row."""
    out = []
    if pd.notna(r.close_spread_home):
        sp = recommend_spread(pm, max(sm, SIGMA_FLOOR_MARGIN), r.close_spread_home,
                              r.home_spread_odds, r.away_spread_odds,
                              kelly_mult=0.25, min_edge=0.025)
        if sp:
            out.append(sp)
    if pd.notna(r.close_total):
        tt = recommend_total(pt, max(st, SIGMA_FLOOR_TOTAL), r.close_total,
                             r.over_odds, r.under_odds, kelly_mult=0.25, min_edge=0.025)
        if tt:
            out.append(tt)
    return out


def _grade(rec, margin, total):
    if rec.market == "spread":
        x = (margin if rec.side == "HOME" else -margin) + rec.line
    else:
        x = (total - rec.line) * (1 if rec.side == "OVER" else -1)
    return 1 if x > 0 else (0 if x < 0 else -1)


def replay(season: int, start: str | None = None, step_days: int = 7) -> dict:
    prev_raw, cur_raw = live.load_games(season - 1), live.load_games(season)
    names = live.team_names(prev_raw, cur_raw)
    prev_long = R.add_tempo_efficiency(live.to_long(prev_raw)) if not prev_raw.empty else pd.DataFrame()
    cur_long = R.add_tempo_efficiency(live.to_long(cur_raw))
    cur_long["date"] = pd.to_datetime(cur_long["date"])

    lines = load_lines(season, list(cur_raw["game_id"].astype(str)))
    games = cur_raw.merge(lines, on="game_id", how="left")
    games["date"] = pd.to_datetime(games["date"])
    for c, d in [("home_spread_odds", -110), ("away_spread_odds", -110),
                 ("over_odds", -110), ("under_odds", -110)]:
        games[c] = games[c].fillna(d)
    has_line = games["close_spread_home"].notna() | games["close_total"].notna()
    print(f"[replay] {len(games)} games, {int(has_line.sum())} with closing lines")

    first = pd.to_datetime(start) if start else games["date"].min() + pd.Timedelta(days=7)
    cuts = pd.date_range(first, games["date"].max(), freq=f"{step_days}D")

    bets, preds, weekly = [], [], []
    for cut in cuts:
        t0 = time.time()
        hist_cur = cur_long[cur_long["date"] < cut].copy()
        n_avg = len(hist_cur) / max(hist_cur["team"].nunique(), 1) if not hist_cur.empty else 0
        w_prev = 0.5 * 12 / (12 + n_avg)
        parts = []
        if not prev_long.empty:
            p = prev_long.copy(); p["weight"] = w_prev; parts.append(p)
        hist_cur["weight"] = 1.0; parts.append(hist_cur)
        hist = _d1(pd.concat(parts, ignore_index=True))
        teams_ok = set(hist["team"])

        ratings = R.fit_adjusted_ratings(hist, alpha=3.0)
        dual = M.DualModel.train(build_training_frame(hist, ratings), FEATURES)

        wk = games[(games["date"] >= cut) & (games["date"] < cut + pd.Timedelta(days=step_days))]
        wk = wk[wk["home_id"].isin(teams_ok) & wk["away_id"].isin(teams_ok)]
        wk = wk.dropna(subset=["home_score", "away_score"])
        if wk.empty:
            continue

        feats = []
        for r in wk.itertuples(index=False):
            neutral = bool(r.neutral)
            pr = ratings.project(r.home_id, r.away_id, neutral=neutral)
            feats.append({
                "proj_margin": pr["proj_margin"], "proj_total": pr["proj_total"],
                "proj_poss": pr["proj_poss"],
                "adj_em_diff": ratings.adj_em(r.home_id) - ratings.adj_em(r.away_id),
                "tempo_sum": ratings.team_tempo(r.home_id) + ratings.team_tempo(r.away_id),
                "custom_hca": 0.0 if neutral else ratings.hca,
            })
        fx = pd.DataFrame(feats)
        mm, ms = dual.margin.predict_with_sigma(fx)
        tm, ts = dual.total.predict_with_sigma(fx)

        wk_bets = 0
        for i, r in enumerate(wk.itertuples(index=False)):
            margin = r.home_score - r.away_score
            total = r.home_score + r.away_score
            preds.append({
                "game_id": r.game_id, "date": r.date.date().isoformat(),
                "margin": margin, "total": total,
                "model_margin": float(mm[i]), "model_total": float(tm[i]),
                "ratings_margin": float(fx.proj_margin[i]), "ratings_total": float(fx.proj_total[i]),
                "close_spread_home": r.close_spread_home, "close_total": r.close_total,
                "month": r.date.strftime("%Y-%m"),
            })
            for variant, pm, pt in (("model", mm[i], tm[i]),
                                    ("ratings_only", fx.proj_margin[i], fx.proj_total[i])):
                for rec in _price(pm, ms[i], pt, ts[i], r):
                    res = _grade(rec, margin, total)
                    bets.append({
                        "variant": variant, "date": r.date.date().isoformat(),
                        "month": r.date.strftime("%Y-%m"),
                        "game_id": r.game_id,
                        "matchup": f"{names.get(r.away_id, r.away_id)} @ {names.get(r.home_id, r.home_id)}",
                        "market": rec.market, "side": rec.side, "line": rec.line,
                        "odds": rec.odds, "edge_pct": round(100 * rec.edge, 2),
                        "stake": round(rec.stake_units, 3), "result": res,
                        "margin": margin, "total": total,
                    })
                    wk_bets += variant == "model"
        weekly.append({"week_of": cut.date().isoformat(), "games": len(wk), "bets": wk_bets,
                       "prior_weight": round(w_prev, 3), "secs": round(time.time() - t0, 1)})
        print(f"  week {cut.date()}: {len(wk)} games, {wk_bets} model bets, prior w={w_prev:.2f}")

    return {"bets": pd.DataFrame(bets), "preds": pd.DataFrame(preds), "weekly": weekly}


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def _record(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    w, l, p = int((b.result == 1).sum()), int((b.result == 0).sum()), int((b.result == -1).sum())
    dec = b["odds"].map(lambda o: american_to_decimal(o) - 1)
    flat = np.where(b.result == 1, dec, np.where(b.result == 0, -1.0, 0.0))
    kel = flat * b["stake"]
    return {
        "bets": int(len(b)), "w": w, "l": l, "p": p,
        "win_pct": round(w / max(w + l, 1), 4),
        "flat_units": round(float(flat.sum()), 2),
        "flat_roi": round(float(flat.sum() / len(b)), 4),
        "kelly_units": round(float(kel.sum()), 2),
        "kelly_roi": round(float(kel.sum() / max(b["stake"].sum(), 1e-9)), 4),
        "avg_edge": round(float(b["edge_pct"].mean()), 2),
    }


def _bankroll(b: pd.DataFrame, start=100.0) -> tuple[list, float]:
    """Kelly stakes as % of current bankroll (1u = 1%). Daily curve + max drawdown."""
    bank, peak, dd, curve = start, start, 0.0, []
    for d, day in b.sort_values("date").groupby("date"):
        pnl = 0.0
        for r in day.itertuples():
            stake = r.stake / 100 * bank
            if r.result == 1:
                pnl += stake * (american_to_decimal(r.odds) - 1)
            elif r.result == 0:
                pnl -= stake
        bank += pnl
        peak = max(peak, bank)
        dd = max(dd, (peak - bank) / peak)
        curve.append({"date": d, "bankroll": round(bank, 2)})
    return curve, round(dd, 4)


def summarize(season: int, out: dict) -> dict:
    bets, preds = out["bets"], out["preds"]
    rmse = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 2)
    mae = lambda e: round(float(np.mean(np.abs(e))), 2)
    acc = {}
    sp = preds.dropna(subset=["close_spread_home"])
    to = preds.dropna(subset=["close_total"])
    acc["games_with_spread"] = int(len(sp))
    acc["games_with_total"] = int(len(to))
    acc["margin"] = {
        "model_rmse": rmse(sp.margin - sp.model_margin), "model_mae": mae(sp.margin - sp.model_margin),
        "ratings_rmse": rmse(sp.margin - sp.ratings_margin),
        "market_rmse": rmse(sp.margin + sp.close_spread_home),
        "market_mae": mae(sp.margin + sp.close_spread_home),
    }
    acc["total"] = {
        "model_rmse": rmse(to.total - to.model_total), "model_mae": mae(to.total - to.model_total),
        "ratings_rmse": rmse(to.total - to.ratings_total),
        "market_rmse": rmse(to.total - to.close_total),
        "market_mae": mae(to.total - to.close_total),
    }

    res = {"season": season, "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "rules": {"bet_line": "DraftKings closing line via ESPN", "min_edge_pct": 2.5,
                     "kelly_mult": 0.25, "max_stake_units": 5.0,
                     "sigma_floor_margin": SIGMA_FLOOR_MARGIN, "sigma_floor_total": SIGMA_FLOOR_TOTAL},
           "accuracy": acc, "weekly": out["weekly"], "variants": {}}

    for v in ("model", "ratings_only"):
        b = bets[bets.variant == v] if not bets.empty else bets
        block = {"all": _record(b),
                 "spread": _record(b[b.market == "spread"]) if not b.empty else {},
                 "total": _record(b[b.market == "total"]) if not b.empty else {}}
        if not b.empty:
            buckets = pd.cut(b.edge_pct, [2.5, 5, 10, 20, 1000], right=False,
                             labels=["2.5-5%", "5-10%", "10-20%", "20%+"])
            block["by_edge"] = {str(k): _record(g) for k, g in b.groupby(buckets, observed=True)}
            block["by_month"] = {k: _record(g) for k, g in b.groupby("month")}
            block["curve"], block["max_drawdown"] = _bankroll(b)
        res["variants"][v] = block
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=live.current_season() - 1)
    ap.add_argument("--start", default=None, help="first week, YYYY-MM-DD")
    args = ap.parse_args()

    live.update_games(args.season - 1)        # prior season, for the early-season blend
    live.update_games(args.season)
    out = replay(args.season, start=args.start)
    res = summarize(args.season, out)

    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "data", f"backtest_{args.season}.json")
    if not out["bets"].empty:
        res["bets"] = out["bets"][out["bets"].variant == "model"].drop(columns="variant") \
            .to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote", path)
    print(json.dumps({k: res[k] for k in ("accuracy",)}, indent=1))
    for v, blk in res["variants"].items():
        print(v, json.dumps({k: blk.get(k) for k in ("all", "spread", "total", "max_drawdown")}))


if __name__ == "__main__":
    main()

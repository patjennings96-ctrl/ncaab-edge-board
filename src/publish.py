"""
publish.py
==========
Write the day's card where the website and the history log can read it.

* docs/data/plays.json      -> read by the GitHub Pages board (docs/index.html)
* data/history/card_log.csv -> every play ever posted, with the line at posting
                               time; fill in closing lines later to track CLV.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import live

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = os.path.join(ROOT, "docs", "data", "plays.json")
LOG_PATH = os.path.join(ROOT, "data", "history", "card_log.csv")


def _side_label(row) -> str:
    """Spread sides become team names; totals stay OVER/UNDER."""
    if row["mkt"] == "SPREAD":
        return row["home"] if row["side"] == "HOME" else row["away"]
    return row["side"]


# "Best bet" tier: the one filter that held up in both halves of last season's
# Jan 1 - Mar 8 backtest (93-66, 58.5%, +20u; Jan 1-Feb 4 58.7%, Feb 5-Mar 8
# 57.6%). Edge Board spreads only: 4+ points off the line, taking the underdog,
# spread under 15. It lost in Nov-Dec (80-87), so it only applies Jan 1 - Mar 8.
BEST_GAP, BEST_MAX_LINE = 4.0, 15.0


def is_best_bet(mkt, line, gap, start) -> bool:
    try:
        line, gap = float(line), float(gap)
    except (TypeError, ValueError):
        return False
    d = pd.to_datetime(start, utc=True, errors="coerce")
    if pd.isna(d) or mkt != "SPREAD":
        return False
    d = d.tz_convert("America/New_York")
    in_window = d.month in (1, 2) or (d.month == 3 and d.day <= 8)
    return in_window and gap >= BEST_GAP and 0 < line < BEST_MAX_LINE


def card_to_records(card: pd.DataFrame) -> list[dict]:
    recs = []
    for _, r in card.iterrows():
        recs.append({
            "game_id": str(r.get("game_id", "") or ""), "start": str(r.get("start", "") or ""),
            "home": r["home"], "away": r["away"], "mkt": r["mkt"],
            "side": _side_label(r), "side_code": r["side"],
            "line": float(r["line"]), "odds": int(r["odds"]),
            "proj": float(r["proj_margin"]), "total": float(r["proj_total"]),
            "p": float(r["model_prob"]), "edge": float(r["edge_pct"]),
            "stake": float(r["stake_units"]),
            "notes": str(r.get("notes", "") or ""),
            "best_line": None if pd.isna(r.get("best_line", np.nan)) else float(r["best_line"]),
            "best_odds": None if pd.isna(r.get("best_odds", np.nan)) else int(r["best_odds"]),
            "book": str(r.get("best_book", "") or ""),
            "agree": bool(r.get("agree", False)),
            "play_to": None if pd.isna(r.get("play_to", np.nan)) else float(r["play_to"]),
            "tracking": float(r["stake_units"]) == 0.0,
            "to_win": float(r.get("to_win", 1.0) if pd.notna(r.get("to_win", 1.0)) else 1.0),
            "reason": str(r.get("reason", "") or ""),
            "gap": None if pd.isna(r.get("gap", np.nan)) else float(r["gap"]),
            "best": is_best_bet(r["mkt"], r.get("best_line", r["line"]) if pd.notna(r.get("best_line", np.nan)) else r["line"],
                                r.get("gap", np.nan), r.get("start", "")),
            "quotes": json.loads(r["quotes"]) if isinstance(r.get("quotes"), str) and r.get("quotes") else [],
        })
    return recs


def write_json(card: pd.DataFrame, mode: str, settings: dict) -> str:
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "mode": mode,                      # "demo" or "live"
        "settings": settings,
        "plays": card_to_records(card) if not card.empty else [],
    }
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    return JSON_PATH


def attach_similar() -> None:
    """Final pass on both plays.json files: drop plays on voided days, then add
    "how have similar plays done" records."""
    import similar
    e = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else None
    c = json.load(open(CAL_JSON_PATH)) if os.path.exists(CAL_JSON_PATH) else None
    for obj, key, model in ((e, "plays", "edge"), (c, "games", "cal")):
        if obj is None:
            continue
        days = void_days(model)
        rows = obj.get(key, [])
        keep = ~az_date(pd.Series([r.get("start") for r in rows], dtype=object)).isin(days) if rows else []
        obj[key] = [r for r, k in zip(rows, keep) if k]
        obj["void_days"] = sorted(days)
    try:
        similar.attach((e or {}).get("plays", []), (c or {}).get("games", []), similar.pool())
    except Exception as exc:
        print(f"[similar] skipped: {exc}")
    for path, obj in ((JSON_PATH, e), (CAL_JSON_PATH, c)):
        if obj is not None:
            with open(path, "w") as f:
                json.dump(obj, f, indent=2)


def append_log(card: pd.DataFrame, mode: str) -> str | None:
    """Append live plays only, so demo runs never pollute the CLV record."""
    if mode != "live" or card.empty:
        return None
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df = pd.DataFrame(card_to_records(card)).drop(columns=["quotes"], errors="ignore")
    # Log each (game, market) once: the first time the model posted it.
    if os.path.exists(LOG_PATH):
        seen = pd.read_csv(LOG_PATH, dtype={"game_id": str})[["game_id", "mkt"]]
        key = df["game_id"] + "|" + df["mkt"]
        df = df[~key.isin(seen["game_id"] + "|" + seen["mkt"])]
    if df.empty:
        return None
    df.insert(0, "posted_at", datetime.now(timezone.utc).isoformat(timespec="minutes"))
    df["close_line"] = ""                  # filled after tip-off
    df["clv_points"] = ""
    df["result"] = ""
    if os.path.exists(LOG_PATH):
        # rewrite rather than append so new columns (best book, agree) stay aligned
        old = pd.read_csv(LOG_PATH, dtype=str)
        df = pd.concat([old, df.fillna("").astype(str)], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return LOG_PATH


def update_clv(closes: pd.DataFrame, games_raw: pd.DataFrame) -> None:
    """
    Fill closing lines, CLV (points) and results into card_log.csv.

    Spread CLV = your line minus the closing line for the same side
                 (took +5, closed +3 -> +2 points of value).
    Total CLV  = OVER: close - line;  UNDER: line - close.
    """
    if not os.path.exists(LOG_PATH):
        return
    log = pd.read_csv(LOG_PATH, dtype={"game_id": str})
    for col in ("close_line", "clv_points", "result"):
        if col not in log.columns:
            log[col] = ""
        log[col] = log[col].astype(object)

    close = closes.set_index("game_id") if not closes.empty else pd.DataFrame()
    res = games_raw.assign(game_id=games_raw["game_id"].astype(str)).drop_duplicates("game_id", keep="last") \
        .set_index("game_id") if not games_raw.empty else pd.DataFrame()

    for i, r in log.iterrows():
        gid = str(r["game_id"])
        if not close.empty and gid in close.index and str(r["close_line"]) in ("", "nan"):
            c = close.loc[gid]
            if r["mkt"] == "SPREAD":
                cl = c["spread_home"] if r["side_code"] == "HOME" else -c["spread_home"]
                clv = r["line"] - cl
            else:
                cl = c["total"]
                clv = (cl - r["line"]) if r["side_code"] == "OVER" else (r["line"] - cl)
            if pd.notna(cl):
                log.at[i, "close_line"] = round(float(cl), 1)
                log.at[i, "clv_points"] = round(float(clv), 1)
        if not res.empty and gid in res.index and str(r["result"]) in ("", "nan"):
            g = res.loc[gid]
            if pd.isna(g["home_score"]):
                continue
            hs, as_ = live.oriented_scores(g, r["home"], r["away"])
            margin, total = hs - as_, hs + as_
            if r["mkt"] == "SPREAD":
                m = margin if r["side_code"] == "HOME" else -margin
                x = m + float(r["line"])
            else:
                x = (total - float(r["line"])) * (1 if r["side_code"] == "OVER" else -1)
            log.at[i, "result"] = "W" if x > 0 else ("P" if x == 0 else "L")
    log.to_csv(LOG_PATH, index=False)


# -----------------------------------------------------------------------------
# Season record for the front page (both models)
# -----------------------------------------------------------------------------
RECORD_PATH = os.path.join(ROOT, "docs", "data", "record.json")
CAL_LOG_PATH = os.path.join(ROOT, "data", "history", "cal_log.csv")


def _season_start(now: datetime | None = None) -> str:
    """Seasons run Nov-Apr; anything posted from Oct 1 belongs to the new season."""
    now = now or datetime.now(timezone.utc)
    y = now.year if now.month >= 10 else now.year - 1
    return f"{y}-10-01"


def _payout(odds) -> float:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        o = -110.0
    if not np.isfinite(o) or o == 0:
        o = -110.0
    return o / 100.0 if o > 0 else 100.0 / -o


def _summ(df: pd.DataFrame, units_col: str = "units") -> dict:
    """W-L-P, units, ROI and CLV. ROI is per unit risked: the stake for real
    stakes, 1 per play for flat 1u records (tracking, both-models)."""
    u = pd.to_numeric(df[units_col], errors="coerce")
    graded = df[u.notna()]
    res = graded["result"] if "result" in graded else np.sign(u[u.notna()]).map({1: "W", -1: "L", 0: "P"})
    out = {"w": int((res == "W").sum()), "l": int((res == "L").sum()), "p": int((res == "P").sum()),
           "units": round(float(u.sum()), 2), "pending": int(len(df) - len(graded))}
    if units_col == "flat" and "odds" in df:
        o = pd.to_numeric(df.get("best_odds", df["odds"]), errors="coerce").fillna(pd.to_numeric(df["odds"], errors="coerce"))
        tw = df["to_win_n"].loc[graded.index] if "to_win_n" in df else 1.0
        risk = float((tw / o.loc[graded.index].map(_payout)).sum())
        out["roi"] = round(out["units"] / risk, 4) if risk > 0 else None
    elif units_col != "units" or "risked" not in df:
        out["roi"] = round(out["units"] / len(graded), 4) if len(graded) else None
    else:
        risk = float(pd.to_numeric(graded["risked"], errors="coerce").sum())
        out["roi"] = round(out["units"] / risk, 4) if risk > 0 else None
    if "clv_points" in df:
        clv = pd.to_numeric(df["clv_points"], errors="coerce").dropna()
        out["clv"] = round(float(clv.mean()), 2) if len(clv) else None
        out["beat_close"] = round(float((clv > 0).mean()), 3) if len(clv) else None
    return out


# Pat's rule: Edge Board spreads over 25.5 points (26+) are off the board. They stay in
# card_log.csv (graded, for the twice-monthly review of the cap) but are left
# out of every Edge Board record and check.
BIG_SPREAD = 25.5


# Days Pat voided (data/history/void_days.csv: date, model edge|cal|both,
# reason). Dates are Arizona dates of tip-off, as the site groups them. Voided
# plays stay in the logs but are dropped from the board, every record and check.
VOID_PATH = os.path.join(ROOT, "data", "history", "void_days.csv")


def void_days(model: str) -> set[str]:
    if not os.path.exists(VOID_PATH):
        return set()
    v = pd.read_csv(VOID_PATH, dtype=str).fillna("")
    return set(v.loc[v["model"].str.lower().isin([model, "both"]), "date"].str.strip())


def az_date(start) -> pd.Series:
    st = pd.to_datetime(start, utc=True, errors="coerce")
    return st.dt.tz_convert("America/Phoenix").dt.strftime("%Y-%m-%d")


def drop_void(log: pd.DataFrame, model: str) -> pd.DataFrame:
    days = void_days(model)
    if not days or log.empty or "start" not in log:
        return log
    return log[~az_date(log["start"]).isin(days)]


def is_big_spread(log: pd.DataFrame) -> pd.Series:
    line = pd.to_numeric(log.get("best_line", log["line"]), errors="coerce").fillna(
        pd.to_numeric(log["line"], errors="coerce"))
    return (log["mkt"] == "SPREAD") & (line.abs() > BIG_SPREAD)


def edge_record(since: str) -> dict:
    if not os.path.exists(LOG_PATH):
        return {}
    log = drop_void(pd.read_csv(LOG_PATH, dtype={"game_id": str}), "edge")
    log = log[log["posted_at"].astype(str) >= since].copy()
    if log.empty:
        return {}
    big = is_big_spread(log)
    big_res = log.loc[big, "result"].astype(str)
    big_summary = {"w": int((big_res == "W").sum()), "l": int((big_res == "L").sum()),
                   "p": int((big_res == "P").sum())}
    log = log[~big].copy()
    if log.empty:
        return {"big_spreads": big_summary}
    log["stake"] = pd.to_numeric(log.get("stake"), errors="coerce").fillna(0.0)
    odds = log["best_odds"] if "best_odds" in log else log["odds"]
    odds = pd.to_numeric(odds, errors="coerce").fillna(pd.to_numeric(log["odds"], errors="coerce"))
    res = log["result"].astype(str)
    log["units"] = np.where(res == "W", log["stake"] * odds.map(_payout),
                   np.where(res == "L", -log["stake"], np.where(res == "P", 0.0, np.nan)))
    log["risked"] = np.where(res.isin(["W", "L", "P"]), log["stake"], np.nan)
    # tracking plays (0u) are graded as a bet to win 1u (risk 1.1u at -110),
    # the same rule as staked plays and Cal's model
    risk1 = 1.0 / odds.map(_payout)
    tw = pd.to_numeric(log.get("to_win", pd.Series(1.0, index=log.index)), errors="coerce").fillna(1.0)
    log["to_win_n"] = tw
    log["flat"] = np.where(res == "W", tw, np.where(res == "L", -risk1 * tw,
                  np.where(res == "P", 0.0, np.nan)))
    staked, tracking = log[log["stake"] > 0], log[log["stake"] <= 0]
    agree = log[log.get("agree", pd.Series(False, index=log.index)).astype(str).str.lower() == "true"]
    out = {"staked": {m: _summ(g) for m, g in staked.groupby("mkt")},
           "tracking": {m: _summ(g, "flat") for m, g in tracking.groupby("mkt")},
           "both_models": {m: _summ(g, "flat") for m, g in agree.groupby("mkt")},
           "all": _summ(staked), "big_spreads": big_summary}
    best = log[[is_best_bet(m, bl if pd.notna(bl) else l, g, s) for m, bl, l, g, s in
                zip(log["mkt"], pd.to_numeric(log.get("best_line"), errors="coerce"), log["line"],
                    pd.to_numeric(log.get("gap"), errors="coerce"), log["start"])]]
    out["best_bets"] = _summ(best, "flat") if not best.empty else {"w": 0, "l": 0, "p": 0}
    rec = log[res.isin(["W", "L", "P"])].sort_values("start", ascending=False).head(40)
    out["recent"] = [{"start": str(r.start), "away": r.away, "home": r.home, "mkt": r.mkt,
                      "side": str(r.side), "line": float(r.line), "stake": float(r.stake),
                      "result": r.result, "units": None if pd.isna(r.units) else round(float(r.units), 2),
                      "clv": None if str(r.clv_points) in ("", "nan") else float(r.clv_points)}
                     for r in rec.itertuples(index=False)]
    return out


def cal_record(since: str) -> dict:
    if not os.path.exists(CAL_LOG_PATH):
        return {}
    log = drop_void(pd.read_csv(CAL_LOG_PATH, dtype={"game_id": str, "units": str}).fillna({"units": ""}), "cal")
    log = log[log["posted_at"].astype(str) >= since].copy()
    if log.empty:
        return {}
    log["units_n"] = pd.to_numeric(log["units"], errors="coerce")
    out = {"record": {m: _summ(g, "units_n") for m, g in log.groupby("market")},
           "all": _summ(log, "units_n")}
    rec = log[log.units_n.notna()].sort_values("start", ascending=False).head(40)
    out["recent"] = [{"start": str(r.start), "away": r.away, "home": r.home, "mkt": r.market,
                      "play": r.play, "line": None if pd.isna(r.line) else float(r.line),
                      "total_line": None if pd.isna(r.total_line) else float(r.total_line),
                      "units": float(r.units_n)} for r in rec.itertuples(index=False)]
    return out


def write_record() -> str:
    since = _season_start()
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "season_start": since, "edge": edge_record(since), "cal": cal_record(since),
               "check": model_check(since)}
    os.makedirs(os.path.dirname(RECORD_PATH), exist_ok=True)
    with open(RECORD_PATH, "w") as f:
        json.dump(payload, f, indent=1)
    return RECORD_PATH


# -----------------------------------------------------------------------------
# Model check: is the staking calibration holding up? (both models, all plays)
# -----------------------------------------------------------------------------
EDGE_BUCKETS = {"SPREAD": [2, 3, 4, 5, 99], "TOTAL": [3, 4, 5, 7, 99]}
CAL_BUCKETS = {"ATS": [5, 7, 10, 99], "TOTAL": [7, 9, 12, 99]}


def _wl(res: pd.Series) -> dict:
    w, l = int((res == "W").sum()), int((res == "L").sum())
    return {"w": w, "l": l, "pct": round(w / (w + l), 3) if w + l else None}


def _label(lo, hi):
    return f"{lo:g}+" if hi >= 99 else f"{lo:g}-{hi:g}"


def _side_split(result: pd.Series, fav: pd.Series, size: pd.Series) -> dict:
    """Favorites vs underdogs, and by size of the spread."""
    out = {"fav": _wl(result[fav]), "dog": _wl(result[~fav])}
    for lo, hi, lab in ((0, 5, "under 5"), (5, 15, "5-15"), (15, 99, "15+")):
        out[lab] = _wl(result[(size >= lo) & (size < hi)])
    return out


def model_check(since: str) -> dict:
    out = {"edge": {}, "cal": {}}
    if os.path.exists(LOG_PATH):
        log = drop_void(pd.read_csv(LOG_PATH, dtype={"game_id": str}), "edge")
        log = log[(log["posted_at"].astype(str) >= since) & log["result"].astype(str).isin(["W", "L", "P"])]
        log = log[~is_big_spread(log)] if not log.empty else log
        if not log.empty and "gap" in log:
            log["gap"] = pd.to_numeric(log["gap"], errors="coerce")
            log["p"] = pd.to_numeric(log.get("p"), errors="coerce")
            agree = log.get("agree", pd.Series("", index=log.index)).astype(str).str.lower() == "true"
            for mkt, edges in EDGE_BUCKETS.items():
                g = log[log["mkt"] == mkt]
                rows = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    b = g[(g["gap"] >= lo) & (g["gap"] < hi)]
                    clv = pd.to_numeric(b.get("clv_points"), errors="coerce").dropna()
                    rows.append({"gap": _label(lo, hi), **_wl(b["result"]),
                                 "expected": round(float(b["p"].mean()), 3) if b["p"].notna().any() else None,
                                 "clv": round(float(clv.mean()), 2) if len(clv) else None})
                out["edge"][mkt] = {"buckets": rows,
                                    "agree": _wl(g[agree.loc[g.index]]["result"]),
                                    "no_agree": _wl(g[~agree.loc[g.index]]["result"])}
                if mkt == "SPREAD" and not g.empty:
                    ln = pd.to_numeric(g["line"], errors="coerce")
                    out["edge"][mkt]["sides"] = _side_split(g["result"], ln < 0, ln.abs())
    if os.path.exists(CAL_LOG_PATH):
        log = drop_void(pd.read_csv(CAL_LOG_PATH, dtype={"game_id": str, "units": str}).fillna({"units": ""}), "cal")
        log = log[log["posted_at"].astype(str) >= since]
        u = pd.to_numeric(log["units"], errors="coerce")
        log = log[u.notna()].assign(result=np.where(u[u.notna()] > 0, "W", np.where(u[u.notna()] < 0, "L", "P")))
        if not log.empty and "gap" in log:
            log["gap"] = pd.to_numeric(log["gap"], errors="coerce")
            for mkt, edges in CAL_BUCKETS.items():
                g = log[log["market"] == mkt]
                if mkt == "ATS" and not g.empty:
                    ln = pd.to_numeric(g["line"], errors="coerce")          # home line
                    fav = (ln < 0) == (g["play"] == "HOME")
                    side_split = _side_split(g["result"], fav, ln.abs())
                out["cal"][mkt] = {"buckets": [{"gap": _label(lo, hi), **_wl(g[(g["gap"] >= lo) & (g["gap"] < hi)]["result"])}
                                               for lo, hi in zip(edges[:-1], edges[1:])]}
                if mkt == "ATS" and not g.empty:
                    out["cal"][mkt]["sides"] = side_split
    return out


# -----------------------------------------------------------------------------
# Full slate: every game with lines, both models' numbers, plays or not
# -----------------------------------------------------------------------------
SLATE_PATH = os.path.join(ROOT, "docs", "data", "slate.json")


def _f(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


def write_slate(preds, cal_priced, card, names: dict) -> str:
    games = []
    cal = cal_priced.set_index(cal_priced["game_id"].astype(str)) \
        if cal_priced is not None and not cal_priced.empty else pd.DataFrame()
    plays = set()
    if card is not None and not card.empty:
        plays = set(card["game_id"].astype(str) + "|" + card["mkt"])
    if preds is not None and not preds.empty:
        for r in preds.itertuples(index=False):
            gid = str(getattr(r, "game_id", ""))
            g = {"game_id": gid, "start": str(getattr(r, "start", "")),
                 "home": names.get(r.home, r.home), "away": names.get(r.away, r.away),
                 "spread_home": _f(r.spread_home), "total": _f(r.total),
                 "edge": {"margin": _f(r.pred_margin), "total": _f(r.pred_total),
                          "spread_play": gid + "|SPREAD" in plays, "total_play": gid + "|TOTAL" in plays},
                 "book": str(getattr(r, "best_home_book", "") or "")}
            if not cal.empty and gid in cal.index:
                c = cal.loc[gid]
                g["cal"] = {"m1_margin": _f(c.m1_home - c.m1_away), "m2_margin": _f(c.m2_home - c.m2_away),
                            "m1_total": _f(c.m1_home + c.m1_away), "m2_total": _f(c.m2_home + c.m2_away),
                            "ats_play": str(c.ats_play or ""), "ou_play": str(c.ou_play or "")}
            games.append(g)
    games.sort(key=lambda g: (g["start"], g["home"]))
    with open(SLATE_PATH, "w") as f:
        json.dump({"updated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                   "games": games}, f, indent=1)
    return SLATE_PATH


# -----------------------------------------------------------------------------
# Plays that have tipped: kept on the board (with live / final status filled in
# by the page from ESPN) until the next day is well under way
# -----------------------------------------------------------------------------
CAL_JSON_PATH = os.path.join(ROOT, "docs", "cal", "data", "plays.json")


def _num(x, default=None):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _recent_window(df: pd.DataFrame, col: str = "start") -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    st = pd.to_datetime(df[col], utc=True, errors="coerce")
    # whole days, as the site groups them (Arizona dates): yesterday and today
    since = now.tz_convert("America/Phoenix").normalize() - pd.Timedelta(days=1)
    return df[(st <= now) & (st >= since)]


def recent_edge_plays() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    log = _recent_window(drop_void(pd.read_csv(LOG_PATH, dtype={"game_id": str}), "edge"))
    out = []
    for r in log.itertuples(index=False):
        g = r._asdict()
        odds = _num(g.get("best_odds")) or _num(g.get("odds"), -110)
        tw = _num(g.get("to_win"), 1.0) or 1.0
        res = str(g.get("result") or "")
        units = tw if res == "W" else (-tw / _payout(odds) if res == "L" else (0.0 if res == "P" else None))
        out.append({"game_id": str(g["game_id"]), "start": str(g["start"]), "home": g["home"], "away": g["away"],
                    "mkt": g["mkt"], "side": str(g["side"]), "side_code": g["side_code"],
                    "line": _num(g["line"]), "best_line": _num(g.get("best_line")), "odds": _num(g.get("odds")),
                    "best_odds": _num(g.get("best_odds")), "book": "" if pd.isna(g.get("book")) else str(g.get("book")),
                    "stake": _num(g.get("stake"), 0.0), "to_win": tw, "p": _num(g.get("p")), "edge": _num(g.get("edge")),
                    "agree": str(g.get("agree")).lower() == "true", "tracking": _num(g.get("stake"), 0.0) == 0.0,
                    "reason": "" if pd.isna(g.get("reason")) else str(g.get("reason")),
                    "result": res if res in ("W", "L", "P") else "", "units": None if units is None else round(units, 2),
                    "started": True})
    return out


def recent_cal_plays() -> list[dict]:
    if not os.path.exists(CAL_LOG_PATH):
        return []
    log = _recent_window(drop_void(pd.read_csv(CAL_LOG_PATH, dtype={"game_id": str, "units": str}).fillna({"units": ""}), "cal"))
    out = []
    for r in log.itertuples(index=False):
        u = _num(r.units)
        out.append({"game_id": str(r.game_id), "start": str(r.start), "home": r.home, "away": r.away,
                    "market": r.market, "play": r.play, "line": _num(r.line), "total_line": _num(r.total_line),
                    "units": u, "result": "" if u is None else ("W" if u > 0 else ("L" if u < 0 else "P")),
                    "started": True})
    return out


def refresh_recent() -> None:
    """Write tipped plays (with their graded result so far) into both play files."""
    for path, plays in ((JSON_PATH, recent_edge_plays()), (CAL_JSON_PATH, recent_cal_plays())):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        d["recent"] = plays
        d["results_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

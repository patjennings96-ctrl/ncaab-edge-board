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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = os.path.join(ROOT, "docs", "data", "plays.json")
LOG_PATH = os.path.join(ROOT, "data", "history", "card_log.csv")


def _side_label(row) -> str:
    """Spread sides become team names; totals stay OVER/UNDER."""
    if row["mkt"] == "SPREAD":
        return row["home"] if row["side"] == "HOME" else row["away"]
    return row["side"]


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
            "tracking": float(r["stake_units"]) == 0.0,
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


def append_log(card: pd.DataFrame, mode: str) -> str | None:
    """Append live plays only, so demo runs never pollute the CLV record."""
    if mode != "live" or card.empty:
        return None
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df = pd.DataFrame(card_to_records(card))
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
    res = games_raw.set_index("game_id") if not games_raw.empty else pd.DataFrame()

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
            margin = float(g["home_score"]) - float(g["away_score"])
            total = float(g["home_score"]) + float(g["away_score"])
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
    if units_col != "units" or "risked" not in df:
        out["roi"] = round(out["units"] / len(graded), 4) if len(graded) else None
    else:
        risk = float(pd.to_numeric(graded["risked"], errors="coerce").sum())
        out["roi"] = round(out["units"] / risk, 4) if risk > 0 else None
    if "clv_points" in df:
        clv = pd.to_numeric(df["clv_points"], errors="coerce").dropna()
        out["clv"] = round(float(clv.mean()), 2) if len(clv) else None
        out["beat_close"] = round(float((clv > 0).mean()), 3) if len(clv) else None
    return out


def edge_record(since: str) -> dict:
    if not os.path.exists(LOG_PATH):
        return {}
    log = pd.read_csv(LOG_PATH, dtype={"game_id": str})
    log = log[log["posted_at"].astype(str) >= since].copy()
    if log.empty:
        return {}
    log["stake"] = pd.to_numeric(log.get("stake"), errors="coerce").fillna(0.0)
    odds = log["best_odds"] if "best_odds" in log else log["odds"]
    odds = pd.to_numeric(odds, errors="coerce").fillna(pd.to_numeric(log["odds"], errors="coerce"))
    res = log["result"].astype(str)
    log["units"] = np.where(res == "W", log["stake"] * odds.map(_payout),
                   np.where(res == "L", -log["stake"], np.where(res == "P", 0.0, np.nan)))
    log["risked"] = np.where(res.isin(["W", "L", "P"]), log["stake"], np.nan)
    # tracking plays (0u) are graded as 1u flat so their record is still visible
    log["flat"] = np.where(res == "W", odds.map(_payout), np.where(res == "L", -1.0,
                  np.where(res == "P", 0.0, np.nan)))
    staked, tracking = log[log["stake"] > 0], log[log["stake"] <= 0]
    agree = log[log.get("agree", pd.Series(False, index=log.index)).astype(str).str.lower() == "true"]
    out = {"staked": {m: _summ(g) for m, g in staked.groupby("mkt")},
           "tracking": {m: _summ(g, "flat") for m, g in tracking.groupby("mkt")},
           "both_models": {m: _summ(g, "flat") for m, g in agree.groupby("mkt")},
           "all": _summ(staked)}
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
    log = pd.read_csv(CAL_LOG_PATH, dtype={"game_id": str, "units": str}).fillna({"units": ""})
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
               "season_start": since, "edge": edge_record(since), "cal": cal_record(since)}
    os.makedirs(os.path.dirname(RECORD_PATH), exist_ok=True)
    with open(RECORD_PATH, "w") as f:
        json.dump(payload, f, indent=1)
    return RECORD_PATH

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
    df.to_csv(LOG_PATH, mode="a", index=False, header=not os.path.exists(LOG_PATH))
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

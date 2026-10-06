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
            "home": r["home"], "away": r["away"], "mkt": r["mkt"],
            "side": _side_label(r),
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
    df.insert(0, "posted_at", datetime.now(timezone.utc).isoformat(timespec="minutes"))
    df["close_line"] = ""                  # fill after tip-off for CLV
    df["result"] = ""
    df.to_csv(LOG_PATH, mode="a", index=False, header=not os.path.exists(LOG_PATH))
    return LOG_PATH

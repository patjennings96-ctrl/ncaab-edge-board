"""
players.py
==========
Player-level data, ratings and lineup adjustments.

Data
----
* ESPN box scores (every game): who played, starters, minutes, box stats, and
  DNP rows with ESPN's reason. Stored per season in data/players/<season>.csv.
* Torvik season player stats (BPM, OBPM, DBPM, minutes %, usage) when the site
  allows the download. Used as the rating source in live mode and to calibrate
  our own box-score rating.
* Injury/availability: ESPN's injury feed, any NCAA availability report rows
  saved in data/availability/, and your manual overrides in
  data/availability_overrides.csv.

Ratings
-------
Each player gets an offensive and defensive rating on the Box Plus-Minus scale
(points per 100 possessions vs an average D-I player). Small samples are pulled
toward a prior: last season's rating for returners (transfers keep their ESPN
id), replacement level for everyone else.

Lineup adjustment
-----------------
Team ratings describe the lineups a team has actually used. Before each game
we compare the lineup we expect tonight with that average:

    strength = 5 * sum(rating_i * minutes_i) / team_minutes      (per 100 poss)
    adjustment = expected_strength_tonight - average_strength_in_rated_games

A bench player missing changes almost nothing (his minutes are a small share
and his replacement is similar). A 32-minute starter rated +6 replaced by a -2
player moves the team about 6 points per 100 possessions, roughly 4 points a game.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import live

ROOT = live.ROOT
PLAYERS_DIR = os.path.join(ROOT, "data", "players")
CALIB_PATH = os.path.join(ROOT, "data", "player_rating_coefs.json")
OVERRIDES_PATH = os.path.join(ROOT, "data", "availability_overrides.csv")
AVAIL_DIR = os.path.join(ROOT, "data", "availability")

PLAYER_COLS = ["game_id", "date", "team_id", "player_id", "name", "pos", "starter",
               "dnp", "reason", "min", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
               "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]

REPLACEMENT = -2.0        # BPM of the player who absorbs missing minutes
PRIOR_MINUTES = 250.0     # minutes of evidence the prior is worth
STAT_KEYS = ["pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
             "oreb", "dreb", "ast", "tov", "stl", "blk", "pf"]


# -----------------------------------------------------------------------------
# ESPN player box scores
# -----------------------------------------------------------------------------
def _ma(s):
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except Exception:
        return None, None


def parse_player_box(game_id: str, date: str, data: dict) -> list[dict]:
    """Player rows from an ESPN /summary payload."""
    rows = []
    for t in data.get("boxscore", {}).get("players", []):
        team_id = str(t.get("team", {}).get("id"))
        for block in t.get("statistics", []):
            keys = block.get("keys") or []
            idx = {k: i for i, k in enumerate(keys)}
            for a in block.get("athletes", []):
                ath = a.get("athlete", {})
                st = a.get("stats") or []

                def g(k):
                    i = idx.get(k)
                    return st[i] if i is not None and i < len(st) else None

                fgm, fga = _ma(g("fieldGoalsMade-fieldGoalsAttempted"))
                tpm, tpa = _ma(g("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = _ma(g("freeThrowsMade-freeThrowsAttempted"))
                dnp = bool(a.get("didNotPlay")) or not st
                rows.append({
                    "game_id": str(game_id), "date": date, "team_id": team_id,
                    "player_id": str(ath.get("id")), "name": ath.get("displayName", ""),
                    "pos": (ath.get("position") or {}).get("abbreviation", ""),
                    "starter": int(bool(a.get("starter"))), "dnp": int(dnp),
                    "reason": a.get("reason") or "",
                    "min": live._to_int(g("minutes")) or 0,
                    "pts": live._to_int(g("points")), "fgm": fgm, "fga": fga,
                    "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                    "oreb": live._to_int(g("offensiveRebounds")),
                    "dreb": live._to_int(g("defensiveRebounds")),
                    "ast": live._to_int(g("assists")), "tov": live._to_int(g("turnovers")),
                    "stl": live._to_int(g("steals")), "blk": live._to_int(g("blocks")),
                    "pf": live._to_int(g("fouls")),
                })
    return rows


def _players_path(season: int) -> str:
    return os.path.join(PLAYERS_DIR, f"{season}.csv")


def load_player_games(season: int) -> pd.DataFrame:
    p = _players_path(season)
    if not os.path.exists(p):
        return pd.DataFrame(columns=PLAYER_COLS)
    return pd.read_csv(p, dtype={"game_id": str, "team_id": str, "player_id": str,
                                 "reason": str}, keep_default_na=False,
                       na_values={c: [""] for c in PLAYER_COLS if c not in ("reason", "name", "pos")})


def update_player_games(season: int, workers: int = 8) -> pd.DataFrame:
    """Fetch player box scores for every stored game that doesn't have them yet."""
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    games = live.load_games(season)
    have = load_player_games(season)
    done = set(have["game_id"]) if not have.empty else set()
    todo = games[~games["game_id"].isin(done)][["game_id", "date"]]
    print(f"[players {season}] {len(done)} games cached, fetching {len(todo)}")
    if todo.empty:
        return have

    def one(r):
        try:
            data = live._get_json(f"{live.ESPN}/summary", {"event": r.game_id})
            return parse_player_box(r.game_id, r.date, data)
        except Exception as exc:
            print(f"  player box failed {r.game_id}: {exc}")
            return []

    new_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(one, todo.itertuples(index=False))):
            new_rows.extend(rows)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)}")
    new = pd.DataFrame(new_rows, columns=PLAYER_COLS)
    # append-only so git stores small deltas
    new.to_csv(_players_path(season), mode="a", index=False,
               header=not os.path.exists(_players_path(season)))
    return load_player_games(season)

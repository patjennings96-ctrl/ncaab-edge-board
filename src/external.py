"""
external.py
===========
Import slots for rating sources that can't be downloaded automatically:

    data/external/torvik/teams.csv        data/external/torvik/players.csv
    data/external/kenpom/teams.csv        data/external/kenpom/players.csv
    data/external/shotquality/teams.csv   data/external/shotquality/players.csv

Save the site's CSV export under these names (any column order). Recognized
columns, case-insensitive:

  teams:   team name (team / teamname / school), offense (adjoe / adjo / adj_o /
           ortg / off_rating / sq_ortg), defense (adjde / adjd / adj_d / drtg /
           def_rating / sq_drtg), tempo (adjt / adj_t / adjtempo / tempo / pace)
  players: player name (player / player_name / name), team, and either
           obpm + dbpm, or bpm (split 60/40 into offense/defense), or o + d

Torvik blocks requests from cloud servers and KenPom's terms forbid scraping,
so these files are meant to be exported by you while logged in. Files older
than MAX_AGE_DAYS are ignored so stale ratings don't linger.

When present, team ratings are blended with ours (each source centered on its
own average first), and player ratings are blended into the player model.
"""

from __future__ import annotations

import os
import re
import time

import pandas as pd

import live

EXT_DIR = os.path.join(live.ROOT, "data", "external")
SOURCES = ("torvik", "kenpom", "shotquality")
MAX_AGE_DAYS = 3
TEAM_BLEND = 0.5        # share of the projection given to external team ratings
PLAYER_BLEND = 0.5      # share of each player's rating given to external ratings

ALIASES = {
    "team": ["team", "teamname", "team_name", "school"],
    "adj_o": ["adjoe", "adjo", "adj_o", "ortg", "off_rating", "sq_ortg", "adjoff", "offense"],
    "adj_d": ["adjde", "adjd", "adj_d", "drtg", "def_rating", "sq_drtg", "adjdef", "defense"],
    "tempo": ["adjt", "adj_t", "adjtempo", "tempo", "pace"],
    "player": ["player", "player_name", "playername", "name"],
    "obpm": ["obpm"], "dbpm": ["dbpm"], "bpm": ["bpm", "gbpm"], "o": ["o"], "d": ["d"],
}


def _pick(df: pd.DataFrame, key: str) -> str | None:
    cols = {re.sub(r"[^a-z0-9_]", "", c.lower()): c for c in df.columns}
    for a in ALIASES[key]:
        if a in cols:
            return cols[a]
    return None


def _fresh(path: str) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < MAX_AGE_DAYS * 86400


def load_team_sources(espn_names: dict) -> list[pd.DataFrame]:
    """Each available source as DataFrame[team_id, off, deff, tempo] centered on its own mean."""
    out = []
    for src in SOURCES:
        path = os.path.join(EXT_DIR, src, "teams.csv")
        if not _fresh(path):
            continue
        df = pd.read_csv(path)
        tc, oc, dc, tpc = (_pick(df, k) for k in ("team", "adj_o", "adj_d", "tempo"))
        if not (tc and oc and dc):
            print(f"[external] {src}/teams.csv: need team, offense and defense columns; skipped")
            continue
        m = live.match_teams(list(df[tc].astype(str)), espn_names)
        d = pd.DataFrame({"team_id": df[tc].astype(str).map(m),
                          "adj_o": pd.to_numeric(df[oc], errors="coerce"),
                          "adj_d": pd.to_numeric(df[dc], errors="coerce"),
                          "tempo": pd.to_numeric(df[tpc], errors="coerce") if tpc else None})
        d = d.dropna(subset=["team_id", "adj_o", "adj_d"]).drop_duplicates("team_id")
        if len(d) < 200:
            print(f"[external] {src}/teams.csv: only {len(d)} teams matched; export all of "
                  "Division I so ratings can be centered. Skipped.")
            continue
        d["off"] = d["adj_o"] - d["adj_o"].mean()
        d["deff"] = d["adj_d"] - d["adj_d"].mean()
        d["source"] = src
        print(f"[external] {src} teams: {len(d)} matched of {len(df)}")
        out.append(d.set_index("team_id"))
    return out


def blend_team_ratings(r, sources: list[pd.DataFrame], weight: float = TEAM_BLEND):
    """Return a copy of AdjustedRatings with offense/defense/tempo blended with external sources."""
    if not sources:
        return r
    import copy
    b = copy.deepcopy(r)
    for t in list(b.off):
        vals = [s.loc[t] for s in sources if t in s.index]
        if not vals:
            continue
        eo = sum(v["off"] for v in vals) / len(vals)
        ed = sum(v["deff"] for v in vals) / len(vals)
        b.off[t] = (1 - weight) * r.off[t] + weight * eo
        b.deff[t] = (1 - weight) * r.deff[t] + weight * ed
        tps = [v["tempo"] for v in vals if pd.notna(v.get("tempo"))]
        if tps and t in b.tempo:
            b.tempo[t] = (1 - weight) * r.tempo[t] + weight * (sum(tps) / len(tps))
    return b


def _norm_name(s: str) -> str:
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def load_player_sources(espn_names: dict, roster: pd.DataFrame) -> pd.DataFrame:
    """
    External player ratings matched to ESPN player ids: DataFrame[player_id, o, d].
    roster: our current player table (index player_id, columns name, team_id).
    """
    rows = []
    key = {(_norm_name(n), t): pid for pid, n, t in zip(roster.index, roster["name"], roster["team_id"])}
    for src in SOURCES:
        path = os.path.join(EXT_DIR, src, "players.csv")
        if not _fresh(path):
            continue
        df = pd.read_csv(path)
        pc, tc = _pick(df, "player"), _pick(df, "team")
        ob, db, bp, oc, dc = (_pick(df, k) for k in ("obpm", "dbpm", "bpm", "o", "d"))
        if not (pc and tc and ((ob and db) or bp or (oc and dc))):
            print(f"[external] {src}/players.csv: need player, team and obpm/dbpm, bpm or o/d; skipped")
            continue
        tm = live.match_teams(list(df[tc].astype(str)), espn_names)
        if ob and db:
            o, d = pd.to_numeric(df[ob], errors="coerce"), pd.to_numeric(df[db], errors="coerce")
        elif oc and dc:
            o, d = pd.to_numeric(df[oc], errors="coerce"), pd.to_numeric(df[dc], errors="coerce")
        else:
            b = pd.to_numeric(df[bp], errors="coerce")
            o, d = 0.6 * b, 0.4 * b
        hit = 0
        for name, team, ov, dv in zip(df[pc], df[tc].astype(str), o, d):
            pid = key.get((_norm_name(name), tm.get(team)))
            if pid and pd.notna(ov) and pd.notna(dv):
                rows.append({"player_id": pid, "o": ov, "d": dv, "source": src})
                hit += 1
        print(f"[external] {src} players: {hit} matched of {len(df)}")
    if not rows:
        return pd.DataFrame(columns=["player_id", "o", "d"])
    return pd.DataFrame(rows).groupby("player_id")[["o", "d"]].mean().reset_index()

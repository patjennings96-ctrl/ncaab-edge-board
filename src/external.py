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
import calendar
import hashlib
import json
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


# -----------------------------------------------------------------------------
# Freshness stamps and KenPom paste
# -----------------------------------------------------------------------------
# A file's modification time is useless on GitHub Actions (every checkout looks
# new), so each source folder keeps stamp.json = {file: {sha256, as_of}}. The
# stamp only moves when the file's contents change; the daily run refreshes
# stamps and commits them.
def _stamp_path(folder: str) -> str:
    return os.path.join(folder, "stamp.json")


def _load_stamp(folder: str) -> dict:
    try:
        with open(_stamp_path(folder)) as f:
            return json.load(f)
    except Exception:
        return {}


KENPOM_PASTE = os.path.join(EXT_DIR, "kenpom", "paste.txt")
_WL = re.compile(r"^\d{1,2}-\d{1,2}$")
_NUM = re.compile(r"^[+-]?\d+(\.\d+)?$")


def parse_kenpom_paste(text: str) -> pd.DataFrame:
    """
    Rows copied straight off kenpom.com's ratings table (select the table,
    copy, paste). Works with tabs or plain spaces; header rows repeated down
    the page are skipped. Each data row reads:
        Rk  Team [seed]  Conf  W-L  NetRtg  ORtg rk  DRtg rk  AdjT rk  ...
    """
    rows = []
    for line in text.splitlines():
        tok = line.replace("\t", " ").split()
        if len(tok) < 8 or not tok[0].isdigit():
            continue
        wl = next((i for i, t in enumerate(tok) if _WL.match(t)), None)
        if wl is None or wl < 3:
            continue
        name = tok[1:wl - 1]                       # token before W-L is the conference
        if len(name) > 1 and name[-1].isdigit():   # tournament seed after the name
            name = name[:-1]
        nums = [t for t in tok[wl + 1:] if _NUM.match(t)]
        if len(nums) < 6:
            continue
        # NetRtg, ORtg, ORtg rank, DRtg, DRtg rank, AdjT, ...
        rows.append({"team": " ".join(name), "adjo": float(nums[1]),
                     "adjd": float(nums[3]), "adjt": float(nums[5])})
    return pd.DataFrame(rows).drop_duplicates("team")


def refresh_stamps() -> None:
    """Convert a KenPom paste to teams.csv and update freshness stamps."""
    if os.path.exists(KENPOM_PASTE):
        with open(KENPOM_PASTE, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if text.strip():
            df = parse_kenpom_paste(text)
            if len(df) >= 200:
                df.to_csv(os.path.join(EXT_DIR, "kenpom", "teams.csv"), index=False)
                print(f"[external] KenPom paste: {len(df)} teams")
            else:
                print(f"[external] KenPom paste: only {len(df)} rows parsed; copy the whole ratings table")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for src in SOURCES:
        folder = os.path.join(EXT_DIR, src)
        if not os.path.isdir(folder):
            continue
        stamp, changed = _load_stamp(folder), False
        for name in ("teams.csv", "players.csv"):
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if stamp.get(name, {}).get("sha256") != h:
                stamp[name] = {"sha256": h, "as_of": now}
                changed = True
        if changed:
            with open(_stamp_path(folder), "w") as f:
                json.dump(stamp, f, indent=1)


def _fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    st = _load_stamp(os.path.dirname(path)).get(os.path.basename(path))
    if st:
        t = calendar.timegm(time.strptime(st["as_of"], "%Y-%m-%dT%H:%M:%SZ"))
        return (time.time() - t) < MAX_AGE_DAYS * 86400
    return (time.time() - os.path.getmtime(path)) < MAX_AGE_DAYS * 86400


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
        m = match_schools(list(df[tc].astype(str)))
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


# -----------------------------------------------------------------------------
# School-name matching (KenPom / Torvik style names -> ESPN ids)
# -----------------------------------------------------------------------------
# Sportsbook names carry mascots ("North Carolina Tar Heels"); rating sites use
# the school only, with their own abbreviations. Normalize both sides against
# ESPN's school names (data/teams.csv) and fix the known quirks here.
SCHOOL_ALIASES = {
    "connecticut": "uconn", "nc state": "nc state", "north carolina state": "nc state",
    "miami fl": "miami", "miami oh": "miami oh", "mississippi": "ole miss",
    "louisiana monroe": "ul monroe", "texas aandm corpus chris": "texas aandm corpus christi",
    "liu": "long island university", "liu brooklyn": "long island university",
    "loyola md": "loyola maryland", "usc upstate": "south carolina upstate",
    "tennessee martin": "ut martin", "nebraska omaha": "omaha", "iu indy": "iu indianapolis",
    "iupui": "iu indianapolis", "texas aandm commerce": "east texas aandm",
    "illinois chicago": "uic", "arkansas pine bluff": "arkansas pine bluff", "penn": "pennsylvania",
    "pitt": "pittsburgh", "grambling state": "grambling", "mcneese state": "mcneese",
    "nicholls state": "nicholls", "southeastern louisiana": "se louisiana",
    "cal baptist": "california baptist", "sam houston state": "sam houston", "seattle": "seattle u",
    "fiu": "florida international", "albany": "ualbany", "appalachian state": "app state",
    "american": "american university", "umkc": "kansas city", "arkansas little rock": "little rock",
    "new orleans": "lsu new orleans", "csun": "cal state northridge", "fgcu": "florida gulf coast",
    "louisiana lafayette": "louisiana", "college of charleston": "charleston",
    "detroit": "detroit mercy", "fort wayne": "purdue fort wayne", "southern mississippi": "southern miss",
    "texas rio grande valley": "ut rio grande valley", "utrgv": "ut rio grande valley",
    "wisconsin milwaukee": "milwaukee", "wisconsin green bay": "green bay", "boston u": "boston university",
    "saint thomas": "st thomas", "st thomas mn": "st thomas", "md eastern shore": "maryland eastern shore",
    "massachusetts lowell": "umass lowell", "southeast missouri": "southeast missouri state", "siue": "siu edwardsville", "uconn": "uconn",
}


def _school_norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and").replace("'", "").replace("’", "")
    s = re.sub(r"[().,\-]", " ", s)
    tok = s.split()
    for i, t in enumerate(tok):          # "Kansas St." / "Cal St. Fullerton" -> state
        if t == "st" and i > 0 and tok[i - 1] != "mount":
            tok[i] = "state"
    return " ".join(tok)


def match_schools(names: list[str]) -> dict:
    """Rating-site school name -> ESPN team id. Unmatched names are logged."""
    import difflib
    teams = pd.read_csv(os.path.join(live.ROOT, "data", "teams.csv"), dtype=str)
    by = {}
    for r in teams.itertuples(index=False):
        for v in (r.location, r.display, r.abbr):
            if isinstance(v, str) and v:
                by.setdefault(_school_norm(v), r.team_id)
    by.update({_school_norm(r.location): r.team_id for r in teams.itertuples(index=False)})
    keys, out, missing = list(by), {}, []
    for n in set(names):
        k = _school_norm(n)
        k = SCHOOL_ALIASES.get(k, k)
        if k in by:
            out[n] = by[k]
            continue
        close = difflib.get_close_matches(k, keys, n=1, cutoff=0.9)
        if close:
            out[n] = by[close[0]]
        else:
            missing.append(n)
    if missing:
        print(f"[external] {len(missing)} school names unmatched: {', '.join(sorted(missing)[:15])}")
    return out

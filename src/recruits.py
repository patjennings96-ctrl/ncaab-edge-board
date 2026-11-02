"""
recruits.py
===========
247Sports Composite player rankings, pasted as plain text (or the page's HTML)
into data/external/recruits/<class year>.txt, once a year.

Each player block in the copied page reads:
    rank / 247 rank / name / name / high school / position / height-weight /
    composite rating (0.8-1.0000) / national, position, state ranks / school
The rating line is the anchor; everything else is located relative to it.
"""

from __future__ import annotations

import os
import re

import pandas as pd

import live

RECRUIT_DIR = os.path.join(live.ROOT, "data", "external", "recruits")
_RAT = re.compile(r"^[01]\.\d{4}$")
_HW = re.compile(r"^\d-\d{1,2}\s*/\s*\d{2,3}$")


def _norm_name(s: str) -> str:
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def parse(text: str) -> pd.DataFrame:
    if "<" in text and ">" in text:                      # HTML: strip tags, one item per line
        text = re.sub(r"<[^>]+>", "\n", text)
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    rows = []
    for i, l in enumerate(lines):
        if not (_RAT.match(l) and i >= 4 and _HW.match(lines[i - 1])):
            continue
        school = lines[i + 2] if i + 2 < len(lines) else ""
        if school.isdigit() or _RAT.match(school) or school.upper() in ("N/A", ""):
            school = ""
        rows.append({"name": lines[i - 4], "pos": lines[i - 2], "htwt": lines[i - 1],
                     "rating": float(l), "school": school})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d.sort_values("rating", ascending=False).drop_duplicates("name").reset_index(drop=True)
    d["rank"] = range(1, len(d) + 1)
    d["key"] = d["name"].map(_norm_name)
    return d


def load(class_year: int) -> pd.DataFrame:
    path = os.path.join(RECRUIT_DIR, f"{class_year}.txt")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse(f.read())


def match_players(recruits: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """recruits + player_id/team_id by name. `people`: player_id, name, team_id.
    Prefers a player at the school 247 lists when a name appears twice."""
    import external as X
    if recruits.empty:
        return recruits
    sch = X.match_schools(list(recruits["school"][recruits["school"] != ""].unique()))
    r = recruits.assign(school_id=recruits["school"].map(sch))
    p = people.assign(key=people["name"].map(_norm_name))[["key", "player_id", "team_id"]].drop_duplicates()
    m = r.merge(p, on="key", how="left")
    m["same"] = m["team_id"] == m["school_id"]
    m = m.sort_values("same", ascending=False).drop_duplicates("key")
    return m.drop(columns="same")

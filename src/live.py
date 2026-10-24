"""
live.py
=======
Live data feeds for the model.

* ESPN public JSON  -> schedule, scores and box scores (FGA, FTA, OREB, TOV)
* The Odds API      -> spreads and totals from US books, reduced to a consensus
* Bart Torvik       -> optional reference ratings (often blocks bots; never required)

Everything is cached in the repo under data/ so each run only fetches what's new:

    data/games/<season>.csv           one row per completed game (home perspective)
    data/history/odds_snapshots.csv   consensus line for every game at every run
    data/team_aliases.csv             manual fixes for team names that don't match

Seasons are named by their spring year (2027 = the 2026-27 season).
"""

from __future__ import annotations

import csv
import difflib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAMES_DIR = os.path.join(ROOT, "data", "games")
SNAP_PATH = os.path.join(ROOT, "data", "history", "odds_snapshots.csv")
ALIAS_PATH = os.path.join(ROOT, "data", "team_aliases.csv")
UNMATCHED_PATH = os.path.join(ROOT, "data", "unmatched_teams.csv")

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
ET = ZoneInfo("America/New_York")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ncaab-edge-board/1.0 (personal research)"})


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------
def _get_json(url: str, params: dict | None = None, tries: int = 4) -> dict | list:
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(f"GET failed after {tries} tries: {url}")


# -----------------------------------------------------------------------------
# Season helpers
# -----------------------------------------------------------------------------
def current_season(today: date | None = None) -> int:
    """Spring year of the season in progress (or about to start)."""
    today = today or datetime.now(ET).date()
    return today.year + 1 if today.month >= 8 else today.year


def season_bounds(season: int) -> tuple[date, date]:
    return date(season - 1, 11, 1), date(season, 4, 10)


# -----------------------------------------------------------------------------
# ESPN: scoreboard + box scores
# -----------------------------------------------------------------------------
def espn_scoreboard(day: date) -> list[dict]:
    """All Division I events on a given (Eastern) date."""
    data = _get_json(f"{ESPN}/scoreboard",
                     {"dates": day.strftime("%Y%m%d"), "groups": "50", "limit": "500"})
    out = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in teams or "away" not in teams:
            continue
        status = comp.get("status", {})
        out.append({
            "game_id": str(ev["id"]),
            "start": comp.get("date") or ev.get("date"),
            "neutral": bool(comp.get("neutralSite", False)),
            "completed": bool(status.get("type", {}).get("completed", False)),
            "periods": int(status.get("period") or 2),
            "home_id": str(teams["home"]["team"]["id"]),
            "away_id": str(teams["away"]["team"]["id"]),
            "home_name": teams["home"]["team"].get("displayName", ""),
            "away_name": teams["away"]["team"].get("displayName", ""),
            "home_score": _to_int(teams["home"].get("score")),
            "away_score": _to_int(teams["away"].get("score")),
        })
    return out


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _split_made_att(s: str) -> int | None:
    try:
        return int(str(s).split("-")[1])
    except (IndexError, ValueError):
        return None


def espn_box(game_id: str) -> dict:
    """{team_id: {fga, fta, oreb, tov}} from the game summary box score."""
    data = _get_json(f"{ESPN}/summary", {"event": game_id})
    out = {}
    for t in data.get("boxscore", {}).get("teams", []):
        stats = {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])}
        out[str(t["team"]["id"])] = {
            "fga": _split_made_att(stats.get("fieldGoalsMade-fieldGoalsAttempted")),
            "fta": _split_made_att(stats.get("freeThrowsMade-freeThrowsAttempted")),
            "oreb": _to_int(stats.get("offensiveRebounds")),
            "tov": _to_int(stats.get("turnovers") or stats.get("totalTurnovers")),
        }
    return out


GAME_COLS = ["game_id", "date", "season", "neutral", "minutes",
             "home_id", "away_id", "home_name", "away_name", "home_score", "away_score",
             "h_fga", "h_fta", "h_oreb", "h_tov", "a_fga", "a_fta", "a_oreb", "a_tov"]


def _games_path(season: int) -> str:
    return os.path.join(GAMES_DIR, f"{season}.csv")


def load_games(season: int) -> pd.DataFrame:
    p = _games_path(season)
    if not os.path.exists(p):
        return pd.DataFrame(columns=GAME_COLS)
    return pd.read_csv(p, dtype={"game_id": str, "home_id": str, "away_id": str})


def update_games(season: int, workers: int = 8, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch every completed game from the last stored date (or season start)
    through yesterday, add box scores, and save. Safe to rerun; it only fetches
    what's missing.
    """
    os.makedirs(GAMES_DIR, exist_ok=True)
    have = load_games(season)
    done_flag = os.path.join(GAMES_DIR, f"{season}.complete")
    if os.path.exists(done_flag):
        return have
    start, end = season_bounds(season)
    season_over = datetime.now(ET).date() - timedelta(days=1) >= end
    yesterday = datetime.now(ET).date() - timedelta(days=1)
    end = min(end, yesterday)
    if not have.empty:
        start = max(start, pd.to_datetime(have["date"]).max().date() - timedelta(days=1))
    if start > end:
        if season_over:
            open(done_flag, "w").close()
        return have

    known = set(have["game_id"]) if not have.empty else set()
    events, d = [], start
    while d <= end:
        for e in espn_scoreboard(d):
            if e["completed"] and e["game_id"] not in known:
                e["date"] = d.isoformat()
                events.append(e)
        d += timedelta(days=1)
    if verbose:
        print(f"[games {season}] {len(events)} new completed games {start}..{end}")
    if season_over:
        open(done_flag, "w").close()
    if not events:
        return have

    def box(e):
        try:
            return e, espn_box(e["game_id"])
        except Exception as exc:                       # keep going on one bad game
            print(f"  box score failed for {e['game_id']}: {exc}")
            return e, {}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for e, b in pool.map(box, events):
            h, a = b.get(e["home_id"], {}), b.get(e["away_id"], {})
            rows.append({
                "game_id": e["game_id"], "date": e["date"], "season": season,
                "neutral": int(e["neutral"]),
                "minutes": 40 + 5 * max(0, e["periods"] - 2),
                "home_id": e["home_id"], "away_id": e["away_id"],
                "home_name": e["home_name"], "away_name": e["away_name"],
                "home_score": e["home_score"], "away_score": e["away_score"],
                "h_fga": h.get("fga"), "h_fta": h.get("fta"),
                "h_oreb": h.get("oreb"), "h_tov": h.get("tov"),
                "a_fga": a.get("fga"), "a_fta": a.get("fta"),
                "a_oreb": a.get("oreb"), "a_tov": a.get("tov"),
            })
    new = pd.DataFrame(rows, columns=GAME_COLS)
    allg = (pd.concat([have, new], ignore_index=True)
              .drop_duplicates("game_id", keep="last")
              .sort_values(["date", "game_id"]))
    allg.to_csv(_games_path(season), index=False)
    return allg


def to_long(games: pd.DataFrame) -> pd.DataFrame:
    """Home-perspective game rows -> the long format ratings.py expects."""
    g = games.dropna(subset=["h_fga", "h_fta", "h_oreb", "h_tov",
                             "a_fga", "a_fta", "a_oreb", "a_tov",
                             "home_score", "away_score"]).copy()
    home_loc = np.where(g["neutral"].astype(int) == 1, "N", "H")
    away_loc = np.where(g["neutral"].astype(int) == 1, "N", "A")
    common = {"game_id": g["game_id"], "date": pd.to_datetime(g["date"]),
              "season": g["season"], "minutes": g["minutes"], "venue_id": g["home_id"]}
    home = pd.DataFrame({**common, "team": g["home_id"], "opp": g["away_id"],
                         "location": home_loc,
                         "team_score": g["home_score"], "opp_score": g["away_score"],
                         "fga": g["h_fga"], "fta": g["h_fta"], "oreb": g["h_oreb"], "tov": g["h_tov"],
                         "opp_fga": g["a_fga"], "opp_fta": g["a_fta"],
                         "opp_oreb": g["a_oreb"], "opp_tov": g["a_tov"]})
    away = pd.DataFrame({**common, "team": g["away_id"], "opp": g["home_id"],
                         "location": away_loc,
                         "team_score": g["away_score"], "opp_score": g["home_score"],
                         "fga": g["a_fga"], "fta": g["a_fta"], "oreb": g["a_oreb"], "tov": g["a_tov"],
                         "opp_fga": g["h_fga"], "opp_fta": g["h_fta"],
                         "opp_oreb": g["h_oreb"], "opp_tov": g["h_tov"]})
    out = pd.concat([home, away], ignore_index=True)
    num = ["team_score", "opp_score", "fga", "fta", "oreb", "tov",
           "opp_fga", "opp_fta", "opp_oreb", "opp_tov", "minutes"]
    out[num] = out[num].astype(float)
    return out


def team_names(*frames: pd.DataFrame) -> dict:
    """ESPN team id -> display name, from any game frames."""
    names = {}
    for f in frames:
        if f is None or f.empty:
            continue
        names.update(dict(zip(f["home_id"], f["home_name"])))
        names.update(dict(zip(f["away_id"], f["away_name"])))
    return names


# -----------------------------------------------------------------------------
# Today's slate (ESPN) + odds (The Odds API)
# -----------------------------------------------------------------------------
MAX_SLATE_DAYS = 21     # how far ahead to price games that already have lines


def upcoming_slate(hours: int = 30, until: date | None = None) -> pd.DataFrame:
    """
    Games that haven't tipped yet: the next `hours`, or every game through the
    Eastern date `until` (capped at MAX_SLATE_DAYS ahead).
    """
    now = datetime.now(timezone.utc)
    today = datetime.now(ET).date()
    if until is None:
        days = [today, today + timedelta(days=1)]
        horizon = now + timedelta(hours=hours)
    else:
        last = min(until, today + timedelta(days=MAX_SLATE_DAYS))
        days = [today + timedelta(days=i) for i in range((last - today).days + 1)]
        horizon = None
    rows = []
    for d in days:
        for e in espn_scoreboard(d):
            if e["completed"] or not e["start"]:
                continue
            start = datetime.fromisoformat(e["start"].replace("Z", "+00:00"))
            if start >= now and (horizon is None or start <= horizon):
                rows.append(e)
    return pd.DataFrame(rows).drop_duplicates("game_id") if rows else pd.DataFrame()


def _decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def _best(quotes: list, better_line) -> tuple:
    """Best (line, price, book) among quotes [(line, price, book)]: the most
    favourable line first, then the best price at that line."""
    if not quotes:
        return np.nan, -110, ""
    line = better_line(q[0] for q in quotes)
    at = [q for q in quotes if q[0] == line]
    q = max(at, key=lambda q: _decimal(q[1]))
    return float(q[0]), int(q[1]), q[2]


def fetch_odds(api_key: str) -> pd.DataFrame:
    """
    Consensus (median) spread/total per game across US books, plus the best
    available number on each side and the book offering it (line shopping).
    Cost: 2 credits per call (2 markets x 1 region).
    """
    data = _get_json(ODDS, {"apiKey": api_key, "regions": "us",
                            "markets": "spreads,totals", "oddsFormat": "american"})
    rows = []
    for ev in data:
        home, away = ev["home_team"], ev["away_team"]
        sp, sp_px_h, sp_px_a, tot, px_o, px_u, books = [], [], [], [], [], [], 0
        q_home, q_away, q_over, q_under = [], [], [], []
        for bk in ev.get("bookmakers", []):
            books += 1
            title = bk.get("title", bk.get("key", ""))
            for m in bk.get("markets", []):
                outs = {o["name"]: o for o in m.get("outcomes", [])}
                if m["key"] == "spreads" and home in outs and away in outs:
                    sp.append(outs[home]["point"])
                    sp_px_h.append(outs[home]["price"])
                    sp_px_a.append(outs[away]["price"])
                    q_home.append((outs[home]["point"], outs[home]["price"], title))
                    q_away.append((outs[away]["point"], outs[away]["price"], title))
                elif m["key"] == "totals" and "Over" in outs and "Under" in outs:
                    tot.append(outs["Over"]["point"])
                    px_o.append(outs["Over"]["price"])
                    px_u.append(outs["Under"]["price"])
                    q_over.append((outs["Over"]["point"], outs["Over"]["price"], title))
                    q_under.append((outs["Under"]["point"], outs["Under"]["price"], title))
        row = {
            "odds_id": ev["id"], "commence": ev["commence_time"],
            "home_name_odds": home, "away_name_odds": away,
            "spread_home": float(np.median(sp)) if sp else np.nan,
            "home_odds": int(np.median(sp_px_h)) if sp_px_h else -110,
            "away_odds": int(np.median(sp_px_a)) if sp_px_a else -110,
            "total": float(np.median(tot)) if tot else np.nan,
            "over_odds": int(np.median(px_o)) if px_o else -110,
            "under_odds": int(np.median(px_u)) if px_u else -110,
            "n_books": books,
        }
        # Best number per side. Spreads: the highest number is best for both
        # sides (home -3 beats -3.5; away +4 beats +3.5). Over: lowest total;
        # under: highest total.
        for side, quotes, pick in (("home", q_home, max), ("away", q_away, max),
                                   ("over", q_over, min), ("under", q_under, max)):
            row[f"best_{side}_line"], row[f"best_{side}_odds"], row[f"best_{side}_book"] = _best(quotes, pick)
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Team-name matching (Odds API names -> ESPN ids)
# -----------------------------------------------------------------------------
def _norm(name: str) -> str:
    s = name.lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_aliases() -> dict:
    if not os.path.exists(ALIAS_PATH):
        return {}
    df = pd.read_csv(ALIAS_PATH, dtype=str)
    return dict(zip(df["source_name"], df["espn_id"]))


def match_teams(odds_names: list[str], espn_names: dict) -> dict:
    """
    Map each sportsbook team name to an ESPN team id. Exact normalized match
    first, then a close fuzzy match. Anything left goes to data/unmatched_teams.csv
    so it can be added to data/team_aliases.csv by hand.
    """
    aliases = _load_aliases()
    by_norm = {_norm(n): i for i, n in espn_names.items()}
    keys = list(by_norm)
    out, missing = {}, []
    for n in set(odds_names):
        if n in aliases:
            out[n] = aliases[n]
            continue
        k = _norm(n)
        if k in by_norm:
            out[n] = by_norm[k]
            continue
        close = difflib.get_close_matches(k, keys, n=1, cutoff=0.88)
        if close:
            out[n] = by_norm[close[0]]
        else:
            missing.append(n)
    if missing:
        os.makedirs(os.path.dirname(UNMATCHED_PATH), exist_ok=True)
        pd.DataFrame({"source_name": sorted(missing)}).to_csv(UNMATCHED_PATH, index=False)
        print(f"[names] {len(missing)} sportsbook names unmatched -> {UNMATCHED_PATH}")
    return out


def build_live_slate(api_key: str, espn_names: dict) -> pd.DataFrame:
    """Join ESPN's upcoming games with odds: every game that has lines posted,
    up to MAX_SLATE_DAYS ahead (one Odds API call covers all of them)."""
    odds = fetch_odds(api_key)
    if odds.empty:
        print("[odds] no NCAAB odds returned")
        return pd.DataFrame()
    last = pd.to_datetime(odds["commence"], utc=True).dt.tz_convert(ET).dt.date.max()
    slate = upcoming_slate(until=last)
    if slate.empty:
        print(f"[slate] no untipped games through {last}")
        return pd.DataFrame()

    names = {**espn_names,
             **dict(zip(slate["home_id"], slate["home_name"])),
             **dict(zip(slate["away_id"], slate["away_name"]))}
    m = match_teams(list(odds["home_name_odds"]) + list(odds["away_name_odds"]), names)
    odds["home_id"] = odds["home_name_odds"].map(m)
    odds["away_id"] = odds["away_name_odds"].map(m)

    j = slate.merge(odds, on=["home_id", "away_id"], how="inner")
    # books sometimes list neutral-site games with home/away flipped
    flipped = slate.merge(odds, left_on=["home_id", "away_id"],
                          right_on=["away_id", "home_id"], how="inner",
                          suffixes=("", "_o"))
    if not flipped.empty:
        flipped = flipped.assign(
            spread_home=-flipped["spread_home"],
            home_odds=flipped["away_odds"], away_odds=flipped["home_odds"],
            **{f"best_home_{k}": flipped[f"best_away_{k}"] for k in ("line", "odds", "book")},
            **{f"best_away_{k}": flipped[f"best_home_{k}"] for k in ("line", "odds", "book")})
        flipped = flipped[[c for c in j.columns if c in flipped.columns]]
        j = pd.concat([j, flipped], ignore_index=True)

    # the same two teams can meet twice in the window: keep the odds whose
    # start time matches ESPN's
    if not j.empty:
        gap = (pd.to_datetime(j["start"], utc=True) - pd.to_datetime(j["commence"], utc=True)).abs()
        j = j[gap <= pd.Timedelta(hours=18)]
        j = j.sort_values("n_books", ascending=False).drop_duplicates("game_id")
    j = j.dropna(subset=["spread_home", "total"], how="all")
    print(f"[slate] {len(slate)} untipped games through {last}, {len(odds)} with odds, {len(j)} matched")
    snapshot_odds(j)
    return j.rename(columns={"home_id": "home", "away_id": "away"})


def snapshot_odds(slate: pd.DataFrame) -> None:
    """Append this run's consensus lines; the last one before tip is the close."""
    if slate.empty:
        return
    os.makedirs(os.path.dirname(SNAP_PATH), exist_ok=True)
    s = slate[["game_id", "start", "home_id", "away_id", "spread_home", "total",
               "home_odds", "away_odds", "over_odds", "under_odds", "n_books"]].copy()
    s.insert(0, "snap_at", datetime.now(timezone.utc).isoformat(timespec="minutes"))
    s.to_csv(SNAP_PATH, mode="a", index=False, header=not os.path.exists(SNAP_PATH))


def closing_lines() -> pd.DataFrame:
    """Last snapshot taken before each game's start."""
    if not os.path.exists(SNAP_PATH):
        return pd.DataFrame()
    s = pd.read_csv(SNAP_PATH, dtype={"game_id": str})
    start = pd.to_datetime(s["start"], utc=True)
    s = s[(pd.to_datetime(s["snap_at"], utc=True) < start)
          & (start <= pd.Timestamp.now(tz="UTC"))]          # only games that have tipped
    return s.sort_values("snap_at").groupby("game_id").tail(1)


# -----------------------------------------------------------------------------
# Torvik (optional reference)
# -----------------------------------------------------------------------------
def fetch_torvik(season: int) -> pd.DataFrame | None:
    """
    Torvik T-Rank ratings if the site allows the request; otherwise None.
    Used only as a cross-check column next to our own ratings.
    """
    try:
        r = SESSION.get("https://barttorvik.com/trank.php",
                        params={"year": season, "csv": 1}, timeout=25)
        if r.status_code != 200 or "," not in r.text[:200]:
            print(f"[torvik] unavailable (HTTP {r.status_code}); skipping")
            return None
        rows = list(csv.reader(r.text.splitlines()))
        # trank csv: team, adjoe, adjde, barthag, ... ; adj tempo near the end
        df = pd.DataFrame({"torvik_team": [x[0] for x in rows if len(x) > 3],
                           "torvik_adj_o": [float(x[1]) for x in rows if len(x) > 3],
                           "torvik_adj_d": [float(x[2]) for x in rows if len(x) > 3]})
        df["torvik_adj_em"] = (df["torvik_adj_o"] - df["torvik_adj_d"]).round(2)
        return df
    except Exception as exc:
        print(f"[torvik] unavailable ({exc}); skipping")
        return None

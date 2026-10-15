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


# -----------------------------------------------------------------------------
# Player ratings
# -----------------------------------------------------------------------------
# Default box-score weights per 100 possessions on the floor, close to the
# published Box Plus-Minus 2.0 raw coefficients. Overwritten by a calibration
# against Torvik BPM when that data is available (data/player_rating_coefs.json).
DEFAULT_COEFS = {
    "off": {"pts": 0.860, "tpm": 0.389, "ast": 0.580, "tov": -0.964, "oreb": 0.613,
            "fga": -0.560, "fta": -0.246},
    "def": {"dreb": 0.116, "stl": 1.369, "blk": 1.327, "pf": -0.367},
}


def load_coefs() -> dict:
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            return json.load(f)
    return DEFAULT_COEFS


def game_possessions(games_raw: pd.DataFrame) -> pd.Series:
    """game_id -> possessions (mean of both teams' estimates)."""
    g = games_raw
    ph = g.h_fga - g.h_oreb + g.h_tov + 0.475 * g.h_fta
    pa = g.a_fga - g.a_oreb + g.a_tov + 0.475 * g.a_fta
    return pd.Series(((ph + pa) / 2).values, index=g.game_id.astype(str))


def season_box(pg: pd.DataFrame, poss: pd.Series) -> pd.DataFrame:
    """Per-player season totals and per-100-possession rates."""
    p = pg[(pg["dnp"] == 0) & (pg["min"] > 0)].copy()
    p["on_poss"] = p["game_id"].map(poss).fillna(68.0) * p["min"] / 40.0
    agg = p.groupby("player_id").agg(
        name=("name", "last"), team_id=("team_id", "last"), games=("game_id", "nunique"),
        minutes=("min", "sum"), on_poss=("on_poss", "sum"),
        **{k: (k, "sum") for k in STAT_KEYS})
    for k in STAT_KEYS:
        agg[f"{k}_100"] = agg[k] * 100.0 / agg["on_poss"].clip(lower=1)
    return agg


def raw_ratings(box: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    """Uncentered offensive and defensive box ratings."""
    out = pd.DataFrame(index=box.index)
    out["raw_o"] = sum(w * box[f"{k}_100"] for k, w in coefs["off"].items()) + coefs.get("off_const", 0.0)
    out["raw_d"] = sum(w * box[f"{k}_100"] for k, w in coefs["def"].items()) + coefs.get("def_const", 0.0)
    return out


def player_ratings(pg: pd.DataFrame, games_raw: pd.DataFrame, team_ratings,
                   prior: pd.DataFrame | None = None, coefs: dict | None = None) -> pd.DataFrame:
    """
    Ratings for every player in `pg` (player games already restricted to the
    period you want, e.g. before a cutoff date).

    1. Box rates per 100 possessions on the floor -> raw O and D ratings.
    2. Center so the minutes-weighted D-I average is 0.
    3. Team adjustment: shift each team's players so that their minutes-weighted
       sum matches the team's adjusted offensive/defensive rating (as BPM does).
    4. Shrink toward a prior by minutes played.
    """
    coefs = coefs or load_coefs()
    poss = game_possessions(games_raw)
    box = season_box(pg, poss)
    if box.empty:
        return pd.DataFrame(columns=["name", "team_id", "minutes", "o", "d", "rating"])
    r = raw_ratings(box, coefs).join(box[["name", "team_id", "minutes", "games"]])

    w = r["minutes"]
    r["o"] = r["raw_o"] - np.average(r["raw_o"], weights=w)
    r["d"] = r["raw_d"] - np.average(r["raw_d"], weights=w)

    # team adjustment, using each player's minutes for his current team
    if team_ratings is not None:
        tm = pg[(pg["dnp"] == 0)].groupby(["team_id", "player_id"])["min"].sum().reset_index()
        tm = tm.merge(r[["o", "d"]], left_on="player_id", right_index=True)
        tot = tm.groupby("team_id")["min"].transform("sum")
        tm["share5"] = 5.0 * tm["min"] / tot
        sums = tm.assign(so=tm.share5 * tm.o, sd=tm.share5 * tm.d).groupby("team_id")[["so", "sd"]].sum()
        mu = team_ratings.mu
        tgt_o = pd.Series({t: team_ratings.adj_o(t) - mu for t in sums.index})
        tgt_d = pd.Series({t: mu - team_ratings.adj_d(t) for t in sums.index})
        off_o = (tgt_o - sums["so"]) / 5.0
        off_d = (tgt_d - sums["sd"]) / 5.0
        r["o"] += r["team_id"].map(off_o).fillna(0.0)
        r["d"] += r["team_id"].map(off_d).fillna(0.0)

    # shrink toward prior (last season for returners, replacement otherwise)
    po = pd.Series(REPLACEMENT * 0.6, index=r.index)
    pdd = pd.Series(REPLACEMENT * 0.4, index=r.index)
    if prior is not None and not prior.empty:
        known = r.index.intersection(prior.index)
        # last season regressed a third of the way back to average
        po.loc[known] = prior.loc[known, "o"] * 0.67
        pdd.loc[known] = prior.loc[known, "d"] * 0.67
    k = r["minutes"] / (r["minutes"] + PRIOR_MINUTES)
    r["o"] = k * r["o"] + (1 - k) * po
    r["d"] = k * r["d"] + (1 - k) * pdd
    r["rating"] = r["o"] + r["d"]
    return r[["name", "team_id", "minutes", "games", "o", "d", "rating"]]


# -----------------------------------------------------------------------------
# Lineups: expected minutes, availability, team strength
# -----------------------------------------------------------------------------
STATUS_PLAY_PROB = {"OUT": 0.0, "SUSPENDED": 0.0, "DOUBTFUL": 0.25,
                    "QUESTIONABLE": 0.6, "GAME-TIME DECISION": 0.6, "GTD": 0.6,
                    "DAY-TO-DAY": 0.75, "PROBABLE": 0.95, "ACTIVE": 1.0, "AVAILABLE": 1.0}

# Chance a rotation player plays after missing his team's last k games, when no
# report says otherwise. Replaced by the empirical rate from data when available.
DEFAULT_RETURN_PROB = {1: 0.55, 2: 0.35, 3: 0.25}


def team_game_matrix(pg: pd.DataFrame, team_id: str) -> pd.DataFrame:
    """Rows = the team's games in date order, columns = players, values = minutes (0 if absent)."""
    t = pg[pg["team_id"] == team_id]
    if t.empty:
        return pd.DataFrame()
    m = t.pivot_table(index=["date", "game_id"], columns="player_id",
                      values="min", aggfunc="sum", fill_value=0).sort_index()
    return m


def expected_minutes(mat: pd.DataFrame, halflife: float = 4.0) -> pd.DataFrame:
    """
    For each player: minutes when he plays (EWMA over games he played),
    consecutive team games missed most recently, and recent rotation role.
    """
    if mat.empty:
        return pd.DataFrame(columns=["min_if_plays", "missed_streak", "was_rotation"])
    out = {}
    for pid in mat.columns:
        s = mat[pid]
        played = s[s > 0]
        if played.empty:
            continue
        min_if = played.ewm(halflife=halflife).mean().iloc[-1]
        streak = 0
        for v in s.values[::-1]:
            if v > 0:
                break
            streak += 1
        out[pid] = {"min_if_plays": float(min_if), "missed_streak": streak,
                    "was_rotation": float(min_if) >= 10.0,
                    "last_played": played.index[-1][0]}
    return pd.DataFrame.from_dict(out, orient="index")


def play_probability(row, status: str | None, return_prob: dict) -> float:
    if status:
        return STATUS_PLAY_PROB.get(status.upper(), 0.6)
    k = int(row["missed_streak"])
    if k == 0:
        return 1.0
    if not row["was_rotation"]:
        return 0.5
    return return_prob.get(min(k, 3), 0.25)


def lineup_strength(minutes: pd.Series, ratings: pd.DataFrame, team_minutes: float = 200.0) -> tuple[float, float]:
    """5 * sum(rating * minutes) / team_minutes, O and D separately; missing minutes at replacement."""
    m = minutes[minutes > 0]
    total = m.sum()
    if total > team_minutes:
        m = m * team_minutes / total
        total = team_minutes
    o = ratings["o"].reindex(m.index).fillna(REPLACEMENT * 0.6)
    d = ratings["d"].reindex(m.index).fillna(REPLACEMENT * 0.4)
    gap = max(team_minutes - total, 0.0)
    so = 5.0 * (float((o * m).sum()) + REPLACEMENT * 0.6 * gap) / team_minutes
    sd = 5.0 * (float((d * m).sum()) + REPLACEMENT * 0.4 * gap) / team_minutes
    return so, sd


def baseline_strength(pg_rated: pd.DataFrame, weights: pd.Series, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted average lineup strength of each team over the games its team
    rating was fit on, using actual minutes played. `weights`: game_id -> weight.
    Vectorized version of lineup_strength over every team-game.
    """
    p = pg_rated[pg_rated["dnp"] == 0][["game_id", "team_id", "player_id", "min"]].copy()
    p["o"] = p["player_id"].map(ratings["o"]).fillna(REPLACEMENT * 0.6)
    p["d"] = p["player_id"].map(ratings["d"]).fillna(REPLACEMENT * 0.4)
    p["mo"], p["md"] = p["min"] * p["o"], p["min"] * p["d"]
    tg = p.groupby(["team_id", "game_id"]).agg(mins=("min", "sum"), mo=("mo", "sum"), md=("md", "sum"))
    tg["so"] = 5.0 * tg["mo"] / tg["mins"].clip(lower=1)
    tg["sd"] = 5.0 * tg["md"] / tg["mins"].clip(lower=1)
    tg = tg.reset_index()
    tg["w"] = tg["game_id"].map(weights).fillna(0.0)
    tg = tg[tg["w"] > 0]
    agg = tg.assign(wo=tg.w * tg.so, wd=tg.w * tg.sd).groupby("team_id")[["wo", "wd", "w"]].sum()
    return pd.DataFrame({"base_o": agg.wo / agg.w, "base_d": agg.wd / agg.w})


def project_lineup(pg_team_season: pd.DataFrame, team_id: str, ratings: pd.DataFrame,
                   statuses: dict | None = None, return_prob: dict | None = None,
                   played_override: set | None = None) -> tuple[float, float, pd.DataFrame]:
    """
    Expected strength tonight for one team.
    statuses: player_id -> report status.  played_override: set of player_ids
    known to play (used by the backtest's perfect-information variant).
    """
    statuses = statuses or {}
    return_prob = return_prob or DEFAULT_RETURN_PROB
    em = expected_minutes(team_game_matrix(pg_team_season, team_id))
    if em.empty:
        return 0.0, 0.0, em
    if played_override is not None:
        em["p_play"] = [1.0 if pid in played_override else 0.0 for pid in em.index]
    else:
        em["p_play"] = [play_probability(r, statuses.get(pid), return_prob)
                        for pid, r in em.iterrows()]
    em["exp_min"] = em["p_play"] * em["min_if_plays"]
    so, sd = lineup_strength(em["exp_min"], ratings)
    return so, sd, em


# -----------------------------------------------------------------------------
# Availability reports
# -----------------------------------------------------------------------------
def load_overrides() -> pd.DataFrame:
    """
    Your manual availability notes: data/availability_overrides.csv with
    columns player_id (or name + team_id), status, through (YYYY-MM-DD, optional).
    """
    if not os.path.exists(OVERRIDES_PATH):
        return pd.DataFrame(columns=["player_id", "name", "team_id", "status", "through"])
    o = pd.read_csv(OVERRIDES_PATH, dtype=str).fillna("")
    today = datetime.now(live.ET).date().isoformat()
    if "through" in o.columns:
        o = o[(o["through"] == "") | (o["through"] >= today)]
    return o


def fetch_espn_injuries() -> pd.DataFrame:
    """League-wide ESPN injury feed (often sparse for college)."""
    try:
        data = live._get_json(f"{live.ESPN}/injuries")
    except Exception as exc:
        print(f"[injuries] ESPN feed unavailable: {exc}")
        return pd.DataFrame(columns=["player_id", "team_id", "status", "source"])
    rows = []
    for team in data.get("injuries", []):
        tid = str(team.get("id") or (team.get("team") or {}).get("id") or "")
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {})
            rows.append({"player_id": str(ath.get("id", "")), "team_id": tid,
                         "name": ath.get("displayName", ""),
                         "status": (inj.get("status") or inj.get("type", {}).get("name") or "").upper(),
                         "source": "espn"})
    return pd.DataFrame(rows, columns=["player_id", "team_id", "name", "status", "source"])


X_REPORTS_PATH = os.path.join(AVAIL_DIR, "x_reports.csv")

# Reporter posts count slightly below official reports.
STATUS_PLAY_PROB.update({"REPORTED OUT": 0.1, "REPORTED DOUBTFUL": 0.3,
                         "REPORTED QUESTIONABLE": 0.6, "REPORTED PROBABLE": 0.9,
                         "REPORTED AVAILABLE": 0.95})


def _match_player(name: str, team: str, roster: pd.DataFrame, team_names: dict) -> str:
    """ESPN player id for a name (+ team text), or '' if not confidently matched."""
    import difflib
    from external import _norm_name
    key = _norm_name(name)
    cand = roster[roster["_nn"] == key]
    if len(cand) == 1:
        return cand.index[0]
    if len(cand) > 1 and team:
        tn = {cand.loc[i, "team_id"]: team_names.get(cand.loc[i, "team_id"], "") for i in cand.index}
        best = difflib.get_close_matches(team.lower(), [v.lower() for v in tn.values()], n=1, cutoff=0.5)
        if best:
            for i in cand.index:
                if tn[cand.loc[i, "team_id"]].lower() == best[0]:
                    return i
    return ""


def load_x_reports(roster: pd.DataFrame, team_names: dict) -> dict:
    """
    Reporter posts collected by the scheduled X scan (data/availability/x_reports.csv):
    columns date, player, team, status, account, url, posted_at, note.
    Only rows dated today or later (Eastern) are used.
    """
    if not os.path.exists(X_REPORTS_PATH) or roster.empty:
        return {}
    x = pd.read_csv(X_REPORTS_PATH, dtype=str).fillna("")
    today = datetime.now(live.ET).date().isoformat()
    x = x[x["date"] >= today]
    out = {}
    for r in x.sort_values("posted_at").itertuples():
        pid = _match_player(r.player, r.team, roster, team_names)
        if pid:
            out[pid] = f"REPORTED {r.status.strip().upper()}"
        else:
            print(f"[x reports] no player match for {r.player} ({r.team})")
    return out


def current_statuses(roster: pd.DataFrame, team_names: dict | None = None) -> dict:
    """
    player_id -> status. Later sources overwrite earlier ones:
    reporter posts (X scan) < ESPN injury feed < saved reports < your overrides.
    """
    roster = roster.copy()
    from external import _norm_name
    roster["_nn"] = roster["name"].map(_norm_name)
    st = load_x_reports(roster, team_names or {})
    inj = fetch_espn_injuries()
    for r in inj.itertuples():
        if r.player_id:
            st[r.player_id] = r.status
    if os.path.isdir(AVAIL_DIR):
        for f in sorted(os.listdir(AVAIL_DIR)):
            if f.endswith(".csv") and os.path.join(AVAIL_DIR, f) != X_REPORTS_PATH:
                a = pd.read_csv(os.path.join(AVAIL_DIR, f), dtype=str).fillna("")
                for r in a.itertuples():
                    if getattr(r, "player_id", ""):
                        st[r.player_id] = r.status.upper()
    ov = load_overrides()
    if not ov.empty:
        for r in ov.itertuples():
            pid = r.player_id or _match_player(r.name, "", roster[roster["team_id"] == r.team_id]
                                               if r.team_id else roster, team_names or {})
            if pid:
                st[pid] = r.status.upper()
    return st


def adjust_projection(proj: dict, adj_home: tuple, adj_away: tuple, weight: float = 1.0) -> dict:
    """
    Apply lineup adjustments (dO, dD per 100 possessions) to a ratings projection.
    Home points move with home offense change and away defense change, and vice versa.
    """
    poss = proj["proj_poss"]
    dho, dhd = adj_home
    dao, dad = adj_away
    ph = proj["proj_home"] + weight * (dho - dad) * poss / 100.0
    pa = proj["proj_away"] + weight * (dao - dhd) * poss / 100.0
    out = dict(proj)
    out.update(proj_home=ph, proj_away=pa, proj_margin=ph - pa, proj_total=ph + pa)
    return out


# -----------------------------------------------------------------------------
# Live context: everything the daily run needs to adjust tonight's games
# -----------------------------------------------------------------------------
# Weights fitted on 2025-26 (lineup_backtest.py, first half fit / second half test):
# margin 0.77 overall (0.64 in Nov-Dec when turnover dominates, 0.97 from January);
# totals showed no accuracy gain, so they get half weight.
LINEUP_WEIGHT_MARGIN = 0.8
LINEUP_WEIGHT_TOTAL = 0.5
MIN_TEAM_GAMES = 3
RETURN_RATES_PATH = os.path.join(ROOT, "data", "return_rates.json")


def cached_return_rates(prev_pg: pd.DataFrame, prev_season: int) -> dict:
    if os.path.exists(RETURN_RATES_PATH):
        with open(RETURN_RATES_PATH) as f:
            d = json.load(f)
        if d.get("season") == prev_season:
            return {int(k): v for k, v in d["rates"].items()}
    from lineup_backtest import return_rates
    rates, n = return_rates(prev_pg)
    with open(RETURN_RATES_PATH, "w") as f:
        json.dump({"season": prev_season, "rates": rates, "n": n}, f, indent=1)
    return rates


def build_context(season: int, prev_raw, cur_raw, prev_long, team_ratings,
                  game_weights: pd.Series, external_players: pd.DataFrame | None = None) -> dict:
    """Player ratings, baselines, return rates and current statuses for the live run."""
    import ratings as R
    prev_pg, cur_pg = load_player_games(season - 1), load_player_games(season)
    n = prev_long.groupby("team").size()
    d1p = set(n[n >= 10].index)
    pl = prev_long[prev_long.team.isin(d1p) & prev_long.opp.isin(d1p)]
    prior = player_ratings(prev_pg[prev_pg.team_id.isin(d1p)], prev_raw,
                           R.fit_adjusted_ratings(pl, alpha=3.0)) if not prev_pg.empty else None
    cur = player_ratings(cur_pg, cur_raw, team_ratings, prior=prior) if not cur_pg.empty else \
        pd.DataFrame(columns=["name", "team_id", "minutes", "games", "o", "d", "rating"])
    if external_players is not None and not external_players.empty:
        cur = blend_external(cur, external_players)
    allr = pd.concat([prior[~prior.index.isin(cur.index)], cur]) if prior is not None else cur
    base = baseline_strength(pd.concat([prev_pg, cur_pg]), game_weights, allr)
    rr = cached_return_rates(prev_pg, season - 1) if not prev_pg.empty else DEFAULT_RETURN_PROB
    cur_pg = cur_pg.copy()
    cur_pg["date"] = pd.to_datetime(cur_pg["date"])
    names = live.team_names(prev_raw, cur_raw)
    statuses = current_statuses(cur, names) if not cur.empty else {}
    print(f"[players] {len(cur)} rated this season, {len(statuses)} with a report status")
    return {"ratings": allr, "current": cur, "base": base, "return_rates": rr,
            "pg": cur_pg, "statuses": statuses}


def team_adjustment(ctx: dict, team_id: str) -> tuple[float, float, str]:
    """(dO, dD) per 100 possessions vs the rated baseline, plus a short note."""
    pg = ctx["pg"]
    t = pg[pg["team_id"] == team_id]
    if t["game_id"].nunique() < MIN_TEAM_GAMES or team_id not in ctx["base"].index:
        return 0.0, 0.0, ""
    so, sd, em = project_lineup(t, team_id, ctx["ratings"], statuses=ctx["statuses"],
                                return_prob=ctx["return_rates"])
    b = ctx["base"].loc[team_id]
    note = ""
    if not em.empty:
        miss = em[(em["p_play"] < 0.75) & (em["min_if_plays"] >= 12)].copy()
        if not miss.empty:
            r = ctx["ratings"]["rating"].reindex(miss.index)
            miss["impact"] = (r.fillna(0) - REPLACEMENT) * miss["min_if_plays"] * (1 - miss["p_play"])
            miss = miss.sort_values("impact", ascending=False).head(3)
            nm = ctx["ratings"]["name"].reindex(miss.index).fillna("?")
            parts = []
            for pid, row in miss.iterrows():
                st = ctx["statuses"].get(pid)
                tag = st.lower() if st else (f"missed {int(row['missed_streak'])}" if row["missed_streak"] else "")
                parts.append(f"{nm[pid]} ({tag})" if tag else nm[pid])
            note = ", ".join(parts)
    return so - b["base_o"], sd - b["base_d"], note


def ratings_table(ctx: dict, names: dict, min_minutes: int = 100) -> pd.DataFrame:
    """Current-season player ratings for the Sheets 'Players' tab."""
    c = ctx["current"]
    c = c[c["minutes"] >= min_minutes].copy()
    c["team"] = c["team_id"].map(names)
    c = c.sort_values("rating", ascending=False)
    c.insert(0, "rank", range(1, len(c) + 1))
    return c[["rank", "name", "team", "minutes", "games", "o", "d", "rating"]].round(2)


def blend_external(cur: pd.DataFrame, ext: pd.DataFrame, weight: float = 0.5) -> pd.DataFrame:
    """
    Blend external player ratings (Torvik/KenPom/ShotQuality exports already
    matched to ESPN player ids, columns player_id, o, d) into ours.
    Both are on the points-per-100 plus-minus scale.
    """
    e = ext.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")
    idx = cur.index.intersection(e.index)
    out = cur.copy()
    for c in ("o", "d"):
        if c in e.columns:
            out.loc[idx, c] = (1 - weight) * out.loc[idx, c] + weight * e.loc[idx, c]
    out["rating"] = out["o"] + out["d"]
    return out

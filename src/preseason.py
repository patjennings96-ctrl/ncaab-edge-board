"""
preseason.py
============
Early-season team ratings built from who is actually on the roster.

Before conference play, a team's rating leans on last season. In the
transfer-portal era the average team returns under 30% of its minutes, so
last season's *team* rating describes a roster that has mostly left. This
module builds a starting rating for every team from:

  1. last season's team rating (offense and defense, opponent-adjusted), and
  2. a roster rating: this season's roster x each player's rating from last
     season, at whatever school he played (transfers carry their rating),
     with newcomers at a below-average default,

mixed by roster continuity (share of projected minutes from returning
players). The mix weights were fit on 2025-26 and checked on held-out teams.

The starting rating then enters the efficiency fit as pseudo-games that fade
as real games accumulate. Efficiency is fit on this season's games only, so
the scoring level is always this season's (this fixes the December under bias
caused by last season's lower scoring). Pace and home court still use last
season's games, shifted by this season's year-over-year change on matching
dates.

Rosters come from ESPN (data/rosters/<season>.csv, refreshed weekly); before
that file exists they are inferred from this season's box scores.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import live
import players as P
import ratings as R

ROSTER_DIR = os.path.join(live.ROOT, "data", "rosters")

NEW_MPG = 8.0            # projected minutes for a player with no D-I history
MAX_MPG = 34.0
PLAYER_REGRESS = 0.67    # last season's player rating, a third back to average
# Starting rating is worth this many games, fading as 12 / (12 + games played).
PRIOR_GAMES = 25.0
# Mix of [last, last*continuity, roster, roster*continuity], offense and
# defense. Fit on 2025-26 final ratings (365 D-I teams). Cross-fit error:
# offense 4.87 vs 5.03 for last season alone, defense 4.52 vs 4.63.
MIX_O = np.array([0.476, 0.615, 0.642, -0.840])
MIX_D = np.array([0.441, 0.655, 0.692, -1.053])


# -----------------------------------------------------------------------------
# Rosters
# -----------------------------------------------------------------------------
def roster_path(season: int) -> str:
    return os.path.join(ROSTER_DIR, f"{season}.csv")


def fetch_rosters(season: int, team_ids) -> pd.DataFrame:
    """ESPN roster for every team id (needs network; runs on GitHub Actions)."""
    rows = []
    for tid in team_ids:
        try:
            d = live._get_json(f"{live.ESPN}/teams/{tid}/roster")
        except Exception as exc:
            print(f"[rosters] {tid}: {exc}")
            continue
        for a in d.get("athletes", []):
            # some responses group athletes by position
            items = a.get("items", [a]) if isinstance(a, dict) else []
            for p in items:
                if "id" not in p:
                    continue
                rows.append({"team_id": str(tid), "player_id": str(p["id"]),
                             "name": p.get("fullName") or p.get("displayName", ""),
                             "pos": (p.get("position") or {}).get("abbreviation", ""),
                             "year": (p.get("experience") or {}).get("displayValue", "")})
    df = pd.DataFrame(rows)
    os.makedirs(ROSTER_DIR, exist_ok=True)
    if not df.empty:
        df.to_csv(roster_path(season), index=False)
    print(f"[rosters] season {season}: {len(df)} players on {df.team_id.nunique() if not df.empty else 0} teams")
    return df


def ensure_rosters(season: int, team_ids, max_age_days: int = 7) -> None:
    """Refresh the ESPN roster file if missing or older than a week."""
    path = roster_path(season)
    if os.path.exists(path):
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        if age < timedelta(days=max_age_days):
            return
    try:
        fetch_rosters(season, team_ids)
    except Exception as exc:
        print(f"[rosters] refresh failed: {exc}")


def load_roster(season: int, cur_pg: pd.DataFrame | None = None) -> pd.DataFrame:
    """team_id, player_id. ESPN roster file, plus anyone who has played this season."""
    parts = []
    if os.path.exists(roster_path(season)):
        parts.append(pd.read_csv(roster_path(season), dtype=str)[["team_id", "player_id"]])
    if cur_pg is not None and not cur_pg.empty:
        # a player's current team = the last team he played for
        p = cur_pg.sort_values("date").drop_duplicates("player_id", keep="last")
        parts.append(p[["team_id", "player_id"]].astype(str))
    if not parts:
        return pd.DataFrame(columns=["team_id", "player_id"])
    r = pd.concat(parts[::-1], ignore_index=True)        # box scores win over the roster file
    return r.drop_duplicates("player_id", keep="first")


# -----------------------------------------------------------------------------
# Starting ratings
# -----------------------------------------------------------------------------
def last_season_players(prev_pg: pd.DataFrame, prev_raw: pd.DataFrame, prev_ratings) -> pd.DataFrame:
    """Last season's player ratings (team-adjusted, regressed) and minutes per game."""
    pr = P.player_ratings(prev_pg, prev_raw, prev_ratings)
    pr = pr.copy()
    pr["o"] *= PLAYER_REGRESS
    pr["d"] *= PLAYER_REGRESS
    pr["mpg"] = pr["minutes"] / pr["games"].clip(lower=1)
    return pr


def roster_strength(roster: pd.DataFrame, last_players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Per team: roster O/D strength (points per 100 vs average), share of
    projected minutes from returning players, and the projected minutes
    (player_id -> minutes) used as the lineup baseline.
    """
    r = roster.merge(last_players[["team_id", "o", "d", "mpg"]].rename(columns={"team_id": "team_last"}),
                     left_on="player_id", right_index=True, how="left")
    r["mpg"] = r["mpg"].fillna(NEW_MPG).clip(upper=MAX_MPG)
    rows, minutes = [], {}
    for t, g in r.groupby("team_id"):
        m = g.set_index("player_id")["mpg"]
        if m.sum() > 200:
            m = m * 200 / m.sum()
        so, sd = P.lineup_strength(m, g.set_index("player_id")[["o", "d"]])
        ret = (g.set_index("player_id")["team_last"] == t).reindex(m.index).fillna(False)
        rows.append({"team": t, "so": so, "sd": sd, "ret_min": float(m[ret].sum() / 200)})
        minutes[t] = m
    return pd.DataFrame(rows).set_index("team"), minutes


def starting_ratings(prev_ratings, rstr: pd.DataFrame, teams) -> dict:
    """team -> (off offset, def offset), points per 100 vs average (def: + is good)."""
    out = {}
    for t in teams:
        lo = prev_ratings.off.get(t)
        ld = -prev_ratings.deff[t] if t in prev_ratings.deff else None
        if t in rstr.index:
            z = rstr.loc[t]
            c = float(z.ret_min)
            if lo is None:          # new to D-I or no history: roster only
                lo = ld = 0.0
            o = MIX_O @ [lo, lo * c, z.so, z.so * c]
            d = MIX_D @ [ld, ld * c, z.sd, z.sd * c]
        elif lo is not None:        # no roster information: last season, regressed
            o, d = 0.6 * lo, 0.6 * ld
        else:
            o = d = 0.0
        out[t] = (float(o), float(d))
    return out


def relevel(prev_long: pd.DataFrame, cur_long: pd.DataFrame, min_rows: int = 200) -> pd.DataFrame:
    """
    Shift last season's efficiency and pace by this season's year-over-year
    change, measured on matching dates (this season to date vs last season
    through the same point). Scoring falls ~3 points per game from November
    to January every year, so comparing early games with last season's full
    average would overshoot.
    """
    if len(cur_long) < min_rows or "date" not in cur_long or "date" not in prev_long:
        return prev_long
    p = prev_long.copy()
    cd = pd.to_datetime(cur_long["date"])
    pdates = pd.to_datetime(p["date"])
    same = p[pdates < cd.max() - pd.DateOffset(years=1) + pd.Timedelta(days=1)]
    if len(same) < min_rows:
        return prev_long
    for c in ("raw_oe", "raw_de", "tempo"):
        p[c] = p[c] + (cur_long[c].mean() - same[c].mean())
    return p


def fit_ratings(prev_long: pd.DataFrame, cur_long: pd.DataFrame, start: dict,
                alpha: float = 3.0, hca: float | None = None):
    """
    prev_long / cur_long: long-format D-I games with tempo/efficiency columns.
    start: team -> (off, def) starting ratings. Returns AdjustedRatings.
    """
    n_avg = len(cur_long) / max(cur_long["team"].nunique(), 1) if len(cur_long) else 0.0
    fade = 12.0 / (12.0 + n_avg)
    pp = prev_long.copy()
    pp["weight"] = 0.5 * fade
    cc = cur_long.copy()
    cc["weight"] = 1.0
    pp = relevel(pp, cc)
    # pace and home court: last season (re-leveled) + this season
    pace = R.fit_adjusted_ratings(pd.concat([pp, cc], ignore_index=True), alpha=alpha)
    mu_now = float(cc["raw_oe"].mean()) if len(cc) >= 200 else pace.mu
    prior = R.AdjustedRatings(mu=mu_now, hca=pace.hca, league_tempo=pace.league_tempo,
                              off={t: v[0] for t, v in start.items()},
                              deff={t: -v[1] for t, v in start.items()}, tempo=pace.tempo)
    # Home court from last season's full, balanced schedule. Early-season
    # schedules (strong teams hosting weak ones) inflate it: fit on this
    # season's games through mid-December it came out 3.4-4.9 points per side
    # vs ~2.5 over a full season.
    if hca is None:
        hca = R.fit_adjusted_ratings(prev_long, alpha=alpha).hca
    eff = R.fit_adjusted_ratings(cc, alpha=alpha, prior_ratings=prior,
                                 prior_weight=PRIOR_GAMES * fade, extra_teams=list(start),
                                 fixed_hca=hca)
    eff.tempo, eff.league_tempo = pace.tempo, pace.league_tempo
    eff.prior_games = PRIOR_GAMES * fade
    return eff


def build(season: int, prev_raw, cur_raw, prev_pg, cur_pg, d1: set):
    """Everything the daily run needs: ratings, start ratings, roster minutes."""
    prev_long = R.add_tempo_efficiency(live.to_long(prev_raw))
    cur_long = R.add_tempo_efficiency(live.to_long(cur_raw)) if not cur_raw.empty else prev_long.iloc[:0]
    prev_long = prev_long[prev_long.team.isin(d1) & prev_long.opp.isin(d1)]
    cur_long = cur_long[cur_long.team.isin(d1) & cur_long.opp.isin(d1)]
    prev_ratings = R.fit_adjusted_ratings(prev_long, alpha=3.0)
    lastp = last_season_players(prev_pg[prev_pg.team_id.isin(d1)], prev_raw, prev_ratings) \
        if not prev_pg.empty else None
    roster = load_roster(season, cur_pg)
    if lastp is not None and not roster.empty:
        rstr, minutes = roster_strength(roster[roster.team_id.isin(d1)], lastp)
    else:
        rstr, minutes = pd.DataFrame(columns=["so", "sd", "ret_min"]), {}
    start = starting_ratings(prev_ratings, rstr, sorted(d1))
    ratings = fit_ratings(prev_long, cur_long, start, hca=prev_ratings.hca)
    print(f"[preseason] {len(rstr)} rosters, avg returning minutes "
          f"{rstr.ret_min.mean() if len(rstr) else float('nan'):.0%}, "
          f"starting rating worth {ratings.prior_games:.1f} games")
    return {"ratings": ratings, "start": start, "roster_strength": rstr,
            "roster_minutes": minutes, "prior_games": ratings.prior_games}

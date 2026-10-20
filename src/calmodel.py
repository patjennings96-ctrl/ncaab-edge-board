"""
calmodel.py
===========
Cal's NCAAB model (the "NCAAB Model" Google Sheet), rebuilt in Python so it can
run on the same daily data and lines as the Edge Board.

The sheet's logic, reproduced formula for formula
-------------------------------------------------
Inputs per team
  Model 1 (TeamRankings stats in the sheet): offensive efficiency OE (points per
  possession), defensive efficiency DE (points allowed per possession),
  possessions per game P, effective possession ratio EPR.
  Model 2 (KenPom in the sheet): AdjO, AdjD (points per 100), AdjT (tempo).

MODEL 1, away team score (home is symmetric):
    base_a  = P_a * EPR_a
    proj1   = base_a * OE_a
    proj2   = base_a * (2 - DE_h)
    proj3   = proj1 * DE_h
    away    = mean(proj1, proj2, proj3)

MODEL 2, away team score:
    o_a = AdjO_a / 100,  d_h = AdjD_h / 100,  control = min(AdjT_a, AdjT_h)
    proj1 = AdjT_a * o_a
    proj2 = o_a * AdjT_a * d_h
    proj3 = o_a * control * d_h
    away  = mean(proj1, proj2, proj3)

Values (both models)
    home ATS value = (home - away) + home_line
    total value    = (home + away) - total_line

Plays (LINES tab)
    ATS:   HOME if both models' ATS value >= 5; AWAY if both <= -5
    Total: Model 2 total values are first adjusted by subtracting that day's
           average Model 2 total value ("Adjust Daily").
           OVER if Model 1 >= 7 and adjusted Model 2 >= 7; UNDER if both <= -7.
    Grading: +1 unit win, -1.1 loss (flat, -110).

Home court: the sheet had none. At Pat's direction this version adds a 3-point
home edge to both models' margins (+1.5 home, -1.5 away, so totals are
unchanged), skipped for neutral-site games.

Data substitutions (the sheet scraped sites that block automated access)
    TeamRankings OE/DE/possessions/EPR -> computed from ESPN box scores, same
        definitions: points / possessions, possessions per game, and
        EPR = (possessions - turnovers + offensive rebounds) / possessions.
    KenPom AdjO/AdjD/AdjT -> the KenPom ratings table pasted into
        data/external/kenpom/paste.txt (same copy-paste as the sheet, at most
        3 days old), otherwise the Edge Board's own adjusted ratings.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import live

ROOT = live.ROOT
CAL_JSON = os.path.join(ROOT, "docs", "cal", "data", "plays.json")
CAL_LOG = os.path.join(ROOT, "data", "history", "cal_log.csv")

HCA = 3.0             # home-court points added to each model's margin (half to each side)
ATS_THRESHOLD = 5.0
TOTAL_THRESHOLD = 7.0
MIN_GAMES = 3          # fewer current-season games -> fall back to last season


# -----------------------------------------------------------------------------
# Model 1 inputs from box scores
# -----------------------------------------------------------------------------
def team_box_stats(games_raw: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date OE, DE, possessions per game and EPR per team (raw, unadjusted)."""
    g = games_raw.dropna(subset=["h_fga", "h_fta", "h_oreb", "h_tov",
                                 "a_fga", "a_fta", "a_oreb", "a_tov",
                                 "home_score", "away_score"])
    if g.empty:
        return pd.DataFrame(columns=["oe", "de", "poss", "epr", "games"])
    ph = g.h_fga - g.h_oreb + g.h_tov + 0.475 * g.h_fta
    pa = g.a_fga - g.a_oreb + g.a_tov + 0.475 * g.a_fta
    poss = (ph + pa) / 2                           # per game, overtime included (as TeamRankings)
    rows = pd.concat([
        pd.DataFrame({"team": g.home_id, "pts": g.home_score, "opp": g.away_score,
                      "poss": poss, "tov": g.h_tov, "oreb": g.h_oreb,
                      "game_poss": (ph + pa) / 2}),
        pd.DataFrame({"team": g.away_id, "pts": g.away_score, "opp": g.home_score,
                      "poss": poss, "tov": g.a_tov, "oreb": g.a_oreb,
                      "game_poss": (ph + pa) / 2}),
    ])
    a = rows.groupby("team").agg(pts=("pts", "sum"), opp=("opp", "sum"),
                                 gp=("game_poss", "sum"), poss=("poss", "mean"),
                                 tov=("tov", "sum"), oreb=("oreb", "sum"),
                                 games=("pts", "size"))
    out = pd.DataFrame({
        "oe": a.pts / a.gp,
        "de": a.opp / a.gp,
        "poss": a.poss,
        "epr": (a.gp - a.tov + a.oreb) / a.gp,
        "games": a.games,
    })
    return out


def model1_inputs(cur_raw: pd.DataFrame, prev_raw: pd.DataFrame | None) -> pd.DataFrame:
    """Current-season stats; teams with fewer than MIN_GAMES use last season's."""
    cur = team_box_stats(cur_raw)
    if prev_raw is None or prev_raw.empty:
        return cur
    prev = team_box_stats(prev_raw)
    thin = prev.index.difference(cur[cur.games >= MIN_GAMES].index)
    return pd.concat([cur[cur.games >= MIN_GAMES], prev.loc[thin]])


# -----------------------------------------------------------------------------
# Model 2 inputs
# -----------------------------------------------------------------------------
def model2_inputs(team_ratings, espn_names: dict | None = None) -> tuple[pd.DataFrame, str]:
    """AdjO/AdjD/AdjT: fresh KenPom export if available, else our adjusted ratings."""
    try:
        import external as X
        src = [s for s in X.load_team_sources(espn_names or {}) if s["source"].iloc[0] == "kenpom"]
        if src:
            k = src[0]
            kp = pd.DataFrame({"adjo": k["adj_o"], "adjd": k["adj_d"], "adjt": k["tempo"]}).dropna()
            # teams KenPom's paste didn't cover (name not matched) use our ratings
            teams = [t for t in team_ratings.off if t not in kp.index]
            fill = pd.DataFrame({"adjo": [team_ratings.adj_o(t) for t in teams],
                                 "adjd": [team_ratings.adj_d(t) for t in teams],
                                 "adjt": [team_ratings.team_tempo(t) for t in teams]}, index=teams)
            return pd.concat([kp, fill]), f"KenPom ({len(kp)} teams)"
    except Exception as exc:                                     # pragma: no cover
        print(f"[cal] KenPom import skipped: {exc}")
    teams = list(team_ratings.off)
    return pd.DataFrame({"adjo": [team_ratings.adj_o(t) for t in teams],
                         "adjd": [team_ratings.adj_d(t) for t in teams],
                         "adjt": [team_ratings.team_tempo(t) for t in teams]},
                        index=teams), "Edge Board adjusted ratings"


# -----------------------------------------------------------------------------
# The two models
# -----------------------------------------------------------------------------
def model1(a: pd.Series, h: pd.Series) -> tuple[float, float]:
    def score(x, opp):
        base = x.poss * x.epr
        p1 = base * x.oe
        p2 = base * (2 - opp.de)
        p3 = p1 * opp.de
        return (p1 + p2 + p3) / 3
    return score(a, h), score(h, a)


def model2(a: pd.Series, h: pd.Series) -> tuple[float, float]:
    control = min(a.adjt, h.adjt)

    def score(x, opp):
        o, d = x.adjo / 100, opp.adjd / 100
        p1 = x.adjt * o
        p2 = o * x.adjt * d
        p3 = o * control * d
        return (p1 + p2 + p3) / 3
    return score(a, h), score(h, a)


def price_slate(slate: pd.DataFrame, m1: pd.DataFrame, m2: pd.DataFrame) -> pd.DataFrame:
    """
    slate: one row per game with home, away (team ids), spread_home, total.
    Returns projections, values and plays exactly as the sheet's LINES tab.
    """
    rows = []
    for g in slate.itertuples(index=False):
        if g.home not in m1.index or g.away not in m1.index or \
           g.home not in m2.index or g.away not in m2.index:
            continue
        a1, h1 = model1(m1.loc[g.away], m1.loc[g.home])
        a2, h2 = model2(m2.loc[g.away], m2.loc[g.home])
        if not bool(getattr(g, "neutral", False)):
            # 3-point home edge: margin moves 3, total unchanged
            h1, a1, h2, a2 = h1 + HCA / 2, a1 - HCA / 2, h2 + HCA / 2, a2 - HCA / 2
        r = {"game_id": getattr(g, "game_id", ""), "start": getattr(g, "start", ""),
             "home": g.home, "away": g.away,
             "line": g.spread_home, "total_line": g.total,
             "m1_away": a1, "m1_home": h1, "m2_away": a2, "m2_home": h2}
        r["m1_ats"] = (h1 - a1) + g.spread_home if pd.notna(g.spread_home) else np.nan
        r["m2_ats"] = (h2 - a2) + g.spread_home if pd.notna(g.spread_home) else np.nan
        r["m1_tot"] = (h1 + a1) - g.total if pd.notna(g.total) else np.nan
        r["m2_tot"] = (h2 + a2) - g.total if pd.notna(g.total) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["m2_tot_adj"] = df["m2_tot"] - df["m2_tot"].mean()          # "Adjust Daily"
    df["ats_play"] = np.select(
        [(df.m1_ats >= ATS_THRESHOLD) & (df.m2_ats >= ATS_THRESHOLD),
         (df.m1_ats <= -ATS_THRESHOLD) & (df.m2_ats <= -ATS_THRESHOLD)], ["HOME", "AWAY"], "")
    df["ou_play"] = np.select(
        [(df.m1_tot >= TOTAL_THRESHOLD) & (df.m2_tot_adj >= TOTAL_THRESHOLD),
         (df.m1_tot <= -TOTAL_THRESHOLD) & (df.m2_tot_adj <= -TOTAL_THRESHOLD)], ["OVER", "UNDER"], "")
    df["ats_play_to"] = [ats_play_to(r) for r in df.itertuples(index=False)]
    n_tot = int(df["m2_tot"].notna().sum())
    df["ou_play_to"] = [ou_play_to(r, n_tot) for r in df.itertuples(index=False)]
    return df


def _half_up(x: float) -> float:
    """Smallest half-point line >= x."""
    return float(np.ceil(round(x * 2, 6)) / 2)


def _half_down(x: float) -> float:
    """Largest half-point line <= x."""
    return float(np.floor(round(x * 2, 6)) / 2)


def ats_play_to(r) -> float | None:
    """
    Worst line, from the picked side's view, at which both models still clear
    the spread threshold. HOME needs margin_k + line >= 5 for both models;
    AWAY needs margin_k + home_line <= -5, i.e. away line >= 5 + margin_k.
    """
    if r.ats_play not in ("HOME", "AWAY"):
        return None
    m = [r.m1_home - r.m1_away, r.m2_home - r.m2_away]
    if r.ats_play == "HOME":
        return _half_up(ATS_THRESHOLD - min(m))          # home line
    return _half_up(ATS_THRESHOLD + max(m))              # away line


def ou_play_to(r, n_games: int) -> float | None:
    """
    Worst total at which both models still clear the total threshold.
    Model 2 is compared after the day's adjustment (minus the slate's average
    Model 2 value), and that average moves with this game's line: if the line
    moves by d, the game's own value moves by -d and the average by -d/n.
    """
    if r.ou_play not in ("OVER", "UNDER") or n_games < 2:
        return None
    t1, t2, t0 = r.m1_home + r.m1_away, r.m2_home + r.m2_away, r.total_line
    avg0 = r.m2_tot - r.m2_tot_adj
    k = 1.0 - 1.0 / n_games
    th = TOTAL_THRESHOLD if r.ou_play == "OVER" else -TOTAL_THRESHOLD
    # Model 2 adjusted value at total T: (t2 - T) - avg0 + (T - t0)/n = th
    t_m2 = (t2 - avg0 - t0 / n_games - th) / k
    t_m1 = t1 - th
    if r.ou_play == "OVER":
        return _half_down(min(t_m1, t_m2))
    return _half_up(max(t_m1, t_m2))


def grade(play: str, line: float, total_line: float, home_score: float, away_score: float) -> float | None:
    """Sheet grading: +1 win, -1.1 loss, 0 push."""
    if play in ("HOME", "AWAY"):
        x = (home_score + line) - away_score
        if play == "AWAY":
            x = -x
    elif play in ("OVER", "UNDER"):
        x = (home_score + away_score) - total_line
        if play == "UNDER":
            x = -x
    else:
        return None
    return 1.0 if x > 0 else (-1.1 if x < 0 else 0.0)


# -----------------------------------------------------------------------------
# Daily run
# -----------------------------------------------------------------------------
def run_daily(slate: pd.DataFrame, cur_raw, prev_raw, team_ratings, names: dict,
              mode: str) -> pd.DataFrame:
    m1 = model1_inputs(cur_raw, prev_raw)
    m2, m2_source = model2_inputs(team_ratings, names)
    priced = price_slate(slate, m1, m2) if not slate.empty else pd.DataFrame()
    write_json(priced, names, mode, m2_source)
    if mode == "live":
        if not priced.empty:
            log_plays(priced, names)
        # grade every run, including days with no slate
        grade_log(pd.concat([prev_raw, cur_raw], ignore_index=True))
    n_ats = int((priced.get("ats_play", pd.Series(dtype=str)) != "").sum()) if not priced.empty else 0
    n_ou = int((priced.get("ou_play", pd.Series(dtype=str)) != "").sum()) if not priced.empty else 0
    print(f"[cal] {len(priced)} games priced, {n_ats} ATS plays, {n_ou} total plays (Model 2: {m2_source})")
    return priced


def _r(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def write_json(priced: pd.DataFrame, names: dict, mode: str, m2_source: str) -> None:
    os.makedirs(os.path.dirname(CAL_JSON), exist_ok=True)
    games = []
    for r in priced.itertuples(index=False) if not priced.empty else []:
        games.append({
            "game_id": str(r.game_id), "start": str(r.start),
            "home": names.get(r.home, r.home), "away": names.get(r.away, r.away),
            "line": _r(r.line), "total_line": _r(r.total_line),
            "m1": {"away": _r(r.m1_away), "home": _r(r.m1_home), "ats": _r(r.m1_ats), "tot": _r(r.m1_tot)},
            "m2": {"away": _r(r.m2_away), "home": _r(r.m2_home), "ats": _r(r.m2_ats),
                   "tot": _r(r.m2_tot), "tot_adj": _r(r.m2_tot_adj)},
            "ats_play": r.ats_play, "ou_play": r.ou_play,
            "ats_play_to": _r(getattr(r, "ats_play_to", None)),
            "ou_play_to": _r(getattr(r, "ou_play_to", None)),
        })
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "mode": mode, "model2_source": m2_source, "hca": HCA,
               "thresholds": {"ats": ATS_THRESHOLD, "total": TOTAL_THRESHOLD},
               "record": log_record(), "games": games}
    with open(CAL_JSON, "w") as f:
        json.dump(payload, f, indent=1)


def log_plays(priced: pd.DataFrame, names: dict) -> None:
    rows = []
    for r in priced.itertuples(index=False):
        for mkt, play in (("ATS", r.ats_play), ("TOTAL", r.ou_play)):
            if play:
                rows.append({"posted_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                             "game_id": str(r.game_id), "start": str(r.start),
                             "away": names.get(r.away, r.away), "home": names.get(r.home, r.home),
                             "market": mkt, "play": play, "line": r.line, "total_line": r.total_line,
                             # the weaker model's margin past the line (drives the play)
                             "gap": round(min(abs(r.m1_ats), abs(r.m2_ats)) if mkt == "ATS"
                                          else min(abs(r.m1_tot), abs(r.m2_tot_adj)), 2),
                             "units": ""})
    if not rows:
        return
    new = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CAL_LOG), exist_ok=True)
    if os.path.exists(CAL_LOG):
        old = pd.read_csv(CAL_LOG, dtype={"game_id": str})
        new = new[~(new.game_id + "|" + new.market).isin(old.game_id + "|" + old.market)]
    if not new.empty:
        new.to_csv(CAL_LOG, mode="a", index=False, header=not os.path.exists(CAL_LOG))


def grade_log(games_raw: pd.DataFrame) -> None:
    if not os.path.exists(CAL_LOG) or games_raw.empty:
        return
    log = pd.read_csv(CAL_LOG, dtype={"game_id": str, "units": str}).fillna({"units": ""})
    res = games_raw.drop_duplicates("game_id").set_index(games_raw.drop_duplicates("game_id")["game_id"].astype(str))
    for i, r in log.iterrows():
        if r["units"] != "" or r["game_id"] not in res.index:
            continue
        g = res.loc[r["game_id"]]
        if pd.isna(g["home_score"]):
            continue
        u = grade(r["play"], r["line"], r["total_line"], g["home_score"], g["away_score"])
        if u is not None:
            log.at[i, "units"] = str(u)
    log.to_csv(CAL_LOG, index=False)


def log_record() -> dict:
    if not os.path.exists(CAL_LOG):
        return {}
    log = pd.read_csv(CAL_LOG, dtype={"units": str}).fillna({"units": ""})
    out = {}
    for mkt, g in log.groupby("market"):
        u = pd.to_numeric(g["units"], errors="coerce").dropna()
        out[mkt] = {"w": int((u > 0).sum()), "l": int((u < 0).sum()), "p": int((u == 0).sum()),
                    "units": round(float(u.sum()), 1), "pending": int(len(g) - len(u))}
    return out

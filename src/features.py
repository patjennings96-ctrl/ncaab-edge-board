"""
features.py
===========
Domain-specific feature engineering. Each builder is leakage-safe: it only uses
information available *before* tip-off (prior games, prior-season rosters,
schedule, venue metadata).

Feature families
----------------
1. Rest & Travel  – days rest, back-to-back / 3-in-N flags, consecutive road
   games, great-circle miles travelled, altitude of the venue.
2. Continuity     – minutes-continuity %, roster turnover, and a usage-weighted
   "minutes available" hit when a high-usage player is out.
3. Dynamic HCA    – per-team home edge via empirical-Bayes shrinkage toward the
   league mean (avoids over-fitting tiny home samples).
4. Recent form    – EWMA of efficiency margin and pace vs the season baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_MILES = 3958.8


# ----------------------------------------------------------------------------- 
# 1. REST & TRAVEL
# -----------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_MILES * np.arcsin(np.sqrt(a)))


def rest_travel_features(team_games: pd.DataFrame,
                         venues: pd.DataFrame) -> pd.DataFrame:
    """
    team_games : long, ONE team's schedule sorted by date, with columns
                 game_id, date, team, venue_id, location.
    venues     : venue_id, lat, lon, altitude_ft, home_team.

    Returns per-game: days_rest, is_b2b, games_last_7, consec_road,
    travel_miles, altitude_ft, altitude_delta.
    """
    g = team_games.sort_values("date").copy().reset_index(drop=True)
    v = venues.set_index("venue_id")

    g["date"] = pd.to_datetime(g["date"])
    g["days_rest"] = g["date"].diff().dt.days
    g["is_b2b"] = (g["days_rest"] <= 1).astype(float)

    # games in trailing 7 days (inclusive of current)
    g["games_last_7"] = [
        int(((g["date"] <= d) & (g["date"] > d - pd.Timedelta(days=7))).sum())
        for d in g["date"]
    ]

    # consecutive away games ending at this game
    consec = []
    run = 0
    for loc in g["location"]:
        run = run + 1 if loc == "A" else 0
        consec.append(run)
    g["consec_road"] = consec

    # travel + altitude (vs the team's HOME venue elevation for jet-lag/altitude)
    lat = g["venue_id"].map(v["lat"])
    lon = g["venue_id"].map(v["lon"])
    alt = g["venue_id"].map(v["altitude_ft"])
    home_alt = v.loc[v["home_team"] == g["team"].iloc[0], "altitude_ft"]
    home_alt = float(home_alt.iloc[0]) if len(home_alt) else float(alt.median())

    miles = [0.0]
    for i in range(1, len(g)):
        miles.append(haversine_miles(lat.iloc[i - 1], lon.iloc[i - 1],
                                     lat.iloc[i], lon.iloc[i]))
    g["travel_miles"] = miles
    g["altitude_ft"] = alt.values
    g["altitude_delta"] = alt.values - home_alt   # +ve => playing higher than home
    return g


# -----------------------------------------------------------------------------
# 2. TEAM CONTINUITY / CHEMISTRY
# -----------------------------------------------------------------------------
def minutes_continuity(prev_minutes: pd.DataFrame,
                       current_roster: pd.Series) -> float:
    """
    Bart-Torvik style minutes continuity: of last season's total minutes, what
    fraction was played by players still on this season's roster.

    prev_minutes  : columns [player_id, minutes] from last season.
    current_roster: Series/array of player_ids on this season's roster.
    """
    returning = set(current_roster)
    total = prev_minutes["minutes"].sum()
    if total == 0:
        return np.nan
    kept = prev_minutes.loc[prev_minutes["player_id"].isin(returning), "minutes"].sum()
    return float(kept / total)


def availability_index(roster_usage: pd.DataFrame,
                       out_players: set) -> float:
    """
    Usage-weighted share of offense currently AVAILABLE. A 28%-usage star being
    out hurts far more than a bench player. Returns value in [0, 1].

    roster_usage : columns [player_id, usage]   (usage = % of possessions used
                   while on floor, ~0.10-0.35).
    out_players  : ids unavailable for this game (injury/suspension).
    """
    if roster_usage.empty:
        return 1.0
    w = roster_usage["usage"].clip(lower=0)
    total = w.sum()
    if total == 0:
        return 1.0
    avail = roster_usage.loc[~roster_usage["player_id"].isin(out_players), "usage"].sum()
    return float(avail / total)


# -----------------------------------------------------------------------------
# 3. DYNAMIC HOME-COURT ADVANTAGE  (empirical-Bayes shrinkage)
# -----------------------------------------------------------------------------
def dynamic_hca(home_games: pd.DataFrame, league_hca: float,
                shrink_games: float = 25.0) -> pd.DataFrame:
    """
    Per-team HCA shrunk toward the league mean. A team with few home games barely
    moves off the league baseline; a team with many home games and a consistent
    home bump (altitude, raucous crowd) earns a custom number.

    home_games : columns [team, home_margin_vs_expected]   where
                 home_margin_vs_expected = actual_margin - neutral_proj_margin
                 for that team's home games (the extra points the venue is worth).
    league_hca : league-average HCA in points.
    shrink_games: pseudo-count; larger => stronger pull to the mean.
    """
    out = []
    for team, grp in home_games.groupby("team"):
        n = len(grp)
        obs = grp["home_margin_vs_expected"].mean() if n else league_hca
        # James-Stein / EB shrink toward the league mean
        hca = (n * obs + shrink_games * league_hca) / (n + shrink_games)
        out.append({"team": team, "n_home": n, "hca_team": round(hca, 2)})
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# 4. RECENT FORM (EWMA) vs SEASON BASELINE
# -----------------------------------------------------------------------------
def ewma_form(team_games: pd.DataFrame, halflife: float = 5.0) -> pd.DataFrame:
    """
    Exponentially weighted recent form. Computed with .shift(1) so a game's
    feature value uses only games BEFORE it (no leakage).

    team_games : long, one team, sorted by date, with 'game_em' (single-game
                 efficiency margin = raw_oe - raw_de) and 'tempo'.
    """
    g = team_games.sort_values("date").copy()
    g["form_em_ewma"] = g["game_em"].shift(1).ewm(halflife=halflife).mean()
    g["form_tempo_ewma"] = g["tempo"].shift(1).ewm(halflife=halflife).mean()
    g["season_em_mean"] = g["game_em"].shift(1).expanding().mean()
    # positive => playing better than season-long baseline lately
    g["form_vs_baseline"] = g["form_em_ewma"] - g["season_em_mean"]
    return g


# -----------------------------------------------------------------------------
# Assembly: build the matchup feature row the model consumes.
# -----------------------------------------------------------------------------
def build_matchup_features(home: str, away: str, ratings, neutral: bool,
                           team_feats: dict, hca_table: dict) -> dict:
    """
    Combine the ratings projection with engineered features into one flat dict.
    `team_feats[team]` holds the latest rest/continuity/form features for a team.
    """
    proj = ratings.project(home, away, neutral=neutral)
    hf, af = team_feats.get(home, {}), team_feats.get(away, {})
    custom_hca = 0.0 if neutral else hca_table.get(home, ratings.hca)

    return {
        "home": home, "away": away, "neutral": int(neutral),
        # ratings projection (the physics-based prior)
        "proj_margin": proj["proj_margin"],
        "proj_total": proj["proj_total"],
        "proj_poss": proj["proj_poss"],
        "adj_em_diff": ratings.adj_em(home) - ratings.adj_em(away),
        "adj_o_home": ratings.adj_o(home), "adj_d_home": ratings.adj_d(home),
        "adj_o_away": ratings.adj_o(away), "adj_d_away": ratings.adj_d(away),
        "tempo_sum": ratings.team_tempo(home) + ratings.team_tempo(away),
        "custom_hca": custom_hca,
        # rest / travel deltas (home minus away)
        "rest_diff": hf.get("days_rest", 3) - af.get("days_rest", 3),
        "b2b_diff": hf.get("is_b2b", 0) - af.get("is_b2b", 0),
        "travel_diff": af.get("travel_miles", 0) - hf.get("travel_miles", 0),
        "altitude_delta_away": af.get("altitude_delta", 0),
        "consec_road_away": af.get("consec_road", 0),
        # continuity / availability (home minus away)
        "continuity_diff": hf.get("continuity", 0.5) - af.get("continuity", 0.5),
        "avail_home": hf.get("availability", 1.0),
        "avail_away": af.get("availability", 1.0),
        # recent form
        "form_diff": hf.get("form_vs_baseline", 0) - af.get("form_vs_baseline", 0),
    }

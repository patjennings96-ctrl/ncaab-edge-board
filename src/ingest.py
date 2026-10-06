"""
ingest.py
=========
Data ingestion + the early-season cold-start fix.

DATA SOURCES (legality/ToS notes matter — read README before scraping anything)
-------------------------------------------------------------------------------
* Box scores / play-by-play / schedules:
    - `cbbpy` (PyPI) — pulls ESPN box & PBP, free.
    - sports-reference (sportsdataverse / hoopR), CollegeBasketballData API.
* Adjusted-efficiency reference ratings:
    - Bart Torvik (barttorvik.com) — has public JSON/CSV endpoints (`trank`,
      team results). Used here as the open reference source.
    - KenPom — PAID and its ToS prohibits scraping. If you subscribe, export
      manually or use your own account; do NOT scrape it. The model is source-
      agnostic: drop any ratings table with [team, adj_o, adj_d, tempo] into the
      pipeline and everything downstream works.
* Market odds (lines + closing lines for CLV):
    - The Odds API (the-odds-api.com) — free tier, has NCAAB spreads/totals.
    - Pinnacle / sportsbook screens for the sharpest closing numbers.
* Injuries / availability:
    - Public injury feeds, beat reporters, team availability reports.
* Roster / continuity / recruiting (early-season):
    - Verbal Commits / 247 / On3 recruiting ranks, transfer-portal trackers,
      prior-season minutes from box scores.

Everything below returns tidy DataFrames. Network calls are wrapped so the
module imports cleanly offline; wire in your real keys/endpoints where marked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Schema the rest of the pipeline expects (long format, one row per team-game).
# -----------------------------------------------------------------------------
GAME_COLUMNS = [
    "game_id", "date", "season", "team", "opp", "location",
    "team_score", "opp_score",
    "fga", "fta", "oreb", "tov",
    "opp_fga", "opp_fta", "opp_oreb", "opp_tov",
    "minutes", "venue_id",
]


def fetch_torvik_ratings(season: int) -> pd.DataFrame:
    """
    Fetch Bart Torvik T-Rank ratings -> [team, adj_o, adj_d, tempo, season].
    Endpoint pattern (verify current path): barttorvik.com/trank.php?year=...&csv=1
    Implement with requests; left as a stub so the module imports offline.
    """
    raise NotImplementedError(
        "Wire up the Torvik CSV/JSON endpoint with `requests`. Return columns "
        "['team','adj_o','adj_d','tempo','season']."
    )


def fetch_odds(date: str, api_key: str) -> pd.DataFrame:
    """
    Fetch NCAAB spreads/totals from The Odds API ->
    [game_id, home, away, spread_home, total, over_odds, under_odds,
     home_ml, away_ml, book, snapshot_ts].
    Store snapshots so you can recover BOTH opening and closing lines (for CLV).
    """
    raise NotImplementedError(
        "Call The Odds API /v4/sports/basketball_ncaab/odds with your key and "
        "normalise to the documented columns. Persist every snapshot for CLV."
    )


# -----------------------------------------------------------------------------
# EARLY-SEASON COLD START
# -----------------------------------------------------------------------------
def preseason_prior(prev_ratings: pd.DataFrame,
                    continuity: pd.DataFrame,
                    recruiting: pd.DataFrame,
                    league_avg_em: float = 0.0) -> pd.DataFrame:
    """
    Build a preseason efficiency-margin prior BEFORE current-season samples exist.

    Blend three signals:
      1. Last season's adjusted EM, regressed toward the mean (talent persists
         but not perfectly — historically ~65-75% year-over-year correlation).
      2. Minutes continuity — teams returning more minutes hold their rating;
         heavy turnover regresses harder toward average.
      3. Incoming talent — recruiting class + transfer portal rank as a bump.

    prev_ratings : [team, adj_em]            (last season)
    continuity   : [team, continuity]        (0-1, fraction of minutes returning)
    recruiting   : [team, talent_score]      (z-scored incoming talent, ~N(0,1))
    """
    df = (prev_ratings.merge(continuity, on="team", how="left")
                      .merge(recruiting, on="team", how="left"))
    df["continuity"] = df["continuity"].fillna(0.5)
    df["talent_score"] = df["talent_score"].fillna(0.0)

    # Regression strength scales with turnover: more continuity -> trust last year.
    # reg_factor in [0.55, 0.85] roughly.
    df["reg_factor"] = 0.55 + 0.30 * df["continuity"]
    regressed = league_avg_em + df["reg_factor"] * (df["adj_em"] - league_avg_em)

    # Talent bump: ~2.5 EM points per z-score of incoming talent (tunable).
    df["prior_em"] = regressed + 2.5 * df["talent_score"]
    return df[["team", "prior_em", "continuity", "reg_factor"]]


def blend_prior_with_current(prior_em: pd.DataFrame,
                             current_ratings,
                             games_played: pd.Series,
                             stabilize_games: float = 12.0) -> pd.DataFrame:
    """
    Shrink current-season ratings toward the preseason prior until enough games
    have been played. Weight on current data = n / (n + stabilize_games), so a
    team with 2 games is mostly prior, a team with 20+ is mostly current.
    """
    rows = []
    for _, p in prior_em.iterrows():
        team = p["team"]
        n = float(games_played.get(team, 0))
        w = n / (n + stabilize_games)
        cur = current_ratings.adj_em(team) if team in current_ratings.off else p["prior_em"]
        rows.append({"team": team,
                     "blended_em": round(w * cur + (1 - w) * p["prior_em"], 2),
                     "weight_current": round(w, 3)})
    return pd.DataFrame(rows)


def synthesize_demo_games(n_teams: int = 60, games_per_team: int = 30,
                          seed: int = 7) -> pd.DataFrame:
    """
    Generate a realistic synthetic season (long format) so the pipeline can be
    smoke-tested with NO network. True team strengths are latent; box scores are
    sampled around them. Used by the test harness and the README example.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{ i:02d}" for i in range(n_teams)]
    true_o = {t: rng.normal(105, 8) for t in teams}     # true adj O
    true_d = {t: rng.normal(105, 8) for t in teams}     # true adj D
    true_tempo = {t: rng.normal(68, 5) for t in teams}
    hca = 3.2

    rows, gid = [], 0
    start = pd.Timestamp("2025-11-04")
    for _ in range(n_teams * games_per_team // 2):
        h, a = rng.choice(teams, 2, replace=False)
        gid += 1
        date = start + pd.Timedelta(days=int(rng.integers(0, 130)))
        poss = (true_tempo[h] * true_tempo[a] / 68.0)
        oe_h = true_o[h] - (true_d[a] - 105) + hca + rng.normal(0, 9)
        oe_a = true_o[a] - (true_d[h] - 105) + rng.normal(0, 9)
        ps_h = max(40, oe_h * poss / 100.0)
        ps_a = max(40, oe_a * poss / 100.0)
        for team, opp, loc, ps, pa in [(h, a, "H", ps_h, ps_a),
                                       (a, h, "A", ps_a, ps_h)]:
            rows.append({
                "game_id": gid, "date": date, "season": 2026,
                "team": team, "opp": opp, "location": loc,
                "team_score": round(ps), "opp_score": round(pa),
                "fga": 55 + rng.integers(-6, 7), "fta": 18 + rng.integers(-6, 8),
                "oreb": 10 + rng.integers(-4, 5), "tov": 12 + rng.integers(-4, 5),
                "opp_fga": 55 + rng.integers(-6, 7), "opp_fta": 18 + rng.integers(-6, 8),
                "opp_oreb": 10 + rng.integers(-4, 5), "opp_tov": 12 + rng.integers(-4, 5),
                "minutes": 40, "venue_id": h,
            })
    return pd.DataFrame(rows)

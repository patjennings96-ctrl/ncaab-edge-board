"""
ratings.py
==========
Tempo-free adjusted-efficiency engine (a transparent KenPom / Bart-Torvik style
replication).

Two things matter for the rest of the model:

1.  **Possessions / Tempo** – everything is expressed *per 100 possessions* so a
    fast team and a slow team can be compared on the same scale.
2.  **Opponent adjustment** – raw efficiency is meaningless without accounting for
    strength of schedule. We solve the whole season as one regularised linear
    system (ridge) instead of the iterative averaging KenPom uses. The ridge
    formulation is mathematically cleaner, converges deterministically, and makes
    point-in-time fitting (no leakage) trivial.

Model
-----
For every (game, offense-team, defense-team) observation:

    OE_obs = mu + off[team] + def[opp] + hca * is_home + eps

    OE_obs  : points scored per 100 possessions in that game
    mu      : Division-I average efficiency (the intercept)
    off[t]  : team t's offensive rating above/below average vs an average defense
    def[o]  : points/100 opponent o concedes vs an average offense
              (a GOOD defense has a NEGATIVE coefficient)
    hca     : home-court bump in points/100

We fit one ridge regression on the stacked offensive rows of every game.
Adjusted ratings are then:

    AdjO[t] = mu + off[t]          (offense vs avg D on neutral court)
    AdjD[t] = mu + def[t]          (defense vs avg O on neutral court; lower = better)
    AdjEM[t]= AdjO[t] - AdjD[t]    (net efficiency margin – the headline number)

CRITICAL: `fit()` must only ever see games strictly *before* the prediction date.
The backtester enforces this; never call it on the full season and then predict
past games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --- Possession-estimate coefficient -----------------------------------------
# Dean Oliver's possession formula. 0.475 is the value most commonly fit to
# college box scores for the FTA term; 0.44 is the NBA convention. Tunable.
FT_POSS_COEF = 0.475


def estimate_possessions(fga: pd.Series, fta: pd.Series,
                         oreb: pd.Series, tov: pd.Series) -> pd.Series:
    """Single-team possession estimate: FGA - OREB + TOV + 0.475*FTA."""
    return fga - oreb + tov + FT_POSS_COEF * fta


def add_tempo_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw possessions, tempo, and raw offensive/defensive efficiency.

    `games` is LONG format – one row per team per game – with columns:
        game_id, date, season, team, opp, location ('H'/'A'/'N'),
        team_score, opp_score,
        fga, fta, oreb, tov,            (team's own box)
        opp_fga, opp_fta, opp_oreb, opp_tov,
        minutes                          (40 + 5*number_of_OTs)
    """
    g = games.copy()
    poss_team = estimate_possessions(g.fga, g.fta, g.oreb, g.tov)
    poss_opp = estimate_possessions(g.opp_fga, g.opp_fta, g.opp_oreb, g.opp_tov)
    # KenPom averages both teams' estimates to get a single game possession count.
    g["poss"] = (poss_team + poss_opp) / 2.0
    g["tempo"] = g["poss"] * 40.0 / g["minutes"]            # possessions / 40 min
    g["raw_oe"] = 100.0 * g["team_score"] / g["poss"]       # points / 100 poss
    g["raw_de"] = 100.0 * g["opp_score"] / g["poss"]
    g["is_home"] = (g["location"] == "H").astype(float)
    g["is_away"] = (g["location"] == "A").astype(float)
    return g


@dataclass
class AdjustedRatings:
    """Fitted ratings table + projection helpers."""
    mu: float                                   # D-I average efficiency
    hca: float                                  # home-court bump (pts / 100 poss)
    league_tempo: float                         # average tempo
    off: Dict[str, float] = field(default_factory=dict)   # offensive coefs
    deff: Dict[str, float] = field(default_factory=dict)  # defensive coefs
    tempo: Dict[str, float] = field(default_factory=dict) # team tempo
    prior_em: Dict[str, float] = field(default_factory=dict)  # for fallbacks

    # ---- team-level accessors -------------------------------------------------
    def adj_o(self, team: str) -> float:
        return self.mu + self.off.get(team, 0.0)

    def adj_d(self, team: str) -> float:
        return self.mu + self.deff.get(team, 0.0)

    def adj_em(self, team: str) -> float:
        return self.adj_o(team) - self.adj_d(team)

    def team_tempo(self, team: str) -> float:
        return self.tempo.get(team, self.league_tempo)

    # ---- game projection ------------------------------------------------------
    def project(self, home: str, away: str, neutral: bool = False) -> dict:
        """
        Project a single game. Returns expected scores, margin and total.

        Pace projection (KenPom-style log5 of tempo):
            poss = tempo_home * tempo_away / league_tempo
        Efficiency projection:
            OE_home = mu + off[home] + def[away] (+ hca if home court)
        """
        poss = self.team_tempo(home) * self.team_tempo(away) / self.league_tempo
        hca = 0.0 if neutral else self.hca

        oe_home = self.mu + self.off.get(home, 0.0) + self.deff.get(away, 0.0) + hca
        oe_away = self.mu + self.off.get(away, 0.0) + self.deff.get(home, 0.0)

        pts_home = oe_home * poss / 100.0
        pts_away = oe_away * poss / 100.0
        return {
            "proj_poss": poss,
            "proj_home": pts_home,
            "proj_away": pts_away,
            "proj_margin": pts_home - pts_away,   # +ve => home favoured
            "proj_total": pts_home + pts_away,
        }


def fit_adjusted_ratings(games_long: pd.DataFrame,
                         alpha: float = 50.0,
                         prior_ratings: "AdjustedRatings | None" = None,
                         prior_weight: float = 0.0) -> AdjustedRatings:
    """
    Fit adjusted O/D ratings + HCA via ridge on the offensive rows of every game.

    Parameters
    ----------
    games_long   : output of add_tempo_efficiency (long format).
    alpha        : ridge penalty. Larger -> heavier shrinkage toward the mean,
                   which is exactly what you want early in the season when each
                   team has few games (see `ingest.py` early-season priors).
    prior_ratings/prior_weight : optionally blend last season's ratings in as
                   pseudo-observations (the "regression to prior year" early-season
                   fix). prior_weight is the number of league-average-strength
                   synthetic games injected per team.
    """
    from scipy import sparse

    g = games_long
    teams = sorted(set(g.team) | set(g.opp))
    t_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_cols = 2 * n_teams + 1

    # Sparse one-hot design: [offense block | defense block | home flag]
    n = len(g)
    off_col = g["team"].map(t_idx).to_numpy()
    def_col = n_teams + g["opp"].map(t_idx).to_numpy()
    home = (g["location"] == "H").to_numpy()
    r_idx = np.concatenate([np.arange(n), np.arange(n), np.flatnonzero(home)])
    c_idx = np.concatenate([off_col, def_col, np.full(home.sum(), n_cols - 1)])
    X = sparse.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(n, n_cols))
    y = g["raw_oe"].to_numpy(dtype=float)
    # Optional per-row weights (e.g. last season's games down-weighted)
    sw = g["weight"].to_numpy(dtype=float) if "weight" in g.columns else np.ones(n)

    # Optional last-season prior injected as synthetic average-context games.
    if prior_ratings is not None and prior_weight > 0:
        rows, cols, extra_y = [], [], []
        for k, t in enumerate(teams):
            rows += [2 * k, 2 * k + 1]
            cols += [t_idx[t], n_teams + t_idx[t]]
            extra_y += [prior_ratings.adj_o(t), prior_ratings.adj_d(t)]
        P = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(2 * n_teams, n_cols))
        X = sparse.vstack([X, P]).tocsr()
        y = np.concatenate([y, np.asarray(extra_y)])
        sw = np.concatenate([sw, np.full(2 * n_teams, prior_weight)])

    ridge = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg",
                  max_iter=5000, tol=1e-6)
    ridge.fit(X, y, sample_weight=sw)

    coef = ridge.coef_
    mu = float(ridge.intercept_)
    off = {t: float(coef[t_idx[t]]) for t in teams}
    deff = {t: float(coef[n_teams + t_idx[t]]) for t in teams}
    hca = float(coef[-1])

    # Tempo: simple per-team mean (could also be opponent-adjusted the same way).
    if "weight" in g.columns:
        wt = g["weight"] * g["tempo"]
        tempo = (wt.groupby(g["team"]).sum() / g["weight"].groupby(g["team"]).sum()).to_dict()
        league_tempo = float(wt.sum() / g["weight"].sum())
    else:
        tempo = g.groupby("team")["tempo"].mean().to_dict()
        league_tempo = float(g["tempo"].mean())

    return AdjustedRatings(mu=mu, hca=hca, league_tempo=league_tempo,
                           off=off, deff=deff, tempo=tempo)


def ratings_table(r: AdjustedRatings) -> pd.DataFrame:
    """Tidy DataFrame of every team's adjusted ratings (Sheets-friendly)."""
    teams = sorted(r.off)
    df = pd.DataFrame({
        "team": teams,
        "adj_o": [round(r.adj_o(t), 2) for t in teams],
        "adj_d": [round(r.adj_d(t), 2) for t in teams],
        "adj_em": [round(r.adj_em(t), 2) for t in teams],
        "tempo": [round(r.team_tempo(t), 1) for t in teams],
    }).sort_values("adj_em", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df

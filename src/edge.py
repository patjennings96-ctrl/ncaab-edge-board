"""
edge.py
=======
Turn a point projection + an error distribution into a *probability* of covering,
compare it to the market's de-vigged probability, and size the bet with fractional
Kelly.

Key ideas
---------
* A projection alone is useless for betting. What matters is P(cover) vs the price.
* We treat the model's margin/total prediction as the mean of a Normal whose
  standard deviation is estimated empirically from walk-forward residuals
  (`backtest.py` measures it – historically ~10-11 pts for margins, ~12-14 for
  totals in college basketball).
* "Edge" = model probability minus the fair (de-vigged) market probability.
* Bet only when edge clears a threshold that comfortably covers the vig, then
  stake a *fraction* of full Kelly because the model's probabilities are
  themselves uncertain (over-betting Kelly is how bankrolls die).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# American-odds break-even at standard -110 juice = 0.5238...
STANDARD_VIG_ODDS = -110


# ---- odds conversions --------------------------------------------------------
def american_to_decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def american_to_prob(american: float) -> float:
    """Raw (vig-inclusive) implied probability of a single American price."""
    return (100.0 / (american + 100.0) if american > 0
            else abs(american) / (abs(american) + 100.0))


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Remove the vig from a two-sided market and return fair probabilities that
    sum to 1 (proportional / "multiplicative" method).
    """
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    s = pa + pb
    return pa / s, pb / s


# ---- cover probabilities -----------------------------------------------------
def prob_cover_spread(proj_margin: float, spread: float, sigma: float) -> float:
    """
    P(home team covers).  `spread` is the HOME line (negative = home favoured,
    e.g. -4.5).  Home covers when actual_margin + spread > 0, i.e.
    margin > -spread.  margin ~ Normal(proj_margin, sigma).
    """
    return float(norm.cdf((proj_margin + spread) / sigma))


def prob_over(proj_total: float, total_line: float, sigma: float) -> float:
    """P(game goes OVER `total_line`).  total ~ Normal(proj_total, sigma)."""
    return float(1.0 - norm.cdf((total_line - proj_total) / sigma))


# ---- Kelly -------------------------------------------------------------------
def kelly_fraction(p: float, american_odds: float) -> float:
    """
    Full-Kelly fraction of bankroll for a single binary bet.
        f* = (p*b - q) / b,  where b = decimal_odds - 1, q = 1 - p.
    Negative -> no bet.
    """
    b = american_to_decimal(american_odds) - 1.0
    f = (p * b - (1.0 - p)) / b
    return max(f, 0.0)


@dataclass
class BetRecommendation:
    market: str            # 'spread' or 'total'
    side: str              # 'HOME'/'AWAY' or 'OVER'/'UNDER'
    line: float
    odds: float
    model_prob: float
    fair_prob: float
    edge: float            # model_prob - fair_prob
    ev_per_unit: float     # expected profit per 1u staked
    kelly_full: float
    stake_units: float     # fractional-Kelly stake, capped

    def as_row(self) -> dict:
        return {
            "market": self.market, "side": self.side, "line": self.line,
            "odds": self.odds,
            "model_prob": round(self.model_prob, 4),
            "fair_prob": round(self.fair_prob, 4),
            "edge_pct": round(100 * self.edge, 2),
            "ev_per_unit": round(self.ev_per_unit, 4),
            "stake_units": round(self.stake_units, 3),
        }


def _ev_per_unit(p: float, american_odds: float) -> float:
    b = american_to_decimal(american_odds) - 1.0
    return p * b - (1.0 - p)


def recommend_spread(proj_margin: float, sigma: float,
                     home_spread: float, home_odds: float, away_odds: float,
                     kelly_mult: float = 0.25, min_edge: float = 0.02,
                     max_stake: float = 5.0) -> BetRecommendation | None:
    """Evaluate both sides of a spread and return the +EV side, if any."""
    p_home = prob_cover_spread(proj_margin, home_spread, sigma)
    p_away = 1.0 - p_home
    fair_home, fair_away = devig_two_way(home_odds, away_odds)

    cands = [
        ("HOME", home_spread, home_odds, p_home, fair_home),
        ("AWAY", -home_spread, away_odds, p_away, fair_away),
    ]
    best = None
    for side, line, odds, p, fair in cands:
        edge = p - fair
        if edge < min_edge:
            continue
        stake = min(kelly_mult * kelly_fraction(p, odds) * 100.0, max_stake)
        if stake <= 0:
            continue
        rec = BetRecommendation("spread", side, line, odds, p, fair, edge,
                                _ev_per_unit(p, odds), kelly_fraction(p, odds), stake)
        if best is None or rec.edge > best.edge:
            best = rec
    return best


def recommend_total(proj_total: float, sigma: float,
                    total_line: float, over_odds: float, under_odds: float,
                    kelly_mult: float = 0.25, min_edge: float = 0.02,
                    max_stake: float = 5.0) -> BetRecommendation | None:
    """Evaluate Over and Under and return the +EV side, if any."""
    p_over = prob_over(proj_total, total_line, sigma)
    p_under = 1.0 - p_over
    fair_over, fair_under = devig_two_way(over_odds, under_odds)

    cands = [
        ("OVER", over_odds, p_over, fair_over),
        ("UNDER", under_odds, p_under, fair_under),
    ]
    best = None
    for side, odds, p, fair in cands:
        edge = p - fair
        if edge < min_edge:
            continue
        stake = min(kelly_mult * kelly_fraction(p, odds) * 100.0, max_stake)
        if stake <= 0:
            continue
        rec = BetRecommendation("total", side, total_line, odds, p, fair, edge,
                                _ev_per_unit(p, odds), kelly_fraction(p, odds), stake)
        if best is None or rec.edge > best.edge:
            best = rec
    return best


# ---- Closing Line Value ------------------------------------------------------
def clv_points(bet_line: float, closing_line: float, side: str) -> float:
    """
    CLV in points. Positive means you got a better number than the close.
    For HOME/OVER a higher (less negative) close vs your line is bad; the sign
    convention below makes "beat the close" always positive.
    """
    if side in ("HOME", "OVER"):
        return closing_line - bet_line
    return bet_line - closing_line


def clv_prob(bet_odds: float, close_odds_same_side: float,
             close_odds_other_side: float) -> float:
    """CLV in probability: your de-vigged win prob at the close minus 0.5-ish."""
    fair_you, _ = devig_two_way(close_odds_same_side, close_odds_other_side)
    return fair_you - american_to_prob(bet_odds)  # crude but directionally right

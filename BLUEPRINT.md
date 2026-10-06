# NCAAB Betting Model — Architectural Blueprint

A production blueprint for projecting college-basketball spreads (ATS) and totals
(O/U), grounded in tempo-free efficiency, opponent adjustment, and disciplined,
leakage-safe validation.

A note on framing before any of the engineering: the deliverable below is a
*rigorous system for finding and sizing edges*, not a promise of profit. The
closing line in major NCAAB markets aggregates an enormous amount of sharp money
and is hard to beat. The honest objective is to build something whose
projections beat the closing number often enough to clear the vig, prove it
out-of-sample, and stake it conservatively. Everything here is engineered around
that objective.

---

## 1. Architectural Overview & Data Ingestion

### 1.1 Pipeline

```
        ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
SOURCES │  Collect    │ → │  Transform   │ → │  Feature      │ → │  Model +     │
        │  (ingest)   │   │  (ratings)   │   │  store        │   │  Edge        │
        └─────────────┘   └──────────────┘   └───────────────┘   └──────┬───────┘
                                                                         │
                                            ┌────────────────────────────┴───────┐
                                            │  Backtest (offline)  Sheets (live)  │
                                            └─────────────────────────────────────┘
```

- **Collect** — box scores, play-by-play, schedule, venue metadata, market lines
  (with snapshots so opening *and* closing numbers are recoverable), injuries,
  rosters/recruiting. Persist raw to a dated store (Parquet/SQLite). Raw data is
  immutable; never overwrite history — point-in-time reconstruction depends on it.
- **Transform** — compute possessions, tempo, raw and opponent-adjusted
  efficiency. This is the analytical core (`ratings.py`).
- **Feature store** — one wide table keyed by `(game_id, side)` plus rolling
  per-team state, all computed with `.shift()`/`< date` filters so a row only
  ever contains pre-tip information.
- **Model + Edge** — dual model → projection → P(cover) → Edge % → Kelly stake.
- **Two consumers** — the *backtest* (offline, walk-forward) and the *live daily
  card* (writes to Sheets).

### 1.2 Data sources

| Layer | Source | Why |
|-------|--------|-----|
| Box / PBP / schedule | `cbbpy`, hoopR, CollegeBasketballData API | possessions, four factors, results |
| Reference ratings | **Bart Torvik** (open CSV/JSON) | sanity-check our adjusted ratings; open ToS |
| Reference ratings | KenPom (**paid, no scraping**) | use your own export only |
| Market lines | The Odds API, Pinnacle | the thing we must beat; snapshot for CLV |
| Injuries / availability | public injury/availability feeds | usage-weighted availability feature |
| Recruiting / transfers | 247/On3, portal trackers | early-season prior |

The system is deliberately **source-agnostic**: any ratings table with
`[team, adj_o, adj_d, tempo]` drops in, so you can swap Torvik for your own
KenPom export without touching downstream code.

### 1.3 The early-season problem

Before ~10–12 games, current-season efficiency is mostly noise. Three-part fix
(`ingest.preseason_prior` + `blend_prior_with_current`):

1. **Regress last year's rating to the mean.** Year-over-year adjusted-EM
   correlation is roughly 0.65–0.75, so a preseason prior is
   `mean + r·(last_year_EM − mean)`.
2. **Modulate by continuity.** Returning minutes predict how much last year
   carries over; the regression factor scales with minutes-continuity
   (`0.55 + 0.30·continuity`). High-turnover teams regress harder.
3. **Add incoming talent.** Recruiting-class + transfer-portal rank, z-scored,
   adds ~2.5 EM points per standard deviation (tunable).

Then **blend**: current-season weight = `n / (n + k)` with `k ≈ 12`. A team with
2 games is ~85% prior; by 25 games it's ~70% current. The ridge in `ratings.py`
also takes a **higher `alpha` early** (heavier shrinkage) and supports injecting
the prior as synthetic average-context observations.

---

## 2. Feature Engineering & Domain Metrics

### 2.1 Possessions, tempo, efficiency

Single-team possession estimate (Dean Oliver):

```
Poss = FGA − OREB + TOV + 0.475·FTA
```

Game possessions = mean of both teams' estimates (KenPom convention). Then:

```
Tempo  = Poss · 40 / minutes                 (possessions per 40, OT-aware)
RawOE  = 100 · PointsFor   / Poss            (points per 100 possessions)
RawDE  = 100 · PointsAgainst / Poss
```

Per-100 normalization is essential: it removes pace, so a 58-possession grind-it-
out team and an 80-possession track meet are compared on offensive/defensive
*quality*, not volume.

### 2.2 Opponent adjustment (the heart)

Raw efficiency is schedule-contaminated. We solve the whole slate as one ridge
regression over the offensive rows of every game:

```
RawOE_obs = μ + off[team] + def[opp] + hca·is_home + ε
```

- `μ` — Division-I average efficiency (intercept).
- `off[t]` — team t's offense vs an average defense.
- `def[o]` — points/100 opponent o concedes vs an average offense (**good defense
  = negative**).
- `hca` — home bump in points/100.

Then `AdjO[t] = μ + off[t]`, `AdjD[t] = μ + def[t]`,
`AdjEM[t] = AdjO[t] − AdjD[t]`.

**Why ridge instead of KenPom's iterative averaging?** It is a single
deterministic linear system (no convergence babysitting), the `alpha` penalty is
*exactly* the early-season shrinkage you want, and point-in-time fitting is
trivial — just pass only the games before the cutoff. The smoke test recovers the
latent team strengths cleanly.

**Pace + score projection** (`AdjustedRatings.project`):

```
Poss     = tempo_home · tempo_away / league_tempo      (log5-style)
OE_home  = μ + off[home] + def[away] (+ hca)
Pts_home = OE_home · Poss / 100
Margin   = Pts_home − Pts_away      Total = Pts_home + Pts_away
```

This projection alone is a strong KenPom-style baseline; the ML layer learns the
residual on top of it.

### 2.3 Rest & travel (`features.rest_travel_features`)

- `days_rest`, `is_b2b`, `games_last_7` — fatigue, common in multi-team events
  and conference clusters.
- `consec_road` — extended road trips compound.
- `travel_miles` — great-circle (haversine) distance between consecutive venues.
- `altitude_ft`, `altitude_delta` — visitors to Denver, Air Force, Wyoming, Utah,
  New Mexico fade late; delta vs the team's home elevation captures the shock.

### 2.4 Continuity & chemistry

- `minutes_continuity` — Torvik-style fraction of last season's minutes returning.
  Critical in the transfer-portal era.
- `availability_index` — **usage-weighted** share of offense available. A 28%-
  usage star being out costs far more than a bench player; weighting by usage
  (possessions used per floor-minute) captures that asymmetry, unlike a raw
  player count.

### 2.5 Dynamic home-court advantage (`features.dynamic_hca`)

A static league HCA throws away real signal — altitude, travel difficulty, and
crowd intensity vary by arena. We estimate per-team HCA but **shrink it toward
the league mean** with empirical Bayes:

```
HCA_team = (n·observed + k·league_HCA) / (n + k),   k ≈ 25 games
```

Teams with few home games barely move off baseline; teams with many home games
and a consistent venue effect earn a custom number. Attendance/crowd-density data
can be folded into `observed` when available.

### 2.6 Recent form vs baseline (`features.ewma_form`)

Exponentially weighted recent efficiency margin (`halflife ≈ 5 games`) minus the
season-long expanding mean gives `form_vs_baseline` — positive when a team is
playing above its season norm. Computed with `.shift(1)` so the current game is
never in its own feature. EWMA beats a flat window because it ages information
smoothly instead of dropping it off a cliff.

---

## 3. Machine-Learning Strategy

### 3.1 Dual-model architecture

Two targets need two models because their error structure differs:

- **MarginModel** → scoring margin → drives **spreads**.
- **TotalModel** → combined points → drives **totals**.

Margin and total are weakly correlated and respond to different features (a
pace-up, two-bad-defense game spikes the total without moving the margin), so
splitting them lets each specialize. From the two you can reconstruct both team
scores if you want exact-score outputs (`home = (total+margin)/2`).

### 3.2 Algorithms (`model.py`)

Each target is an **ensemble** of:

1. **Gradient-boosted trees** (LightGBM, or sklearn `HistGradientBoosting` as a
   zero-extra-dependency fallback). Trees capture non-linear interactions a
   linear projection misses — tired team **and** altitude **and** fast opponent.
2. **Bayesian Ridge.** College box-score data is high-variance / low signal; a
   heavily regularized linear model is brutally hard to beat, barely overfits,
   and yields a predictive variance. It anchors the ensemble.

Averaging them (default 0.6 GBM / 0.4 ridge) cuts variance — the whole game when
irreducible noise (~10–11 points on a margin) dwarfs any extractable edge. And
all of it sits **on top of the ratings projection**, which is itself a strong
prior; the models learn corrections, not the whole signal from scratch.

### 3.3 Loss functions

- **Point head: Huber**, not plain MSE. Blowouts (30+ point margins) are heavy-
  tailed; Huber caps their leverage so a few garbage-time results don't warp the
  fit.
- **Counts: Poisson/Tweedie** are defensible for raw scores, but modeling margin
  and total directly with Huber is simpler and empirically competitive.
- **Uncertainty: quantile heads (q=0.16 / 0.84 ≈ ±1σ)** plus the empirical
  walk-forward residual std. This is the crucial piece: a point prediction can't
  be bet. We need **σ** to convert a projection into **P(cover)** via a Normal and
  compare it to the market's price. The edge layer (§5) consumes `(mean, σ)`.

The deepest point: minimizing RMSE to the *actual score* is not the same as
*beating the line*. The line is the benchmark. So the model output is converted
to a cover probability and bet only when it diverges from the de-vigged market
probability by more than the vig — RMSE is a diagnostic, CLV is the verdict.

---

## 4. Backtesting & Validation (`backtest.py`)

### 4.1 Walk-forward, no leakage

The one rule that makes or breaks everything: **to predict games on date D,
nothing the model sees may come from date ≥ D** — not ratings, not features, not
fitted weights. `walk_forward` marches the calendar in steps; at each step it
refits ratings and the model on the expanding past, then scores the next slice.
The most common silent bug in public models is fitting season-long adjusted
ratings and then "backtesting" past games with them — that leaks the future into
the rating and manufactures fake edge. This framework structurally prevents it.

### 4.2 Success metrics

| Metric | What it tells you |
|--------|-------------------|
| Margin / Total **RMSE & MAE** | projection accuracy (diagnostic) |
| **ATS hit-rate** | break-even at -110 is **52.38%** |
| **ROI** (flat and Kelly) | profit per unit staked — noisy in small samples |
| **CLV** | did we beat the closing line? **the leading indicator** |
| **Kelly bankroll** | simulated growth + max drawdown |

### 4.3 Closing Line Value

Record the line when you'd bet and the closing line; `clv_points` is signed so
"got a better number than the close" is always positive, with a probability
version via de-vig. **Positive average CLV with flat short-run ROI is variance;
negative CLV is a broken model regardless of short-run wins.** CLV stabilizes far
faster than ROI, so it's the primary health check before risking real money.

---

## 5. Production Implementation

The runnable modules:

- `ratings.py` — `add_tempo_efficiency`, `fit_adjusted_ratings` (the
  preprocessing that computes adjusted efficiency and pace).
- `model.py` — `DualModel.train` / `.predict` (GBM + Bayesian Ridge ensemble with
  quantile-based σ).
- `edge.py` — `recommend_spread` / `recommend_total`: de-vig → P(cover) → Edge %
  → fractional-Kelly stake.

### 5.1 Edge and Kelly math

For a spread with home line `L` and projection `margin ~ Normal(μ, σ)`:

```
P(home covers) = Φ( (μ + L) / σ )
fair_prob      = devig(home_odds, away_odds)
Edge           = P_model − fair_prob
```

Stake via **fractional Kelly**:

```
b      = decimal_odds − 1                  (≈ 0.909 at -110)
f*     = (P·b − (1−P)) / b                 (full Kelly)
stake  = clamp(kelly_mult · f*, 0, max)    kelly_mult = 0.25
```

Quarter-Kelly is non-negotiable: full Kelly assumes your probabilities are
*exactly right*, which they never are. Model error means full Kelly massively
over-bets and invites ruinous drawdowns; ¼-Kelly keeps growth while surviving the
inevitable cold streaks. A hard `max_stake` cap backstops it.

### 5.2 Verified behavior

The package was smoke-tested end-to-end on a synthetic season:

- Ridge recovers the latent team strengths and a sensible HCA.
- The dual model trains and emits `(mean, σ)` for margin and total.
- The edge layer produces Edge % and fractional-Kelly stakes.
- `walk_forward` runs without leakage errors and reports RMSE/MAE/ATS/ROI/CLV.

The synthetic ATS/ROI numbers are **inflated on purpose** (the fake lines hug the
projection). On real, efficient markets, expect most candidate edges to vanish
into the vig — which is exactly why the CLV-first validation discipline exists.

---

## Risk & honesty checklist

- Sports betting is **−EV by default** because of the vig; a real edge is rare,
  small, and fragile.
- A good backtest is a **hypothesis**, not proof. Paper-trade, then bet tiny.
- **CLV first.** If you're not beating closing lines, you don't have an edge.
- Respect data ToS (don't scrape KenPom). Respect your jurisdiction's laws and
  age limits. Stake only what you can afford to lose.
- This is engineering, not financial advice.

# NCAAB ATS / Totals Model

A modular, leakage-safe pipeline for projecting college-basketball spreads and
totals, comparing projections to market lines, and sizing bets with fractional
Kelly. Outputs land in Google Sheets and can refresh daily.

> **Read this first — honest expectations.** Major NCAAB markets (especially the
> *closing* line) are very efficient. Most public models lose money after the
> -110 vig. The realistic goal is not "guaranteed profit" — it is to **beat the
> closing line consistently (positive CLV)**, which is the only reliable leading
> indicator of a real edge. A backtest that looks profitable is not proof; it is
> a hypothesis. The synthetic demo numbers in this repo are inflated by design
> (the fake "lines" sit near the projection) and are **not** representative of
> live results. Treat any edge skeptically, paper-trade first, and never stake
> money you can't afford to lose. This is software, not financial advice.

## Architecture

```
ingest.py    -> data sources + early-season prior (cold start)
ratings.py   -> possessions, tempo, opponent-adjusted O/D efficiency (ridge)
features.py  -> rest/travel/altitude, continuity, dynamic HCA, EWMA form
model.py     -> dual model: margin (spreads) + total (O/U), GBM + Bayesian Ridge
edge.py      -> de-vig, P(cover), Edge %, fractional Kelly
backtest.py  -> walk-forward CV: RMSE/MAE, ATS%, ROI, CLV, Kelly bankroll
sheets.py    -> Google Sheets export
run_daily.py -> orchestration entrypoint
```

See **BLUEPRINT.md** for the full methodology and math justifications.

## Quick start (no credentials needed)

```bash
pip install -r requirements.txt
cd src
python run_daily.py                 # demo card from synthetic data -> docs/data/plays.json
```

To run on real data, implement the `fetch_*` functions in `ingest.py` and call
`run_daily.run(demo=False, sheet_id=...)`.

## Data sources

| Need | Source | Notes |
|------|--------|-------|
| Box / play-by-play / schedule | `cbbpy`, hoopR, CollegeBasketballData API | free |
| Reference ratings | **Bart Torvik** (public CSV/JSON) | open; primary reference here |
| Reference ratings | KenPom | **paid; ToS forbids scraping.** Use your own export, don't scrape |
| Market lines + closing lines | The Odds API (free tier), Pinnacle screens | snapshot every line for CLV |
| Injuries / availability | public injury feeds, availability reports | drives the usage-availability feature |
| Recruiting / transfers | 247/On3/Verbal Commits, portal trackers | early-season prior only |

The model is **source-agnostic**: any ratings table with
`[team, adj_o, adj_d, tempo]` plugs straight in.

## Hosting and daily refresh (GitHub)

The board is a static page in `docs/` served by GitHub Pages. The
**Refresh board** workflow (`.github/workflows/daily.yml`) reruns the model and
commits a new `docs/data/plays.json`, which the page loads on open.

| When (UTC) | What |
|------------|------|
| 12:15 daily | Refit ratings and rebuild the card |
| 16:15, 19:15, 22:15, 00:15 | Re-price the slate as lines move (live mode only) |

### One-time setup

1. **Pages:** Settings → Pages → Source: *Deploy from a branch* → `main` / `/docs`.
2. **Actions permissions:** Settings → Actions → General → Workflow permissions →
   *Read and write*.
3. **First run:** Actions → Refresh board → *Run workflow*.
4. **Go live:** Settings → Secrets and variables → Actions →
   - Secret `ODDS_API_KEY` = your key from the-odds-api.com
   - Variable `MODEL_MODE` = `live`
   The first live run downloads last season's games and box scores from ESPN
   (several thousand games), so it takes longer than later runs.
5. **Google Sheets (optional):**
   - Variable `SHEET_ID` = the id from your sheet's URL
   - Secret `GOOGLE_CREDS` = the full service-account JSON. Share the sheet with
     that service account's email.

### What live mode keeps in the repo

| File | Contents |
|------|----------|
| `data/games/<season>.csv` | Every completed D-I game with box-score possessions inputs (ESPN) |
| `data/history/odds_snapshots.csv` | Consensus spread/total for each game at every run |
| `data/history/card_log.csv` | Each play the first time it was posted, then its closing line, CLV (points) and W/L/P, filled in automatically |
| `data/unmatched_teams.csv` | Sportsbook team names that didn't match an ESPN team |
| `data/team_aliases.csv` | Your manual fixes: columns `source_name,espn_id` |

Ratings come from our own adjusted-efficiency fit on ESPN box scores. Early in
the season, last season's games are blended in at weight
`0.5 × 12 / (12 + games played)`, so the prior fades as data arrives.
Torvik's site blocks most automated downloads; `live.fetch_torvik` tries it and
skips quietly when blocked. KenPom requires your own subscription export.

Note: GitHub Pages on a **private** repo needs a paid GitHub plan. On a free
plan, a public repo makes the board (and your picks) public.

## Validation discipline

- Every rating and feature for a game on date *D* is computed only from games
  **before** *D* (`backtest.walk_forward` enforces this).
- Success = positive CLV first, then ROI over a large sample. Break-even ATS at
  -110 is **52.38%**. Anything you can't reproduce out-of-sample is overfitting.

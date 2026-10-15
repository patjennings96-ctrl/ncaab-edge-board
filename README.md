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

## Injuries and lineups

Every player gets an offensive and defensive rating (points per 100 possessions
vs an average D-I player, Box Plus-Minus scale) from ESPN box scores, adjusted
so each team's players add up to the team's rating, and pulled toward last
season's rating (or replacement level) until he has played enough minutes.

Before each game the model compares the lineup it expects tonight with the
lineups behind the team's rating:

* **Expected minutes:** recent minutes in games the player played.
* **Will he play?** From, in order of priority: your notes in
  `data/availability_overrides.csv`, saved reports in `data/availability/`,
  ESPN's injury feed, and otherwise his recent absences. A rotation player who
  missed the last game played the next one 49% of the time last season, 34%
  after two missed games and 8.5% after three or more.
* **Impact:** `5 × Σ rating × minutes / 200`, with missing minutes filled at
  replacement level (−2). A bench player out barely moves the line; a
  32-minute starter rated +6 is worth about 4 points.

The adjustment applies at 0.8 weight to margins and 0.5 to totals, the weights
fitted on 2025-26 (`src/lineup_backtest.py`, results in
`docs/data/lineup_backtest_2026.json`). On the second half of that season it
cut margin error from 11.24 to 11.14 points, and when the adjusted projection
disagreed with the closing spread by 3+ points its side covered 53.6% (334
games), vs 48.0% without it. Totals did not improve. Two-thirds of team-games
move less than 1 point; 2% move more than 3.

Adding a note (ESPN's college injury feed is often empty):

```
player_id,name,team_id,status,through
,Caleb Wilson,153,OUT,2026-02-28
```

Use `player_id` or `name` + `team_id` (ESPN ids; see the Players tab).
Status: OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, ACTIVE. Rows past `through` are ignored.

The board shows a "Lineup:" note under any game where a player is expected out.

### X scan (@JonRothstein, @CorbieBO)

Scheduled tasks open these two accounts in your Chrome (Claude in Chrome, your
logged-in X) and record availability news for that day's games in
`data/availability/x_reports.csv`:

```
date,player,team,status,account,url,posted_at,note
```

Pushing that file re-prices the board immediately. Reporter posts count
slightly below official reports (REPORTED OUT = 10% chance to play), and ESPN
reports and your overrides take precedence. Picks, predictions and opinions are
ignored. Each scan appends what it read to `data/history/x_scan_log.csv`.

Schedule (Eastern time, Nov-Apr), from 2024-26 tip times:

| Pass | When | Why |
|------|------|-----|
| First tip | Mon 1:50pm, Tue-Wed 3:50pm, Thu 11:50am, Fri-Sun 10:50am | ~1 hour before that weekday's median first tip |
| Evening | Daily 5:50pm | 1 hour before the 7pm ET wave (most common tip time; 2/3 of games tip 5pm ET or later) |

Your computer must be on with Chrome open and the Claude extension connected;
otherwise that pass is skipped and the model uses ESPN and recent absences.

## Torvik, KenPom and ShotQuality

Torvik blocks requests from cloud servers (GitHub included), KenPom's terms
forbid scraping, and ShotQuality is a paid service, so these are import slots
rather than automatic feeds. Save an export as:

```
data/external/torvik/teams.csv      data/external/torvik/players.csv
data/external/kenpom/teams.csv      data/external/kenpom/players.csv
data/external/shotquality/teams.csv data/external/shotquality/players.csv
```

Team files need a team name plus offense and defense ratings (tempo optional);
player files need name, team and OBPM/DBPM, BPM, or O/D. Common column names
from each site are recognized (`src/external.py`). Team ratings are blended 50/50
with ours and player ratings 50/50 with ours. Files older than 3 days are
ignored. These blends have not been backtested, because historical exports
aren't available.

## Cal's model (second site)

`docs/cal/` is a separate board for Cal's "NCAAB Model" Google Sheet, rebuilt in
`src/calmodel.py` formula for formula: Model 1 (raw efficiency × possessions ×
effective possession ratio), Model 2 (AdjO/AdjD/AdjT), each averaging three
projections, and a play only when both agree by 5+ on a side or 7+ on a total
(Model 2 totals shifted by the day's average). It prices the same slate and lines
as the Edge Board on every run and logs plays to `data/history/cal_log.csv`.

Data substitutions: TeamRankings stats are computed from ESPN box scores with
the same definitions; KenPom comes from `data/external/kenpom/teams.csv` if
fresh, otherwise the Edge Board's adjusted ratings.

Home court: the sheet had none; this version adds 3 points to each model's
margin (+1.5 home, -1.5 road, totals unchanged), none on neutral courts.

`src/cal_backtest.py` replays 2025-26 against DraftKings closes: spreads 362-356
(-29.6u, 718 plays; without home court it was 560-546, -40.6u, 94% road teams),
totals 211-187 (+5.3u).

## Validation discipline

- Every rating and feature for a game on date *D* is computed only from games
  **before** *D* (`backtest.walk_forward` enforces this).
- Success = positive CLV first, then ROI over a large sample. Break-even ATS at
  -110 is **52.38%**. Anything you can't reproduce out-of-sample is overfitting.

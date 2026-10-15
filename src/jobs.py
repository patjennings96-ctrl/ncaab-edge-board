"""
jobs.py
=======
One-off data jobs that need internet access. They run on GitHub Actions via
.github/workflows/job.yml whenever jobs/request.txt changes; the first line of
that file is the command, e.g.

    backfill_players 2025 2026
    probe_torvik 2026

Output logs go to jobs/out/<command>.log and data files are committed.
"""

from __future__ import annotations

import os
import sys

import live

OUT_DIR = os.path.join(live.ROOT, "jobs", "out")

TORVIK_PROBES = [
    "https://barttorvik.com/getadvstats.php?year={y}&csv=1",
    "https://barttorvik.com/getadvstats.php?year={y}&csv=1&start=20260101&end=20260131",
    "https://barttorvik.com/{y}_team_results.csv",
    "https://barttorvik.com/{y}_team_results.json",
    "https://barttorvik.com/trank.php?year={y}&json=1",
    "https://barttorvik.com/trank.php?year={y}&csv=1",
    "https://barttorvik.com/timemachine/team_results/20260115_team_results.json.gz",
    "https://barttorvik.com/{y}_super_sked.json",
    "https://barttorvik.com/getgamestats.php?year={y}&csv=1",
    "https://barttorvik.com/{y}_fffinal.csv",
]


def probe_torvik(year: int) -> None:
    import gzip
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    for tmpl in TORVIK_PROBES:
        url = tmpl.format(y=year)
        try:
            r = live.SESSION.get(url, timeout=40)
            body = r.content
            if url.endswith(".gz") and body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            text = body.decode("utf-8", "replace")
            lines.append(f"=== {url}\nHTTP {r.status_code}  bytes {len(body)}  "
                         f"type {r.headers.get('content-type')}\n{text[:2500]}\n")
        except Exception as exc:
            lines.append(f"=== {url}\nERROR {exc}\n")
    with open(os.path.join(OUT_DIR, "probe_torvik.log"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(l[:400] for l in lines))


def backfill_players(*seasons: int) -> None:
    import players
    for s in seasons:
        df = players.update_player_games(int(s))
        print(f"season {s}: {len(df)} player rows, {df['game_id'].nunique()} games")


def lineup_backtest(season: int = 2026) -> None:
    import sys as _s
    import lineup_backtest as L
    _s.argv = ["lineup_backtest.py", "--season", str(season)]
    L.main()


def start_times(*seasons: int) -> None:
    """Tip-off time (UTC) for every game, from ESPN's daily scoreboards."""
    from datetime import timedelta
    import pandas as pd
    for season in seasons:
        start, end = live.season_bounds(int(season))
        rows, d = [], start
        while d <= end:
            for e in live.espn_scoreboard(d):
                rows.append({"game_id": e["game_id"], "date": d.isoformat(), "start": e["start"],
                             "home_id": e["home_id"], "away_id": e["away_id"]})
            d += timedelta(days=1)
        out = os.path.join(live.ROOT, "data", "games", f"{season}_starts.csv")
        pd.DataFrame(rows).drop_duplicates("game_id").to_csv(out, index=False)
        print(f"season {season}: {len(rows)} games with start times -> {out}")


def probe_url(*urls) -> None:
    """Fetch URLs and save status + the start of each body (for checking access rules)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    for url in urls:
        try:
            r = live.SESSION.get(url, timeout=40)
            lines.append(f"=== {url}\nHTTP {r.status_code} bytes {len(r.content)}\n{r.text[:6000]}\n")
        except Exception as exc:
            lines.append(f"=== {url}\nERROR {exc}\n")
    with open(os.path.join(OUT_DIR, "probe_url.log"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(l[:300] for l in lines))


def main(argv):
    if not argv:
        print("usage: jobs.py <command> [args]")
        return
    cmd, args = argv[0], argv[1:]
    fn = {"probe_torvik": probe_torvik, "backfill_players": backfill_players,
          "lineup_backtest": lineup_backtest, "start_times": start_times,
          "probe_url": probe_url}.get(cmd)
    if fn is None:
        raise SystemExit(f"unknown command {cmd}")
    fn(*[int(a) if a.isdigit() and cmd != "probe_url" else a for a in args])


if __name__ == "__main__":
    main(sys.argv[1:])

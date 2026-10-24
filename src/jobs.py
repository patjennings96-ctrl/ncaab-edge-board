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
            import re as _re
            links = sorted(set(l for l in _re.findall(r'href="([^"]*)"', r.text)
                               if _re.search(r"term|legal|polic|ncaa-basketball/matchup", l)))
            lines.append(f"=== {url}\nHTTP {r.status_code} bytes {len(r.content)}\nLINKS {links[:400]}\n{r.text[:3000]}\n")
        except Exception as exc:
            lines.append(f"=== {url}\nERROR {exc}\n")
    with open(os.path.join(OUT_DIR, "probe_url.log"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(l[:300] for l in lines))


def espn_teams() -> None:
    """ESPN team id, school (location), mascot (name) and display name for every team."""
    import pandas as pd
    d = live._get_json(f"{live.ESPN}/teams", {"limit": "1000"})
    rows = []
    for sp in d.get("sports", []):
        for lg in sp.get("leagues", []):
            for t in lg.get("teams", []):
                t = t.get("team", t)
                rows.append({"team_id": str(t["id"]), "location": t.get("location", ""),
                             "mascot": t.get("name", ""), "display": t.get("displayName", ""),
                             "abbr": t.get("abbreviation", "")})
    out = os.path.join(live.ROOT, "data", "teams.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} teams -> {out}")


def rosters(season: int) -> None:
    """ESPN rosters for every D-I team -> data/rosters/<season>.csv."""
    import pandas as pd
    import preseason as PS
    teams = pd.read_csv(os.path.join(live.ROOT, "data", "teams.csv"), dtype=str)
    g = live.load_games(int(season) - 1)
    d1 = set(g.home_id.astype(str)) | set(g.away_id.astype(str))
    ids = [t for t in teams.team_id if t in d1]
    PS.fetch_rosters(int(season), ids)


def odds_probe() -> None:
    """What The Odds API has for NCAAB right now: events (free) and odds by
    book and date (2 credits). Never prints the key."""
    import collections
    key = os.environ.get("ODDS_API_KEY", "")
    base = "https://api.the-odds-api.com/v4/sports/basketball_ncaab"
    lines = []
    r = live.SESSION.get(f"{base}/events", params={"apiKey": key}, timeout=40)
    ev = r.json() if r.ok else []
    lines.append(f"events: HTTP {r.status_code}, {len(ev)} events, "
                 f"credits left {r.headers.get('x-requests-remaining')}")
    by_day = collections.Counter(e["commence_time"][:10] for e in ev)
    lines.append("events by UTC date: " + ", ".join(f"{d} {n}" for d, n in sorted(by_day.items())))
    for regions in ("us", "us,us2"):
        r = live.SESSION.get(f"{base}/odds", params={"apiKey": key, "regions": regions,
                             "markets": "spreads,totals", "oddsFormat": "american"}, timeout=40)
        od = r.json() if r.ok else []
        books = collections.Counter(b["key"] for e in od for b in e.get("bookmakers", []))
        days = collections.Counter(e["commence_time"][:10] for e in od if e.get("bookmakers"))
        lines.append(f"odds regions={regions}: HTTP {r.status_code}, {len(od)} events, "
                     f"{sum(1 for e in od if e.get('bookmakers'))} with any book, "
                     f"credits left {r.headers.get('x-requests-remaining')}")
        lines.append("  books: " + ", ".join(f"{b} {n}" for b, n in books.most_common()))
        lines.append("  priced events by UTC date: " + ", ".join(f"{d} {n}" for d, n in sorted(days.items())))
        for e in od[:8]:
            lines.append(f"  {e['commence_time']} {e['away_team']} @ {e['home_team']}: "
                         + ", ".join(b["key"] for b in e.get("bookmakers", [])))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "odds_probe.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    if not argv:
        print("usage: jobs.py <command> [args]")
        return
    cmd, args = argv[0], argv[1:]
    fn = {"probe_torvik": probe_torvik, "backfill_players": backfill_players,
          "lineup_backtest": lineup_backtest, "start_times": start_times,
          "probe_url": probe_url, "espn_teams": espn_teams, "rosters": rosters,
          "odds_probe": odds_probe}.get(cmd)
    if fn is None:
        raise SystemExit(f"unknown command {cmd}")
    fn(*[int(a) if a.isdigit() and cmd != "probe_url" else a for a in args])


if __name__ == "__main__":
    main(sys.argv[1:])

"""
similar.py
==========
"How have plays like this one done?" for every posted play, both models.

A play is described by a few traits that mattered in the backtests:

  Edge Board spread  phase, gap bucket, underdog or favorite, spread size
  Edge Board total   phase, gap bucket, over or under
  Cal spread         phase, weaker model's gap bucket, underdog or favorite, spread size
  Cal total          phase, weaker model's gap bucket, over or under

phase   Nov-Dec ("non-conference"), Jan 1 - Mar 8 ("conference"), later ("March")
size    under 5, 5-15, 15+ points

The pool is last season's walk-forward backtest (graded at the close) plus
every graded play this season. Matching starts with all the traits; if fewer
than MIN_N similar plays exist it drops the least important trait (side, then
size) until there are enough. The result carries the record, win %, units at
-110 and a description of what was matched, so the site can show it as is.

`score` is the win % shrunk toward 50% by SHRINK plays, so a 12-3 bucket does
not outrank a 300-250 one. It is what the "best by similar plays" list sorts by.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import live

HIST = os.path.join(live.ROOT, "data", "history")
MIN_N = 40
SHRINK = 50

EDGE_GAPS = {"SPREAD": [2, 3, 4, 5, 99], "TOTAL": [3, 4, 5, 7, 99]}
CAL_GAPS = {"ATS": [5, 7, 10, 99], "TOTAL": [7, 9, 12, 99]}
PHASE_LABEL = {"early": "Nov-Dec", "conf": "Jan-Mar 8", "late": "after Mar 8"}


def phase(ts) -> str:
    d = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(d):
        return "early"
    d = d.tz_convert("America/New_York")
    if d.month in (10, 11, 12):
        return "early"
    if d.month in (1, 2) or (d.month == 3 and d.day <= 8):
        return "conf"
    return "late"


def _bucket(gap: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= gap < hi:
            return f"{lo}-{hi}" if hi < 99 else f"{lo}+"
    return f"<{edges[0]}"


def _size(line: float) -> str:
    a = abs(line)
    return "under 5" if a < 5 else ("5-15" if a < 15 else "15+")


# -----------------------------------------------------------------------------
# The pool of past plays
# -----------------------------------------------------------------------------
def _edge_backtest() -> pd.DataFrame:
    import model_compare
    try:
        d = model_compare.load(2026)
    except Exception as exc:                       # backtest files missing
        print(f"[similar] edge backtest unavailable: {exc}")
        return pd.DataFrame()
    out = []
    dm = d.edge_tempo_lineup_m - d.close_m
    s = d[dm.abs() >= EDGE_GAPS["SPREAD"][0]]
    dm = dm[s.index]
    out.append(pd.DataFrame({
        "model": "edge", "mkt": "SPREAD", "start": s.date.astype(str),
        "gap": dm.abs(), "side": np.where(np.sign(dm) != np.sign(s.close_m), "dog", "fav"),
        "size": s.close_m.abs().map(_size),
        "hit": np.sign(s.margin - s.close_m) * np.sign(dm)}))
    dt = d.edge_tempo_lineup_t - d.close_t
    t = d[dt.abs() >= EDGE_GAPS["TOTAL"][0]]
    dt = dt[t.index]
    out.append(pd.DataFrame({
        "model": "edge", "mkt": "TOTAL", "start": t.date.astype(str),
        "gap": dt.abs(), "side": np.where(dt > 0, "over", "under"), "size": "",
        "hit": np.sign(t.total - t.close_t) * np.sign(dt)}))
    return pd.concat(out, ignore_index=True)


def _cal_backtest() -> pd.DataFrame:
    p = os.path.join(HIST, "cal_backtest_rows_2026.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p).dropna(subset=["home_score", "away_score"])
    margin = d.home_score - d.away_score
    out = []
    a = d[d.ats_play.isin(["HOME", "AWAY"])]
    home = a.ats_play == "HOME"
    pick_line = np.where(home, a.line, -a.line)
    x = np.where(home, margin[a.index], -margin[a.index]) + pick_line
    out.append(pd.DataFrame({
        "model": "cal", "mkt": "ATS", "start": a.start.astype(str),
        "gap": np.minimum(a.m1_ats.abs(), a.m2_ats.abs()),
        "side": np.where(pick_line > 0, "dog", "fav"), "size": a.line.abs().map(_size),
        "hit": np.sign(x)}))
    t = d[d.ou_play.isin(["OVER", "UNDER"])]
    tot = (t.home_score + t.away_score) - t.total_line
    out.append(pd.DataFrame({
        "model": "cal", "mkt": "TOTAL", "start": t.start.astype(str),
        "gap": np.minimum(t.m1_tot.abs(), t.m2_tot_adj.abs()),
        "side": np.where(t.ou_play == "OVER", "over", "under"), "size": "",
        "hit": np.sign(tot) * np.where(t.ou_play == "OVER", 1, -1)}))
    return pd.concat(out, ignore_index=True)


def _edge_live() -> pd.DataFrame:
    p = os.path.join(HIST, "card_log.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    import publish
    d = publish.drop_void(pd.read_csv(p, dtype={"game_id": str}), "edge")
    d = d[d.result.isin(["W", "L", "P"])].copy()
    d["gap"] = pd.to_numeric(d.gap, errors="coerce")
    d = d.dropna(subset=["gap"])
    sp = d.mkt == "SPREAD"
    return pd.DataFrame({
        "model": "edge", "mkt": d.mkt, "start": d.start.astype(str), "gap": d.gap,
        "side": np.where(sp, np.where(d.line > 0, "dog", "fav"),
                         np.where(d.side_code == "OVER", "over", "under")),
        "size": np.where(sp, d.line.abs().map(_size), ""),
        "hit": d.result.map({"W": 1, "L": -1, "P": 0})})


def _cal_live() -> pd.DataFrame:
    p = os.path.join(HIST, "cal_log.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    import publish
    d = publish.drop_void(pd.read_csv(p, dtype={"game_id": str, "units": str}), "cal")
    d["u"] = pd.to_numeric(d.units, errors="coerce")
    d["gap"] = pd.to_numeric(d.gap, errors="coerce")
    d = d.dropna(subset=["u", "gap"])
    ats = d.market == "ATS"
    pick_line = np.where(d.play == "HOME", d.line, -d.line)
    return pd.DataFrame({
        "model": "cal", "mkt": d.market, "start": d.start.astype(str), "gap": d.gap.abs(),
        "side": np.where(ats, np.where(pick_line > 0, "dog", "fav"),
                         np.where(d.play == "OVER", "over", "under")),
        "size": np.where(ats, d.line.abs().map(_size), ""),
        "hit": np.sign(d.u)})


def pool() -> pd.DataFrame:
    parts = [f() for f in (_edge_backtest, _cal_backtest, _edge_live, _cal_live)]
    p = pd.concat([x for x in parts if not x.empty], ignore_index=True)
    p["phase"] = p.start.map(phase)
    gaps = {("edge", m): e for m, e in EDGE_GAPS.items()} | {("cal", m): e for m, e in CAL_GAPS.items()}
    p["gbin"] = [_bucket(g, gaps[(mo, mk)]) for g, mo, mk in zip(p.gap, p.model, p.mkt)]
    return p


# -----------------------------------------------------------------------------
# Lookup
# -----------------------------------------------------------------------------
def lookup(P: pd.DataFrame, model: str, mkt: str, start, gap: float, side: str, size: str = "") -> dict:
    gaps = (EDGE_GAPS if model == "edge" else CAL_GAPS)[mkt]
    key = {"phase": phase(start), "gbin": _bucket(abs(gap), gaps), "side": side, "size": size}
    base = P[(P.model == model) & (P.mkt == mkt)]
    order = ["phase", "gbin", "size", "side"] if size else ["phase", "gbin", "side"]
    while True:
        m = base
        for k in order:
            m = m[m[k] == key[k]]
        if len(m) >= MIN_N or len(order) <= 2:
            break
        order = order[:-1]
    w, l = int((m.hit > 0).sum()), int((m.hit < 0).sum())
    pct = w / max(w + l, 1)
    words = {"phase": PHASE_LABEL[key["phase"]], "gbin": f"{key['gbin']} pt gap",
             "side": {"dog": "underdogs", "fav": "favorites", "over": "overs", "under": "unders"}[side],
             "size": f"{size} spreads"}
    return {"w": w, "l": l, "pct": round(pct, 3), "units": round(w - 1.1 * l, 1),
            "score": round((w + SHRINK * 0.5) / (w + l + SHRINK), 4),
            "match": ", ".join(words[k] for k in order)}


def attach(edge_plays: list[dict], cal_games: list[dict], P: pd.DataFrame | None = None) -> None:
    """Add `similar` to each Edge play and `ats_similar` / `ou_similar` to Cal's games, in place."""
    P = pool() if P is None else P
    for r in edge_plays:
        if r.get("gap") is None:
            continue
        if r["mkt"] == "SPREAD":
            r["similar"] = lookup(P, "edge", "SPREAD", r["start"], r["gap"],
                                  "dog" if r["line"] > 0 else "fav", _size(r["line"]))
        else:
            r["similar"] = lookup(P, "edge", "TOTAL", r["start"], r["gap"],
                                  "over" if r["side_code"] == "OVER" else "under")
    for g in cal_games:
        m1, m2 = g.get("m1", {}), g.get("m2", {})
        if g.get("ats_play") in ("HOME", "AWAY"):
            pl = g["line"] if g["ats_play"] == "HOME" else -g["line"]
            g["ats_similar"] = lookup(P, "cal", "ATS", g["start"], min(abs(m1["ats"]), abs(m2["ats"])),
                                      "dog" if pl > 0 else "fav", _size(g["line"]))
        if g.get("ou_play") in ("OVER", "UNDER"):
            t2 = m2.get("tot_adj", m2.get("tot"))
            g["ou_similar"] = lookup(P, "cal", "TOTAL", g["start"], min(abs(m1["tot"]), abs(t2)),
                                     "over" if g["ou_play"] == "OVER" else "under")

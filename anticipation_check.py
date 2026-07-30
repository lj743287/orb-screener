#!/usr/bin/env python3
"""
anticipation_check.py — Stage B (intraday)

Reads anticipation_watch.json (coiled pre-breakout setups from
anticipation_scan.py) and checks each one for the moment Stockbee actually
buys: the day the range expands out of the quiet base.

    BREAKOUT  price above the consolidation pivot, today's range already
              wider than the recent average, projected volume above
              yesterday's, and trading in the top 40% of today's range.
              This is day one of the momentum burst.

    EARLY     price above the pivot but volume/range not yet confirming,
              OR pressing right under the pivot (within 0.5%) on elevated
              volume pace. Watch into the close.

Because the base day was QUIET by construction, "volume above yesterday" is
the correct Stockbee test here — unlike a day-2 test against a burst day.

Every row carries the day's low (the Stockbee stop) and the resulting risk %,
so position size drops straight out of it.

Writes setups.html (GitHub Pages report), setups.json and setups.txt
(TradingView watchlist import). Requires env var TWELVE_DATA_API_KEY.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
QUOTE_URL = "https://api.twelvedata.com/quote"

BATCH = 40
BATCH_DELAY = 50
MIN_SESSION_FRAC = 0.15
NEAR_PIVOT_PCT = 0.5      # "pressing the pivot" tolerance
CLOSE_POS_MAX = 0.40      # must trade in top 40% of today's range
RANGE_EXP_MULT = 1.0      # today's range% vs recent avg range%

WATCH_FILE = "anticipation_watch.json"
OUT_HTML = "setups.html"
OUT_JSON = "setups.json"
OUT_TXT = "setups.txt"


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "sb-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def session_fraction():
    """Fraction of the US regular session elapsed (ET 09:30-16:00)."""
    utc = datetime.now(timezone.utc)
    offset = -4 if 3 <= utc.month <= 10 else -5
    et = utc + timedelta(hours=offset)
    open_t = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = et.replace(hour=16, minute=0, second=0, microsecond=0)
    if et <= open_t:
        return 0.0
    if et >= close_t:
        return 1.0
    return (et - open_t).total_seconds() / (close_t - open_t).total_seconds()


def fetch_quotes(symbols):
    quotes = {}
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        url = f"{QUOTE_URL}?symbol={','.join(chunk)}&apikey={API_KEY}"
        try:
            data = json.loads(http_get(url))
        except Exception:
            data = {}
        if len(chunk) == 1:
            data = {chunk[0]: data}
        for sym in chunk:
            q = data.get(sym)
            if isinstance(q, dict) and "close" in q:
                quotes[sym] = q
        if i + BATCH < len(symbols):
            time.sleep(BATCH_DELAY)
    return quotes


def evaluate(stock, q, frac):
    try:
        price = float(q["close"])
        hi = float(q["high"])
        lo = float(q["low"])
        vol = float(q.get("volume") or 0)
    except (KeyError, ValueError, TypeError):
        return None

    prev_close = stock["close"]
    pivot = stock["pivot"]
    prev_vol = stock["prev_volume"]
    avg_rng = stock.get("avg_range_10") or stock.get("adr_pct") or 1.0

    pct = 100.0 * (price - prev_close) / prev_close if prev_close else 0.0
    proj_vol = vol / max(frac, MIN_SESSION_FRAC)
    rng = hi - lo
    rng_pct = 100.0 * rng / price if price else 0.0
    close_pos = (hi - price) / rng if rng > 0 else 0.0
    dist = 100.0 * (price - pivot) / pivot if pivot else 0.0

    above = price > pivot
    near = -NEAR_PIVOT_PCT <= dist <= 0.0
    vol_ok = proj_vol > prev_vol
    rng_ok = rng_pct >= RANGE_EXP_MULT * avg_rng
    pos_ok = close_pos <= CLOSE_POS_MAX

    if above and vol_ok and rng_ok and pos_ok:
        status = "BREAKOUT"
    elif above or (near and vol_ok):
        status = "EARLY"
    else:
        return None

    risk_pct = 100.0 * (price - lo) / price if price else 0.0
    return {
        "symbol": stock["symbol"], "exchange": stock.get("exchange", ""),
        "status": status, "score": stock["score"],
        "price": round(price, 2), "pct": round(pct, 2),
        "pivot": pivot, "dist": round(dist, 2),
        "stop": round(lo, 2), "risk_pct": round(risk_pct, 2),
        "adr_pct": stock.get("adr_pct", 0), "consol_days": stock["consol_days"],
        "quals": stock.get("quals", ""),
        "proj_vol": round(proj_vol), "prev_vol": round(prev_vol),
        "vol_ok": vol_ok, "rng_ok": rng_ok, "pos_ok": pos_ok,
    }


def write_html(rows, frac, watched, found):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tr = ""
    for r in rows:
        badge = (f"<span class='bo'>BREAKOUT</span>" if r["status"] == "BREAKOUT"
                 else "<span class='early'>EARLY</span>")
        checks = (f"vol {'✓' if r['vol_ok'] else '✗'} · "
                  f"range {'✓' if r['rng_ok'] else '✗'} · "
                  f"pos {'✓' if r['pos_ok'] else '✗'}")
        tr += (f"<tr><td class='sym'>{r['symbol']}</td>"
               f"<td>{badge}</td><td class='sc'>{r['score']:.0f}</td>"
               f"<td>${r['price']:.2f}</td><td class='big'>{r['pct']:+.2f}%</td>"
               f"<td>${r['pivot']:.2f}</td><td>{r['dist']:+.2f}%</td>"
               f"<td class='stop'>${r['stop']:.2f}</td>"
               f"<td class='risk'>{r['risk_pct']:.1f}%</td>"
               f"<td>{r['adr_pct']:.1f}%</td>"
               f"<td>{r['consol_days']}d · {r['quals']}</td>"
               f"<td class='chk'>{checks}</td></tr>\n")
    if not tr:
        tr = ("<tr><td colspan='12' class='none'>No setups triggering yet — "
              "the watch list is coiled but quiet.</td></tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stockbee Anticipation Setups</title><style>
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:16px;line-height:1.6}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #21262d;font-size:14px;white-space:nowrap}}
th{{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase}}
.sym{{font-weight:700;font-size:15px}} .big{{font-weight:700;color:#3fb950}}
.sc{{font-weight:700;color:#d29922}} .stop{{color:#f85149}} .risk{{color:#f0883e}}
.bo{{background:#238636;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700}}
.early{{background:#9e6a03;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700}}
.chk{{color:#8b949e;font-size:12px}} .none{{color:#8b949e;padding:24px}}
a{{color:#58a6ff}}
</style></head><body>
<h1>Stockbee Anticipation Setups</h1>
<div class="meta">Generated {ts} · session {frac*100:.0f}% elapsed ·
{found} setups found overnight, {watched} watched today ·
<a href="setups.txt" download>Download TradingView watchlist (.txt)</a><br>
BREAKOUT = above pivot with range expansion, volume pace and strong close ·
EARLY = above pivot unconfirmed, or pressing the pivot on volume ·
STOP = today's low (Stockbee stop) · RISK = distance from price to that stop</div>
<table><tr><th>Symbol</th><th>Status</th><th>Score</th><th>Price</th><th>Today</th>
<th>Pivot</th><th>vs Pivot</th><th>Stop</th><th>Risk</th><th>ADR</th>
<th>Base</th><th>Checks</th></tr>
{tr}</table></body></html>"""
    with open(OUT_HTML, "w") as f:
        f.write(html)


def main():
    if not API_KEY:
        sys.exit("TWELVE_DATA_API_KEY not set")
    if not os.path.exists(WATCH_FILE):
        sys.exit(f"{WATCH_FILE} missing — run anticipation_scan.py first")
    with open(WATCH_FILE) as f:
        watch = json.load(f)
    stocks = watch.get("stocks", [])
    found = watch.get("found", len(stocks))
    frac = session_fraction()
    print(f"Watching {len(stocks)} coiled setups, session {frac*100:.0f}% elapsed")
    if frac <= 0.0:
        print("Market not open yet — exiting")
        return
    quotes = fetch_quotes([s["symbol"] for s in stocks])
    rows = []
    for s in stocks:
        q = quotes.get(s["symbol"])
        if q:
            r = evaluate(s, q, frac)
            if r:
                rows.append(r)
    rows.sort(key=lambda x: (x["status"] != "BREAKOUT", -x["score"]))
    write_html(rows, frac, len(stocks), found)
    tv = [f"{r['exchange']}:{r['symbol']}" if r["exchange"] else r["symbol"]
          for r in rows]
    with open(OUT_TXT, "w") as f:
        f.write(",".join(tv) + ("\n" if tv else ""))
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "session_fraction": round(frac, 3), "results": rows}, f, indent=1)
    print(f"Wrote {OUT_HTML} / {OUT_JSON} / {OUT_TXT}: "
          f"{sum(1 for r in rows if r['status']=='BREAKOUT')} BREAKOUT, "
          f"{sum(1 for r in rows if r['status']=='EARLY')} EARLY")


if __name__ == "__main__":
    main()

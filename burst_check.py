#!/usr/bin/env python3
"""
burst_check.py — Stage B (intraday, ~11:30 ET)
Reads burst_watch.json (yesterday's burst stocks from burst_scan.py), pulls
live quotes for just those names, and tests for a SECOND burst in progress:
    price >= 4% above yesterday's close
    volume, projected to full-day pace, >= 60% of the burst day's volume
    trading in top 30% of today's range so far
FULL = all three pass.  EARLY = price test only (volume/range pending).
Writes bursts.html (styled report served by GitHub Pages), bursts.json, and
bursts.txt — a TradingView watchlist import file (EXCHANGE:SYMBOL, comma-
separated). Falls back to bare symbols if the watch file predates exchange
tagging.

Requires env var TWELVE_DATA_API_KEY. Quotes are batched 40 symbols per call,
paced within the Grow 55 credit budget.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
QUOTE_URL = "https://api.twelvedata.com/quote"

PCT_THRESHOLD = 4.0
RANGE_TOP = 0.30
DAY2_VOL_RATIO = 0.60   # day-2 projected volume must be >= this fraction of
                        # the burst day's volume (day 2 of a real sequence
                        # rarely out-trades day 1, so 1.0 is miscalibrated)
BATCH = 40
BATCH_DELAY = 50          # seconds between batches (40 credits per batch)
MIN_SESSION_FRAC = 0.15   # clamp for volume projection

WATCH_FILE = "burst_watch.json"
OUT_HTML = "bursts.html"
OUT_JSON = "bursts.json"
OUT_TXT = "bursts.txt"


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "burst-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def session_fraction():
    """Fraction of the US regular session elapsed (ET 09:30–16:00)."""
    utc = datetime.now(timezone.utc)
    # US Eastern offset: -4 (EDT, roughly Mar–Nov) else -5. Good enough for
    # a scheduler that runs mid-session on trading days.
    edt_start = utc.replace(month=3, day=8)
    est_start = utc.replace(month=11, day=1)
    offset = -4 if edt_start <= utc.replace(tzinfo=timezone.utc) < est_start else -5
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
        o = float(q["open"])
        h = float(q["high"])
        l = float(q["low"])
        vol = float(q.get("volume") or 0)
    except (KeyError, ValueError, TypeError):
        return None
    y_close, y_vol = stock["y_close"], stock["y_volume"]
    if y_close <= 0:
        return None
    pct = 100.0 * (price - y_close) / y_close
    proj_vol = vol / max(frac, MIN_SESSION_FRAC)
    rng = h - l
    top_pos = (h - price) / rng if rng > 0 else 0.0
    pct_ok = pct >= PCT_THRESHOLD
    vol_ok = proj_vol >= DAY2_VOL_RATIO * y_vol
    rng_ok = top_pos <= RANGE_TOP
    status = "FULL" if (pct_ok and vol_ok and rng_ok) else ("EARLY" if pct_ok else None)
    if not status:
        return None
    return {"symbol": stock["symbol"],
            "exchange": stock.get("exchange", ""), "status": status,
            "y_pct": stock["y_pct"], "today_pct": round(pct, 2),
            "price": price, "proj_vol": round(proj_vol),
            "y_volume": round(y_vol), "vol_ok": vol_ok, "rng_ok": rng_ok}


def write_html(rows, frac, watch):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tr = ""
    for r in rows:
        badge = ("<span class='full'>FULL</span>" if r["status"] == "FULL"
                 else "<span class='early'>EARLY</span>")
        checks = f"vol {'✓' if r['vol_ok'] else '✗'} · range {'✓' if r['rng_ok'] else '✗'}"
        tr += (f"<tr><td class='sym'>{r['symbol']}</td><td>{badge}</td>"
               f"<td>+{r['y_pct']}%</td><td class='big'>+{r['today_pct']}%</td>"
               f"<td>${r['price']:.2f}</td>"
               f"<td>{r['proj_vol']:,} vs {r['y_volume']:,}</td>"
               f"<td class='chk'>{checks}</td></tr>\n")
    if not tr:
        tr = "<tr><td colspan='7' class='none'>No second-burst candidates right now.</td></tr>"
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Second Burst Watch</title><style>
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;max-width:900px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d;font-size:14px}}
th{{color:#8b949e;font-weight:600;font-size:12px;text-transform:uppercase}}
.sym{{font-weight:700;font-size:15px}} .big{{font-weight:700;color:#3fb950}}
.full{{background:#238636;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700}}
.early{{background:#9e6a03;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700}}
.chk{{color:#8b949e;font-size:12px}} .none{{color:#8b949e;padding:24px}}
</style></head><body>
<h1>Second Burst Watch</h1>
<div class="meta">Generated {ts} · session {frac*100:.0f}% elapsed ·
watching {watch} stocks that burst yesterday · FULL = price + volume pace +
range all confirm · EARLY = price only, volume pending ·
<a href="bursts.txt" download style="color:#58a6ff">Download TradingView
watchlist (.txt)</a></div>
<table><tr><th>Symbol</th><th>Status</th><th>Yesterday</th><th>Today</th>
<th>Price</th><th>Proj vol vs yesterday</th><th>Checks</th></tr>
{tr}</table></body></html>"""
    with open(OUT_HTML, "w") as f:
        f.write(html)


def main():
    if not API_KEY:
        sys.exit("TWELVE_DATA_API_KEY not set")
    if not os.path.exists(WATCH_FILE):
        sys.exit(f"{WATCH_FILE} missing — run burst_scan.py first")
    with open(WATCH_FILE) as f:
        watch = json.load(f)
    stocks = watch.get("stocks", [])
    frac = session_fraction()
    print(f"Watching {len(stocks)} stocks, session {frac*100:.0f}% elapsed")
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
    rows.sort(key=lambda x: (x["status"] != "FULL", -x["today_pct"]))
    write_html(rows, frac, len(stocks))
    tv_syms = [f"{r['exchange']}:{r['symbol']}" if r["exchange"] else r["symbol"]
               for r in rows]
    with open(OUT_TXT, "w") as f:
        f.write(",".join(tv_syms) + ("\n" if tv_syms else ""))
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "session_fraction": round(frac, 3), "results": rows}, f, indent=1)
    print(f"Wrote {OUT_HTML} / {OUT_JSON} / {OUT_TXT}: "
          f"{sum(1 for r in rows if r['status']=='FULL')} FULL, "
          f"{sum(1 for r in rows if r['status']=='EARLY')} EARLY")


if __name__ == "__main__":
    main()

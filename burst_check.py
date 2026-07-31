#!/usr/bin/env python3
"""
burst_check.py — Stage B (intraday, ~1 hour after the US open)

Reads burst_watch.json (yesterday's burst stocks from burst_scan.py), pulls
live quotes for just those names, and tests for a SECOND burst in progress:
    price >= 4% above yesterday's close
    volume, projected to full-day pace, >= 60% of the burst day's volume
    trading in top 30% of today's range so far
FULL = all three pass.  EARLY = price test only (volume/range pending).

Writes bursts.html (its own standalone report), bursts.json, bursts.txt, and
data/burst_check.json — the "Second burst" tab on the combined dashboard.

Requires env var TWELVE_DATA_API_KEY. Quotes are batched 40 symbols per call,
paced within the Grow 55 credit budget.

Scheduling: run this at 14:30 and 15:30 UTC on weekdays. One of those is an
hour after the US open and one is not, depending on the time of year; the
script works out which and skips the run that is too early. It also skips
once the session has closed.

Outputs (unchanged): bursts.html, bursts.json, bursts.txt
Outputs (new):       data/burst_check.json  -- shared format for the dashboard
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta, date

# NEW: shared output writer, so this scan can appear on the combined page.
from scan_output import write_scan, write_failure

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

# NEW: only report on a session that is properly under way. Below this the
# volume projection is multiplying a handful of opening trades by six or more
# and cannot be trusted; at or above 1.0 the session has closed.
MIN_RUN_FRAC = 0.12       # roughly 45 minutes into the 6.5 hour session

WATCH_FILE = "burst_watch.json"
OUT_HTML = "bursts.html"
OUT_JSON = "bursts.json"
OUT_TXT = "bursts.txt"

# NEW: identity of this scan on the combined dashboard.
SCAN_ID = "burst_check"
SCAN_LABEL = "Second burst"


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "burst-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _nth_weekday(year, month, weekday, n):
    """Date of the nth given weekday in a month. weekday: Monday=0, Sunday=6."""
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + timedelta(days=shift + 7 * (n - 1))


def us_eastern_offset(utc_dt):
    """NEW: correct US Eastern UTC offset for a given moment.

    US daylight saving runs from 02:00 local on the second Sunday in March to
    02:00 local on the first Sunday in November. The previous version assumed
    fixed dates of 8 March and 1 November, which is wrong by up to a week
    twice a year -- and while it is wrong the session-elapsed figure is out by
    a full hour, which throws the volume projection out by roughly 15%.
    """
    year = utc_dt.year
    # 02:00 local on the changeover days, expressed in UTC.
    dst_start = datetime.combine(
        _nth_weekday(year, 3, 6, 2), datetime.min.time(),
        tzinfo=timezone.utc) + timedelta(hours=7)    # 02:00 EST = 07:00 UTC
    dst_end = datetime.combine(
        _nth_weekday(year, 11, 6, 1), datetime.min.time(),
        tzinfo=timezone.utc) + timedelta(hours=6)    # 02:00 EDT = 06:00 UTC
    return -4 if dst_start <= utc_dt < dst_end else -5


def session_fraction():
    """Fraction of the US regular session elapsed (ET 09:30-16:00)."""
    utc = datetime.now(timezone.utc)
    et = utc + timedelta(hours=us_eastern_offset(utc))
    if et.weekday() >= 5:
        return 0.0
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


def to_scan_rows(rows):
    """NEW: translate results into the shared dashboard format."""
    out = []
    for r in rows:
        out.append({
            "symbol": r.get("symbol", ""),
            "exchange": r.get("exchange", ""),
            "close": r.get("price"),
            "status": r.get("status", ""),
            "today_pct": r.get("today_pct"),
            "y_pct": r.get("y_pct"),
            "proj_volM": round((r.get("proj_vol") or 0) / 1e6, 2),
            "burst_volM": round((r.get("y_volume") or 0) / 1e6, 2),
            "volume_ok": "yes" if r.get("vol_ok") else "no",
            "range_ok": "yes" if r.get("rng_ok") else "no",
        })
    return out


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

    # NEW: only run on a session that is properly under way, and never after
    # the close. Two scheduled times cover the US clock change; whichever one
    # lands too early simply stops here and leaves the previous result alone.
    if frac <= 0.0:
        print("Market not open (or weekend) — leaving previous result in place")
        return
    if frac >= 1.0:
        print("Session has closed — leaving previous result in place")
        return
    if frac < MIN_RUN_FRAC:
        print(f"Only {frac*100:.0f}% into the session (need "
              f"{MIN_RUN_FRAC*100:.0f}%) — too early to project volume, "
              f"leaving previous result in place")
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

    n_full = sum(1 for r in rows if r["status"] == "FULL")
    n_early = sum(1 for r in rows if r["status"] == "EARLY")

    # --- NEW: shared output for the combined dashboard ---------------------
    write_scan(
        SCAN_ID,
        to_scan_rows(rows),
        label=SCAN_LABEL,
        meta={
            "sort": "FULL first, then biggest move",
            "watching": len(stocks),
            "session_elapsed": f"{frac*100:.0f}%",
            "full": n_full,
            "early": n_early,
            "burst_date": watch.get("burst_date", ""),
        },
    )

    print(f"Wrote {OUT_HTML} / {OUT_JSON} / {OUT_TXT}: {n_full} FULL, {n_early} EARLY")


if __name__ == "__main__":
    # NEW: if the check falls over, leave a note so the dashboard can show a
    # red banner on this tab. The error is re-raised so the GitHub Actions run
    # still shows as failed.
    try:
        main()
    except Exception as exc:
        try:
            write_failure(SCAN_ID, exc, label=SCAN_LABEL)
        except Exception:
            pass
        raise

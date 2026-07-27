#!/usr/bin/env python3
"""
burst_scan.py — Stage A (overnight batch)
Scans the full NASDAQ/NYSE common-stock universe for stocks whose most recent
COMPLETED daily bar was a Stockbee burst day:
    close >= 4% above prior close
    volume > prior day volume
    close in top 30% of day's range
    close >= $3, volume >= 100k (junk/liquidity floor)
Writes burst_watch.json for burst_check.py (Stage B) to test intraday.

Designed for GitHub Actions (cron, pre-US-open). Requires env var
TWELVE_DATA_API_KEY. Paced to ~50 requests/min for the Grow 55 plan.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
TD_URL = "https://api.twelvedata.com/time_series"
UNIVERSE_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]

PCT_THRESHOLD = 4.0        # min % gain vs prior close
RANGE_TOP = 0.30           # close within top 30% of range
MIN_PRICE = 3.0
MIN_VOLUME = 100_000
REQUESTS_PER_MIN = 50
BAD_NAME_WORDS = ("WARRANT", "RIGHT", "UNIT", "PREFERRED", "NOTES", "DEBENTURE")
BAD_SYMBOL_CHARS = set(".$^~=")

OUT_FILE = "burst_watch.json"


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "burst-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def load_universe():
    symbols = []
    for url in UNIVERSE_URLS:
        is_nasdaq_file = "nasdaqlisted" in url
        text = http_get(url)
        lines = [l for l in text.splitlines() if "|" in l]
        header = lines[0].split("|")
        idx = {name: i for i, name in enumerate(header)}
        sym_col = "Symbol" if "Symbol" in idx else "ACT Symbol"
        for line in lines[1:]:
            f = line.split("|")
            if len(f) < len(header) or f[0].startswith("File Creation"):
                continue
            sym = f[idx[sym_col]].strip()
            name = f[idx["Security Name"]].upper() if "Security Name" in idx else ""
            etf = f[idx["ETF"]].strip() if "ETF" in idx else "N"
            test = f[idx["Test Issue"]].strip() if "Test Issue" in idx else "N"
            exch = f[idx["Exchange"]].strip() if "Exchange" in idx else "Q"
            if not sym or etf == "Y" or test == "Y":
                continue
            if exch not in ("Q", "N", "A", ""):  # NASDAQ file has no Exchange col
                continue
            if any(c in BAD_SYMBOL_CHARS for c in sym):
                continue
            if any(w in name for w in BAD_NAME_WORDS):
                continue
            if is_nasdaq_file:
                tv_exch = "NASDAQ"
            else:
                tv_exch = "NYSE" if exch == "N" else "AMEX"
            symbols.append((sym, tv_exch))
    # de-dup keeping first exchange seen
    seen, out = set(), []
    for sym, ex in sorted(symbols):
        if sym not in seen:
            seen.add(sym)
            out.append((sym, ex))
    return out


def us_market_today_incomplete():
    """Return today's US-Eastern date string if the regular session has not
    yet completed (bar still in progress), else None."""
    utc = datetime.now(timezone.utc)
    offset = -4 if 3 <= utc.month <= 10 else -5  # approx EDT/EST
    et = utc + timedelta(hours=offset)
    if et.weekday() >= 5:
        return None
    # Bar considered complete a few minutes after the 16:00 ET close
    if et.hour < 16 or (et.hour == 16 and et.minute < 10):
        return et.strftime("%Y-%m-%d")
    return None


INCOMPLETE_TODAY = None  # set in main()


def fetch_last_bars(symbol):
    url = (f"{TD_URL}?symbol={symbol}&interval=1day&outputsize=3"
           f"&apikey={API_KEY}")
    try:
        data = json.loads(http_get(url))
    except Exception:
        return None
    vals = data.get("values")
    if not vals:
        return None
    # Drop the in-progress bar if the scan is running mid-session, so the
    # burst test always applies to the last COMPLETED trading day.
    if INCOMPLETE_TODAY and vals[0].get("datetime", "")[:10] == INCOMPLETE_TODAY:
        vals = vals[1:]
    if len(vals) < 2:
        return None
    return vals  # newest first


def is_burst(bar, prior):
    try:
        c, o = float(bar["close"]), float(bar["open"])
        h, l = float(bar["high"]), float(bar["low"])
        v = float(bar["volume"])
        pc, pv = float(prior["close"]), float(prior["volume"])
    except (KeyError, ValueError, TypeError):
        return None
    if c < MIN_PRICE or v < MIN_VOLUME or pc <= 0:
        return None
    pct = 100.0 * (c - pc) / pc
    rng = h - l
    top_pos = (h - c) / rng if rng > 0 else 0.0
    if pct >= PCT_THRESHOLD and v > pv and top_pos <= RANGE_TOP:
        return {"y_pct": round(pct, 2), "y_close": c, "y_volume": v,
                "y_dollar_vol": round(c * v)}
    return None


def main():
    if not API_KEY:
        sys.exit("TWELVE_DATA_API_KEY not set")
    global INCOMPLETE_TODAY
    INCOMPLETE_TODAY = us_market_today_incomplete()
    if INCOMPLETE_TODAY:
        print(f"US session in progress — excluding {INCOMPLETE_TODAY} bar, "
              f"testing last completed day")
    universe = load_universe()
    print(f"Universe: {len(universe)} symbols")
    delay = 60.0 / REQUESTS_PER_MIN
    hits, scanned = [], 0
    for sym, exch in universe:
        bars = fetch_last_bars(sym)
        scanned += 1
        if bars:
            res = is_burst(bars[0], bars[1])
            if res:
                res["symbol"] = sym
                res["exchange"] = exch
                res["date"] = bars[0].get("datetime", "")
                hits.append(res)
                print(f"BURST {sym}  +{res['y_pct']}%")
        if scanned % 500 == 0:
            print(f"...{scanned}/{len(universe)} scanned, {len(hits)} bursts")
        time.sleep(delay)
    hits.sort(key=lambda x: -x["y_pct"])
    out = {"generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "burst_date": hits[0]["date"] if hits else "",
           "count": len(hits), "stocks": hits}
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {OUT_FILE}: {len(hits)} burst stocks")


if __name__ == "__main__":
    main()

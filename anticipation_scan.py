#!/usr/bin/env python3
"""
anticipation_scan.py — Stage A (overnight batch)

Implements Pradeep Bonde's (Stockbee) ANTICIPATION scan: find stocks in an
established momentum trend that are currently in a quiet, contracted,
low-volatility consolidation — i.e. coiled BEFORE the momentum burst, not
after it.

Trend qualifier (at least one of Stockbee's three scans):
    Double Trouble : c / min(low,252) >= 1.8
    TI65 1% Mover  : avgc7 / avgc65   >  1.05
    MDT            : c / avgc126      >  1.19

Universal gates (his scan lines):
    min(volume, last 3 days) >= 100,000
    today's % change between -1% and +1%      <- the "quiet" gate
    close >= $3

Ritual gates (his checklist for a good anticipation setup):
    3-12 days since the last momentum burst   (consolidation length)
    at least one prior burst in the last 40d  (established trend / linear leg)
    no 4% breakdown during the consolidation  (orderly pullback)
    not up 3 days in a row
    pullback from consolidation high <= 20%   (shallow, not broken)

Quality score 0-100 ranks the survivors on volatility contraction, volume
contraction, pullback shallowness, trend strength, consolidation length and
today's range narrowness.

Writes anticipation_watch.json for anticipation_check.py (Stage B), including
each stock's PIVOT (consolidation high) — the level whose break starts the
burst.

Requires env var TWELVE_DATA_API_KEY. Paced to ~50 req/min (Grow 55 plan).

Outputs (unchanged): anticipation_watch.json  -- consumed by anticipation_check.py
Outputs (new):       data/anticipation.json   -- shared format for the combined dashboard
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

# NEW: shared output writer used by all the overnight scans.
from scan_output import write_scan, write_failure

# NEW: shared bar cache, filled once per night by fetch_bars.py. If it is
# absent this scan falls back to fetching its own data, exactly as before.
from bars_cache import CACHE

API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
TD_URL = "https://api.twelvedata.com/time_series"
UNIVERSE_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]

# ── Stockbee scan constants ──────────────────────────────────────────────────
DT_RATIO = 1.80           # Double Trouble:  c / 252-day low
TI65_RATIO = 1.05         # TI65 1% Mover:   avgc7 / avgc65
MDT_RATIO = 1.19          # MDT:             c / avgc126
MIN_VOL_3D = 100_000      # minv3.1
QUIET_PCT = 1.0           # |today's % change| must be <= this
MIN_PRICE = 3.0

# ── Ritual constants ─────────────────────────────────────────────────────────
BURST_PCT = 4.0           # what counts as a momentum burst / breakdown day
CONSOL_MIN = 3            # days since last burst, lower bound
CONSOL_MAX = 12           # days since last burst, upper bound
PRIOR_BURST_LOOKBACK = 40 # must have burst at least once in this window
MAX_PULLBACK = 20.0       # % off the consolidation high
MAX_UP_DAYS = 3           # reject if up this many days in a row

BARS = 260                # daily bars to fetch (same API cost as 3)
MIN_BARS = 70             # need at least this much history
REQUESTS_PER_MIN = 50
WATCH_CAP = 250           # keep the top N by score (keeps Stage B fast)

BAD_NAME_WORDS = ("WARRANT", "RIGHT", "UNIT", "PREFERRED", "NOTES", "DEBENTURE")
BAD_SYMBOL_CHARS = set(".$^~=")
OUT_FILE = "anticipation_watch.json"

INCOMPLETE_TODAY = None   # set in main()

# NEW: identity of this scan on the combined dashboard.
SCAN_ID = "anticipation"
SCAN_LABEL = "Stockbee Anticipation"


# ── helpers ──────────────────────────────────────────────────────────────────
def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "sb-anticipation/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def us_market_today_incomplete():
    """Today's US-Eastern date if the regular session has not finished yet."""
    utc = datetime.now(timezone.utc)
    offset = -4 if 3 <= utc.month <= 10 else -5
    et = utc + timedelta(hours=offset)
    if et.weekday() >= 5:
        return None
    if et.hour < 16 or (et.hour == 16 and et.minute < 10):
        return et.strftime("%Y-%m-%d")
    return None


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
            if any(c in BAD_SYMBOL_CHARS for c in sym):
                continue
            if any(w in name for w in BAD_NAME_WORDS):
                continue
            tv_exch = "NASDAQ" if is_nasdaq_file else ("NYSE" if exch == "N" else "AMEX")
            symbols.append((sym, tv_exch))
    seen, out = set(), []
    for sym, ex in sorted(symbols):
        if sym not in seen:
            seen.add(sym)
            out.append((sym, ex))
    return out


def load_universe_cached():
    """NEW: prefer the universe the shared cache was built from.

    fetch_bars.py already downloaded and filtered the NASDAQ Trader files, so
    reusing its list saves two downloads and guarantees this scan and the
    cache agree on which symbols exist.
    """
    if CACHE.available:
        universe = CACHE.universe()
        if universe:
            print(f"Universe from cache: {len(universe)} symbols")
            return universe
    print("No cache — building universe from NASDAQ Trader files")
    return load_universe()


def fetch_bars(symbol):
    """Return bars OLDEST-first as dicts of floats, or None."""
    # NEW: try the shared cache first. Ordering matters enormously here --
    # every calculation below indexes from the end (c[-1] is today, l[-252:]
    # is the last year), so newest_first=False is essential. Handed the bars
    # the other way round this scan would not crash, it would read the year
    # backwards and produce confident nonsense.
    cached = CACHE.get(symbol, newest_first=False)
    if cached:
        if len(cached) < MIN_BARS:
            return None
        # Already in the same shape this function returns, and the
        # in-progress bar was dropped when the cache was built.
        return cached

    url = f"{TD_URL}?symbol={symbol}&interval=1day&outputsize={BARS}&apikey={API_KEY}"
    try:
        data = json.loads(http_get(url))
    except Exception:
        return None
    vals = data.get("values")
    if not vals:
        return None
    # Drop the in-progress bar when running mid-session.
    if INCOMPLETE_TODAY and vals[0].get("datetime", "")[:10] == INCOMPLETE_TODAY:
        vals = vals[1:]
    if len(vals) < MIN_BARS:
        return None
    bars = []
    for v in reversed(vals):                      # oldest first
        try:
            bars.append({
                "d": v.get("datetime", "")[:10],
                "o": float(v["open"]), "h": float(v["high"]),
                "l": float(v["low"]),  "c": float(v["close"]),
                "v": float(v["volume"]),
            })
        except (KeyError, ValueError, TypeError):
            return None
    return bars


# ── the anticipation test ────────────────────────────────────────────────────
def evaluate(bars):
    n = len(bars)
    c = [b["c"] for b in bars]
    h = [b["h"] for b in bars]
    l = [b["l"] for b in bars]
    v = [b["v"] for b in bars]

    px = c[-1]
    if px < MIN_PRICE:
        return None
    if min(v[-3:]) < MIN_VOL_3D:
        return None

    pct_today = 100.0 * (c[-1] - c[-2]) / c[-2] if c[-2] else 0.0
    if abs(pct_today) > QUIET_PCT:                # the "quiet" gate
        return None

    # ── trend qualifiers (need at least one) ──
    quals = []
    lookback252 = l[-252:] if n >= 252 else l
    if min(lookback252) > 0 and px / min(lookback252) >= DT_RATIO:
        quals.append("DT")
    if n >= 65:
        a7, a65 = mean(c[-7:]), mean(c[-65:])
        if a65 > 0 and a7 / a65 > TI65_RATIO:
            quals.append("TI65")
    if n >= 126:
        a126 = mean(c[-126:])
        if a126 > 0 and px / a126 > MDT_RATIO:
            quals.append("MDT")
    if not quals:
        return None

    # ── daily % changes and burst/breakdown days ──
    pct = [0.0] * n
    for i in range(1, n):
        pct[i] = 100.0 * (c[i] - c[i - 1]) / c[i - 1] if c[i - 1] else 0.0
    burst_day = [pct[i] >= BURST_PCT and v[i] > v[i - 1] for i in range(n)]
    bdown_day = [pct[i] <= -BURST_PCT for i in range(n)]

    # days since the most recent burst = consolidation length
    days_since = None
    for i in range(n - 1, max(n - PRIOR_BURST_LOOKBACK - 1, 0), -1):
        if burst_day[i]:
            days_since = (n - 1) - i
            break
    if days_since is None:                        # no established leg
        return None
    if not (CONSOL_MIN <= days_since <= CONSOL_MAX):
        return None

    consol = slice(n - days_since, n)             # bars after the burst day

    # orderly: no 4% breakdown inside the consolidation
    if any(bdown_day[i] for i in range(n - days_since, n)):
        return None

    # not up 3 days in a row
    ups = 0
    for i in range(n - 1, n - 1 - MAX_UP_DAYS, -1):
        if i >= 1 and c[i] > c[i - 1]:
            ups += 1
        else:
            break
    if ups >= MAX_UP_DAYS:
        return None

    # pivot = consolidation high (the level whose break starts the burst)
    pivot = max(h[consol]) if days_since > 0 else h[-1]
    pullback_pct = 100.0 * (pivot - px) / pivot if pivot else 0.0
    if pullback_pct > MAX_PULLBACK:
        return None

    # ── contraction measures ──
    def rng_pct(i):
        return 100.0 * (h[i] - l[i]) / c[i] if c[i] else 0.0

    consol_rng = mean([rng_pct(i) for i in range(n - days_since, n)])
    base_start = max(0, n - days_since - 20)
    prior_rng = mean([rng_pct(i) for i in range(base_start, n - days_since)]) or 1e-9
    vol_ratio = mean(v[consol]) / (mean(v[base_start:n - days_since]) or 1e-9)
    rng_ratio = consol_rng / prior_rng
    adr20 = mean([rng_pct(i) for i in range(max(0, n - 20), n)])
    today_rng = rng_pct(n - 1)

    # ── quality score 0-100 ──
    s_vola = 25.0 * max(0.0, min(1.0, (1.0 - rng_ratio) / 0.5))    # tighter = better
    s_vol = 20.0 * max(0.0, min(1.0, (1.0 - vol_ratio) / 0.5))     # drier = better
    s_pull = 20.0 * max(0.0, min(1.0, 1.0 - pullback_pct / MAX_PULLBACK))
    s_trend = 15.0 * (len(quals) / 3.0)
    ideal = 4 <= days_since <= 8
    s_len = 10.0 if ideal else 5.0
    s_narrow = 10.0 * max(0.0, min(1.0, 1.0 - (today_rng / adr20 if adr20 else 1.0)))
    score = round(s_vola + s_vol + s_pull + s_trend + s_len + s_narrow, 1)

    return {
        "close": round(px, 2),
        "prev_volume": round(v[-1]),
        "pivot": round(pivot, 2),
        "dist_to_pivot": round(pullback_pct, 2),
        "consol_days": days_since,
        "quals": "+".join(quals),
        "rng_ratio": round(rng_ratio, 2),
        "vol_ratio": round(vol_ratio, 2),
        "adr_pct": round(adr20, 2),
        "avg_range_10": round(mean([rng_pct(i) for i in range(max(0, n - 10), n)]), 2),
        "score": score,
        "date": bars[-1]["d"],
    }


def to_scan_rows(kept):
    """NEW: translate anticipation setups into the shared dashboard format.

    anticipation_watch.json keeps its own field names because
    anticipation_check.py reads it and must not be disturbed. This makes a
    separate copy for the combined page, with the columns in the order that
    is most useful to read at a glance and volume in millions.
    """
    rows = []
    for s in kept:
        vol = s.get("prev_volume") or 0
        rows.append({
            "symbol": s.get("symbol", ""),
            "exchange": s.get("exchange", ""),
            "close": s.get("close"),
            "score": s.get("score"),
            "pivot": s.get("pivot"),
            "pct_to_pivot": s.get("dist_to_pivot"),
            "consol_days": s.get("consol_days"),
            "quals": s.get("quals", ""),
            "rng_ratio": s.get("rng_ratio"),
            "vol_ratio": s.get("vol_ratio"),
            "adr_pct": s.get("adr_pct"),
            "avg_range_10": s.get("avg_range_10"),
            "volM": round(vol / 1e6, 2),
            "setup_date": s.get("date", ""),
        })
    return rows


def main():
    if not API_KEY:
        sys.exit("TWELVE_DATA_API_KEY not set")
    global INCOMPLETE_TODAY
    INCOMPLETE_TODAY = us_market_today_incomplete()
    if INCOMPLETE_TODAY:
        print(f"US session in progress — excluding {INCOMPLETE_TODAY} bar, "
              f"testing last completed day")
    universe = load_universe_cached()
    print(f"Universe: {len(universe)} symbols")
    # NEW: no pacing needed when the data comes off local disk.
    delay = 0.0 if CACHE.available else 60.0 / REQUESTS_PER_MIN
    hits, scanned = [], 0
    for sym, exch in universe:
        bars = fetch_bars(sym)
        scanned += 1
        if bars:
            res = evaluate(bars)
            if res:
                res["symbol"] = sym
                res["exchange"] = exch
                hits.append(res)
                print(f"SETUP {sym:6s} score {res['score']:5.1f}  "
                      f"{res['quals']:12s} consol {res['consol_days']}d  "
                      f"pivot {res['pivot']}  ({res['dist_to_pivot']}% away)")
        if scanned % 500 == 0:
            print(f"...{scanned}/{len(universe)} scanned, {len(hits)} setups")
        if delay:
            time.sleep(delay)
    hits.sort(key=lambda x: -x["score"])
    kept = hits[:WATCH_CAP]
    out = {"generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "setup_date": kept[0]["date"] if kept else "",
           "found": len(hits), "count": len(kept), "stocks": kept}
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {OUT_FILE}: {len(hits)} setups found, kept top {len(kept)}")
    CACHE.report()

    # --- NEW: shared output for the combined dashboard ---------------------
    # Same stocks, same order (best score first), different packaging.
    write_scan(
        SCAN_ID,
        to_scan_rows(kept),
        label=SCAN_LABEL,
        meta={
            "sort": "score descending",
            "universe": len(universe),
            "found": len(hits),
            "cap": WATCH_CAP,
            "setup_date": (kept[0]["date"] if kept else ""),
        },
    )


if __name__ == "__main__":
    # NEW: if the scan falls over, leave a note so the dashboard can show a
    # red banner on this tab instead of a silently empty table. The error is
    # re-raised so the GitHub Actions run still shows as failed.
    try:
        main()
    except Exception as exc:
        try:
            write_failure(SCAN_ID, exc, label=SCAN_LABEL)
        except Exception:
            pass
        raise

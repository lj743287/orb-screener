#!/usr/bin/env python3
"""
build_universe.py

Builds the daily scanner's ticker list from Twelve Data:
- Source: /stocks?exchange=NASDAQ and /stocks?exchange=NYSE
- Filter: stocks only (exclude ETFs) using Twelve Data 'type'
- Filter: price > 1.50 using /price endpoint
- Writes: daily_universe.txt (one symbol per line) sorted A-Z

Hardened for GitHub Actions:
- Short timeouts, retries, and unbuffered logging (flush=True)
- Refuses to overwrite an existing list with a much smaller one

Runs weekly. Output filename comes from UNIVERSE_OUTFILE and the workflow
sets it to daily_universe.txt, which is the file daily_scanner.py reads.
"""

import os
import time
import json
import math
from typing import List, Optional
import urllib.parse
import urllib.request
import urllib.error

BASE_URL = "https://api.twelvedata.com"

# NEW: safety floor. A rebuilt list smaller than this fraction of the existing
# one is treated as a bad data day, not a real change -- the old list is kept
# and the run fails loudly. A genuine market does not shed a third of its
# listings in a week, so if this ever trips, something upstream is broken and
# silently scanning half a universe for the next week would be much worse
# than a red tick in the Actions tab.
MIN_KEEP_FRACTION = 0.70


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


class RateLimiter:
    """Simple per-minute throttler (rolling 60s window)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.window_start = time.time()
        self.count = 0

    def wait(self) -> None:
        if self.max_per_minute <= 0:
            return

        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= 60:
            self.window_start = now
            self.count = 0

        if self.count >= self.max_per_minute:
            sleep_for = max(0.0, 60 - elapsed) + 0.25
            print(f"[pacing] Hit {self.max_per_minute}/min, sleeping {sleep_for:.2f}s", flush=True)
            time.sleep(sleep_for)
            self.window_start = time.time()
            self.count = 0

        self.count += 1


def http_get_json(url: str, timeout: int = 10, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "daily-stock-scanner/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
            return json.loads(data)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            backoff = 1.5 * attempt
            print(f"[http] attempt {attempt}/{retries} failed: {type(e).__name__}: {e} | sleeping {backoff:.1f}s", flush=True)
            time.sleep(backoff)
    raise RuntimeError(f"HTTP failed after {retries} retries. URL={url} LastError={last_err}")


def build_stocks_catalog(apikey: str, exchange: str) -> List[dict]:
    q = urllib.parse.urlencode({"apikey": apikey, "exchange": exchange})
    url = f"{BASE_URL}/stocks?{q}"
    data = http_get_json(url)
    if data.get("status") != "ok":
        raise RuntimeError(f"/stocks failed for {exchange}: {data}")
    return data.get("data", [])


def is_stock(rec: dict) -> bool:
    t = (rec.get("type") or "").strip()
    keep = {"Common Stock", "Depositary Receipt", "American Depositary Receipt", "REIT"}
    return t in keep


def fetch_price(apikey: str, symbol: str) -> Optional[float]:
    q = urllib.parse.urlencode({"apikey": apikey, "symbol": symbol})
    url = f"{BASE_URL}/price?{q}"
    data = http_get_json(url)

    if "price" in data and data.get("price") not in (None, ""):
        try:
            return float(data["price"])
        except (TypeError, ValueError):
            return None

    if data.get("status") == "error":
        return None

    return None


def count_existing(path: str) -> int:
    """NEW: how many symbols the current list holds, 0 if there isn't one."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return len([l for l in f.read().splitlines() if l.strip()])
    except Exception:
        return 0


def write_universe(path: str, symbols: List[str]) -> None:
    symbols = sorted(set(s.strip().upper() for s in symbols if s and s.strip()))
    with open(path, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")


def main() -> int:
    apikey = os.getenv("TWELVE_DATA_API_KEY") or os.getenv("TWELVEDATA_API_KEY") or ""
    if not apikey:
        print("ERROR: Missing Twelve Data API key. Set TWELVE_DATA_API_KEY (or TWELVEDATA_API_KEY).", flush=True)
        return 2

    min_price = env_float("UNIVERSE_MIN_PRICE", 1.50)
    max_per_minute = env_int("TD_MAX_REQUESTS_PER_MINUTE", 50)
    max_symbols = env_int("UNIVERSE_MAX_SYMBOLS", 0)  # 0 = no cap
    out_path = os.getenv("UNIVERSE_OUTFILE", "daily_universe.txt")

    existing = count_existing(out_path)

    limiter = RateLimiter(max_per_minute)

    print("[universe] Starting build...", flush=True)
    print(f"[universe] min_price={min_price:.2f} max_per_minute={max_per_minute} max_symbols={max_symbols or 'none'} outfile={out_path}", flush=True)
    print(f"[universe] existing list holds {existing} symbols", flush=True)

    exchanges = ["NASDAQ", "NYSE"]
    all_recs: List[dict] = []

    print("[universe] Fetching symbol catalog from /stocks ...", flush=True)
    for ex in exchanges:
        limiter.wait()
        recs = build_stocks_catalog(apikey, ex)
        print(f"[universe] {ex}: {len(recs)} rows from /stocks", flush=True)
        all_recs.extend(recs)

    stock_recs = [r for r in all_recs if is_stock(r)]
    print(f"[universe] After stock-only filter: {len(stock_recs)}", flush=True)

    symbols = sorted(set((r.get("symbol") or "").strip() for r in stock_recs if (r.get("symbol") or "").strip()))
    print(f"[universe] Unique symbols pre-price-filter: {len(symbols)}", flush=True)

    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]
        print(f"[universe] TEST MODE: limiting to first {max_symbols} symbols", flush=True)

    kept: List[str] = []
    skipped_no_price = 0
    skipped_below = 0

    print(f"[universe] Applying price filter > {min_price:.2f} using /price ...", flush=True)

    for i, sym in enumerate(symbols, start=1):
        limiter.wait()
        px = fetch_price(apikey, sym)
        if px is None or (isinstance(px, float) and (math.isnan(px) or math.isinf(px))):
            skipped_no_price += 1
        else:
            if px > min_price:
                kept.append(sym)
            else:
                skipped_below += 1

        if i % 50 == 0:
            print(f"[universe] progress {i}/{len(symbols)} kept={len(kept)} no_price={skipped_no_price} below={skipped_below}", flush=True)

    # NEW: refuse to replace a healthy list with a suspiciously small one.
    # Skipped in test mode, where a small result is the whole point.
    if existing and not max_symbols:
        floor = int(existing * MIN_KEEP_FRACTION)
        if len(kept) < floor:
            print("", flush=True)
            print(f"ERROR: refusing to overwrite {out_path}.", flush=True)
            print(f"  existing list : {existing} symbols", flush=True)
            print(f"  rebuilt list  : {len(kept)} symbols", flush=True)
            print(f"  minimum kept  : {floor} ({MIN_KEEP_FRACTION:.0%} of existing)", flush=True)
            print(f"  no_price={skipped_no_price} below_threshold={skipped_below}", flush=True)
            print("", flush=True)
            print("A market does not lose that many listings in a week, so this "
                  "is far more likely to be a bad response from Twelve Data. "
                  "The existing list has been left untouched and the scans will "
                  "carry on using it. Re-run this workflow by hand once the "
                  "provider looks healthy.", flush=True)
            return 3

    write_universe(out_path, kept)

    print("[universe] Done.", flush=True)
    print(f"[universe] wrote: {out_path}", flush=True)
    print(f"[universe] kept: {len(kept)} (was {existing})", flush=True)
    print(f"[universe] skipped_no_price: {skipped_no_price}", flush=True)
    print(f"[universe] skipped_below_threshold: {skipped_below}", flush=True)
    print("[universe] preview:", ", ".join(kept[:20]), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

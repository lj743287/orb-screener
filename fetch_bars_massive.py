#!/usr/bin/env python3
"""Build the existing ORB daily-bar cache from Massive's grouped daily API.

This file contains no screening logic. It writes the same cache structure as
fetch_bars.py so the existing production screener.py can run unchanged.

Massive's grouped daily endpoint returns OHLCV for the whole US stock market
for one date, so the initial 250-session cache requires roughly 250 requests
rather than one request per symbol.

Required env var:
  MASSIVE_API_KEY

Optional env vars:
  CACHE_FILE                    default cache/massive_bars.json.gz
  BARS                          default 250 completed market sessions
  MASSIVE_REQUESTS_PER_MIN      default 4.5 (below Massive Basic's 5 req/min limit)
  MAX_SYMBOLS                   optional universe cap for testing
  MIN_COVERAGE                  default 0.90
"""

import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

# Reuse the production universe and incomplete-session handling verbatim.
from fetch_bars import load_universe, incomplete_today


BASE_URL = "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{date}"
API_KEY = os.environ.get("MASSIVE_API_KEY", "")
CACHE_FILE = os.environ.get("CACHE_FILE", "cache/massive_bars.json.gz")
BARS = int(os.environ.get("BARS", "250"))
REQUESTS_PER_MIN = float(os.environ.get("MASSIVE_REQUESTS_PER_MIN", "4.5"))
MAX_SYMBOLS = int(os.environ.get("MAX_SYMBOLS", "0") or "0")
MIN_COVERAGE = float(os.environ.get("MIN_COVERAGE", "0.90"))

_last_request_at = 0.0


def pace_request():
    global _last_request_at
    if REQUESTS_PER_MIN <= 0:
        _last_request_at = time.monotonic()
        return
    min_gap = 60.0 / REQUESTS_PER_MIN
    now = time.monotonic()
    wait = min_gap - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def request_day(day, tries=8):
    """Fetch one grouped market day with conservative free-tier pacing."""
    url = BASE_URL.format(date=day)
    params = {
        "adjusted": "true",
        "include_otc": "false",
        "apiKey": API_KEY,
    }
    delay = 2.0

    for attempt in range(tries):
        try:
            pace_request()
            r = requests.get(url, params=params, timeout=60)

            if r.status_code == 429:
                raw_retry = r.headers.get("Retry-After", "")
                try:
                    retry_after = float(raw_retry) if raw_retry else 0.0
                except ValueError:
                    retry_after = 0.0
                wait = max(retry_after, 65.0)
                print(
                    f"Massive rate limit (429); waiting {wait:.0f}s before retry "
                    f"{attempt + 1}/{tries}",
                    flush=True,
                )
                time.sleep(wait)
                delay = 2.0
                continue

            if r.status_code >= 500:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue

            r.raise_for_status()
            payload = r.json()
            status = str(payload.get("status", "")).upper()
            if status not in ("", "OK"):
                raise RuntimeError(f"Massive returned status {status}: {payload}")
            return payload

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            if attempt == tries - 1:
                raise
            print(f"Massive request error for {day}: {exc}; retrying", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)

    raise RuntimeError(f"Massive request failed for {day} after retries")


def main():
    if not API_KEY:
        sys.exit("MASSIVE_API_KEY not set")

    universe = load_universe()
    if MAX_SYMBOLS:
        universe = universe[:MAX_SYMBOLS]
        print(f"TEST MODE: capped at {MAX_SYMBOLS} symbols", flush=True)

    exchanges = {sym: exch for sym, exch in universe}
    wanted = set(exchanges)
    bars = {sym: [] for sym in wanted}

    skip = incomplete_today()
    cursor = datetime.now(timezone.utc).date()
    if skip and cursor.isoformat() == skip:
        print(
            f"US session in progress - starting before {skip} so the ORB screen "
            "sees the same last completed day as production",
            flush=True,
        )
        cursor -= timedelta(days=1)

    print(
        f"Massive free ORB cache: {len(wanted)} symbols, target {BARS} completed "
        f"market sessions, request ceiling {REQUESTS_PER_MIN:g}/min",
        flush=True,
    )

    sessions = 0
    calls = 0
    empty_weekdays = 0
    started = time.time()
    oldest_session = None
    newest_session = None

    # 250 US sessions fit comfortably inside 400 calendar days. The wider
    # guard prevents an accidental infinite loop if the provider repeatedly
    # returns empty data.
    max_calendar_days = max(400, int(BARS * 2.2))
    scanned_days = 0

    while sessions < BARS and scanned_days < max_calendar_days:
        day = cursor.isoformat()
        cursor -= timedelta(days=1)
        scanned_days += 1

        day_obj = datetime.fromisoformat(day).date()
        if day_obj.weekday() >= 5:
            continue

        payload = request_day(day)
        calls += 1
        results = payload.get("results") or []

        if not results:
            empty_weekdays += 1
            continue

        market_rows = 0
        for row in results:
            sym = str(row.get("T", "")).strip().upper()
            if sym not in wanted:
                continue
            try:
                bar = {
                    "d": day,
                    "o": float(row["o"]),
                    "h": float(row["h"]),
                    "l": float(row["l"]),
                    "c": float(row["c"]),
                    "v": float(row["v"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            bars[sym].append(bar)
            market_rows += 1

        # A real trading day should contain many in-universe stocks. Requiring
        # at least 100 avoids counting a malformed or partial response as one of
        # the 250 sessions.
        if market_rows < 100:
            for sym in wanted:
                if bars[sym] and bars[sym][-1].get("d") == day:
                    bars[sym].pop()
            print(
                f"Ignoring suspiciously small grouped response for {day}: "
                f"{market_rows} in-universe rows",
                flush=True,
            )
            continue

        sessions += 1
        if newest_session is None:
            newest_session = day
        oldest_session = day

        if sessions % 25 == 0 or sessions == BARS:
            elapsed = (time.time() - started) / 60
            covered = sum(1 for rows in bars.values() if rows)
            print(
                f"  {sessions}/{BARS} sessions - {covered}/{len(wanted)} symbols "
                f"with data, {calls} API calls, {elapsed:.1f} min",
                flush=True,
            )

    if sessions < BARS:
        sys.exit(
            f"ERROR: only collected {sessions}/{BARS} completed market sessions "
            f"after scanning {scanned_days} calendar days"
        )

    bars = {sym: rows[:BARS] for sym, rows in bars.items() if rows}
    missing = sorted(wanted - set(bars))

    os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)
    out = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": "massive",
        "feed": "grouped_daily",
        "adjustment": "split",
        "bars_requested": BARS,
        "dropped_date": skip or "",
        "newest_session": newest_session or "",
        "oldest_session": oldest_session or "",
        "universe": len(universe),
        "count": len(bars),
        "failed": missing,
        "api_calls": calls,
        "empty_weekdays": empty_weekdays,
        "exchanges": exchanges,
        "bars": bars,
    }

    tmp = CACHE_FILE + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    os.replace(tmp, CACHE_FILE)

    elapsed = (time.time() - started) / 60
    size_mb = os.path.getsize(CACHE_FILE) / 1e6
    coverage = len(bars) / len(wanted) if wanted else 0.0
    print(
        f"Wrote {CACHE_FILE}: {len(bars)}/{len(wanted)} symbols "
        f"({coverage:.1%}), {sessions} sessions, {size_mb:.1f} MB, "
        f"{calls} API calls, {elapsed:.1f} min",
        flush=True,
    )

    if coverage < MIN_COVERAGE:
        sys.exit(
            f"ERROR: Massive cache coverage only {len(bars)}/{len(wanted)} "
            f"({coverage:.1%}), below required {MIN_COVERAGE:.0%}. "
            "Refusing to treat this as a valid comparison run."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

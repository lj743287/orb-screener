#!/usr/bin/env python3
"""Build TradingView-importable watchlists from the exact ORB production universe.

This does NOT alter screener logic. It reuses fetch_bars.load_universe(), which
already returns the filtered NASDAQ/NYSE/AMEX universe together with the
TradingView exchange prefix for each symbol.

TradingView watchlists are limited to 1,000 symbols, so the universe is split
into chunks of at most 1,000 and written as comma-separated EXCHANGE:SYMBOL
lists ready for TradingView import.
"""

from pathlib import Path

from fetch_bars import load_universe

OUT_DIR = Path("tradingview_watchlists")
CHUNK_SIZE = 1000


def main():
    universe = load_universe()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale chunks from previous runs.
    for old in OUT_DIR.glob("orb_universe_*.txt"):
        old.unlink()

    print(f"Production universe: {len(universe)} symbols")

    for start in range(0, len(universe), CHUNK_SIZE):
        chunk = universe[start:start + CHUNK_SIZE]
        number = start // CHUNK_SIZE + 1
        path = OUT_DIR / f"orb_universe_{number:02d}.txt"
        values = [f"{exchange}:{symbol}" for symbol, exchange in chunk]
        path.write_text(",".join(values), encoding="utf-8")
        print(f"{path}: {len(values)} symbols")


if __name__ == "__main__":
    main()

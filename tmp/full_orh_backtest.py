#!/usr/bin/env python3
"""End-to-end ORB + earnings + MA + 30-minute ORH backtest.

The primary strategy follows the current Trading Strategy Tracker defaults:
* Daily WATCHLIST PASS = ORB PASS + positive previous EPS surprise + MA GO.
* Exclude entries within seven calendar days of the next known earnings date.
* Opening range = 09:30-10:00 America/New_York.
* Buy the first post-10:00 break above the opening-range high.
* Initial stop = opening-range low. Do not reject sub-2% stops; reject >8%.
* Move stop to breakeven after a confirmed daily close >= 2.5% above entry.
* Exit at the initial/breakeven stop or a +45% hard target.
* Fixed $15,000 position, $200,000 starting capital.

It also reports gate-off, uncapped-stop and optional-partial variants. Alpaca
one-minute IEX bars are used for the entry day. Split-adjusted daily bars are
used after entry. Any later daily bar that touches stop and target is treated
conservatively as stop-first and counted in the ambiguity field.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
OUT = Path(os.environ.get("ORH_OUT", "tmp/full_orh_backtest_out"))
CACHE_FILE = os.environ.get("CACHE_FILE", "cache/full_orh_bars.json.gz")
SIGNAL_START = pd.Timestamp(os.environ.get("SIGNAL_START", "2025-07-01"))
SIGNAL_END = pd.Timestamp(os.environ.get("SIGNAL_END", "2026-06-30"))
DATA_END = pd.Timestamp(os.environ.get("DATA_END", datetime.now(UTC).date().isoformat()))
ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_FEED = os.environ.get("ALPACA_FEED", "iex").strip().lower()
API_KEY = os.environ.get("APCA_API_KEY_ID", "")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY", "")
RPM = int(os.environ.get("REQUESTS_PER_MIN", "170"))
EARN_WORKERS = int(os.environ.get("EARN_WORKERS", "6"))
INTRA_BATCH = int(os.environ.get("INTRA_BATCH", "18"))
TARGET_PCT = float(os.environ.get("TARGET_PCT", "45")) / 100.0
BE_TRIGGER_PCT = float(os.environ.get("BE_TRIGGER_PCT", "2.5")) / 100.0
PARTIAL_TRIGGER_PCT = float(os.environ.get("PARTIAL_TRIGGER_PCT", "15")) / 100.0
PARTIAL_FRACTION = float(os.environ.get("PARTIAL_FRACTION", "25")) / 100.0
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "8")) / 100.0
POSITION_VALUE = float(os.environ.get("POSITION_VALUE", "15000"))
STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", "200000"))

MEANS = np.array([1.6053478743377458, 0.8587446273550717, 5.228070175438597,
                  8.271929824561404, 0.7042055314591412, 1.5652731406790044])
SDS = np.array([0.4884305133688643, 0.2336522673244613, 6.918111353312391,
                5.140627799258959, 0.19733635029076077, 1.105743621784014])
COEFS = np.array([0.35814256501291186, 0.15962122502434276, -0.30038853716369424,
                  0.005477613398225789, 0.08267834158482865, -0.03172365295762897])
INTERCEPT = -0.0746060880381915


def iso_value(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, dict):
        return {k: iso_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [iso_value(v) for v in value]
    return value


class AlpacaClient:
    def __init__(self):
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")
        self.headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
        self.spacing = 60.0 / max(1, RPM)
        self.last_call = 0.0
        self.calls = 0

    def _wait(self):
        delay = self.spacing - (time.monotonic() - self.last_call)
        if delay > 0:
            time.sleep(delay)

    def get(self, params, tries=6):
        import requests

        pause = 1.0
        for attempt in range(tries):
            self._wait()
            try:
                response = requests.get(ALPACA_URL, headers=self.headers, params=params, timeout=90)
                self.last_call = time.monotonic()
                self.calls += 1
                if response.status_code in (401, 403):
                    raise PermissionError(f"Alpaca rejected {ALPACA_FEED} access ({response.status_code})")
                if response.status_code == 429:
                    time.sleep(max(float(response.headers.get("Retry-After", 0) or 0), pause))
                    pause = min(pause * 2, 30)
                    continue
                if response.status_code >= 500:
                    time.sleep(pause)
                    pause = min(pause * 2, 30)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == tries - 1:
                    raise
                time.sleep(pause)
                pause = min(pause * 2, 30)
        raise RuntimeError("Alpaca request failed after retries")

    def bars(self, symbols, timeframe, start, end, batch_size=50):
        collected = {symbol: [] for symbol in symbols}
        for offset in range(0, len(symbols), batch_size):
            group = symbols[offset:offset + batch_size]
            page_token = None
            while True:
                params = {
                    "symbols": ",".join(group), "timeframe": timeframe,
                    "start": start, "end": end, "limit": 10000,
                    "adjustment": "split", "feed": ALPACA_FEED, "sort": "asc",
                }
                if page_token:
                    params["page_token"] = page_token
                payload = self.get(params)
                for symbol, rows in (payload.get("bars") or {}).items():
                    if symbol in collected:
                        collected[symbol].extend(rows)
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        return collected


def load_cache():
    with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("feed") != ALPACA_FEED:
        raise RuntimeError(f"Expected {ALPACA_FEED} cache, got {payload.get('feed')!r}")
    return payload


def cache_frame(rows):
    if not rows:
        return None
    frame = pd.DataFrame(rows).rename(columns={"d": "datetime", "o": "open", "h": "high",
                                                   "l": "low", "c": "close", "v": "volume"})
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.set_index("datetime").sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])


def alpaca_frame(rows):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(NY)
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return frame[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr_wilder(frame, n=20):
    previous = frame.close.shift(1)
    true_range = pd.concat([(frame.high - frame.low).abs(),
                            (frame.high - previous).abs(),
                            (frame.low - previous).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False).mean()


def bars_since(condition):
    values = np.asarray(condition.fillna(False), dtype=bool)
    indices = np.where(values)[0]
    return 21 if len(indices) == 0 else min(len(values) - 1 - int(indices[-1]), 21)


def ma_features(frame):
    if frame is None or len(frame) < 60:
        return None
    close, high, low, volume = frame.close, frame.high, frame.low, frame.volume
    atr20 = atr_wilder(frame, 20)
    ema10, ema20 = ema(close, 10), ema(close, 20)
    range3 = (high.rolling(3).max() - low.rolling(3).min()) / atr20
    volume_ratio = volume.rolling(5).mean() / volume.rolling(20).mean()
    reset_age = bars_since(close <= ema10)
    recent_highs = high.iloc[-20:].to_numpy()
    high_age = len(recent_highs) - 1 - int(np.argmax(recent_highs))
    high20, low20 = high.iloc[-20:].max(), low.iloc[-20:].min()
    close_position = (close.iloc[-1] - low20) / (high20 - low20) if high20 > low20 else 0.5
    prior_high = high.shift(1).rolling(20).max().iloc[-1]
    distance_high = (prior_high - close.iloc[-1]) / atr20.iloc[-1] if atr20.iloc[-1] > 0 else np.nan
    momentum15 = ((close.iloc[-1] / close.iloc[-16]) - 1) * 100 if close.iloc[-16] else np.nan
    ema20_distance = (close.iloc[-1] - ema20.iloc[-1]) / atr20.iloc[-1] if atr20.iloc[-1] > 0 else np.nan
    stretch50 = (close.iloc[-1] - low.iloc[-50:].min()) / atr20.iloc[-1] if atr20.iloc[-1] > 0 else np.nan
    values = np.array([range3.iloc[-1], volume_ratio.iloc[-1], reset_age, high_age,
                       close_position, distance_high], dtype=float)
    if np.any(np.isnan(values)):
        return None
    score = 100 / (1 + math.exp(-(INTERCEPT + float(np.dot(COEFS, (values - MEANS) / SDS)))))
    weak = bool(momentum15 <= 4.6 and ema20_distance <= 0.72)
    exhausted = bool(stretch50 > 11.5)
    normal = bool(score >= 41 and not weak and not exhausted)
    exceptional = bool(35 <= score < 41 and momentum15 >= 20 and ema20_distance >= 1.40 and not exhausted)
    return {
        "ma_score": score, "ma_go": int(normal or exceptional), "ma_exceptional": int(exceptional),
        "mom15": momentum15, "ema20_dist_atr": ema20_distance, "stretch50_atr": stretch50,
        "range3_atr": values[0], "vol_ratio5": values[1], "reset_age": int(reset_age),
        "high_age": int(high_age), "close_pos20": close_position,
        "dist_high20_atr": distance_high, "weak": int(weak), "exhausted": int(exhausted),
    }


def prior_slice(frame, session):
    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    copy = frame.copy()
    copy.index = index
    return copy[copy.index.normalize() < pd.Timestamp(session).normalize()]


def fetch_earnings(symbol):
    import yfinance as yf

    for attempt in range(3):
        try:
            earnings = yf.Ticker(symbol).get_earnings_dates(limit=32)
            if earnings is not None and not earnings.empty:
                earnings = earnings.copy()
                earnings.index = pd.to_datetime(earnings.index)
                return symbol, earnings
        except Exception:
            pass
        time.sleep(1 + 2 * attempt)
    return symbol, None


def earnings_for_date(earnings, date):
    output = {
        "earnings_history_known": 0, "prev_earnings_pass": 0,
        "prev_eps_est": np.nan, "prev_eps_reported": np.nan,
        "prev_eps_surprise_pct": np.nan, "prev_earnings_date": None,
        "next_earnings_known": 0, "next_earnings_date": None,
        "days_to_next_earnings": np.nan, "earnings_blackout_7d": np.nan,
    }
    if earnings is None or earnings.empty:
        return output
    target = pd.Timestamp(date)
    target = (target.tz_localize(NY) if target.tzinfo is None else target.tz_convert(NY)).normalize()
    target += pd.Timedelta(hours=9, minutes=30)
    frame = earnings.copy()
    index = pd.to_datetime(frame.index)
    index = index.tz_localize(NY) if index.tz is None else index.tz_convert(NY)
    frame.index = index
    frame = frame.sort_index()
    previous = frame[frame.index < target]
    if not previous.empty:
        row = previous.iloc[-1]
        estimate = pd.to_numeric(row.get("EPS Estimate", np.nan), errors="coerce")
        reported = pd.to_numeric(row.get("Reported EPS", np.nan), errors="coerce")
        output["prev_earnings_date"] = previous.index[-1].isoformat()
        if not pd.isna(estimate) and not pd.isna(reported):
            output.update(earnings_history_known=1, prev_eps_est=float(estimate),
                          prev_eps_reported=float(reported), prev_earnings_pass=int(reported > estimate))
            if estimate != 0:
                output["prev_eps_surprise_pct"] = float((reported - estimate) / abs(estimate) * 100)
    upcoming = frame[frame.index >= target]
    if not upcoming.empty:
        next_date = upcoming.index[0]
        days = (next_date.date() - target.date()).days
        output.update(next_earnings_known=1, next_earnings_date=next_date.isoformat(),
                      days_to_next_earnings=int(days), earnings_blackout_7d=int(0 <= days <= 7))
    return output


def market_regime(frame, sessions):
    result = {}
    for session in sessions:
        prior = prior_slice(frame, session)
        if len(prior) < 20:
            result[pd.Timestamp(session).strftime("%Y-%m-%d")] = np.nan
            continue
        result[pd.Timestamp(session).strftime("%Y-%m-%d")] = int(
            ema(prior.close, 10).iloc[-1] > ema(prior.close, 20).iloc[-1]
        )
    return result


def session_bounds(date):
    day = pd.Timestamp(date).date()
    start = datetime.combine(day, dt_time(9, 30), NY).astimezone(UTC)
    end = datetime.combine(day, dt_time(16, 1), NY).astimezone(UTC)
    return start.isoformat(), end.isoformat()


def opening_range_signal(frame, date):
    if frame is None or frame.empty:
        return None
    day = pd.Timestamp(date).date()
    bars = frame[frame.index.date == day]
    if bars.empty:
        return None
    opening = bars[(bars.index.time >= dt_time(9, 30)) & (bars.index.time < dt_time(10, 0))]
    after = bars[(bars.index.time >= dt_time(10, 0)) & (bars.index.time < dt_time(16, 0))]
    if opening.empty or after.empty:
        return None
    opening_high = float(opening.high.max())
    opening_low = float(opening.low.min())
    triggers = after[after.high >= opening_high]
    if triggers.empty or opening_high <= opening_low:
        return {
            "orh_triggered": 0, "opening_high": opening_high, "opening_low": opening_low,
            "opening_minutes": len(opening), "session_minutes": len(bars),
        }
    trigger_time = triggers.index[0]
    trigger_bar = triggers.iloc[0]
    entry = max(opening_high, float(trigger_bar.open))
    stop_pct = (entry - opening_low) / entry
    return {
        "orh_triggered": 1, "opening_high": opening_high, "opening_low": opening_low,
        "opening_minutes": len(opening), "session_minutes": len(bars),
        "trigger_time": trigger_time.isoformat(), "entry_price": entry,
        "initial_stop": opening_low, "stop_pct": stop_pct,
        "entry_gap_pct": (entry / opening_high - 1) * 100,
        "entry_bar_stop_touch": int(float(trigger_bar.low) <= opening_low),
    }


@dataclass
class TradeOutcome:
    exit_date: str
    exit_price: float
    exit_reason: str
    return_pct: float
    pnl_usd: float
    holding_sessions: int
    be_activated: int
    partial_taken: int
    ambiguity_days: int
    open_at_end: int


def _exit_result(entry, realised, remaining, exit_price, reason, holding, be, partial, ambiguity, open_end):
    total_return = realised + remaining * (exit_price / entry - 1)
    return TradeOutcome(
        exit_date="", exit_price=float(exit_price), exit_reason=reason,
        return_pct=float(total_return * 100), pnl_usd=float(total_return * POSITION_VALUE),
        holding_sessions=int(holding), be_activated=int(be), partial_taken=int(partial),
        ambiguity_days=int(ambiguity), open_at_end=int(open_end),
    )


def simulate_trade(entry_day_minutes, daily, signal_date, entry, initial_stop, partial_enabled=False):
    signal_day = pd.Timestamp(signal_date).normalize()
    stop = float(initial_stop)
    target = entry * (1 + TARGET_PCT)
    partial_price = entry * (1 + PARTIAL_TRIGGER_PCT)
    realised = 0.0
    remaining = 1.0
    partial_taken = False
    be_active = False
    ambiguity = 0

    post_opening_range = entry_day_minutes[
        (entry_day_minutes.index.time >= dt_time(10, 0)) &
        (entry_day_minutes.index.time < dt_time(16, 0))
    ]
    trigger_candidates = post_opening_range[post_opening_range.high >= entry]
    if trigger_candidates.empty:
        raise ValueError("simulate_trade called without a post-10:00 ORH trigger")
    trigger_rows = post_opening_range[post_opening_range.index >= trigger_candidates.index[0]]
    for timestamp, bar in trigger_rows.iterrows():
        stop_hit = float(bar.low) <= stop
        target_hit = float(bar.high) >= target
        partial_hit = partial_enabled and not partial_taken and float(bar.high) >= partial_price
        if stop_hit and (target_hit or partial_hit):
            ambiguity += 1
        if stop_hit:
            fill = min(stop, float(bar.open)) if float(bar.open) < stop else stop
            result = _exit_result(entry, realised, remaining, fill, "initial_stop", 1,
                                  be_active, partial_taken, ambiguity, False)
            result.exit_date = timestamp.isoformat()
            return result
        if partial_hit:
            realised += PARTIAL_FRACTION * (partial_price / entry - 1)
            remaining -= PARTIAL_FRACTION
            partial_taken = True
        if target_hit:
            result = _exit_result(entry, realised, remaining, target, "target", 1,
                                  be_active, partial_taken, ambiguity, False)
            result.exit_date = timestamp.isoformat()
            return result

    last_close = float(entry_day_minutes.iloc[-1].close)
    if last_close >= entry * (1 + BE_TRIGGER_PCT):
        stop = entry
        be_active = True

    future = daily[daily.index.normalize() > signal_day]
    holding = 1
    for timestamp, bar in future.iterrows():
        holding += 1
        stop_hit = float(bar.low) <= stop
        target_hit = float(bar.high) >= target
        partial_hit = partial_enabled and not partial_taken and float(bar.high) >= partial_price
        if stop_hit and (target_hit or partial_hit):
            ambiguity += 1
        if stop_hit:
            fill = float(bar.open) if float(bar.open) < stop else stop
            reason = "breakeven_stop" if be_active else "initial_stop"
            result = _exit_result(entry, realised, remaining, fill, reason, holding,
                                  be_active, partial_taken, ambiguity, False)
            result.exit_date = pd.Timestamp(timestamp).isoformat()
            return result
        if partial_hit:
            realised += PARTIAL_FRACTION * (partial_price / entry - 1)
            remaining -= PARTIAL_FRACTION
            partial_taken = True
        if target_hit:
            result = _exit_result(entry, realised, remaining, target, "target", holding,
                                  be_active, partial_taken, ambiguity, False)
            result.exit_date = pd.Timestamp(timestamp).isoformat()
            return result
        if float(bar.close) >= entry * (1 + BE_TRIGGER_PCT):
            stop = entry
            be_active = True

    if future.empty:
        last_date = signal_day
        final_close = last_close
    else:
        last_date = future.index[-1]
        final_close = float(future.iloc[-1].close)
    result = _exit_result(entry, realised, remaining, final_close, "open_mark_to_market", holding,
                          be_active, partial_taken, ambiguity, True)
    result.exit_date = pd.Timestamp(last_date).isoformat()
    return result


def outcome_metrics(frame, return_column="pure_return_pct", pnl_column="pure_pnl_usd"):
    if frame.empty:
        return {"n": 0}
    returns = frame[return_column].astype(float)
    pnl = frame[pnl_column].astype(float)
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "n": int(len(frame)), "symbols": int(frame.symbol.nunique()),
        "closed": int((frame.pure_open_at_end == 0).sum()), "open_at_end": int(frame.pure_open_at_end.sum()),
        "win_rate": float((pnl > 0).mean()), "mean_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()), "total_pnl_usd": float(pnl.sum()),
        "mean_pnl_usd": float(pnl.mean()),
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) else None,
        "targets": int((frame.pure_exit_reason == "target").sum()),
        "initial_stops": int((frame.pure_exit_reason == "initial_stop").sum()),
        "breakeven_stops": int((frame.pure_exit_reason == "breakeven_stop").sum()),
        "median_holding_sessions": float(frame.pure_holding_sessions.median()),
        "ambiguous_trades": int((frame.pure_ambiguity_days > 0).sum()),
    }


def portfolio_simulation(frame, use_market_gate, cooldown_sessions=5):
    eligible = frame[(frame.stop_ok_8pct == 1)].copy()
    if use_market_gate:
        eligible = eligible[eligible.market_bull == 1]
    eligible = eligible.sort_values(["date", "trigger_time", "ma_score", "symbol"],
                                    ascending=[True, True, False, True])
    cash = STARTING_CAPITAL
    active = []
    accepted = []
    last_stop_date = {}
    skipped_capacity = skipped_open = skipped_cooldown = 0
    for _, row in eligible.iterrows():
        entry_date = str(row.date)[:10]
        still_active = []
        for trade in active:
            if str(trade["pure_exit_date"])[:10] < entry_date:
                cash += POSITION_VALUE + trade["pure_pnl_usd"]
                if trade["pure_exit_reason"] == "initial_stop":
                    last_stop_date[trade["symbol"]] = str(trade["pure_exit_date"])[:10]
            else:
                still_active.append(trade)
        active = still_active
        if any(t["symbol"] == row.symbol for t in active):
            skipped_open += 1
            continue
        if row.symbol in last_stop_date:
            before = np.datetime64(last_stop_date[row.symbol])
            now = np.datetime64(entry_date)
            if int(np.busday_count(before, now)) < cooldown_sessions:
                skipped_cooldown += 1
                continue
        if cash < POSITION_VALUE:
            skipped_capacity += 1
            continue
        cash -= POSITION_VALUE
        record = row.to_dict()
        active.append(record)
        accepted.append(record)
    for trade in active:
        cash += POSITION_VALUE + trade["pure_pnl_usd"]
    accepted_frame = pd.DataFrame(accepted)
    return {
        "market_gate": bool(use_market_gate), "accepted": int(len(accepted_frame)),
        "cooldown_sessions": int(cooldown_sessions),
        "skipped_capacity": int(skipped_capacity), "skipped_symbol_open": int(skipped_open),
        "skipped_cooldown": int(skipped_cooldown), "ending_equity_usd": float(cash),
        "total_return_pct": float((cash / STARTING_CAPITAL - 1) * 100),
        "trade_metrics": outcome_metrics(accepted_frame) if len(accepted_frame) else {"n": 0},
    }


def self_test():
    index = pd.date_range("2026-01-05 09:30", periods=390, freq="1min", tz=NY)
    frame = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                          "close": 100.0, "volume": 1000.0}, index=index)
    frame.loc[frame.index < pd.Timestamp("2026-01-05 10:00", tz=NY), "high"] = 101.0
    frame.loc[frame.index < pd.Timestamp("2026-01-05 10:00", tz=NY), "low"] = 99.0
    frame.loc[pd.Timestamp("2026-01-05 10:05", tz=NY), ["open", "high", "low", "close"]] = [100.5, 101.2, 100.4, 101.1]
    signal = opening_range_signal(frame, "2026-01-05")
    assert signal and signal["orh_triggered"] == 1
    assert abs(signal["entry_price"] - 101.0) < 1e-9
    assert abs(signal["initial_stop"] - 99.0) < 1e-9
    daily_index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    daily = pd.DataFrame({"open": [100, 102, 104], "high": [102, 104, 150],
                          "low": [99, 100.5, 103], "close": [102, 103.6, 146],
                          "volume": [1, 1, 1]}, index=daily_index)
    outcome = simulate_trade(frame, daily, "2026-01-05", 101.0, 99.0, False)
    assert outcome.exit_reason == "target"
    assert abs(outcome.return_pct - 45.0) < 1e-8

    stopped = frame.copy()
    stopped.loc[pd.Timestamp("2026-01-05 10:06", tz=NY), ["open", "high", "low", "close"]] = [100.5, 100.6, 98.5, 99.0]
    stop_outcome = simulate_trade(stopped, daily, "2026-01-05", 101.0, 99.0, False)
    assert stop_outcome.exit_reason == "initial_stop"
    assert abs(stop_outcome.return_pct - ((99.0 / 101.0 - 1) * 100)) < 1e-8

    be_daily = daily.copy()
    be_daily.loc[pd.Timestamp("2026-01-06"), ["high", "low", "close"]] = [104.0, 100.0, 104.0]
    be_daily.loc[pd.Timestamp("2026-01-07"), ["open", "high", "low", "close"]] = [102.0, 103.0, 100.5, 101.0]
    be_outcome = simulate_trade(frame, be_daily, "2026-01-05", 101.0, 99.0, False)
    assert be_outcome.exit_reason == "breakeven_stop"
    assert abs(be_outcome.return_pct) < 1e-8

    partial_outcome = simulate_trade(frame, daily, "2026-01-05", 101.0, 99.0, True)
    assert partial_outcome.exit_reason == "target"
    expected_partial_return = PARTIAL_FRACTION * PARTIAL_TRIGGER_PCT + (1 - PARTIAL_FRACTION) * TARGET_PCT
    assert abs(partial_outcome.return_pct - expected_partial_return * 100) < 1e-8
    print("self-test passed")


def main():
    import screener

    OUT.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    daily_frames = {symbol: cache_frame(rows) for symbol, rows in cache.get("bars", {}).items()}
    daily_frames = {symbol: frame for symbol, frame in daily_frames.items() if frame is not None}
    client = AlpacaClient()

    calendar_start = (SIGNAL_START - pd.Timedelta(days=10)).tz_localize(NY).tz_convert(UTC).isoformat()
    calendar_end = (DATA_END + pd.Timedelta(days=2)).tz_localize(NY).tz_convert(UTC).isoformat()
    references = client.bars(["SPY", "ONEQ"], "1Day", calendar_start, calendar_end, batch_size=2)
    spy = alpaca_frame(references.get("SPY"))
    oneq = alpaca_frame(references.get("ONEQ"))
    if spy is None or oneq is None:
        raise RuntimeError("Could not obtain SPY/ONEQ reference bars")
    sessions = [timestamp.tz_localize(None).normalize() for timestamp in spy.index
                if SIGNAL_START <= timestamp.tz_localize(None).normalize() <= SIGNAL_END]
    regimes = market_regime(oneq.tz_localize(None), sessions)
    print(f"signal sessions={len(sessions)} {sessions[0].date()} to {sessions[-1].date()} symbols={len(daily_frames)}", flush=True)

    candidates = []
    symbols_ok = 0
    for number, (symbol, daily) in enumerate(sorted(daily_frames.items()), 1):
        if len(daily) < 60:
            continue
        symbols_ok += 1
        for session in sessions:
            history = prior_slice(daily, session)
            if len(history) < 60:
                continue
            passed, metrics, criteria = screener.compute_screen(history)
            if not passed:
                continue
            architecture = ma_features(history)
            if architecture is None:
                continue
            row = {"date": session.strftime("%Y-%m-%d"), "symbol": symbol, "orb_pass": 1,
                   "market_bull": regimes.get(session.strftime("%Y-%m-%d"), np.nan)}
            row.update({key: value for key, value in metrics.items() if np.isscalar(value)})
            row.update({f"crit_{key}": int(bool(value)) for key, value in criteria.items()})
            row.update(architecture)
            candidates.append(row)
        if number % 500 == 0:
            print(f"daily {number}/{len(daily_frames)} orb_pairs={len(candidates)}", flush=True)
    daily_candidates = pd.DataFrame(candidates)
    daily_candidates.to_csv(OUT / "daily_candidates.csv", index=False)
    ma_candidates = daily_candidates[daily_candidates.ma_go == 1].copy()
    print(f"ORB={len(daily_candidates)} MA={len(ma_candidates)}", flush=True)

    earnings = {}
    symbols = sorted(ma_candidates.symbol.unique())
    with ThreadPoolExecutor(max_workers=EARN_WORKERS) as executor:
        futures = [executor.submit(fetch_earnings, symbol) for symbol in symbols]
        for number, future in enumerate(as_completed(futures), 1):
            symbol, frame = future.result()
            earnings[symbol] = frame
            if number % 100 == 0 or number == len(futures):
                print(f"earnings {number}/{len(futures)}", flush=True)
    filtered_rows = []
    for _, source in ma_candidates.iterrows():
        row = source.to_dict()
        row.update(earnings_for_date(earnings.get(source.symbol), source.date))
        filtered_rows.append(row)
    filtered = pd.DataFrame(filtered_rows)
    filtered["earnings_rule_base"] = ((filtered.earnings_history_known == 1) &
                                       (filtered.prev_earnings_pass == 1)).astype(int)
    filtered["earnings_rule_strict"] = ((filtered.earnings_rule_base == 1) &
                                         (filtered.next_earnings_known == 1) &
                                         (filtered.earnings_blackout_7d == 0)).astype(int)
    filtered.to_csv(OUT / "earnings_candidates.csv", index=False)
    watchlist = filtered[filtered.earnings_rule_strict == 1].copy()
    print(f"strict watchlist pairs={len(watchlist)} symbols={watchlist.symbol.nunique()}", flush=True)

    output_rows = []
    intraday_failures = []
    grouped = list(watchlist.groupby("date"))
    for number, (date, group) in enumerate(grouped, 1):
        symbols = sorted(group.symbol.unique())
        start, end = session_bounds(date)
        try:
            raw = client.bars(symbols, "1Min", start, end, batch_size=INTRA_BATCH)
        except Exception as exc:
            intraday_failures.append({"date": date, "symbols": symbols, "error": str(exc)})
            continue
        frames = {symbol: alpaca_frame(raw.get(symbol)) for symbol in symbols}
        for _, source in group.iterrows():
            row = source.to_dict()
            minute_frame = frames.get(source.symbol)
            signal = opening_range_signal(minute_frame, date)
            if signal is None:
                row.update(orh_data_known=0, orh_triggered=np.nan)
                output_rows.append(row)
                continue
            row.update(orh_data_known=1)
            row.update(signal)
            if signal.get("orh_triggered") != 1:
                output_rows.append(row)
                continue
            row["stop_ok_8pct"] = int(signal["stop_pct"] <= MAX_STOP_PCT)
            day_minutes = minute_frame[minute_frame.index.date == pd.Timestamp(date).date()]
            daily = daily_frames[source.symbol]
            pure = simulate_trade(day_minutes, daily, date, signal["entry_price"], signal["initial_stop"], False)
            partial = simulate_trade(day_minutes, daily, date, signal["entry_price"], signal["initial_stop"], True)
            for prefix, outcome in [("pure", pure), ("partial", partial)]:
                for key, value in outcome.__dict__.items():
                    row[f"{prefix}_{key}"] = value
            output_rows.append(row)
        if number % 20 == 0 or number == len(grouped):
            print(f"intraday {number}/{len(grouped)} rows={len(output_rows)} api_calls={client.calls}", flush=True)

    results = pd.DataFrame(output_rows)
    results.to_csv(OUT / "full_orh_results.csv", index=False)
    triggered = results[results.orh_triggered == 1].copy()
    primary = triggered[triggered.stop_ok_8pct == 1].copy()
    bull = primary[primary.market_bull == 1].copy()
    partial_primary = primary.copy()
    partial_primary["pure_return_pct"] = partial_primary.partial_return_pct
    partial_primary["pure_pnl_usd"] = partial_primary.partial_pnl_usd
    partial_primary["pure_exit_reason"] = partial_primary.partial_exit_reason
    partial_primary["pure_open_at_end"] = partial_primary.partial_open_at_end
    partial_primary["pure_holding_sessions"] = partial_primary.partial_holding_sessions
    partial_primary["pure_ambiguity_days"] = partial_primary.partial_ambiguity_days

    summary = {
        "period": {"signal_start": SIGNAL_START.strftime("%Y-%m-%d"),
                   "signal_end": SIGNAL_END.strftime("%Y-%m-%d"),
                   "data_end": DATA_END.strftime("%Y-%m-%d"), "sessions": len(sessions)},
        "rules": {"orh": "first break after 10:00 ET above 09:30-10:00 high",
                  "initial_stop": "09:30-10:00 low", "max_stop_pct": MAX_STOP_PCT * 100,
                  "breakeven": f"next session after daily close >= {BE_TRIGGER_PCT * 100:.1f}%",
                  "target_pct": TARGET_PCT * 100, "position_value_usd": POSITION_VALUE,
                  "starting_capital_usd": STARTING_CAPITAL,
                  "market_gate": "ONEQ EMA10 > EMA20 reported on and off",
                  "partial_variant": f"sell {PARTIAL_FRACTION * 100:.0f}% at +{PARTIAL_TRIGGER_PCT * 100:.0f}%"},
        "coverage": {"universe": int(cache.get("universe", 0)), "daily_symbols_ok": symbols_ok,
                     "orb_pairs": len(daily_candidates), "orb_ma_pairs": len(ma_candidates),
                     "positive_earnings_pairs": int(filtered.earnings_rule_base.sum()),
                     "strict_watchlist_pairs": len(watchlist),
                     "orh_data_known": int((results.orh_data_known == 1).sum()),
                     "orh_triggered": len(triggered), "stop_ok_8pct": len(primary),
                     "market_bull_primary": len(bull), "intraday_failures": len(intraday_failures),
                     "alpaca_api_calls": client.calls},
        "outcomes": {"primary_gate_off_pure": outcome_metrics(primary),
                     "primary_gate_on_pure": outcome_metrics(bull),
                     "all_stop_widths_gate_off_pure": outcome_metrics(triggered),
                     "primary_gate_off_optional_partial": outcome_metrics(partial_primary,
                                                                           "pure_return_pct",
                                                                           "pure_pnl_usd")},
        "portfolio": {
            "gate_off_no_cooldown": portfolio_simulation(primary, False, 0),
            "gate_on_no_cooldown": portfolio_simulation(primary, True, 0),
            "gate_off_5_session_cooldown": portfolio_simulation(primary, False, 5),
            "gate_on_5_session_cooldown": portfolio_simulation(primary, True, 5),
        },
        "data_limitations": [
            "Earnings history is current yfinance data, not guaranteed point-in-time TradingView estimates.",
            "Later stop/target same-day touches use conservative stop-first ordering.",
            "Daily and intraday data use Alpaca IEX rather than consolidated SIP.",
            "The current listed universe creates survivorship bias.",
        ],
    }
    (OUT / "intraday_failures.json").write_text(json.dumps(iso_value(intraday_failures), indent=2))
    (OUT / "summary.json").write_text(json.dumps(iso_value(summary), indent=2))
    print(json.dumps(iso_value(summary), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()

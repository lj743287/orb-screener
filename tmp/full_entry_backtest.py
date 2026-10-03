#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bars_cache import CACHE

OUT = Path("tmp/full_backtest_out")
OUT.mkdir(parents=True, exist_ok=True)

PCT_THRESH = 7.5
QUIET_DAYS = 12
CLEAN_DAYS = 20
TI_MIN = 1.05
PARTIAL_DAY = 3
PARTIAL_PCT = 0.50
MIN_WARMUP = 65

def cache_frame(symbol):
    rows = CACHE.get(symbol, newest_first=False)
    if not rows:
        return None
    d = pd.DataFrame(rows).rename(columns={"d":"date","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    if "date" not in d.columns:
        return None
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["date","open","high","low","close","volume"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

def add_signal_columns(d):
    x = d.copy()
    x["prev_close"] = x["close"].shift(1)
    x["prev_vol"] = x["volume"].shift(1)
    x["pct_chg"] = np.where(x["prev_close"] != 0, 100.0*(x["close"]-x["prev_close"])/x["prev_close"], np.nan)
    rng = x["high"] - x["low"]
    x["close_pos_top"] = np.where(rng > 0, (x["high"]-x["close"])/rng, 0.0)
    x["close_pos_bottom"] = np.where(rng > 0, (x["close"]-x["low"])/rng, 0.0)
    x["vol_ok"] = x["volume"] > x["prev_vol"]
    x["bull_flag"] = (x["pct_chg"] >= PCT_THRESH) & x["vol_ok"] & (x["close_pos_top"] <= 0.30)
    x["bear_raw"] = (x["pct_chg"] <= -PCT_THRESH) & x["vol_ok"] & (x["close_pos_bottom"] <= 0.30)
    x["sma7"] = x["close"].rolling(7, min_periods=7).mean()
    x["sma10"] = x["close"].rolling(10, min_periods=10).mean()
    x["sma20"] = x["close"].rolling(20, min_periods=20).mean()
    x["sma65"] = x["close"].rolling(65, min_periods=65).mean()
    x["ti65"] = x["sma7"] / x["sma65"]
    x["greens_before"] = x["bull_flag"].astype(float).rolling(QUIET_DAYS, min_periods=1).sum().shift(1).fillna(0.0)
    x["reds_before"] = x["bear_raw"].astype(float).rolling(CLEAN_DAYS, min_periods=1).sum().shift(1).fillna(0.0)
    x["quiet_ok"] = x["greens_before"] == 0.0
    x["clean_ok"] = x["reds_before"] == 0.0
    x["ti_ok"] = x["ti65"] >= TI_MIN
    x["buy_signal"] = x["bull_flag"] & x["quiet_ok"] & x["clean_ok"] & x["ti_ok"]
    return x

def stop_fill(open_px, low_px, stop_px):
    if not np.isfinite(stop_px):
        return None
    if open_px <= stop_px:
        return float(open_px)
    if low_px <= stop_px:
        return float(stop_px)
    return None

def horizon_stats(x, entry_i, entry_px):
    out = {}
    for n in (3,5,10,20):
        end_i = min(entry_i+n-1, len(x)-1)
        f = x.iloc[entry_i:end_i+1]
        complete = len(f) == n
        out[f"full_{n}d"] = int(complete)
        out[f"ret_{n}d_pct"] = (float(f.iloc[-1]["close"])/entry_px-1.0)*100.0 if complete else np.nan
        out[f"mfe_{n}d_pct"] = (float(f["high"].max())/entry_px-1.0)*100.0 if complete else np.nan
        out[f"mae_{n}d_pct"] = (float(f["low"].min())/entry_px-1.0)*100.0 if complete else np.nan
    return out

def simulate_trade(x, signal_i, ma_len):
    entry_i = signal_i + 1
    if entry_i >= len(x):
        return None
    sig = x.iloc[signal_i]
    entry = x.iloc[entry_i]
    entry_px = float(entry["open"])
    sig_close = float(sig["close"])
    sig_low = float(sig["low"])
    gap_below_close = entry_px < sig_close
    initial_stop = sig_low if gap_below_close else sig_close

    if entry_px <= sig_low:
        gross = 0.0
        return {
            "entry_date":entry["date"],"entry_price":entry_px,"signal_close":sig_close,"signal_low":sig_low,
            "initial_stop":initial_stop,"gap_below_signal_close":int(gap_below_close),"gap_below_signal_low":1,
            "partial_reached":0,"partial_price":np.nan,"exit_date":entry["date"],"exit_price_remainder":entry_px,
            "exit_reason":"open_below_signal_low","hold_days":1,"gross_return_pct":gross,
            "net20bps_return_pct":gross-0.20,"initial_risk_pct":np.nan,"r_multiple":np.nan,
        }

    initial_risk_pct = 100.0*(entry_px-initial_stop)/entry_px if entry_px > initial_stop else np.nan
    partial_i = entry_i + PARTIAL_DAY - 1
    last_i = len(x)-1
    pre_partial_end = min(partial_i, last_i)

    for j in range(entry_i, pre_partial_end+1):
        row = x.iloc[j]
        fill = stop_fill(float(row["open"]), float(row["low"]), initial_stop)
        if fill is not None:
            gross = (fill/entry_px-1.0)*100.0
            return {
                "entry_date":entry["date"],"entry_price":entry_px,"signal_close":sig_close,"signal_low":sig_low,
                "initial_stop":initial_stop,"gap_below_signal_close":int(gap_below_close),
                "gap_below_signal_low":int(entry_px <= sig_low),"partial_reached":0,"partial_price":np.nan,
                "exit_date":row["date"],"exit_price_remainder":fill,"exit_reason":"initial_stop",
                "hold_days":j-entry_i+1,"gross_return_pct":gross,"net20bps_return_pct":gross-0.20,
                "initial_risk_pct":initial_risk_pct,
                "r_multiple":gross/initial_risk_pct if np.isfinite(initial_risk_pct) and initial_risk_pct > 0 else np.nan,
            }

    if partial_i > last_i:
        final = x.iloc[last_i]
        gross = (float(final["close"])/entry_px-1.0)*100.0
        return {
            "entry_date":entry["date"],"entry_price":entry_px,"signal_close":sig_close,"signal_low":sig_low,
            "initial_stop":initial_stop,"gap_below_signal_close":int(gap_below_close),
            "gap_below_signal_low":int(entry_px <= sig_low),"partial_reached":0,"partial_price":np.nan,
            "exit_date":final["date"],"exit_price_remainder":float(final["close"]),
            "exit_reason":"open_at_data_end_before_partial","hold_days":last_i-entry_i+1,
            "gross_return_pct":gross,"net20bps_return_pct":gross-0.20,"initial_risk_pct":initial_risk_pct,
            "r_multiple":gross/initial_risk_pct if np.isfinite(initial_risk_pct) and initial_risk_pct > 0 else np.nan,
        }

    partial_row = x.iloc[partial_i]
    partial_px = float(partial_row["close"])
    partial_ret = (partial_px/entry_px-1.0)*100.0
    ma_col = "sma10" if ma_len == 10 else "sma20"

    for j in range(partial_i+1, len(x)):
        row = x.iloc[j]
        stop_px = float(x.iloc[j-1][ma_col])
        if not np.isfinite(stop_px):
            continue
        fill = stop_fill(float(row["open"]), float(row["low"]), stop_px)
        if fill is not None:
            rem_ret = (fill/entry_px-1.0)*100.0
            gross = PARTIAL_PCT*partial_ret + (1.0-PARTIAL_PCT)*rem_ret
            return {
                "entry_date":entry["date"],"entry_price":entry_px,"signal_close":sig_close,"signal_low":sig_low,
                "initial_stop":initial_stop,"gap_below_signal_close":int(gap_below_close),
                "gap_below_signal_low":int(entry_px <= sig_low),"partial_reached":1,"partial_price":partial_px,
                "exit_date":row["date"],"exit_price_remainder":fill,"exit_reason":f"{ma_len}ma_stop",
                "hold_days":j-entry_i+1,"gross_return_pct":gross,"net20bps_return_pct":gross-0.20,
                "initial_risk_pct":initial_risk_pct,
                "r_multiple":gross/initial_risk_pct if np.isfinite(initial_risk_pct) and initial_risk_pct > 0 else np.nan,
            }

    final = x.iloc[-1]
    rem_ret = (float(final["close"])/entry_px-1.0)*100.0
    gross = PARTIAL_PCT*partial_ret + (1.0-PARTIAL_PCT)*rem_ret
    return {
        "entry_date":entry["date"],"entry_price":entry_px,"signal_close":sig_close,"signal_low":sig_low,
        "initial_stop":initial_stop,"gap_below_signal_close":int(gap_below_close),
        "gap_below_signal_low":int(entry_px <= sig_low),"partial_reached":1,"partial_price":partial_px,
        "exit_date":final["date"],"exit_price_remainder":float(final["close"]),"exit_reason":"open_at_data_end",
        "hold_days":len(x)-entry_i,"gross_return_pct":gross,"net20bps_return_pct":gross-0.20,
        "initial_risk_pct":initial_risk_pct,
        "r_multiple":gross/initial_risk_pct if np.isfinite(initial_risk_pct) and initial_risk_pct > 0 else np.nan,
    }

def safe_float(v):
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except Exception:
        return None

def metric_block(df, return_col="gross_return_pct"):
    if df.empty:
        return {"n":0}
    r = pd.to_numeric(df[return_col], errors="coerce").dropna()
    if r.empty:
        return {"n":int(len(df))}
    wins = r[r > 0]
    losses = r[r < 0]
    gain_sum = wins.sum()
    loss_sum = abs(losses.sum())
    return {
        "n":int(len(r)),
        "win_rate":float((r > 0).mean()),
        "mean_return_pct":float(r.mean()),
        "median_return_pct":float(r.median()),
        "avg_winner_pct":float(wins.mean()) if len(wins) else None,
        "avg_loser_pct":float(losses.mean()) if len(losses) else None,
        "profit_factor":float(gain_sum/loss_sum) if loss_sum > 0 else None,
        "p10_return_pct":float(r.quantile(0.10)),
        "p90_return_pct":float(r.quantile(0.90)),
        "median_hold_days":float(pd.to_numeric(df["hold_days"], errors="coerce").median()),
        "initial_stop_before_partial_rate":float((df["exit_reason"] == "initial_stop").mean()),
        "partial_reached_rate":float(pd.to_numeric(df["partial_reached"], errors="coerce").mean()),
        "gap_below_signal_close_rate":float(pd.to_numeric(df["gap_below_signal_close"], errors="coerce").mean()),
    }

def followthrough_block(signals):
    out = {"n":int(len(signals))}
    for n in (3,5,10,20):
        g = signals[signals[f"full_{n}d"] == 1].copy()
        out[f"n_{n}d"] = int(len(g))
        if g.empty:
            continue
        ret = pd.to_numeric(g[f"ret_{n}d_pct"], errors="coerce")
        mfe = pd.to_numeric(g[f"mfe_{n}d_pct"], errors="coerce")
        mae = pd.to_numeric(g[f"mae_{n}d_pct"], errors="coerce")
        out[f"median_ret_{n}d_pct"] = safe_float(ret.median())
        out[f"mean_ret_{n}d_pct"] = safe_float(ret.mean())
        out[f"positive_ret_{n}d_rate"] = safe_float((ret > 0).mean())
        out[f"median_mfe_{n}d_pct"] = safe_float(mfe.median())
        out[f"median_mae_{n}d_pct"] = safe_float(mae.median())
    return out

def main():
    if not CACHE.available:
        raise RuntimeError("Alpaca cache not available")

    symbols = CACHE.symbols
    print(f"Cache universe: {len(symbols)} symbols", flush=True)
    prepared = {}
    latest_dates = []

    for i, sym in enumerate(symbols, 1):
        d = cache_frame(sym)
        if d is None or len(d) < MIN_WARMUP + 5:
            continue
        x = add_signal_columns(d)
        prepared[sym] = x
        latest_dates.append(x["date"].max())
        if i % 1000 == 0:
            print(f"Prepared {i}/{len(symbols)}", flush=True)

    if not latest_dates:
        raise RuntimeError("No usable daily histories")

    test_end = pd.Timestamp(max(latest_dates)).normalize()
    test_start = (test_end - pd.DateOffset(years=1)).normalize()
    print(f"Test period: {test_start.date()} to {test_end.date()}", flush=True)

    signal_rows = []
    trade_rows_10 = []
    trade_rows_20 = []

    for i, (sym, x) in enumerate(prepared.items(), 1):
        mask = x["buy_signal"] & (x["date"] >= test_start) & (x["date"] <= test_end)
        sig_idxs = list(np.flatnonzero(mask.to_numpy()))
        next_free_10 = -1
        next_free_20 = -1

        for si in sig_idxs:
            if si + 1 >= len(x):
                continue
            sig = x.iloc[si]
            entry_i = si + 1
            entry = x.iloc[entry_i]
            entry_px = float(entry["open"])
            row = {
                "symbol":sym,"signal_date":sig["date"],"signal_close":float(sig["close"]),
                "signal_low":float(sig["low"]),"signal_pct_chg":float(sig["pct_chg"]),
                "signal_volume":float(sig["volume"]),"prior_volume":float(sig["prev_vol"]),
                "volume_ratio":float(sig["volume"]/sig["prev_vol"]) if sig["prev_vol"] > 0 else np.nan,
                "close_pos_top_pct":float(sig["close_pos_top"]*100.0),"ti65":float(sig["ti65"]),
                "entry_date":entry["date"],"entry_open":entry_px,
                "next_open_gap_vs_signal_close_pct":(entry_px/float(sig["close"])-1.0)*100.0,
            }
            row.update(horizon_stats(x, entry_i, entry_px))
            signal_rows.append(row)

            if si >= next_free_10:
                tr10 = simulate_trade(x, si, 10)
                if tr10 is not None:
                    tr10.update({"symbol":sym,"signal_date":sig["date"],"ma_len":10})
                    trade_rows_10.append(tr10)
                    ed = pd.Timestamp(tr10["exit_date"])
                    after = x.index[x["date"] > ed]
                    next_free_10 = int(after[0]) if len(after) else len(x)

            if si >= next_free_20:
                tr20 = simulate_trade(x, si, 20)
                if tr20 is not None:
                    tr20.update({"symbol":sym,"signal_date":sig["date"],"ma_len":20})
                    trade_rows_20.append(tr20)
                    ed = pd.Timestamp(tr20["exit_date"])
                    after = x.index[x["date"] > ed]
                    next_free_20 = int(after[0]) if len(after) else len(x)

        if i % 500 == 0:
            print(f"Scanned {i}/{len(prepared)} | signals {len(signal_rows)} | 10MA {len(trade_rows_10)} | 20MA {len(trade_rows_20)}", flush=True)

    signals = pd.DataFrame(signal_rows)
    t10 = pd.DataFrame(trade_rows_10)
    t20 = pd.DataFrame(trade_rows_20)
    signals.to_csv(OUT/"stockbee_signals.csv", index=False)
    t10.to_csv(OUT/"strategy_10ma_trades.csv", index=False)
    t20.to_csv(OUT/"strategy_20ma_trades.csv", index=False)

    summary = {
        "period":{
            "start":str(test_start.date()),"end":str(test_end.date()),
            "data_source":"Alpaca IEX split-adjusted completed 1D bars",
            "universe":"current NASDAQ/NYSE common-stock universe from existing orb-screener cache",
            "survivorship_bias_warning":True,
        },
        "rules":{
            "burst_pct_min":PCT_THRESH,"volume_gt_prior_day":True,"close_in_top_pct_of_range":30,
            "quiet_days_no_green":QUIET_DAYS,"clean_days_no_red":CLEAN_DAYS,"ti65_min":TI_MIN,
            "entry":"next trading day open",
            "initial_stop":"signal-day close; if next open below it, signal-day low",
            "partial":"50% at Day 3 close if initial stop not hit",
            "remainder":"selected MA from last completed day, active as stop next session",
            "stop_gap_fill":"next open if gap below stop, otherwise stop price",
            "net_cost_sensitivity":"gross return less 0.20 percentage points per trade",
        },
        "coverage":{
            "cache_symbols":int(len(symbols)),"usable_symbols":int(len(prepared)),
            "signals":int(len(signals)),"trades_10ma":int(len(t10)),"trades_20ma":int(len(t20)),
        },
        "signal_followthrough":followthrough_block(signals) if not signals.empty else {"n":0},
        "strategy_10ma_gross":metric_block(t10,"gross_return_pct") if not t10.empty else {"n":0},
        "strategy_10ma_net20bps":metric_block(t10,"net20bps_return_pct") if not t10.empty else {"n":0},
        "strategy_20ma_gross":metric_block(t20,"gross_return_pct") if not t20.empty else {"n":0},
        "strategy_20ma_net20bps":metric_block(t20,"net20bps_return_pct") if not t20.empty else {"n":0},
    }

    if not signals.empty:
        for label, lo, hi in [("7_5_to_10",7.5,10.0),("10_to_15",10.0,15.0),("15_plus",15.0,np.inf)]:
            g = signals[(signals["signal_pct_chg"] >= lo) & (signals["signal_pct_chg"] < hi)]
            summary.setdefault("signal_burst_bands", {})[label] = followthrough_block(g)
        for label, lo, hi in [("under_1_5x",0.0,1.5),("1_5_to_2x",1.5,2.0),("2x_plus",2.0,np.inf)]:
            g = signals[(signals["volume_ratio"] >= lo) & (signals["volume_ratio"] < hi)]
            summary.setdefault("signal_volume_bands", {})[label] = followthrough_block(g)

    with open(OUT/"summary.json","w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str), flush=True)

if __name__ == "__main__":
    main()

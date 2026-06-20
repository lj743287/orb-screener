#!/usr/bin/env python3
"""
Overnight ORB-continuation screener.
Scans the NYSE/NASDAQ common-stock universe via Twelve Data, applies Leighton's
validated entry parameters, and writes a watchlist (CSV + a GitHub Pages page).

Run by GitHub Actions after the US close. Network calls are throttled and
auto-retry on Twelve Data's per-minute rate limit so it can run unattended.

Env:
  TWELVE_DATA_KEY   (required)   your Twelve Data API key
  MAX_SYMBOLS       (optional)   cap the universe to protect daily credits
  THROTTLE_SEC      (optional)   seconds between calls (default 1.2)
"""
import os, io, time, json, datetime as dt
import requests
import numpy as np
import pandas as pd

API_KEY    = os.environ.get("TWELVE_DATA_KEY", "")
MAX_SYMBOLS= int(os.environ.get("MAX_SYMBOLS", "0"))      # 0 = no cap
THROTTLE   = float(os.environ.get("THROTTLE_SEC", "1.2"))
REGIME_SYM = os.environ.get("REGIME_SYMBOL", "ONEQ")      # Nasdaq Composite proxy
OUT_DIR    = "output"
DOCS_DIR   = "docs"

# ------- tunable screen parameters (Leighton's validated set) -------
P = dict(ADR_MAX=6.0, RUNUP_MIN=45.0, RUNUP_MAX=100.0, PRICE_MIN=5.0,
         BASE_MIN=8, RUNUP_LB=60, MA_TOL=3.0)


# ============================ universe ============================
def load_universe():
    """NASDAQ Trader listing files -> common-stock symbols (ex ETFs/test issues)."""
    urls = ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"]
    syms = []
    for u in urls:
        txt = requests.get(u, timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|", dtype=str)
        first = df.columns[0]
        df = df[~df[first].astype(str).str.contains("File Creation Time", na=False)]
        etf_col  = "ETF" if "ETF" in df.columns else None
        test_col = "Test Issue" if "Test Issue" in df.columns else None
        sym_col  = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        if etf_col:  df = df[df[etf_col] != "Y"]
        if test_col: df = df[df[test_col] != "Y"]
        s = df[sym_col].dropna().astype(str).str.strip()
        s = s[(s.str.len() > 0) & (s.str.upper() != "NAN")]
        s = s[~s.str.contains(r"[.$^]", regex=True, na=False)]   # drop warrants/units/pfd
        syms += s.tolist()
    return sorted({x for x in syms if isinstance(x, str) and x})


# ============================ data ============================
def td_daily(symbol, outputsize=160, tries=6):
    params = dict(symbol=symbol, interval="1day", outputsize=outputsize,
                  order="ASC", timezone="America/New_York", apikey=API_KEY)
    for _ in range(tries):
        j = requests.get("https://api.twelvedata.com/time_series",
                         params=params, timeout=30).json()
        if "values" in j:
            d = pd.DataFrame(j["values"])
            d["datetime"] = pd.to_datetime(d["datetime"])
            d = d.set_index("datetime").sort_index()
            for c in ["open", "high", "low", "close", "volume"]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            time.sleep(THROTTLE)
            return d[["open", "high", "low", "close", "volume"]]
        msg = str(j.get("message", "")).lower()
        if any(k in msg for k in ("credit", "run out", "limit")):
            time.sleep(61); continue
        return None    # bad symbol / no data
    return None


# ============================ screen (pure, testable) ============================
def compute_screen(d, regime_ok, p=P):
    """Return (passed: bool, metrics: dict) for one symbol's daily frame."""
    if d is None or len(d) < 60:
        return False, {}
    c, h, l = d["close"], d["high"], d["low"]
    sma10, sma20, sma50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    adr = 100.0 * ((h / l).rolling(20).mean().iloc[-1] - 1.0)
    price = float(c.iloc[-1])

    run_lb = min(p["RUNUP_LB"], len(d) - 1)
    run_low  = float(l.iloc[-run_lb:].min())
    run_high = float(h.iloc[-(p["BASE_MIN"] + 5):].max())
    runup = (run_high - run_low) / run_low * 100.0 if run_low > 0 else np.nan

    base = d.iloc[-p["BASE_MIN"]:]
    base_high = float(base["high"].max())
    base_low  = float(base["low"].min())
    base_depth = (base_high - base_low) / base_high * 100.0 if base_high > 0 else np.nan
    rng = (h - l)
    contracting = rng.iloc[-p["BASE_MIN"]:].mean() < rng.iloc[-2*p["BASE_MIN"]:-p["BASE_MIN"]].mean()
    lows_up = l.iloc[-p["BASE_MIN"]:].min() > l.iloc[-2*p["BASE_MIN"]:-p["BASE_MIN"]].min()

    def near(ma):
        return l.iloc[-1] <= ma.iloc[-1] * (1 + p["MA_TOL"]/100) and ma.iloc[-1] > ma.iloc[-6]
    surf = near(sma10) or near(sma20) or near(sma50)

    crit = dict(
        adr   = adr < p["ADR_MAX"],
        runup = (not np.isnan(runup)) and p["RUNUP_MIN"] <= runup <= p["RUNUP_MAX"],
        price = price >= p["PRICE_MIN"],
        base  = bool(contracting) and price < base_high,
        hl    = bool(lows_up),
        surf  = bool(surf),
        regime= bool(regime_ok),
    )
    passed = all(crit.values())
    metrics = dict(price=round(price, 2), adr=round(adr, 2),
                   runup=round(float(runup), 1) if not np.isnan(runup) else None,
                   base_depth=round(float(base_depth), 1) if not np.isnan(base_depth) else None,
                   **{f"ok_{k}": v for k, v in crit.items()})
    return passed, metrics


# ============================ outputs ============================
def write_outputs(rows):
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DOCS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    df.to_csv(os.path.join(OUT_DIR, "candidates.csv"), index=False)
    cols = ["symbol", "price", "adr", "runup", "base_depth"]
    body = (df[cols].to_html(index=False, border=0)
            if len(df) else "<p>No candidates today.</p>")
    html = f"""<!doctype html><meta charset="utf-8">
<title>ORB Continuation Watchlist</title>
<style>body{{font-family:system-ui;margin:2rem;background:#0f1115;color:#e6e6e6}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem .8rem;border-bottom:1px solid #333;text-align:right}}
th:first-child,td:first-child{{text-align:left;font-weight:600}}h1{{font-size:1.2rem}}</style>
<h1>ORB Continuation Watchlist <small>({len(df)} candidates · {stamp})</small></h1>
{body}"""
    with open(os.path.join(DOCS_DIR, "index.html"), "w") as f:
        f.write(html)
    print(f"Wrote {len(df)} candidates to {OUT_DIR}/candidates.csv and {DOCS_DIR}/index.html")


# ============================ main ============================
def main():
    if not API_KEY:
        raise SystemExit("TWELVE_DATA_KEY not set")
    universe = load_universe()
    if MAX_SYMBOLS:
        universe = universe[:MAX_SYMBOLS]
    print(f"Universe: {len(universe)} symbols. Regime via {REGIME_SYM}.")

    rd = td_daily(REGIME_SYM)
    if rd is None or len(rd) < 20:
        regime_ok = False
    else:
        rc = rd["close"]
        m10, m20 = rc.rolling(10).mean(), rc.rolling(20).mean()
        regime_ok = bool(rc.iloc[-1] > m10.iloc[-1] > m10.iloc[-2] and rc.iloc[-1] > m20.iloc[-1])
    print(f"Market regime OK: {regime_ok}")

    rows = []
    for i, sym in enumerate(universe, 1):
        try:
            passed, m = compute_screen(td_daily(sym), regime_ok)
            if passed:
                rows.append(dict(symbol=sym, **{k: v for k, v in m.items() if not k.startswith("ok_")}))
                print(f"  [{i}/{len(universe)}] HIT {sym}")
        except Exception as e:
            print(f"  [{i}/{len(universe)}] {sym} err: {str(e)[:60]}")
    write_outputs(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Overnight ORB-continuation screener (NYSE/NASDAQ) via Twelve Data.
Surfaces liquid stocks in an intact uptrend that are COILING below a recent high
(set up for a breakout next session). Prints a filter funnel; sorts calmest-first.
No market-regime gate (assess the market yourself).

Env: TWELVE_DATA_KEY (req), MAX_SYMBOLS (opt cap), THROTTLE_SEC (default 1.2)
"""
import os, io, time, datetime as dt
import requests
import numpy as np
import pandas as pd

API_KEY    = os.environ.get("TWELVE_DATA_KEY", "")
MAX_SYMBOLS= int(os.environ.get("MAX_SYMBOLS", "0"))
THROTTLE   = float(os.environ.get("THROTTLE_SEC", "1.2"))
OUT_DIR, DOCS_DIR = "output", "docs"

# ---- screen parameters ----
P = dict(ADR_MIN=2.0, ADR_MAX=6.0, RUNUP_MIN=45.0, RUNUP_MAX=200.0, PRICE_MIN=5.0,
         BASE_MIN=8, RUNUP_LB=60, MA_TOL=7.0, PEAK_MIN_BACK=2, PULLBACK_MIN=0.5,
         DOLLAR_VOL_MIN=5_000_000)


def load_universe():
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
        name_col = "Security Name" if "Security Name" in df.columns else None
        sym_col  = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        if etf_col:  df = df[df[etf_col] != "Y"]
        if test_col: df = df[df[test_col] != "Y"]
        if name_col:
            bad = r"\b(?:unit|units|warrant|warrants|right|rights|preferred|depositary)\b"
            df = df[~df[name_col].astype(str).str.contains(bad, case=False, regex=True, na=False)]
        s = df[sym_col].dropna().astype(str).str.strip()
        s = s[(s.str.len() > 0) & (s.str.upper() != "NAN")]
        s = s[~s.str.contains(r"[.$^]", regex=True, na=False)]
        s = s[~((s.str.len() == 5) & (s.str[-1].isin(["U", "W", "R"])))]
        syms += s.tolist()
    return sorted({x for x in syms if isinstance(x, str) and x})


def td_daily(symbol, outputsize=260, tries=6):
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
        return None
    return None


def compute_screen(d, p=P):
    """Return (passed, metrics, crit). Pattern + quality gates only."""
    if d is None or len(d) < 60:
        return False, {}, {}
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    sma10, sma20, sma50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    adr = 100.0 * ((h / l).rolling(20).mean().iloc[-1] - 1.0)
    price = float(c.iloc[-1])
    dvol = float((c * v).rolling(20).mean().iloc[-1])          # avg $ volume, 20d

    run_lb = min(p["RUNUP_LB"], len(d) - 1)
    run_low  = float(l.iloc[-run_lb:].min())
    run_high = float(h.iloc[-(p["BASE_MIN"] + 5):].max())
    runup = (run_high - run_low) / run_low * 100.0 if run_low > 0 else np.nan

    # --- base: must be COILING below a recent high, not printing new highs ---
    win = d.iloc[-(p["BASE_MIN"] + 4):]                       # ~12-bar look
    hh = win["high"].to_numpy()
    peak_back = len(hh) - 1 - int(hh.argmax())                # bars since the window high
    recent_high = float(hh.max())
    base_low = float(win["low"].min())
    base_depth = (recent_high - base_low) / recent_high * 100.0 if recent_high > 0 else np.nan
    rng = (h - l)
    contracting = rng.iloc[-p["BASE_MIN"]:].mean() < rng.iloc[-2*p["BASE_MIN"]:-p["BASE_MIN"]].mean()
    coiling = (peak_back >= p["PEAK_MIN_BACK"]) and (price < recent_high * (1 - p["PULLBACK_MIN"]/100))
    base_ok = bool(contracting and coiling)

    lows_up = l.iloc[-p["BASE_MIN"]:].min() > l.iloc[-2*p["BASE_MIN"]:-p["BASE_MIN"]].min()

    def near(ma):
        return l.iloc[-1] <= ma.iloc[-1] * (1 + p["MA_TOL"]/100) and ma.iloc[-1] > ma.iloc[-6]
    surf = bool(near(sma10) or near(sma20) or near(sma50))
    ext10 = (price - sma10.iloc[-1]) / sma10.iloc[-1] * 100 if sma10.iloc[-1] else np.nan

    # trend gates: intermediate (above 50-MA) hard; primary (above 200-MA) soft for young names
    above50  = bool(not pd.isna(sma50.iloc[-1]) and price > sma50.iloc[-1])
    has200   = bool(not pd.isna(sma200.iloc[-1]))
    above200 = bool((not has200) or (price > sma200.iloc[-1]))

    crit = dict(
        price = bool(price >= p["PRICE_MIN"]),
        liq   = bool(dvol >= p["DOLLAR_VOL_MIN"]),
        adr   = bool(p["ADR_MIN"] <= adr < p["ADR_MAX"]),
        runup = bool((not np.isnan(runup)) and p["RUNUP_MIN"] <= runup <= p["RUNUP_MAX"]),
        base  = base_ok,
        hl    = bool(lows_up),
        surf  = surf,
        t50   = above50,
        t200  = above200,
    )
    passed = all(crit.values())
    metrics = dict(price=round(price, 2), adr=round(adr, 2),
                   runup=round(float(runup), 1) if not np.isnan(runup) else None,
                   base_depth=round(float(base_depth), 1) if not np.isnan(base_depth) else None,
                   ext10=round(float(ext10), 1) if not np.isnan(ext10) else None,
                   dvolM=round(dvol/1e6, 1),
                   trend200=("yes" if (has200 and price > sma200.iloc[-1]) else ("n/a" if not has200 else "no")))
    return passed, metrics, crit


def write_outputs(rows):
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DOCS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    if len(df) and "adr" in df.columns:
        df = df.sort_values("adr", ascending=True).reset_index(drop=True)  # calmest first
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    df.to_csv(os.path.join(OUT_DIR, "candidates.csv"), index=False)
    cols = [c for c in ["symbol","price","adr","runup","base_depth","ext10","dvolM","trend200"] if c in df.columns]
    body = (df[cols].to_html(index=False, border=0) if len(df) else "<p>No candidates today.</p>")
    html = f"""<!doctype html><meta charset="utf-8">
<title>ORB Continuation Watchlist</title>
<style>body{{font-family:system-ui;margin:2rem;background:#0f1115;color:#e6e6e6}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem .8rem;border-bottom:1px solid #333;text-align:right}}
th:first-child,td:first-child{{text-align:left;font-weight:600}}h1{{font-size:1.2rem}}small{{color:#9aa}}</style>
<h1>ORB Continuation Watchlist <small>({len(df)} candidates &middot; {stamp})</small></h1>
<p><small>Liquid, above 50-MA, coiling below a recent high &middot; sorted by ADR (calmest first). dvolM = avg $ volume (millions).</small></p>
{body}"""
    with open(os.path.join(DOCS_DIR, "index.html"), "w") as f:
        f.write(html)
    print(f"Wrote {len(df)} candidates -> {OUT_DIR}/candidates.csv, {DOCS_DIR}/index.html")


def main():
    if not API_KEY:
        raise SystemExit("TWELVE_DATA_KEY not set")
    universe = load_universe()
    if MAX_SYMBOLS:
        universe = universe[:MAX_SYMBOLS]
    print(f"Universe: {len(universe)} symbols.")

    keys = ["price", "liq", "adr", "runup", "base", "hl", "surf", "t50", "t200"]
    crits, rows = [], []
    for i, sym in enumerate(universe, 1):
        try:
            passed, m, crit = compute_screen(td_daily(sym))
            if not crit:
                continue
            crits.append(crit)
            if passed:
                rows.append(dict(symbol=sym, **m))
                print(f"  [{i}/{len(universe)}] HIT {sym}")
        except Exception as e:
            print(f"  [{i}/{len(universe)}] {sym} err: {str(e)[:60]}")

    print(f"\nFUNNEL ({len(crits)} symbols with >=60 bars):")
    for k in keys:
        print(f"  {k:>6}: {sum(1 for cr in crits if cr[k]):>5} pass individually")
    print("  stacked (in order):")
    cum = crits
    for k in keys:
        cum = [cr for cr in cum if cr[k]]
        print(f"    + {k:<6} -> {len(cum):>5}")

    write_outputs(rows)


if __name__ == "__main__":
    main()

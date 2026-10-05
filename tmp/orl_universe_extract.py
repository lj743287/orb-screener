#!/usr/bin/env python3
import gzip, json, os, sys, time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
from fetch_bars import load_universe

BASE = "https://data.alpaca.markets/v2/stocks/bars"
KEY = os.environ.get("APCA_API_KEY_ID","")
SECRET = os.environ.get("APCA_API_SECRET_KEY","")
START = os.environ.get("ORL_START","2026-03-01")
END = os.environ.get("ORL_END","2026-10-06")
BATCH = int(os.environ.get("BATCH_SIZE","50"))
RPM = int(os.environ.get("REQUESTS_PER_MIN","170"))
OUT = os.environ.get("OUT_FILE","tmp/orl_out/orl_universe.json.gz")
FEED = os.environ.get("ALPACA_FEED","iex").strip().lower() or "iex"
NY = ZoneInfo("America/New_York")
HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}

def chunks(xs,n):
    for i in range(0,len(xs),n):
        yield xs[i:i+n]

def get_page(params, tries=7):
    delay=1.0
    for attempt in range(tries):
        r=requests.get(BASE, headers=HEADERS, params=params, timeout=90)
        if r.status_code==429:
            time.sleep(max(float(r.headers.get("Retry-After","0") or 0), delay))
            delay=min(delay*2,30)
            continue
        if r.status_code>=500:
            time.sleep(delay); delay=min(delay*2,30); continue
        return r
    return r

def fetch_group(group):
    out={s:{} for s in group}
    token=None
    calls=0
    while True:
        params={"symbols":",".join(group),"timeframe":"30Min",
                "start":START+"T00:00:00Z","end":END+"T00:00:00Z",
                "limit":10000,"adjustment":"split","feed":FEED,"sort":"asc"}
        if token: params["page_token"]=token
        r=get_page(params)
        calls += 1
        if r.status_code in (401,403):
            raise PermissionError(f"{FEED} denied ({r.status_code})")
        r.raise_for_status()
        payload=r.json()
        for sym, rows in (payload.get("bars") or {}).items():
            if sym not in out: continue
            for row in rows:
                try:
                    dt=datetime.fromisoformat(str(row["t"]).replace("Z","+00:00")).astimezone(NY)
                    if dt.hour==9 and dt.minute==30:
                        d=dt.date().isoformat()
                        out[sym][d]={"orl":float(row["l"]),"orh":float(row["h"])}
                except Exception:
                    continue
        token=payload.get("next_page_token")
        if not token: break
        if RPM: time.sleep(60.0/RPM)
    return out,calls

def main():
    if not KEY or not SECRET:
        sys.exit("Missing Alpaca credentials")
    if FEED not in {"iex","sip"}:
        sys.exit("ALPACA_FEED must be iex or sip")
    print(f"Using Alpaca feed: {FEED}", flush=True)
    universe=[s for s,_ in load_universe()]
    print(f"Universe size: {len(universe)}", flush=True)
    merged={}
    failed=[]
    calls=0
    groups=list(chunks(universe,BATCH))
    for idx,g in enumerate(groups,1):
        try:
            data,c=fetch_group(g); calls+=c
            for s,days in data.items():
                if days: merged[s]=days
        except Exception as e:
            failed.append({"symbols":g,"error":str(e)})
            print(f"Batch {idx}/{len(groups)} failed: {e}", flush=True)
        if idx%10==0 or idx==len(groups):
            print(f"{idx}/{len(groups)} batches; {len(merged)} symbols; {calls} calls", flush=True)
        if RPM: time.sleep(60.0/RPM)
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    payload={"generated_utc":datetime.now(timezone.utc).isoformat(),
             "provider":"alpaca","feed":FEED,"adjustment":"split",
             "start":START,"end_exclusive":END,"universe_size":len(universe),
             "symbols_with_orl":len(merged),"api_calls":calls,"failed_batches":failed,
             "orl":merged}
    with gzip.open(OUT,"wt",encoding="utf-8") as f:
        json.dump(payload,f,separators=(",",":"))
    print(f"Wrote {OUT}", flush=True)
    if not merged:
        sys.exit("No ORL rows produced")

if __name__=="__main__":
    main()

"""Authenticated read-only Alpaca daily OHLCV export. Never falls back to IEX."""
import argparse,datetime,gzip,json,os,re,time,urllib.parse,urllib.request,urllib.error
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--symbols',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--output',default='research_export/sip_daily.json.gz');a=p.parse_args()
    symbols=sorted(set(x.strip().upper() for x in a.symbols.split(',') if x.strip()))
    if not symbols or len(symbols)>250 or any(not re.fullmatch(r'[A-Z0-9.\-]{1,12}',x) for x in symbols):raise SystemExit('Supply 1 to 250 comma-separated equity symbols.')
    start=datetime.date.fromisoformat(a.start);end=datetime.date.fromisoformat(a.end)
    if start>end or end>=datetime.datetime.now(datetime.timezone.utc).date():raise SystemExit('Use a completed historical range, ending before today (UTC).')
    key=os.environ.get('APCA_API_KEY_ID');secret=os.environ.get('APCA_API_SECRET_KEY')
    if not key or not secret:raise SystemExit('Alpaca credential environment bindings are missing.')
    out=[];missing=[]
    for pos in range(0,len(symbols),20):
        chunk=symbols[pos:pos+20];token=None;seen=set();page=0
        while True:
            q={'symbols':','.join(chunk),'timeframe':'1Day','start':a.start+'T00:00:00Z','end':a.end+'T23:59:59Z','adjustment':'split','feed':'sip','sort':'asc','limit':'10000'}
            if token:q['page_token']=token
            req=urllib.request.Request('https://data.alpaca.markets/v2/stocks/bars?'+urllib.parse.urlencode(q),headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,'Accept':'application/json'})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req,timeout=60) as r:data=json.load(r)
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (429,500,502,503,504) and attempt<4:time.sleep(min(30,2**attempt*2));continue
                    raise SystemExit('Alpaca historical SIP request failed with HTTP '+str(e.code)+'. No feed substitution was made.')
            for sym,bars in (data.get('bars') or {}).items():
                seen.add(sym)
                for b in bars:
                    # US equity daily timestamps refer to that exchange session date.
                    out.append([sym,b['t'][:10],b['o'],b['h'],b['l'],b['c'],b['v']])
            token=data.get('next_page_token');page+=1
            if not token:break
            if page>=100:raise SystemExit('Pagination guard reached; refusing an incomplete success.')
            time.sleep(.4)
        missing.extend(s for s in chunk if s not in seen)
        time.sleep(.5)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(target,'wt') as f:json.dump(sorted(out,key=lambda r:(r[0],r[1])),f,separators=(',',':'))
    target.with_suffix('.manifest.json').write_text(json.dumps({'feed':'sip','adjustment':'split','start':a.start,'end':a.end,'symbols_requested':symbols,'missing_symbols':missing,'bars':len(out)},indent=2))
    print('SIP export complete:',len(out),'bars;',len(missing),'symbols without data. No credentials were exported.')

if __name__=='__main__':main()

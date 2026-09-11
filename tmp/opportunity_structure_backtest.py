import os, re, json, subprocess, math, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = pd.Timestamp('2026-08-04')
END_DATE = pd.Timestamp('2026-09-04')
INTRADAY_START = '2026-08-01'
INTRADAY_END = '2026-09-12'

TRADES = [
 {'Symbol': 'CTRN', 'Entry date': '2026-08-03', 'Avg entry': 69.43, 'Realized $': 229},
 {'Symbol': 'GO', 'Entry date': '2026-08-03', 'Avg entry': 9.65, 'Realized $': 191},
 {'Symbol': 'LIND', 'Entry date': '2026-08-03', 'Avg entry': 32.07, 'Realized $': 761},
 {'Symbol': 'AFRM', 'Entry date': '2026-08-04', 'Avg entry': 79.51, 'Realized $': -17},
 {'Symbol': 'LQDA', 'Entry date': '2026-08-04', 'Avg entry': 87.4, 'Realized $': 339},
 {'Symbol': 'UFI', 'Entry date': '2026-08-04', 'Avg entry': 6.77, 'Realized $': 23},
 {'Symbol': 'ACAD', 'Entry date': '2026-08-05', 'Avg entry': 28.64, 'Realized $': 15},
 {'Symbol': 'OSPN', 'Entry date': '2026-08-05', 'Avg entry': 16.72, 'Realized $': -411},
 {'Symbol': 'SATL', 'Entry date': '2026-08-06', 'Avg entry': 5.06, 'Realized $': 326},
 {'Symbol': 'SDGR', 'Entry date': '2026-08-06', 'Avg entry': 17.84, 'Realized $': -12},
 {'Symbol': 'LMT', 'Entry date': '2026-08-10', 'Avg entry': 604.1, 'Realized $': -562},
 {'Symbol': 'PBF', 'Entry date': '2026-08-11', 'Avg entry': 69.09, 'Realized $': 578},
 {'Symbol': 'XNCR', 'Entry date': '2026-08-11', 'Avg entry': 22.01, 'Realized $': 6063},
 {'Symbol': 'KNSA', 'Entry date': '2026-08-12', 'Avg entry': 79.0, 'Realized $': -347},
 {'Symbol': 'OTLY', 'Entry date': '2026-08-13', 'Avg entry': 13.73, 'Realized $': 1935},
 {'Symbol': 'OII', 'Entry date': '2026-08-14', 'Avg entry': 52.72, 'Realized $': -108},
 {'Symbol': 'OPHC', 'Entry date': '2026-08-14', 'Avg entry': 9.39, 'Realized $': 340},
 {'Symbol': 'RDNT', 'Entry date': '2026-08-20', 'Avg entry': 77.75, 'Realized $': -600},
 {'Symbol': 'MATV', 'Entry date': '2026-08-21', 'Avg entry': 12.44, 'Realized $': -674},
 {'Symbol': 'III', 'Entry date': '2026-08-24', 'Avg entry': 5.01, 'Realized $': -396},
 {'Symbol': 'JRSH', 'Entry date': '2026-08-24', 'Avg entry': 5.65, 'Realized $': -191},
 {'Symbol': 'QMCO', 'Entry date': '2026-08-26', 'Avg entry': 22.67, 'Realized $': -22},
 {'Symbol': 'TPG', 'Entry date': '2026-08-27', 'Avg entry': 53.74, 'Realized $': -525},
 {'Symbol': 'FORR', 'Entry date': '2026-08-28', 'Avg entry': 12.52, 'Realized $': -570},
 {'Symbol': 'RCMT', 'Entry date': '2026-08-31', 'Avg entry': 42.61, 'Realized $': -448},
]

ORB = dict(ADR_MIN=3.5, ADR_MAX=7.5, RUNUP_MIN=40.0, RUNUP_MAX=1000.0,
           PRICE_MIN=3.0, BASE_MIN=6, RUNUP_LB=60, MA_TOL=7.0,
           PEAK_MIN_BACK=2, PULLBACK_MIN=0.5)

MEANS = np.array([1.6053478743377458, 0.8587446273550717, 5.228070175438597,
                  8.271929824561404, 0.7042055314591412, 1.5652731406790044])
SDS = np.array([0.4884305133688643, 0.2336522673244613, 6.918111353312391,
                5.140627799258959, 0.19733635029076077, 1.105743621784014])
COEFS = np.array([0.35814256501291186, 0.15962122502434276, -0.30038853716369424,
                  0.005477613398225789, 0.08267834158482865, -0.03172365295762897])
INTERCEPT = -0.0746060880381915

def git_candidate_snapshots():
    out = subprocess.check_output(['git','log','--all','--grep=Overnight scans 2026-','--format=%H|%cI|%s'], text=True)
    by_date = {}
    for line in out.splitlines():
        parts = line.split('|',2)
        if len(parts) != 3: continue
        sha, ci, subj = parts
        m = re.search(r'Overnight scans (2026-\d\d-\d\d)', subj)
        if not m: continue
        d = pd.Timestamp(m.group(1))
        if d < START_DATE or d > END_DATE: continue
        if d.date() in by_date: continue
        try:
            raw = subprocess.check_output(['git','show',f'{sha}:data/orb.json'], text=True, stderr=subprocess.DEVNULL)
            j = json.loads(raw)
        except Exception:
            continue
        syms = [r.get('symbol','').strip().upper() for r in j.get('rows',[]) if r.get('symbol')]
        by_date[d.date()] = {'sha':sha,'symbols':sorted(set(syms)), 'old_count':len(syms)}
    return by_date

def normalize_download(df, symbol=None):
    if df is None or df.empty: return None
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        if symbol is not None:
            if symbol in x.columns.get_level_values(0): x = x[symbol]
            elif symbol in x.columns.get_level_values(1): x = x.xs(symbol, axis=1, level=1)
    x.columns = [str(c).lower() for c in x.columns]
    x = x.rename(columns={'adj close':'adj_close'})
    need = ['open','high','low','close','volume']
    if not all(c in x.columns for c in need): return None
    x = x[need].copy().dropna(subset=['open','high','low','close'])
    idx = pd.to_datetime(x.index)
    if idx.tz is not None: idx = idx.tz_convert('America/New_York')
    x.index = idx
    return x.sort_index()

def fetch_daily(symbol):
    try:
        df = yf.download(symbol, start='2025-10-01', end='2026-09-06', interval='1d', auto_adjust=False, progress=False, threads=False)
        return symbol, normalize_download(df, symbol)
    except Exception: return symbol, None

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def atr_wilder(d, n=20):
    prev = d['close'].shift(1)
    tr = pd.concat([(d['high']-d['low']).abs(), (d['high']-prev).abs(), (d['low']-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def evaluate_orb(d):
    if d is None or len(d) < 60: return False, {}
    p=ORB; c=d.close; h=d.high; l=d.low; v=d.volume
    sma10=c.rolling(10).mean(); sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean(); sma200=c.rolling(200).mean()
    adr=100*((h/l).rolling(20).mean().iloc[-1]-1); price=float(c.iloc[-1]); run_lb=min(p['RUNUP_LB'],len(d)-1)
    run_low=float(l.iloc[-run_lb:].min()); run_high=float(h.iloc[-(p['BASE_MIN']+5):].max()); runup=((run_high-run_low)/run_low*100) if run_low>0 else np.nan
    win=d.iloc[-(p['BASE_MIN']+4):]; hh=win.high.to_numpy(); peak_back=len(hh)-1-int(hh.argmax()); recent_high=float(hh.max()); base_low=float(win.low.min())
    base_depth=((recent_high-base_low)/recent_high*100) if recent_high>0 else np.nan; rng=h-l
    contracting=rng.iloc[-p['BASE_MIN']:].mean() < rng.iloc[-2*p['BASE_MIN']:-p['BASE_MIN']].mean(); pulled_in=price < recent_high*(1-p['PULLBACK_MIN']/100)
    base_ok=bool(contracting and peak_back>=p['PEAK_MIN_BACK'] and pulled_in); lows_up=l.iloc[-p['BASE_MIN']:].min() > l.iloc[-2*p['BASE_MIN']:-p['BASE_MIN']].min()
    def ma_status(ma):
        rising=bool(ma.iloc[-1]>ma.iloc[-6]); nearby=bool(l.iloc[-1] <= ma.iloc[-1]*(1+p['MA_TOL']/100)); return rising,nearby
    r10,n10=ma_status(sma10); r20,n20=ma_status(sma20); r50,n50=ma_status(sma50); surf=(r10 and n10) or (r20 and n20) or (r50 and n50)
    above50=bool(not pd.isna(sma50.iloc[-1]) and price>sma50.iloc[-1]); has200=not pd.isna(sma200.iloc[-1]); above200=bool((not has200) or price>sma200.iloc[-1])
    crit=dict(price=price>=p['PRICE_MIN'], adr=p['ADR_MIN']<=adr<p['ADR_MAX'], runup=(not np.isnan(runup) and p['RUNUP_MIN']<=runup<=p['RUNUP_MAX']), base=base_ok, hl=bool(lows_up), surf=bool(surf), t50=above50, t200=above200)
    return all(crit.values()), dict(price=price,adr=adr,runup=runup,base_depth=base_depth,peak_back=peak_back,**{f'crit_{k}':int(v) for k,v in crit.items()})

def bars_since(cond):
    arr=np.asarray(cond.fillna(False),dtype=bool); inds=np.where(arr)[0]
    if len(inds)==0: return 21
    return min(len(arr)-1-int(inds[-1]),21)

def ma_features(d):
    if d is None or len(d)<60: return None
    c,h,l,v=d.close,d.high,d.low,d.volume; atr20=atr_wilder(d,20); e10=ema(c,10); e20=ema(c,20)
    high3=h.rolling(3).max(); low3=l.rolling(3).min(); range3=(high3-low3)/atr20; vol_ratio=v.rolling(5).mean()/v.rolling(20).mean(); reset_age=bars_since(c<=e10)
    wh=h.iloc[-20:].to_numpy(); high_age=len(wh)-1-int(np.argmax(wh)); high20=h.iloc[-20:].max(); low20=l.iloc[-20:].min(); pos=(c.iloc[-1]-low20)/(high20-low20) if high20>low20 else .5
    prior_high=h.shift(1).rolling(20).max().iloc[-1]; dist=(prior_high-c.iloc[-1])/atr20.iloc[-1] if atr20.iloc[-1]>0 else np.nan
    mom15=((c.iloc[-1]/c.iloc[-16])-1)*100 if len(c)>=16 and c.iloc[-16]!=0 else np.nan; ema20dist=(c.iloc[-1]-e20.iloc[-1])/atr20.iloc[-1] if atr20.iloc[-1]>0 else np.nan
    low50=l.iloc[-50:].min(); stretch=(c.iloc[-1]-low50)/atr20.iloc[-1] if atr20.iloc[-1]>0 else np.nan
    vals=np.array([range3.iloc[-1],vol_ratio.iloc[-1],reset_age,high_age,pos,dist],dtype=float)
    if np.any(np.isnan(vals)): return None
    z=(vals-MEANS)/SDS; logit=INTERCEPT+float(np.dot(COEFS,z)); score=100/(1+math.exp(-logit)); weak=(mom15<=4.6 and ema20dist<=0.72); exhausted=(stretch>11.5)
    normal=score>=41 and not weak and not exhausted; exceptional=(35<=score<41 and mom15>=20 and ema20dist>=1.40 and not exhausted)
    return dict(ma_score=score, ma_go=int(normal or exceptional), mom15=mom15, ema20_dist_atr=ema20dist, stretch50_atr=stretch, range3_atr=vals[0],vol_ratio5=vals[1],reset_age=reset_age,high_age=high_age,close_pos20=pos,dist_high20_atr=dist, weak=int(weak),exhausted=int(exhausted))

def fetch_earnings(symbol):
    try:
        e=yf.Ticker(symbol).get_earnings_dates(limit=24)
        if e is None or e.empty: return symbol, None
        e=e.copy(); e.index=pd.to_datetime(e.index); return symbol,e
    except Exception: return symbol,None

def earnings_before(e, date):
    if e is None or e.empty: return None
    target=pd.Timestamp(date).tz_localize('America/New_York') + pd.Timedelta(hours=9,minutes=30); idx=e.index
    if idx.tz is None: idx=idx.tz_localize('America/New_York')
    else: idx=idx.tz_convert('America/New_York')
    x=e.copy(); x.index=idx; x=x[x.index<target]
    if x.empty: return None
    row=x.sort_index().iloc[-1]; est=pd.to_numeric(row.get('EPS Estimate',np.nan),errors='coerce'); rep=pd.to_numeric(row.get('Reported EPS',np.nan),errors='coerce')
    if pd.isna(est) or pd.isna(rep): return None
    surprise=(rep-est)/abs(est)*100 if est!=0 else np.nan
    return dict(earnings_pass=int(rep>est), eps_est=est, eps_reported=rep, eps_surprise_pct=surprise, earnings_date=x.sort_index().index[-1].isoformat())

def fetch_intraday(symbol):
    try:
        df=yf.download(symbol,start=INTRADAY_START,end=INTRADAY_END,interval='30m',auto_adjust=False,prepost=False,progress=False,threads=False)
        return symbol,normalize_download(df,symbol)
    except Exception: return symbol,None

def regular_session(x):
    if x is None or x.empty: return x
    idx=x.index
    if idx.tz is None: idx=idx.tz_localize('America/New_York')
    else: idx=idx.tz_convert('America/New_York')
    y=x.copy(); y.index=idx; return y.between_time('09:30','15:59')

def structure_features(x, date):
    x=regular_session(x)
    if x is None or x.empty: return None
    date=pd.Timestamp(date).date(); day=x[x.index.date==date]
    if len(day)<3: return None
    c1=day.iloc[0]; c2=day.iloc[1]; c3=day.iloc[2]; hist=x[x.index<day.index[1]]; prior=hist.iloc[:-1] if len(hist)>1 else hist.iloc[0:0]
    if len(prior)<10: return None
    prior10=hist.iloc[-10:]; prior6=hist.iloc[-6:]; prior4=hist.iloc[-4:]; median_range20=float((prior.tail(20).high-prior.tail(20).low).median()); median_body10=float((prior10.close-prior10.open).abs().median()); vol50=float(prior.tail(50).volume.mean()) if len(prior)>=10 else np.nan
    local_res=float(prior6.high.max()); touches=int(((prior10.high-local_res).abs()/local_res<=0.01).sum()) if local_res>0 else 0; c2_range=float(c2.high-c2.low); c2_body=float(c2.close-c2.open)
    c2_close_pos=(c2.close-c2.low)/c2_range if c2_range>0 else .5; body_ratio=c2_body/(median_body10 if median_body10>0 else 1e-9); range_ratio=c2_range/(median_range20 if median_range20>0 else 1e-9); vol_ratio=c2.volume/(vol50 if vol50>0 else np.nan)
    closes=prior4.close.to_numpy(dtype=float); slope=float(np.polyfit(np.arange(len(closes)),closes,1)[0]) if len(closes)>=3 else np.nan; level_slope=abs(slope)*(len(closes)-1)/(median_range20 if median_range20>0 else 1e-9); close_spread=(closes.max()-closes.min())/(median_range20 if median_range20>0 else 1e-9)
    levelling=(level_slope<=0.75 and close_spread<=2.0); thrust=(c2_body>0 and body_ratio>=1.2 and range_ratio>=1.1 and c2_close_pos>=0.70 and (pd.isna(vol_ratio) or vol_ratio>=0.8)); orh=(c2.close>c1.high); resistance_break=(orh and c2.close>local_res and touches>=1); level_thrust=(orh and levelling and thrust)
    dates=sorted(set(prior.index.date)); prev_date=dates[-1] if dates else None; prior_day=prior[prior.index.date==prev_date] if prev_date else prior.iloc[0:0]; prev_day_high=float(prior_day.high.max()) if not prior_day.empty else np.nan; prior3dates=dates[-3:]; prior3=prior[np.isin(prior.index.date,prior3dates)] if prior3dates else prior.iloc[0:0]; prior3_high=float(prior3.high.max()) if not prior3.empty else np.nan
    break_prevday=bool(orh and not pd.isna(prev_day_high) and c2.close>prev_day_high); break_3day=bool(orh and not pd.isna(prior3_high) and c2.close>prior3_high); range_or_resistance=bool(orh and (resistance_break or break_prevday or break_3day)); valid_structure=bool(range_or_resistance or level_thrust)
    entry=float(c3.open); future=x[x.index>=day.index[2]]; sess=[]
    for dd in future.index.date:
        if dd not in sess: sess.append(dd)
    def horizon(n):
        use=set(sess[:n]); f=future[np.isin(future.index.date,list(use))]
        if f.empty or entry<=0: return (np.nan,np.nan,np.nan)
        return (f.high.max()/entry-1)*100,(f.low.min()/entry-1)*100,float(f.iloc[-1].close/entry-1)*100
    mfe5,mae5,ret5=horizon(5); mfe10,mae10,ret10=horizon(10)
    return dict(c1_open=float(c1.open),c1_high=float(c1.high),c1_low=float(c1.low),c1_close=float(c1.close),c2_open=float(c2.open),c2_high=float(c2.high),c2_low=float(c2.low),c2_close=float(c2.close),entry_c3_open=entry,orh_trigger=int(orh),local_res=local_res,touches_1pct=touches,resistance_break=int(resistance_break),break_prevday=int(break_prevday),break_3day=int(break_3day),range_or_resistance=int(range_or_resistance),levelling=int(levelling),level_slope_norm=level_slope,close_spread_norm=close_spread,thrust=int(thrust),level_thrust=int(level_thrust),c2_body_ratio=body_ratio,c2_range_ratio=range_ratio,c2_close_pos=c2_close_pos,c2_vol_ratio50=vol_ratio,valid_structure=int(valid_structure),mfe5_pct=mfe5,mae5_pct=mae5,ret5_pct=ret5,mfe10_pct=mfe10,mae10_pct=mae10,ret10_pct=ret10)

def main():
    os.makedirs('tmp/backtest_out',exist_ok=True); snaps=git_candidate_snapshots(); candidate_pairs=[]
    for d,meta in sorted(snaps.items()):
        for s in meta['symbols']: candidate_pairs.append((pd.Timestamp(d),s,meta['sha'],meta['old_count']))
    actual=pd.DataFrame(TRADES); actual['Entry date']=pd.to_datetime(actual['Entry date']); all_syms=sorted(set([s for _,s,_,_ in candidate_pairs])|set(actual.Symbol)); print('snapshots',len(snaps),'candidate pairs',len(candidate_pairs),'symbols',len(all_syms))
    daily={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(fetch_daily,s) for s in all_syms]
        for i,f in enumerate(as_completed(futs),1):
            s,d=f.result(); daily[s]=d
            if i%50==0: print('daily',i,'/',len(futs))
    screen_rows=[]
    for dt,s,sha,old_count in candidate_pairs:
        d=daily.get(s)
        if d is None: continue
        idx=pd.to_datetime(d.index)
        if idx.tz is not None: dd=d[idx.tz_convert(None)<dt]
        else: dd=d[idx<dt]
        if len(dd)<60: continue
        opass,om=evaluate_orb(dd); mf=ma_features(dd)
        if mf is None: continue
        screen_rows.append(dict(date=dt.date().isoformat(),symbol=s,source_sha=sha,old_scan_count=old_count,orb_pass=int(opass),**om,**mf))
    screen=pd.DataFrame(screen_rows); screen.to_csv('tmp/backtest_out/reconstructed_screen.csv',index=False); pre=screen[(screen.orb_pass==1)&(screen.ma_go==1)].copy(); print('ORB+MA pairs',len(pre),'symbols',pre.symbol.nunique())
    earnings={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(fetch_earnings,s) for s in sorted(pre.symbol.unique())]
        for f in as_completed(futs):
            s,e=f.result(); earnings[s]=e
    erows=[]
    for _,r in pre.iterrows():
        e=earnings_before(earnings.get(r.symbol),r.date); rr=r.to_dict(); rr.update(e or {'earnings_pass':0,'eps_est':np.nan,'eps_reported':np.nan,'eps_surprise_pct':np.nan,'earnings_date':''}); erows.append(rr)
    final=pd.DataFrame(erows)
    if not final.empty: final['watchlist_pass']=((final.orb_pass==1)&(final.ma_go==1)&(final.earnings_pass==1)).astype(int)
    final.to_csv('tmp/backtest_out/final_screen_candidates.csv',index=False); flagged=final[final.watchlist_pass==1].copy() if not final.empty else final; print('final flagged pairs',len(flagged),'symbols',flagged.symbol.nunique() if not flagged.empty else 0)
    intraday_syms=sorted(set(actual.Symbol)|(set(flagged.symbol) if not flagged.empty else set())); intraday={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(fetch_intraday,s) for s in intraday_syms]
        for i,f in enumerate(as_completed(futs),1):
            s,x=f.result(); intraday[s]=x
            if i%20==0: print('intraday',i,'/',len(futs))
    arows=[]
    for _,r in actual.iterrows():
        sf=structure_features(intraday.get(r.Symbol),r['Entry date']); base={'date':r['Entry date'].date().isoformat(),'symbol':r.Symbol,'actual_entry':r['Avg entry'],'actual_pnl':r['Realized $'],'actual_win':int(r['Realized $']>0)}
        if sf:
            base.update(sf); base['entry_match_pct']=(base['actual_entry']/sf['entry_c3_open']-1)*100 if sf['entry_c3_open'] else np.nan
        arows.append(base)
    actual_struct=pd.DataFrame(arows); actual_struct.to_csv('tmp/backtest_out/actual_trade_structures.csv',index=False)
    frows=[]; actual_key={(r['Entry date'].date().isoformat(),r.Symbol):(r['Avg entry'],r['Realized $']) for _,r in actual.iterrows()}
    if not flagged.empty:
        for _,r in flagged.iterrows():
            sf=structure_features(intraday.get(r.symbol),pd.Timestamp(r.date)); rr=r.to_dict(); key=(str(r.date),r.symbol); rr['bought']=int(key in actual_key)
            rr['actual_entry'],rr['actual_pnl']=actual_key.get(key,(np.nan,np.nan))
            if sf: rr.update(sf)
            frows.append(rr)
    opp=pd.DataFrame(frows); opp.to_csv('tmp/backtest_out/flagged_opportunities.csv',index=False)
    summary={'scan_dates':len(snaps),'candidate_pairs_recovered':len(candidate_pairs),'unique_candidate_symbols':len(set(s for _,s,_,_ in candidate_pairs)),'orb_ma_pairs':int(len(pre)),'final_flagged_pairs':int(len(flagged)) if not flagged.empty else 0,'actual_trades_aug':int(len(actual_struct)),'actual_with_intraday':int(actual_struct['orh_trigger'].notna().sum()) if 'orh_trigger' in actual_struct else 0}
    if not actual_struct.empty and 'valid_structure' in actual_struct:
        z=actual_struct.dropna(subset=['valid_structure']); summary['actual_structure_rate']=float(z.valid_structure.mean()) if len(z) else None
        for label,grp in [('winners',z[z.actual_win==1]),('losers',z[z.actual_win==0])]:
            summary[label]={'n':int(len(grp)),'valid_structure_rate':float(grp.valid_structure.mean()) if len(grp) else None,'resistance_rate':float(grp.range_or_resistance.mean()) if len(grp) else None,'level_thrust_rate':float(grp.level_thrust.mean()) if len(grp) else None,'median_entry_match_pct':float(grp.entry_match_pct.abs().median()) if 'entry_match_pct' in grp and len(grp) else None}
    if not opp.empty and 'valid_structure' in opp:
        z=opp.dropna(subset=['valid_structure']); valid=z[(z.orh_trigger==1)&(z.valid_structure==1)]; summary['flagged_with_intraday']=int(len(z)); summary['flagged_orh']=int((z.orh_trigger==1).sum()); summary['flagged_valid_structure']=int(len(valid))
        if len(valid):
            summary['valid_structure_outcomes']={'n':int(len(valid)),'median_mfe5':float(valid.mfe5_pct.median()),'median_mae5':float(valid.mae5_pct.median()),'median_mfe10':float(valid.mfe10_pct.median()),'median_ret10':float(valid.ret10_pct.median()),'bought_n':int(valid.bought.sum()),'unbought_n':int((valid.bought==0).sum())}
            for name,g in [('bought',valid[valid.bought==1]),('unbought',valid[valid.bought==0])]:
                if len(g): summary[name]={'n':int(len(g)),'median_mfe5':float(g.mfe5_pct.median()),'median_mae5':float(g.mae5_pct.median()),'median_mfe10':float(g.mfe10_pct.median()),'median_ret10':float(g.ret10_pct.median()),'pct_mfe10_ge10':float((g.mfe10_pct>=10).mean())}
    with open('tmp/backtest_out/summary.json','w') as f: json.dump(summary,f,indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()

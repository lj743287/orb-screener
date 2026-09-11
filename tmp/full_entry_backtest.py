#!/usr/bin/env python3
import json, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import screener
from bars_cache import CACHE

OUT=Path('tmp/full_backtest_out'); OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-08-03'); END=pd.Timestamp('2026-09-04')
INTRA_START='2026-07-24'; INTRA_END='2026-09-12'; INTRA_BATCH=40; EARN_WORKERS=8
MEANS=np.array([1.6053478743377458,0.8587446273550717,5.228070175438597,8.271929824561404,0.7042055314591412,1.5652731406790044])
SDS=np.array([0.4884305133688643,0.2336522673244613,6.918111353312391,5.140627799258959,0.19733635029076077,1.105743621784014])
COEFS=np.array([0.35814256501291186,0.15962122502434276,-0.30038853716369424,0.005477613398225789,0.08267834158482865,-0.03172365295762897])
INTERCEPT=-0.0746060880381915

def cache_frame(symbol):
    rows=CACHE.get(symbol,newest_first=False)
    if not rows:return None
    d=pd.DataFrame(rows).rename(columns={'d':'datetime','o':'open','h':'high','l':'low','c':'close','v':'volume'})
    if 'datetime' not in d:return None
    d['datetime']=pd.to_datetime(d['datetime']); d=d.set_index('datetime').sort_index()
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d[['open','high','low','close','volume']].dropna(subset=['open','high','low','close'])

def test_sessions():
    try:
        spy=yf.download('SPY',start=START.strftime('%Y-%m-%d'),end=(END+pd.Timedelta(days=2)).strftime('%Y-%m-%d'),interval='1d',auto_adjust=False,progress=False,threads=False)
        if spy is not None and not spy.empty:
            ds=pd.to_datetime(spy.index).tz_localize(None).normalize(); ds=[d for d in ds if START<=d<=END]
            if ds:return ds
    except Exception:pass
    return list(pd.bdate_range(START,END))

def ema(s,span):return s.ewm(span=span,adjust=False).mean()
def atr_wilder(d,n=20):
    prev=d.close.shift(1); tr=pd.concat([(d.high-d.low).abs(),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def bars_since(cond):
    arr=np.asarray(cond.fillna(False),dtype=bool); idx=np.where(arr)[0]
    return 21 if len(idx)==0 else min(len(arr)-1-int(idx[-1]),21)

def ma_features(d):
    if d is None or len(d)<60:return None
    c,h,l,v=d.close,d.high,d.low,d.volume; atr20=atr_wilder(d,20); e10=ema(c,10); e20=ema(c,20)
    range3=(h.rolling(3).max()-l.rolling(3).min())/atr20; vol_ratio=v.rolling(5).mean()/v.rolling(20).mean(); reset_age=bars_since(c<=e10)
    wh=h.iloc[-20:].to_numpy(); high_age=len(wh)-1-int(np.argmax(wh)); high20=h.iloc[-20:].max(); low20=l.iloc[-20:].min(); close_pos=(c.iloc[-1]-low20)/(high20-low20) if high20>low20 else .5
    prior_high=h.shift(1).rolling(20).max().iloc[-1]; dist_high=(prior_high-c.iloc[-1])/atr20.iloc[-1] if atr20.iloc[-1]>0 else np.nan
    mom15=((c.iloc[-1]/c.iloc[-16])-1)*100 if len(c)>=16 and c.iloc[-16]!=0 else np.nan; ema20_dist=(c.iloc[-1]-e20.iloc[-1])/atr20.iloc[-1] if atr20.iloc[-1]>0 else np.nan
    low50=l.iloc[-50:].min(); stretch50=(c.iloc[-1]-low50)/atr20.iloc[-1] if atr20.iloc[-1]>0 else np.nan
    vals=np.array([range3.iloc[-1],vol_ratio.iloc[-1],reset_age,high_age,close_pos,dist_high],dtype=float)
    if np.any(np.isnan(vals)):return None
    z=(vals-MEANS)/SDS; score=100/(1+math.exp(-(INTERCEPT+float(np.dot(COEFS,z)))))
    weak=bool(mom15<=4.6 and ema20_dist<=0.72); exhausted=bool(stretch50>11.5)
    normal=bool(score>=41 and not weak and not exhausted); exceptional=bool(35<=score<41 and mom15>=20 and ema20_dist>=1.40 and not exhausted)
    return dict(ma_score=score,ma_go=int(normal or exceptional),mom15=mom15,ema20_dist_atr=ema20_dist,stretch50_atr=stretch50,range3_atr=vals[0],vol_ratio5=vals[1],reset_age=int(reset_age),high_age=int(high_age),close_pos20=close_pos,dist_high20_atr=dist_high,weak=int(weak),exhausted=int(exhausted))

def prior_slice(d,session):
    x=d.copy(); idx=pd.to_datetime(x.index); idx=idx.tz_localize(None) if idx.tz is not None else idx; x.index=idx
    return x[x.index.normalize()<pd.Timestamp(session).normalize()]

def fetch_earnings(symbol):
    for attempt in range(2):
        try:
            e=yf.Ticker(symbol).get_earnings_dates(limit=32)
            if e is not None and not e.empty:
                e=e.copy(); e.index=pd.to_datetime(e.index); return symbol,e
        except Exception:pass
        time.sleep(1+attempt)
    return symbol,None

def earnings_for_date(e,date):
    out=dict(earnings_history_known=0,prev_earnings_pass=0,prev_eps_est=np.nan,prev_eps_reported=np.nan,prev_eps_surprise_pct=np.nan,prev_earnings_date=None,next_earnings_known=0,next_earnings_date=None,days_to_next_earnings=np.nan,earnings_blackout_7d=np.nan)
    if e is None or e.empty:return out
    target=pd.Timestamp(date)
    if target.tzinfo is None:target=target.tz_localize('America/New_York')
    target=target.normalize()+pd.Timedelta(hours=9,minutes=30)
    x=e.copy(); idx=pd.to_datetime(x.index); idx=idx.tz_localize('America/New_York') if idx.tz is None else idx.tz_convert('America/New_York'); x.index=idx; x=x.sort_index()
    prev=x[x.index<target]
    if not prev.empty:
        r=prev.iloc[-1]; est=pd.to_numeric(r.get('EPS Estimate',np.nan),errors='coerce'); rep=pd.to_numeric(r.get('Reported EPS',np.nan),errors='coerce'); out['prev_earnings_date']=prev.index[-1].isoformat()
        if not pd.isna(est) and not pd.isna(rep):
            out['earnings_history_known']=1; out['prev_eps_est']=float(est); out['prev_eps_reported']=float(rep); out['prev_earnings_pass']=int(rep>est)
            if est!=0:out['prev_eps_surprise_pct']=float((rep-est)/abs(est)*100)
    nxt=x[x.index>=target]
    if not nxt.empty:
        nd=nxt.index[0]; days=(nd.date()-target.date()).days; out['next_earnings_known']=1; out['next_earnings_date']=nd.isoformat(); out['days_to_next_earnings']=int(days); out['earnings_blackout_7d']=int(0<=days<=7)
    return out

def flatten_frame(df,symbol):
    if df is None or df.empty:return None
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        if symbol in x.columns.get_level_values(0):x=x[symbol]
        elif symbol in x.columns.get_level_values(1):x=x.xs(symbol,axis=1,level=1)
        else:return None
    x.columns=[str(c).lower() for c in x.columns]; need=['open','high','low','close','volume']
    if not all(c in x.columns for c in need):return None
    x=x[need].dropna(subset=['open','high','low','close']).copy(); x.index=pd.to_datetime(x.index)
    return x.sort_index()

def download_intraday_batch(symbols,retries=2):
    tickers=' '.join(symbols)
    for attempt in range(retries+1):
        try:
            d=yf.download(tickers=tickers,start=INTRA_START,end=INTRA_END,interval='30m',auto_adjust=False,group_by='ticker',threads=True,prepost=False,progress=False)
            if d is not None and not d.empty:return d
        except Exception as e:print('intraday batch error',repr(e),flush=True)
        time.sleep(2+attempt*3)
    return None

def regular_session(x):
    if x is None or x.empty:return x
    y=x.copy(); idx=pd.to_datetime(y.index); idx=idx.tz_localize('America/New_York') if idx.tz is None else idx.tz_convert('America/New_York'); y.index=idx
    return y.sort_index().between_time('09:30','15:59')

def structure_features(x,date):
    x=regular_session(x)
    if x is None or x.empty:return None
    target=pd.Timestamp(date).date(); day=x[x.index.date==target]
    if len(day)<3:return None
    c1,c2,c3=day.iloc[0],day.iloc[1],day.iloc[2]; prior=x[x.index<day.index[0]]
    if len(prior)<20:return None
    dates=sorted(set(prior.index.date))
    if len(dates)<3:return None
    prevdates=dates[-3:]; prevday=prior[prior.index.date==dates[-1]]; prior3=prior[np.isin(prior.index.date,prevdates)]; p12=prior.tail(12); p20=prior.tail(20); p50=prior.tail(50)
    med_rng=float((p20.high-p20.low).median()); med_body=float((p20.close-p20.open).abs().median()); vol50=float(p50.volume.mean())
    prevday_high=float(prevday.high.max()); prior3_high=float(prior3.high.max()); local12_high=float(p12.high.max()); prior_close=float(prior.iloc[-1].close)
    levels=sorted(set([prevday_high,local12_high,prior3_high])); near=[v for v in levels if v>=prior_close*.985]; resistance=min(near) if near else local12_high
    def stats(b):
        rng=float(b.high-b.low); body=float(b.close-b.open); cp=(b.close-b.low)/rng if rng>0 else .5
        return body/(med_body if med_body>0 else 1e-9),rng/(med_rng if med_rng>0 else 1e-9),cp,float(b.volume)/(vol50 if vol50>0 else np.nan)
    b1,r1,cp1,v1=stats(c1); b2,r2,cp2,v2=stats(c2)
    break_c1=bool(c1.close>resistance and prior_close<=resistance); break_c2=bool(c2.close>resistance and c1.close<=resistance); break_prevday=bool(max(c1.close,c2.close)>prevday_high and prior_close<=prevday_high); break_3day=bool(max(c1.close,c2.close)>prior3_high and prior_close<=prior3_high); resistance_break=bool(break_c1 or break_c2 or break_prevday or break_3day)
    p4=prior.tail(4); closes=p4.close.to_numpy(float); spread=(closes.max()-closes.min())/(med_rng if med_rng>0 else 1e-9); net=abs(closes[-1]-closes[0])/(med_rng if med_rng>0 else 1e-9); levelling=bool(spread<=2.0 and net<=1.25)
    thrust1=bool(b1>=1.5 and r1>=1.25 and cp1>=.70 and (pd.isna(v1) or v1>=.8)); thrust2=bool(b2>=1.5 and r2>=1.25 and cp2>=.70 and (pd.isna(v2) or v2>=.8)); level_thrust=bool(levelling and (thrust1 or thrust2))
    hour_low=min(float(c1.low),float(c2.low)); hour_high=max(float(c1.high),float(c2.high)); hr=hour_high-hour_low; hour_close_pos=(float(c2.close)-hour_low)/hr if hr>0 else .5; first_hour_hold=bool(hour_close_pos>=.45)
    valid=bool((resistance_break or level_thrust) and first_hour_hold); entry=float(c3.open); future=x[x.index>=day.index[2]]; sessions=[]
    for dd in future.index.date:
        if dd not in sessions:sessions.append(dd)
    def horizon(n):
        if len(sessions)<n or entry<=0:return np.nan,np.nan,np.nan,0
        f=future[np.isin(future.index.date,sessions[:n])]
        if f.empty:return np.nan,np.nan,np.nan,0
        return (float(f.high.max())/entry-1)*100,(float(f.low.min())/entry-1)*100,(float(f.iloc[-1].close)/entry-1)*100,1
    mfe5,mae5,ret5,full5=horizon(5); mfe10,mae10,ret10,full10=horizon(10)
    return dict(entry_c3=entry,resistance=resistance,break_c1=int(break_c1),break_c2=int(break_c2),break_prevday=int(break_prevday),break_3day=int(break_3day),resistance_break=int(resistance_break),levelling=int(levelling),thrust_c1=int(thrust1),thrust_c2=int(thrust2),level_thrust=int(level_thrust),first_hour_hold=int(first_hour_hold),valid_structure=int(valid),c1_body_ratio=b1,c1_range_ratio=r1,c1_close_pos=cp1,c1_vol_ratio=v1,c2_body_ratio=b2,c2_range_ratio=r2,c2_close_pos=cp2,c2_vol_ratio=v2,level_spread_norm=spread,level_net_norm=net,sessions_available=len(sessions),full_5d=full5,mfe5_pct=mfe5,mae5_pct=mae5,ret5_pct=ret5,full_10d=full10,mfe10_pct=mfe10,mae10_pct=mae10,ret10_pct=ret10)

def metrics(g):
    out={'n':int(len(g))}
    if not len(g):return out
    out['median_ma_score']=float(g.ma_score.median()); g5=g[g.full_5d==1]; out['n_full5']=int(len(g5))
    if len(g5):out.update(median_mfe5=float(g5.mfe5_pct.median()),median_mae5=float(g5.mae5_pct.median()),median_ret5=float(g5.ret5_pct.median()),pct_mfe5_ge5=float((g5.mfe5_pct>=5).mean()),pct_mfe5_ge10=float((g5.mfe5_pct>=10).mean()),pct_mae5_le_minus5=float((g5.mae5_pct<=-5).mean()),pct_ret5_positive=float((g5.ret5_pct>0).mean()))
    g10=g[g.full_10d==1]; out['n_full10']=int(len(g10))
    if len(g10):out.update(median_mfe10=float(g10.mfe10_pct.median()),median_mae10=float(g10.mae10_pct.median()),median_ret10=float(g10.ret10_pct.median()),pct_mfe10_ge10=float((g10.mfe10_pct>=10).mean()),pct_mfe10_ge20=float((g10.mfe10_pct>=20).mean()),pct_ret10_positive=float((g10.ret10_pct>0).mean()))
    return out

def scrub(v):
    if isinstance(v,dict):return {k:scrub(x) for k,x in v.items()}
    if isinstance(v,list):return [scrub(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return None if np.isnan(v) else float(v)
    return v

def main():
    if not CACHE.available:raise RuntimeError('Alpaca cache not available')
    sessions=test_sessions(); universe=CACHE.symbols; print('sessions',len(sessions),sessions[0],sessions[-1],'universe',len(universe),flush=True)
    rows=[]; symbols_ok=0
    for i,sym in enumerate(universe,1):
        d=cache_frame(sym)
        if d is None or len(d)<60:continue
        symbols_ok+=1
        for session in sessions:
            hist=prior_slice(d,session)
            if len(hist)<60:continue
            passed,om,crit=screener.compute_screen(hist)
            if not passed:continue
            maf=ma_features(hist)
            if not maf:continue
            row={'date':pd.Timestamp(session).strftime('%Y-%m-%d'),'symbol':sym,'orb_pass':1}; row.update({k:v for k,v in om.items() if np.isscalar(v)}); row.update({f'crit_{k}':int(bool(v)) for k,v in crit.items()}); row.update(maf); rows.append(row)
        if i%500==0:print('daily',i,'/',len(universe),'ok',symbols_ok,'orb pairs',len(rows),flush=True)
    daily=pd.DataFrame(rows); daily.to_csv(OUT/'orb_ma_candidates.csv',index=False)
    if daily.empty:(OUT/'summary.json').write_text(json.dumps({'error':'no candidates'})); return
    daily['date']=pd.to_datetime(daily.date); ma=daily[daily.ma_go==1].copy(); print('ORB pairs',len(daily),'ORB+MA',len(ma),flush=True)
    earn_syms=sorted(ma.symbol.unique()); earnings={}
    with ThreadPoolExecutor(max_workers=EARN_WORKERS) as ex:
        fut=[ex.submit(fetch_earnings,s) for s in earn_syms]
        for i,f in enumerate(as_completed(fut),1):
            s,e=f.result(); earnings[s]=e
            if i%100==0 or i==len(fut):print('earnings',i,'/',len(fut),flush=True)
    erows=[]
    for _,r in ma.iterrows():rr=r.to_dict(); rr.update(earnings_for_date(earnings.get(r.symbol),r.date)); erows.append(rr)
    filtered=pd.DataFrame(erows); filtered['earnings_rule_base']=((filtered.earnings_history_known==1)&(filtered.prev_earnings_pass==1)).astype(int); filtered['earnings_rule_strict']=((filtered.earnings_rule_base==1)&(filtered.next_earnings_known==1)&(filtered.earnings_blackout_7d==0)).astype(int); filtered.to_csv(OUT/'earnings_filtered_candidates.csv',index=False)
    base=filtered[filtered.earnings_rule_base==1].copy(); intra_syms=sorted(base.symbol.unique()); print('positive prior earnings pairs',len(base),'symbols',len(intra_syms),flush=True)
    intra={}; errors=[]; batches=[intra_syms[i:i+INTRA_BATCH] for i in range(0,len(intra_syms),INTRA_BATCH)]
    for bi,batch in enumerate(batches,1):
        raw=download_intraday_batch(batch)
        if raw is None:errors.append(batch); continue
        for sym in batch:
            x=flatten_frame(raw,sym)
            if x is not None and not x.empty:intra[sym]=x
        if bi%5==0 or bi==len(batches):print('intraday',bi,'/',len(batches),'symbols ok',len(intra),flush=True)
    out=[]
    for _,r in base.iterrows():
        rr=r.to_dict(); f=structure_features(intra.get(r.symbol),r.date)
        if f:rr.update(f)
        out.append(rr)
    results=pd.DataFrame(out); results.to_csv(OUT/'full_results.csv',index=False); z=results.dropna(subset=['valid_structure']).copy(); valid=z[z.valid_structure==1].copy(); clean=valid[(valid.next_earnings_known==1)&(valid.earnings_blackout_7d==0)].copy(); blocked=valid[(valid.next_earnings_known==1)&(valid.earnings_blackout_7d==1)].copy(); unknown=valid[valid.next_earnings_known==0].copy(); strict=valid[valid.earnings_rule_strict==1].copy()
    if len(strict):strict.sort_values(['mfe10_pct','mfe5_pct'],ascending=False,na_position='last').head(100).to_csv(OUT/'top_strict_opportunities.csv',index=False)
    bands={}
    for lo,hi,label in [(35,41,'35_40'),(41,50,'41_49'),(50,60,'50_59'),(60,101,'60_plus')]:bands[label]=metrics(strict[(strict.ma_score>=lo)&(strict.ma_score<hi)])
    summary={'period':{'start':START.strftime('%Y-%m-%d'),'end':END.strftime('%Y-%m-%d'),'sessions':len(sessions),'entry_proxy':'C3 open / 10:30 ET'},'rules':{'orb_source':'current screener.py compute_screen using full Alpaca cache universe','momentum_architecture':'score>=41 excluding weak/exhausted, plus exceptional 35-40 route','prior_earnings':'most recent reported EPS > estimate','earnings_blackout':'exclude 0-7 calendar days before next earnings','structure':'(resistance break OR levelling+upward thrust) AND first-hour hold'},'coverage':{'universe':len(universe),'daily_symbols_ok':symbols_ok,'orb_pairs':len(daily),'orb_ma_pairs':len(ma),'earnings_symbols':len(earn_syms),'positive_prior_earnings_pairs':int((filtered.earnings_rule_base==1).sum()),'next_earnings_known_pairs':int(((filtered.earnings_rule_base==1)&(filtered.next_earnings_known==1)).sum()),'blackout_pairs_pre_structure':int(((filtered.earnings_rule_base==1)&(filtered.next_earnings_known==1)&(filtered.earnings_blackout_7d==1)).sum()),'strict_pairs_pre_structure':int((filtered.earnings_rule_strict==1).sum()),'intraday_symbols_requested':len(intra_syms),'intraday_symbols_ok':len(intra),'pairs_with_structure_data':len(z),'valid_structure_pairs':len(valid),'strict_valid_pairs':len(strict),'blocked_valid_pairs':len(blocked),'unknown_calendar_valid_pairs':len(unknown),'intraday_failed_batches':len(errors)},'outcomes':{'all_valid_positive_prior_earnings':metrics(valid),'clean_over_7d_known_calendar':metrics(clean),'strict_clean_over_7d':metrics(strict),'would_be_blocked_0_to_7d':metrics(blocked),'unknown_next_earnings':metrics(unknown)},'strict_score_bands':bands}
    s5=strict[strict.full_5d==1]; b5=blocked[blocked.full_5d==1]
    if len(s5) and len(b5):summary['earnings_blackout_effect_5d']={'clean_n':len(s5),'blocked_n':len(b5),'clean_median_mfe5':float(s5.mfe5_pct.median()),'blocked_median_mfe5':float(b5.mfe5_pct.median()),'clean_median_mae5':float(s5.mae5_pct.median()),'blocked_median_mae5':float(b5.mae5_pct.median()),'clean_median_ret5':float(s5.ret5_pct.median()),'blocked_median_ret5':float(b5.ret5_pct.median()),'clean_pct_ret5_positive':float((s5.ret5_pct>0).mean()),'blocked_pct_ret5_positive':float((b5.ret5_pct>0).mean())}
    (OUT/'intraday_failed_batches.json').write_text(json.dumps(errors,indent=2)); (OUT/'summary.json').write_text(json.dumps(scrub(summary),indent=2)); print(json.dumps(scrub(summary),indent=2),flush=True)

if __name__=='__main__':main()

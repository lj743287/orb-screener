import os, math, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf

ART='tmp/prior_artifact/final_screen_candidates.csv'
OUT='tmp/refined_out'
os.makedirs(OUT,exist_ok=True)

TRADES=[('CTRN','2026-08-03',69.43,229),('GO','2026-08-03',9.65,191),('LIND','2026-08-03',32.07,761),('AFRM','2026-08-04',79.51,-17),('LQDA','2026-08-04',87.4,339),('UFI','2026-08-04',6.77,23),('ACAD','2026-08-05',28.64,15),('OSPN','2026-08-05',16.72,-411),('SATL','2026-08-06',5.06,326),('SDGR','2026-08-06',17.84,-12),('LMT','2026-08-10',604.1,-562),('PBF','2026-08-11',69.09,578),('XNCR','2026-08-11',22.01,6063),('KNSA','2026-08-12',79,-347),('OTLY','2026-08-13',13.73,1935),('OII','2026-08-14',52.72,-108),('OPHC','2026-08-14',9.39,340),('RDNT','2026-08-20',77.75,-600),('MATV','2026-08-21',12.44,-674),('III','2026-08-24',5.01,-396),('JRSH','2026-08-24',5.65,-191),('QMCO','2026-08-26',22.67,-22),('TPG','2026-08-27',53.74,-525),('FORR','2026-08-28',12.52,-570),('RCMT','2026-08-31',42.61,-448)]

def norm(df,sym):
    if df is None or df.empty:return None
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        if sym in x.columns.get_level_values(0):x=x[sym]
        elif sym in x.columns.get_level_values(1):x=x.xs(sym,axis=1,level=1)
    x.columns=[str(c).lower() for c in x.columns]
    need=['open','high','low','close','volume']
    if not all(c in x.columns for c in need):return None
    x=x[need].dropna(subset=['open','high','low','close']).copy(); idx=pd.to_datetime(x.index)
    if idx.tz is None:idx=idx.tz_localize('America/New_York')
    else:idx=idx.tz_convert('America/New_York')
    x.index=idx
    return x.sort_index().between_time('09:30','15:59')

def fetch(sym):
    try:
        d=yf.download(sym,start='2026-07-24',end='2026-09-12',interval='30m',auto_adjust=False,prepost=False,progress=False,threads=False)
        return sym,norm(d,sym)
    except Exception:return sym,None

def features(x,date):
    if x is None or x.empty:return None
    target=pd.Timestamp(date).date(); day=x[x.index.date==target]
    if len(day)<3:return None
    c1,c2,c3=day.iloc[0],day.iloc[1],day.iloc[2]
    prior=x[x.index<day.index[0]]
    if len(prior)<20:return None
    dates=sorted(set(prior.index.date)); prevdates=dates[-3:]
    prevday=prior[prior.index.date==dates[-1]]
    prior3=prior[np.isin(prior.index.date,prevdates)]
    p12=prior.tail(12); p20=prior.tail(20); p50=prior.tail(50)
    med_rng=float((p20.high-p20.low).median()); med_body=float((p20.close-p20.open).abs().median()); vol50=float(p50.volume.mean())
    prevday_high=float(prevday.high.max()); prior3_high=float(prior3.high.max()); local12_high=float(p12.high.max()); prior_close=float(prior.iloc[-1].close)
    # horizontal resistance can be the nearest of local 12-bar, prior-day, or 3-day levels above/near the prior close.
    levels=sorted(set([prevday_high,local12_high,prior3_high]))
    near_levels=[v for v in levels if v>=prior_close*0.985]
    resistance=min(near_levels) if near_levels else local12_high
    def candle_stats(b):
        rng=float(b.high-b.low); body=float(b.close-b.open); cp=(b.close-b.low)/rng if rng>0 else .5
        return body/(med_body if med_body>0 else 1e-9),rng/(med_rng if med_rng>0 else 1e-9),cp,float(b.volume)/(vol50 if vol50>0 else np.nan)
    b1,r1,cp1,v1=candle_stats(c1); b2,r2,cp2,v2=candle_stats(c2)
    # Break/hold above a visible pre-existing intraday/daily resistance level in either of the first two bars.
    break_c1=bool(c1.close>resistance and prior_close<=resistance)
    break_c2=bool(c2.close>resistance and c1.close<=resistance)
    break_prevday=bool(max(c1.close,c2.close)>prevday_high and prior_close<=prevday_high)
    break_3day=bool(max(c1.close,c2.close)>prior3_high and prior_close<=prior3_high)
    resistance_break=bool(break_c1 or break_c2 or break_prevday or break_3day)
    # Levelling: final four 30m bars before the entry day are relatively flat/tight versus normal intraday range.
    p4=prior.tail(4); closes=p4.close.to_numpy(float); spread=(closes.max()-closes.min())/(med_rng if med_rng>0 else 1e-9); net=abs(closes[-1]-closes[0])/(med_rng if med_rng>0 else 1e-9)
    levelling=bool(spread<=2.0 and net<=1.25)
    # Upward thrust can occur in C1 or C2. It must be an expansion green candle that closes strongly.
    thrust1=bool(b1>=1.5 and r1>=1.25 and cp1>=.70 and (pd.isna(v1) or v1>=.8))
    thrust2=bool(b2>=1.5 and r2>=1.25 and cp2>=.70 and (pd.isna(v2) or v2>=.8))
    level_thrust=bool(levelling and (thrust1 or thrust2))
    # First-hour hold: avoids treating a pure opening spike that is immediately fully rejected as the same quality of structure.
    hour_low=min(float(c1.low),float(c2.low)); hour_high=max(float(c1.high),float(c2.high)); hour_rng=hour_high-hour_low; hour_close_pos=(float(c2.close)-hour_low)/hour_rng if hour_rng>0 else .5
    holds=bool(hour_close_pos>=.45)
    valid=bool((resistance_break or level_thrust) and holds)
    entry=float(c3.open); future=x[x.index>=day.index[2]]; sess=[]
    for dd in future.index.date:
        if dd not in sess:sess.append(dd)
    def horizon(n):
        f=future[np.isin(future.index.date,sess[:n])]
        if f.empty or entry<=0:return np.nan,np.nan,np.nan
        return (float(f.high.max())/entry-1)*100,(float(f.low.min())/entry-1)*100,(float(f.iloc[-1].close)/entry-1)*100
    mfe5,mae5,ret5=horizon(5); mfe10,mae10,ret10=horizon(10)
    return dict(entry_c3=entry,resistance=resistance,break_c1=int(break_c1),break_c2=int(break_c2),break_prevday=int(break_prevday),break_3day=int(break_3day),resistance_break=int(resistance_break),levelling=int(levelling),thrust_c1=int(thrust1),thrust_c2=int(thrust2),level_thrust=int(level_thrust),first_hour_hold=int(holds),valid_structure=int(valid),c1_body_ratio=b1,c1_range_ratio=r1,c1_close_pos=cp1,c1_vol_ratio=v1,c2_body_ratio=b2,c2_range_ratio=r2,c2_close_pos=cp2,c2_vol_ratio=v2,level_spread_norm=spread,level_net_norm=net,mfe5_pct=mfe5,mae5_pct=mae5,ret5_pct=ret5,mfe10_pct=mfe10,mae10_pct=mae10,ret10_pct=ret10)

def main():
    final=pd.read_csv(ART); flagged=final[final.watchlist_pass==1].copy(); flagged['date']=pd.to_datetime(flagged.date)
    actual=pd.DataFrame(TRADES,columns=['symbol','date','actual_entry','actual_pnl']); actual.date=pd.to_datetime(actual.date)
    syms=sorted(set(flagged.symbol)|set(actual.symbol)); print('flagged',len(flagged),'symbols',len(set(flagged.symbol)),'download symbols',len(syms))
    data={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut=[ex.submit(fetch,s) for s in syms]
        for i,f in enumerate(as_completed(fut),1):
            s,x=f.result();data[s]=x
            if i%50==0:print('downloaded',i,'/',len(fut))
    rows=[]
    for _,r in flagged.iterrows():
        f=features(data.get(r.symbol),r.date)
        rr=r.to_dict()
        if f:rr.update(f)
        rows.append(rr)
    opp=pd.DataFrame(rows); opp.to_csv(OUT+'/refined_opportunities.csv',index=False)
    a=[]
    for _,r in actual.iterrows():
        f=features(data.get(r.symbol),r.date); rr=r.to_dict(); rr['win']=int(r.actual_pnl>0)
        if f:
            rr.update(f); rr['entry_match_pct']=(r.actual_entry/f['entry_c3']-1)*100
        a.append(rr)
    ad=pd.DataFrame(a);ad.to_csv(OUT+'/refined_actual.csv',index=False)
    summ={}
    az=ad.dropna(subset=['valid_structure']) if 'valid_structure' in ad else ad.iloc[0:0]
    summ['actual_n']=int(len(az)); summ['actual_valid_rate']=float(az.valid_structure.mean()) if len(az) else None
    for name,g in [('winners',az[az.win==1]),('losers',az[az.win==0])]:
        summ[name]={'n':int(len(g)),'valid_rate':float(g.valid_structure.mean()) if len(g) else None,'resistance_break_rate':float(g.resistance_break.mean()) if len(g) else None,'level_thrust_rate':float(g.level_thrust.mean()) if len(g) else None,'c1_thrust_rate':float(g.thrust_c1.mean()) if len(g) else None,'c2_thrust_rate':float(g.thrust_c2.mean()) if len(g) else None,'median_entry_match_abs_pct':float(g.entry_match_pct.abs().median()) if len(g) else None}
    z=opp.dropna(subset=['valid_structure']) if 'valid_structure' in opp else opp.iloc[0:0]; valid=z[z.valid_structure==1]
    summ['flagged_with_intraday']=int(len(z));summ['valid_structure_n']=int(len(valid));summ['valid_structure_rate']=float(len(valid)/len(z)) if len(z) else None
    if len(valid):
        summ['valid_outcomes']={'median_mfe5':float(valid.mfe5_pct.median()),'median_mae5':float(valid.mae5_pct.median()),'median_mfe10':float(valid.mfe10_pct.median()),'median_ret10':float(valid.ret10_pct.median()),'pct_mfe10_ge10':float((valid.mfe10_pct>=10).mean()),'pct_mfe10_ge20':float((valid.mfe10_pct>=20).mean())}
        for lo,hi in [(35,41),(41,50),(50,60),(60,101)]:
            g=valid[(valid.ma_score>=lo)&(valid.ma_score<hi)];summ[f'score_{lo}_{hi}']={'n':int(len(g)),'median_mfe10':float(g.mfe10_pct.median()) if len(g) else None,'median_mae5':float(g.mae5_pct.median()) if len(g) else None,'median_ret10':float(g.ret10_pct.median()) if len(g) else None,'pct_mfe10_ge10':float((g.mfe10_pct>=10).mean()) if len(g) else None}
    with open(OUT+'/summary.json','w') as f:json.dump(summ,f,indent=2)
    print(json.dumps(summ,indent=2))

if __name__=='__main__':main()

import os, math
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

TRADES = [
 ('CTRN','2026-08-03',69.43,229),('GO','2026-08-03',9.65,191),('LIND','2026-08-03',32.07,761),
 ('AFRM','2026-08-04',79.51,-17),('LQDA','2026-08-04',87.4,339),('UFI','2026-08-04',6.77,23),
 ('ACAD','2026-08-05',28.64,15),('OSPN','2026-08-05',16.72,-411),('SATL','2026-08-06',5.06,326),
 ('SDGR','2026-08-06',17.84,-12),('LMT','2026-08-10',604.1,-562),('PBF','2026-08-11',69.09,578),
 ('XNCR','2026-08-11',22.01,6063),('KNSA','2026-08-12',79,-347),('OTLY','2026-08-13',13.73,1935),
 ('OII','2026-08-14',52.72,-108),('OPHC','2026-08-14',9.39,340),('RDNT','2026-08-20',77.75,-600),
 ('MATV','2026-08-21',12.44,-674),('III','2026-08-24',5.01,-396),('JRSH','2026-08-24',5.65,-191),
 ('QMCO','2026-08-26',22.67,-22),('TPG','2026-08-27',53.74,-525),('FORR','2026-08-28',12.52,-570),('RCMT','2026-08-31',42.61,-448)
]
OUT=Path('tmp/entry_charts'); OUT.mkdir(parents=True,exist_ok=True)

def norm(df,sym):
    if df is None or df.empty:return None
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        if sym in x.columns.get_level_values(0): x=x[sym]
        elif sym in x.columns.get_level_values(1): x=x.xs(sym,axis=1,level=1)
    x.columns=[str(c).lower() for c in x.columns]
    need=['open','high','low','close','volume']
    if not all(c in x.columns for c in need):return None
    x=x[need].dropna().copy(); idx=pd.to_datetime(x.index)
    if idx.tz is None:idx=idx.tz_localize('America/New_York')
    else:idx=idx.tz_convert('America/New_York')
    x.index=idx
    return x.between_time('09:30','15:59')

for sym,ds,entry,pnl in TRADES:
    try:
        d=pd.Timestamp(ds)
        raw=yf.download(sym,start=(d-pd.Timedelta(days=8)).strftime('%Y-%m-%d'),end=(d+pd.Timedelta(days=3)).strftime('%Y-%m-%d'),interval='30m',auto_adjust=False,prepost=False,progress=False,threads=False)
        x=norm(raw,sym)
        if x is None or x.empty: continue
        dates=sorted(set(x.index.date)); target=d.date()
        if target not in dates: continue
        ix=dates.index(target); keep=set(dates[max(0,ix-3):min(len(dates),ix+2)])
        y=x[np.isin(x.index.date,list(keep))].copy()
        # infer first bar on entry day whose range contains the avg entry
        td=y[y.index.date==target]; candidates=td[(td.low<=entry)&(td.high>=entry)]
        inferred=candidates.index[0] if not candidates.empty else None
        prevdates=[z for z in dates if z<target]; prevhigh=np.nan
        if prevdates:
            pdx=x[x.index.date==prevdates[-1]]; prevhigh=float(pdx.high.max()) if len(pdx) else np.nan
        title=f'{sym} {ds}  P&L ${pnl:+.0f}  entry {entry:.2f}'
        hlines=[entry]; colors=['dodgerblue']; styles=['--']; widths=[1.4]
        if not pd.isna(prevhigh): hlines.append(prevhigh);colors.append('gray');styles.append(':');widths.append(1.0)
        ap=[]
        fig,axes=mpf.plot(y.rename(columns=str.capitalize),type='candle',volume=True,style='charles',figsize=(12,6),title=title,hlines=dict(hlines=hlines,colors=colors,linestyle=styles,linewidths=widths),returnfig=True,warn_too_much_data=1000)
        ax=axes[0]
        # mark session starts and entry-time candidate
        for dt in sorted(set(y.index.date)):
            first=y[y.index.date==dt].index[0]
            loc=y.index.get_loc(first); ax.axvline(loc,color='lightgray',alpha=.35,linewidth=.8)
        if inferred is not None:
            loc=y.index.get_loc(inferred); ax.axvline(loc,color='purple',alpha=.8,linewidth=1.2)
            ax.text(loc,ax.get_ylim()[1],' inferred entry bar',rotation=90,va='top',fontsize=8)
        fig.tight_layout(); fig.savefig(OUT/f'{ds}_{sym}_{"WIN" if pnl>0 else "LOSS"}.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    except Exception as e:
        print('ERR',sym,e)
print('charts',len(list(OUT.glob('*.png'))))

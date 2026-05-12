"""Quick robustness check for lqd_mom4_z."""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings('ignore')

START,END,OOS_START,MIN_TRAIN = '2022-04-26','2026-04-26','2025-01-03',100
BASE_R2 = 0.7332

def wk(sym):
    for kw in [dict(start=START,end=END,interval='1wk'),dict(start=START,end=END,interval='1wk',auto_adjust=False)]:
        try:
            dl=yf.download(sym,progress=False,**kw)
            if dl.empty: continue
            if isinstance(dl.columns,pd.MultiIndex): dl.columns=dl.columns.get_level_values(0)
            s=dl['Close'] if 'Close' in dl.columns else dl.iloc[:,0]
            if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
            s=s.dropna(); idx=pd.to_datetime(s.index)
            if getattr(idx,'tz',None) is not None: idx=idx.tz_localize(None)
            s.index=idx; return s.resample('W-FRI').last()
        except: continue
    raise RuntimeError(sym)

def residualize(target,base):
    X=np.column_stack([np.ones(len(base)),base.to_numpy()])
    c,*_=np.linalg.lstsq(X,target.to_numpy(),rcond=None)
    return pd.Series(target.to_numpy()-X@c,index=target.index)

def fit(frame,factors):
    y=frame['log_TSLA'].to_numpy()
    X=np.column_stack([np.ones(len(frame))]+[frame[c].to_numpy() for c in factors])
    b,*_=np.linalg.lstsq(X,y,rcond=None); yhat=X@b; r=y-yhat
    n,k=X.shape; s2=(r@r)/(n-k); cov=s2*np.linalg.pinv(X.T@X)
    se=np.sqrt(np.diag(cov)); t=b/se; p=2*(1-stats.t.cdf(np.abs(t),df=n-k))
    return dict(beta=b,p=p,resid=pd.Series(r,index=frame.index))

def select_events(frame,forced,evs,p_thr=0.10):
    events=[e for e in evs if frame[e].nunique()>1]
    factors=forced+events; m=fit(frame,factors)
    while events:
        ep=[(e,float(m['p'][1+factors.index(e)])) for e in events]
        worst,wp=max(ep,key=lambda x:x[1])
        if wp<=p_thr: break
        events.remove(worst); factors=forced+events; m=fit(frame,factors)
    return factors,events,m

def walk_forward(frame,forced,evs):
    pred=pd.Series(index=frame.index,dtype=float)
    for date,row in frame.iterrows():
        train=frame.loc[:date].iloc[:-1]
        if len(train)<MIN_TRAIN: continue
        facs,_,m_tr=select_events(train,forced,evs)
        xv=np.array([1.0]+[float(row[c]) for c in facs])
        pred.loc[date]=float(xv@m_tr['beta'])
    return pred

def oos_r2(frame,pred):
    idx=pred.dropna().index; idx=idx[idx>=pd.Timestamp(OOS_START)]
    actual=np.exp(frame.loc[idx,'log_TSLA'].to_numpy())
    p=np.exp(pred.loc[idx].to_numpy())
    ss_tot=((actual-actual.mean())**2).sum()
    return float(1-((actual-p)**2).sum()/ss_tot)

print('Fetching...')
raw={k:wk(v) for k,v in {'TSLA':'TSLA','QQQ':'QQQ','DXY':'DX-Y.NYB','VIX':'^VIX','NVDA':'NVDA','ARKK':'ARKK','RBOB':'RB=F','IEF':'IEF','SHY':'SHY','LQD':'LQD'}.items()}
df=pd.DataFrame(raw).resample('W-FRI').last().ffill().dropna()
base=pd.DataFrame(index=df.index)
base['log_TSLA']=np.log(df['TSLA']); base['log_QQQ']=np.log(df['QQQ'])
base['log_DXY']=np.log(df['DXY']); base['log_VIX']=np.log(df['VIX'])
base['NVDA_excess']=residualize(np.log(df['NVDA']),base['log_QQQ'])
base['ARKK_excess']=residualize(np.log(df['ARKK']),base['log_QQQ'])
lr=np.log(df['RBOB']); base['RBOB_zscore_52w']=(lr-lr.rolling(52,min_periods=20).mean())/lr.rolling(52,min_periods=20).std()
cl=np.log(df['IEF'])-np.log(df['SHY']); base['curve_IEF_SHY_zscore_52w']=(cl-cl.rolling(52,min_periods=20).mean())/cl.rolling(52,min_periods=20).std()
EVENT_DEFS=[('Split_squeeze_2020','2020-08-11'),('SP500_inclusion','2020-11-16'),('Hertz_1T_peak','2021-10-25'),('Twitter_overhang','2022-04-25'),('Twitter_close','2022-10-27'),('AI_day_2023','2023-07-19'),('Trump_election','2024-11-06'),('DOGE_brand_damage','2025-02-15'),('Musk_exits_DOGE','2025-04-22'),('TrillionPay','2025-09-05'),('Tariff_shock','2026-02-01'),('Robotaxi_Austin','2025-06-22')]
for name,dt in EVENT_DEFS:
    d0=pd.Timestamp(dt); base[f'E_{name}']=((base.index>=d0)&(base.index<d0+pd.Timedelta(weeks=8))).astype(int)
ll=np.log(df['LQD']); raw_mom4=ll.diff(4)*100
base['lqd_mom4_z']=(raw_mom4-raw_mom4.rolling(52,min_periods=20).mean())/raw_mom4.rolling(52,min_periods=20).std()
FORCED=['log_QQQ','log_DXY','log_VIX','NVDA_excess','ARKK_excess','RBOB_zscore_52w','curve_IEF_SHY_zscore_52w']
f_eq=base.dropna(subset=FORCED)
evs=[f'E_{n}' for n,_ in EVENT_DEFS if f_eq[f'E_{n}'].nunique()>1]

print('lqd_mom4_z robustness:')
tests=[
  ('v6.4 + lqd_mom4_z (primary)',       FORCED+['lqd_mom4_z']),
  ('v6.4 - RBOB + lqd_mom4_z',          [f for f in FORCED if f!='RBOB_zscore_52w']+['lqd_mom4_z']),
  ('v6.4 - curve + lqd_mom4_z',         [f for f in FORCED if f!='curve_IEF_SHY_zscore_52w']+['lqd_mom4_z']),
  ('v6.4 - VIX + lqd_mom4_z',           [f for f in FORCED if f!='log_VIX']+['lqd_mom4_z']),
  ('v6.4 - DXY + lqd_mom4_z',           [f for f in FORCED if f!='log_DXY']+['lqd_mom4_z']),
]
for label,forced in tests:
    sub=f_eq.dropna(subset=[c for c in forced if c in f_eq.columns])
    pred=walk_forward(sub,[c for c in forced if c in sub.columns],evs)
    r2=oos_r2(sub,pred)
    print(f'  {label:<45}  OOS R2={r2:.4f}  delta={((r2-BASE_R2)*100):+.2f}pp')

_,_,mf=select_events(f_eq,FORCED,evs)
resid=mf['resid']
oos_idx=f_eq.index[f_eq.index>=pd.Timestamp(OOS_START)]
mom4_oos=f_eq.loc[oos_idx,'lqd_mom4_z'].dropna()
res_oos=resid.reindex(mom4_oos.index).dropna()
common=mom4_oos.index.intersection(res_oos.index)
c,p=stats.pearsonr(mom4_oos.loc[common],res_oos.loc[common])
print(f'  OOS sub-period corr (lqd_mom4_z vs resid): {c:+.4f} p={p:.4f} n={len(common)}')
print('Done.')

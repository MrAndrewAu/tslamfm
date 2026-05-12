"""Robustness checks for vix_ts_zscore_52w and vix_ts_mom4_z."""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings('ignore')

START, END, OOS_START, MIN_TRAIN = '2022-04-26', '2026-04-26', '2025-01-03', 100
BASE_R2 = 0.7332

def wk(sym):
    for kw in [dict(start=START,end=END,interval='1wk'),
               dict(start=START,end=END,interval='1wk',auto_adjust=False)]:
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
    ss_tot=((y-y.mean())**2).sum(); r2=1-(r**2).sum()/ss_tot
    return dict(beta=b,p=p,r2=float(r2),resid=pd.Series(r,index=frame.index))

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

def oos_subcorr(col,frame,resid):
    idx=frame.index[frame.index>=pd.Timestamp(OOS_START)]
    cand=frame.loc[idx,col].dropna(); res=resid.reindex(cand.index).dropna()
    common=cand.index.intersection(res.index)
    if len(common)<10: return np.nan,np.nan,0
    c,p=stats.pearsonr(cand.loc[common],res.loc[common])
    return c,p,len(common)

print('Fetching...')
raw={k:wk(v) for k,v in {
    'TSLA':'TSLA','QQQ':'QQQ','DXY':'DX-Y.NYB','VIX':'^VIX',
    'NVDA':'NVDA','ARKK':'ARKK','RBOB':'RB=F','IEF':'IEF','SHY':'SHY','VIX3M':'^VIX3M'
}.items()}

df=pd.DataFrame({k:v for k,v in raw.items() if k!='VIX3M'}).resample('W-FRI').last().ffill().dropna()
base=pd.DataFrame(index=df.index)
base['log_TSLA']=np.log(df['TSLA']); base['log_QQQ']=np.log(df['QQQ'])
base['log_DXY']=np.log(df['DXY']); base['log_VIX']=np.log(df['VIX'])
base['NVDA_excess']=residualize(np.log(df['NVDA']),base['log_QQQ'])
base['ARKK_excess']=residualize(np.log(df['ARKK']),base['log_QQQ'])
lr=np.log(df['RBOB']); base['RBOB_zscore_52w']=(lr-lr.rolling(52,min_periods=20).mean())/lr.rolling(52,min_periods=20).std()
cl=np.log(df['IEF'])-np.log(df['SHY']); base['curve_IEF_SHY_zscore_52w']=(cl-cl.rolling(52,min_periods=20).mean())/cl.rolling(52,min_periods=20).std()

EVENT_DEFS=[
    ('Split_squeeze_2020','2020-08-11'),('SP500_inclusion','2020-11-16'),
    ('Hertz_1T_peak','2021-10-25'),('Twitter_overhang','2022-04-25'),
    ('Twitter_close','2022-10-27'),('AI_day_2023','2023-07-19'),
    ('Trump_election','2024-11-06'),('DOGE_brand_damage','2025-02-15'),
    ('Musk_exits_DOGE','2025-04-22'),('TrillionPay','2025-09-05'),
    ('Tariff_shock','2026-02-01'),('Robotaxi_Austin','2025-06-22'),
]
for name,dt in EVENT_DEFS:
    d0=pd.Timestamp(dt); base[f'E_{name}']=((base.index>=d0)&(base.index<d0+pd.Timedelta(weeks=8))).astype(int)

# Build VIX term-structure factors
vix3m=raw['VIX3M'].reindex(base.index,method='ffill')
vix1m=df['VIX'].reindex(base.index,method='ffill')
ts_ratio=np.log(vix3m/vix1m)
ts_zs=(ts_ratio-ts_ratio.rolling(52,min_periods=20).mean())/ts_ratio.rolling(52,min_periods=20).std()
base['vix_ts_zscore_52w']=ts_zs
raw_mom4=ts_ratio.diff(4)
base['vix_ts_mom4_z']=(raw_mom4-raw_mom4.rolling(52,min_periods=20).mean())/raw_mom4.rolling(52,min_periods=20).std()

FORCED=['log_QQQ','log_DXY','log_VIX','NVDA_excess','ARKK_excess','RBOB_zscore_52w','curve_IEF_SHY_zscore_52w']
f_eq=base.dropna(subset=FORCED)
evs=[f'E_{n}' for n,_ in EVENT_DEFS if f_eq[f'E_{n}'].nunique()>1]

_,_,mf=select_events(f_eq,FORCED,evs)
resid_v64=mf['resid']

print('\n--- vix_ts_zscore_52w robustness ---')
tests=[
    ('v6.4 + vix_ts_zscore_52w (primary)',           FORCED+['vix_ts_zscore_52w']),
    ('v6.4 - VIX + vix_ts_zscore_52w',              [f for f in FORCED if f!='log_VIX']+['vix_ts_zscore_52w']),
    ('v6.4 - curve + vix_ts_zscore_52w',             [f for f in FORCED if f!='curve_IEF_SHY_zscore_52w']+['vix_ts_zscore_52w']),
    ('v6.4 - ARKK + vix_ts_zscore_52w',             [f for f in FORCED if f!='ARKK_excess']+['vix_ts_zscore_52w']),
    ('v6.4 - DXY + vix_ts_zscore_52w',              [f for f in FORCED if f!='log_DXY']+['vix_ts_zscore_52w']),
    # Both zscore and mom4 together
    ('v6.4 + vix_ts_zscore_52w + vix_ts_mom4_z',    FORCED+['vix_ts_zscore_52w','vix_ts_mom4_z']),
]
print(f"  {'test':<50}  OOS R2   delta")
for label,forced in tests:
    sub=f_eq.dropna(subset=[c for c in forced if c in f_eq.columns])
    pred=walk_forward(sub,[c for c in forced if c in sub.columns],evs)
    r2=oos_r2(sub,pred)
    print(f'  {label:<50}  {r2:.4f}  {((r2-BASE_R2)*100):+.2f}pp')

c_oos,p_oos,n_oos=oos_subcorr('vix_ts_zscore_52w',f_eq,resid_v64)
print(f'  OOS sub-period corr (vix_ts_zscore_52w): {c_oos:+.4f}  p={p_oos:.4f}  n={n_oos}')

print('\n--- vix_ts_mom4_z robustness ---')
tests2=[
    ('v6.4 + vix_ts_mom4_z (primary)',               FORCED+['vix_ts_mom4_z']),
    ('v6.4 - VIX + vix_ts_mom4_z',                  [f for f in FORCED if f!='log_VIX']+['vix_ts_mom4_z']),
    ('v6.4 - curve + vix_ts_mom4_z',                 [f for f in FORCED if f!='curve_IEF_SHY_zscore_52w']+['vix_ts_mom4_z']),
]
print(f"  {'test':<50}  OOS R2   delta")
for label,forced in tests2:
    sub=f_eq.dropna(subset=[c for c in forced if c in f_eq.columns])
    pred=walk_forward(sub,[c for c in forced if c in sub.columns],evs)
    r2=oos_r2(sub,pred)
    print(f'  {label:<50}  {r2:.4f}  {((r2-BASE_R2)*100):+.2f}pp')

c_oos2,p_oos2,n_oos2=oos_subcorr('vix_ts_mom4_z',f_eq,resid_v64)
print(f'  OOS sub-period corr (vix_ts_mom4_z): {c_oos2:+.4f}  p={p_oos2:.4f}  n={n_oos2}')

print('\nDone.')

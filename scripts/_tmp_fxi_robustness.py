"""FXI robustness checks: tenant-swap tests and OOS sub-period correlation."""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings('ignore')

START, END = '2022-04-26', '2026-04-26'
OOS_START  = '2025-01-03'
MIN_TRAIN  = 100
EVENT_P_THR = 0.10

def wk(sym):
    for kw in [dict(start=START,end=END,interval='1wk'),
               dict(start=START,end=END,interval='1wk',auto_adjust=False)]:
        try:
            dl = yf.download(sym, progress=False, **kw)
            if dl.empty: continue
            if isinstance(dl.columns, pd.MultiIndex):
                dl.columns = dl.columns.get_level_values(0)
            s = dl['Close'] if 'Close' in dl.columns else dl.iloc[:,0]
            if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
            s = s.dropna(); idx = pd.to_datetime(s.index)
            if getattr(idx,'tz',None) is not None: idx = idx.tz_localize(None)
            s.index = idx; return s.resample('W-FRI').last()
        except: continue
    raise RuntimeError(sym)

def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef,*_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X@coef, index=target.index)

def fit(frame, factors):
    y = frame['log_TSLA'].to_numpy()
    X = np.column_stack([np.ones(len(frame))]+[frame[c].to_numpy() for c in factors])
    b,*_ = np.linalg.lstsq(X,y,rcond=None); yhat=X@b; r=y-yhat
    n,k=X.shape; s2=(r@r)/(n-k); cov=s2*np.linalg.pinv(X.T@X)
    se=np.sqrt(np.diag(cov)); t=b/se; p=2*(1-stats.t.cdf(np.abs(t),df=n-k))
    ss_tot=((y-y.mean())**2).sum(); r2=1-(r**2).sum()/ss_tot
    return dict(beta=b,p=p,r2=float(r2),
                fitted=pd.Series(yhat,index=frame.index),
                resid=pd.Series(r,index=frame.index))

def select_events(frame, forced, candidate_events, p_thr=EVENT_P_THR):
    events=[e for e in candidate_events if frame[e].nunique()>1]
    factors=forced+events; m=fit(frame,factors)
    while events:
        event_ps=[(e,float(m['p'][1+factors.index(e)])) for e in events]
        worst,worst_p=max(event_ps,key=lambda x:x[1])
        if worst_p<=p_thr: break
        events.remove(worst); factors=forced+events; m=fit(frame,factors)
    return factors,events,m

def walk_forward(frame, forced, candidate_events, min_train=MIN_TRAIN):
    pred_log=pd.Series(index=frame.index,dtype=float)
    for date,row in frame.iterrows():
        train=frame.loc[:date].iloc[:-1]
        if len(train)<min_train: continue
        facs,_,m_tr=select_events(train,forced,candidate_events)
        xv=np.array([1.0]+[float(row[c]) for c in facs])
        pred_log.loc[date]=float(xv@m_tr['beta'])
    return pred_log

def oos_metrics(frame, pred_log):
    idx=pred_log.dropna().index; idx_oos=idx[idx>=pd.Timestamp(OOS_START)]
    actual=np.exp(frame.loc[idx_oos,'log_TSLA'].to_numpy())
    pred=np.exp(pred_log.loc[idx_oos].to_numpy())
    ss_tot=((actual-actual.mean())**2).sum()
    r2=1-((actual-pred)**2).sum()/ss_tot
    mae=np.mean(np.abs(actual-pred))/np.mean(actual)*100
    return float(r2),float(mae),int(len(actual))

print('Fetching...')
raw={k:wk(v) for k,v in {
    'TSLA':'TSLA','QQQ':'QQQ','DXY':'DX-Y.NYB','VIX':'^VIX',
    'NVDA':'NVDA','ARKK':'ARKK','RBOB':'RB=F','IEF':'IEF','SHY':'SHY','FXI':'FXI'
}.items()}
df=pd.DataFrame(raw).resample('W-FRI').last().ffill().dropna()

base=pd.DataFrame(index=df.index)
base['log_TSLA']=np.log(df['TSLA']); base['log_QQQ']=np.log(df['QQQ'])
base['log_DXY']=np.log(df['DXY']); base['log_VIX']=np.log(df['VIX'])
base['NVDA_excess']=residualize(np.log(df['NVDA']),base['log_QQQ'])
base['ARKK_excess']=residualize(np.log(df['ARKK']),base['log_QQQ'])
log_rbob=np.log(df['RBOB'])
base['RBOB_zscore_52w']=(log_rbob-log_rbob.rolling(52,min_periods=20).mean())/log_rbob.rolling(52,min_periods=20).std()
cl=np.log(df['IEF'])-np.log(df['SHY'])
base['curve_IEF_SHY_zscore_52w']=(cl-cl.rolling(52,min_periods=20).mean())/cl.rolling(52,min_periods=20).std()

EVENT_DEFS=[
    ('Split_squeeze_2020','2020-08-11'),('SP500_inclusion','2020-11-16'),
    ('Hertz_1T_peak','2021-10-25'),('Twitter_overhang','2022-04-25'),
    ('Twitter_close','2022-10-27'),('AI_day_2023','2023-07-19'),
    ('Trump_election','2024-11-06'),('DOGE_brand_damage','2025-02-15'),
    ('Musk_exits_DOGE','2025-04-22'),('TrillionPay','2025-09-05'),
    ('Tariff_shock','2026-02-01'),('Robotaxi_Austin','2025-06-22'),
]
for name,dt in EVENT_DEFS:
    d0=pd.Timestamp(dt)
    base[f'E_{name}']=((base.index>=d0)&(base.index<d0+pd.Timedelta(weeks=8))).astype(int)

log_fxi=np.log(df['FXI'])
base['FXI_zscore_52w']=(log_fxi-log_fxi.rolling(52,min_periods=20).mean())/log_fxi.rolling(52,min_periods=20).std()

FORCED=['log_QQQ','log_DXY','log_VIX','NVDA_excess','ARKK_excess','RBOB_zscore_52w','curve_IEF_SHY_zscore_52w']
f_eq=base.dropna(subset=FORCED)
active_events=[f'E_{n}' for n,_ in EVENT_DEFS if f_eq[f'E_{n}'].nunique()>1]

print('\nRobustness walk-forwards...')
print(f"  {'label':<48}  OOS R2    MAE     delta")

def run(label, forced):
    sub=f_eq.dropna(subset=[c for c in forced if c in f_eq.columns])
    pred=walk_forward(sub, [c for c in forced if c in sub.columns], active_events)
    r2,mae,n=oos_metrics(sub, pred)
    return r2, mae, n

base_r2, base_mae, _ = run('v6.4 baseline', FORCED)
print(f"  {'v6.4 baseline':<48}  {base_r2:.4f}  {base_mae:.2f}%  (ref)")

tests = [
    ('v6.4 + FXI_zscore_52w (primary)',       FORCED+['FXI_zscore_52w']),
    ('v6.4 - VIX (alone)',                     [f for f in FORCED if f!='log_VIX']),
    ('v6.4 - VIX + FXI_zscore_52w',           [f for f in FORCED if f!='log_VIX']+['FXI_zscore_52w']),
    ('v6.4 - DXY + FXI_zscore_52w',           [f for f in FORCED if f!='log_DXY']+['FXI_zscore_52w']),
    ('v6.4 - RBOB + FXI_zscore_52w',          [f for f in FORCED if f!='RBOB_zscore_52w']+['FXI_zscore_52w']),
    ('v6.4 - curve + FXI_zscore_52w',         [f for f in FORCED if f!='curve_IEF_SHY_zscore_52w']+['FXI_zscore_52w']),
]
for label, forced in tests:
    r2,mae,_ = run(label, forced)
    delta=(r2-base_r2)*100
    print(f"  {label:<48}  {r2:.4f}  {mae:.2f}%  {delta:+.2f}pp")

# OOS sub-period correlation
print('\nOOS sub-period correlation (FXI_zscore_52w vs v6.4 residual):')
_,_,mf=select_events(f_eq, FORCED, active_events)
resid_v64=mf['resid']
# Full sample
fxi_all=f_eq['FXI_zscore_52w'].dropna()
resid_all=resid_v64.reindex(fxi_all.index).dropna()
common=fxi_all.index.intersection(resid_all.index)
c_full,p_full=stats.pearsonr(fxi_all.loc[common],resid_all.loc[common])
print(f'  Full IS  (n={len(common)}): corr={c_full:.4f} p={p_full:.4f}')
# OOS only
oos_idx=f_eq.index[f_eq.index>=pd.Timestamp(OOS_START)]
fxi_oos=f_eq.loc[oos_idx,'FXI_zscore_52w'].dropna()
resid_oos=resid_v64.reindex(fxi_oos.index).dropna()
common_oos=fxi_oos.index.intersection(resid_oos.index)
c_oos,p_oos=stats.pearsonr(fxi_oos.loc[common_oos],resid_oos.loc[common_oos])
print(f'  OOS only (n={len(common_oos)}): corr={c_oos:.4f} p={p_oos:.4f}')

print('\nDone.')

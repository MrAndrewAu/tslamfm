"""
SMH (VanEck Semiconductor ETF) as a contender to NVDA_excess in v6.3.

Three variants tested against v6.3 walk-forward baseline:
  A. Replace NVDA_excess with SMH_excess
  B. Add SMH_excess alongside NVDA_excess (both in model)
  C. Replace NVDA_excess with SMH_excess AND add NVDA_excess back as a 2nd factor
     (i.e. SMH_excess + NVDA_excess — same as B, but labeled clearly)

SMH_excess = log(SMH) - (a + b*log(QQQ)) : residual after removing market beta,
             same construction as NVDA_excess and ARKK_excess.

Acceptance bar: +2pp WF OOS R² lift vs v6.3 (for additions).
For substitution: if substitute OOS R² >= current within 1pp, accept on parsimony.
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s.resample("W-FRI").last()

def zscore_52(s): return (s-s.rolling(52,min_periods=20).mean())/s.rolling(52,min_periods=20).std()
def residualize(t,b):
    X=np.column_stack([np.ones(len(b)),b.to_numpy()])
    c,*_=np.linalg.lstsq(X,t.to_numpy(),rcond=None)
    return pd.Series(t.to_numpy()-X@c,index=t.index)

print("Fetching data...")
raw={k:wk(v) for k,v in {"TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB","VIX":"^VIX",
     "NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F","IEF":"IEF","SHY":"SHY","SMH":"SMH"}.items()}
df=pd.DataFrame(raw).ffill().dropna()
print(f"  {len(df)} weeks  {df.index[0].date()} → {df.index[-1].date()}")

f=pd.DataFrame(index=df.index)
f["log_TSLA"]=np.log(df["TSLA"]); f["log_QQQ"]=np.log(df["QQQ"])
f["log_DXY"]=np.log(df["DXY"]);   f["log_VIX"]=np.log(df["VIX"])
f["NVDA_excess"]=residualize(np.log(df["NVDA"]),f["log_QQQ"])
f["ARKK_excess"]=residualize(np.log(df["ARKK"]),f["log_QQQ"])
f["SMH_excess"] =residualize(np.log(df["SMH"]), f["log_QQQ"])
f["RBOB_zscore_52w"]=zscore_52(np.log(df["RBOB"]))
f["curve_IEF_SHY_zscore_52w"]=zscore_52(np.log(df["IEF"])-np.log(df["SHY"]))

EVENT_DEFS=[("Split_squeeze_2020","2020-08-11"),("SP500_inclusion","2020-11-16"),
    ("Hertz_1T_peak","2021-10-25"),("Twitter_overhang","2022-04-25"),
    ("Twitter_close","2022-10-27"),("AI_day_2023","2023-07-19"),
    ("Trump_election","2024-11-06"),("DOGE_brand_damage","2025-02-15"),
    ("Musk_exits_DOGE","2025-04-22"),("TrillionPay","2025-09-05"),
    ("Tariff_shock","2026-02-01"),("Robotaxi_Austin","2025-06-22")]
for name,dt in EVENT_DEFS:
    d0=pd.Timestamp(dt)
    f[f"E_{name}"]=((f.index>=d0)&(f.index<d0+pd.Timedelta(weeks=8))).astype(int)
f=f.dropna()
active=[f"E_{n}" for n,_ in EVENT_DEFS if f[f"E_{n}"].nunique()>1]

V63   =["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess",
        "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"]+active
V_sub =["log_QQQ","log_DXY","log_VIX","SMH_excess", "ARKK_excess",
        "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"]+active   # substitute
V_add =["log_QQQ","log_DXY","log_VIX","NVDA_excess","SMH_excess","ARKK_excess",
        "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"]+active   # addition

# ── Descriptive: SMH_excess vs NVDA_excess ────────────────────────────────────
print("\n=== SMH_excess vs NVDA_excess: descriptive ===")
corr_smh_nvda,_=stats.pearsonr(f["SMH_excess"],f["NVDA_excess"])
print(f"  corr(SMH_excess, NVDA_excess) = {corr_smh_nvda:+.3f}")
print(f"  SMH_excess  mean={f['SMH_excess'].mean():+.4f}  std={f['SMH_excess'].std():.4f}")
print(f"  NVDA_excess mean={f['NVDA_excess'].mean():+.4f}  std={f['NVDA_excess'].std():.4f}")

# ── IS full-sample fit for all variants ──────────────────────────────────────
def fit_is(frame, factors):
    y=frame["log_TSLA"].to_numpy()
    X=np.column_stack([np.ones(len(frame))]+[frame[c].to_numpy() for c in factors])
    b,*_=np.linalg.lstsq(X,y,rcond=None)
    n,k=X.shape; r=y-X@b; s2=(r@r)/(n-k)
    cov=s2*np.linalg.pinv(X.T@X); se=np.sqrt(np.diag(cov))
    t=b/se; p=2*(1-stats.t.cdf(np.abs(t),df=n-k))
    r2=1-(r@r)/((y-y.mean())@(y-y.mean()))
    return dict(b=b,se=se,t=t,p=p,r2=r2,factors=factors)

print("\n=== IS regression coefficients ===")
for label,V in [("v6.3 (NVDA)",V63),("v6.3_sub (SMH replaces NVDA)",V_sub),
                ("v6.3_add (NVDA + SMH)",V_add)]:
    m=fit_is(f,V)
    print(f"\n  {label}  IS R²={m['r2']:.4f}")
    for fac in ["log_QQQ","NVDA_excess","SMH_excess","ARKK_excess",
                "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"]:
        if fac not in V: continue
        idx=V.index(fac)+1
        print(f"    {fac:<35} β={m['b'][idx]:>+8.4f}  p={m['p'][idx]:.4f}")

# ── Walk-forward ──────────────────────────────────────────────────────────────
def wf(frame, factors, min_train=104):
    dates=frame.index.tolist(); preds,actuals=[],[]
    for t in range(min_train,len(dates)):
        tr=frame.iloc[:t]; row=frame.iloc[t:t+1]
        y_tr=tr["log_TSLA"].to_numpy()
        X_tr=np.column_stack([np.ones(t)]+[tr[c].to_numpy() for c in factors])
        b,*_=np.linalg.lstsq(X_tr,y_tr,rcond=None)
        X_row=np.column_stack([np.ones(1)]+[row[c].to_numpy() for c in factors])
        preds.append(float((X_row@b).ravel()[0]))
        actuals.append(float(row["log_TSLA"].iloc[0]))
    preds,actuals=np.array(preds),np.array(actuals)
    dates_pred=np.array(dates[min_train:])
    oos=dates_pred>=pd.Timestamp(OOS_START)
    r2_all=1-((actuals-preds)**2).sum()/((actuals-actuals.mean())**2).sum()
    r2_oos=(1-((actuals[oos]-preds[oos])**2).sum()/((actuals[oos]-actuals[oos].mean())**2).sum()) if oos.sum()>5 else np.nan
    mae_oos=(np.mean(np.abs(np.exp(actuals[oos])-np.exp(preds[oos])))/np.mean(np.exp(actuals[oos]))*100) if oos.sum()>5 else np.nan
    return r2_all, r2_oos, mae_oos

print("\n=== Walk-forward OOS comparison ===")
print(f"  {'variant':<40} {'WF_all':>8} {'OOS_R2':>8} {'OOS_MAE':>9}  verdict")
r2_all_base,r2_oos_base,mae_base=wf(f,V63)
print(f"  {'v6.3 baseline (NVDA_excess)':<40} {r2_all_base:>8.4f} {r2_oos_base:>8.4f} {mae_base:>8.2f}%  (ref)")

for label,V,note in [
    ("SMH substitutes NVDA",   V_sub, "substitute"),
    ("SMH added (NVDA + SMH)", V_add, "addition"),
]:
    r2_all,r2_oos,mae=wf(f,V)
    dR2=(r2_oos-r2_oos_base)*100
    if note=="substitute":
        # within 1pp → accept on parsimony; better → prefer
        verdict=("PREFER_SMH" if dR2>=1.0 else ("EQUIV" if dR2>=-1.0 else "KEEP_NVDA"))
    else:
        verdict=("ACCEPT" if dR2>=2.0 else "REJECT")
    print(f"  {label:<40} {r2_all:>8.4f} {r2_oos:>8.4f} {mae:>8.2f}%  {dR2:>+.2f}pp  {verdict}")

# ── Rolling beta: how stable is SMH_excess beta vs NVDA_excess beta? ─────────
print("\n=== Rolling 52w IS beta: SMH_excess vs NVDA_excess ===")
print("  (Assesses which has more stable, reliable signal across sub-periods)")
ROLL=52
roll_nvda=[]; roll_smh=[]
for i in range(ROLL,len(f)):
    sub=f.iloc[i-ROLL:i]
    y=sub["log_TSLA"].to_numpy()
    for col,store in [("NVDA_excess",roll_nvda),("SMH_excess",roll_smh)]:
        X=np.column_stack([np.ones(ROLL),sub["log_QQQ"].to_numpy(),sub[col].to_numpy()])
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        store.append(b[2])

roll_nvda=np.array(roll_nvda); roll_smh=np.array(roll_smh)
print(f"  NVDA_excess rolling β: mean={roll_nvda.mean():+.3f}  std={roll_nvda.std():.3f}  "
      f"min={roll_nvda.min():+.3f}  max={roll_nvda.max():+.3f}")
print(f"  SMH_excess  rolling β: mean={roll_smh.mean():+.3f}  std={roll_smh.std():.3f}  "
      f"min={roll_smh.min():+.3f}  max={roll_smh.max():+.3f}")
print(f"  (Lower std = more stable = preferable as a persistent factor)")

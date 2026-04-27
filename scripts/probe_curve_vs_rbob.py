"""
Orthogonality probe: is curve_IEF_SHY_zscore_52w just a duplicate of
RBOB_zscore_52w? Both ACCEPT against v6 / v6.1 baseline with OOS-dominant
patterns. Need to know if their lifts stack or substitute.
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s

def residualize(t, b):
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, t.to_numpy(), rcond=None)
    return pd.Series(t.to_numpy() - X @ coef, index=t.index)

S = {k: wk(v) for k, v in {
    "TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB","VIX":"^VIX",
    "NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F","IEF":"IEF","SHY":"SHY",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()
f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

def zscore(s, w=52, mp=20):
    return (s - s.rolling(w, min_periods=mp).mean()) / s.rolling(w, min_periods=mp).std()

f["RBOB_z"]  = zscore(np.log(df["RBOB"]))
f["CURVE_z"] = zscore(np.log(df["IEF"]) - np.log(df["SHY"]))

EVENTS = [("Twitter_close","2022-10-27"),("AI_day_2023","2023-07-19"),
          ("Trump_election","2024-11-06"),("DOGE_brand_damage","2025-02-15"),
          ("Musk_exits_DOGE","2025-04-22"),("TrillionPay","2025-09-05"),
          ("Tariff_shock","2026-02-01")]
for n, d in EVENTS:
    d0 = pd.Timestamp(d)
    f[f"E_{n}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
ev = [c for c in f if c.startswith("E_") and f[c].nunique() > 1]
print(f"n={len(f)}  {f.index[0].date()} -> {f.index[-1].date()}")

# Pairwise correlation: are RBOB_z and CURVE_z basically the same signal?
print(f"\nFull-sample corr(RBOB_z, CURVE_z): {f['RBOB_z'].corr(f['CURVE_z']):+.3f}")
oos_idx = f.index >= pd.Timestamp(OOS_START)
print(f"OOS-only   corr(RBOB_z, CURVE_z): {f.loc[oos_idx,'RBOB_z'].corr(f.loc[oos_idx,'CURVE_z']):+.3f}")

# Walk-forward: v6 / v6.1 / v6.1+CURVE / v6+CURVE_only / v6+RBOB+CURVE
V6  = ["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess"] + ev
V61 = V6 + ["RBOB_z"]

def wf(facs, label):
    y = f["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(f))] + [f[c].to_numpy() for c in facs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    r2_in = 1 - ((y-yhat)**2).sum()/((y-y.mean())**2).sum()
    mae_in = float(np.mean(np.abs(np.exp(y-yhat)-1))*100)
    ho = f.loc[OOS_START:]
    oa, op = [], []
    for date, row in ho.iterrows():
        train = f.loc[:date].iloc[:-1]
        if len(train) < 100: continue
        Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in facs])
        bt, *_ = np.linalg.lstsq(Xt, train["log_TSLA"].to_numpy(), rcond=None)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        op.append(float(np.exp(xv@bt)))
        oa.append(float(np.exp(row["log_TSLA"])))
    oa, op = np.array(oa), np.array(op)
    mae = float(np.mean(np.abs(op/oa-1))*100)
    r2 = 1 - ((oa-op)**2).sum()/((oa-oa.mean())**2).sum()
    return r2_in, mae_in, r2, mae, label

print(f"\n{'spec':<40}{'In R2':>9}{'In MAE':>9}{'OOS R2':>9}{'OOS MAE':>10}")
for facs, label in [
    (V6,                       "v6"),
    (V6 + ["CURVE_z"],         "v6 + CURVE_z (no RBOB)"),
    (V61,                      "v6.1 (=v6 + RBOB_z)"),
    (V61 + ["CURVE_z"],        "v6.1 + CURVE_z"),
]:
    r2i, mai, r2, ma, lbl = wf(facs, label)
    print(f"  {lbl:<38}{r2i:>9.4f}{mai:>9.2f}%{r2:>9.4f}{ma:>10.2f}%")

# Coefficients in the joint model
y = f["log_TSLA"].to_numpy()
all_f = V61 + ["CURVE_z"]
X = np.column_stack([np.ones(len(f))] + [f[c].to_numpy() for c in all_f])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
r = y - X @ b
n, k = X.shape
s2 = (r @ r) / (n - k)
se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
from scipy.stats import t as tdist
print(f"\nJoint v6.1+CURVE_z coefficients:")
for nm, bb, ss in zip(["Intercept"] + all_f, b, se):
    tv = bb / ss
    pv = 2 * (1 - tdist.cdf(abs(tv), df=n-k))
    print(f"  {nm:<25} beta={bb:+.4f}  se={ss:.4f}  t={tv:+.2f}  p={pv:.3e}")

"""
#1 Daily frequency rebuild vs v6.3 weekly baseline.

Rebuilds v6.3 factor set at 1-day frequency and compares OOS metrics.
If daily OOS R² > weekly, there is intra-week structure worth exploiting.
If not, weekly aggregation is the right sampling frequency.

Methodology:
  - Same factors as v6.3: log_QQQ, log_DXY, log_VIX, NVDA_excess, ARKK_excess,
    RBOB_zscore_52w, curve_IEF_SHY_zscore_52w + 8 active event dummies
  - z-scores use 260-day rolling window (≈52 weeks × 5 days) — same lookback
  - OOS split: same calendar date 2025-01-03
  - Walk-forward: expanding window, same logic
  - Acceptance: daily WF OOS R² must beat weekly WF OOS R² by +2pp to justify
    switching frequency (switching cost: complexity, illiquidity of daily use)

Also tests: does adding lag-1 daily return to the weekly model help?
  (Would indicate daily momentum/autocorrelation worth capturing)
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
Z_WINDOW_D  = 260   # ≈52 weeks in trading days
Z_MIN_D     = 100

def dl(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1d")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s

def residualize_daily(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)

def zscore_roll(s, window=260, min_p=100):
    mu = s.rolling(window, min_periods=min_p).mean()
    sd = s.rolling(window, min_periods=min_p).std()
    return (s - mu) / sd

def wf_oos(frame, factors, min_train=520):
    """Expanding-window walk-forward. min_train≈2y of trading days."""
    dates = frame.index.tolist()
    preds, actuals = [], []
    for t in range(min_train, len(dates)):
        train = frame.iloc[:t]
        row   = frame.iloc[t:t+1]
        y_tr  = train["log_TSLA"].to_numpy()
        X_tr  = np.column_stack([np.ones(len(train))] +
                                 [train[c].to_numpy() for c in factors])
        b, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        X_row = np.column_stack([np.ones(1)] + [row[c].to_numpy() for c in factors])
        preds.append(float((X_row @ b).ravel()[0]))
        actuals.append(float(row["log_TSLA"].iloc[0]))
    preds, actuals = np.array(preds), np.array(actuals)
    r2  = 1 - ((actuals - preds)**2).sum() / ((actuals - actuals.mean())**2).sum()
    mae = np.mean(np.abs(np.exp(actuals) - np.exp(preds))) / np.mean(np.exp(actuals)) * 100
    return r2, mae

def wf_oos_weekly(frame, factors, min_train=104):
    dates = frame.index.tolist()
    preds, actuals = [], []
    for t in range(min_train, len(dates)):
        train = frame.iloc[:t]
        row   = frame.iloc[t:t+1]
        y_tr  = train["log_TSLA"].to_numpy()
        X_tr  = np.column_stack([np.ones(len(train))] +
                                 [train[c].to_numpy() for c in factors])
        b, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        X_row = np.column_stack([np.ones(1)] + [row[c].to_numpy() for c in factors])
        preds.append(float((X_row @ b).ravel()[0]))
        actuals.append(float(row["log_TSLA"].iloc[0]))
    preds, actuals = np.array(preds), np.array(actuals)
    r2  = 1 - ((actuals - preds)**2).sum() / ((actuals - actuals.mean())**2).sum()
    mae = np.mean(np.abs(np.exp(actuals) - np.exp(preds))) / np.mean(np.exp(actuals)) * 100
    return r2, mae

# ── Download daily ────────────────────────────────────────────────────────────
print("Fetching daily closes...")
tickers = {"TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB",
           "VIX": "^VIX", "NVDA": "NVDA", "ARKK": "ARKK",
           "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY"}
raw = {k: dl(v) for k, v in tickers.items()}
dfd = pd.DataFrame(raw).ffill().dropna()
print(f"  {len(dfd)} trading days, {dfd.index[0].date()} → {dfd.index[-1].date()}")

# ── Build daily feature frame ─────────────────────────────────────────────────
fd = pd.DataFrame(index=dfd.index)
fd["log_TSLA"] = np.log(dfd["TSLA"])
fd["log_QQQ"]  = np.log(dfd["QQQ"])
fd["log_DXY"]  = np.log(dfd["DXY"])
fd["log_VIX"]  = np.log(dfd["VIX"])
fd["NVDA_excess"] = residualize_daily(np.log(dfd["NVDA"]), fd["log_QQQ"])
fd["ARKK_excess"] = residualize_daily(np.log(dfd["ARKK"]), fd["log_QQQ"])
log_rbob = np.log(dfd["RBOB"])
fd["RBOB_zscore_260d"] = zscore_roll(log_rbob, Z_WINDOW_D, Z_MIN_D)
curve_log = np.log(dfd["IEF"]) - np.log(dfd["SHY"])
fd["curve_IEF_SHY_zscore_260d"] = zscore_roll(curve_log, Z_WINDOW_D, Z_MIN_D)

EVENT_DEFS = [
    ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion",   "2020-11-16"),
    ("Hertz_1T_peak",      "2021-10-25"), ("Twitter_overhang",  "2022-04-25"),
    ("Twitter_close",      "2022-10-27"), ("AI_day_2023",       "2023-07-19"),
    ("Trump_election",     "2024-11-06"), ("DOGE_brand_damage", "2025-02-15"),
    ("Musk_exits_DOGE",    "2025-04-22"), ("TrillionPay",       "2025-09-05"),
    ("Tariff_shock",       "2026-02-01"), ("Robotaxi_Austin",   "2025-06-22"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    fd[f"E_{name}"] = ((fd.index >= d0) & (fd.index < d0 + pd.Timedelta(weeks=8))).astype(int)

fd = fd.dropna()
active_events_d = [f"E_{n}" for n, _ in EVENT_DEFS if fd[f"E_{n}"].nunique() > 1]
print(f"  After construction: {len(fd)} days, {len(active_events_d)} active events")

V_DAILY = (["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
            "RBOB_zscore_260d", "curve_IEF_SHY_zscore_260d"] + active_events_d)

# ── IS fit (daily) ────────────────────────────────────────────────────────────
y_d  = fd["log_TSLA"].to_numpy()
X_d  = np.column_stack([np.ones(len(fd))] + [fd[c].to_numpy() for c in V_DAILY])
b_d, *_ = np.linalg.lstsq(X_d, y_d, rcond=None)
r2_is_d  = 1 - ((y_d - X_d @ b_d)**2).sum() / ((y_d - y_d.mean())**2).sum()
mae_is_d = np.mean(np.abs(np.exp(y_d) - np.exp(X_d @ b_d))) / np.mean(np.exp(y_d)) * 100

# OOS simple (IS-trained betas applied to OOS)
oos_d  = fd[fd.index >= pd.Timestamp(OOS_START)]
is_d   = fd[fd.index <  pd.Timestamp(OOS_START)]
y_is_d = is_d["log_TSLA"].to_numpy()
X_is_d = np.column_stack([np.ones(len(is_d))] + [is_d[c].to_numpy() for c in V_DAILY])
b_is_d, *_ = np.linalg.lstsq(X_is_d, y_is_d, rcond=None)
y_oos_d = oos_d["log_TSLA"].to_numpy()
X_oos_d = np.column_stack([np.ones(len(oos_d))] + [oos_d[c].to_numpy() for c in V_DAILY])
pred_oos_d = X_oos_d @ b_is_d
r2_oos_d   = 1 - ((y_oos_d - pred_oos_d)**2).sum() / ((y_oos_d - y_oos_d.mean())**2).sum()
mae_oos_d  = np.mean(np.abs(np.exp(y_oos_d) - np.exp(pred_oos_d))) / np.mean(np.exp(y_oos_d)) * 100

print(f"\n=== Daily model IS vs OOS ===")
print(f"  IS  R²={r2_is_d:.4f}  MAE={mae_is_d:.2f}%  (n={len(fd)} days)")
print(f"  OOS R²={r2_oos_d:.4f}  MAE={mae_oos_d:.2f}%  (n={len(oos_d)} days)")

# ── Walk-forward daily ────────────────────────────────────────────────────────
print("\n  Running daily walk-forward (slow — ~900 expanding fits)...")
r2_wf_d, mae_wf_d = wf_oos(fd, V_DAILY, min_train=520)
print(f"  WF  R²={r2_wf_d:.4f}  MAE={mae_wf_d:.2f}%")

# ── Weekly baseline (v6.3) for comparison ────────────────────────────────────
print("\n=== Weekly v6.3 baseline (re-run for apples-to-apples) ===")
def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s.resample("W-FRI").last()

def zscore_52(s): return (s - s.rolling(52,min_periods=20).mean()) / s.rolling(52,min_periods=20).std()

raw_w = {k: wk(v) for k, v in {"TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB",
         "VIX":"^VIX","NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F","IEF":"IEF","SHY":"SHY"}.items()}
dfw = pd.DataFrame(raw_w).ffill().dropna()
fw = pd.DataFrame(index=dfw.index)
fw["log_TSLA"] = np.log(dfw["TSLA"])
fw["log_QQQ"]  = np.log(dfw["QQQ"])
fw["log_DXY"]  = np.log(dfw["DXY"])
fw["log_VIX"]  = np.log(dfw["VIX"])
def resid_w(t, b): 
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    c, *_ = np.linalg.lstsq(X, t.to_numpy(), rcond=None)
    return pd.Series(t.to_numpy() - X @ c, index=t.index)
fw["NVDA_excess"] = resid_w(np.log(dfw["NVDA"]), fw["log_QQQ"])
fw["ARKK_excess"] = resid_w(np.log(dfw["ARKK"]), fw["log_QQQ"])
fw["RBOB_zscore_52w"] = zscore_52(np.log(dfw["RBOB"]))
fw["curve_IEF_SHY_zscore_52w"] = zscore_52(np.log(dfw["IEF"]) - np.log(dfw["SHY"]))
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    fw[f"E_{name}"] = ((fw.index >= d0) & (fw.index < d0 + pd.Timedelta(weeks=8))).astype(int)
fw = fw.dropna()
active_w = [f"E_{n}" for n, _ in EVENT_DEFS if fw[f"E_{n}"].nunique() > 1]
V63 = ["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess",
       "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"] + active_w

y_w  = fw["log_TSLA"].to_numpy()
X_w  = np.column_stack([np.ones(len(fw))] + [fw[c].to_numpy() for c in V63])
b_w, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
r2_is_w  = 1 - ((y_w - X_w @ b_w)**2).sum() / ((y_w - y_w.mean())**2).sum()
oos_w    = fw[fw.index >= pd.Timestamp(OOS_START)]
is_w_fr  = fw[fw.index <  pd.Timestamp(OOS_START)]
y_is_w2  = is_w_fr["log_TSLA"].to_numpy()
X_is_w2  = np.column_stack([np.ones(len(is_w_fr))] + [is_w_fr[c].to_numpy() for c in V63])
b_is_w2, *_ = np.linalg.lstsq(X_is_w2, y_is_w2, rcond=None)
y_oos_w  = oos_w["log_TSLA"].to_numpy()
X_oos_w  = np.column_stack([np.ones(len(oos_w))] + [oos_w[c].to_numpy() for c in V63])
pred_w   = X_oos_w @ b_is_w2
r2_oos_w = 1 - ((y_oos_w - pred_w)**2).sum() / ((y_oos_w - y_oos_w.mean())**2).sum()
mae_oos_w= np.mean(np.abs(np.exp(y_oos_w) - np.exp(pred_w)))/np.mean(np.exp(y_oos_w))*100

r2_wf_w, mae_wf_w = wf_oos_weekly(fw, V63)
print(f"  IS  R²={r2_is_w:.4f}  n={len(fw)} weeks")
print(f"  OOS R²={r2_oos_w:.4f}  MAE={mae_oos_w:.2f}%  (n={len(oos_w)} weeks)")
print(f"  WF  R²={r2_wf_w:.4f}  MAE={mae_wf_w:.2f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Frequency comparison summary ===")
print(f"  {'metric':<30} {'daily':>10} {'weekly':>10} {'winner'}")
print(f"  {'IS R²':<30} {r2_is_d:>10.4f} {r2_is_w:>10.4f}  {'daily' if r2_is_d > r2_is_w else 'weekly'}")
print(f"  {'OOS R² (fixed IS betas)':<30} {r2_oos_d:>10.4f} {r2_oos_w:>10.4f}  {'daily' if r2_oos_d > r2_oos_w else 'weekly'}")
print(f"  {'WF OOS R²':<30} {r2_wf_d:>10.4f} {r2_wf_w:>10.4f}  {'daily' if r2_wf_d > r2_wf_w else 'weekly'}")
print(f"  {'WF OOS MAE':<30} {mae_wf_d:>9.2f}% {mae_wf_w:>9.2f}%  {'daily' if mae_wf_d < mae_wf_w else 'weekly'}")
dR2_wf = (r2_wf_d - r2_wf_w) * 100
verdict = "SWITCH to daily" if dR2_wf >= 2.0 else "KEEP weekly"
print(f"\n  Daily WF lift vs weekly: {dR2_wf:+.2f}pp  →  {verdict}")

# ── Lag-1 daily return added to weekly model ──────────────────────────────────
print("\n=== Bonus: lag-1 daily TSLA return added to weekly model ===")
# Use Friday close-to-close daily return, align to weekly frame
daily_ret = np.log(dfd["TSLA"]).diff().resample("W-FRI").last()
fw["lag1d_ret"] = daily_ret.shift(1).reindex(fw.index)
fw2 = fw.dropna()
V63_lag = V63 + ["lag1d_ret"]
r2_wf_lag, mae_wf_lag = wf_oos_weekly(fw2, V63_lag)
dR2_lag = (r2_wf_lag - r2_wf_w) * 100
print(f"  v6.3 + lag1d_ret  WF R²={r2_wf_lag:.4f}  MAE={mae_wf_lag:.2f}%  lift={dR2_lag:+.2f}pp")
print(f"  {'ACCEPT' if dR2_lag >= 2.0 else 'REJECT'} (bar: +2pp)")

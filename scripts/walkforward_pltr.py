"""
Walk-forward OOS test: v6 baseline vs v6 + PLTR_zscore_52w.

STATUS: REJECTED (April 2026)
  Notable case: PLTR_zscore_52w PASSED BOTH pre-gates that other rejects failed.
    - Lagged corr_t+1 = +0.323 (p<0.001, n=190)
    - OOS sub-period corr = +0.245 (p=0.043) -- same sign, still significant
  Yet walk-forward still failed:
    - v6 baseline:           OOS R2 = 0.6586  MAE = 7.81%
    - v6 + PLTR_zscore_52w:  OOS R2 = 0.5538  MAE = 8.27%   (-10.48pp)
  Likely cause: multicollinearity with NVDA_excess / ARKK_excess. The factor
  adds in-sample fit (R2 0.854 -> 0.887) but coefficient is unstable across
  rolling refits, so OOS predictions degrade.

  COST_excess_vs_QQQ never reached walk-forward (failed OOS sub-period gate:
  full-sample lagged corr -0.193 p=0.005 but OOS sub-period flipped to +0.046).
  See analyze_equity_signal.py output for both candidate-tests.

KEY LESSON: Even passing the OOS sub-period sanity check is NOT sufficient.
Walk-forward R^2 lift is the only gate that matters. Multicollinearity with
existing factors can void otherwise-valid univariate signals.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s

def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)

print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "PLTR": "PLTR",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

log_pltr = np.log(df["PLTR"])
rmean = log_pltr.rolling(52, min_periods=20).mean()
rstd  = log_pltr.rolling(52, min_periods=20).std()
f["PLTR_zscore_52w"] = (log_pltr - rmean) / rstd

# Events
EVENT_DEFS = [
    ("Split_squeeze_2020", "2020-08-11"),
    ("SP500_inclusion",    "2020-11-16"),
    ("Hertz_1T_peak",      "2021-10-25"),
    ("Twitter_overhang",   "2022-04-25"),
    ("Twitter_close",      "2022-10-27"),
    ("AI_day_2023",        "2023-07-19"),
    ("Trump_election",     "2024-11-06"),
    ("DOGE_brand_damage",  "2025-02-15"),
    ("Musk_exits_DOGE",    "2025-04-22"),
    ("TrillionPay",        "2025-09-05"),
    ("Tariff_shock",       "2026-02-01"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"  n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}")

active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
print(f"  active events: {len(active_events)}")

V6 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess"] + active_events
V7 = V6 + ["PLTR_zscore_52w"]

def walk_forward(frame, factors, label):
    y_full = frame["log_TSLA"].to_numpy()
    X_full = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b_in, *_ = np.linalg.lstsq(X_full, y_full, rcond=None)
    yhat_in = X_full @ b_in
    r2_in = 1 - ((y_full - yhat_in) ** 2).sum() / ((y_full - y_full.mean()) ** 2).sum()
    mae_in = float(np.mean(np.abs(np.exp(y_full - yhat_in) - 1)) * 100)

    ho = frame.loc[OOS_START:]
    oos_actual, oos_pred = [], []
    for date, row in ho.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < 100:
            continue
        Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in factors])
        yt = train["log_TSLA"].to_numpy()
        bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        xv = np.array([1.0] + [float(row[c]) for c in factors])
        oos_pred.append(float(np.exp(xv @ bt)))
        oos_actual.append(float(np.exp(row["log_TSLA"])))
    oos_actual = np.array(oos_actual); oos_pred = np.array(oos_pred)
    mae_oos = float(np.mean(np.abs(oos_pred / oos_actual - 1)) * 100)
    ss_tot = ((oos_actual - oos_actual.mean()) ** 2).sum()
    ss_res = ((oos_actual - oos_pred) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    print(f"  {label:<32}  In R2={r2_in:.4f}  In MAE={mae_in:5.2f}%  |  OOS R2={r2_oos:.4f}  OOS MAE={mae_oos:5.2f}%  (n_oos={len(oos_actual)})")
    return r2_oos, mae_oos

print("\nWalk-forward results:")
r2_v6, _ = walk_forward(f, V6, "v6 baseline")
r2_v7, _ = walk_forward(f, V7, "v6 + PLTR_zscore_52w")
print(f"\n  Delta OOS R^2: {(r2_v7 - r2_v6)*100:+.2f}pp")
print(f"  Acceptance bar: >= +2.00pp")
print(f"  Verdict: {'ACCEPT' if (r2_v7 - r2_v6) >= 0.02 else 'REJECT'}")

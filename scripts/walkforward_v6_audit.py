"""
Leave-one-out walk-forward audit of v6 factors.

STATUS: ALL FACTORS PASS (April 2026)
  Each v6 factor earns at least the +2pp OOS R^2 bar applied to candidates:
    log_QQQ      : +187.91pp  (structural; model collapses without it)
    ARKK_excess  :  +20.48pp  (biggest single-factor contributor)
    log_DXY      :  +17.89pp
    NVDA_excess  :   +7.49pp  (decisively earns its spot)
    all events   :   +2.66pp
    log_VIX      :   +2.19pp  (marginal; weakest tenant, watch it)

Question: do NVDA_excess and ARKK_excess actually earn their spot in v6,
under the same walk-forward bar we apply to candidate factors?

Method: refit v6 with each factor dropped in turn; compare OOS R^2.
Acceptance: factor must contribute >= +2pp OOS R^2 (same bar as candidates).

NOTE: Candidate-test (lagged corr + OOS sub-period) returns 0.000 by OLS
construction for any factor already in the model -- v6 residuals are
orthogonal to the regressors that fit them. Only leave-one-out walk-forward
is meaningful for in-model factors.
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
    "NVDA": "NVDA", "ARKK": "ARKK",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

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

V6 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess"] + active_events

def walk_forward(frame, factors):
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
    return r2_in, mae_in, r2_oos, mae_oos

print("\nLeave-one-out walk-forward:")
print(f"  {'variant':<32}  {'In R2':>7}  {'In MAE':>7}  {'OOS R2':>7}  {'OOS MAE':>8}  {'dR2 OOS':>9}")
r2_in_full, mae_in_full, r2_oos_full, mae_oos_full = walk_forward(f, V6)
print(f"  {'v6 full':<32}  {r2_in_full:>7.4f}  {mae_in_full:>6.2f}%  {r2_oos_full:>7.4f}  {mae_oos_full:>7.2f}%  {'(ref)':>9}")

drop_targets = ["NVDA_excess", "ARKK_excess", "log_QQQ", "log_DXY", "log_VIX"]
for drop in drop_targets:
    factors = [c for c in V6 if c != drop]
    r2_in, mae_in, r2_oos, mae_oos = walk_forward(f, factors)
    delta = (r2_oos_full - r2_oos) * 100
    print(f"  {'v6 - ' + drop:<32}  {r2_in:>7.4f}  {mae_in:>6.2f}%  {r2_oos:>7.4f}  {mae_oos:>7.2f}%  {delta:>+8.2f}pp")

# Also v6 minus all events (sanity)
factors = [c for c in V6 if not c.startswith("E_")]
r2_in, mae_in, r2_oos, mae_oos = walk_forward(f, factors)
delta = (r2_oos_full - r2_oos) * 100
print(f"  {'v6 - all_events':<32}  {r2_in:>7.4f}  {mae_in:>6.2f}%  {r2_oos:>7.4f}  {mae_oos:>7.2f}%  {delta:>+8.2f}pp")

print("\n  Acceptance: each factor should contribute >= +2.00pp OOS R^2 lift.")

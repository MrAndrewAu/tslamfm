"""
Robustness checks for XLU_zscore_52w before promoting to v6.1.

STATUS: REJECTED -- failed all three robustness tests (April 2026)

Initial walk-forward delta vs v6 was +1.60pp (under the +2pp ex-ante bar).
Three robustness tests run before deciding:

  (a) Variant robustness: FAIL.
        v6 + XLU_zscore_52w     : +1.60pp
        v6 + log_XLU            : -0.54pp
        v6 + XLU_excess_vs_QQQ  : -0.54pp
      Signal concentrated in single transform. Real factors usually show
      up across reasonable transforms; concentration in zscore_52w smells
      like spurious regime-matching to 2024-2026 window.

  (b) Tenant-swap (replace marginal log_VIX): FAIL.
        v6 - log_VIX            : -3.37pp
        v6 - log_VIX + XLU_zsc  : +0.55pp
      XLU is a partial substitute for VIX (both reach for "defensive
      rotation / fear" signal), not a complement. VIX does it better.

  (c) Net contribution on top of v6: still +1.60pp. Never clears +2pp.

Verdict: REJECT. Gate held.
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

print("Fetching...")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "XLU": "XLU",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

log_xlu = np.log(df["XLU"])
f["log_XLU"]            = log_xlu
f["XLU_excess_vs_QQQ"]  = residualize(log_xlu, f["log_QQQ"])
rmean = log_xlu.rolling(52, min_periods=20).mean()
rstd  = log_xlu.rolling(52, min_periods=20).std()
f["XLU_zscore_52w"]     = (log_xlu - rmean) / rstd

EVENT_DEFS = [
    ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion", "2020-11-16"),
    ("Hertz_1T_peak", "2021-10-25"),      ("Twitter_overhang", "2022-04-25"),
    ("Twitter_close", "2022-10-27"),      ("AI_day_2023", "2023-07-19"),
    ("Trump_election", "2024-11-06"),     ("DOGE_brand_damage", "2025-02-15"),
    ("Musk_exits_DOGE", "2025-04-22"),    ("TrillionPay", "2025-09-05"),
    ("Tariff_shock", "2026-02-01"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"  n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}")
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V6 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess"] + active_events

def wf(frame, factors):
    y_full = frame["log_TSLA"].to_numpy()
    X_full = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b_in, *_ = np.linalg.lstsq(X_full, y_full, rcond=None)
    yhat_in = X_full @ b_in
    r2_in = 1 - ((y_full - yhat_in) ** 2).sum() / ((y_full - y_full.mean()) ** 2).sum()
    mae_in = float(np.mean(np.abs(np.exp(y_full - yhat_in) - 1)) * 100)
    ho = frame.loc[OOS_START:]
    oa, op = [], []
    for date, row in ho.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < 100: continue
        Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in factors])
        yt = train["log_TSLA"].to_numpy()
        bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        xv = np.array([1.0] + [float(row[c]) for c in factors])
        op.append(float(np.exp(xv @ bt)))
        oa.append(float(np.exp(row["log_TSLA"])))
    oa = np.array(oa); op = np.array(op)
    mae_oos = float(np.mean(np.abs(op / oa - 1)) * 100)
    ss_tot = ((oa - oa.mean()) ** 2).sum()
    ss_res = ((oa - op) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    return r2_in, mae_in, r2_oos, mae_oos

def show(label, factors, ref_r2_oos):
    r2_in, mae_in, r2_oos, mae_oos = wf(f, factors)
    delta = (r2_oos - ref_r2_oos) * 100 if ref_r2_oos is not None else None
    dstr = f"{delta:+8.2f}pp" if delta is not None else "  (ref) "
    print(f"  {label:<48}  In R2={r2_in:.4f}  In MAE={mae_in:5.2f}%  |  OOS R2={r2_oos:.4f}  OOS MAE={mae_oos:5.2f}%  {dstr}")
    return r2_oos

print("\n--- v6 reference ---")
ref = show("v6 baseline", V6, None)

print("\n--- (a) variant robustness ---")
show("v6 + XLU_zscore_52w", V6 + ["XLU_zscore_52w"], ref)
show("v6 + log_XLU",        V6 + ["log_XLU"], ref)
show("v6 + XLU_excess_vs_QQQ", V6 + ["XLU_excess_vs_QQQ"], ref)

print("\n--- (b) tenant swap (drop log_VIX) ---")
V6_no_vix = [c for c in V6 if c != "log_VIX"]
show("v6 - log_VIX (no XLU)",                V6_no_vix, ref)
show("v6 - log_VIX + XLU_zscore_52w",        V6_no_vix + ["XLU_zscore_52w"], ref)

print("\n--- (c) both VIX and XLU ---")
show("v6 + XLU_zscore_52w (already shown)",  V6 + ["XLU_zscore_52w"], ref)

print("\nGate: ANY variant must clear +2.00pp to promote.")

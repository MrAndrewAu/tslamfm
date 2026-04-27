"""
Generic walk-forward OOS test: v6 baseline vs v6 + a single new factor.

Usage:
    python scripts/walkforward_generic.py XLU zscore_52w
    python scripts/walkforward_generic.py RIVN zscore_52w
    python scripts/walkforward_generic.py AAPL excess_vs_QQQ

Transform options: log, excess_vs_QQQ, zscore_52w

Acceptance: OOS R^2 lift >= +2pp over v6 baseline.
"""
from __future__ import annotations
import sys, warnings, numpy as np, pandas as pd, yfinance as yf
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

def build_factor(sym: str, transform: str, weekly: pd.DataFrame) -> pd.Series:
    log_x = np.log(weekly[sym])
    if transform == "log":
        return log_x
    if transform == "excess_vs_QQQ":
        return residualize(log_x, np.log(weekly["QQQ"]))
    if transform == "zscore_52w":
        rmean = log_x.rolling(52, min_periods=20).mean()
        rstd = log_x.rolling(52, min_periods=20).std()
        return (log_x - rmean) / rstd
    raise ValueError(f"unknown transform: {transform}")

if len(sys.argv) < 3:
    print("usage: walkforward_generic.py SYMBOL TRANSFORM")
    print("  TRANSFORM = log | excess_vs_QQQ | zscore_52w")
    sys.exit(1)
sym, transform = sys.argv[1], sys.argv[2]
factor_name = f"{sym}_{transform}"

print(f"Walk-forward: v6.4 baseline vs v6.4 + {factor_name}")
print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK",
    "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY",
    sym: sym,
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])
log_rbob = np.log(df["RBOB"])
f["RBOB_zscore_52w"] = (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) / log_rbob.rolling(52, min_periods=20).std()
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = (curve_log - curve_log.rolling(52, min_periods=20).mean()) / curve_log.rolling(52, min_periods=20).std()
f[factor_name] = build_factor(sym, transform, df)

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
    ("Robotaxi_Austin",    "2025-06-22"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"  n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}")
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V63 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
       "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_events
V_NEW = V63 + [factor_name]

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
    print(f"  {label:<40}  In R2={r2_in:.4f}  In MAE={mae_in:5.2f}%  |  OOS R2={r2_oos:.4f}  OOS MAE={mae_oos:5.2f}%  (n_oos={len(oos_actual)})")
    return r2_oos, mae_oos


def compute_vif(frame, factors):
    """Variance inflation factor for each factor (1/(1-R²_i) from regressing
    factor_i on all others). High VIF => multicollinear with existing set."""
    X = frame[factors].to_numpy().astype(float)
    vifs = {}
    for j, fname in enumerate(factors):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        Xo = np.column_stack([np.ones(len(others)), others])
        b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        yhat = Xo @ b
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vifs[fname] = 1.0 / max(1.0 - r2, 1e-10)
    return vifs


VIF_WARN = 5.0   # common rule of thumb; > 10 is severe

print()
print("VIF screen (multicollinearity check):")
f_clean = f.dropna(subset=V_NEW)
vifs_new = compute_vif(f_clean, V_NEW)
for fname in V_NEW:
    v = vifs_new[fname]
    flag   = "  ← WARN: multicollinear" if v > VIF_WARN else ""
    marker = "  *candidate*" if fname == factor_name else ""
    print(f"  {fname:<40}  VIF={v:6.2f}{marker}{flag}")
candidate_vif = vifs_new.get(factor_name, float('inf'))
if candidate_vif > VIF_WARN:
    print(f"\n  *** Candidate VIF={candidate_vif:.2f} > {VIF_WARN}: likely multicollinear with existing factors. ***")
    print(f"  Walk-forward collapse expected (PLTR/AAPL pattern). Continuing for documentation.\n")
else:
    print(f"\n  Candidate VIF={candidate_vif:.2f} — OK (below {VIF_WARN} threshold). Proceeding to walk-forward.\n")

r2_v63, _ = walk_forward(f, V63, "v6.4 baseline")
r2_new, _ = walk_forward(f, V_NEW, f"v6.4 + {factor_name}")
delta = (r2_new - r2_v63) * 100
print(f"\n  Delta OOS R^2: {delta:+.2f}pp")
print(f"  Acceptance bar: >= +2.00pp")
verdict = "ACCEPT" if delta >= 2.0 else "REJECT"
print(f"  Verdict: {verdict}")

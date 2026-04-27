"""
Event dummy diagnostic + candidate screening (v6.2 baseline).

Step 1: Find the biggest unexplained residual weeks in history.json.
Step 2: Map them to known TSLA-specific events.
Step 3: Test new event dummies (8-week windows) in joint regression with v6.2.
Step 4: Walk-forward those that pass p<0.10.

Candidate events (post-v6.2-cutoff or missing from current set):
  E_Q1_2025_delivery_miss  2025-04-02  Q1 deliveries: 336k vs ~390k expected (-14%)
  E_Robotaxi_Austin        2025-06-22  Austin commercial robotaxi launch
  E_Q2_2025_earnings       2025-07-23  Q2 2025 earnings (margin recovery narrative)
  E_Q4_2025_earnings       2026-01-29  Q4 2025 earnings / 2026 guidance
  E_Q1_2026_earnings       2026-04-22  Q1 2026 earnings (just announced)
  E_Tariff_prolonged       2026-03-30  Sustained tariff regime post-shock window expiry
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "public" / "data" / "history.json"
START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s

def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)

def zscore_52(s):
    return (s - s.rolling(52, min_periods=20).mean()) / s.rolling(52, min_periods=20).std()

# ---------- residuals from history.json ----------
print("=== Step 1: Largest unexplained residuals in v6.2 ===\n")
rows = json.loads(HIST.read_text())
hist = pd.DataFrame(rows)
hist["date"] = pd.to_datetime(hist["date"])
hist = hist.set_index("date").sort_index()
hist["resid_pct"] = (hist["actual"] / hist["fitted"] - 1) * 100

print(f"  {'date':<12} {'actual':>8} {'fitted':>8} {'resid%':>8}")
print("  " + "-" * 44)
# Top 15 by absolute residual
top = hist["resid_pct"].abs().nlargest(15)
for d, v in top.items():
    r = hist.loc[d]
    print(f"  {str(d.date()):<12} {r['actual']:>8.2f} {r['fitted']:>8.2f} {r['resid_pct']:>+8.1f}%")

print(f"\n  Recent 2025-2026 residuals:")
print(f"  {'date':<12} {'actual':>8} {'fitted':>8} {'resid%':>8}")
print("  " + "-" * 44)
recent = hist.loc["2025-01-01":]
for d, r in recent.iterrows():
    print(f"  {str(d.date()):<12} {r['actual']:>8.2f} {r['fitted']:>8.2f} {r['resid_pct']:>+8.1f}%")

# ---------- rebuild v6.2 frame ----------
print("\n\n=== Step 2: Candidate event dummy screening ===\n")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY",
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
f["RBOB_zscore_52w"] = zscore_52(log_rbob)
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = zscore_52(curve_log)

# Existing events
EXISTING_EVENTS = [
    ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion", "2020-11-16"),
    ("Hertz_1T_peak", "2021-10-25"),      ("Twitter_overhang", "2022-04-25"),
    ("Twitter_close", "2022-10-27"),      ("AI_day_2023", "2023-07-19"),
    ("Trump_election", "2024-11-06"),     ("DOGE_brand_damage", "2025-02-15"),
    ("Musk_exits_DOGE", "2025-04-22"),    ("TrillionPay", "2025-09-05"),
    ("Tariff_shock", "2026-02-01"),
]
for name, dt in EXISTING_EVENTS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

# Candidate new events
CANDIDATE_EVENTS = [
    ("Q1_2025_delivery_miss",  "2025-04-04",  "Q1 2025 deliveries: 336k vs ~390k expected (-14% miss)"),
    ("Robotaxi_Austin",        "2025-06-22",  "Austin commercial robotaxi service launch"),
    ("Q2_2025_earnings",       "2025-07-23",  "Q2 2025 earnings / margin recovery narrative"),
    ("Q4_2025_earnings",       "2026-01-29",  "Q4 2025 earnings + 2026 guidance"),
    ("Q1_2026_earnings",       "2026-04-22",  "Q1 2026 earnings"),
    ("Tariff_prolonged",       "2026-03-30",  "Sustained tariff regime after initial shock window expires"),
]
for name, dt, _ in CANDIDATE_EVENTS:
    d0 = pd.Timestamp(dt)
    f[f"C_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"Frame: n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}\n")

active_existing = [f"E_{n}" for n, _ in EXISTING_EVENTS if f[f"E_{n}"].nunique() > 1]
V62 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
       "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_existing

# Screen each candidate: marginal t-stat in joint regression
y = f["log_TSLA"].to_numpy()

def joint_tstat(candidate_col):
    cols = V62 + [candidate_col]
    X = np.column_stack([np.ones(len(f))] + [f[c].to_numpy() for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    resid = y - X @ b
    s2 = (resid @ resid) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    idx = len(V62) + 1  # last coefficient
    t = b[idx] / se[idx]
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - k))
    return float(b[idx]), float(se[idx]), float(t), float(p)

print(f"  {'event':<30} {'start':<12} {'weeks_active':>12} {'beta':>8} {'se':>8} {'t':>8} {'p':>8}  verdict")
print("  " + "-" * 100)
passes = []
for name, dt, desc in CANDIDATE_EVENTS:
    col = f"C_{name}"
    n_active = int(f[col].sum())
    if n_active == 0:
        print(f"  {name:<30} {dt:<12} {'(no obs)':>12}")
        continue
    beta, se, t, p = joint_tstat(col)
    v = "PASS p<0.10" if p < 0.10 else ("BORDERLINE" if p < 0.20 else "REJECT")
    if p < 0.10: passes.append((name, dt, desc))
    print(f"  {name:<30} {dt:<12} {n_active:>12} {beta:>+8.3f} {se:>8.3f} {t:>+8.2f} {p:>8.4f}  {v}")
    print(f"  {'':30} {desc}")

# ---------- walk-forward for passing candidates ----------
if not passes:
    print("\nNo candidates passed p<0.10. Done.")
    raise SystemExit(0)

print(f"\n\n=== Step 3: Walk-forward for {len(passes)} passing candidate(s) ===\n")

def wf(frame, factors_list, oos_start=OOS_START):
    y_full = frame["log_TSLA"].to_numpy()
    X_full = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors_list])
    b_in, *_ = np.linalg.lstsq(X_full, y_full, rcond=None)
    yhat_in = X_full @ b_in
    r2_in = 1 - ((y_full - yhat_in) ** 2).sum() / ((y_full - y_full.mean()) ** 2).sum()
    mae_in = float(np.mean(np.abs(np.exp(y_full - yhat_in) - 1)) * 100)
    ho = frame.loc[oos_start:]
    oa, op = [], []
    for date, row in ho.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < 100: continue
        Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in factors_list])
        yt = train["log_TSLA"].to_numpy()
        bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        xv = np.array([1.0] + [float(row[c]) for c in factors_list])
        op.append(float(np.exp(xv @ bt)))
        oa.append(float(np.exp(row["log_TSLA"])))
    oa, op = np.array(oa), np.array(op)
    mae = float(np.mean(np.abs(op / oa - 1)) * 100)
    r2_oos = 1 - ((oa - op) ** 2).sum() / ((oa - oa.mean()) ** 2).sum()
    return r2_in, mae_in, r2_oos, mae

print(f"  {'variant':<50}{'In R2':>9}{'In MAE':>9}{'OOS R2':>9}{'OOS MAE':>10}{'dR2':>11}")
r2_ref_in, mae_ref_in, r2_ref, mae_ref = wf(f, V62)
print(f"  {'v6.2 baseline':<50}{r2_ref_in:>9.4f}{mae_ref_in:>9.2f}%{r2_ref:>9.4f}{mae_ref:>10.2f}%{'(ref)':>11}")
accepted = []
for name, dt, desc in passes:
    col = f"C_{name}"
    r2_in, mae_in, r2_oos, mae = wf(f, V62 + [col])
    delta = (r2_oos - r2_ref) * 100
    v = "ACCEPT" if delta >= 1.0 else "REJECT"   # lower bar for events: >= +1pp
    if delta >= 1.0: accepted.append((name, dt, desc))
    print(f"  {'v6.2 + ' + name:<50}{r2_in:>9.4f}{mae_in:>9.2f}%{r2_oos:>9.4f}{mae:>10.2f}%{delta:>+9.2f}pp  {v}")
    print(f"  {'':50} {desc}")

print(f"\n  Acceptance bar: >= +1.00pp (event dummies are sparse; 1pp is the right bar)")
print(f"\n  Accepted: {[n for n,_,_ in accepted] if accepted else 'none'}")

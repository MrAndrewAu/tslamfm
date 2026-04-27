"""
Macro / regime factors — candidate-test + walk-forward in one script.

Targets variance v6 (QQQ/NVDA/ARKK/VIX/DXY + events) doesn't already see:
  - HYG/LQD spread       : credit-cycle stress (high-yield vs investment-grade)
  - Copper/Gold ratio    : global growth vs safe-haven (classic regime)
  - ^MOVE                : bond-market volatility (orthogonal to equity VIX)

For each: build raw and 52w z-score variants. Apply standard candidate gates
(lagged corr p<0.05 & |corr|>=0.15 & n>=50, OOS sub-period same sign and
significant). Anything passing both gates -> walk-forward (>= +2pp OOS R^2
lift over v6).

ETF/instrument fallbacks for things yfinance is flaky on:
  - Copper/Gold:  HG=F / GC=F (futures), fallback to CPER / GLD (ETFs)
  - MOVE:         ^MOVE
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
    if s.empty:
        return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s

def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)

def load_residuals():
    rows = json.loads(HIST.read_text())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return np.log(df["actual"]) - np.log(df["fitted"])

def report(name, factor, resid):
    s = factor.dropna().reindex(resid.index, method="nearest", tolerance=pd.Timedelta("3D")).dropna()
    r = resid.loc[s.index]
    out = {"factor": name, "n": int(len(s))}
    if len(s) < 20:
        return {**out, "status": "insufficient"}
    c0, p0 = stats.pearsonr(s, r)
    out["corr_t"], out["pval_t"] = float(c0), float(p0)
    s_lag = s.iloc[:-1]
    r_next = resid.reindex(s.index).shift(-1).iloc[:-1]
    mask = s_lag.notna() & r_next.notna()
    if mask.sum() >= 20:
        c1, p1 = stats.pearsonr(s_lag[mask], r_next[mask])
        out["corr_t+1"], out["pval_t+1"] = float(c1), float(p1)
    else:
        out["corr_t+1"], out["pval_t+1"] = np.nan, np.nan
    oos_idx = s.index[s.index >= pd.Timestamp(OOS_START)]
    if len(oos_idx) >= 20:
        c_oos, p_oos = stats.pearsonr(s.loc[oos_idx], r.loc[oos_idx])
        out["corr_oos"], out["pval_oos"] = float(c_oos), float(p_oos)
    else:
        out["corr_oos"], out["pval_oos"] = np.nan, np.nan
    return out

def verdict_str(row):
    p = row.get("pval_t+1"); c = row.get("corr_t+1"); n = row.get("n", 0)
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "INCONCLUSIVE"
    if n < 50:
        return f"INCONCLUSIVE (n={n})"
    if p < 0.05 and abs(c) > 0.15:
        return f"PROMISING (corr={c:+.3f}, p={p:.3f})"
    if p < 0.10:
        return f"WEAK (p={p:.3f})"
    return "REJECT"

def passes_oos_gate(row):
    """Same gate we adopted after PLTR §10.8: OOS sub-period must be same sign
    and p<0.10 (relaxed from 0.05 because n_oos~70 is small)."""
    c1 = row.get("corr_t+1"); c_oos = row.get("corr_oos"); p_oos = row.get("pval_oos")
    if c1 is None or c_oos is None or p_oos is None: return False
    if any(isinstance(x, float) and np.isnan(x) for x in [c1, c_oos, p_oos]): return False
    return (np.sign(c1) == np.sign(c_oos)) and (p_oos < 0.10)

# ---------- build candidate factors ----------
print("Fetching weekly closes...")
def safe(sym, fallback=None):
    s = wk(sym)
    if s.empty and fallback:
        print(f"  {sym} empty; falling back to {fallback}")
        s = wk(fallback)
    return s

raw = {
    "HYG":   safe("HYG"),
    "LQD":   safe("LQD"),
    "HG":    safe("HG=F", fallback="CPER"),
    "GC":    safe("GC=F", fallback="GLD"),
    "MOVE":  safe("^MOVE"),
}
for k, s in raw.items():
    print(f"  {k}: {len(s)} rows")

# Align everything weekly
W = pd.DataFrame({k: v for k, v in raw.items()}).resample("W-FRI").last().ffill()

factors = {}
# Credit spread: log(HYG) - log(LQD). Negative = stress (HYG underperforms LQD).
hyg_minus_lqd = np.log(W["HYG"]) - np.log(W["LQD"])
factors["HYG_minus_LQD"] = hyg_minus_lqd
factors["HYG_minus_LQD_zscore_52w"] = (
    (hyg_minus_lqd - hyg_minus_lqd.rolling(52, min_periods=20).mean())
    / hyg_minus_lqd.rolling(52, min_periods=20).std()
)

# Copper/Gold ratio: log(HG) - log(GC). High = growth, low = risk-off.
cu_au = np.log(W["HG"]) - np.log(W["GC"])
factors["CopperGold_log_ratio"] = cu_au
factors["CopperGold_zscore_52w"] = (
    (cu_au - cu_au.rolling(52, min_periods=20).mean())
    / cu_au.rolling(52, min_periods=20).std()
)

# MOVE: bond-vol index
log_move = np.log(W["MOVE"])
factors["log_MOVE"] = log_move
factors["MOVE_zscore_52w"] = (
    (log_move - log_move.rolling(52, min_periods=20).mean())
    / log_move.rolling(52, min_periods=20).std()
)

# ---------- candidate-test ----------
resid = load_residuals()
print(f"\n{'factor':<32}{'n':>5}{'corr_t':>10}{'p':>8}{'corr_t+1':>11}{'p':>8}{'corr_oos':>11}{'p':>8}  verdict")
print("-" * 122)
def fmt(x, fmt="{:>+.3f}"):
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return fmt.format(x)
    return "  n/a"

results = []
for name, s in factors.items():
    r = report(name, s, resid)
    r["verdict"] = verdict_str(r)
    r["passes_both_gates"] = (
        ("PROMISING" in r["verdict"]) and passes_oos_gate(r)
    )
    results.append(r)
    print(f"{name:<32}"
          f"{r.get('n',0):>5}"
          f"{fmt(r.get('corr_t')):>10}"
          f"{fmt(r.get('pval_t'), '{:>.3f}'):>8}"
          f"{fmt(r.get('corr_t+1')):>11}"
          f"{fmt(r.get('pval_t+1'), '{:>.3f}'):>8}"
          f"{fmt(r.get('corr_oos')):>11}"
          f"{fmt(r.get('pval_oos'), '{:>.3f}'):>8}"
          f"  {r['verdict']}{' ✓BOTH' if r['passes_both_gates'] else ''}")

promoted = [r["factor"] for r in results if r["passes_both_gates"]]
if not promoted:
    print("\nNo factors passed both gates. Stopping before walk-forward.")
    raise SystemExit(0)

# ---------- walk-forward for promoted factors ----------
print(f"\nPromoting {len(promoted)} factor(s) to walk-forward: {promoted}")

S2 = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK",
}.items()}
df = pd.DataFrame(S2).resample("W-FRI").last().ffill().dropna()
f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

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

# Attach promoted factors to f, aligned
for fac_name in promoted:
    f[fac_name] = factors[fac_name].reindex(f.index, method="nearest", tolerance=pd.Timedelta("3D"))

f = f.dropna()
print(f"  walk-forward frame n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}")
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V6 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess"] + active_events

def wf(frame, factors_list):
    y_full = frame["log_TSLA"].to_numpy()
    X_full = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors_list])
    b_in, *_ = np.linalg.lstsq(X_full, y_full, rcond=None)
    yhat_in = X_full @ b_in
    r2_in = 1 - ((y_full - yhat_in) ** 2).sum() / ((y_full - y_full.mean()) ** 2).sum()
    mae_in = float(np.mean(np.abs(np.exp(y_full - yhat_in) - 1)) * 100)
    ho = frame.loc[OOS_START:]
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
    oa = np.array(oa); op = np.array(op)
    mae_oos = float(np.mean(np.abs(op / oa - 1)) * 100)
    ss_tot = ((oa - oa.mean()) ** 2).sum()
    ss_res = ((oa - op) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot
    return r2_in, mae_in, r2_oos, mae_oos

print(f"\n  {'variant':<48}  {'In R2':>7}  {'In MAE':>7}  {'OOS R2':>7}  {'OOS MAE':>8}  {'dR2':>9}")
r2_in, mae_in, r2_oos, mae_oos = wf(f, V6)
print(f"  {'v6 baseline':<48}  {r2_in:>7.4f}  {mae_in:>6.2f}%  {r2_oos:>7.4f}  {mae_oos:>7.2f}%  {'(ref)':>9}")
ref = r2_oos
for fac_name in promoted:
    r2_in, mae_in, r2_oos, mae_oos = wf(f, V6 + [fac_name])
    delta = (r2_oos - ref) * 100
    verdict = "ACCEPT" if delta >= 2.0 else "REJECT"
    print(f"  {('v6 + ' + fac_name):<48}  {r2_in:>7.4f}  {mae_in:>6.2f}%  {r2_oos:>7.4f}  {mae_oos:>7.2f}%  {delta:>+7.2f}pp  {verdict}")
print("\n  Acceptance bar: >= +2.00pp")

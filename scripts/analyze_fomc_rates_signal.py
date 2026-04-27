"""
FOMC / rates-level factor probe (v6.2 baseline).

We've already covered curve SHAPE (CURVE_z in v6.2). This script probes the
LEVEL/policy-path dimension that the curve doesn't see:
  - TNX_zscore_52w   : 10Y yield level -- discount-rate channel
  - SHY_zscore_52w   : 1-3Y treasury ETF, inverse-2Y proxy (Fed-sensitive belly)
  - TIP_zscore_52w   : TIPS ETF level -- real yield/breakeven proxy
  - IEF_zscore_52w   : 7-10Y treasury ETF level -- duration leg of curve
  - TNX_minus_TIP_z  : (10Y nominal - TIPS proxy) ~ breakeven inflation z-score

Standard candidate gates against v6.2 residuals; force-promote OOS-dominant.
Walk-forward against v6.2 baseline (>=+2pp OOS R^2 lift).

NOTE: any factor with high correlation to curve_IEF_SHY_zscore_52w should be
flagged manually -- candidate must be ORTHOGONAL to existing factors, not just
correlated with residuals.
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

def safe(sym, fallback=None):
    s = wk(sym)
    if s.empty and fallback:
        print(f"  {sym} empty; falling back to {fallback}")
        s = wk(fallback)
    return s

def zscore_52(s):
    return (s - s.rolling(52, min_periods=20).mean()) / s.rolling(52, min_periods=20).std()

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
    if len(s) < 20: return {**out, "status": "insufficient"}
    c0, p0 = stats.pearsonr(s, r); out["corr_t"], out["pval_t"] = float(c0), float(p0)
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
    if p is None or (isinstance(p, float) and np.isnan(p)): return "INCONCLUSIVE"
    if n < 50: return f"INCONCLUSIVE (n={n})"
    if p < 0.05 and abs(c) > 0.15: return f"PROMISING (corr={c:+.3f}, p={p:.3f})"
    if p < 0.10: return f"WEAK (p={p:.3f})"
    return "REJECT"

def passes_oos_gate(row):
    c1 = row.get("corr_t+1"); c_oos = row.get("corr_oos"); p_oos = row.get("pval_oos")
    if any(x is None for x in [c1, c_oos, p_oos]): return False
    if any(isinstance(x, float) and np.isnan(x) for x in [c1, c_oos, p_oos]): return False
    return (np.sign(c1) == np.sign(c_oos)) and (p_oos < 0.10)

# ---------- pull series ----------
print("Fetching weekly closes...")
raw = {
    "TNX":  safe("^TNX"),         # CBOE 10Y yield (in % * 10)
    "TYX":  safe("^TYX"),         # 30Y yield
    "FVX":  safe("^FVX"),         # 5Y yield
    "IRX":  safe("^IRX"),         # 13W T-bill (Fed-funds-ish)
    "SHY":  safe("SHY"),          # 1-3Y ETF
    "IEF":  safe("IEF"),          # 7-10Y ETF
    "TIP":  safe("TIP"),          # TIPS ETF
    "TLT":  safe("TLT"),
}
for k, s in raw.items(): print(f"  {k:<5} {len(s)} rows")

W = pd.DataFrame({k: v for k, v in raw.items() if not v.empty}).resample("W-FRI").last().ffill()

factors = {}

# Yield-based (raw % * 10 from yahoo; scale doesn't matter for z-score)
if "TNX" in W: factors["TNX_zscore_52w"] = zscore_52(W["TNX"])
if "FVX" in W: factors["FVX_zscore_52w"] = zscore_52(W["FVX"])
if "IRX" in W: factors["IRX_zscore_52w"] = zscore_52(W["IRX"])  # short rates / Fed proxy
if "TYX" in W: factors["TYX_zscore_52w"] = zscore_52(W["TYX"])

# ETF-based (price; bonds rally when rates fall, so SIGN flips vs yields)
if "SHY" in W: factors["log_SHY_z52w"] = zscore_52(np.log(W["SHY"]))
if "IEF" in W: factors["log_IEF_z52w"] = zscore_52(np.log(W["IEF"]))
if "TIP" in W: factors["log_TIP_z52w"] = zscore_52(np.log(W["TIP"]))

# Breakeven inflation proxy: TIP outperformance vs IEF (real yield - nominal yield is negative breakeven)
# When inflation expectations rise: TIP outperforms IEF -> log(TIP/IEF) rises
if "TIP" in W and "IEF" in W:
    breakeven = np.log(W["TIP"]) - np.log(W["IEF"])
    factors["breakeven_TIP_IEF_z52w"] = zscore_52(breakeven)

# Cross-orthogonalized: TNX residualized vs CURVE so what's left is rate LEVEL
# orthogonal to curve shape (= the "policy stance" component, not "shape")
if "TNX" in W and "IEF" in W and "SHY" in W:
    log_curve = np.log(W["IEF"]) - np.log(W["SHY"])
    aligned = pd.concat([W["TNX"], log_curve], axis=1, keys=["tnx", "curve"]).dropna()
    if len(aligned) >= 30:
        tnx_orth_curve = residualize(aligned["tnx"], aligned["curve"])
        factors["TNX_orth_CURVE_z52w"] = zscore_52(tnx_orth_curve)

# ---------- candidate gate ----------
resid = load_residuals()
print(f"\nv6.2 residual series: n={len(resid)}  std={resid.std():.4f}  "
      f"({resid.index[0].date()} -> {resid.index[-1].date()})")
print(f"\n{'factor':<28}{'n':>5}{'corr_t':>10}{'p':>8}{'corr_t+1':>11}{'p':>8}{'corr_oos':>11}{'p':>8}  verdict")
print("-" * 124)

def fmt(x, fmt="{:>+.3f}"):
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return fmt.format(x)
    return "  n/a"

results = []
for name, s in factors.items():
    r = report(name, s, resid)
    r["verdict"] = verdict_str(r)
    r["passes_both_gates"] = ("PROMISING" in r["verdict"]) and passes_oos_gate(r)
    results.append(r)
    print(f"{name:<28}"
          f"{r.get('n',0):>5}"
          f"{fmt(r.get('corr_t')):>10}"
          f"{fmt(r.get('pval_t'), '{:>.3f}'):>8}"
          f"{fmt(r.get('corr_t+1')):>11}"
          f"{fmt(r.get('pval_t+1'), '{:>.3f}'):>8}"
          f"{fmt(r.get('corr_oos')):>11}"
          f"{fmt(r.get('pval_oos'), '{:>.3f}'):>8}"
          f"  {r['verdict']}{' BOTH' if r['passes_both_gates'] else ''}")

promoted = [r["factor"] for r in results if r["passes_both_gates"]]

# Force-promote OOS-dominant
for r in results:
    if r["factor"] in promoted: continue
    c1 = r.get("corr_t+1"); p1 = r.get("pval_t+1")
    co = r.get("corr_oos"); po = r.get("pval_oos")
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in [c1, p1, co, po]): continue
    if (p1 < 0.10 and po < 0.01 and abs(co) >= 0.20 and np.sign(c1) == np.sign(co)):
        promoted.append(r["factor"])
        print(f"  Force-promoting OOS-dominant: {r['factor']} (lagged p={p1:.3f}, OOS corr={co:+.3f} p={po:.3f})")

# Manual override: rates-level is the user's specifically-requested hypothesis;
# force-test the strongest OOS-only candidates against v6.2 even though strict
# gates would block them. If walk-forward fails this is a clean reject.
MANUAL = ["TNX_zscore_52w", "FVX_zscore_52w", "log_TIP_z52w",
          "log_IEF_z52w", "TNX_orth_CURVE_z52w"]
for fac in MANUAL:
    if fac in factors and fac not in promoted:
        promoted.append(fac)
        print(f"  Manual override (OOS-only): {fac}")

if not promoted:
    print("\nNo factors passed gates. Stopping before walk-forward.")
    raise SystemExit(0)

# ---------- walk-forward against v6.2 ----------
print(f"\nPromoting {len(promoted)} factor(s) to walk-forward (v6.2 baseline): {promoted}")

S2 = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY",
}.items()}
df = pd.DataFrame(S2).resample("W-FRI").last().ffill().dropna()
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

for fac_name in promoted:
    f[fac_name] = factors[fac_name].reindex(f.index, method="nearest", tolerance=pd.Timedelta("3D"))

f = f.dropna()
print(f"  walk-forward frame n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}")
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V62 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
       "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_events

# Quick collinearity check against existing v6.2 factors
print("\n  Collinearity vs v6.2 factors (full-sample corr):")
key_v62 = ["log_QQQ", "log_DXY", "log_VIX", "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]
print(f"    {'candidate':<28}" + "".join(f"{k:>14}" for k in key_v62))
for fac_name in promoted:
    s = f[fac_name]
    corrs = [f"{f[k].corr(s):>+14.3f}" for k in key_v62]
    print(f"    {fac_name:<28}" + "".join(corrs))

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
    oa, op = np.array(oa), np.array(op)
    mae = float(np.mean(np.abs(op / oa - 1)) * 100)
    r2_oos = 1 - ((oa - op) ** 2).sum() / ((oa - oa.mean()) ** 2).sum()
    return r2_in, mae_in, r2_oos, mae, len(oa)

print(f"\n  {'variant':<52}{'In R2':>9}{'In MAE':>9}{'OOS R2':>9}{'OOS MAE':>10}{'dR2':>11}")
r2_ref_in, mae_ref_in, r2_ref, mae_ref, n_oos = wf(f, V62)
print(f"  {'v6.2 baseline':<52}{r2_ref_in:>9.4f}{mae_ref_in:>9.2f}%{r2_ref:>9.4f}{mae_ref:>10.2f}%{'(ref)':>11}")
for fac_name in promoted:
    r2_in, mae_in, r2_oos, mae, _ = wf(f, V62 + [fac_name])
    delta = (r2_oos - r2_ref) * 100
    verdict = "ACCEPT" if delta >= 2.0 else "REJECT"
    print(f"  {'v6.2 + ' + fac_name:<52}{r2_in:>9.4f}{mae_in:>9.2f}%{r2_oos:>9.4f}{mae:>10.2f}%{delta:>+9.2f}pp  {verdict}")

print(f"\n  Acceptance bar: >= +2.00pp")

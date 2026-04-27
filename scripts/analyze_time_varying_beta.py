"""
Time-varying TSLA-QQQ beta probe (v6.2 baseline).

Hypothesis: the assumption of a constant beta on log(QQQ) leaves regime-shift
variance in v6.2 residuals. Test whether external regime indicators (VIX,
ARKK_excess, CURVE, MOVE, time buckets) can drive an interaction term
log(QQQ) * regime that lifts OOS R^2.

Approach:
  1. Diagnostic: compute rolling 26w beta(TSLA, QQQ) and report time variation.
  2. Construct interaction factors: log_QQQ_centered * regime_z
     (centering log_QQQ avoids absorbing the main effect into the interaction)
  3. Apply standard candidate gate (lagged + OOS) against v6.2 residuals.
  4. Walk-forward against v6.2 baseline (+2pp OOS R^2 bar).

Note: a *naive* "rolling beta as factor" approach is rejected -- it would feed
TSLA's own past returns back into the predictor, repeating the v5 persistence
backdoor. Interaction terms with EXTERNAL regime signals are clean.
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

# ---------- pull base series and v6.2 components ----------
print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY", "MOVE": "^MOVE",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()
print(f"  n={len(df)}  {df.index[0].date()} -> {df.index[-1].date()}")

# Log returns for rolling-beta diagnostic
ret_tsla = np.log(df["TSLA"]).diff()
ret_qqq  = np.log(df["QQQ"]).diff()

# ---------- DIAGNOSTIC: how much does beta(TSLA, QQQ) vary? ----------
print("\n=== Diagnostic: rolling 26w beta(TSLA, QQQ) ===")
WIN = 26
rolling_cov = ret_tsla.rolling(WIN).cov(ret_qqq)
rolling_var = ret_qqq.rolling(WIN).var()
rolling_beta = (rolling_cov / rolling_var).dropna()

print(f"  rolling beta (n={len(rolling_beta)})")
print(f"    mean   {rolling_beta.mean():+.3f}")
print(f"    std    {rolling_beta.std():+.3f}")
print(f"    min    {rolling_beta.min():+.3f}  ({rolling_beta.idxmin().date()})")
print(f"    25%    {rolling_beta.quantile(0.25):+.3f}")
print(f"    50%    {rolling_beta.median():+.3f}")
print(f"    75%    {rolling_beta.quantile(0.75):+.3f}")
print(f"    max    {rolling_beta.max():+.3f}  ({rolling_beta.idxmax().date()})")
# Quick AR(1) on beta -- is it slow-moving (regime-like) or noise?
b_lag = rolling_beta.shift(1).dropna()
b_now = rolling_beta.loc[b_lag.index]
ar1 = float(np.corrcoef(b_lag, b_now)[0, 1])
print(f"    AR(1) coef: {ar1:+.3f}  (close to 1 = persistent regime; close to 0 = noise)")

# ---------- regime indicators (EXTERNAL to TSLA) ----------
log_qqq = np.log(df["QQQ"])
log_qqq_centered = log_qqq - log_qqq.mean()  # avoid co-absorbing main effect

log_vix = np.log(df["VIX"])
vix_z = (log_vix - log_vix.rolling(52, min_periods=20).mean()) / log_vix.rolling(52, min_periods=20).std()

log_rbob = np.log(df["RBOB"])
# residualize ARKK vs QQQ (matches v6.2 construction)
ARKK_excess = residualize(np.log(df["ARKK"]), log_qqq)
NVDA_excess = residualize(np.log(df["NVDA"]), log_qqq)

curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
curve_z = (curve_log - curve_log.rolling(52, min_periods=20).mean()) / curve_log.rolling(52, min_periods=20).std()

log_move = np.log(df["MOVE"])
move_z = (log_move - log_move.rolling(52, min_periods=20).mean()) / log_move.rolling(52, min_periods=20).std()

# Time bucket: "post-Trump" (after 2024-11-06)
trump_era = pd.Series((df.index >= pd.Timestamp("2024-11-06")).astype(float), index=df.index)
# Demean so it doesn't absorb baseline shift
trump_era_dm = trump_era - trump_era.mean()

# ---------- interaction factors ----------
factors = {
    "QQQ_x_VIXz":       log_qqq_centered * vix_z,
    "QQQ_x_ARKKexc":    log_qqq_centered * ARKK_excess,
    "QQQ_x_NVDAexc":    log_qqq_centered * NVDA_excess,
    "QQQ_x_CURVEz":     log_qqq_centered * curve_z,
    "QQQ_x_MOVEz":      log_qqq_centered * move_z,
    "QQQ_x_TrumpEra":   log_qqq_centered * trump_era_dm,
}

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

# Force-promote OOS-dominant factors (lagged WEAK 0.05<=p<0.10, OOS strong)
for r in results:
    if r["factor"] in promoted: continue
    c1 = r.get("corr_t+1"); p1 = r.get("pval_t+1")
    co = r.get("corr_oos"); po = r.get("pval_oos")
    if any(x is None or (isinstance(x, float) and np.isnan(x))
           for x in [c1, p1, co, po]): continue
    if (p1 < 0.10 and po < 0.01 and abs(co) >= 0.20
        and np.sign(c1) == np.sign(co)):
        promoted.append(r["factor"])
        print(f"  Force-promoting OOS-dominant: {r['factor']} "
              f"(lagged p={p1:.3f}, OOS corr={co:+.3f} p={po:.3f})")

# Manual override: walk-forward all interactions regardless. The diagnostic
# question is about regression structure (does an interaction help?), not
# correlation pre-screening. Cheap to test all 6 in walk-forward.
for fac in factors.keys():
    if fac not in promoted:
        promoted.append(fac)
        print(f"  Forcing all interactions to walk-forward: {fac}")

# ---------- walk-forward against v6.2 baseline ----------
print(f"\nPromoting {len(promoted)} factor(s) to walk-forward (v6.2 baseline)")

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = log_qqq
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = log_vix
f["NVDA_excess"] = NVDA_excess
f["ARKK_excess"] = ARKK_excess
f["RBOB_zscore_52w"] = (
    (log_rbob - log_rbob.rolling(52, min_periods=20).mean())
    / log_rbob.rolling(52, min_periods=20).std()
)
f["curve_IEF_SHY_zscore_52w"] = curve_z

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
    f[fac_name] = factors[fac_name].reindex(f.index)

f = f.dropna()
print(f"  walk-forward frame n={len(f)} weeks  {f.index[0].date()} -> {f.index[-1].date()}")
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V62 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
       "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_events

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

print(f"\n  {'variant':<46}{'In R2':>9}{'In MAE':>9}{'OOS R2':>9}{'OOS MAE':>10}{'dR2':>11}")
r2_ref_in, mae_ref_in, r2_ref, mae_ref, n_oos = wf(f, V62)
print(f"  {'v6.2 baseline':<46}{r2_ref_in:>9.4f}{mae_ref_in:>9.2f}%{r2_ref:>9.4f}{mae_ref:>10.2f}%{'(ref)':>11}")
for fac_name in promoted:
    r2_in, mae_in, r2_oos, mae, _ = wf(f, V62 + [fac_name])
    delta = (r2_oos - r2_ref) * 100
    verdict = "ACCEPT" if delta >= 2.0 else "REJECT"
    print(f"  {'v6.2 + ' + fac_name:<46}{r2_in:>9.4f}{mae_in:>9.2f}%{r2_oos:>9.4f}{mae:>10.2f}%{delta:>+9.2f}pp  {verdict}")

# Also test pairs of interactions (best 2 by lift) to see if effect is additive
print(f"\n  Acceptance bar: >= +2.00pp")

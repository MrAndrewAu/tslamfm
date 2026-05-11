"""
Structural swap test: XLK (Technology Select Sector SPDR) vs QQQ as the
tech-market beta anchor in the v6.4 factor set.

QQQ = Nasdaq-100 (100 largest non-financial Nasdaq stocks, ~60% tech).
XLK = S&P 500 Technology sector (~70 pure-tech stocks, AAPL+MSFT ≈ 45%).

Hypotheses:
  H1 — XLK's tighter tech focus gives a cleaner beta signal for TSLA's
       AI / growth multiple, reducing residual noise.
  H2 — NVDA and ARKK, residualized on XLK instead of QQQ, may carry more
       orthogonal information (less of their raw move is already absorbed
       by the base factor).

Three variants tested:
  A. Drop-in swap: log_XLK replaces log_QQQ; NVDA_excess and ARKK_excess
     still residualized on log_QQQ (measures pure substitution).
  B. Full swap: log_XLK replaces log_QQQ AND NVDA/ARKK residualized on
     log_XLK (measures the full re-residualization effect).
  C. Both together: log_QQQ + log_XLK retained side-by-side (checks
     whether XLK adds orthogonal info on top of QQQ).

Acceptance bar: +2pp walk-forward OOS R² lift vs v6.4 baseline.
Same methodology as every prior candidate test.

Note: TSLA is a Nasdaq-100 component (QQQ) but NOT in the Technology
Select Sector (XLK is classified under the S&P GICS tech sector; TSLA is
Consumer Discretionary). This means XLK has no self-inclusion issue,
while QQQ has a very small one (~2% weight at peak). Neither is material.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
MIN_TRAIN   = 100
EVENT_P_THR = 0.10
LAMBDA      = 0.94
BARRIER_PP  = 2.0

# ── data ─────────────────────────────────────────────────────────────────────

def wk(sym):
    candidates = []
    try:
        s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
        candidates.append(s)
    except Exception:
        pass
    try:
        dl = yf.download(sym, start=START, end=END, interval="1wk",
                         progress=False, auto_adjust=False)
        if not dl.empty and "Close" in dl.columns:
            candidates.append(dl["Close"])
    except Exception:
        pass
    for s in candidates:
        s = s.dropna()
        if s.empty:
            continue
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx
        return s.resample("W-FRI").last()
    raise RuntimeError(f"Failed to fetch weekly close for {sym}")


def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


# ── OLS / walk-forward helpers ────────────────────────────────────────────────

def fit(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    r = y - yhat
    n, k = X.shape
    s2 = (r @ r) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss_tot
    return dict(beta=b, p=p, r2=float(r2),
                fitted=pd.Series(yhat, index=frame.index),
                resid=pd.Series(r, index=frame.index),
                factors=list(factors))


def select_events(frame, forced, candidate_events, p_thr=EVENT_P_THR):
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    m = fit(frame, factors)
    while events:
        event_ps = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst, worst_p = max(event_ps, key=lambda x: x[1])
        if worst_p <= p_thr:
            break
        events.remove(worst)
        factors = forced + events
        m = fit(frame, factors)
    return factors, events, m


def walk_forward(frame, forced, candidate_events, min_train=MIN_TRAIN):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, _, m_tr = select_events(train, forced, candidate_events)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred_log.loc[date] = float(xv @ m_tr["beta"])
    return pred_log


def oos_metrics(frame, pred_log):
    idx = pred_log.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    pred   = np.exp(pred_log.loc[idx_oos].to_numpy())
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    ss_res = ((actual - pred) ** 2).sum()
    r2  = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(actual - pred)) / np.mean(actual) * 100
    return float(r2), float(mae), int(len(actual))


# ── fetch data ────────────────────────────────────────────────────────────────

print("=" * 65)
print("XLK vs QQQ as tech-beta anchor — structural swap test")
print("=" * 65)
print("\nFetching weekly closes (QQQ, XLK, and all v6.4 instruments)...")

S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "XLK": "XLK",
    "DXY":  "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK",
    "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()
print(f"  {len(df)} weeks  {df.index[0].date()} → {df.index[-1].date()}")

# ── correlation snapshot: XLK vs QQQ ─────────────────────────────────────────
log_qqq = np.log(df["QQQ"])
log_xlk = np.log(df["XLK"])
log_tsla = np.log(df["TSLA"])
corr_qqq_xlk = float(np.corrcoef(log_qqq, log_xlk)[0, 1])
corr_tsla_qqq = float(np.corrcoef(log_tsla, log_qqq)[0, 1])
corr_tsla_xlk = float(np.corrcoef(log_tsla, log_xlk)[0, 1])
print(f"\nCorrelation snapshot (log levels, n={len(df)}):")
print(f"  corr(log_QQQ, log_XLK)   = {corr_qqq_xlk:.4f}")
print(f"  corr(log_TSLA, log_QQQ)  = {corr_tsla_qqq:.4f}")
print(f"  corr(log_TSLA, log_XLK)  = {corr_tsla_xlk:.4f}")

# Residual of XLK on QQQ: how much XLK adds beyond QQQ
xlk_excess = residualize(log_xlk, log_qqq)
corr_tsla_xlk_excess = float(np.corrcoef(
    log_tsla.reindex(xlk_excess.index).dropna(),
    xlk_excess.reindex(log_tsla.dropna().index).dropna()
)[0, 1])
print(f"  corr(log_TSLA, XLK_excess_vs_QQQ) = {corr_tsla_xlk_excess:.4f}")
print(f"  (XLK_excess captures what XLK adds beyond QQQ)")

# ── build shared feature matrix ───────────────────────────────────────────────

base = pd.DataFrame(index=df.index)
base["log_TSLA"] = np.log(df["TSLA"])
base["log_QQQ"]  = np.log(df["QQQ"])
base["log_XLK"]  = np.log(df["XLK"])
base["log_DXY"]  = np.log(df["DXY"])
base["log_VIX"]  = np.log(df["VIX"])

# NVDA / ARKK residualized on QQQ (v6.4 canonical)
base["NVDA_excess_QQQ"]  = residualize(np.log(df["NVDA"]), base["log_QQQ"])
base["ARKK_excess_QQQ"]  = residualize(np.log(df["ARKK"]), base["log_QQQ"])

# NVDA / ARKK residualized on XLK (for full-swap variant)
base["NVDA_excess_XLK"]  = residualize(np.log(df["NVDA"]), base["log_XLK"])
base["ARKK_excess_XLK"]  = residualize(np.log(df["ARKK"]), base["log_XLK"])

# rolling factors
log_rbob = np.log(df["RBOB"])
base["RBOB_zscore_52w"]  = (
    (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) /
    log_rbob.rolling(52, min_periods=20).std()
)
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
base["curve_IEF_SHY_zscore_52w"] = (
    (curve_log - curve_log.rolling(52, min_periods=20).mean()) /
    curve_log.rolling(52, min_periods=20).std()
)

# event dummies (same list as v6.4 builder)
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
    base[f"E_{name}"] = ((base.index >= d0) &
                          (base.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = base.dropna(subset=[
    "log_TSLA", "log_QQQ", "log_XLK", "log_DXY", "log_VIX",
    "NVDA_excess_QQQ", "ARKK_excess_QQQ",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w",
])
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
n_all = len(f)
n_is  = (f.index < pd.Timestamp(OOS_START)).sum()
n_oos = (f.index >= pd.Timestamp(OOS_START)).sum()
print(f"\nWorking frame: {n_all} weeks  IS={n_is}  OOS={n_oos}")

# ── define the three variant factor sets ─────────────────────────────────────

CONTINUOUS_BASE = [
    "log_DXY", "log_VIX",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w",
]

# v6.4 baseline: QQQ as anchor, NVDA/ARKK residualized on QQQ
FORCED_V64 = ["log_QQQ", "log_DXY", "log_VIX",
              "NVDA_excess_QQQ", "ARKK_excess_QQQ",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

# Variant A: drop-in swap (XLK replaces QQQ, NVDA/ARKK still on QQQ)
FORCED_A = ["log_XLK", "log_DXY", "log_VIX",
            "NVDA_excess_QQQ", "ARKK_excess_QQQ",
            "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

# Variant B: full swap (XLK replaces QQQ, NVDA/ARKK residualized on XLK)
FORCED_B = ["log_XLK", "log_DXY", "log_VIX",
            "NVDA_excess_XLK", "ARKK_excess_XLK",
            "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

# Variant C: additive (QQQ + XLK both included, NVDA/ARKK on QQQ)
FORCED_C = ["log_QQQ", "log_XLK", "log_DXY", "log_VIX",
            "NVDA_excess_QQQ", "ARKK_excess_QQQ",
            "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

# ── in-sample diagnostics ─────────────────────────────────────────────────────

print("\n[IS diagnostics] Fitting on full window with backward event selection...")
variants = {
    "v6.4  (QQQ anchor, baseline)": FORCED_V64,
    "Var A (XLK drop-in)":          FORCED_A,
    "Var B (XLK full swap)":         FORCED_B,
    "Var C (QQQ + XLK additive)":   FORCED_C,
}

print(f"\n  {'variant':<38} {'IS R²':>7} {'events kept':>12}  factor p-values")
for label, forced in variants.items():
    facs, events, m = select_events(f, forced, active_events)
    n_ev = len(events)
    # show p-values of the forced continuous factors
    forced_ps = " | ".join(
        f"{c.split('_')[1] if '_' in c else c}:{m['p'][1+facs.index(c)]:.3f}"
        for c in forced
    )
    print(f"  {label:<38} {m['r2']:>7.4f} {n_ev:>12}  {forced_ps}")

# ── walk-forward OOS ──────────────────────────────────────────────────────────

print("\n[Walk-forward OOS]  (this may take ~60 seconds)...")
print(f"  {'variant':<38} {'OOS R²':>7} {'OOS MAE':>9} {'Δ vs v64':>10}  verdict")

results = {}
for label, forced in variants.items():
    pred = walk_forward(f, forced, active_events)
    r2, mae, n_oos_wf = oos_metrics(f, pred)
    results[label] = (r2, mae)
    if "baseline" in label:
        base_r2 = r2

for label, (r2, mae) in results.items():
    delta = (r2 - base_r2) * 100
    if "baseline" in label:
        verdict = "(ref)"
    elif delta >= BARRIER_PP:
        verdict = "ACCEPT"
    else:
        verdict = "REJECT"
    print(f"  {label:<38} {r2:>7.4f} {mae:>8.2f}%  {delta:>+9.2f}pp  {verdict}")

# ── VIF / multicollinearity for Variant C ────────────────────────────────────

print("\n[Multicollinearity: log_XLK vs log_QQQ]")
# Regress log_XLK on log_QQQ to get VIF
y_vif = f["log_XLK"].to_numpy()
X_vif = np.column_stack([np.ones(len(f)), f["log_QQQ"].to_numpy()])
b_vif, *_ = np.linalg.lstsq(X_vif, y_vif, rcond=None)
yhat_vif = X_vif @ b_vif
ss_tot_v = ((y_vif - y_vif.mean()) ** 2).sum()
r2_vif   = 1 - ((y_vif - yhat_vif) ** 2).sum() / ss_tot_v
vif      = 1 / (1 - r2_vif) if r2_vif < 1 else float("inf")
print(f"  R²(log_XLK ~ log_QQQ) = {r2_vif:.4f}   VIF = {vif:.1f}")
print(f"  {'HIGH MULTICOLLINEARITY — Variant C will be unstable' if vif > 5 else 'moderate collinearity'}")

# ── coefficient stability check for Var B ─────────────────────────────────────

print("\n[Coefficient stability: log_XLK beta across rolling windows]")
window_sizes = [80, 100, 120, 150, len(f)]
print(f"  {'window':>8}  {'beta_XLK':>10}  {'p_XLK':>8}  {'beta_QQQ (v64)':>15}  {'p_QQQ':>8}")
for w in window_sizes:
    sub = f.iloc[-w:] if w < len(f) else f
    facs_b, _, m_b = select_events(sub, FORCED_B, active_events)
    facs_v, _, m_v = select_events(sub, FORCED_V64, active_events)
    if "log_XLK" in facs_b:
        idx_xlk = facs_b.index("log_XLK") + 1
        b_xlk = m_b["beta"][idx_xlk]
        p_xlk = m_b["p"][idx_xlk]
    else:
        b_xlk, p_xlk = np.nan, np.nan
    if "log_QQQ" in facs_v:
        idx_qqq = facs_v.index("log_QQQ") + 1
        b_qqq = m_v["beta"][idx_qqq]
        p_qqq = m_v["p"][idx_qqq]
    else:
        b_qqq, p_qqq = np.nan, np.nan
    label = "full" if w == len(f) else f"last {w}w"
    print(f"  {label:>8}  {b_xlk:>10.4f}  {p_xlk:>8.4f}  {b_qqq:>15.4f}  {p_qqq:>8.4f}")

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
base_r2_val, base_mae_val = results["v6.4  (QQQ anchor, baseline)"]
print(f"  v6.4 baseline: OOS R²={base_r2_val:.4f}  MAE={base_mae_val:.2f}%")
print(f"  Acceptance bar: +{BARRIER_PP}pp OOS R² lift")
print()
print("  Structural note:")
print("  QQQ and XLK are highly correlated (≥0.95 on log levels).")
print("  VIF > 10 is expected for Variant C — adding both is likely unstable.")
print("  Variants A & B test whether XLK is a *better anchor* than QQQ;")
print("  they do NOT add a net new factor, they swap the backbone.")
print()
print("  Economic interpretation if Variant B accepted:")
print("  XLK's S&P 500 tech classification is tighter than QQQ's Nasdaq-100,")
print("  and TSLA is NOT in XLK (Consumer Discretionary sector), eliminating")
print("  any minor self-inclusion feedback. A positive result would suggest")
print("  TSLA's AI premium is better explained by pure S&P tech than by the")
print("  broader Nasdaq growth basket.")

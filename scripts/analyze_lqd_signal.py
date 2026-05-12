"""
Probe: LQD (iShares IG Corporate Bond ETF) as additive v6.4 factor.

Economic story: Investment-grade credit spreads (LQD excess vs IEF) measure
financial-conditions tightening beyond what the yield-curve slope captures.
When IG spreads widen, the cost of capital rises for growth companies like TSLA
even if the Treasury curve is unchanged. HYG (junk) was catastrophically
destructive (§10.18); IG credit is a distinct, more regime-stable signal
that doesn't carry the same default-risk noise.

Transforms tested:
  lqd_excess_vs_ief   — LQD residualized on IEF (isolates credit premium
                         from rate-level moves; primary)
  lqd_zscore_52w      — raw 52-week log z-score of LQD
  lqd_spread_z        — (log_LQD − log_IEF) z-score (simplified spread proxy)
  lqd_mom4_z          — 4-week log-return z-score

Acceptance gate: +1pp walk-forward OOS R² lift (May 2026 threshold).
Robustness: tenant-swap (drop curve) + OOS sub-period corr.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END   = "2022-04-26", "2026-04-26"
OOS_START    = "2025-01-03"
MIN_TRAIN    = 100
EVENT_P_THR  = 0.10
BARRIER_PP   = 1.0
VIF_WARN     = 5.0

# ── Data fetch ────────────────────────────────────────────────────────────────

def wk(sym):
    for kwargs in [
        dict(start=START, end=END, interval="1wk"),
        dict(start=START, end=END, interval="1wk", auto_adjust=False),
    ]:
        try:
            dl = yf.download(sym, progress=False, **kwargs)
            if dl.empty:
                continue
            if isinstance(dl.columns, pd.MultiIndex):
                dl.columns = dl.columns.get_level_values(0)
            s = dl["Close"] if "Close" in dl.columns else dl.iloc[:, 0]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            idx = pd.to_datetime(s.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            s.index = idx
            return s.resample("W-FRI").last()
        except Exception:
            continue
    raise RuntimeError(f"Failed to fetch weekly close for {sym}")

# ── OLS helpers ───────────────────────────────────────────────────────────────

def residualize(target, base_s):
    X = np.column_stack([np.ones(len(base_s)), base_s.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


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
                resid=pd.Series(r, index=frame.index))


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
    idx     = pred_log.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual  = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    pred    = np.exp(pred_log.loc[idx_oos].to_numpy())
    ss_tot  = ((actual - actual.mean()) ** 2).sum()
    r2      = 1 - ((actual - pred) ** 2).sum() / ss_tot
    mae     = np.mean(np.abs(actual - pred)) / np.mean(actual) * 100
    return float(r2), float(mae), int(len(actual))


def compute_vif(frame, new_factor, forced):
    cols = forced + [new_factor]
    sub = frame[cols].dropna()
    X = sub.to_numpy().astype(float)
    j = len(forced)
    y = X[:, j]
    others = X[:, :j]
    Xo = np.column_stack([np.ones(len(others)), others])
    b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    yhat = Xo @ b
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return 1.0 / max(1.0 - r2, 1e-10)


def lagged_corr_vs_resid(col, frame, resid):
    c_lag = frame[col].shift(1)
    common = resid.index.intersection(c_lag.dropna().index)
    if len(common) < 20:
        return np.nan, np.nan
    return stats.pearsonr(c_lag.loc[common], resid.loc[common])


def oos_subcorr(col, frame, resid):
    oos_idx = frame.index[frame.index >= pd.Timestamp(OOS_START)]
    cand    = frame.loc[oos_idx, col].dropna()
    res     = resid.reindex(cand.index).dropna()
    common  = cand.index.intersection(res.index)
    if len(common) < 10:
        return np.nan, np.nan, 0
    c, p = stats.pearsonr(cand.loc[common], res.loc[common])
    return c, p, len(common)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch data
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("LQD (IG credit) probe vs v6.4")
print("=" * 68)
print("\n[1] Fetching weekly closes...")

TICKERS = {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY",
    "LQD": "LQD",
}
raw = {}
for k, v in TICKERS.items():
    try:
        raw[k] = wk(v)
        print(f"  {k:<8} ({v:<12})  {len(raw[k])} rows")
    except Exception as e:
        print(f"  {k:<8} FAILED: {e}")

core_keys = [k for k in TICKERS if k != "LQD"]
core_df = (pd.DataFrame({k: raw[k] for k in core_keys if k in raw})
           .resample("W-FRI").last().ffill().dropna())
print(f"\n  Core: {len(core_df)} weeks  "
      f"{core_df.index[0].date()} -> {core_df.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build v6.4 factor frame
# ═══════════════════════════════════════════════════════════════════════════════

base = pd.DataFrame(index=core_df.index)
base["log_TSLA"] = np.log(core_df["TSLA"])
base["log_QQQ"]  = np.log(core_df["QQQ"])
base["log_DXY"]  = np.log(core_df["DXY"])
base["log_VIX"]  = np.log(core_df["VIX"])
base["NVDA_excess"] = residualize(np.log(core_df["NVDA"]), base["log_QQQ"])
base["ARKK_excess"] = residualize(np.log(core_df["ARKK"]), base["log_QQQ"])
log_rbob = np.log(core_df["RBOB"])
base["RBOB_zscore_52w"] = (
    (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) /
     log_rbob.rolling(52, min_periods=20).std()
)
curve_log = np.log(core_df["IEF"]) - np.log(core_df["SHY"])
base["curve_IEF_SHY_zscore_52w"] = (
    (curve_log - curve_log.rolling(52, min_periods=20).mean()) /
     curve_log.rolling(52, min_periods=20).std()
)

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
    base[f"E_{name}"] = (
        (base.index >= d0) & (base.index < d0 + pd.Timedelta(weeks=8))
    ).astype(int)

FORCED_V64 = [
    "log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w",
]
f_eq = base.dropna(subset=FORCED_V64)
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f_eq[f"E_{n}"].nunique() > 1]
print(f"  Factor frame: {len(f_eq)} weeks  "
      f"IS={(f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build LQD transforms
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building LQD transforms...")

if "LQD" in raw:
    lqd = raw["LQD"].reindex(f_eq.index, method="ffill")
    ief = core_df["IEF"].reindex(f_eq.index, method="ffill")
    log_lqd = np.log(lqd)
    log_ief  = np.log(ief)

    # Primary: excess of LQD over IEF (credit premium residualized on duration)
    lqd_aligned = log_lqd.reindex(f_eq.index).dropna()
    ief_aligned  = log_ief.reindex(lqd_aligned.index)
    f_eq["lqd_excess_vs_ief"] = residualize(lqd_aligned, ief_aligned).reindex(f_eq.index)

    # Z-score of log(LQD/IEF) spread
    spread = log_lqd - log_ief
    f_eq["lqd_spread_z"] = (
        (spread - spread.rolling(52, min_periods=20).mean()) /
         spread.rolling(52, min_periods=20).std()
    )

    # Raw LQD z-score
    f_eq["lqd_zscore_52w"] = (
        (log_lqd - log_lqd.rolling(52, min_periods=20).mean()) /
         log_lqd.rolling(52, min_periods=20).std()
    )

    # Fast 4-week momentum
    raw_mom4 = log_lqd.diff(4) * 100
    f_eq["lqd_mom4_z"] = (
        (raw_mom4 - raw_mom4.rolling(52, min_periods=20).mean()) /
         raw_mom4.rolling(52, min_periods=20).std()
    )

CANDIDATES = [
    "lqd_excess_vs_ief",
    "lqd_spread_z",
    "lqd_zscore_52w",
    "lqd_mom4_z",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. v6.4 baseline walk-forward
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Running v6.4 baseline walk-forward...")
base_pred = walk_forward(f_eq, FORCED_V64, active_events)
base_r2, base_mae, base_n = oos_metrics(f_eq, base_pred)
print(f"  v6.4 baseline  OOS R2={base_r2:.4f}  MAE={base_mae:.2f}%  n={base_n}")

_, _, m_full = select_events(f_eq, FORCED_V64, active_events)
resid_v64 = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Candidate screen + walk-forward
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Candidate screen (lagged-corr vs v6.4 residual + VIF):")
print(f"\n  {'factor':<28}  {'corr_lag1':>9}  {'p':>6}  {'VIF':>6}  screen")
print("  " + "-" * 64)

screen_results = {}
for col in CANDIDATES:
    sub = f_eq[col].dropna()
    if len(sub) < 30:
        print(f"  {col:<28}  insufficient data")
        continue
    corr, p = lagged_corr_vs_resid(col, f_eq, resid_v64)
    vif = compute_vif(f_eq.dropna(subset=[col]), col, FORCED_V64)
    corr_pass = (not np.isnan(p)) and (p < 0.10)
    vif_pass  = vif < VIF_WARN
    flag = ("corr-pass" if corr_pass else "corr-fail") + \
           " / " + ("vif-ok" if vif_pass else "vif-HIGH")
    print(f"  {col:<28}  {corr:>+9.4f}  {p:>6.4f}  {vif:>6.2f}  {flag}")
    screen_results[col] = dict(corr=corr, p=p, vif=vif,
                                corr_pass=corr_pass, vif_pass=vif_pass)

print("\n[5] Walk-forward for all transforms:")
print(f"\n  {'factor':<28}  {'OOS R2':>7}  {'MAE':>7}  {'delta':>8}  verdict")
print("  " + "-" * 68)

wf_results = {}
for col in CANDIDATES:
    sub = f_eq.dropna(subset=[col])
    if len(sub) < MIN_TRAIN + 20:
        print(f"  {col:<28}  insufficient data")
        continue
    pred = walk_forward(sub, FORCED_V64 + [col], active_events)
    r2, mae, n = oos_metrics(sub, pred)
    delta = (r2 - base_r2) * 100
    verdict = ("ACCEPT" if delta >= BARRIER_PP else "REJECT")
    sr = screen_results.get(col, {})
    vif_str = f"VIF={sr.get('vif', float('nan')):.2f}"
    print(f"  {col:<28}  {r2:>7.4f}  {mae:>6.2f}%  {delta:>+8.2f}pp  "
          f"{verdict}  ({vif_str})")
    wf_results[col] = dict(r2=r2, mae=mae, delta=delta, verdict=verdict)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. OOS sub-period correlation
# ═══════════════════════════════════════════════════════════════════════════════

primary = "lqd_excess_vs_ief"
print(f"\n[6] OOS sub-period correlation ({primary} vs v6.4 residual):")
c_full, p_full = lagged_corr_vs_resid(primary, f_eq, resid_v64)
print(f"  Full IS  lagged corr: {c_full:+.4f}  p={p_full:.4f}")
c_oos, p_oos, n_oos = oos_subcorr(primary, f_eq, resid_v64)
print(f"  OOS only contemp corr: {c_oos:+.4f}  p={p_oos:.4f}  n={n_oos}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Robustness tenant-swaps (only if primary passes gate)
# ═══════════════════════════════════════════════════════════════════════════════

primary_res = wf_results.get(primary, {})
if primary_res.get("delta", -999) >= BARRIER_PP:
    print(f"\n[7] Robustness tenant-swap tests for {primary}:")
    print(f"\n  {'test':<48}  {'OOS R2':>7}  {'delta':>8}")
    print("  " + "-" * 66)
    swap_tests = [
        ("v6.4 + lqd_excess_vs_ief (primary)",
         FORCED_V64 + [primary]),
        ("v6.4 - curve + lqd_excess_vs_ief",
         [f for f in FORCED_V64 if f != "curve_IEF_SHY_zscore_52w"] + [primary]),
        ("v6.4 - VIX + lqd_excess_vs_ief",
         [f for f in FORCED_V64 if f != "log_VIX"] + [primary]),
        ("v6.4 - DXY + lqd_excess_vs_ief",
         [f for f in FORCED_V64 if f != "log_DXY"] + [primary]),
    ]
    for label, forced in swap_tests:
        sub = f_eq.dropna(subset=[c for c in forced if c in f_eq.columns])
        pred = walk_forward(sub, [c for c in forced if c in sub.columns], active_events)
        r2, mae, _ = oos_metrics(sub, pred)
        delta = (r2 - base_r2) * 100
        print(f"  {label:<48}  {r2:>7.4f}  {delta:>+8.2f}pp")
else:
    print(f"\n[7] Robustness skipped -- {primary} did not clear the +{BARRIER_PP}pp gate.")

print("\n" + "=" * 68)
print("Done.")

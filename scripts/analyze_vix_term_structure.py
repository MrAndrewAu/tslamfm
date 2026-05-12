"""
Probe: VIX term structure (VIX3M/VIX ratio) as additive v6.4 factor.

Economic story: log_VIX level is already in the model (fear level).
The VIX3M/VIX RATIO captures the SHAPE of the vol term structure:
  - ratio > 1 (contango): near-term calm, long-term uncertainty → sustained dread
  - ratio < 1 (backwardation): acute panic (VIX spike), short-lived fear
When the term structure inverts (VIX > VIX3M), it signals short-term panic
that typically resolves quickly — TSLA tends to recover faster than the
contango-dread regime. Orthogonal to the VIX level already in the model.

^VIX3M (CBOE 3-Month Volatility Index) is available free on yfinance.

Transforms tested:
  vix_ts_ratio        — log(VIX3M/VIX), raw ratio (primary)
  vix_ts_zscore_52w   — 52-week z-score of log(VIX3M/VIX)
  vix_ts_mom4_z       — 4-week momentum z-score of ratio
  vix_backwardation   — binary: 1 when VIX > VIX3M (acute panic regime)

Acceptance gate: +1pp walk-forward OOS R² lift.
Robustness: tenant-swap (drop log_VIX) + OOS sub-period corr.
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

# ── helpers ───────────────────────────────────────────────────────────────────

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
    raise RuntimeError(f"Failed to fetch {sym}")


def residualize(target, base_s):
    X = np.column_stack([np.ones(len(base_s)), base_s.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


def fit(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b; r = y - yhat
    n, k = X.shape; s2 = (r @ r) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov)); t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss_tot
    return dict(beta=b, p=p, r2=float(r2),
                fitted=pd.Series(yhat, index=frame.index),
                resid=pd.Series(r, index=frame.index))


def select_events(frame, forced, evs, p_thr=EVENT_P_THR):
    events = [e for e in evs if frame[e].nunique() > 1]
    factors = forced + events; m = fit(frame, factors)
    while events:
        ep = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst, wp = max(ep, key=lambda x: x[1])
        if wp <= p_thr: break
        events.remove(worst); factors = forced + events; m = fit(frame, factors)
    return factors, events, m


def walk_forward(frame, forced, evs, min_train=MIN_TRAIN):
    pred = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train: continue
        facs, _, m_tr = select_events(train, forced, evs)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred.loc[date] = float(xv @ m_tr["beta"])
    return pred


def oos_metrics(frame, pred):
    idx = pred.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    p = np.exp(pred.loc[idx_oos].to_numpy())
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - ((actual - p) ** 2).sum() / ss_tot
    mae = np.mean(np.abs(actual - p)) / np.mean(actual) * 100
    return float(r2), float(mae), int(len(actual))


def compute_vif(frame, col, forced):
    cols = forced + [col]
    sub = frame[cols].dropna().to_numpy().astype(float)
    j = len(forced); y = sub[:, j]; others = sub[:, :j]
    Xo = np.column_stack([np.ones(len(others)), others])
    b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    ss_res = ((y - Xo @ b) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return 1.0 / max(1.0 - r2, 1e-10)


def lagged_corr(col, frame, resid):
    c_lag = frame[col].shift(1)
    common = resid.index.intersection(c_lag.dropna().index)
    if len(common) < 20: return np.nan, np.nan
    return stats.pearsonr(c_lag.loc[common], resid.loc[common])


def oos_subcorr(col, frame, resid):
    idx = frame.index[frame.index >= pd.Timestamp(OOS_START)]
    cand = frame.loc[idx, col].dropna()
    res = resid.reindex(cand.index).dropna()
    common = cand.index.intersection(res.index)
    if len(common) < 10: return np.nan, np.nan, 0
    c, p = stats.pearsonr(cand.loc[common], res.loc[common])
    return c, p, len(common)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("VIX term structure (VIX3M/VIX) probe vs v6.4")
print("=" * 68)
print("\n[1] Fetching weekly closes...")

TICKERS = {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY",
    "VIX3M": "^VIX3M",
}
raw = {}
for k, v in TICKERS.items():
    try:
        raw[k] = wk(v)
        print(f"  {k:<8} ({v:<12})  {len(raw[k])} rows")
    except Exception as e:
        print(f"  {k:<8} FAILED: {e}")

if "VIX3M" not in raw or raw["VIX3M"].empty:
    print("\n  ERROR: VIX3M not available on yfinance. Cannot proceed.")
    raise SystemExit(1)

core_keys = [k for k in TICKERS if k != "VIX3M"]
core_df = (pd.DataFrame({k: raw[k] for k in core_keys if k in raw})
           .resample("W-FRI").last().ffill().dropna())
print(f"\n  Core: {len(core_df)} weeks  "
      f"{core_df.index[0].date()} -> {core_df.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. v6.4 factor frame
# ═══════════════════════════════════════════════════════════════════════════════

base = pd.DataFrame(index=core_df.index)
base["log_TSLA"] = np.log(core_df["TSLA"])
base["log_QQQ"]  = np.log(core_df["QQQ"])
base["log_DXY"]  = np.log(core_df["DXY"])
base["log_VIX"]  = np.log(core_df["VIX"])
base["NVDA_excess"] = residualize(np.log(core_df["NVDA"]), base["log_QQQ"])
base["ARKK_excess"] = residualize(np.log(core_df["ARKK"]), base["log_QQQ"])
lr = np.log(core_df["RBOB"])
base["RBOB_zscore_52w"] = (lr - lr.rolling(52, min_periods=20).mean()) / lr.rolling(52, min_periods=20).std()
cl = np.log(core_df["IEF"]) - np.log(core_df["SHY"])
base["curve_IEF_SHY_zscore_52w"] = (cl - cl.rolling(52, min_periods=20).mean()) / cl.rolling(52, min_periods=20).std()

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
    base[f"E_{name}"] = ((base.index >= d0) & (base.index < d0 + pd.Timedelta(weeks=8))).astype(int)

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
# 3. Build VIX term-structure transforms
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building VIX term-structure transforms...")

vix3m = raw["VIX3M"].reindex(f_eq.index, method="ffill")
vix1m = core_df["VIX"].reindex(f_eq.index, method="ffill")

# log(VIX3M/VIX) — positive = contango (calm near-term), negative = backwardation (panic)
ts_ratio = np.log(vix3m / vix1m)
f_eq["vix_ts_ratio"] = ts_ratio

ts_zs = (ts_ratio - ts_ratio.rolling(52, min_periods=20).mean()) / ts_ratio.rolling(52, min_periods=20).std()
f_eq["vix_ts_zscore_52w"] = ts_zs

raw_mom4 = ts_ratio.diff(4)
f_eq["vix_ts_mom4_z"] = (raw_mom4 - raw_mom4.rolling(52, min_periods=20).mean()) / raw_mom4.rolling(52, min_periods=20).std()

# Binary: backwardation regime (VIX > VIX3M, i.e. ts_ratio < 0)
f_eq["vix_backwardation"] = (ts_ratio < 0).astype(float)

CANDIDATES = ["vix_ts_ratio", "vix_ts_zscore_52w", "vix_ts_mom4_z", "vix_backwardation"]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Baseline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Running v6.4 baseline walk-forward...")
base_pred = walk_forward(f_eq, FORCED_V64, active_events)
base_r2, base_mae, base_n = oos_metrics(f_eq, base_pred)
print(f"  v6.4 baseline  OOS R2={base_r2:.4f}  MAE={base_mae:.2f}%  n={base_n}")

_, _, m_full = select_events(f_eq, FORCED_V64, active_events)
resid_v64 = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Screen + walk-forward
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Candidate screen:")
print(f"\n  {'factor':<26}  {'corr_lag1':>9}  {'p':>6}  {'VIF':>6}  screen")
print("  " + "-" * 62)

screen_results = {}
for col in CANDIDATES:
    if f_eq[col].dropna().shape[0] < 30:
        print(f"  {col:<26}  insufficient data"); continue
    corr, p = lagged_corr(col, f_eq, resid_v64)
    vif = compute_vif(f_eq.dropna(subset=[col]), col, FORCED_V64)
    cp = (not np.isnan(p)) and (p < 0.10)
    vp = vif < VIF_WARN
    flag = ("corr-pass" if cp else "corr-fail") + " / " + ("vif-ok" if vp else "vif-HIGH")
    print(f"  {col:<26}  {corr:>+9.4f}  {p:>6.4f}  {vif:>6.2f}  {flag}")
    screen_results[col] = dict(corr=corr, p=p, vif=vif, cp=cp, vp=vp)

print("\n[5] Walk-forward:")
print(f"\n  {'factor':<26}  {'OOS R2':>7}  {'MAE':>7}  {'delta':>8}  verdict")
print("  " + "-" * 66)

wf_results = {}
for col in CANDIDATES:
    sub = f_eq.dropna(subset=[col])
    if len(sub) < MIN_TRAIN + 20:
        print(f"  {col:<26}  insufficient data"); continue
    pred = walk_forward(sub, FORCED_V64 + [col], active_events)
    r2, mae, n = oos_metrics(sub, pred)
    delta = (r2 - base_r2) * 100
    verdict = "ACCEPT" if delta >= BARRIER_PP else "REJECT"
    sr = screen_results.get(col, {})
    print(f"  {col:<26}  {r2:>7.4f}  {mae:>6.2f}%  {delta:>+8.2f}pp  "
          f"{verdict}  (VIF={sr.get('vif', float('nan')):.2f})")
    wf_results[col] = dict(r2=r2, mae=mae, delta=delta, verdict=verdict)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. OOS sub-period corr for primary
# ═══════════════════════════════════════════════════════════════════════════════

primary = "vix_ts_ratio"
print(f"\n[6] OOS sub-period correlation ({primary}):")
c_full, p_full = lagged_corr(primary, f_eq, resid_v64)
print(f"  Full IS lagged corr:   {c_full:+.4f}  p={p_full:.4f}")
c_oos, p_oos, n_oos = oos_subcorr(primary, f_eq, resid_v64)
print(f"  OOS contemp corr:      {c_oos:+.4f}  p={p_oos:.4f}  n={n_oos}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Robustness (only if any candidate clears gate)
# ═══════════════════════════════════════════════════════════════════════════════

accepted = [col for col, r in wf_results.items() if r.get("delta", -999) >= BARRIER_PP]
if accepted:
    col = accepted[0]
    print(f"\n[7] Robustness tenant-swap for {col}:")
    print(f"\n  {'test':<50}  {'OOS R2':>7}  {'delta':>8}")
    print("  " + "-" * 68)
    swap_tests = [
        (f"v6.4 + {col} (primary)",             FORCED_V64 + [col]),
        (f"v6.4 - VIX + {col}",                 [f for f in FORCED_V64 if f != "log_VIX"] + [col]),
        (f"v6.4 - curve + {col}",               [f for f in FORCED_V64 if f != "curve_IEF_SHY_zscore_52w"] + [col]),
        (f"v6.4 - ARKK + {col}",                [f for f in FORCED_V64 if f != "ARKK_excess"] + [col]),
    ]
    for label, forced in swap_tests:
        sub = f_eq.dropna(subset=[c for c in forced if c in f_eq.columns])
        pred = walk_forward(sub, [c for c in forced if c in sub.columns], active_events)
        r2, _, _ = oos_metrics(sub, pred)
        delta = (r2 - base_r2) * 100
        print(f"  {label:<50}  {r2:>7.4f}  {delta:>+8.2f}pp")
    # OOS sub-period for accepted col
    c_oos2, p_oos2, n_oos2 = oos_subcorr(col, f_eq, resid_v64)
    print(f"\n  OOS sub-period corr ({col}): {c_oos2:+.4f}  p={p_oos2:.4f}  n={n_oos2}")
else:
    print(f"\n[7] Robustness skipped -- no candidate cleared +{BARRIER_PP}pp.")

print("\n" + "=" * 68)
print("Done.")

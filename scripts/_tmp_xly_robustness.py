"""
Robustness gauntlet for XLY_excess (level) vs v6.5.

Checks:
  1. Tenant-swap: add XLY_excess while dropping each forced factor in turn.
     A genuine signal should be robust even when individual tenants are removed.
  2. OOS sub-period correlation (confirmed +0.268, p=0.03 in probe).
  3. TSLA self-reference check: TSLA is ~18% of XLY. Does a consumer proxy
     WITHOUT TSLA produce similar signal? Test with:
       - AMZN_excess  (largest XLY component ~25%; no TSLA contamination)
       - XRT_excess   (SPDR Retail ETF — pure consumer-retail, no TSLA)
     If XLY_excess signal is circularity (TSLA driving XLY), AMZN/XRT
     should NOT replicate it. If it IS genuine consumer macro, both should
     show positive signal.

v6.5 forced factors:
  log_QQQ, log_DXY, log_VIX, NVDA_excess, ARKK_excess,
  RBOB_zscore_52w, curve_IEF_SHY_zscore_52w, vix_ts_zscore_52w
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
    factors = forced + events
    m = fit(frame, factors)
    while events:
        ep = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst, wp = max(ep, key=lambda x: x[1])
        if wp <= p_thr:
            break
        events.remove(worst)
        factors = forced + events
        m = fit(frame, factors)
    return factors, events, m


def walk_forward(frame, forced, evs, min_train=MIN_TRAIN):
    pred = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, _, m_tr = select_events(train, forced, evs)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred.loc[date] = float(xv @ m_tr["beta"])
    return pred


def oos_r2(frame, pred):
    idx = pred.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    p = np.exp(pred.loc[idx_oos].to_numpy())
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - ((actual - p) ** 2).sum() / ss_tot
    mae = np.mean(np.abs(actual - p)) / np.mean(actual) * 100
    return float(r2), float(mae), int(len(actual))


def oos_subcorr(col, frame, resid):
    idx = frame.index[frame.index >= pd.Timestamp(OOS_START)]
    cand = frame.loc[idx, col].dropna()
    res = resid.reindex(cand.index).dropna()
    common = cand.index.intersection(res.index)
    if len(common) < 10:
        return np.nan, np.nan, 0
    c, p = stats.pearsonr(cand.loc[common], res.loc[common])
    return c, p, len(common)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("XLY_excess robustness gauntlet vs v6.5")
print("=" * 68)
print("\n[1] Fetching weekly closes...")

TICKERS = {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY", "VIX3M": "^VIX3M",
    "XLY": "XLY", "AMZN": "AMZN", "XRT": "XRT",
}
raw = {}
for k, v in TICKERS.items():
    try:
        raw[k] = wk(v)
        print(f"  {k:<8} ({v:<12})  {len(raw[k])} rows")
    except Exception as e:
        print(f"  {k:<8} FAILED: {e}")

core_df = (pd.DataFrame({k: raw[k] for k in raw})
           .resample("W-FRI").last().ffill().dropna())
print(f"\n  Core: {len(core_df)} weeks  "
      f"{core_df.index[0].date()} -> {core_df.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. v6.5 factor frame
# ═══════════════════════════════════════════════════════════════════════════════

f = pd.DataFrame(index=core_df.index)
f["log_TSLA"] = np.log(core_df["TSLA"])
f["log_QQQ"]  = np.log(core_df["QQQ"])
f["log_DXY"]  = np.log(core_df["DXY"])
f["log_VIX"]  = np.log(core_df["VIX"])
f["NVDA_excess"] = residualize(np.log(core_df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(core_df["ARKK"]), f["log_QQQ"])
lr = np.log(core_df["RBOB"])
f["RBOB_zscore_52w"] = (lr - lr.rolling(52, min_periods=20).mean()) / lr.rolling(52, min_periods=20).std()
cl = np.log(core_df["IEF"]) - np.log(core_df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = (cl - cl.rolling(52, min_periods=20).mean()) / cl.rolling(52, min_periods=20).std()
ts = np.log(core_df["VIX3M"]) - np.log(core_df["VIX"])
f["vix_ts_zscore_52w"] = (ts - ts.rolling(52, min_periods=20).mean()) / ts.rolling(52, min_periods=20).std()

# Candidate and self-reference proxies
f["XLY_excess"]  = residualize(np.log(core_df["XLY"]),  f["log_QQQ"])
f["AMZN_excess"] = residualize(np.log(core_df["AMZN"]), f["log_QQQ"])
f["XRT_excess"]  = residualize(np.log(core_df["XRT"]),  f["log_QQQ"])

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
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

FORCED_V65 = [
    "log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w", "vix_ts_zscore_52w",
]
f_eq = f.dropna(subset=FORCED_V65)
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f_eq[f"E_{n}"].nunique() > 1]
print(f"\n  v6.5 factor frame: {len(f_eq)} weeks  "
      f"IS={(f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Baselines
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Baselines (v6.5 and v6.5 + XLY_excess)...")
pred_base = walk_forward(f_eq, FORCED_V65, active_events)
r2_base, mae_base, _ = oos_r2(f_eq, pred_base)
print(f"  v6.5 baseline:            OOS R²={r2_base:.4f}  MAE={mae_base:.2f}%")

pred_xly = walk_forward(f_eq, FORCED_V65 + ["XLY_excess"], active_events)
r2_xly, mae_xly, _ = oos_r2(f_eq, pred_xly)
delta_xly = (r2_xly - r2_base) * 100
print(f"  v6.5 + XLY_excess:        OOS R²={r2_xly:.4f}  Δ={delta_xly:+.2f}pp  MAE={mae_xly:.2f}%")

# Full-sample residuals for correlation checks
_, _, m_full = select_events(f_eq, FORCED_V65, active_events)
resid_full = m_full["resid"]
c_oos, p_oos, n_oos = oos_subcorr("XLY_excess", f_eq, resid_full)
print(f"  OOS sub-period corr:      {c_oos:+.3f}  p={p_oos:.3f}  n={n_oos}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Tenant-swap tests
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Tenant-swap (drop one forced factor + XLY_excess)...")
tenant_results = []
for drop in FORCED_V65:
    reduced = [c for c in FORCED_V65 if c != drop]
    pred_r = walk_forward(f_eq, reduced + ["XLY_excess"], active_events)
    r2_r, _, _ = oos_r2(f_eq, pred_r)
    delta_r = (r2_r - r2_base) * 100
    print(f"  − {drop:<30}  OOS R²={r2_r:.4f}  Δ vs v6.5={delta_r:+.2f}pp")
    tenant_results.append((drop, r2_r, delta_r))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Self-reference check: AMZN_excess and XRT_excess
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Self-reference check (TSLA ~18% of XLY)...")
print("  If XLY_excess is circular (TSLA driving XLY), AMZN/XRT should NOT lift.")
print("  If genuine consumer macro, AMZN/XRT should also show positive lift.")
print()

for proxy, label in [
    ("AMZN_excess", "AMZN_excess (largest XLY component, no TSLA)"),
    ("XRT_excess",  "XRT_excess  (retail ETF, no TSLA component)"),
]:
    sub = f_eq.dropna(subset=[proxy])
    pred_p = walk_forward(sub, FORCED_V65 + [proxy], active_events)
    r2_p, mae_p, _ = oos_r2(sub, pred_p)
    delta_p = (r2_p - r2_base) * 100
    c_p, p_p, n_p = oos_subcorr(proxy, f_eq, resid_full)
    print(f"  {label}")
    print(f"    OOS R²={r2_p:.4f}  Δ={delta_p:+.2f}pp  OOS-corr={c_p:+.3f}(p={p_p:.2f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("SUMMARY — XLY_excess robustness")
print("=" * 68)
print(f"  v6.5 baseline:        OOS R²={r2_base:.4f}")
print(f"  + XLY_excess:         OOS R²={r2_xly:.4f}  Δ={delta_xly:+.2f}pp")
print(f"  OOS sub-period corr:  {c_oos:+.3f}  p={p_oos:.3f}  n={n_oos}")
print()
print("  Tenant-swap results:")
for drop, r2_r, delta_r in tenant_results:
    print(f"    − {drop:<30}  Δ={delta_r:+.2f}pp")
print()

# Decision logic
passes = 0
flags = []
if delta_xly >= 1.0:
    passes += 1
else:
    flags.append(f"PRIMARY LIFT {delta_xly:+.2f}pp < +1pp gate")
if c_oos > 0:
    passes += 1
else:
    flags.append(f"OOS corr sign-flipped ({c_oos:+.3f})")
min_tenant = min(d for _, _, d in tenant_results)
if min_tenant > -5.0:
    passes += 1
else:
    flags.append(f"Tenant swap worst case {min_tenant:+.2f}pp")

print(f"  Checks passed: {passes}/3")
if flags:
    print("  Flags:")
    for fl in flags:
        print(f"    ✗ {fl}")
if passes == 3:
    print("  → All checks passed. Promote to full gauntlet if self-reference clean.")
elif passes >= 2:
    print("  → Partial pass. Review self-reference check before decision.")
else:
    print("  → REJECT.")

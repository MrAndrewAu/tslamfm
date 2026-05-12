"""
Probe: Fed balance sheet (WALCL) as additive v6.4 factor.

Data source: FRED series WALCL — "Assets: Total Assets: Total Assets
(Less Eliminations from Consolidation)" — the Fed's weekly balance sheet.
Units: millions of dollars. Published Wednesdays with data through the
prior Wednesday. We apply a 1-week release-lag shift (conservative).

Economic story:
  WALCL IS the QE/QT policy instrument — not downstream like M2.
  - QE: Fed buys Treasuries/MBS → balance sheet expands → reserves flood
    into risk assets → multiples inflate. TSLA as a high-beta growth stock
    should be disproportionately sensitive.
  - QT: Fed stops reinvesting maturities → balance sheet shrinks → risk
    premium rises → speculative multiples compress.
  WALCL collapsed ~$1T (2022-2023 QT), rebounded partially in 2024.
  This is the cleanest version of the liquidity-regime hypothesis: it IS
  the policy instrument, not a lagged consequence (unlike M2).
  Orthogonal to: yield curve (prices the rate level), VIX (fear), RBOB
  (inflation demand), QQQ (price of tech equity).

Transforms:
  A. yoy_z   — YoY % change z-scored (regime: QE vs QT)
  B. zscore_52w — z-score of log level vs 52w (trend-adjusted level)
  C. mom4_z  — 4-week change z-scored (fast-moving flow proxy)

Acceptance bar: +2pp walk-forward OOS R² lift vs v6.4 baseline.
"""

import io, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request as _urllib
from scipy import stats

warnings.filterwarnings("ignore")

START, END   = "2022-04-26", "2026-04-26"
OOS_START    = "2025-01-03"
MIN_TRAIN    = 100
EVENT_P_THR  = 0.10
BARRIER_PP   = 2.0
VIF_WARN     = 5.0
RELEASE_LAG  = 1   # weeks

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# ── FRED fetch ────────────────────────────────────────────────────────────────

def fetch_fred(series_id):
    url = FRED_BASE + series_id
    try:
        with _urllib.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        date_col = next(c for c in df.columns if "date" in c.lower())
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        s_wk = s.resample("W-FRI").last().ffill()
        s_wk.index = s_wk.index + pd.Timedelta(weeks=RELEASE_LAG)
        print(f"  {series_id}: {len(s_wk)} weekly rows after {RELEASE_LAG}w lag"
              f"  ({s_wk.index[0].date()} → {s_wk.index[-1].date()})")
        return s_wk
    except Exception as e:
        raise RuntimeError(f"FRED fetch failed for {series_id}: {e}") from e

# ── OLS helpers ───────────────────────────────────────────────────────────────

def wk(sym):
    for kwargs in [
        dict(start=START, end=END, interval="1wk"),
        dict(start=START, end=END, interval="1wk", auto_adjust=False),
    ]:
        try:
            dl = yf.download(sym, progress=False, **kwargs)
            if dl.empty:
                continue
            # Flatten MultiIndex columns (yfinance ≥0.2 returns them)
            if isinstance(dl.columns, pd.MultiIndex):
                dl.columns = dl.columns.get_level_values(0)
            s = dl["Close"] if "Close" in dl.columns else dl.iloc[:, 0]
            # squeeze DataFrame → Series if needed
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


def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
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


def compute_vif(frame, factors):
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


def lagged_corr(candidate, resid_v64):
    c_lag = candidate.shift(1)
    common = resid_v64.index.intersection(c_lag.dropna().index)
    if len(common) < 20:
        return np.nan, np.nan
    return stats.pearsonr(c_lag.loc[common], resid_v64.loc[common])

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch WALCL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Fed Balance Sheet (WALCL) — additive factor probe vs v6.4")
print("=" * 65)
print(f"\n[1] Fetching FRED WALCL (1-week release-lag shift applied)...")

walcl = fetch_fred("WALCL")

# Sanity check — should be ~$6-7T range in millions
recent = walcl.iloc[-4:]
print(f"  Recent WALCL (millions $): {recent.values.round(0)}")
print(f"  → ${walcl.iloc[-1]/1e6:.2f}T  (should be ~$6-7T range)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Equity factor frame (v6.4)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Fetching weekly equity closes...")
raw = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB",
    "VIX":  "^VIX",  "NVDA": "NVDA",  "ARKK": "ARKK",
    "RBOB": "RB=F",  "IEF":  "IEF",   "SHY":  "SHY",
}.items()}
df = pd.DataFrame(raw).resample("W-FRI").last().ffill().dropna()
print(f"  {len(df)} weeks  {df.index[0].date()} → {df.index[-1].date()}")

base = pd.DataFrame(index=df.index)
base["log_TSLA"] = np.log(df["TSLA"])
base["log_QQQ"]  = np.log(df["QQQ"])
base["log_DXY"]  = np.log(df["DXY"])
base["log_VIX"]  = np.log(df["VIX"])
base["NVDA_excess"] = residualize(np.log(df["NVDA"]), base["log_QQQ"])
base["ARKK_excess"] = residualize(np.log(df["ARKK"]), base["log_QQQ"])
log_rbob = np.log(df["RBOB"])
base["RBOB_zscore_52w"] = (
    (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) /
    log_rbob.rolling(52, min_periods=20).std()
)
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
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
    base[f"E_{name}"] = ((base.index >= d0) &
                          (base.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f_eq = base.dropna(subset=[
    "log_TSLA", "log_QQQ", "log_DXY", "log_VIX",
    "NVDA_excess", "ARKK_excess",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w",
])
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f_eq[f"E_{n}"].nunique() > 1]
FORCED_V64 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

print(f"  Equity frame: {len(f_eq)} weeks  IS={(f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build WALCL transforms
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Building WALCL transforms (look-ahead free)...")
log_w = np.log(walcl)

walcl_yoy_z_raw    = log_w.diff(52) * 100
walcl_yoy_z        = (walcl_yoy_z_raw -
                      walcl_yoy_z_raw.rolling(52, min_periods=20).mean()) / \
                     walcl_yoy_z_raw.rolling(52, min_periods=20).std()

walcl_zscore_52w   = (log_w - log_w.rolling(52, min_periods=20).mean()) / \
                     log_w.rolling(52, min_periods=20).std()

walcl_mom4_raw     = log_w.diff(4) * 100
walcl_mom4_z       = (walcl_mom4_raw -
                      walcl_mom4_raw.rolling(52, min_periods=20).mean()) / \
                     walcl_mom4_raw.rolling(52, min_periods=20).std()

transforms = {
    "WALCL_yoy_z":     walcl_yoy_z,
    "WALCL_zscore_52w": walcl_zscore_52w,
    "WALCL_mom4_z":    walcl_mom4_z,
}

for name, s in transforms.items():
    aligned = s.reindex(f_eq.index, method="ffill")
    f_eq[name] = aligned
    print(f"  {name:<25}  {aligned.dropna().shape[0]} non-null rows  "
          f"mean={aligned.mean():.3f}  std={aligned.std():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. v6.4 baseline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] v6.4 baseline walk-forward...")
pred_base = walk_forward(f_eq, FORCED_V64, active_events)
base_r2, base_mae, n_oos = oos_metrics(f_eq, pred_base)
print(f"  OOS R²={base_r2:.4f}  MAE={base_mae:.2f}%  n_oos={n_oos}")

_, _, m_full = select_events(f_eq, FORCED_V64, active_events)
resid_v64 = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Correlation screen + VIF
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5] Candidate screen (lagged corr vs v6.4 residuals + VIF)...")
print(f"  {'candidate':<25}  {'n':>5}  {'corr_lag1':>10}  {'p':>8}  {'VIF':>6}  screen")

for cname in transforms:
    sub = f_eq.dropna(subset=[cname])
    if len(sub) < 50:
        print(f"  {cname:<25}  (too few rows: {len(sub)})")
        continue
    corr, p = lagged_corr(sub[cname], resid_v64.reindex(sub.index).dropna())
    vifs = compute_vif(sub.dropna(subset=FORCED_V64 + [cname]),
                       FORCED_V64 + [cname])
    vif = vifs.get(cname, np.nan)
    c_verdict = "corr-pass" if abs(corr) >= 0.10 and p < 0.20 else "corr-fail"
    v_verdict = "vif-ok" if vif < VIF_WARN else "vif-HIGH"
    print(f"  {cname:<25}  {len(sub):>5}  {corr:>10.4f}  {p:>8.4f}  {vif:>6.2f}  "
          f"{c_verdict} | {v_verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Walk-forward for all three transforms
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6] Walk-forward OOS (all transforms)...")
print(f"  {'variant':<38}  {'OOS R²':>7}  {'MAE%':>7}  {'Δ':>8}  verdict")
print(f"  {'v6.4 baseline':<38}  {base_r2:>7.4f}  {base_mae:>6.2f}%  (ref)")

for cname in transforms:
    sub = f_eq.dropna(subset=[cname])
    if len(sub) < MIN_TRAIN + 20:
        print(f"  {'v6.4 + ' + cname:<38}  (too few rows)")
        continue
    pred = walk_forward(sub, FORCED_V64 + [cname], active_events)
    r2, mae, _ = oos_metrics(sub, pred)
    delta = (r2 - base_r2) * 100
    verdict = "ACCEPT" if delta >= BARRIER_PP else "REJECT"
    print(f"  {'v6.4 + ' + cname:<38}  {r2:>7.4f}  {mae:>6.2f}%  {delta:>+7.2f}pp  {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Recent data snapshot
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7] WALCL transforms vs v6.4 residual (last 16 observations):")
print(f"  {'date':<12}  {'yoy_z':>8}  {'z52w':>8}  {'mom4_z':>8}  {'v64_resid':>10}")
snap = f_eq[["WALCL_yoy_z", "WALCL_zscore_52w", "WALCL_mom4_z"]].join(
    resid_v64.rename("resid")).dropna()
for dt, row in snap.iloc[-16:].iterrows():
    print(f"  {str(dt.date()):<12}  {row['WALCL_yoy_z']:>8.3f}  "
          f"{row['WALCL_zscore_52w']:>8.3f}  {row['WALCL_mom4_z']:>8.3f}  "
          f"{row['resid']:>10.4f}")

print("\n" + "=" * 65)
print(f"  v6.4 baseline OOS R²={base_r2:.4f}  MAE={base_mae:.2f}%")
print(f"  Acceptance bar: +{BARRIER_PP}pp")
print(f"  Note: WALCL is the direct QE/QT instrument (not downstream like M2).")
print(f"  curve_IEF_SHY_zscore_52w already encodes rate-regime; this tests")
print(f"  whether the SIZE of the balance sheet adds orthogonal info.")

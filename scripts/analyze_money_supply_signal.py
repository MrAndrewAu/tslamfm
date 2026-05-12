"""
Probe: US M1 and M2 money supply as additive v6.4 factors.

Data source: FRED (Federal Reserve Economic Data) public CSV endpoint.
  No API key required. URLs:
    Weekly M1 (NSA): https://fred.stlouisfed.org/graph/fredgraph.csv?id=WM1NS
    Weekly M2 (NSA): https://fred.stlouisfed.org/graph/fredgraph.csv?id=WM2NS
  Release lag: FRED publishes weekly M1/M2 on Thursdays with data through
  the prior Monday. We shift all series forward by 1 week before aligning
  to Friday closes to ensure ZERO look-ahead.

Economic story:
  M2 growth = liquidity expansion → inflates multiples on risk assets.
  QT (M2 contraction 2022-2023) was a documented headwind for speculative
  equities including TSLA. If the liquidity regime genuinely drives TSLA's
  factor residual, the YoY growth rate (z-scored) should predict it.
  This is orthogonal to: QQQ (price), VIX (fear), curve (rate pricing),
  RBOB (inflation demand). Plausibly uncorrelated — worth testing.

Candidates (both M1 and M2, same transforms):
  A. yoy_z   — YoY % growth, z-scored vs 52-week rolling history
  B. zscore_52w — z-score of log level vs trailing 52 weeks (trend-adjusted)
  C. mom_z   — 4-week (MoM-proxy) % change, z-scored (faster-moving)

Acceptance bar: +2pp walk-forward OOS R² lift vs v6.4 baseline.
"""

import io, warnings
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import urllib.request as _urllib
    def _fetch_csv(url):
        with _urllib.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8")
except Exception:
    _fetch_csv = None

from scipy import stats
warnings.filterwarnings("ignore")

START, END   = "2022-04-26", "2026-04-26"
OOS_START    = "2025-01-03"
MIN_TRAIN    = 100
EVENT_P_THR  = 0.10
BARRIER_PP   = 2.0
VIF_WARN     = 5.0
RELEASE_LAG  = 1   # weeks to shift FRED data forward (conservative look-ahead guard)

# ── FRED fetch ────────────────────────────────────────────────────────────────

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

def fetch_fred(series_id):
    """Download a FRED series as a weekly-aligned pandas Series.
    Returns log-level series resampled to W-FRI, shifted by RELEASE_LAG weeks."""
    url = FRED_BASE + series_id
    try:
        raw = _fetch_csv(url)
        df = pd.read_csv(io.StringIO(raw))
        # FRED uses "observation_date" as the date column name
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        s  = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        # Resample to weekly Friday close (forward-fill within-week)
        s_wk = s.resample("W-FRI").last().ffill()
        # Shift forward by RELEASE_LAG weeks to enforce no look-ahead
        s_wk.index = s_wk.index + pd.Timedelta(weeks=RELEASE_LAG)
        print(f"  {series_id}: {len(s_wk)} weekly rows after {RELEASE_LAG}w lag shift  "
              f"({s_wk.index[0].date()} → {s_wk.index[-1].date()})")
        return s_wk
    except Exception as e:
        raise RuntimeError(f"FRED fetch failed for {series_id}: {e}") from e


# ── OLS helpers ───────────────────────────────────────────────────────────────

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
    """Weekly lagged correlation (candidate at t vs resid at t+1)."""
    c_lag = candidate.shift(1)
    common = resid_v64.index.intersection(c_lag.dropna().index)
    if len(common) < 20:
        return np.nan, np.nan
    return stats.pearsonr(c_lag.loc[common], resid_v64.loc[common])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch FRED money supply
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("US M1 / M2 Money Supply — additive factor probe vs v6.4")
print("=" * 65)
print(f"\n[1] Fetching FRED data (1-week release-lag shift applied)...")

m1_wk = fetch_fred("WM1NS")   # Weekly M1, not seasonally adjusted (billions $)
m2_wk = fetch_fred("WM2NS")   # Weekly M2, not seasonally adjusted (billions $)

# Show recent values for sanity check
print(f"\n  M1 (recent): {m1_wk.iloc[-3].round(1)} → {m1_wk.iloc[-1].round(1)} B$")
print(f"  M2 (recent): {m2_wk.iloc[-3].round(1)} → {m2_wk.iloc[-1].round(1)} B$")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Weekly price data + v6.4 factor frame
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

print(f"  Equity frame: {len(f_eq)} weeks  IS={( f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build money supply transforms and align to equity frame
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Building money supply transforms (look-ahead free)...")

def build_money_transforms(m_wk, label, frame_index):
    """Returns dict of {name: weekly Series} aligned to frame_index."""
    log_m = np.log(m_wk)
    # YoY growth = % change vs 52 weeks ago (look-behind only)
    yoy = log_m.diff(52) * 100   # approx YoY % in log-space ≈ true YoY %
    yoy_z = (yoy - yoy.rolling(52, min_periods=20).mean()) / yoy.rolling(52, min_periods=20).std()
    # z-score of log level vs trailing 52w
    zscore = (log_m - log_m.rolling(52, min_periods=20).mean()) / log_m.rolling(52, min_periods=20).std()
    # 4-week MoM-proxy change (z-scored)
    mom4  = log_m.diff(4) * 100
    mom4_z = (mom4 - mom4.rolling(52, min_periods=20).mean()) / mom4.rolling(52, min_periods=20).std()

    out = {}
    for name, s in [(f"{label}_yoy_z", yoy_z),
                    (f"{label}_zscore_52w", zscore),
                    (f"{label}_mom4_z", mom4_z)]:
        # align to equity frame index (forward-fill any small gaps)
        aligned = s.reindex(frame_index, method="ffill")
        out[name] = aligned
        n_valid = aligned.dropna().shape[0]
        print(f"  {name:<30}  {n_valid} non-null rows")
    return out

m1_transforms = build_money_transforms(m1_wk, "M1", f_eq.index)
m2_transforms = build_money_transforms(m2_wk, "M2", f_eq.index)
all_transforms = {**m1_transforms, **m2_transforms}

# Add to frame
for name, s in all_transforms.items():
    f_eq[name] = s

# ═══════════════════════════════════════════════════════════════════════════════
# 4. v6.4 baseline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] v6.4 baseline walk-forward...")
pred_base = walk_forward(f_eq, FORCED_V64, active_events)
base_r2, base_mae, n_oos = oos_metrics(f_eq, pred_base)
print(f"  OOS R²={base_r2:.4f}  MAE={base_mae:.2f}%  n_oos={n_oos}")

# ── v6.4 residuals for lagged-correlation screen ──────────────────────────────
_, _, m_full = select_events(f_eq, FORCED_V64, active_events)
resid_v64 = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Candidate screen: lagged correlation + VIF
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5] Candidate screen (lagged corr vs v6.4 residuals + VIF)...")
print(f"  {'candidate':<30}  {'n':>5}  {'corr_lag1':>10}  {'p':>8}  {'VIF':>6}  screen")

screened = []
for cname, cs in all_transforms.items():
    sub = f_eq.dropna(subset=[cname])
    if len(sub) < 50:
        print(f"  {cname:<30}  (too few rows: {len(sub)})")
        continue
    # Lagged corr
    corr, p = lagged_corr(sub[cname], resid_v64.reindex(sub.index).dropna())
    # VIF
    vif_factors = FORCED_V64 + [cname]
    vif_sub = sub.dropna(subset=vif_factors)
    vifs = compute_vif(vif_sub, vif_factors)
    vif = vifs.get(cname, np.nan)
    verdict = []
    if abs(corr) >= 0.10 and p < 0.20:
        verdict.append("corr-pass")
    else:
        verdict.append("corr-fail")
    if vif < VIF_WARN:
        verdict.append("vif-ok")
    else:
        verdict.append("vif-HIGH")
    screened.append((cname, corr, p, vif, " | ".join(verdict)))
    print(f"  {cname:<30}  {len(sub):>5}  {corr:>10.4f}  {p:>8.4f}  {vif:>6.2f}  {' | '.join(verdict)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Walk-forward for all candidates (run all regardless — document fully)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6] Walk-forward OOS for all candidates...")
print(f"  {'variant':<40}  {'OOS R²':>7}  {'MAE%':>7}  {'Δ':>8}  verdict")
print(f"  {'v6.4 baseline':<40}  {base_r2:>7.4f}  {base_mae:>6.2f}%  (ref)")

for cname in all_transforms:
    sub = f_eq.dropna(subset=[cname])
    if len(sub) < MIN_TRAIN + 20:
        print(f"  {'v6.4 + ' + cname:<40}  (too few rows: {len(sub)})")
        continue
    pred = walk_forward(sub, FORCED_V64 + [cname], active_events)
    r2, mae, _ = oos_metrics(sub, pred)
    delta = (r2 - base_r2) * 100
    verdict = "ACCEPT" if delta >= BARRIER_PP else "REJECT"
    print(f"  {'v6.4 + ' + cname:<40}  {r2:>7.4f}  {mae:>6.2f}%  {delta:>+7.2f}pp  {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Visual snapshot: M2 YoY growth vs TSLA log residual
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7] M2 YoY growth vs v6.4 residual (last 20 observations):")
print(f"  {'date':<12}  {'M2_yoy_z':>10}  {'M1_yoy_z':>10}  {'v64_resid':>10}")
snap = f_eq[["M2_yoy_z", "M1_yoy_z"]].join(resid_v64.rename("resid")).dropna()
for dt, row in snap.iloc[-20:].iterrows():
    print(f"  {str(dt.date()):<12}  {row['M2_yoy_z']:>10.3f}  {row['M1_yoy_z']:>10.3f}  {row['resid']:>10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  v6.4 baseline OOS R²={base_r2:.4f}  MAE={base_mae:.2f}%")
print(f"  Acceptance bar: +{BARRIER_PP}pp")
print()
print("  Data: FRED WM1NS / WM2NS  |  Release lag: 1 week shift applied")
print("  Economic story: M2 YoY growth → liquidity regime → risk multiples")
print()
print("  Note: M2 contracted 2022-2023 (Fed QT) and expanded in 2020-2021.")
print("  The 4-year window (2022-2026) spans the full QT + re-expansion cycle,")
print("  so this is a well-powered test of the liquidity-premium thesis.")

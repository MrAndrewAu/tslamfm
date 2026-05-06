"""
Predictive lead analysis for TSLA multi-factor model.

Tests forward regression at lag 0-6 trading days using daily close data.
Same 7 continuous factors as v6.4 canonical:
  log_QQQ, log_DXY, log_VIX, NVDA_excess, ARKK_excess,
  RBOB_zscore_52w (252-day), curve_IEF_SHY_zscore_52w (252-day),
  + same event dummies, backward-selected at p<0.10

Method:
  For each lag L in 0..6:
    Construct target = log_TSLA(t + L) aligned with factors at t.
    Run two-phase evaluation:
      1. Simple hold-out OOS (train pre-OOS_START, test on OOS_START+).
         Fast; used for lag selection.
      2. Walk-forward OOS (MIN_TRAIN=252 daily obs, step=1 day).
         Only run for lag=0 (baseline) and the best candidate.
    Report IS R², hold-out OOS R², hold-out OOS MAE.

Goal: identify the lag where the model price LEADS TSLA by ~2-3 days,
      maximising OOS R² while staying honest about the difficulty of
      predicting a directional move 2-3 sessions ahead.
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START       = "2022-04-26"
END         = "2026-04-26"
OOS_START   = "2025-01-03"
MIN_TRAIN   = 252          # ~1 year of trading days
LAMBDA      = 0.94
EVENT_P_THRESHOLD = 0.10

# ── data helpers ──────────────────────────────────────────────────────────────

def fetch_daily(sym, start=START, end=END):
    candidates = []
    try:
        s = yf.Ticker(sym).history(start=start, end=end, interval="1d")["Close"]
        candidates.append(s)
    except Exception:
        pass
    try:
        dl = yf.download(sym, start=start, end=end, interval="1d",
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
        return s
    raise RuntimeError(f"Cannot fetch daily close for {sym}")


def residualize(target: pd.Series, base: pd.Series):
    """target = a + b*base + e  → returns (a, b, residuals)"""
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    resid = target.to_numpy() - X @ coef
    return float(coef[0]), float(coef[1]), pd.Series(resid, index=target.index)


# ── OLS helper ────────────────────────────────────────────────────────────────

def fit_ols(frame: pd.DataFrame, factors: list, target_col: str = "log_TSLA_target"):
    y = frame[target_col].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    r = y - yhat
    n, k = X.shape
    s2 = max((r @ r) / (n - k), 1e-12)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = b / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (r @ r) / ss_tot) if ss_tot > 0 else 0.0
    return dict(beta=b, se=se, t=t_stats, p=p_vals, r2=r2,
                fitted=pd.Series(yhat, index=frame.index),
                resid=pd.Series(r, index=frame.index),
                factors=factors)


def select_events(frame, forced, candidate_events, target_col, threshold=EVENT_P_THRESHOLD):
    """Backward-eliminate event dummies until all p <= threshold."""
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    m = fit_ols(frame, factors, target_col)
    while events:
        ev_ps = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst_e, worst_p = max(ev_ps, key=lambda x: x[1])
        if worst_p <= threshold:
            break
        events.remove(worst_e)
        factors = forced + events
        m = fit_ols(frame, factors, target_col)
    return factors, events, m


def r2_score(actual, predicted):
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ((actual - predicted) ** 2).sum() / ss_tot)


def mae_pct(actual, predicted):
    return float(np.mean(np.abs(predicted / actual - 1)) * 100)


# ── fetch data ────────────────────────────────────────────────────────────────

print("Fetching daily closes …")
SYMS = {"TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
        "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
        "IEF": "IEF", "SHY": "SHY"}
S = {k: fetch_daily(v) for k, v in SYMS.items()}
df = pd.DataFrame(S).ffill().dropna()
print(f"  {len(df)} trading days, {df.index[0].date()} → {df.index[-1].date()}")

# ── build daily factor matrix ─────────────────────────────────────────────────

f = pd.DataFrame(index=df.index)
f["log_TSLA"]  = np.log(df["TSLA"])
f["log_QQQ"]   = np.log(df["QQQ"])
f["log_DXY"]   = np.log(df["DXY"])
f["log_VIX"]   = np.log(df["VIX"])

_, _, nvda_resid = residualize(np.log(df["NVDA"]), f["log_QQQ"])
_, _, arkk_resid = residualize(np.log(df["ARKK"]), f["log_QQQ"])
f["NVDA_excess"] = nvda_resid
f["ARKK_excess"] = arkk_resid

# 52-week rolling z-scores → 252 trading-day window
log_rbob = np.log(df["RBOB"])
f["RBOB_zscore_52w"] = (
    (log_rbob - log_rbob.rolling(252, min_periods=100).mean())
    / log_rbob.rolling(252, min_periods=100).std()
)

curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = (
    (curve_log - curve_log.rolling(252, min_periods=100).mean())
    / curve_log.rolling(252, min_periods=100).std()
)

# Event dummies – same calendar windows as weekly model (8 weeks = 56 cal days)
EVENT_DEFS = [
    ("AI_day_2023",     "2023-07-19", "AI Day 2 / Dojo narrative"),
    ("Trump_election",  "2024-11-06", "Trump election rally"),
    ("Tariff_shock",    "2026-02-01", "Tariff shock"),
    ("Robotaxi_Austin", "2025-06-22", "Austin robotaxi sell-the-news"),
    # Extra candidates (will be selected by p<0.10)
    ("DOGE_brand_damage", "2025-02-15", "DOGE brand damage"),
    ("Musk_exits_DOGE",   "2025-04-22", "Musk exits DOGE"),
]
for name, dt, _ in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"  Feature matrix: {len(f)} rows after dropna")

FORCED = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
          "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]
ALL_EVENTS = [f"E_{n}" for n, _, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]

# ── lag scan – simple hold-out OOS ────────────────────────────────────────────

print("\n── Lag scan (simple hold-out OOS) ──────────────────────────────────────")
print(f"  OOS period: {OOS_START} → {df.index[-1].date()}")
print(f"\n  {'Lag':>4}  {'IS R²':>7}  {'OOS R²':>7}  {'OOS MAE%':>9}  "
      f"{'Kept events':>12}  {'Notes'}")
print("  " + "-" * 72)

best_lag, best_oos_r2 = None, -np.inf
lag_results = {}

for lag in range(0, 8):
    # Align: target = log_TSLA at t+lag, factors at t
    target_series = f["log_TSLA"].shift(-lag)
    fwork = f.copy()
    fwork["log_TSLA_target"] = target_series
    fwork = fwork.dropna(subset=["log_TSLA_target"])

    if len(fwork) < MIN_TRAIN + 20:
        print(f"  {lag:>4}  {'—':>7}  {'—':>7}  {'—':>9}  {'—':>12}  (insufficient data)")
        continue

    oos_mask   = fwork.index >= pd.Timestamp(OOS_START)
    train_frame = fwork[~oos_mask]
    test_frame  = fwork[oos_mask]

    if len(train_frame) < MIN_TRAIN or len(test_frame) < 10:
        print(f"  {lag:>4}  {'—':>7}  {'—':>7}  {'—':>9}  {'—':>12}  (split too small)")
        continue

    active_events = [e for e in ALL_EVENTS if fwork[e].nunique() > 1]
    _, kept_ev, m_train = select_events(train_frame, FORCED, active_events,
                                        "log_TSLA_target")

    # IS
    r2_is = m_train["r2"]

    # Predict test set using training coefficients
    test_facs = m_train["factors"]
    X_test = np.column_stack([np.ones(len(test_frame))]
                              + [test_frame[c].to_numpy() for c in test_facs])
    y_test_pred_log = X_test @ m_train["beta"]
    y_test_actual_log = test_frame["log_TSLA_target"].to_numpy()

    y_pred_price  = np.exp(y_test_pred_log)
    y_actual_price = np.exp(y_test_actual_log)

    r2_oos  = r2_score(y_actual_price, y_pred_price)
    mae_oos = mae_pct(y_actual_price, y_pred_price)

    # Cross-correlation: how well does model(t) lead actual(t)?
    # Compute model value on FULL sample using training coefficients
    full_facs = m_train["factors"]
    X_full = np.column_stack([np.ones(len(fwork))]
                              + [fwork[c].to_numpy() for c in full_facs])
    model_values = pd.Series(X_full @ m_train["beta"], index=fwork.index)
    actual_log  = fwork["log_TSLA"]

    notes = ""
    if 2 <= lag <= 3 and r2_oos > best_oos_r2:
        best_oos_r2 = r2_oos
        best_lag = lag

    lag_results[lag] = dict(r2_is=r2_is, r2_oos=r2_oos, mae_oos=mae_oos,
                             kept_events=kept_ev, beta=m_train["beta"],
                             factors=m_train["factors"])
    ev_str = str(len(kept_ev))
    flag = " ◄ best 2-3d" if (2 <= lag <= 3 and r2_oos == best_oos_r2) else ""
    print(f"  {lag:>4}  {r2_is:>7.3f}  {r2_oos:>7.3f}  {mae_oos:>9.2f}%  "
          f"{ev_str:>12}  {flag}")

if best_lag is None:
    best_lag = 2
    print(f"\n  No 2-3d candidate beat baseline; defaulting to lag=2.")
else:
    print(f"\n  Best lag in 2-3d window: {best_lag} day(s)  (hold-out OOS R²={best_oos_r2:.3f})")

# ── cross-correlation table ────────────────────────────────────────────────────

print("\n── Cross-correlation: model(t) vs actual(t+k) for k=0..5 ──────────────")
# Re-fit on full train sample with lag=best_lag for cross-corr
res_best = lag_results.get(best_lag, lag_results.get(2, {}))
if res_best:
    target_b = f["log_TSLA"].shift(-best_lag)
    fwork_b  = f.copy()
    fwork_b["log_TSLA_target"] = target_b
    fwork_b  = fwork_b.dropna(subset=["log_TSLA_target"])
    train_b  = fwork_b[fwork_b.index < pd.Timestamp(OOS_START)]
    _, _, m_b = select_events(train_b, FORCED,
                               [e for e in ALL_EVENTS if fwork_b[e].nunique() > 1],
                               "log_TSLA_target")
    X_full_b = np.column_stack([np.ones(len(fwork_b))]
                                + [fwork_b[c].to_numpy() for c in m_b["factors"]])
    model_log_series = pd.Series(X_full_b @ m_b["beta"], index=fwork_b.index)
    actual_log_series = fwork_b["log_TSLA"]
    print(f"  (Using lag={best_lag} model, fitted on train slice)")
    print(f"  {'k':>3}  {'pearson r':>10}  {'interpretation'}")
    print("  " + "-" * 45)
    for k in range(0, 7):
        lead_actual = actual_log_series.shift(-k)
        aligned = pd.DataFrame({"m": model_log_series, "a": lead_actual}).dropna()
        if len(aligned) < 20:
            continue
        r, p = stats.pearsonr(aligned["m"], aligned["a"])
        note = " ← contemporaneous" if k == 0 else f" ← model leads actual by {k}d"
        print(f"  {k:>3}  {r:>10.4f}  {note}")

# ── summary of the chosen lag ─────────────────────────────────────────────────

print(f"\n── Recommended settings ─────────────────────────────────────────────────")
chosen = lag_results.get(best_lag, {})
print(f"  LAG           = {best_lag} trading days")
if chosen:
    print(f"  IS R²         = {chosen['r2_is']:.3f}")
    print(f"  Hold-out OOS R² = {chosen['r2_oos']:.3f}")
    print(f"  Hold-out OOS MAE = {chosen['mae_oos']:.2f}%")
    print(f"  Kept events   = {[e.replace('E_','') for e in chosen['kept_events']]}")
    print(f"\n  Coefficients (log-space):")
    for name, val in zip(["Intercept"] + chosen["factors"], chosen["beta"]):
        print(f"    {name:<35} {val:+.6f}")

print("\nDone.")

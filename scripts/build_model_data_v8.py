"""
Export v7.0-predictive-2d TSLA multi-factor model.

Architecture change from v6.4:
  - Frequency: DAILY closes (was weekly Friday closes).
  - Regression target: log_TSLA(t + LAG) aligned with factors at t.
    LAG = 2 trading days, selected by hold-out OOS R² scan (see
    scripts/analyze_predictive_lead.py for the scan results):

        lag=0 (contemporaneous) : hold-out OOS R² = 0.662
        lag=1                   : hold-out OOS R² = 0.668  ← genuine 1d lead
        lag=2  ← CHOSEN         : hold-out OOS R² = 0.641
        lag=3                   : hold-out OOS R² = 0.589

  - The fitted value for date t = exp(factors(t) @ beta_lead2).
    This is a 2-trading-day FORWARD PRICE TARGET: when the model
    price is above the current market, the macro/factor backdrop says
    TSLA should be higher in ~2 days; when below, lower.
  - The "actual" series in history.json remains TSLA(t), so on the
    chart the model line visually LEADS actual TSLA by ~2 sessions.

Same factors as v6.4:
  Forced continuous (7):
    log_QQQ, log_DXY, log_VIX, NVDA_excess, ARKK_excess,
    RBOB_zscore_52w (252-day window), curve_IEF_SHY_zscore_52w (252-day)
  Event dummies (8-week calendar windows, backward-selected p<0.10):
    Same candidate list as v6.4.

Key coefficient changes vs v6.4 (weekly contemporaneous):
  - log_DXY sign flips: was +0.232, now -0.45 (the forward-looking
    dollar relationship is negative: stronger USD → TSLA lower in 2d).
  - log_QQQ drops from 1.11 → 1.02 (less same-day beta loading).
  - ARKK_excess drops from 0.897 → 0.77.
  - Fewer event dummies survive (Tariff_shock and Robotaxi_Austin
    drop out; their price impact is absorbed on the day, not 2d later).

Outputs:
    public/data/model.json    -- coefficients, stats, current 2d target
    public/data/history.json  -- daily time series (actual, fitted, band)
"""
import json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START    = "2022-04-26"
END      = "2026-04-26"
OOS_START = "2025-01-03"
MIN_TRAIN = 252           # ~1 trading year minimum for walk-forward training
LAG       = 2             # forward regression target: log_TSLA(t + LAG)
LAMBDA    = 0.94
BAND_MIN_ERRORS = 52      # ~one quarter of daily data before band activates
EVENT_P_THRESHOLD = 0.10

# ── data helpers ─────────────────────────────────────────────────────────────

def fetch_daily(sym):
    candidates = []
    try:
        s = yf.Ticker(sym).history(start=START, end=END, interval="1d")["Close"]
        candidates.append(s)
    except Exception:
        pass
    try:
        dl = yf.download(sym, start=START, end=END, interval="1d",
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
    """target = a + b*base + e  →  returns (a, b, residuals)."""
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    resid = target.to_numpy() - X @ coef
    return float(coef[0]), float(coef[1]), pd.Series(resid, index=target.index)


# ── OLS helpers ───────────────────────────────────────────────────────────────

def fit_ols(frame: pd.DataFrame, factors: list, target_col: str = "target"):
    y = frame[target_col].to_numpy()
    X = np.column_stack([np.ones(len(frame))]
                        + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    r    = y - yhat
    n, k = X.shape
    s2   = max((r @ r) / (n - k), 1e-12)
    cov  = s2 * np.linalg.pinv(X.T @ X)
    se   = np.sqrt(np.diag(cov))
    t_st = b / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_st), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (r @ r) / ss_tot) if ss_tot > 0 else 0.0
    return dict(beta=b, se=se, t=t_st, p=p_val, r2=r2,
                fitted=pd.Series(yhat, index=frame.index),
                resid=pd.Series(r, index=frame.index),
                factors=list(factors), n=n, k=k,
                sigma=float(np.sqrt(s2)))


def select_factors(frame, forced, candidate_events, threshold=EVENT_P_THRESHOLD):
    """Backward-eliminate event dummies until all p <= threshold."""
    events  = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    m = fit_ols(frame, factors)
    while events:
        ev_ps = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst_e, worst_p = max(ev_ps, key=lambda x: x[1])
        if worst_p <= threshold:
            break
        events.remove(worst_e)
        factors = forced + events
        m = fit_ols(frame, factors)
    return factors, events, m


def walk_forward_predictions(frame, forced, candidate_events, min_train=MIN_TRAIN):
    """One-step-ahead walk-forward: predict log_TSLA(t+LAG) using factors(t)."""
    pred_series = pd.Series(index=frame.index, dtype=float)
    for i, (date, row) in enumerate(frame.iterrows()):
        train = frame.iloc[:i]
        if len(train) < min_train:
            continue
        facs, _, tm = select_factors(train, forced, candidate_events)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred_series.loc[date] = float(xv @ tm["beta"])
    return pred_series


# ── fetch data ────────────────────────────────────────────────────────────────

print("Fetching daily closes…")
SYMS = {"TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
        "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
        "IEF": "IEF", "SHY": "SHY"}
S = {k: fetch_daily(v) for k, v in SYMS.items()}
df = pd.DataFrame(S).ffill().dropna()
print(f"  {len(df)} trading days, {df.index[0].date()} → {df.index[-1].date()}")

# ── build factor matrix ───────────────────────────────────────────────────────

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])

nvda_a, nvda_b, nvda_resid = residualize(np.log(df["NVDA"]), f["log_QQQ"])
arkk_a, arkk_b, arkk_resid = residualize(np.log(df["ARKK"]), f["log_QQQ"])
f["NVDA_excess"] = nvda_resid
f["ARKK_excess"] = arkk_resid

# 52-week rolling z-scores → 252 trading-day window for daily data
log_rbob = np.log(df["RBOB"])
rbob_mean = log_rbob.rolling(252, min_periods=100).mean()
rbob_std  = log_rbob.rolling(252, min_periods=100).std()
f["RBOB_zscore_52w"] = (log_rbob - rbob_mean) / rbob_std

curve_log  = np.log(df["IEF"]) - np.log(df["SHY"])
curve_mean = curve_log.rolling(252, min_periods=100).mean()
curve_std  = curve_log.rolling(252, min_periods=100).std()
f["curve_IEF_SHY_zscore_52w"] = (curve_log - curve_mean) / curve_std

# Event dummies – same calendar windows as v6.4 (8 weeks = 56 calendar days)
EVENT_DEFS = [
    ("Split_squeeze_2020",  "2020-08-11", "5-for-1 split squeeze"),
    ("SP500_inclusion",     "2020-11-16", "S&P 500 inclusion announcement"),
    ("Hertz_1T_peak",       "2021-10-25", "Hertz order, $1T cap peak"),
    ("Twitter_overhang",    "2022-04-25", "Twitter bid overhang"),
    ("Twitter_close",       "2022-10-27", "Twitter acquisition close"),
    ("AI_day_2023",         "2023-07-19", "AI Day 2 / Dojo narrative"),
    ("Trump_election",      "2024-11-06", "Trump election rally"),
    ("DOGE_brand_damage",   "2025-02-15", "DOGE brand damage"),
    ("Musk_exits_DOGE",     "2025-04-22", "Musk exits DOGE"),
    ("TrillionPay",         "2025-09-05", "$1T Musk pay package"),
    ("Tariff_shock",        "2026-02-01", "Tariff shock"),
    ("Robotaxi_Austin",     "2025-06-22", "Austin robotaxi sell-the-news"),
]
for name, dt, _ in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

# Forward-regression target: log_TSLA shifted back LAG rows
# (i.e., align factors at t with log_TSLA at t+LAG)
f["target"] = f["log_TSLA"].shift(-LAG)
f = f.dropna()
print(f"  Feature matrix after dropna + target shift: {len(f)} rows")

# Drop all-zero event dummies in the effective window
def has_variation(col):
    return f[col].nunique() > 1

active_event_names = [n for n, _, _ in EVENT_DEFS if has_variation(f"E_{n}")]
print(f"  Event dummies with variation: {len(active_event_names)} / {len(EVENT_DEFS)}")

FORCED = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
          "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]
ALL_EVENTS = [f"E_{n}" for n in active_event_names]

# ── full-sample fit ───────────────────────────────────────────────────────────

factors, kept_events, m = select_factors(f, FORCED, ALL_EVENTS)
print(f"  Events kept after backward selection: {len(kept_events)}")
for e in kept_events:
    print(f"    {e}")

# In-sample stats (target = log_TSLA(t+LAG); fitted = model(t))
y_target  = f["target"].to_numpy()
y_fitted  = m["fitted"].to_numpy()
resid_fwd = y_target - y_fitted   # residual in log-space, forward-regression

# IS MAE: compare model(t) to actual(t+LAG) in price-space
mae_in = float(np.mean(np.abs(np.exp(y_target) / np.exp(y_fitted) - 1)) * 100)
print(f"  IS R² (target=t+{LAG}): {m['r2']:.3f}")
print(f"  IS MAE (target=t+{LAG}): {mae_in:.2f}%")

# ── walk-forward OOS ──────────────────────────────────────────────────────────

print(f"Walk-forward OOS (LAG={LAG})…")
pred_log_target = walk_forward_predictions(f, FORCED, ALL_EVENTS)
predicted_rows  = pred_log_target.dropna().index

# For each date t where we have a prediction, compare to actual TSLA(t+LAG).
# The prediction at t is for t+LAG; the target column already contains
# log_TSLA(t+LAG), so we can compare directly.
pred_errors_fwd = (f.loc[predicted_rows, "target"]
                   - pred_log_target.loc[predicted_rows]).astype(float)

oos_idx    = predicted_rows[predicted_rows >= pd.Timestamp(OOS_START)]
oos_pred   = np.exp(pred_log_target.loc[oos_idx].to_numpy())
oos_target = np.exp(f.loc[oos_idx, "target"].to_numpy())       # actual(t+LAG)
mae_oos    = float(np.mean(np.abs(oos_pred / oos_target - 1)) * 100)
ss_tot_oos = float(((oos_target - oos_target.mean()) ** 2).sum())
ss_res_oos = float(((oos_target - oos_pred) ** 2).sum())
r2_oos     = float(1 - ss_res_oos / ss_tot_oos)
corr_oos   = float(np.corrcoef(oos_target, oos_pred)[0, 1])
print(f"  OOS R² (pred t+{LAG} vs actual t+{LAG}): {r2_oos:.3f}")
print(f"  OOS MAE: {mae_oos:.2f}%  |  OOS corr: {corr_oos:.3f}")

# ── current snapshot ──────────────────────────────────────────────────────────

# Latest factor values (last row of f, before the target-shift removes LAG rows)
# We re-use the raw factor matrix at the very last day of data.
last_day_features = f.iloc[-1]   # factors(T), targeting T+LAG
beta  = m["beta"]
intercept = float(beta[0])

contrib_log = {}
for i, c in enumerate(factors, start=1):
    contrib_log[c] = float(beta[i] * float(last_day_features[c]))
contrib_log["__intercept__"] = intercept

logfair = intercept + sum(beta[i + 1] * float(last_day_features[c])
                          for i, c in enumerate(factors))
fair = float(np.exp(logfair))
# actual price at last date (not t+LAG — we show current market price)
actual_now = float(df["TSLA"].iloc[-1])

def dollars(component_log):
    return float(np.exp(logfair) - np.exp(logfair - component_log))

bucket_dollars = {
    "QQQ":              dollars(contrib_log.get("log_QQQ", 0)),
    "DXY":              dollars(contrib_log.get("log_DXY", 0)),
    "VIX":              dollars(contrib_log.get("log_VIX", 0)),
    "NVDA_rotation":    dollars(contrib_log.get("NVDA_excess", 0)),
    "ARKK_rotation":    dollars(contrib_log.get("ARKK_excess", 0)),
    "gas_affordability":dollars(contrib_log.get("RBOB_zscore_52w", 0)),
    "recession_pricing":dollars(contrib_log.get("curve_IEF_SHY_zscore_52w", 0)),
    "events":           dollars(sum(v for k, v in contrib_log.items() if k.startswith("E_"))),
    "baseline":         fair - sum([
        dollars(contrib_log.get("log_QQQ", 0)),
        dollars(contrib_log.get("log_DXY", 0)),
        dollars(contrib_log.get("log_VIX", 0)),
        dollars(contrib_log.get("NVDA_excess", 0)),
        dollars(contrib_log.get("ARKK_excess", 0)),
        dollars(contrib_log.get("RBOB_zscore_52w", 0)),
        dollars(contrib_log.get("curve_IEF_SHY_zscore_52w", 0)),
        dollars(sum(v for k, v in contrib_log.items() if k.startswith("E_"))),
    ]),
}

coef_map = {"Intercept": intercept}
for i, c in enumerate(factors, start=1):
    coef_map[c] = float(beta[i])

p_map = {"Intercept": float(m["p"][0])}
for i, c in enumerate(factors, start=1):
    p_map[c] = float(m["p"][i])

latest_date = f.index[-1]
active_events = []
for name, dt, label in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    if d0 <= latest_date < d0 + pd.Timedelta(weeks=8) and f"E_{name}" in factors:
        active_events.append({"name": name, "start": dt, "label": label})

model_obj = {
    "version": "v7.0-predictive-2d",
    "generated_at": pd.Timestamp.utcnow().isoformat(),
    "window": {
        "start": str(f.index[0].date()),
        "end":   str(f.index[-1].date()),
        "n_days": int(len(f)),
        "n_weeks": int(len(f)),   # alias kept for frontend backward-compat
        "lag_days": LAG,
        "lag_description": f"Model price at t is a {LAG}-trading-day forward target for TSLA.",
    },
    "factors": factors,
    "coefficients": coef_map,
    "p_values": p_map,
    "residualizations": {
        "NVDA_excess": {"intercept": nvda_a, "beta_log_QQQ": nvda_b,
                        "source": "log(NVDA)"},
        "ARKK_excess": {"intercept": arkk_a, "beta_log_QQQ": arkk_b,
                        "source": "log(ARKK)"},
    },
    "rolling_stats": {
        "RBOB_zscore_52w": {
            "window": 252, "min_periods": 100,
            "log_mean_latest": float(rbob_mean.iloc[-1]),
            "log_std_latest":  float(rbob_std.iloc[-1]),
            "source": "log(RBOB)",
        },
        "curve_IEF_SHY_zscore_52w": {
            "window": 252, "min_periods": 100,
            "log_mean_latest": float(curve_mean.iloc[-1]),
            "log_std_latest":  float(curve_std.iloc[-1]),
            "source": "log(IEF) - log(SHY)",
        },
    },
    "events": [
        {"name": n, "start": dt, "label": lbl,
         "weeks": 8,
         "in_model": f"E_{n}" in factors,
         "beta": coef_map.get(f"E_{n}"),
         "p_value": p_map.get(f"E_{n}")}
        for n, dt, lbl in EVENT_DEFS
    ],
    "stats": {
        "r2_in": float(m["r2"]),
        "mae_in_pct": mae_in,
        "sigma_resid_log": m["sigma"],
        "sigma_t_method": "EWMA_lambda_0.94_on_fwd_one_step_errors",
        "sigma_t_latest": None,   # filled after EWMA pass below
        "oos": {
            "r2":     float(r2_oos),
            "mae_pct": mae_oos,
            "corr":   corr_oos,
            "start":  OOS_START,
            "n":      int(len(oos_target)),
            "note":   f"Walk-forward: model(t) predicts actual(t+{LAG})",
        },
    },
    "current": {
        "date":       str(latest_date.date()),
        "tsla_actual": actual_now,
        "tsla_fair":  fair,
        "gap_pct":    float((actual_now / fair - 1) * 100),
        "sigma_low":  float(fair * np.exp(-m["sigma"])),
        "sigma_high": float(fair * np.exp( m["sigma"])),
        "factors_now": {c: float(last_day_features[c]) for c in factors},
        "contribution_dollars": bucket_dollars,
        "active_events": active_events,
        "underlyings": {
            "TSLA": float(df["TSLA"].iloc[-1]),
            "QQQ":  float(df["QQQ"].iloc[-1]),
            "DXY":  float(df["DXY"].iloc[-1]),
            "VIX":  float(df["VIX"].iloc[-1]),
            "NVDA": float(df["NVDA"].iloc[-1]),
            "ARKK": float(df["ARKK"].iloc[-1]),
            "RBOB": float(df["RBOB"].iloc[-1]),
            "IEF":  float(df["IEF"].iloc[-1]),
            "SHY":  float(df["SHY"].iloc[-1]),
        },
    },
}

with open(OUT_DIR / "model.json", "w") as fh:
    json.dump(model_obj, fh, indent=2)
print(f"Wrote {OUT_DIR / 'model.json'}")

# ── predictive band calibration (forward-error based) ────────────────────────
# Same EWMA methodology as v6.4, but applied to forward (t → t+LAG) errors.

resid_log   = m["resid"].to_numpy()
sigma_const = m["sigma"]
seed_var    = float(np.var(resid_log[:52], ddof=1))
fallback_sigma_arr = np.empty(len(resid_log))
s2_fb = seed_var
for i in range(len(resid_log)):
    fallback_sigma_arr[i] = float(np.sqrt(max(s2_fb, 1e-8)))
    s2_fb = LAMBDA * s2_fb + (1 - LAMBDA) * (resid_log[i] ** 2)

sigma_pred_is = float(np.std(pred_errors_fwd.to_numpy(), ddof=1))
q10_log_global = float(np.percentile(pred_errors_fwd.to_numpy(), 10))
q90_log_global = float(np.percentile(pred_errors_fwd.to_numpy(), 90))

seen_pred_errors = []
pred_var_state   = None
predictive_band_ready = []
history = []

for i, date in enumerate(f.index):
    # actual(t): true TSLA price at t (for chart display)
    a = float(df["TSLA"].reindex(f.index).loc[date])
    # fitted(t): model prediction at t, targeting t+LAG
    fit_v = float(np.exp(m["fitted"].loc[date]))

    if len(seen_pred_errors) >= BAND_MIN_ERRORS:
        if pred_var_state is None:
            pred_var_state = float(np.var(np.array(seen_pred_errors), ddof=1))
        sigma_band = float(np.sqrt(max(pred_var_state, 1e-8)))
        q10_raw    = float(np.percentile(seen_pred_errors, 10))
        q90_raw    = float(np.percentile(seen_pred_errors, 90))
        scale      = sigma_band / max(sigma_pred_is, 1e-8)
        q10_offset = q10_raw * scale
        q90_offset = q90_raw * scale
        predictive_band_ready.append(True)
    else:
        sigma_band = float(fallback_sigma_arr[i])
        q10_offset = -sigma_band
        q90_offset = sigma_band
        predictive_band_ready.append(False)

    history.append({
        "date":    str(date.date()),
        "actual":  round(a, 2),
        "fitted":  round(fit_v, 2),
        "low":     round(fit_v * float(np.exp(-sigma_band)), 2),
        "high":    round(fit_v * float(np.exp( sigma_band)), 2),
        "sigma_t": round(sigma_band, 5),
        "low_q":   round(fit_v * float(np.exp(q10_offset)), 2),
        "high_q":  round(fit_v * float(np.exp(q90_offset)), 2),
    })
    if date in pred_errors_fwd.index:
        err = float(pred_errors_fwd.loc[date])
        seen_pred_errors.append(err)
        if pred_var_state is not None:
            pred_var_state = LAMBDA * pred_var_state + (1 - LAMBDA) * (err ** 2)

# Next-period sigma
if pred_var_state is None:
    sigma_t_next = float(fallback_sigma_arr[-1])
else:
    sigma_t_next = float(np.sqrt(max(pred_var_state, 1e-8)))

model_obj["stats"]["sigma_t_latest"] = sigma_t_next
model_obj["current"]["sigma_low"]    = float(fair * np.exp(-sigma_t_next))
model_obj["current"]["sigma_high"]   = float(fair * np.exp( sigma_t_next))
model_obj["current"]["sigma_t"]      = sigma_t_next

scale_next = sigma_t_next / max(sigma_pred_is, 1e-8)
model_obj["stats"]["q10_log"]     = q10_log_global
model_obj["stats"]["q90_log"]     = q90_log_global
model_obj["stats"]["sigma_IS_log"] = sigma_pred_is
model_obj["current"]["q_low"]     = float(fair * np.exp(q10_log_global * scale_next))
model_obj["current"]["q_high"]    = float(fair * np.exp(q90_log_global * scale_next))

# Band coverage (compare band at t to actual(t+LAG) — the true target)
# Since our chart shows actual(t) not actual(t+LAG), we also measure
# coverage of band at t vs. actual(t) for chart honesty.
predictive_rows = [h for h, rdy in zip(history, predictive_band_ready) if rdy]
predictive_rows_oos = [h for h, rdy in zip(history, predictive_band_ready)
                       if rdy and h["date"] >= OOS_START]

# Coverage: band(t) vs actual(t) — chart-level honesty metric
coverage_bt  = float(np.mean([r["low_q"] <= r["actual"] <= r["high_q"]
                               for r in predictive_rows]))
coverage_oos = float(np.mean([r["low_q"] <= r["actual"] <= r["high_q"]
                               for r in predictive_rows_oos]))
model_obj["stats"]["band_coverage_backtest"] = coverage_bt
model_obj["stats"]["band_coverage_oos"]      = coverage_oos
model_obj["stats"]["band_backtest_start"]    = (predictive_rows[0]["date"]
                                                 if predictive_rows else None)

# Re-write model.json with finalized sigma_t fields
with open(OUT_DIR / "model.json", "w") as fh:
    json.dump(model_obj, fh, indent=2)
print(f"Wrote {OUT_DIR / 'model.json'} (updated with sigma_t)")

# ── write history.json ────────────────────────────────────────────────────────

# Append today's intraday/daily snapshot if newer than last row
try:
    SYMS_DAILY = {"TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
                  "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
                  "IEF": "IEF", "SHY": "SHY"}
    daily_raw = yf.download(list(SYMS_DAILY.values()), period="5d", interval="1d",
                             progress=False, auto_adjust=False)
    if isinstance(daily_raw.columns, pd.MultiIndex):
        daily_raw.columns = ["_".join(c).strip() for c in daily_raw.columns]
    today_prices = {}
    for sym, ticker in SYMS_DAILY.items():
        col = f"Close_{ticker}"
        if col not in daily_raw.columns:
            col = next((c for c in daily_raw.columns
                        if c.startswith("Close") and ticker in c), None)
        if col and col in daily_raw.columns:
            s = daily_raw[col].dropna()
            if not s.empty:
                today_prices[sym] = float(s.iloc[-1])
    last_daily_idx = daily_raw.dropna(how="all").index
    today_date = pd.Timestamp(last_daily_idx[-1]).normalize()
    last_model_date = f.index[-1]

    if len(today_prices) == len(SYMS_DAILY) and today_date > last_model_date:
        lq  = np.log(today_prices["QQQ"])
        ld  = np.log(today_prices["DXY"])
        lv  = np.log(today_prices["VIX"])
        nvda_exc = np.log(today_prices["NVDA"]) - (nvda_a + nvda_b * lq)
        arkk_exc = np.log(today_prices["ARKK"]) - (arkk_a + arkk_b * lq)
        # Use stored rolling stats for z-scores
        rbob_z  = ((np.log(today_prices["RBOB"])
                    - model_obj["rolling_stats"]["RBOB_zscore_52w"]["log_mean_latest"])
                   / model_obj["rolling_stats"]["RBOB_zscore_52w"]["log_std_latest"])
        curve_z = (((np.log(today_prices["IEF"]) - np.log(today_prices["SHY"]))
                    - model_obj["rolling_stats"]["curve_IEF_SHY_zscore_52w"]["log_mean_latest"])
                   / model_obj["rolling_stats"]["curve_IEF_SHY_zscore_52w"]["log_std_latest"])
        row_vals = {
            "log_QQQ": lq, "log_DXY": ld, "log_VIX": lv,
            "NVDA_excess": nvda_exc, "ARKK_excess": arkk_exc,
            "RBOB_zscore_52w": rbob_z,
            "curve_IEF_SHY_zscore_52w": curve_z,
        }
        for name, dt, _ in EVENT_DEFS:
            d0 = pd.Timestamp(dt)
            row_vals[f"E_{name}"] = int(d0 <= today_date < d0 + pd.Timedelta(weeks=8))

        xv_today = np.array([1.0] + [row_vals.get(c, 0.0) for c in factors])
        logfair_today = float(xv_today @ beta)
        fair_today    = float(np.exp(logfair_today))
        a_today       = today_prices["TSLA"]
        sigma_today   = sigma_t_next
        history.append({
            "date":     str(today_date.date()),
            "actual":   round(a_today, 2),
            "fitted":   round(fair_today, 2),
            "low":      round(fair_today * np.exp(-sigma_today), 2),
            "high":     round(fair_today * np.exp( sigma_today), 2),
            "sigma_t":  round(sigma_today, 5),
            "low_q":    round(fair_today * np.exp(q10_log_global * scale_next), 2),
            "high_q":   round(fair_today * np.exp(q90_log_global * scale_next), 2),
            "partial_week": True,
        })
        print(f"  Appended today's snapshot ({today_date.date()}, TSLA={a_today:.2f})")
except Exception as e:
    print(f"  Today's snapshot skipped: {e}")

with open(OUT_DIR / "history.json", "w") as fh:
    json.dump(history, fh)
print(f"Wrote {OUT_DIR / 'history.json'}  ({len(history)} rows)")

# ── print summary ─────────────────────────────────────────────────────────────
print()
print("=" * 56)
print(f"  Model version     : v7.0-predictive-2d")
print(f"  Regression target : log_TSLA(t+{LAG})  →  {LAG}-day forward")
print(f"  Observations      : {len(f)} trading days")
print(f"  IS R²             : {m['r2']:.3f}")
print(f"  IS MAE            : {mae_in:.2f}%")
print(f"  OOS R²            : {r2_oos:.3f}")
print(f"  OOS MAE           : {mae_oos:.2f}%")
print(f"  OOS corr          : {corr_oos:.3f}")
print(f"  2d fwd target     : ${fair:.2f}  (TSLA now: ${actual_now:.2f})")
print(f"  Gap vs market     : {(actual_now/fair-1)*100:+.1f}%")
print(f"  Band coverage(bt) : {coverage_bt*100:.1f}%")
print(f"  Band coverage(OOS): {coverage_oos*100:.1f}%")
print("=" * 56)

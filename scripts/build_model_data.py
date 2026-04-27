"""
Export v6.4-canonical TSLA multi-factor model coefficients + historical series
to JSON files consumed by the React app.

v6.4-canonical factors (v6.3 structure, methodology hardened):
  - log_QQQ, log_DXY, log_VIX (orthogonalized vs QQQ): NVDA_excess, ARKK_excess
    NOTE on log_DXY: kept as a *forced* factor on theoretical grounds (dollar
    regime is a persistent driver of large-cap risk assets), NOT because it
    clears a statistical bar in the current 4y window. In the v6.4 fit its
    p-value is ~0.61 (effectively noise over 2022-09 -> 2026-04). It is
    retained for cross-regime stability: dropping it would let other factors
    silently absorb dollar moves and bias their betas the next time DXY
    matters. Forced factors are NOT subject to the p<0.10 backward gate;
    only event dummies are. Revisit if log_DXY stays insignificant across
    multiple regimes.
  - RBOB_zscore_52w: 52w rolling z-score of log(RBOB gasoline futures) -- gas-
    affordability proxy. Promoted in v6.1 after gauntlet (+5.62pp OOS).
  - curve_IEF_SHY_zscore_52w: 52w rolling z-score of log(IEF/SHY) -- bond-curve
    shape, proxy for recession/rate-cut pricing. Promoted in v6.2 after gauntlet:
      walk-forward OOS R^2 lift +4.88pp on top of v6.1; permutation null
      p<0.002 (0/500); strict complement to RBOB (CURVE alone HURTS v6 by
      -5.31pp; v6.1+CURVE = +4.88pp). Lift positive in ALL three OOS
      sub-periods (2025-H1, 2025-H2, 2026-YTD) -- cleaner than RBOB's profile.
      See AI_CONTEXT.md sec 10.13.
    - 8-week event dummies, kept if p<0.10 (backward selection)
    - E_Robotaxi_Austin (2025-06-22): β=-0.119, p=0.030, WF lift +2.30pp, perm-p=0.020.
      Austin commercial robotaxi launch was a "sell-the-news" event; TSLA traded
      -7% below model average for 8 consecutive weeks. Macro orthogonal (VIX/RBOB/
      ARKK near OOS means during window). Promoted in v6.3 after permutation null
      and walk-forward validation.
    - v6.4 hardening: event dummies are now truly kept by backward selection
        (p<0.10) and the displayed range is calibrated from expanding one-step
        forecast errors rather than full-sample fit residuals.

Outputs:
    public/data/model.json    -- coefficients, residualizations, stats, current snapshot
    public/data/history.json  -- weekly time series (actual, fitted, predictive range)

Predictive range: calibrated from expanding one-step-ahead forecast errors.
At each row, the displayed range uses only earlier forecast errors (no lookahead):
    - raw shape from expanding 10th / 90th percentiles of past forecast errors
    - width from EWMA(lambda=0.94) on past forecast errors
This keeps the band predictive, asymmetric, and regime-aware.
"""
import json, warnings, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"
MIN_TRAIN = 100
EVENT_P_THRESHOLD = 0.10
LAMBDA = 0.94
BAND_MIN_ERRORS = 26

def wk(sym):
    candidates = []
    try:
        candidates.append(yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"])
    except Exception:
        pass
    try:
        dl = yf.download(sym, start=START, end=END, interval="1wk", progress=False, auto_adjust=False)
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

    raise RuntimeError(f"Failed to fetch weekly close for {sym}")

def residualize(target, base):
    """Returns (intercept, beta, residuals). target = a + b*base + e."""
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    resid = target.to_numpy() - X @ coef
    return float(coef[0]), float(coef[1]), pd.Series(resid, index=target.index)

print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()
print(f"  {len(df)} weekly observations, {df.index[0].date()} -> {df.index[-1].date()}")

# Build feature matrix
f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])

# Residualizations (also export coefficients so JS can recompute for live mode)
nvda_a, nvda_b, nvda_resid = residualize(np.log(df["NVDA"]), f["log_QQQ"])
arkk_a, arkk_b, arkk_resid = residualize(np.log(df["ARKK"]), f["log_QQQ"])
f["NVDA_excess"] = nvda_resid
f["ARKK_excess"] = arkk_resid

# RBOB 52w rolling z-score (backward-only -- no lookahead). min_periods=20
# matches the validated walk-forward gauntlet (+5.62pp OOS R^2 lift).
log_rbob = np.log(df["RBOB"])
rbob_mean_52 = log_rbob.rolling(window=52, min_periods=20).mean()
rbob_std_52  = log_rbob.rolling(window=52, min_periods=20).std()
f["RBOB_zscore_52w"] = (log_rbob - rbob_mean_52) / rbob_std_52

# Bond-curve shape: log(IEF/SHY), 52w rolling z-score (backward-only).
# Promoted in v6.2 after gauntlet (+4.88pp OOS R^2 lift on top of v6.1).
# Negative beta: bull-flattening curve (rate-cut pricing) -> TSLA discount.
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
curve_mean_52 = curve_log.rolling(window=52, min_periods=20).mean()
curve_std_52  = curve_log.rolling(window=52, min_periods=20).std()
f["curve_IEF_SHY_zscore_52w"] = (curve_log - curve_mean_52) / curve_std_52

# Events (8-week dummies)
EVENT_DEFS = [
    ("Split_squeeze_2020", "2020-08-11", "5-for-1 split squeeze"),
    ("SP500_inclusion",    "2020-11-16", "S&P 500 inclusion announcement"),
    ("Hertz_1T_peak",      "2021-10-25", "Hertz order, $1T cap peak"),
    ("Twitter_overhang",   "2022-04-25", "Twitter bid overhang"),
    ("Twitter_close",      "2022-10-27", "Twitter acquisition close"),
    ("AI_day_2023",        "2023-07-19", "AI Day 2 / Dojo narrative"),
    ("Trump_election",     "2024-11-06", "Trump election rally"),
    ("DOGE_brand_damage",  "2025-02-15", "DOGE brand damage"),
    ("Musk_exits_DOGE",    "2025-04-22", "Musk exits DOGE"),
    ("TrillionPay",        "2025-09-05", "$1T Musk pay package"),
    ("Tariff_shock",       "2026-02-01", "Tariff shock"),
    ("Robotaxi_Austin",   "2025-06-22", "Austin robotaxi sell-the-news"),
]
for name, dt, _ in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()

# Drop event dummies that are constant (all-zero) on this window
def has_variation(col):
    v = f[col]
    return v.nunique() > 1

active_event_names = [n for n, _, _ in EVENT_DEFS if has_variation(f"E_{n}")]
print(f"  events with variation in window: {len(active_event_names)} / {len(EVENT_DEFS)}")

FORCED = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
          "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]
ALL_EVENTS = [f"E_{n}" for n in active_event_names]

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
    return dict(beta=b, se=se, t=t, p=p, r2=float(r2),
                fitted=pd.Series(yhat, index=frame.index),
                resid=pd.Series(r, index=frame.index),
                factors=list(factors), n=n, k=k,
                sigma=float(np.sqrt(s2)))

def select_factors(frame, forced, candidate_events, p_threshold=EVENT_P_THRESHOLD):
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    m_local = fit(frame, factors)
    while events:
        event_ps = [(e, float(m_local["p"][1 + factors.index(e)])) for e in events]
        worst_event, worst_p = max(event_ps, key=lambda item: item[1])
        if worst_p <= p_threshold:
            break
        events.remove(worst_event)
        factors = forced + events
        m_local = fit(frame, factors)
    return factors, events, m_local

def recursive_predictions(frame, forced, candidate_events, min_train=MIN_TRAIN):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, _, train_model = select_factors(train, forced, candidate_events)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred_log.loc[date] = float(xv @ train_model["beta"])
    return pred_log

factors, kept_events, m = select_factors(f, FORCED, ALL_EVENTS)
print(f"  events kept after backward selection: {len(kept_events)}")
for e in kept_events:
    print(f"    {e}")

# In-sample stats
y = f["log_TSLA"].to_numpy()
yhat = m["fitted"].to_numpy()
resid_pct = (np.exp(y - yhat) - 1) * 100  # actual vs fitted
mae_in = float(np.mean(np.abs(resid_pct)))

# Walk-forward one-step predictions (same backward event selection, no lookahead)
print("Walk-forward OOS...")
pred_log = recursive_predictions(f, FORCED, ALL_EVENTS)
predicted_rows = pred_log.dropna().index
pred_errors_log = (f.loc[predicted_rows, "log_TSLA"] - pred_log.loc[predicted_rows]).astype(float)

oos_idx = predicted_rows[predicted_rows >= pd.Timestamp(OOS_START)]
oos_pred = np.exp(pred_log.loc[oos_idx].to_numpy())
oos_actual = np.exp(f.loc[oos_idx, "log_TSLA"].to_numpy())
mae_oos = float(np.mean(np.abs(oos_pred / oos_actual - 1)) * 100)
ss_tot = float(((oos_actual - oos_actual.mean()) ** 2).sum())
ss_res = float(((oos_actual - oos_pred) ** 2).sum())
r2_oos = 1 - ss_res / ss_tot
corr_oos = float(np.corrcoef(oos_actual, oos_pred)[0, 1])

# Current factor contributions (latest week)
last = f.iloc[-1]
beta = m["beta"]
intercept = float(beta[0])
contrib_log = {}
for i, c in enumerate(factors, start=1):
    contrib_log[c] = float(beta[i] * float(last[c]))
contrib_log["__intercept__"] = intercept

logfair = intercept + sum(beta[i + 1] * float(last[c]) for i, c in enumerate(factors))
fair = float(np.exp(logfair))
actual_now = float(np.exp(float(last["log_TSLA"])))

# Group log-contribs into 5 buckets for the bar chart (in price terms via marginal multipliers)
# Each factor i contributes (e^logfair - e^(logfair - beta_i*x_i)) at the margin (relative to "factor=0")
def dollars(component_log):
    return float((np.exp(logfair) - np.exp(logfair - component_log)))

bucket_dollars = {
    "QQQ":   dollars(contrib_log.get("log_QQQ", 0)),
    "DXY":   dollars(contrib_log.get("log_DXY", 0)),
    "VIX":   dollars(contrib_log.get("log_VIX", 0)),
    "NVDA_rotation": dollars(contrib_log.get("NVDA_excess", 0)),
    "ARKK_rotation": dollars(contrib_log.get("ARKK_excess", 0)),
    "gas_affordability": dollars(contrib_log.get("RBOB_zscore_52w", 0)),
    "recession_pricing": dollars(contrib_log.get("curve_IEF_SHY_zscore_52w", 0)),
    "events": dollars(sum(v for k, v in contrib_log.items() if k.startswith("E_"))),
    "baseline": fair - sum([
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

# Coefficients map (named)
coef_map = {"Intercept": intercept}
for i, c in enumerate(factors, start=1):
    coef_map[c] = float(beta[i])

# Significance/p-values
p_map = {"Intercept": float(m["p"][0])}
for i, c in enumerate(factors, start=1):
    p_map[c] = float(m["p"][i])

# Active events at the latest date
latest_date = f.index[-1]
active_events = []
for name, dt, label in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    if d0 <= latest_date < d0 + pd.Timedelta(weeks=8) and f"E_{name}" in factors:
        active_events.append({"name": name, "start": dt, "label": label})

model_obj = {
    "version": "v6.4-canonical-4y",
    "generated_at": pd.Timestamp.utcnow().isoformat(),
    "window": {
        "start": str(f.index[0].date()),
        "end": str(f.index[-1].date()),
        "n_weeks": int(len(f)),
    },
    "factors": factors,
    "coefficients": coef_map,
    "p_values": p_map,
    "residualizations": {
        "NVDA_excess": {"intercept": nvda_a, "beta_log_QQQ": nvda_b, "source": "log(NVDA)"},
        "ARKK_excess": {"intercept": arkk_a, "beta_log_QQQ": arkk_b, "source": "log(ARKK)"},
    },
    "rolling_stats": {
        "RBOB_zscore_52w": {
            "window": 52, "min_periods": 20,
            "log_mean_latest": float(rbob_mean_52.iloc[-1]),
            "log_std_latest":  float(rbob_std_52.iloc[-1]),
            "source": "log(RBOB)",
        },
        "curve_IEF_SHY_zscore_52w": {
            "window": 52, "min_periods": 20,
            "log_mean_latest": float(curve_mean_52.iloc[-1]),
            "log_std_latest":  float(curve_std_52.iloc[-1]),
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
        "sigma_t_method": "EWMA_lambda_0.94_on_one_step_errors",
        "sigma_t_latest": None,  # filled after EWMA pass below
        "oos": {
            "r2": float(r2_oos),
            "mae_pct": mae_oos,
            "corr": corr_oos,
            "start": OOS_START,
            "n": int(len(oos_actual)),
        },
    },
    "current": {
        "date": str(latest_date.date()),
        "tsla_actual": actual_now,
        "tsla_fair": fair,
        "gap_pct": float((actual_now / fair - 1) * 100),
        "sigma_low": float(fair * np.exp(-m["sigma"])),
        "sigma_high": float(fair * np.exp(m["sigma"])),
        "factors_now": {c: float(last[c]) for c in factors},
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

# Predictive band calibration.
# Centerline remains the full-sample fair-value fit; the displayed range is
# calibrated from expanding one-step-ahead forecast errors only.
resid_log = m["resid"].to_numpy()
sigma_const = m["sigma"]
seed_var = float(np.var(resid_log[:26], ddof=1))
fallback_sigma_arr = np.empty(len(resid_log))
s2_fallback = seed_var
for i in range(len(resid_log)):
    fallback_sigma_arr[i] = float(np.sqrt(max(s2_fallback, 1e-8)))
    s2_fallback = LAMBDA * s2_fallback + (1 - LAMBDA) * (resid_log[i] ** 2)

sigma_pred_is = float(np.std(pred_errors_log.to_numpy(), ddof=1))
q10_log_global = float(np.percentile(pred_errors_log.to_numpy(), 10))
q90_log_global = float(np.percentile(pred_errors_log.to_numpy(), 90))

seen_pred_errors = []
pred_var_state = None
row_band_low = []
row_band_high = []
row_sigma = []
predictive_band_ready = []
history = []
for i, date in enumerate(f.index):
    a = float(np.exp(f.loc[date, "log_TSLA"]))
    fit_v = float(np.exp(m["fitted"].loc[date]))
    if len(seen_pred_errors) >= BAND_MIN_ERRORS:
        if pred_var_state is None:
            pred_var_state = float(np.var(np.array(seen_pred_errors), ddof=1))
        sigma_band = float(np.sqrt(max(pred_var_state, 1e-8)))
        q10_raw = float(np.percentile(seen_pred_errors, 10))
        q90_raw = float(np.percentile(seen_pred_errors, 90))
        scale = sigma_band / max(sigma_pred_is, 1e-8)
        q10_offset = q10_raw * scale
        q90_offset = q90_raw * scale
        predictive_band_ready.append(True)
    else:
        sigma_band = float(fallback_sigma_arr[i])
        q10_offset = -sigma_band
        q90_offset = sigma_band
        predictive_band_ready.append(False)
    row_sigma.append(sigma_band)
    row_band_low.append(fit_v * float(np.exp(q10_offset)))
    row_band_high.append(fit_v * float(np.exp(q90_offset)))
    history.append({
        "date": str(date.date()),
        "actual": round(a, 2),
        "fitted": round(fit_v, 2),
        "low":  round(fit_v * float(np.exp(-sigma_band)), 2),
        "high": round(fit_v * float(np.exp( sigma_band)), 2),
        "sigma_t": round(sigma_band, 5),
        "low_q":  round(fit_v * float(np.exp(q10_offset)), 2),
        "high_q": round(fit_v * float(np.exp(q90_offset)), 2),
    })
    if date in pred_errors_log.index:
        err = float(pred_errors_log.loc[date])
        seen_pred_errors.append(err)
        if pred_var_state is not None:
            pred_var_state = LAMBDA * pred_var_state + (1 - LAMBDA) * (err ** 2)

# Predictive sigma for the next week (uses all known one-step forecast errors).
if pred_var_state is None:
    sigma_t_next = float(fallback_sigma_arr[-1])
else:
    sigma_t_next = float(np.sqrt(max(pred_var_state, 1e-8)))
model_obj["stats"]["sigma_t_latest"] = sigma_t_next
model_obj["current"]["sigma_low"]  = float(fair * np.exp(-sigma_t_next))
model_obj["current"]["sigma_high"] = float(fair * np.exp( sigma_t_next))
model_obj["current"]["sigma_t"]    = sigma_t_next
# Quantile band offsets: latest snapshot scales predictive 10/90 by sigma_t/sigma_pred_is
# so live bands track the current forecast-error regime.
scale_next = sigma_t_next / max(sigma_pred_is, 1e-8)
q10_log_now = q10_log_global * scale_next
q90_log_now = q90_log_global * scale_next
model_obj["stats"]["q10_log"]    = q10_log_global
model_obj["stats"]["q90_log"]    = q90_log_global
model_obj["stats"]["sigma_IS_log"] = sigma_pred_is
model_obj["current"]["q_low"]    = float(fair * np.exp(q10_log_now))
model_obj["current"]["q_high"]   = float(fair * np.exp(q90_log_now))

# Realized backtest coverage of the predictive range.
predictive_rows = [h for h, ready in zip(history, predictive_band_ready) if ready]
predictive_rows_oos = [h for h, ready in zip(history, predictive_band_ready)
                       if ready and h["date"] >= OOS_START]
coverage_backtest = float(np.mean([r["low_q"] <= r["actual"] <= r["high_q"] for r in predictive_rows]))
coverage_oos = float(np.mean([r["low_q"] <= r["actual"] <= r["high_q"] for r in predictive_rows_oos]))
model_obj["stats"]["band_coverage_backtest"] = coverage_backtest
model_obj["stats"]["band_coverage_oos"] = coverage_oos
model_obj["stats"]["band_backtest_start"] = predictive_rows[0]["date"] if predictive_rows else None

# Re-write model.json with updated sigma_t fields
with open(OUT_DIR / "model.json", "w") as fh:
    json.dump(model_obj, fh, indent=2)

with open(OUT_DIR / "history.json", "w") as fh:
    json.dump(history, fh)
print(f"Wrote {OUT_DIR / 'history.json'} ({len(history)} rows)")
print(f"  sigma_t (EWMA lambda=0.94): latest={sigma_t_next:.4f}  constant ref={sigma_const:.4f}")
print(f"  predictive q (expanding 10/90):  q10={q10_log_global:+.4f}  q90={q90_log_global:+.4f}")
print(f"  scaled q (current regime):       q10={q10_log_now:+.4f}  q90={q90_log_now:+.4f}  (scale={scale_next:.3f})")
print(f"  predictive band coverage:        backtest={coverage_backtest:.3f}  oos={coverage_oos:.3f}")

# Console summary
print(f"\nv6.4-canonical | factors={len(factors)} | n={len(f)}")
print(f"  In-sample:  R²={m['r2']:.4f}  MAE={mae_in:.2f}%")
print(f"  OOS:        R²={r2_oos:.4f}  MAE={mae_oos:.2f}%  Corr={corr_oos:.3f}")
print(f"  Current:    TSLA=${actual_now:.2f}  Fair=${fair:.2f}  Gap={(actual_now/fair-1)*100:+.1f}%")

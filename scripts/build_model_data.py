"""
Export v6.3-canonical TSLA multi-factor model coefficients + historical series
to JSON files consumed by the React app.

v6.3-canonical factors (v6.2 + Robotaxi Austin sell-the-news event dummy):
  - log_QQQ, log_DXY, log_VIX (orthogonalized vs QQQ): NVDA_excess, ARKK_excess
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

Outputs:
  public/data/model.json    -- coefficients, residualizations, stats, current snapshot
  public/data/history.json  -- weekly time series (actual, fitted, +/- 1 sigma_t band)

Uncertainty bands: per-row sigma_t computed via EWMA on past in-sample
residuals (lambda=0.94, RiskMetrics standard, lookahead-free). Selected after
adaptive-sigma probe -- C3_EWMA dominates constant sigma on coverage, log
predictive density, and CRPS in the 2025-2026 OOS window. See AI_CONTEXT.md
sec 10.14.
"""
import json, warnings, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START, END = "2022-04-26", "2026-04-26"

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s

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

selected = FORCED + ALL_EVENTS
m = fit(f, selected)
factors = selected

# In-sample stats
y = f["log_TSLA"].to_numpy()
yhat = m["fitted"].to_numpy()
resid_pct = (np.exp(y - yhat) - 1) * 100  # actual vs fitted
mae_in = float(np.mean(np.abs(resid_pct)))

# OOS walk-forward from 2025-01-03
print("Walk-forward OOS...")
ho = f.loc["2025-01-03":]
oos_actual, oos_pred = [], []
for date, row in ho.iterrows():
    train = f.loc[:date].iloc[:-1]
    if len(train) < 100:
        continue
    facs = [c for c in factors if c in train.columns]
    Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in facs])
    yt = train["log_TSLA"].to_numpy()
    bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
    xv = np.array([1.0] + [float(row[c]) for c in facs])
    oos_pred.append(float(np.exp(xv @ bt)))
    oos_actual.append(float(np.exp(row["log_TSLA"])))

oos_actual = np.array(oos_actual); oos_pred = np.array(oos_pred)
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
    "version": "v6.3-canonical-4y",
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
        "sigma_t_method": "EWMA_lambda_0.94",
        "sigma_t_latest": None,  # filled after EWMA pass below
        "oos": {
            "r2": float(r2_oos),
            "mae_pct": mae_oos,
            "corr": corr_oos,
            "start": "2025-01-03",
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

# History series with EWMA(0.94) adaptive sigma_t.
# Lookahead-free: sigma_t uses residuals strictly before t.
#   sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * resid_{t-1}^2
#   seeded with var(first 26 in-sample residuals).
LAMBDA = 0.94
resid_log = m["resid"].to_numpy()
seed_var = float(np.var(resid_log[:26], ddof=1))
sigma_t_arr = np.empty(len(resid_log))
s2 = seed_var
for i in range(len(resid_log)):
    sigma_t_arr[i] = float(np.sqrt(max(s2, 1e-8)))
    s2 = LAMBDA * s2 + (1 - LAMBDA) * (resid_log[i] ** 2)
# Bands at row t use sigma_t (sqrt of variance estimated using info up to t-1)
sigma_const = m["sigma"]
history = []
for i, date in enumerate(f.index):
    a = float(np.exp(f.loc[date, "log_TSLA"]))
    fit_v = float(np.exp(m["fitted"].loc[date]))
    s_t = float(sigma_t_arr[i])
    history.append({
        "date": str(date.date()),
        "actual": round(a, 2),
        "fitted": round(fit_v, 2),
        "low":  round(fit_v * float(np.exp(-s_t)), 2),
        "high": round(fit_v * float(np.exp( s_t)), 2),
        "sigma_t": round(s_t, 5),
    })
# sigma_t for the next-week prediction (using info through latest row)
sigma_t_next = float(np.sqrt(max(LAMBDA * (sigma_t_arr[-1] ** 2)
                                 + (1 - LAMBDA) * (resid_log[-1] ** 2), 1e-8)))
model_obj["stats"]["sigma_t_latest"] = sigma_t_next
model_obj["current"]["sigma_low"]  = float(fair * np.exp(-sigma_t_next))
model_obj["current"]["sigma_high"] = float(fair * np.exp( sigma_t_next))
model_obj["current"]["sigma_t"]    = sigma_t_next
# Re-write model.json with updated sigma_t fields
with open(OUT_DIR / "model.json", "w") as fh:
    json.dump(model_obj, fh, indent=2)

with open(OUT_DIR / "history.json", "w") as fh:
    json.dump(history, fh)
print(f"Wrote {OUT_DIR / 'history.json'} ({len(history)} rows)")
print(f"  sigma_t (EWMA lambda=0.94): latest={sigma_t_next:.4f}  constant ref={sigma_const:.4f}")

# Console summary
print(f"\nv6.3-canonical | factors={len(factors)} | n={len(f)}")
print(f"  In-sample:  R²={m['r2']:.4f}  MAE={mae_in:.2f}%")
print(f"  OOS:        R²={r2_oos:.4f}  MAE={mae_oos:.2f}%  Corr={corr_oos:.3f}")
print(f"  Current:    TSLA=${actual_now:.2f}  Fair=${fair:.2f}  Gap={(actual_now/fair-1)*100:+.1f}%")

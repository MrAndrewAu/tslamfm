"""
Export v6-canonical TSLA multi-factor model coefficients + historical series
to JSON files consumed by the React app.

v6-canonical factors:
  - log_QQQ, log_DXY, log_VIX (orthogonalized vs QQQ): NVDA_excess, ARKK_excess
  - 8-week event dummies, kept if p<0.10 (backward selection)

Outputs:
  public/data/model.json    -- coefficients, residualizations, stats, current snapshot
  public/data/history.json  -- weekly time series (actual, fitted, +/- 1 sigma band)
"""
import json, warnings, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START, END = "2020-04-26", "2026-04-26"

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
    "NVDA": "NVDA", "ARKK": "ARKK"
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
]
for name, dt, _ in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()

FORCED = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess"]
ALL_EVENTS = [f"E_{n}" for n, _, _ in EVENT_DEFS]

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
    "events": dollars(sum(v for k, v in contrib_log.items() if k.startswith("E_"))),
    "baseline": fair - sum([
        dollars(contrib_log.get("log_QQQ", 0)),
        dollars(contrib_log.get("log_DXY", 0)),
        dollars(contrib_log.get("log_VIX", 0)),
        dollars(contrib_log.get("NVDA_excess", 0)),
        dollars(contrib_log.get("ARKK_excess", 0)),
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
    "version": "v6-canonical",
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
        },
    },
}

with open(OUT_DIR / "model.json", "w") as fh:
    json.dump(model_obj, fh, indent=2)
print(f"Wrote {OUT_DIR / 'model.json'}")

# History series
sigma = m["sigma"]
history = []
for date in f.index:
    a = float(np.exp(f.loc[date, "log_TSLA"]))
    fit_v = float(np.exp(m["fitted"].loc[date]))
    history.append({
        "date": str(date.date()),
        "actual": round(a, 2),
        "fitted": round(fit_v, 2),
        "low":  round(fit_v * float(np.exp(-sigma)), 2),
        "high": round(fit_v * float(np.exp( sigma)), 2),
    })
with open(OUT_DIR / "history.json", "w") as fh:
    json.dump(history, fh)
print(f"Wrote {OUT_DIR / 'history.json'} ({len(history)} rows)")

# Console summary
print(f"\nv6-canonical | factors={len(factors)} | n={len(f)}")
print(f"  In-sample:  R²={m['r2']:.4f}  MAE={mae_in:.2f}%")
print(f"  OOS:        R²={r2_oos:.4f}  MAE={mae_oos:.2f}%  Corr={corr_oos:.3f}")
print(f"  Current:    TSLA=${actual_now:.2f}  Fair=${fair:.2f}  Gap={(actual_now/fair-1)*100:+.1f}%")

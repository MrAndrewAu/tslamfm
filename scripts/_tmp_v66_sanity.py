"""
Sanity checks for v6.6 tuned estimator.

Outputs:
  - OLS vs ridge coefficient comparison on the selected v6.6 factor set
  - OOS sub-period metrics for v6.5 OLS vs v6.6 ridge(lambda=0.30)
  - Recent history band misses from generated history.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"
MIN_TRAIN = 100
EVENT_P_THRESHOLD = 0.10
RIDGE_LAMBDA = 5.00


def wk(sym):
    for kwargs in [
        dict(start=START, end=END, interval="1wk"),
        dict(start=START, end=END, interval="1wk", auto_adjust=False),
    ]:
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
    raise RuntimeError(f"Failed to fetch {sym}")


def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


def solve_beta(X, y, ridge_lambda=0.0):
    if ridge_lambda <= 0:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    features = X[:, 1:]
    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    X_scaled = np.column_stack([np.ones(len(X)), (features - means) / stds])
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    beta_scaled = np.linalg.solve(X_scaled.T @ X_scaled + ridge_lambda * penalty, X_scaled.T @ y)
    beta = np.empty_like(beta_scaled)
    beta[1:] = beta_scaled[1:] / stds
    beta[0] = beta_scaled[0] - np.sum(beta_scaled[1:] * means / stds)
    return beta


def fit(frame, factors, ridge_lambda=0.0):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    beta = solve_beta(X, y, ridge_lambda=ridge_lambda)
    yhat = X @ beta
    r = y - yhat
    n, k = X.shape
    s2 = (r @ r) / (n - k)
    beta_ols = solve_beta(X, y, ridge_lambda=0.0)
    r_ols = y - X @ beta_ols
    s2_ols = (r_ols @ r_ols) / (n - k)
    cov = s2_ols * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta_ols / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    return {"beta": beta, "p": p, "fitted": pd.Series(yhat, index=frame.index), "resid": pd.Series(r, index=frame.index)}


def select_factors(frame, forced, candidate_events):
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    model = fit(frame, factors, ridge_lambda=0.0)
    while events:
        event_ps = [(e, float(model["p"][1 + factors.index(e)])) for e in events]
        worst_event, worst_p = max(event_ps, key=lambda item: item[1])
        if worst_p <= EVENT_P_THRESHOLD:
            break
        events.remove(worst_event)
        factors = forced + events
        model = fit(frame, factors, ridge_lambda=0.0)
    return factors, events


def recursive_predictions(frame, forced, candidate_events, ridge_lambda=0.0):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < MIN_TRAIN:
            continue
        factors, _ = select_factors(train, forced, candidate_events)
        model = fit(train, factors, ridge_lambda=ridge_lambda)
        xv = np.array([1.0] + [float(row[c]) for c in factors])
        pred_log.loc[date] = float(xv @ model["beta"])
    return pred_log


def score(frame, pred_log, start, end=None):
    idx = pred_log.dropna().index
    idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    actual = np.exp(frame.loc[idx, "log_TSLA"].to_numpy())
    pred = np.exp(pred_log.loc[idx].to_numpy())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    ss_res = float(((actual - pred) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(pred / actual - 1)) * 100)
    corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 2 else float("nan")
    return len(actual), r2, mae, corr


def build_frame():
    tickers = {
        "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
        "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
        "IEF": "IEF", "SHY": "SHY", "VIX3M": "^VIX3M",
    }
    raw = {k: wk(v) for k, v in tickers.items()}
    df = pd.DataFrame(raw).resample("W-FRI").last().ffill().dropna()
    frame = pd.DataFrame(index=df.index)
    frame["log_TSLA"] = np.log(df["TSLA"])
    frame["log_QQQ"] = np.log(df["QQQ"])
    frame["log_DXY"] = np.log(df["DXY"])
    frame["log_VIX"] = np.log(df["VIX"])
    frame["NVDA_excess"] = residualize(np.log(df["NVDA"]), frame["log_QQQ"])
    frame["ARKK_excess"] = residualize(np.log(df["ARKK"]), frame["log_QQQ"])
    rbob = np.log(df["RBOB"])
    frame["RBOB_zscore_52w"] = (rbob - rbob.rolling(52, min_periods=20).mean()) / rbob.rolling(52, min_periods=20).std()
    curve = np.log(df["IEF"]) - np.log(df["SHY"])
    frame["curve_IEF_SHY_zscore_52w"] = (curve - curve.rolling(52, min_periods=20).mean()) / curve.rolling(52, min_periods=20).std()
    vix_ts = np.log(df["VIX3M"]) - np.log(df["VIX"])
    frame["vix_ts_zscore_52w"] = (vix_ts - vix_ts.rolling(52, min_periods=20).mean()) / vix_ts.rolling(52, min_periods=20).std()
    event_defs = [
        ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion", "2020-11-16"),
        ("Hertz_1T_peak", "2021-10-25"), ("Twitter_overhang", "2022-04-25"),
        ("Twitter_close", "2022-10-27"), ("AI_day_2023", "2023-07-19"),
        ("Trump_election", "2024-11-06"), ("DOGE_brand_damage", "2025-02-15"),
        ("Musk_exits_DOGE", "2025-04-22"), ("TrillionPay", "2025-09-05"),
        ("Tariff_shock", "2026-02-01"), ("Robotaxi_Austin", "2025-06-22"),
    ]
    for name, dt in event_defs:
        d0 = pd.Timestamp(dt)
        frame[f"E_{name}"] = ((frame.index >= d0) & (frame.index < d0 + pd.Timedelta(weeks=8))).astype(int)
    frame = frame.dropna()
    forced = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess", "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w", "vix_ts_zscore_52w"]
    events = [f"E_{name}" for name, _ in event_defs if frame[f"E_{name}"].nunique() > 1]
    return frame, forced, events


frame, forced, events = build_frame()
factors, kept_events = select_factors(frame, forced, events)
ols_model = fit(frame, factors, ridge_lambda=0.0)
ridge_model = fit(frame, factors, ridge_lambda=RIDGE_LAMBDA)

print("Selected events:", ", ".join(kept_events))
print("\nCoefficient comparison (full-sample selected factor set):")
print(f"{'factor':<32} {'OLS':>12} {'ridge':>12} {'shrink%':>10}")
print(f"{'Intercept':<32} {ols_model['beta'][0]:>12.5f} {ridge_model['beta'][0]:>12.5f} {'n/a':>10}")
for i, factor in enumerate(factors, start=1):
    old = float(ols_model["beta"][i])
    new = float(ridge_model["beta"][i])
    shrink = (1 - abs(new) / max(abs(old), 1e-12)) * 100
    print(f"{factor:<32} {old:>12.5f} {new:>12.5f} {shrink:>9.1f}%")

print("\nWalk-forward sub-period metrics:")
pred_ols = recursive_predictions(frame, forced, events, ridge_lambda=0.0)
pred_ridge = recursive_predictions(frame, forced, events, ridge_lambda=RIDGE_LAMBDA)
periods = [
    ("Full OOS", "2025-01-03", None),
    ("2025 H1", "2025-01-03", "2025-06-30"),
    ("2025 H2", "2025-07-01", "2025-12-31"),
    ("2026 YTD", "2026-01-01", None),
]
print(f"{'period':<10} {'n':>3} {'OLS R2':>8} {'ridge R2':>9} {'OLS MAE':>8} {'ridge MAE':>9} {'OLS Corr':>9} {'ridge Corr':>10}")
for label, start, end in periods:
    n1, r2o, maeo, corro = score(frame, pred_ols, start, end)
    n2, r2r, maer, corrr = score(frame, pred_ridge, start, end)
    print(f"{label:<10} {n2:>3} {r2o:>8.3f} {r2r:>9.3f} {maeo:>8.2f} {maer:>9.2f} {corro:>9.3f} {corrr:>10.3f}")

history = json.loads((ROOT / "public/data/history.json").read_text())
rows = [r for r in history if not r.get("partial_week")]
print("\nRecent generated history band status:")
print(f"{'date':<12} {'actual':>8} {'fitted':>8} {'low_q':>8} {'high_q':>8} {'inside_q':>9}")
for row in rows[-12:]:
    inside = row["low_q"] <= row["actual"] <= row["high_q"]
    print(f"{row['date']:<12} {row['actual']:>8.2f} {row['fitted']:>8.2f} {row['low_q']:>8.2f} {row['high_q']:>8.2f} {str(inside):>9}")

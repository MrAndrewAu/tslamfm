"""
Probe historical window length for the v6.6 estimator.

Compares the current 4y-ish production start against candidate starts while keeping
the rest of the model fixed:
  - same factors
  - same OLS event selection p<0.10
  - same standardized ridge final coefficients, lambda=5.00
  - same OOS start, 2025-01-03

This is an experiment only. It does not write model.json/history.json.
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

END = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())
OOS_START = "2025-01-03"
MIN_TRAIN = 100
EVENT_P_THRESHOLD = 0.10
RIDGE_LAMBDA = 5.00

WINDOWS = [
    ("current_4y", "2022-04-26"),
    ("try_aug2022", "2022-03-26"),
    ("try_3_5y", "2022-11-12"),
    ("try_4_5y", "2021-11-12"),
    ("try_8y", "2018-05-12"),
]


def wk(sym, start):
    for kwargs in [
        dict(start=start, end=END, interval="1wk"),
        dict(start=start, end=END, interval="1wk", auto_adjust=False),
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


def beta_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def beta_standardized_ridge(X, y, ridge_lambda=RIDGE_LAMBDA):
    if ridge_lambda <= 0:
        return beta_ols(X, y)
    features = X[:, 1:]
    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    X_scaled = np.column_stack([np.ones(len(X)), (features - means) / stds])
    penalty = np.eye(X_scaled.shape[1])
    penalty[0, 0] = 0.0
    beta_scaled = np.linalg.solve(X_scaled.T @ X_scaled + ridge_lambda * penalty, X_scaled.T @ y)
    beta = np.empty_like(beta_scaled)
    beta[1:] = beta_scaled[1:] / stds
    beta[0] = beta_scaled[0] - np.sum(beta_scaled[1:] * means / stds)
    return beta


def fit(frame, factors, ridge=False):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    beta = beta_standardized_ridge(X, y) if ridge else beta_ols(X, y)
    yhat = X @ beta

    beta_ref = beta_ols(X, y)
    resid_ref = y - X @ beta_ref
    n, k = X.shape
    s2 = (resid_ref @ resid_ref) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - stats.t.cdf(np.abs(beta_ref / se), df=n - k))
    return {"beta": beta, "p": p, "fitted": pd.Series(yhat, index=frame.index)}


def select_factors(frame, forced, candidate_events):
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    model = fit(frame, factors, ridge=False)
    while events:
        event_ps = [(e, float(model["p"][1 + factors.index(e)])) for e in events]
        worst_event, worst_p = max(event_ps, key=lambda item: item[1])
        if worst_p <= EVENT_P_THRESHOLD:
            break
        events.remove(worst_event)
        factors = forced + events
        model = fit(frame, factors, ridge=False)
    return factors, events


def recursive_predictions(frame, forced, candidate_events):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < MIN_TRAIN:
            continue
        factors, _ = select_factors(train, forced, candidate_events)
        model = fit(train, factors, ridge=True)
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


def build_frame(start):
    tickers = {
        "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
        "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
        "IEF": "IEF", "SHY": "SHY", "VIX3M": "^VIX3M",
    }
    raw = {k: wk(v, start) for k, v in tickers.items()}
    df = pd.DataFrame(raw).resample("W-FRI").last().ffill().dropna()
    today = pd.Timestamp.today().normalize()
    days_since_friday = (today.weekday() - 4) % 7
    last_completed_friday = today - pd.Timedelta(days=days_since_friday)
    df = df[df.index <= last_completed_friday]

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
    forced = [
        "log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
        "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w", "vix_ts_zscore_52w",
    ]
    events = [f"E_{name}" for name, _ in event_defs if frame[f"E_{name}"].nunique() > 1]
    return frame, forced, events


print("Window length probe: v6.6 estimator, standardized ridge lambda=5.00")
print(f"Fetch END={END}; OOS_START={OOS_START}\n")
summary = []
for label, start in WINDOWS:
    frame, forced, events = build_frame(start)
    selected, kept = select_factors(frame, forced, events)
    pred = recursive_predictions(frame, forced, events)
    full = score(frame, pred, OOS_START)
    h1 = score(frame, pred, "2025-01-03", "2025-06-30")
    h2 = score(frame, pred, "2025-07-01", "2025-12-31")
    ytd = score(frame, pred, "2026-01-01")
    model = fit(frame, selected, ridge=True)
    fitted = np.exp(model["fitted"].to_numpy())
    actual = np.exp(frame["log_TSLA"].to_numpy())
    mae_in = float(np.mean(np.abs(fitted / actual - 1)) * 100)
    print(f"{label}: start={start} effective={frame.index[0].date()}->{frame.index[-1].date()} n={len(frame)} events={','.join(kept)}")
    print(f"  In-sample MAE={mae_in:.2f}%")
    print(f"  Full OOS n={full[0]} R2={full[1]:.4f} MAE={full[2]:.2f}% Corr={full[3]:.3f}")
    print(f"  2025 H1  n={h1[0]} R2={h1[1]:.4f} MAE={h1[2]:.2f}% Corr={h1[3]:.3f}")
    print(f"  2025 H2  n={h2[0]} R2={h2[1]:.4f} MAE={h2[2]:.2f}% Corr={h2[3]:.3f}")
    print(f"  2026 YTD n={ytd[0]} R2={ytd[1]:.4f} MAE={ytd[2]:.2f}% Corr={ytd[3]:.3f}\n")
    summary.append((label, start, len(frame), full[1], full[2], full[3]))

base = summary[0]
print("Summary vs current_4y baseline")
for label, start, n, r2, mae, corr in summary:
    print(
        f"  {label:<11} n={n:<3} R2={r2:.4f} ({(r2-base[3])*100:+.2f}pp) "
        f"MAE={mae:.2f}% ({mae-base[4]:+.2f}pp) Corr={corr:.3f} ({corr-base[5]:+.3f})"
    )
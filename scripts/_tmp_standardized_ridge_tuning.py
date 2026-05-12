"""
Compare unstandardized vs standardized ridge for v6.5/v6.6 factor weights.

Why: plain ridge on raw features is scale-dependent. Standardized ridge centers
and scales predictors in each training slice, leaves the intercept unpenalized,
and converts coefficients back to raw-feature space for scoring/export.
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"
MIN_TRAIN = 100
EVENT_P_THRESHOLD = 0.10


def wk(sym):
    for kwargs in [dict(start=START, end=END, interval="1wk"), dict(start=START, end=END, interval="1wk", auto_adjust=False)]:
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


def build_frame():
    tickers = {
        "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
        "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
        "IEF": "IEF", "SHY": "SHY", "VIX3M": "^VIX3M",
    }
    raw = {k: wk(v) for k, v in tickers.items()}
    df = pd.DataFrame(raw).resample("W-FRI").last().ffill().dropna()
    f = pd.DataFrame(index=df.index)
    f["log_TSLA"] = np.log(df["TSLA"])
    f["log_QQQ"] = np.log(df["QQQ"])
    f["log_DXY"] = np.log(df["DXY"])
    f["log_VIX"] = np.log(df["VIX"])
    f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
    f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])
    rbob = np.log(df["RBOB"])
    f["RBOB_zscore_52w"] = (rbob - rbob.rolling(52, min_periods=20).mean()) / rbob.rolling(52, min_periods=20).std()
    curve = np.log(df["IEF"]) - np.log(df["SHY"])
    f["curve_IEF_SHY_zscore_52w"] = (curve - curve.rolling(52, min_periods=20).mean()) / curve.rolling(52, min_periods=20).std()
    vix_ts = np.log(df["VIX3M"]) - np.log(df["VIX"])
    f["vix_ts_zscore_52w"] = (vix_ts - vix_ts.rolling(52, min_periods=20).mean()) / vix_ts.rolling(52, min_periods=20).std()
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
        f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)
    f = f.dropna()
    forced = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess", "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w", "vix_ts_zscore_52w"]
    events = [f"E_{name}" for name, _ in event_defs if f[f"E_{name}"].nunique() > 1]
    return f, forced, events


def ols_beta(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def raw_ridge_beta(X, y, lam):
    if lam <= 0:
        return ols_beta(X, y)
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)


def standardized_ridge_beta(X, y, lam):
    if lam <= 0:
        return ols_beta(X, y)
    Z = X[:, 1:]
    means = Z.mean(axis=0)
    stds = Z.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    Zs = (Z - means) / stds
    Xs = np.column_stack([np.ones(len(X)), Zs])
    penalty = np.eye(Xs.shape[1])
    penalty[0, 0] = 0.0
    beta_std = np.linalg.solve(Xs.T @ Xs + lam * penalty, Xs.T @ y)
    beta_raw = np.empty_like(beta_std)
    beta_raw[1:] = beta_std[1:] / stds
    beta_raw[0] = beta_std[0] - np.sum(beta_std[1:] * means / stds)
    return beta_raw


def fit(frame, factors, method="ols", lam=0.0):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    if method == "standardized_ridge":
        beta = standardized_ridge_beta(X, y, lam)
    elif method == "raw_ridge":
        beta = raw_ridge_beta(X, y, lam)
    else:
        beta = ols_beta(X, y)
    yhat = X @ beta
    r = y - yhat
    n, k = X.shape
    beta_ols = ols_beta(X, y)
    r_ols = y - X @ beta_ols
    s2 = (r_ols @ r_ols) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - stats.t.cdf(np.abs(beta_ols / se), df=n - k))
    return {"beta": beta, "p": p}


def select_factors(frame, forced, events):
    kept = [e for e in events if frame[e].nunique() > 1]
    factors = forced + kept
    model = fit(frame, factors, method="ols")
    while kept:
        event_ps = [(e, float(model["p"][1 + factors.index(e)])) for e in kept]
        worst_event, worst_p = max(event_ps, key=lambda item: item[1])
        if worst_p <= EVENT_P_THRESHOLD:
            break
        kept.remove(worst_event)
        factors = forced + kept
        model = fit(frame, factors, method="ols")
    return factors


def recursive_predictions(frame, forced, events, method="ols", lam=0.0):
    pred = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < MIN_TRAIN:
            continue
        factors = select_factors(train, forced, events)
        model = fit(train, factors, method=method, lam=lam)
        pred.loc[date] = float(np.array([1.0] + [float(row[c]) for c in factors]) @ model["beta"])
    return pred


def score(frame, pred):
    idx = pred.dropna().index
    idx = idx[idx >= pd.Timestamp(OOS_START)]
    actual = np.exp(frame.loc[idx, "log_TSLA"].to_numpy())
    forecast = np.exp(pred.loc[idx].to_numpy())
    r2 = 1 - ((actual - forecast) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    mae = np.mean(np.abs(forecast / actual - 1)) * 100
    corr = np.corrcoef(actual, forecast)[0, 1]
    return float(r2), float(mae), float(corr), len(actual)


frame, forced, events = build_frame()
grid = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
print("method,lambda,r2,mae,corr,n")
all_results = []
for method in ["raw_ridge", "standardized_ridge"]:
    for lam in grid:
        pred = recursive_predictions(frame, forced, events, method=method, lam=lam)
        r2, mae, corr, n = score(frame, pred)
        all_results.append((method, lam, r2, mae, corr, n))
        print(f"{method},{lam:.2f},{r2:.4f},{mae:.2f},{corr:.3f},{n}")
print("\nBest by R2/MAE/Corr:")
for method in ["raw_ridge", "standardized_ridge"]:
    subset = [r for r in all_results if r[0] == method]
    best = max(subset, key=lambda r: (r[2], -r[3], r[4]))
    print(f"{method}: lambda={best[1]:.2f} R2={best[2]:.4f} MAE={best[3]:.2f}% Corr={best[4]:.3f}")

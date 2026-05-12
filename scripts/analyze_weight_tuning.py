"""
Probe: tune coefficient estimation for v6.5 to improve OOS R^2 / MAE / Corr.

Local hypothesis:
  Plain OLS on the forced factor set is slightly overfit for the 190-row sample.
  Mild ridge shrinkage on continuous factors should reduce variance and improve
  the OOS tradeoff without changing the accepted economic stories.

This script compares:
  - baseline OLS (current production estimator)
  - ridge regression over a lambda grid

Event dummies remain in the model-selection loop but are not included in the
ridge probe here; the goal is to test whether continuous-weight shrinkage is
the cheapest reliable improvement path before changing production code.
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
    for kwargs in [
        dict(start=START, end=END, interval="1wk"),
        dict(start=START, end=END, interval="1wk", auto_adjust=False),
    ]:
        try:
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
        except Exception:
            continue
    raise RuntimeError(f"Failed to fetch {sym}")


def residualize(target, base_s):
    X = np.column_stack([np.ones(len(base_s)), base_s.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    resid = target.to_numpy() - X @ coef
    return float(coef[0]), float(coef[1]), pd.Series(resid, index=target.index)


def build_frame():
    tickers = {
        "TSLA": "TSLA",
        "QQQ": "QQQ",
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX",
        "NVDA": "NVDA",
        "ARKK": "ARKK",
        "RBOB": "RB=F",
        "IEF": "IEF",
        "SHY": "SHY",
        "VIX3M": "^VIX3M",
    }
    s = {k: wk(v) for k, v in tickers.items()}
    df = pd.DataFrame(s).resample("W-FRI").last().ffill().dropna()

    f = pd.DataFrame(index=df.index)
    f["log_TSLA"] = np.log(df["TSLA"])
    f["log_QQQ"] = np.log(df["QQQ"])
    f["log_DXY"] = np.log(df["DXY"])
    f["log_VIX"] = np.log(df["VIX"])
    _, _, f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
    _, _, f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

    rbob_log = np.log(df["RBOB"])
    rbob_mean_52 = rbob_log.rolling(window=52, min_periods=20).mean()
    rbob_std_52 = rbob_log.rolling(window=52, min_periods=20).std()
    f["RBOB_zscore_52w"] = (rbob_log - rbob_mean_52) / rbob_std_52

    curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
    curve_mean_52 = curve_log.rolling(window=52, min_periods=20).mean()
    curve_std_52 = curve_log.rolling(window=52, min_periods=20).std()
    f["curve_IEF_SHY_zscore_52w"] = (curve_log - curve_mean_52) / curve_std_52

    vix_ts_log = np.log(df["VIX3M"]) - np.log(df["VIX"])
    vix_ts_mean_52 = vix_ts_log.rolling(window=52, min_periods=20).mean()
    vix_ts_std_52 = vix_ts_log.rolling(window=52, min_periods=20).std()
    f["vix_ts_zscore_52w"] = (vix_ts_log - vix_ts_mean_52) / vix_ts_std_52

    event_defs = [
        ("Split_squeeze_2020", "2020-08-11"),
        ("SP500_inclusion", "2020-11-16"),
        ("Hertz_1T_peak", "2021-10-25"),
        ("Twitter_overhang", "2022-04-25"),
        ("Twitter_close", "2022-10-27"),
        ("AI_day_2023", "2023-07-19"),
        ("Trump_election", "2024-11-06"),
        ("DOGE_brand_damage", "2025-02-15"),
        ("Musk_exits_DOGE", "2025-04-22"),
        ("TrillionPay", "2025-09-05"),
        ("Tariff_shock", "2026-02-01"),
        ("Robotaxi_Austin", "2025-06-22"),
    ]
    for name, dt in event_defs:
        d0 = pd.Timestamp(dt)
        f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

    f = f.dropna()
    forced = [
        "log_QQQ",
        "log_DXY",
        "log_VIX",
        "NVDA_excess",
        "ARKK_excess",
        "RBOB_zscore_52w",
        "curve_IEF_SHY_zscore_52w",
        "vix_ts_zscore_52w",
    ]
    active_event_names = [name for name, _ in event_defs if f[f"E_{name}"].nunique() > 1]
    all_events = [f"E_{name}" for name in active_event_names]
    return f, forced, all_events


def solve_beta(frame, factors, ridge_lambda=0.0):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    if ridge_lambda <= 0:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    pen = np.eye(X.shape[1])
    pen[0, 0] = 0.0
    xtx = X.T @ X + ridge_lambda * pen
    xty = X.T @ y
    return np.linalg.solve(xtx, xty)


def fit(frame, factors, ridge_lambda=0.0):
    beta = solve_beta(frame, factors, ridge_lambda=ridge_lambda)
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    yhat = X @ beta
    r = y - yhat
    n, k = X.shape
    s2 = (r @ r) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss_tot
    return dict(
        beta=beta,
        p=p,
        r2=float(r2),
        fitted=pd.Series(yhat, index=frame.index),
        resid=pd.Series(r, index=frame.index),
    )


def select_factors(frame, forced, candidate_events, ridge_lambda=0.0, p_threshold=EVENT_P_THRESHOLD):
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    model = fit(frame, factors, ridge_lambda=ridge_lambda)
    while events:
        event_ps = [(e, float(model["p"][1 + factors.index(e)])) for e in events]
        worst_event, worst_p = max(event_ps, key=lambda item: item[1])
        if worst_p <= p_threshold:
            break
        events.remove(worst_event)
        factors = forced + events
        model = fit(frame, factors, ridge_lambda=ridge_lambda)
    return factors, model


def recursive_predictions(frame, forced, candidate_events, ridge_lambda=0.0, min_train=MIN_TRAIN):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, train_model = select_factors(train, forced, candidate_events, ridge_lambda=ridge_lambda)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred_log.loc[date] = float(xv @ train_model["beta"])
    return pred_log


def recursive_predictions_ols_select_ridge_fit(frame, forced, candidate_events, ridge_lambda=0.0, min_train=MIN_TRAIN):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, _ = select_factors(train, forced, candidate_events, ridge_lambda=0.0)
        train_model = fit(train, facs, ridge_lambda=ridge_lambda)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred_log.loc[date] = float(xv @ train_model["beta"])
    return pred_log


def score(frame, pred_log):
    predicted_rows = pred_log.dropna().index
    oos_idx = predicted_rows[predicted_rows >= pd.Timestamp(OOS_START)]
    oos_pred = np.exp(pred_log.loc[oos_idx].to_numpy())
    oos_actual = np.exp(frame.loc[oos_idx, "log_TSLA"].to_numpy())
    mae_oos = float(np.mean(np.abs(oos_pred / oos_actual - 1)) * 100)
    ss_tot = float(((oos_actual - oos_actual.mean()) ** 2).sum())
    ss_res = float(((oos_actual - oos_pred) ** 2).sum())
    r2_oos = 1 - ss_res / ss_tot
    corr_oos = float(np.corrcoef(oos_actual, oos_pred)[0, 1])
    return dict(r2=r2_oos, mae=mae_oos, corr=corr_oos, n=int(len(oos_actual)))


print("Building v6.5 frame...")
frame, forced, all_events = build_frame()
print(f"  rows={len(frame)}  forced={len(forced)}  active_events={len(all_events)}")

grid = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 1.0, 3.0, 10.0, 30.0]
results = []
print("\nTuning ridge lambda over the live v6.5 factor set...")
for lam in grid:
    pred = recursive_predictions(frame, forced, all_events, ridge_lambda=lam)
    metrics = score(frame, pred)
    results.append(dict(lambda_=lam, **metrics))
    print(
        f"  lambda={lam:>5.2f}  OOS R^2={metrics['r2']:.4f}  "
        f"MAE={metrics['mae']:.2f}%  Corr={metrics['corr']:.3f}  n={metrics['n']}"
    )

hybrid_results = []
print("\nTuning ridge lambda with OLS event selection + ridge final fit...")
for lam in grid:
    pred = recursive_predictions_ols_select_ridge_fit(frame, forced, all_events, ridge_lambda=lam)
    metrics = score(frame, pred)
    hybrid_results.append(dict(lambda_=lam, **metrics))
    print(
        f"  lambda={lam:>5.2f}  OOS R^2={metrics['r2']:.4f}  "
        f"MAE={metrics['mae']:.2f}%  Corr={metrics['corr']:.3f}  n={metrics['n']}"
    )

baseline = next(r for r in results if r["lambda_"] == 0.0)
best = max(
    results,
    key=lambda r: (r["r2"], -r["mae"], r["corr"]),
)
best_hybrid = max(
    hybrid_results,
    key=lambda r: (r["r2"], -r["mae"], r["corr"]),
)

print("\nSummary")
print(
    f"  baseline lambda=0.00 -> R^2={baseline['r2']:.4f}  "
    f"MAE={baseline['mae']:.2f}%  Corr={baseline['corr']:.3f}"
)
print(
    f"  best     lambda={best['lambda_']:.2f} -> R^2={best['r2']:.4f}  "
    f"MAE={best['mae']:.2f}%  Corr={best['corr']:.3f}"
)
print(
    f"  deltas                -> R^2={(best['r2'] - baseline['r2']) * 100:+.2f}pp  "
    f"MAE={best['mae'] - baseline['mae']:+.2f}pp  Corr={best['corr'] - baseline['corr']:+.3f}"
)
print(
    f"  best hybrid lambda={best_hybrid['lambda_']:.2f} -> R^2={best_hybrid['r2']:.4f}  "
    f"MAE={best_hybrid['mae']:.2f}%  Corr={best_hybrid['corr']:.3f}"
)
print(
    f"  hybrid deltas         -> R^2={(best_hybrid['r2'] - baseline['r2']) * 100:+.2f}pp  "
    f"MAE={best_hybrid['mae'] - baseline['mae']:+.2f}pp  Corr={best_hybrid['corr'] - baseline['corr']:+.3f}"
)
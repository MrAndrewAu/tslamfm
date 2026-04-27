import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

START, END = '2022-04-26', '2026-04-26'
OOS_START = '2025-01-03'
MIN_TRAIN = 100
EVENT_P_THRESHOLD = 0.10
LAMBDA = 0.94
BAND_MIN_ERRORS = 26


def wk(sym):
    candidates = []
    try:
        candidates.append(yf.Ticker(sym).history(start=START, end=END, interval='1wk')['Close'])
    except Exception:
        pass
    try:
        dl = yf.download(sym, start=START, end=END, interval='1wk', progress=False, auto_adjust=False)
        if not dl.empty and 'Close' in dl.columns:
            candidates.append(dl['Close'])
    except Exception:
        pass
    for series in candidates:
        series = series.dropna()
        if series.empty:
            continue
        idx = pd.to_datetime(series.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_localize(None)
        series.index = idx
        return series
    raise RuntimeError(sym)


def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    resid = target.to_numpy() - X @ coef
    return float(coef[0]), float(coef[1]), pd.Series(resid, index=target.index)


def fit(frame, factors):
    y = frame['log_TSLA'].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    s2 = (resid @ resid) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    return {
        'beta': beta,
        'p': p,
        'fitted': pd.Series(yhat, index=frame.index),
        'resid': pd.Series(resid, index=frame.index),
        'sigma': float(np.sqrt(s2)),
    }


def select_factors(frame, forced, candidate_events):
    events = [event for event in candidate_events if frame[event].nunique() > 1]
    factors = forced + events
    model = fit(frame, factors)
    while events:
        event_ps = [(event, float(model['p'][1 + factors.index(event)])) for event in events]
        worst_event, worst_p = max(event_ps, key=lambda item: item[1])
        if worst_p <= EVENT_P_THRESHOLD:
            break
        events.remove(worst_event)
        factors = forced + events
        model = fit(frame, factors)
    return factors, events, model


def recursive_predictions(frame, forced, candidate_events):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < MIN_TRAIN:
            continue
        factors, _, train_model = select_factors(train, forced, candidate_events)
        xv = np.array([1.0] + [float(row[c]) for c in factors])
        pred_log.loc[date] = float(xv @ train_model['beta'])
    return pred_log


raw = {k: wk(v) for k, v in {
    'TSLA': 'TSLA', 'QQQ': 'QQQ', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX',
    'NVDA': 'NVDA', 'ARKK': 'ARKK', 'RBOB': 'RB=F', 'IEF': 'IEF', 'SHY': 'SHY',
}.items()}
df = pd.DataFrame(raw).resample('W-FRI').last().ffill().dropna()
frame = pd.DataFrame(index=df.index)
frame['log_TSLA'] = np.log(df['TSLA'])
frame['log_QQQ'] = np.log(df['QQQ'])
frame['log_DXY'] = np.log(df['DXY'])
frame['log_VIX'] = np.log(df['VIX'])
_, _, frame['NVDA_excess'] = residualize(np.log(df['NVDA']), frame['log_QQQ'])
_, _, frame['ARKK_excess'] = residualize(np.log(df['ARKK']), frame['log_QQQ'])
log_rbob = np.log(df['RBOB'])
frame['RBOB_zscore_52w'] = (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) / log_rbob.rolling(52, min_periods=20).std()
curve_log = np.log(df['IEF']) - np.log(df['SHY'])
frame['curve_IEF_SHY_zscore_52w'] = (curve_log - curve_log.rolling(52, min_periods=20).mean()) / curve_log.rolling(52, min_periods=20).std()

EVENT_DEFS = [
    ('Split_squeeze_2020', '2020-08-11'), ('SP500_inclusion', '2020-11-16'),
    ('Hertz_1T_peak', '2021-10-25'), ('Twitter_overhang', '2022-04-25'),
    ('Twitter_close', '2022-10-27'), ('AI_day_2023', '2023-07-19'),
    ('Trump_election', '2024-11-06'), ('DOGE_brand_damage', '2025-02-15'),
    ('Musk_exits_DOGE', '2025-04-22'), ('TrillionPay', '2025-09-05'),
    ('Tariff_shock', '2026-02-01'), ('Robotaxi_Austin', '2025-06-22'),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    frame[f'E_{name}'] = ((frame.index >= d0) & (frame.index < d0 + pd.Timedelta(weeks=8))).astype(int)
frame = frame.dropna()
forced = ['log_QQQ', 'log_DXY', 'log_VIX', 'NVDA_excess', 'ARKK_excess', 'RBOB_zscore_52w', 'curve_IEF_SHY_zscore_52w']
events = [f'E_{name}' for name, _ in EVENT_DEFS if frame[f'E_{name}'].nunique() > 1]
factors, kept_events, model = select_factors(frame, forced, events)
pred_log = recursive_predictions(frame, forced, events)
pred_idx = pred_log.dropna().index
pred_errors = (frame.loc[pred_idx, 'log_TSLA'] - pred_log.loc[pred_idx]).astype(float)
resid_log = model['resid'].to_numpy()
seed_var = float(np.var(resid_log[:26], ddof=1))
fallback_sigma = np.empty(len(resid_log))
state = seed_var
for i in range(len(resid_log)):
    fallback_sigma[i] = float(np.sqrt(max(state, 1e-8)))
    state = LAMBDA * state + (1 - LAMBDA) * (resid_log[i] ** 2)

sigma_pred_is = float(np.std(pred_errors.to_numpy(), ddof=1))
results = []
for exponent in [0.0, 0.25, 0.5, 0.75, 1.0]:
    seen = []
    pred_var_state = None
    rows = []
    for i, date in enumerate(frame.index):
        fit_v = float(np.exp(model['fitted'].loc[date]))
        actual = float(np.exp(frame.loc[date, 'log_TSLA']))
        if len(seen) >= BAND_MIN_ERRORS:
            if pred_var_state is None:
                pred_var_state = float(np.var(np.array(seen), ddof=1))
            sigma_band = float(np.sqrt(max(pred_var_state, 1e-8)))
            q10_raw = float(np.percentile(seen, 10))
            q90_raw = float(np.percentile(seen, 90))
            scale = (sigma_band / max(sigma_pred_is, 1e-8)) ** exponent
            low = fit_v * float(np.exp(q10_raw * scale))
            high = fit_v * float(np.exp(q90_raw * scale))
            rows.append((date.strftime('%Y-%m-%d'), actual, low, high))
        if date in pred_errors.index:
            err = float(pred_errors.loc[date])
            seen.append(err)
            if pred_var_state is not None:
                pred_var_state = LAMBDA * pred_var_state + (1 - LAMBDA) * (err ** 2)
    full = float(np.mean([low <= actual <= high for _, actual, low, high in rows]))
    oos_rows = [row for row in rows if row[0] >= OOS_START]
    oos = float(np.mean([low <= actual <= high for _, actual, low, high in oos_rows]))
    results.append({'exponent': exponent, 'coverage_full': full, 'coverage_oos': oos})

print(json.dumps({'kept_events': kept_events, 'results': results}, indent=2))

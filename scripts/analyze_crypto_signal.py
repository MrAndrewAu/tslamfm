"""
Crypto-signal prediction model for TSLA: 1-week and 4-week forward returns.

Features (all lagged — available *before* the target return is realized):
  BTC_zscore_52w, BTC_excess_vs_QQQ, ETH_zscore_52w, ETH_excess_vs_QQQ
  + base controls: QQQ_ret_1w, log_VIX

Targets:
  y_1w[t] = log( TSLA[t+1] / TSLA[t] )   next-week log return
  y_4w[t] = log( TSLA[t+4] / TSLA[t] )   next-4-week log return

Walk-forward OLS (expanding window, min_train=52 weeks).
OOS evaluation starts: 2025-01-03.

Outputs:
  • Per-horizon table: OOS R², Pearson IC, Spearman IC, Hit Rate, MAE
  • Feature importance (in-sample beta / σ_feature)
  • Current live forecast using most-recent available features
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END = "2022-04-26", "2026-04-26"
OOS_START  = "2025-01-03"
MIN_TRAIN  = 52   # weeks before first OOS prediction


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def wk(sym: str) -> pd.Series:
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s


def residualize(target: pd.Series, base: pd.Series) -> pd.Series:
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


def build_frame() -> pd.DataFrame:
    """Fetch all series and return a weekly feature / target frame."""
    print("Fetching weekly closes …")
    raw = {k: wk(v) for k, v in {
        "TSLA": "TSLA", "QQQ": "QQQ", "BTC": "BTC-USD",
        "ETH": "ETH-USD", "VIX": "^VIX",
    }.items()}
    df = pd.DataFrame(raw).resample("W-FRI").last().ffill()

    f = pd.DataFrame(index=df.index)

    # --- base controls ---
    f["log_QQQ"]    = np.log(df["QQQ"])
    f["log_VIX"]    = np.log(df["VIX"])
    f["QQQ_ret_1w"] = f["log_QQQ"].diff()   # lagged 1 week when shifted below

    # --- BTC features ---
    log_btc = np.log(df["BTC"])
    rmean_btc = log_btc.rolling(52, min_periods=20).mean()
    rstd_btc  = log_btc.rolling(52, min_periods=20).std()
    f["BTC_zscore_52w"]    = (log_btc - rmean_btc) / rstd_btc
    f["BTC_excess_vs_QQQ"] = residualize(log_btc, f["log_QQQ"])

    # --- ETH features ---
    log_eth = np.log(df["ETH"])
    rmean_eth = log_eth.rolling(52, min_periods=20).mean()
    rstd_eth  = log_eth.rolling(52, min_periods=20).std()
    f["ETH_zscore_52w"]    = (log_eth - rmean_eth) / rstd_eth
    f["ETH_excess_vs_QQQ"] = residualize(log_eth, f["log_QQQ"])

    # --- TSLA forward log returns (targets) ---
    log_tsla = np.log(df["TSLA"])
    f["y_1w"] = log_tsla.shift(-1) - log_tsla   # realized 1-week ahead
    f["y_4w"] = log_tsla.shift(-4) - log_tsla   # realized 4-week ahead

    f = f.dropna(subset=["BTC_zscore_52w", "ETH_zscore_52w"])
    return f


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

FEATURES = [
    "BTC_zscore_52w", "BTC_excess_vs_QQQ",
    "ETH_zscore_52w", "ETH_excess_vs_QQQ",
    "QQQ_ret_1w", "log_VIX",
]


def walk_forward(frame: pd.DataFrame, target: str) -> dict:
    """
    Expanding-window OLS.  At each OOS date t we:
      1. Train on all rows s < t where both X and y[s] are observed.
      2. Predict y_hat[t] using X[t] (features at t are available; target at t is future).
      3. Collect realized y[t] once it arrives and compare.

    For 4-week targets, note y_4w[t] requires TSLA price 4 weeks after t.
    We handle this by only scoring rows where y_4w is not NaN (i.e., we
    have enough future data).
    """
    oos_dates = frame.index[frame.index >= pd.Timestamp(OOS_START)]
    X_all = np.column_stack([np.ones(len(frame))]
                            + [frame[c].to_numpy() for c in FEATURES])
    y_all = frame[target].to_numpy()

    preds, actuals, dates = [], [], []

    for date in oos_dates:
        i = frame.index.get_loc(date)
        # training rows: all rows before this date where target is not NaN
        train_mask = np.arange(len(frame)) < i
        valid_mask = train_mask & ~np.isnan(y_all)
        if valid_mask.sum() < MIN_TRAIN:
            continue
        # test row: features must be complete; target can be NaN (future)
        x_test = X_all[i]
        if np.any(np.isnan(x_test)):
            continue
        y_test = y_all[i]
        if np.isnan(y_test):
            continue   # realized return not yet available (edge of data)

        X_tr = X_all[valid_mask]
        y_tr = y_all[valid_mask]
        b, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        preds.append(float(x_test @ b))
        actuals.append(float(y_test))
        dates.append(date)

    preds   = np.array(preds)
    actuals = np.array(actuals)
    n       = len(preds)

    if n < 5:
        return {"target": target, "n_oos": n, "status": "insufficient OOS data"}

    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    ss_res = ((actuals - preds) ** 2).sum()
    r2     = float(1 - ss_res / ss_tot)
    ic_p   = float(stats.pearsonr(preds, actuals)[0])
    ic_s   = float(stats.spearmanr(preds, actuals)[0])
    hit    = float(np.mean(np.sign(preds) == np.sign(actuals)))
    mae    = float(np.mean(np.abs(actuals - preds)) * 100)   # in %

    return {
        "target":  target,
        "n_oos":   n,
        "r2_oos":  r2,
        "ic_pearson": ic_p,
        "ic_spearman": ic_s,
        "hit_rate": hit,
        "mae_pct":  mae,
        "dates":    dates,
        "preds":    preds,
        "actuals":  actuals,
    }


# ---------------------------------------------------------------------------
# Feature importance (full-sample betas, standardised)
# ---------------------------------------------------------------------------

def feature_importance(frame: pd.DataFrame, target: str) -> None:
    sub = frame.dropna(subset=[target] + FEATURES)
    X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in FEATURES])
    y = sub[target].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    sds = [sub[c].std() for c in FEATURES]
    print(f"\n  Feature importance ({target}, full-sample standardised beta):")
    for name, beta, sd in zip(FEATURES, b[1:], sds):
        print(f"    {name:<28}  β={beta*sd:+.4f}")


# ---------------------------------------------------------------------------
# Live forecast
# ---------------------------------------------------------------------------

def live_forecast(frame: pd.DataFrame) -> None:
    """Use the full historical sample to fit coefficients, then predict on
    the most recent row that has complete features (but may lack future target)."""
    last = frame.dropna(subset=FEATURES).iloc[-1]
    print(f"\n{'='*60}")
    print(f"  LIVE FORECAST  (features as of {last.name.date()})")
    print(f"{'='*60}")
    for target, label in [("y_1w", "1-week"), ("y_4w", "4-week")]:
        sub = frame.dropna(subset=[target] + FEATURES)
        X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in FEATURES])
        y = sub[target].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        x_new = np.array([1.0] + [float(last[c]) for c in FEATURES])
        pred_log = float(x_new @ b)
        pred_pct = (np.exp(pred_log) - 1) * 100
        direction = "UP" if pred_log > 0 else "DOWN"
        print(f"  {label:>8}  predicted log-return: {pred_log:+.4f}  "
              f"({pred_pct:+.2f}%)  [{direction}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    frame = build_frame()
    print(f"  {len(frame)} weekly rows  "
          f"{frame.index[0].date()} → {frame.index[-1].date()}\n")

    hdr = (f"{'horizon':<8}{'n_oos':>7}{'OOS R²':>9}"
           f"{'IC(P)':>8}{'IC(S)':>8}{'HitRate':>9}{'MAE%':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for target, label in [("y_1w", "1-week"), ("y_4w", "4-week")]:
        r = walk_forward(frame, target)
        results[target] = r
        if "status" in r:
            print(f"  {label:<8}  {r['status']}")
            continue
        print(f"  {label:<8}"
              f"{r['n_oos']:>7}"
              f"{r['r2_oos']:>+9.4f}"
              f"{r['ic_pearson']:>+8.3f}"
              f"{r['ic_spearman']:>+8.3f}"
              f"{r['hit_rate']:>8.1%}"
              f"{r['mae_pct']:>7.2f}")

    for target, label in [("y_1w", "1-week"), ("y_4w", "4-week")]:
        feature_importance(frame, target)

    live_forecast(frame)


if __name__ == "__main__":
    main()

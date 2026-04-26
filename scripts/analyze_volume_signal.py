"""
Test whether TSLA traded volume adds predictive power to v6-canonical-4y.

Procedure (mirrors analyze_options_signal.py):
  1. Pull weekly TSLA + QQQ volumes from yfinance over the model window.
  2. Build three candidate transforms:
       log_volume          -- raw log(weekly volume)
       volume_zscore_52w   -- (log_vol - rolling_mean_52w) / rolling_std_52w
       volume_excess       -- residual of log(TSLA_vol) ~ log(QQQ_vol)
  3. Correlate each against model residuals (resid_log = log(actual)-log(fitted))
     contemporaneously (corr_t) and one week ahead (corr_t+1).
  4. Verdict per project bar: PROMISING needs p<0.05, |corr|>0.15, n>=50
     on the LAGGED correlation (the only one that matters for forecasting).

Run AFTER build_model_data.py so history.json exists.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "public" / "data" / "history.json"
START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"


def weekly_volume(sym: str) -> pd.Series:
    h = yf.Ticker(sym).history(start=START, end=END, interval="1wk")
    s = h["Volume"].astype(float)
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s


def residualize(target: pd.Series, base: pd.Series) -> pd.Series:
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


def load_residuals() -> pd.DataFrame:
    rows = json.loads(HIST.read_text())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["resid_log"] = np.log(df["actual"]) - np.log(df["fitted"])
    return df[["actual", "fitted", "resid_log"]]


def report(name: str, factor: pd.Series, resid: pd.Series) -> dict:
    s = factor.dropna()
    s = s.reindex(resid.index, method="nearest", tolerance=pd.Timedelta("3D")).dropna()
    r = resid.loc[s.index]
    out = {"factor": name, "n": int(len(s))}
    if len(s) < 20:
        return {**out, "status": "insufficient"}

    c0, p0 = stats.pearsonr(s, r)
    out["corr_t"], out["pval_t"] = float(c0), float(p0)

    # factor at t  vs  residual at t+1
    s_lag = s.iloc[:-1]
    r_next = resid.reindex(s.index).shift(-1).iloc[:-1]
    mask = s_lag.notna() & r_next.notna()
    if mask.sum() >= 20:
        c1, p1 = stats.pearsonr(s_lag[mask], r_next[mask])
        out["corr_t+1"], out["pval_t+1"] = float(c1), float(p1)
    else:
        out["corr_t+1"], out["pval_t+1"] = np.nan, np.nan

    oos_idx = s.index[s.index >= pd.Timestamp(OOS_START)]
    if len(oos_idx) >= 20:
        c_oos, p_oos = stats.pearsonr(s.loc[oos_idx], r.loc[oos_idx])
        out["corr_oos"], out["pval_oos"] = float(c_oos), float(p_oos)
    else:
        out["corr_oos"], out["pval_oos"] = np.nan, np.nan
    return out


def verdict(row: dict) -> str:
    p = row.get("pval_t+1"); c = row.get("corr_t+1"); n = row.get("n", 0)
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "INCONCLUSIVE"
    if n < 50:
        return f"INCONCLUSIVE (n={n})"
    if p < 0.05 and abs(c) > 0.15:
        return f"PROMISING (corr={c:+.3f}, p={p:.3f})"
    if p < 0.10:
        return f"WEAK (p={p:.3f})"
    return "REJECT"


def main():
    if not HIST.exists():
        print("history.json missing — run build_model_data.py first.")
        return
    print("Fetching weekly volumes...")
    tsla = weekly_volume("TSLA")
    qqq = weekly_volume("QQQ")

    weekly = pd.concat({"TSLA": tsla, "QQQ": qqq}, axis=1).resample("W-FRI").last().ffill().dropna()
    log_tsla = np.log(weekly["TSLA"])
    log_qqq = np.log(weekly["QQQ"])

    factors = pd.DataFrame(index=weekly.index)
    factors["log_volume"] = log_tsla
    rmean = log_tsla.rolling(52, min_periods=20).mean()
    rstd = log_tsla.rolling(52, min_periods=20).std()
    factors["volume_zscore_52w"] = (log_tsla - rmean) / rstd
    factors["volume_excess"] = residualize(log_tsla, log_qqq)

    resid = load_residuals()["resid_log"]

    rows = [report(c, factors[c], resid) for c in factors.columns]
    for r in rows:
        r["verdict"] = verdict(r)

    print(f"\nResiduals: {len(resid)} weeks")
    print(f"Volumes:   {len(weekly)} weeks")
    print()
    print(f"{'factor':<22}{'n':>5}{'corr_t':>10}{'p':>8}{'corr_t+1':>11}{'p':>8}{'corr_oos':>11}{'p':>8}  verdict")
    print("-" * 115)
    def f(x, fmt="{:>+.3f}"):
        if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
            return fmt.format(x)
        return "  n/a"
    for r in rows:
        print(f"{r['factor']:<22}"
              f"{r['n']:>5}"
              f"{f(r.get('corr_t')):>10}"
              f"{f(r.get('pval_t'), '{:>.3f}'):>8}"
              f"{f(r.get('corr_t+1')):>11}"
              f"{f(r.get('pval_t+1'), '{:>.3f}'):>8}"
              f"{f(r.get('corr_oos')):>11}"
              f"{f(r.get('pval_oos'), '{:>.3f}'):>8}"
              f"  {r['verdict']}")

    print()
    print("Decision rule: integrate only if PROMISING + improves OOS R^2 by >= 2pp")
    print("when added to build_model_data.py.")


if __name__ == "__main__":
    main()

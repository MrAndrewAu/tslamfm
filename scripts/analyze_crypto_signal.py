"""
Re-test crypto (BTC, then ETH) as candidate v7 factors.

Same discipline:
  1. Pull weekly close.
  2. Build candidate transforms (raw log, excess vs QQQ, rolling z-score).
  3. Correlate with v6 residuals contemporaneously and with one-week lag.
  4. Verdict: PROMISING needs p<0.05, |corr|>0.15, n>=50 on lagged corr.

If a transform clears PROMISING -> proceed to walk-forward test.
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


def load_residuals() -> pd.Series:
    rows = json.loads(HIST.read_text())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return np.log(df["actual"]) - np.log(df["fitted"])


def report(name: str, factor: pd.Series, resid: pd.Series) -> dict:
    s = factor.dropna()
    s = s.reindex(resid.index, method="nearest", tolerance=pd.Timedelta("3D")).dropna()
    r = resid.loc[s.index]
    out = {"factor": name, "n": int(len(s))}
    if len(s) < 20:
        return {**out, "status": "insufficient"}
    c0, p0 = stats.pearsonr(s, r)
    out["corr_t"], out["pval_t"] = float(c0), float(p0)
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


def build_factors(sym: str, qqq: pd.Series) -> dict[str, pd.Series]:
    px = wk(sym)
    weekly = pd.concat({sym: px, "QQQ": qqq}, axis=1).resample("W-FRI").last().ffill().dropna()
    log_x = np.log(weekly[sym])
    log_q = np.log(weekly["QQQ"])
    rmean = log_x.rolling(52, min_periods=20).mean()
    rstd = log_x.rolling(52, min_periods=20).std()
    return {
        f"log_{sym}": log_x,
        f"{sym}_excess_vs_QQQ": residualize(log_x, log_q),
        f"{sym}_zscore_52w": (log_x - rmean) / rstd,
    }


def main():
    if not HIST.exists():
        print("history.json missing — run build_model_data.py first.")
        return
    resid = load_residuals()
    qqq = wk("QQQ")

    for sym in ("BTC-USD", "ETH-USD"):
        print(f"\n=== {sym} ===")
        factors = build_factors(sym, qqq)
        rows = []
        for name, s in factors.items():
            r = report(name, s, resid)
            r["verdict"] = verdict(r)
            rows.append(r)
        print(f"{'factor':<28}{'n':>5}{'corr_t':>10}{'p':>8}{'corr_t+1':>11}{'p':>8}{'corr_oos':>11}{'p':>8}  verdict")
        print("-" * 118)
        def f(x, fmt="{:>+.3f}"):
            if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
                return fmt.format(x)
            return "  n/a"
        for r in rows:
            print(f"{r['factor']:<28}"
                  f"{r['n']:>5}"
                  f"{f(r.get('corr_t')):>10}"
                  f"{f(r.get('pval_t'), '{:>.3f}'):>8}"
                  f"{f(r.get('corr_t+1')):>11}"
                  f"{f(r.get('pval_t+1'), '{:>.3f}'):>8}"
                  f"{f(r.get('corr_oos')):>11}"
                  f"{f(r.get('pval_oos'), '{:>.3f}'):>8}"
                  f"  {r['verdict']}")


if __name__ == "__main__":
    main()

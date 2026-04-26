"""
Test microstructure (FINRA off-exchange + short-volume + accumulation/distribution)
factors against v6-canonical-4y model residuals.

Candidate factors (weekly, from daily aggregation):
  short_ratio       -- ShortVolume / FINRA_TotalVolume (within FINRA-reported)
  off_exch_ratio    -- FINRA_TotalVolume / consolidated_volume_yfinance
  off_exch_zscore   -- 52-week z-score of off_exch_ratio (de-trended)
  short_zscore      -- 52-week z-score of short_ratio
  up_down_vol_4w    -- rolling 4w sum(vol on up days) / sum(vol on down days)
                       (Wyckoff-style accumulation/distribution proxy)

Verdict (per project bar): PROMISING needs p<0.05, |corr|>0.15, n>=50 on
LAGGED correlation (factor at t vs residual at t+1).

Run AFTER fetch_finra_volume.py.
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
FINRA = ROOT / "public" / "data" / "finra_volume.csv"
START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"


def load_residuals() -> pd.DataFrame:
    rows = json.loads(HIST.read_text())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def yfin_daily() -> pd.DataFrame:
    h = yf.Ticker("TSLA").history(start=START, end=END, interval="1d")
    idx = pd.to_datetime(h.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    h.index = idx
    return h[["Open", "Close", "Volume"]].rename(columns=str.lower)


def build_weekly_factors() -> pd.DataFrame:
    finra = pd.read_csv(FINRA, index_col="date", parse_dates=["date"]).sort_index()
    px = yfin_daily()

    daily = finra.join(px, how="inner")
    daily["short_ratio"] = daily["short_vol"] / daily["finra_total_vol"]
    daily["off_exch_ratio"] = daily["finra_total_vol"] / daily["volume"]
    # Cap extreme outliers from data oddities (>1 means FINRA reported more
    # than consolidated tape, which happens occasionally on splits/errors).
    daily["off_exch_ratio"] = daily["off_exch_ratio"].clip(upper=1.0)

    # Up/down day classification using close-to-close
    daily["ret"] = daily["close"].pct_change()
    daily["up_vol"] = np.where(daily["ret"] > 0, daily["volume"], 0.0)
    daily["dn_vol"] = np.where(daily["ret"] < 0, daily["volume"], 0.0)

    # Resample to W-FRI: ratios use mean, volumes use sum
    weekly = pd.DataFrame(index=pd.date_range(daily.index.min(), daily.index.max(), freq="W-FRI"))
    weekly["short_ratio"] = daily["short_ratio"].resample("W-FRI").mean()
    weekly["off_exch_ratio"] = daily["off_exch_ratio"].resample("W-FRI").mean()
    up_w = daily["up_vol"].resample("W-FRI").sum()
    dn_w = daily["dn_vol"].resample("W-FRI").sum()
    # 4-week rolling accumulation/distribution
    up4 = up_w.rolling(4, min_periods=2).sum()
    dn4 = dn_w.rolling(4, min_periods=2).sum()
    weekly["up_down_vol_4w"] = up4 / dn4.replace(0, np.nan)

    # 52-week z-scores
    for c in ("short_ratio", "off_exch_ratio"):
        m = weekly[c].rolling(52, min_periods=20).mean()
        s = weekly[c].rolling(52, min_periods=20).std()
        weekly[f"{c}_zscore"] = (weekly[c] - m) / s

    return weekly.dropna(how="all")


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


def main():
    if not HIST.exists():
        print("history.json missing — run build_model_data.py first.")
        return
    if not FINRA.exists():
        print("finra_volume.csv missing — run fetch_finra_volume.py first.")
        return

    resid_df = load_residuals()
    resid = np.log(resid_df["actual"]) - np.log(resid_df["fitted"])

    weekly = build_weekly_factors()

    print(f"Residuals:    {len(resid)} weeks  ({resid.index[0].date()} -> {resid.index[-1].date()})")
    print(f"Microstruct.: {len(weekly)} weeks ({weekly.index[0].date()} -> {weekly.index[-1].date()})\n")

    factors = ["short_ratio", "short_ratio_zscore",
               "off_exch_ratio", "off_exch_ratio_zscore",
               "up_down_vol_4w"]

    rows = []
    for f in factors:
        if f not in weekly.columns:
            continue
        r = report(f, weekly[f], resid)
        r["verdict"] = verdict(r)
        rows.append(r)

    print(f"{'factor':<25}{'n':>5}{'corr_t':>10}{'p':>8}{'corr_t+1':>11}{'p':>8}{'corr_oos':>11}{'p':>8}  verdict")
    print("-" * 118)
    def f(x, fmt="{:>+.3f}"):
        if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
            return fmt.format(x)
        return "  n/a"
    for r in rows:
        print(f"{r['factor']:<25}"
              f"{r['n']:>5}"
              f"{f(r.get('corr_t')):>10}"
              f"{f(r.get('pval_t'), '{:>.3f}'):>8}"
              f"{f(r.get('corr_t+1')):>11}"
              f"{f(r.get('pval_t+1'), '{:>.3f}'):>8}"
              f"{f(r.get('corr_oos')):>11}"
              f"{f(r.get('pval_oos'), '{:>.3f}'):>8}"
              f"  {r['verdict']}")

    print()
    print("Decision: integrate only if PROMISING + improves OOS R^2 by >= 2pp")
    print("when added to build_model_data.py.")


if __name__ == "__main__":
    main()

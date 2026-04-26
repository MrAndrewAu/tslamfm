"""
Test whether option-chain features add real predictive power to the
v6-canonical-4y TSLA model — BEFORE committing to a paid Theta data plan.

Procedure:
  1. Load existing model output (history.json) to get weekly fitted values
     and residuals (actual - fitted, in log space).
  2. Load options_features.parquet from fetch_theta_options.py.
  3. For each candidate factor (skew_25d, iv_atm_30d, pc_oi_ratio,
     pc_vol_ratio, term_slope), report:
        - corr(factor_t, residual_t)            -- contemporaneous fit
        - corr(factor_t, residual_{t+1})        -- predictive (the only
                                                   one that matters)
        - delta-OOS-R2 if added to the regression on a walk-forward basis
  4. Print a recommendation: integrate / skip / inconclusive.

A factor only earns inclusion if BOTH:
  - lagged correlation with future residual is statistically significant
    (|t| > 2, n > 50)
  - walk-forward OOS R² strictly improves vs current 0.66 baseline by
    at least 2 percentage points (else it's noise dressed as signal).

Run AFTER fetch_theta_options.py has produced the parquet.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "public" / "data" / "history.json"
OPTS = ROOT / "public" / "data" / "options_features.csv"

# Walk-forward OOS start (matches build_model_data.py)
OOS_START = "2025-01-03"


def load_residuals() -> pd.DataFrame:
    """Returns DataFrame indexed by date with columns: actual, fitted, resid_log."""
    rows = json.loads(HIST.read_text())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["resid_log"] = np.log(df["actual"]) - np.log(df["fitted"])
    return df[["actual", "fitted", "resid_log"]]


def align(resid: pd.DataFrame, opts: pd.DataFrame) -> pd.DataFrame:
    """Reindex options to weekly Friday and merge."""
    o = opts.copy()
    o.index = pd.to_datetime(o.index)
    # Forward-fill to nearest Friday (markets close Fri, options EOD Fri)
    j = resid.join(o.reindex(resid.index, method="nearest", tolerance="3D"))
    return j.dropna(subset=["resid_log"])


def report_factor(name: str, df: pd.DataFrame) -> dict:
    """Contemporaneous and 1-week-ahead correlations with residual."""
    s = df[name].dropna()
    r = df.loc[s.index, "resid_log"]

    out = {"factor": name, "n": int(len(s))}

    if len(s) < 20:
        out["status"] = "insufficient data"
        return out

    # Contemporaneous
    c0, p0 = stats.pearsonr(s, r)
    out["corr_t"] = float(c0)
    out["pval_t"] = float(p0)

    # Predictive: factor at t vs residual at t+1
    s_aligned = s.iloc[:-1]
    r_next    = df["resid_log"].reindex(s.index).shift(-1).iloc[:-1]
    mask = s_aligned.notna() & r_next.notna()
    if mask.sum() >= 20:
        c1, p1 = stats.pearsonr(s_aligned[mask], r_next[mask])
        out["corr_t+1"] = float(c1)
        out["pval_t+1"] = float(p1)
    else:
        out["corr_t+1"] = np.nan
        out["pval_t+1"] = np.nan

    # OOS-period stats only
    oos = df[df.index >= OOS_START]
    s_oos = oos[name].dropna()
    if len(s_oos) >= 20:
        r_oos = oos.loc[s_oos.index, "resid_log"]
        c_oos, p_oos = stats.pearsonr(s_oos, r_oos)
        out["corr_oos"] = float(c_oos)
        out["pval_oos"] = float(p_oos)
    else:
        out["corr_oos"] = np.nan
        out["pval_oos"] = np.nan

    return out


def verdict(stats_row: dict) -> str:
    p = stats_row.get("pval_t+1")
    c = stats_row.get("corr_t+1")
    n = stats_row.get("n", 0)
    if p is None or np.isnan(p):
        return "INCONCLUSIVE — too little data"
    if n < 50:
        return f"INCONCLUSIVE — only {n} obs"
    if p < 0.05 and abs(c) > 0.15:
        return f"PROMISING — predictive corr {c:+.3f}, p={p:.3f}"
    if p < 0.10:
        return f"WEAK — borderline (p={p:.3f}). Need more data."
    return "REJECT — no predictive signal"


def main():
    if not HIST.exists():
        print("history.json missing — run build_model_data.py first.")
        return
    if not OPTS.exists():
        print("options_features.csv missing — run fetch_marketdata_options.py first.")
        return

    resid = load_residuals()
    opts = pd.read_csv(OPTS, index_col="date", parse_dates=["date"])
    df = align(resid, opts)

    print(f"Residuals: {len(resid)} weeks  ({resid.index[0].date()} -> {resid.index[-1].date()})")
    print(f"Options:   {len(opts)} weeks  ({opts.index[0].date()} -> {opts.index[-1].date()})")
    print(f"Joined:    {df.dropna(subset=opts.columns.tolist()).shape[0]} usable rows")
    print()

    candidates = [c for c in opts.columns if c not in ("spot",)]
    results = []
    for c in candidates:
        if c not in df.columns:
            continue
        r = report_factor(c, df)
        r["verdict"] = verdict(r)
        results.append(r)

    print(f"{'factor':<14}{'n':>5}{'corr_t':>10}{'p':>8}{'corr_t+1':>11}{'p':>8}{'corr_oos':>11}{'p':>8}  verdict")
    print("-" * 110)
    for r in results:
        def f(x, fmt="{:>+.3f}"):
            return fmt.format(x) if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else "  n/a"
        print(f"{r['factor']:<14}"
              f"{r['n']:>5}"
              f"{f(r.get('corr_t')):>10}"
              f"{f(r.get('pval_t'), '{:>.3f}'):>8}"
              f"{f(r.get('corr_t+1')):>11}"
              f"{f(r.get('pval_t+1'), '{:>.3f}'):>8}"
              f"{f(r.get('corr_oos')):>11}"
              f"{f(r.get('pval_oos'), '{:>.3f}'):>8}"
              f"  {r['verdict']}")

    print()
    print("Decision rule: only integrate factors with PROMISING verdict")
    print("AND verify they improve OOS R² >= 2pp in build_model_data.py.")


if __name__ == "__main__":
    main()

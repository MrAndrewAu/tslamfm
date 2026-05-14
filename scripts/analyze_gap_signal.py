#!/usr/bin/env python3
"""
analyze_gap_signal.py
=====================
Tests whether the v6.6 fair-value gap  (actual − fitted) / fitted
predicts mean reversion in TSLA over the next 4, 8, and 13 weeks.

Signal hypothesis:
  gap > 0  (actual > fair value)  →  stock stretched above fundamentals  →  downside risk
  gap < 0  (actual < fair value)  →  stock beaten below fundamentals    →  upside potential

NOTE on IS vs OOS:
  In-sample (pre-2025-01-03): 'fitted' is a full-sample OLS prediction; the model
  was estimated on these very prices, so IS gaps are biased toward zero and
  autocorrelation tests on them are not honest forward-prediction tests.
  OOS (2025-01-03+): genuinely walk-forward predictions — these are the honest test.

Run from repo root:
    $env:PYTHONIOENCODING='utf-8'
    python scripts/analyze_gap_signal.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── config ────────────────────────────────────────────────────────────────────

HISTORY_PATH = Path(__file__).parent.parent / "public" / "data" / "history.json"
OOS_START    = "2025-01-03"  # first walk-forward prediction date

HORIZONS = [4, 8, 13]       # weeks ahead for forward-return tests

# Signal tiers: (display_label, gap_lo_exclusive, gap_hi_inclusive)
# Sorted extreme-positive → extreme-negative for display
TIERS = [
    ("VERY STRETCHED ABOVE  gap > +25%",     0.25,  99.0),
    ("STRETCHED ABOVE       gap +15% – +25%", 0.15,  0.25),
    ("ELEVATED              gap  +5% – +15%", 0.05,  0.15),
    ("NEAR FAIR VALUE       gap  ±5%",        -0.05,  0.05),
    ("DEPRESSED             gap  -5% – -15%", -0.15, -0.05),
    ("STRETCHED BELOW       gap -15% – -25%", -0.25, -0.15),
    ("VERY STRETCHED BELOW  gap < -25%",      -99.0, -0.25),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    with open(HISTORY_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if "partial_week" in df.columns:
        df = df[~df["partial_week"].fillna(False).astype(bool)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["gap"] = (df["actual"] - df["fitted"]) / df["fitted"]
    df["oos"]  = df["date"] >= OOS_START
    return df


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        df[f"fwd_{h}w"] = df["actual"].shift(-h) / df["actual"] - 1
    return df


def pearson(x: np.ndarray, y: np.ndarray):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 6:
        return n, np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    return n, r, p


def spearman(x: np.ndarray, y: np.ndarray):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 6:
        return n, np.nan, np.nan
    r, p = stats.spearmanr(x, y)
    return n, r, p


def verdict(r, p) -> str:
    if np.isnan(r):
        return "n/a"
    if r < -0.15 and p < 0.05:
        return "SIGNAL ✓ (mean reversion)"
    if r < -0.10 and p < 0.10:
        return "signal ~ (borderline)"
    if r < -0.05:
        return f"weak negative (p={p:.2f})"
    if r > 0.10 and p < 0.10:
        return "MOMENTUM (price chasing)"
    return "noise"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_history()
    df = add_forward_returns(df)

    oos = df[df["oos"]].copy()
    ins = df[~df["oos"]].copy()

    sep = "─" * 68

    print("=" * 68)
    print("  TSLA v6.6  Gap Signal — Mean Reversion Backtest")
    print("=" * 68)
    print(f"\n  Total weekly rows (ex partial):   {len(df)}")
    print(f"  In-sample  (pre {OOS_START}):  {len(ins)}"
          f"  ← fitted from full-sample OLS; gaps biased small")
    print(f"  OOS (walk-forward, {OOS_START}+): {len(oos)}"
          f"  ← honest out-of-sample predictions")

    # ── gap distribution ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Gap Distribution  (gap = (actual − fair_value) / fair_value)")
    print(sep)
    for label, sub in [("In-sample (IS)", ins), ("OOS (walk-forward)", oos)]:
        g = sub["gap"] * 100
        q = np.percentile(g, [10, 25, 50, 75, 90])
        print(f"\n  {label}  n={len(sub)}")
        print(f"    mean={g.mean():+.1f}%  std={g.std():.1f}%"
              f"  min={g.min():+.1f}%  max={g.max():+.1f}%")
        print(f"    p10={q[0]:+.1f}%  p25={q[1]:+.1f}%  "
              f"median={q[2]:+.1f}%  p75={q[3]:+.1f}%  p90={q[4]:+.1f}%")

    # ── current reading ───────────────────────────────────────────────────────
    latest = df.dropna(subset=["actual", "fitted"]).iloc[-1]
    curr_gap = latest["gap"] * 100
    curr_tier = "NEAR FAIR VALUE"
    for lbl, lo, hi in TIERS:
        if lo < latest["gap"] <= hi:
            curr_tier = lbl.strip()
            break
    print(f"\n{sep}")
    print("  Current Reading")
    print(sep)
    print(f"  Date:        {latest['date'].date()}")
    print(f"  Actual:     ${latest['actual']:.2f}")
    print(f"  Fair value: ${latest['fitted']:.2f}")
    print(f"  Gap:         {curr_gap:+.1f}%")
    print(f"  Signal tier: {curr_tier}")

    # ── correlation tests ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Pearson  r(gap_t, fwd_return_t+N)")
    print("  Negative r supports mean-reversion hypothesis.")
    print(sep)
    header = f"  {'Sample':<22}  {'H':>4}  {'n':>4}  {'r':>7}  {'p':>6}  verdict"
    print(header)
    print(f"  {'-'*22}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*26}")
    for sub, lbl in [(ins, "IS (biased)"), (oos, "OOS (honest)"), (df, "Full sample")]:
        for h in HORIZONS:
            n, r, p = pearson(sub["gap"].values, sub[f"fwd_{h}w"].values)
            r_s = f"{r:+.3f}" if not np.isnan(r) else "    —"
            p_s = f"{p:.3f}" if not np.isnan(p) else "    —"
            print(f"  {lbl:<22}  {h:>3}w  {n:>4}  {r_s:>7}  {p_s:>6}  {verdict(r, p)}")
        print()

    print(f"\n{sep}")
    print("  Spearman  ρ(gap_t, fwd_return_t+N)  (rank-based, robust to outliers)")
    print(sep)
    print(header.replace("r(", "ρ(").replace("  r  ", "  ρ  "))
    print(f"  {'-'*22}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*26}")
    for sub, lbl in [(ins, "IS (biased)"), (oos, "OOS (honest)"), (df, "Full sample")]:
        for h in HORIZONS:
            n, r, p = spearman(sub["gap"].values, sub[f"fwd_{h}w"].values)
            r_s = f"{r:+.3f}" if not np.isnan(r) else "    —"
            p_s = f"{p:.3f}" if not np.isnan(p) else "    —"
            print(f"  {lbl:<22}  {h:>3}w  {n:>4}  {r_s:>7}  {p_s:>6}  {verdict(r, p)}")
        print()

    # ── signal tier hit rates ─────────────────────────────────────────────────
    for dataset, ds_label in [(df, "Full Sample (IS+OOS)"), (oos, "OOS Only (walk-forward)")]:
        print(f"\n{sep}")
        print(f"  Signal Tier Hit Rates — {ds_label}  (n={len(dataset)})")
        print(f"  'hit%' = % of observations where gap correctly predicted reversion direction")
        print(sep)
        for h in HORIZONS:
            col = f"fwd_{h}w"
            print(f"\n  {h}-week forward return")
            print(f"  {'Tier':<47}  {'n':>4}  {'med_fwd':>8}  {'hit%':>6}  {'mean_fwd':>9}")
            print(f"  {'-'*47}  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*9}")
            for label, lo, hi in TIERS:
                sub = dataset[(dataset["gap"] > lo) & (dataset["gap"] <= hi)
                              & dataset[col].notna()]
                n = len(sub)
                if n == 0:
                    print(f"  {label:<47}  {n:>4}  {'—':>8}  {'—':>6}  {'—':>9}")
                    continue
                fwd = sub[col] * 100
                med = fwd.median()
                mean = fwd.mean()
                mid = (lo + hi) / 2
                if mid > 0.001:          # stretched above → expect negative return
                    hit = (fwd < 0).mean() * 100
                elif mid < -0.001:       # stretched below → expect positive return
                    hit = (fwd > 0).mean() * 100
                else:
                    hit = np.nan
                hit_s = f"{hit:.0f}%" if not np.isnan(hit) else "  —"
                print(f"  {label:<47}  {n:>4}  {med:>+7.1f}%  {hit_s:>6}  {mean:>+8.1f}%")

    # ── gap persistence (autocorrelation) ─────────────────────────────────────
    print(f"\n{sep}")
    print("  Gap Persistence — OOS autocorrelation")
    print("  How long does a stretched reading typically persist?")
    print(sep)
    oos_g = oos["gap"].dropna().values
    print(f"  {'lag':>5}  {'r':>7}  {'p':>7}  note")
    for lag in [1, 2, 4, 8]:
        if len(oos_g) > lag + 2:
            n, r, p = pearson(oos_g[:-lag], oos_g[lag:])
            note = "persistent" if r > 0.5 else ("moderate" if r > 0.25 else "weak")
            print(f"  {lag:>3}w   {r:>+.3f}  {p:>7.3f}  {note}")

    # ── extreme readings with what happened next ──────────────────────────────
    print(f"\n{sep}")
    print("  Extreme Readings  |gap| > 15%  — what happened next?")
    print(sep)
    extreme = df[df["gap"].abs() > 0.15].copy().sort_values("gap", ascending=False)
    if len(extreme) == 0:
        print("  No readings with |gap| > 15% found.")
    else:
        print(f"  {'Date':<12}  {'gap':>7}  {'actual':>8}  {'fair_val':>8}  "
              f"{'4w':>7}  {'8w':>7}  {'13w':>7}  sample")
        print(f"  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*8}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
        for _, row in extreme.iterrows():
            tag = "OOS" if row["oos"] else "IS"
            def fmt(v):
                return f"{v*100:>+6.1f}%" if not np.isnan(v) else "     —"
            print(f"  {str(row['date'].date()):<12}  {row['gap']*100:>+6.1f}%  "
                  f"${row['actual']:>7.0f}  ${row['fitted']:>7.0f}  "
                  f"{fmt(row.get('fwd_4w',  np.nan))}  "
                  f"{fmt(row.get('fwd_8w',  np.nan))}  "
                  f"{fmt(row.get('fwd_13w', np.nan))}  {tag}")

    # ── key observations ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Key Questions for the Reframe Decision")
    print(sep)
    print("""
  1. Is r(gap, fwd_return) consistently negative across horizons?
       → Yes  = gap is a mean-reversion signal (proceed with reframe)
       → No   = current fair value is not a reliable reversal anchor

  2. Are the OOS and full-sample signs consistent?
       → Same sign = regime-stable signal (more credible)
       → Opposite  = regime-dependent (do not trade on it)

  3. What gap size is needed for a useful signal?
       → Look at hit% in the ±15%, ±25% tiers
       → A hit% > 60% at |gap| > 15% with n ≥ 10 is a reasonable minimum

  4. How long does a gap persist before mean-reverting?
       → Autocorrelation lag-1 tells you whether to watch weekly or quarterly

  HONEST CAVEAT: with n≈69 OOS rows, individual p-values are weak.
  Focus on directional consistency (sign stable across horizons and samples)
  and empirical hit rates rather than statistical significance alone.
""")


if __name__ == "__main__":
    main()

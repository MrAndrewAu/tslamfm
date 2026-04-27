"""
Adaptive sigma_t probe (v6.2 baseline).

Currently history.json bands use a single constant sigma = sqrt(SSR/(n-k))
from in-sample residuals. This is miscalibrated across volatility regimes:
during high-VIX periods the true uncertainty is wider; during calm periods
it's tighter. We measure that miscalibration and propose an adaptive sigma_t.

Candidates (all lookahead-free in walk-forward):
  C0  constant sigma                           -- current baseline
  C1  sigma_t = a + b * log(VIX_t)            -- externally-driven
  C2  sigma_t = rolling_std_26w(past residuals)  -- empirical
  C3  EWMA: s_t^2 = lambda*s_{t-1}^2 + (1-l)*e_{t-1}^2 (lambda=0.94)
        RiskMetrics standard

Scoring (OOS only, walk-forward):
  - Coverage @ +/-1 sigma  (target 0.683)
  - Coverage @ +/-2 sigma  (target 0.954)
  - Mean |residual|/sigma  (target 0.798 if Gaussian)
  - Mean log predictive density under Gaussian (higher = better)
  - Pinball loss equivalents

Acceptance: improves at least 2 of (coverage error, mean-|z|, log-pdf) vs C0
without making the others materially worse.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s

def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)

# ---------- assemble v6.2 frame ----------
print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])
log_rbob = np.log(df["RBOB"])
f["RBOB_zscore_52w"] = (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) / log_rbob.rolling(52, min_periods=20).std()
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = (curve_log - curve_log.rolling(52, min_periods=20).mean()) / curve_log.rolling(52, min_periods=20).std()

EVENT_DEFS = [
    ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion", "2020-11-16"),
    ("Hertz_1T_peak", "2021-10-25"),      ("Twitter_overhang", "2022-04-25"),
    ("Twitter_close", "2022-10-27"),      ("AI_day_2023", "2023-07-19"),
    ("Trump_election", "2024-11-06"),     ("DOGE_brand_damage", "2025-02-15"),
    ("Musk_exits_DOGE", "2025-04-22"),    ("TrillionPay", "2025-09-05"),
    ("Tariff_shock", "2026-02-01"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"  frame n={len(f)}  {f.index[0].date()} -> {f.index[-1].date()}")

active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V62 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
       "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_events

# ---------- walk-forward fit + collect (resid, vix, in-sample sigma) per week ----------
print("Walk-forward to collect per-week residuals + sigma proxies...")
records = []
for date in f.loc[OOS_START:].index:
    train = f.loc[:date].iloc[:-1]
    if len(train) < 100: continue
    Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in V62])
    yt = train["log_TSLA"].to_numpy()
    bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)

    # In-sample residuals on train (used for adaptive sigma calibration)
    resid_train_log = yt - Xt @ bt
    n_train, k_train = Xt.shape
    sigma_const = float(np.sqrt((resid_train_log @ resid_train_log) / (n_train - k_train)))

    # Predict next week
    row = f.loc[date]
    xv = np.array([1.0] + [float(row[c]) for c in V62])
    yhat_log = float(xv @ bt)
    actual_log = float(row["log_TSLA"])
    resid_log = actual_log - yhat_log

    records.append({
        "date": date,
        "actual_log": actual_log,
        "fitted_log": yhat_log,
        "resid_log": resid_log,
        "log_VIX": float(row["log_VIX"]),
        "sigma_const": sigma_const,
        "train_resid_log": resid_train_log,  # for rolling/EWMA calibration
        "train_log_vix": train["log_VIX"].to_numpy(),
    })

n_oos = len(records)
print(f"  collected {n_oos} OOS predictions")

# ---------- score each candidate ----------
ROLL_WIN = 26
EWMA_LAMBDA = 0.94

def candidate_C0(rec):  # constant sigma from train
    return rec["sigma_const"]

def candidate_C1(rec):
    """sigma_t = a + b * log(VIX_t), fit on train residuals' |resid| or sqrt(resid^2)."""
    e2 = rec["train_resid_log"] ** 2
    lv = rec["train_log_vix"]
    # Use last len(e2) elements of lv (they line up: train_resid_log was computed
    # from the same train rows that produced lv).
    lv = lv[-len(e2):]
    # Regress log(e^2 + eps) on log_VIX (log-link to keep sigma>0)
    eps = 1e-8
    y = np.log(e2 + eps)
    X = np.column_stack([np.ones(len(lv)), lv])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    log_sigma2_t = coef[0] + coef[1] * rec["log_VIX"]
    s = float(np.sqrt(np.exp(log_sigma2_t)))
    return max(s, 1e-4)

def candidate_C2(rec):
    """sigma_t = std of last ROLL_WIN train residuals (lookahead-free)."""
    e = rec["train_resid_log"][-ROLL_WIN:]
    if len(e) < 8: return rec["sigma_const"]
    return float(np.std(e, ddof=1))

def candidate_C3(rec):
    """EWMA sigma_t (lambda=0.94 RiskMetrics)."""
    e = rec["train_resid_log"]
    s2 = float(np.var(e[:26], ddof=1)) if len(e) >= 26 else float(np.var(e, ddof=1))
    for ei in e[26:]:
        s2 = EWMA_LAMBDA * s2 + (1 - EWMA_LAMBDA) * (ei ** 2)
    return float(np.sqrt(max(s2, 1e-8)))

CANDIDATES = {
    "C0_constant":      candidate_C0,
    "C1_VIX_regressed": candidate_C1,
    "C2_rolling26w":    candidate_C2,
    "C3_EWMA_l0.94":    candidate_C3,
}

print(f"\nScoring {n_oos} OOS weeks across {len(CANDIDATES)} sigma candidates...\n")
print(f"  {'candidate':<22}{'mean_sigma':>12}{'cov_1s':>10}{'cov_2s':>10}{'mean|z|':>10}"
      f"{'mean_logpdf':>14}{'crps_proxy':>12}")
print("-" * 92)

for name, fn in CANDIDATES.items():
    sigmas = np.array([fn(r) for r in records])
    resids = np.array([r["resid_log"] for r in records])
    z = resids / sigmas
    # Coverage
    cov_1s = float(np.mean(np.abs(z) <= 1.0))
    cov_2s = float(np.mean(np.abs(z) <= 2.0))
    mean_abs_z = float(np.mean(np.abs(z)))
    # Log predictive density (Gaussian)
    logpdf = -0.5 * np.log(2 * np.pi) - np.log(sigmas) - 0.5 * (resids ** 2) / (sigmas ** 2)
    mean_logpdf = float(np.mean(logpdf))
    # CRPS proxy: E|Z| * sigma where Z~N(0,1) implies 0.7979*sigma; we use |resid| / sigma * sigma = |resid| as no-info baseline; actual CRPS for Gaussian is sigma*(z*(2*Phi(z)-1)+2*phi(z)-1/sqrt(pi))
    from scipy.stats import norm
    crps = sigmas * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    mean_crps = float(np.mean(crps))
    print(f"  {name:<22}{np.mean(sigmas):>12.4f}{cov_1s:>10.3f}{cov_2s:>10.3f}{mean_abs_z:>10.3f}"
          f"{mean_logpdf:>14.4f}{mean_crps:>12.5f}")

print(f"\n  Targets:  cov_1s=0.683  cov_2s=0.954  mean|z|=0.798  (for calibrated Gaussian)")
print(f"  Higher mean_logpdf better; lower crps_proxy better.")

# ---------- in-sample VIX -> volatility relationship ----------
# fit on full sample to characterize the regression coef and check stability
all_resids = np.concatenate([np.array([r["resid_log"] for r in records])])
all_lvix = np.array([r["log_VIX"] for r in records])
print(f"\n  In-OOS-sample diagnostic: corr(|resid|, log_VIX) = {np.corrcoef(np.abs(all_resids), all_lvix)[0,1]:+.3f}")
print(f"  In-OOS-sample diagnostic: corr(resid^2,  log_VIX) = {np.corrcoef(all_resids**2, all_lvix)[0,1]:+.3f}")

"""
Structural swap test: SPY (S&P 500) vs QQQ (Nasdaq-100) as the
tech-market beta anchor in the v6.4 factor set.

QQQ = Nasdaq-100 (~60% tech, growth-tilted, TSLA ~2% weight at peak).
SPY = S&P 500 (broad market, ~30% tech, more value/defensive mix).

Hypotheses:
  H1 — SPY is a "purer" market-beta anchor because it contains all sectors;
       the tech-specific premium is then entirely absorbed by NVDA_excess
       and ARKK_excess residualized against SPY.
  H2 — The QQQ premium over SPY (Nasdaq growth tilt) is itself a signal for
       TSLA's speculative multiple; stripping it out via residualization
       could degrade fit.

Three variants tested (same framework as analyze_xlk_vs_qqq.py):
  A. Drop-in swap: log_SPY replaces log_QQQ; NVDA/ARKK still on QQQ.
  B. Full swap: log_SPY replaces log_QQQ AND NVDA/ARKK residualized on SPY.
  C. Additive: log_QQQ + log_SPY both retained (checks orthogonal info).

Acceptance bar: +2pp walk-forward OOS R² lift vs v6.4 baseline.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
MIN_TRAIN   = 100
EVENT_P_THR = 0.10
BARRIER_PP  = 2.0

# ── helpers ───────────────────────────────────────────────────────────────────

def wk(sym):
    candidates = []
    try:
        s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
        candidates.append(s)
    except Exception:
        pass
    try:
        dl = yf.download(sym, start=START, end=END, interval="1wk",
                         progress=False, auto_adjust=False)
        if not dl.empty and "Close" in dl.columns:
            candidates.append(dl["Close"])
    except Exception:
        pass
    for s in candidates:
        s = s.dropna()
        if s.empty:
            continue
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx
        return s.resample("W-FRI").last()
    raise RuntimeError(f"Failed to fetch weekly close for {sym}")


def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


def fit(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    r = y - yhat
    n, k = X.shape
    s2 = (r @ r) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss_tot
    return dict(beta=b, p=p, r2=float(r2),
                fitted=pd.Series(yhat, index=frame.index),
                factors=list(factors))


def select_events(frame, forced, candidate_events, p_thr=EVENT_P_THR):
    events = [e for e in candidate_events if frame[e].nunique() > 1]
    factors = forced + events
    m = fit(frame, factors)
    while events:
        event_ps = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst, worst_p = max(event_ps, key=lambda x: x[1])
        if worst_p <= p_thr:
            break
        events.remove(worst)
        factors = forced + events
        m = fit(frame, factors)
    return factors, events, m


def walk_forward(frame, forced, candidate_events, min_train=MIN_TRAIN):
    pred_log = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, _, m_tr = select_events(train, forced, candidate_events)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred_log.loc[date] = float(xv @ m_tr["beta"])
    return pred_log


def oos_metrics(frame, pred_log):
    idx = pred_log.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    pred   = np.exp(pred_log.loc[idx_oos].to_numpy())
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2  = 1 - ((actual - pred) ** 2).sum() / ss_tot
    mae = np.mean(np.abs(actual - pred)) / np.mean(actual) * 100
    return float(r2), float(mae)


# ── fetch data ────────────────────────────────────────────────────────────────

print("=" * 65)
print("SPY vs QQQ as market-beta anchor — structural swap test")
print("=" * 65)
print("\nFetching weekly closes...")

S = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "SPY": "SPY",
    "DXY":  "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK",
    "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()
print(f"  {len(df)} weeks  {df.index[0].date()} → {df.index[-1].date()}")

# ── correlation snapshot ──────────────────────────────────────────────────────
log_qqq  = np.log(df["QQQ"])
log_spy  = np.log(df["SPY"])
log_tsla = np.log(df["TSLA"])
corr_qqq_spy   = float(np.corrcoef(log_qqq, log_spy)[0, 1])
corr_tsla_qqq  = float(np.corrcoef(log_tsla, log_qqq)[0, 1])
corr_tsla_spy  = float(np.corrcoef(log_tsla, log_spy)[0, 1])
spy_excess      = residualize(log_spy, log_qqq)
qqq_excess      = residualize(log_qqq, log_spy)
corr_tsla_spy_ex = float(np.corrcoef(log_tsla, spy_excess)[0, 1])
corr_tsla_qqq_ex = float(np.corrcoef(log_tsla, qqq_excess)[0, 1])

# VIF
X_vif = np.column_stack([np.ones(len(df)), log_qqq.to_numpy()])
b_vif, *_ = np.linalg.lstsq(X_vif, log_spy.to_numpy(), rcond=None)
r2_vif = 1 - ((log_spy.to_numpy() - X_vif @ b_vif) ** 2).sum() / \
             ((log_spy.to_numpy() - log_spy.mean()) ** 2).sum()
vif = 1 / (1 - r2_vif) if r2_vif < 1 else float("inf")

print(f"\nCorrelation snapshot (log levels, n={len(df)}):")
print(f"  corr(log_QQQ, log_SPY)        = {corr_qqq_spy:.4f}   VIF={vif:.1f}")
print(f"  corr(log_TSLA, log_QQQ)       = {corr_tsla_qqq:.4f}")
print(f"  corr(log_TSLA, log_SPY)       = {corr_tsla_spy:.4f}")
print(f"  corr(log_TSLA, SPY_excess_QQQ)= {corr_tsla_spy_ex:.4f}  (unique SPY info beyond QQQ)")
print(f"  corr(log_TSLA, QQQ_excess_SPY)= {corr_tsla_qqq_ex:.4f}  (unique QQQ info beyond SPY)")

# ── build factor frame ────────────────────────────────────────────────────────

base = pd.DataFrame(index=df.index)
base["log_TSLA"] = np.log(df["TSLA"])
base["log_QQQ"]  = np.log(df["QQQ"])
base["log_SPY"]  = np.log(df["SPY"])
base["log_DXY"]  = np.log(df["DXY"])
base["log_VIX"]  = np.log(df["VIX"])

base["NVDA_excess_QQQ"] = residualize(np.log(df["NVDA"]), base["log_QQQ"])
base["ARKK_excess_QQQ"] = residualize(np.log(df["ARKK"]), base["log_QQQ"])
base["NVDA_excess_SPY"] = residualize(np.log(df["NVDA"]), base["log_SPY"])
base["ARKK_excess_SPY"] = residualize(np.log(df["ARKK"]), base["log_SPY"])

log_rbob = np.log(df["RBOB"])
base["RBOB_zscore_52w"] = (
    (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) /
    log_rbob.rolling(52, min_periods=20).std()
)
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
base["curve_IEF_SHY_zscore_52w"] = (
    (curve_log - curve_log.rolling(52, min_periods=20).mean()) /
    curve_log.rolling(52, min_periods=20).std()
)

EVENT_DEFS = [
    ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion",   "2020-11-16"),
    ("Hertz_1T_peak",      "2021-10-25"), ("Twitter_overhang",  "2022-04-25"),
    ("Twitter_close",      "2022-10-27"), ("AI_day_2023",       "2023-07-19"),
    ("Trump_election",     "2024-11-06"), ("DOGE_brand_damage", "2025-02-15"),
    ("Musk_exits_DOGE",    "2025-04-22"), ("TrillionPay",       "2025-09-05"),
    ("Tariff_shock",       "2026-02-01"), ("Robotaxi_Austin",   "2025-06-22"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    base[f"E_{name}"] = ((base.index >= d0) &
                          (base.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = base.dropna(subset=[
    "log_TSLA", "log_QQQ", "log_SPY", "log_DXY", "log_VIX",
    "NVDA_excess_QQQ", "ARKK_excess_QQQ",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w",
])
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
n_is  = (f.index < pd.Timestamp(OOS_START)).sum()
n_oos = (f.index >= pd.Timestamp(OOS_START)).sum()
print(f"\nWorking frame: {len(f)} weeks  IS={n_is}  OOS={n_oos}")

# ── define variants ───────────────────────────────────────────────────────────

FORCED_V64 = ["log_QQQ", "log_DXY", "log_VIX",
              "NVDA_excess_QQQ", "ARKK_excess_QQQ",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

FORCED_A   = ["log_SPY", "log_DXY", "log_VIX",
              "NVDA_excess_QQQ", "ARKK_excess_QQQ",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

FORCED_B   = ["log_SPY", "log_DXY", "log_VIX",
              "NVDA_excess_SPY", "ARKK_excess_SPY",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

FORCED_C   = ["log_QQQ", "log_SPY", "log_DXY", "log_VIX",
              "NVDA_excess_QQQ", "ARKK_excess_QQQ",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]

variants = {
    "v6.4  (QQQ anchor, baseline)": FORCED_V64,
    "Var A (SPY drop-in)":          FORCED_A,
    "Var B (SPY full swap)":        FORCED_B,
    "Var C (QQQ + SPY additive)":   FORCED_C,
}

# ── IS diagnostics ────────────────────────────────────────────────────────────

print("\n[IS diagnostics]")
print(f"  {'variant':<38} {'IS R²':>7} {'events':>7}")
for label, forced in variants.items():
    facs, events, m = select_events(f, forced, active_events)
    print(f"  {label:<38} {m['r2']:>7.4f} {len(events):>7}")

# ── walk-forward OOS ──────────────────────────────────────────────────────────

print("\n[Walk-forward OOS]  (this may take ~60 seconds)...")
print(f"  {'variant':<38} {'OOS R²':>7} {'OOS MAE':>9} {'Δ vs v64':>10}  verdict")

results = {}
for label, forced in variants.items():
    pred = walk_forward(f, forced, active_events)
    r2, mae = oos_metrics(f, pred)
    results[label] = (r2, mae)
    if "baseline" in label:
        base_r2 = r2

for label, (r2, mae) in results.items():
    delta = (r2 - base_r2) * 100
    verdict = "(ref)" if "baseline" in label else ("ACCEPT" if delta >= BARRIER_PP else "REJECT")
    print(f"  {label:<38} {r2:>7.4f} {mae:>8.2f}%  {delta:>+9.2f}pp  {verdict}")

# ── coefficient stability for Var B (SPY full swap) ──────────────────────────

print("\n[Coefficient stability: beta_SPY vs beta_QQQ across rolling windows]")
print(f"  {'window':>8}  {'beta_SPY':>10}  {'p_SPY':>8}  {'beta_QQQ(v64)':>14}  {'p_QQQ':>8}")
for w in [80, 100, 120, 150, len(f)]:
    sub = f.iloc[-w:] if w < len(f) else f
    facs_b, _, m_b = select_events(sub, FORCED_B, active_events)
    facs_v, _, m_v = select_events(sub, FORCED_V64, active_events)
    b_spy = m_b["beta"][facs_b.index("log_SPY") + 1] if "log_SPY" in facs_b else np.nan
    p_spy = m_b["p"][facs_b.index("log_SPY") + 1]   if "log_SPY" in facs_b else np.nan
    b_qqq = m_v["beta"][facs_v.index("log_QQQ") + 1] if "log_QQQ" in facs_v else np.nan
    p_qqq = m_v["p"][facs_v.index("log_QQQ") + 1]   if "log_QQQ" in facs_v else np.nan
    label = "full" if w == len(f) else f"last {w}w"
    print(f"  {label:>8}  {b_spy:>10.4f}  {p_spy:>8.4f}  {b_qqq:>14.4f}  {p_qqq:>8.4f}")

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
base_r2_val, base_mae_val = results["v6.4  (QQQ anchor, baseline)"]
print(f"  v6.4 baseline: OOS R²={base_r2_val:.4f}  MAE={base_mae_val:.2f}%")
print(f"  Acceptance bar: +{BARRIER_PP}pp OOS R² lift")
print()
print("  Key diagnostic: corr(log_TSLA, QQQ_excess_SPY) vs")
print(f"  corr(log_TSLA, SPY_excess_QQQ):")
print(f"    QQQ-unique signal for TSLA: {corr_tsla_qqq_ex:+.4f}")
print(f"    SPY-unique signal for TSLA: {corr_tsla_spy_ex:+.4f}")
print()
print("  If QQQ_excess_SPY is positively correlated with TSLA, it means")
print("  the Nasdaq growth premium over broad market is itself a driver —")
print("  swapping to SPY strips out a genuine signal.")

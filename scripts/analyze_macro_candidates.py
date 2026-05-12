"""
Probe: TLT, FXI, CNY, GLD, HYG as additive v6.4 factors.

Tests all candidates in one pass against the v6.4 baseline.
Each is screened for lagged correlation + VIF, then all proceed to
walk-forward regardless of screen result (full documentation).

Candidates:
  TLT  — iShares 20yr Treasury ETF. Captures long-rate LEVEL (not curve
          shape, which IEF/SHY already encodes). 2022 rate shock was a level
          event; the curve factor captures shape. Genuinely orthogonal story.
  FXI  — iShares China Large-Cap ETF. TSLA Shanghai = ~40% of global volume.
          China macro cycle (stimulus, consumer sentiment, trade tension) is
          not captured by DXY (which is euro/yen-weighted).
  CNY  — USD/CNY offshore rate (USDCNY=X). More direct China FX proxy than FXI.
          When CNY weakens vs USD, Shanghai revenues translate back at discount.
  GLD  — SPDR Gold ETF. Safe-haven / real-asset regime. When gold outperforms
          equities, capital is fleeing into non-productive assets → TSLA premium
          compresses. Orthogonal to VIX (which is implied vol, not safe-haven flow).
  HYG  — iShares HY Bond ETF. Credit risk channel, distinct from equity vol (VIX).
          HY spreads widened before VIX in 2022. HYG_excess_vs_IEF isolates
          credit-risk premium from rate levels.

All transforms: zscore_52w (primary), yoy_z (secondary), mom4_z (fast).
Acceptance bar: +1pp walk-forward OOS R² lift (lowered from 2pp, May 2026).
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END   = "2022-04-26", "2026-04-26"
OOS_START    = "2025-01-03"
MIN_TRAIN    = 100
EVENT_P_THR  = 0.10
BARRIER_PP   = 1.0
VIF_WARN     = 5.0

# ── Data fetch ────────────────────────────────────────────────────────────────

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
    raise RuntimeError(f"Failed to fetch weekly close for {sym}")

# ── OLS helpers ───────────────────────────────────────────────────────────────

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
                resid=pd.Series(r, index=frame.index))


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
    idx     = pred_log.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual  = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    pred    = np.exp(pred_log.loc[idx_oos].to_numpy())
    ss_tot  = ((actual - actual.mean()) ** 2).sum()
    r2      = 1 - ((actual - pred) ** 2).sum() / ss_tot
    mae     = np.mean(np.abs(actual - pred)) / np.mean(actual) * 100
    return float(r2), float(mae), int(len(actual))


def compute_vif(frame, factors):
    X = frame[factors].dropna().to_numpy().astype(float)
    vifs = {}
    for j, fname in enumerate(factors):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        Xo = np.column_stack([np.ones(len(others)), others])
        b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        yhat = Xo @ b
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vifs[fname] = 1.0 / max(1.0 - r2, 1e-10)
    return vifs


def lagged_corr(candidate, resid_v64):
    c_lag = candidate.shift(1)
    common = resid_v64.index.intersection(c_lag.dropna().index)
    if len(common) < 20:
        return np.nan, np.nan
    return stats.pearsonr(c_lag.loc[common], resid_v64.loc[common])


def zscore_52w(s):
    log_s = np.log(s)
    return (log_s - log_s.rolling(52, min_periods=20).mean()) / \
            log_s.rolling(52, min_periods=20).std()


def yoy_z(s):
    log_s = np.log(s)
    raw = log_s.diff(52) * 100
    return (raw - raw.rolling(52, min_periods=20).mean()) / \
            raw.rolling(52, min_periods=20).std()


def mom4_z(s):
    log_s = np.log(s)
    raw = log_s.diff(4) * 100
    return (raw - raw.rolling(52, min_periods=20).mean()) / \
            raw.rolling(52, min_periods=20).std()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch all data
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("Macro candidates probe: TLT, FXI, CNY, GLD, HYG vs v6.4")
print("=" * 68)
print("\n[1] Fetching weekly closes...")

CORE = {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY",
}
CANDIDATES_RAW = {
    "TLT":  "TLT",
    "FXI":  "FXI",
    "CNY":  "USDCNY=X",   # USD/CNY: higher = weaker CNY
    "GLD":  "GLD",
    "HYG":  "HYG",
}

raw = {}
for k, v in {**CORE, **CANDIDATES_RAW}.items():
    try:
        raw[k] = wk(v)
        print(f"  {k:<6} ({v})  {len(raw[k])} rows")
    except Exception as e:
        print(f"  {k:<6} FAILED: {e}")

# Align core frame
core_df = pd.DataFrame({k: raw[k] for k in CORE if k in raw})\
            .resample("W-FRI").last().ffill().dropna()
print(f"\n  Core frame: {len(core_df)} weeks  {core_df.index[0].date()} → {core_df.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build v6.4 factor frame
# ═══════════════════════════════════════════════════════════════════════════════

base = pd.DataFrame(index=core_df.index)
base["log_TSLA"] = np.log(core_df["TSLA"])
base["log_QQQ"]  = np.log(core_df["QQQ"])
base["log_DXY"]  = np.log(core_df["DXY"])
base["log_VIX"]  = np.log(core_df["VIX"])
base["NVDA_excess"] = residualize(np.log(core_df["NVDA"]), base["log_QQQ"])
base["ARKK_excess"] = residualize(np.log(core_df["ARKK"]), base["log_QQQ"])
log_rbob = np.log(core_df["RBOB"])
base["RBOB_zscore_52w"] = (log_rbob - log_rbob.rolling(52, min_periods=20).mean()) / \
                           log_rbob.rolling(52, min_periods=20).std()
curve_log = np.log(core_df["IEF"]) - np.log(core_df["SHY"])
base["curve_IEF_SHY_zscore_52w"] = \
    (curve_log - curve_log.rolling(52, min_periods=20).mean()) / \
     curve_log.rolling(52, min_periods=20).std()

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

FORCED_V64 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"]
f_eq = base.dropna(subset=FORCED_V64)
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f_eq[f"E_{n}"].nunique() > 1]

print(f"  Factor frame: {len(f_eq)} weeks  IS={(f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build candidate transforms
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building candidate transforms...")

# TLT — long-rate level. Positive TLT (falling yields) → should be TSLA-positive.
if "TLT" in raw:
    f_eq["TLT_zscore_52w"]  = zscore_52w(raw["TLT"]).reindex(f_eq.index, method="ffill")
    f_eq["TLT_yoy_z"]       = yoy_z(raw["TLT"]).reindex(f_eq.index, method="ffill")
    f_eq["TLT_mom4_z"]      = mom4_z(raw["TLT"]).reindex(f_eq.index, method="ffill")

# FXI — China large-cap. When FXI runs, TSLA Shanghai revenue / sentiment lifts.
if "FXI" in raw:
    f_eq["FXI_zscore_52w"]  = zscore_52w(raw["FXI"]).reindex(f_eq.index, method="ffill")
    f_eq["FXI_excess_vs_QQQ"] = residualize(
        np.log(raw["FXI"].reindex(f_eq.index, method="ffill").dropna()),
        base["log_QQQ"].reindex(
            raw["FXI"].reindex(f_eq.index, method="ffill").dropna().index)
    ).reindex(f_eq.index)
    f_eq["FXI_yoy_z"]       = yoy_z(raw["FXI"]).reindex(f_eq.index, method="ffill")

# CNY — USD/CNY rate. Higher = weaker CNY → TSLA Shanghai revenues shrink in USD.
# Expected negative beta: when USDCNY rises (CNY weakens), TSLA suffers.
if "CNY" in raw:
    f_eq["CNY_zscore_52w"]  = zscore_52w(raw["CNY"]).reindex(f_eq.index, method="ffill")
    f_eq["CNY_yoy_z"]       = yoy_z(raw["CNY"]).reindex(f_eq.index, method="ffill")
    f_eq["CNY_mom4_z"]      = mom4_z(raw["CNY"]).reindex(f_eq.index, method="ffill")

# GLD — gold ETF. Safe-haven flow signal. Expected negative beta.
if "GLD" in raw:
    f_eq["GLD_zscore_52w"]  = zscore_52w(raw["GLD"]).reindex(f_eq.index, method="ffill")
    f_eq["GLD_excess_vs_QQQ"] = residualize(
        np.log(raw["GLD"].reindex(f_eq.index, method="ffill").dropna()),
        base["log_QQQ"].reindex(
            raw["GLD"].reindex(f_eq.index, method="ffill").dropna().index)
    ).reindex(f_eq.index)
    f_eq["GLD_yoy_z"]       = yoy_z(raw["GLD"]).reindex(f_eq.index, method="ffill")

# HYG — high-yield bond ETF. Credit risk channel.
# HYG_excess_vs_IEF = credit spread proxy: HY returns minus IG rate returns.
if "HYG" in raw:
    f_eq["HYG_zscore_52w"]  = zscore_52w(raw["HYG"]).reindex(f_eq.index, method="ffill")
    log_hyg = np.log(raw["HYG"].reindex(f_eq.index, method="ffill").dropna())
    log_ief  = np.log(core_df["IEF"].reindex(log_hyg.index))
    f_eq["HYG_excess_vs_IEF"] = (log_hyg - log_ief).reindex(f_eq.index)
    f_eq["HYG_yoy_z"]       = yoy_z(raw["HYG"]).reindex(f_eq.index, method="ffill")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. v6.4 baseline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] v6.4 baseline walk-forward...")
pred_base = walk_forward(f_eq, FORCED_V64, active_events)
base_r2, base_mae, n_oos = oos_metrics(f_eq, pred_base)
print(f"  OOS R²={base_r2:.4f}  MAE={base_mae:.2f}%  n_oos={n_oos}")

_, _, m_full = select_events(f_eq, FORCED_V64, active_events)
resid_v64 = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Screen all candidates
# ═══════════════════════════════════════════════════════════════════════════════

# Group candidates by ticker for organised output
CANDIDATE_GROUPS = {
    "TLT (long-rate level)":     ["TLT_zscore_52w", "TLT_yoy_z", "TLT_mom4_z"],
    "FXI (China large-cap)":     ["FXI_zscore_52w", "FXI_excess_vs_QQQ", "FXI_yoy_z"],
    "CNY (USD/CNY FX)":          ["CNY_zscore_52w", "CNY_yoy_z", "CNY_mom4_z"],
    "GLD (gold safe-haven)":     ["GLD_zscore_52w", "GLD_excess_vs_QQQ", "GLD_yoy_z"],
    "HYG (credit risk)":         ["HYG_zscore_52w", "HYG_excess_vs_IEF", "HYG_yoy_z"],
}

all_candidates = [c for group in CANDIDATE_GROUPS.values() for c in group]
# Filter to only those that actually exist in f_eq
all_candidates = [c for c in all_candidates if c in f_eq.columns]

print(f"\n[4] Correlation screen + VIF (all {len(all_candidates)} candidates)...")
print(f"  {'candidate':<28}  {'n':>5}  {'corr_lag1':>10}  {'p':>8}  {'VIF':>6}  screen")

screen_results = {}
for cname in all_candidates:
    sub = f_eq.dropna(subset=[cname])
    if len(sub) < 50:
        print(f"  {cname:<28}  (too few rows: {len(sub)})")
        continue
    corr, p = lagged_corr(sub[cname], resid_v64.reindex(sub.index).dropna())
    vif_sub = sub.dropna(subset=FORCED_V64 + [cname])
    vifs = compute_vif(vif_sub, FORCED_V64 + [cname])
    vif = vifs.get(cname, np.nan)
    c_verdict = "corr-pass" if (not np.isnan(corr) and abs(corr) >= 0.10 and p < 0.20) else "corr-fail"
    v_verdict = "vif-ok" if vif < VIF_WARN else "vif-HIGH"
    screen_results[cname] = (len(sub), corr, p, vif, c_verdict, v_verdict)
    print(f"  {cname:<28}  {len(sub):>5}  {corr:>10.4f}  {p:>8.4f}  {vif:>6.2f}  "
          f"{c_verdict} | {v_verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Walk-forward — all candidates
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5] Walk-forward OOS ({BARRIER_PP}pp acceptance bar)...")
print(f"  {'variant':<40}  {'OOS R²':>7}  {'MAE%':>7}  {'Δ':>8}  verdict")
print(f"  {'v6.4 baseline':<40}  {base_r2:>7.4f}  {base_mae:>6.2f}%  (ref)")

wf_results = {}
for group_name, candidates in CANDIDATE_GROUPS.items():
    candidates = [c for c in candidates if c in f_eq.columns]
    if not candidates:
        continue
    print(f"  --- {group_name} ---")
    for cname in candidates:
        sub = f_eq.dropna(subset=[cname])
        if len(sub) < MIN_TRAIN + 20:
            print(f"  {'v6.4 + ' + cname:<40}  (too few rows)")
            continue
        pred = walk_forward(sub, FORCED_V64 + [cname], active_events)
        r2, mae, _ = oos_metrics(sub, pred)
        delta = (r2 - base_r2) * 100
        verdict = "ACCEPT" if delta >= BARRIER_PP else "REJECT"
        wf_results[cname] = (r2, mae, delta, verdict)
        marker = " ◄◄◄" if verdict == "ACCEPT" else ""
        print(f"  {'v6.4 + ' + cname:<40}  {r2:>7.4f}  {mae:>6.2f}%  {delta:>+7.2f}pp  {verdict}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)
print(f"  v6.4 baseline OOS R²={base_r2:.4f}  MAE={base_mae:.2f}%")
print(f"  Acceptance bar: +{BARRIER_PP}pp")
print()

accepted = [(c, r) for c, r in wf_results.items() if r[3] == "ACCEPT"]
if accepted:
    print(f"  ACCEPTED ({len(accepted)} candidates):")
    for c, (r2, mae, delta, _) in sorted(accepted, key=lambda x: -x[1][2]):
        sr = screen_results.get(c, ())
        print(f"    {c:<30}  Δ={delta:+.2f}pp  OOS R²={r2:.4f}  MAE={mae:.2f}%")
        if sr:
            print(f"      corr_lag1={sr[1]:.4f} (p={sr[2]:.4f})  VIF={sr[3]:.2f}  "
                  f"{sr[4]} | {sr[5]}")
else:
    print("  No candidates cleared the acceptance bar.")

print()
near_miss = [(c, r) for c, r in wf_results.items()
             if r[3] == "REJECT" and r[2] > 0]
if near_miss:
    print(f"  Positive but below bar:")
    for c, (r2, mae, delta, _) in sorted(near_miss, key=lambda x: -x[1][2]):
        print(f"    {c:<30}  Δ={delta:+.2f}pp")

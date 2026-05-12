"""
Probe: DJT (Trump Media & Technology Group) as additive v6.5 factor.

Economic / political story:
  DJT is a direct market proxy for Donald Trump's political capital and
  policy influence. TSLA's fate in 2025-2026 is unusually entangled with
  Musk's relationship with Trump (DOGE, tariffs, regulatory environment,
  federal EV credits). DJT's price movements may capture:
    - Trump political strength → Musk access / regulatory favor
    - Tariff / trade policy expectations (DJT spikes on hawkish trade signals)
    - Policy uncertainty / risk-on vs risk-off for TSLA specifically (not just
      for the broad market — log_QQQ and log_VIX already capture that)

  The hypothesis: after conditioning on the standard macro factors (QQQ, VIX,
  etc.), residual DJT moves that are NOT explained by the market carry an
  incremental "Trump-Musk nexus" signal for TSLA.

  Key confound: DJT listed Oct 2022 via DWAC SPAC. Data available ~Oct 2022+.
  Sample is shorter than v6.5's 2022-04 start — need to check effective n.

  Second confound: high idiosyncratic volatility in DJT (meme-stock behavior).
  Test zscore to dampen level effects.

Transforms tested:
  DJT_excess        — log(DJT) residualized on log_QQQ (removes market beta)
  DJT_excess_zscore — 52w z-score of above
  DJT_excess_mom4_z — 4-week momentum z-score of above (likely artefact, flag)
  log_DJT           — raw log (direct level signal, not residualized)

v6.5 baseline: OOS R² = 0.7554
Acceptance gate: +1pp walk-forward OOS R² lift.
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
BARRIER_PP  = 1.0

# ── helpers ───────────────────────────────────────────────────────────────────

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
    raise RuntimeError(f"Failed to fetch {sym}")


def residualize(target, base_s):
    aligned = base_s.reindex(target.index).dropna()
    target = target.reindex(aligned.index)
    X = np.column_stack([np.ones(len(aligned)), aligned.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)


def fit(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b; r = y - yhat
    n, k = X.shape; s2 = (r @ r) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov)); t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss_tot
    return dict(beta=b, p=p, r2=float(r2),
                fitted=pd.Series(yhat, index=frame.index),
                resid=pd.Series(r, index=frame.index))


def select_events(frame, forced, evs, p_thr=EVENT_P_THR):
    events = [e for e in evs if frame[e].nunique() > 1]
    factors = forced + events
    m = fit(frame, factors)
    while events:
        ep = [(e, float(m["p"][1 + factors.index(e)])) for e in events]
        worst, wp = max(ep, key=lambda x: x[1])
        if wp <= p_thr:
            break
        events.remove(worst)
        factors = forced + events
        m = fit(frame, factors)
    return factors, events, m


def walk_forward(frame, forced, evs, min_train=MIN_TRAIN):
    pred = pd.Series(index=frame.index, dtype=float)
    for date, row in frame.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue
        facs, _, m_tr = select_events(train, forced, evs)
        xv = np.array([1.0] + [float(row[c]) for c in facs])
        pred.loc[date] = float(xv @ m_tr["beta"])
    return pred


def oos_r2(frame, pred):
    idx = pred.dropna().index
    idx_oos = idx[idx >= pd.Timestamp(OOS_START)]
    actual = np.exp(frame.loc[idx_oos, "log_TSLA"].to_numpy())
    p = np.exp(pred.loc[idx_oos].to_numpy())
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - ((actual - p) ** 2).sum() / ss_tot
    mae = np.mean(np.abs(actual - p)) / np.mean(actual) * 100
    return float(r2), float(mae), int(len(actual))


def compute_vif(frame, col, forced):
    cols = forced + [col]
    sub = frame[cols].dropna().to_numpy().astype(float)
    j = len(forced)
    y = sub[:, j]
    others = sub[:, :j]
    Xo = np.column_stack([np.ones(len(others)), others])
    b, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    ss_res = ((y - Xo @ b) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return 1.0 / max(1.0 - r2, 1e-10)


def oos_subcorr(col, frame, resid):
    idx = frame.index[frame.index >= pd.Timestamp(OOS_START)]
    cand = frame.loc[idx, col].dropna()
    res = resid.reindex(cand.index).dropna()
    common = cand.index.intersection(res.index)
    if len(common) < 10:
        return np.nan, np.nan, 0
    c, p = stats.pearsonr(cand.loc[common], res.loc[common])
    return c, p, len(common)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fetch — note DJT has shorter history (DWAC listed ~Oct 2022)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("DJT (Trump Media) signal probe vs v6.5")
print("=" * 68)
print("\n[1] Fetching weekly closes...")

TICKERS = {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY", "VIX3M": "^VIX3M",
    "DJT": "DJT",
}
raw = {}
for k, v in TICKERS.items():
    try:
        raw[k] = wk(v)
        print(f"  {k:<8} ({v:<12})  {len(raw[k])} rows  "
              f"first={raw[k].index[0].date()}  last={raw[k].index[-1].date()}")
    except Exception as e:
        print(f"  {k:<8} FAILED: {e}")

if "DJT" not in raw or raw["DJT"].empty:
    print("\n  ERROR: DJT data unavailable. Trying DWAC as fallback...")
    try:
        raw["DJT"] = wk("DWAC")
        print(f"  DWAC fallback: {len(raw['DJT'])} rows")
    except Exception as e:
        print(f"  DWAC also failed: {e}")
        raise SystemExit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build factor frame (DJT data starts ~Oct 2022, so effective window shorter)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building factor frame...")

core_keys = [k for k in TICKERS if k != "DJT"]
core_df = (pd.DataFrame({k: raw[k] for k in core_keys if k in raw})
           .resample("W-FRI").last().ffill().dropna())

f = pd.DataFrame(index=core_df.index)
f["log_TSLA"] = np.log(core_df["TSLA"])
f["log_QQQ"]  = np.log(core_df["QQQ"])
f["log_DXY"]  = np.log(core_df["DXY"])
f["log_VIX"]  = np.log(core_df["VIX"])
f["NVDA_excess"] = residualize(np.log(core_df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(core_df["ARKK"]), f["log_QQQ"])
lr = np.log(core_df["RBOB"])
f["RBOB_zscore_52w"] = (lr - lr.rolling(52, min_periods=20).mean()) / lr.rolling(52, min_periods=20).std()
cl = np.log(core_df["IEF"]) - np.log(core_df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = (cl - cl.rolling(52, min_periods=20).mean()) / cl.rolling(52, min_periods=20).std()
ts = np.log(core_df["VIX3M"]) - np.log(core_df["VIX"])
f["vix_ts_zscore_52w"] = (ts - ts.rolling(52, min_periods=20).mean()) / ts.rolling(52, min_periods=20).std()

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
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

FORCED_V65 = [
    "log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
    "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w", "vix_ts_zscore_52w",
]
f_eq = f.dropna(subset=FORCED_V65)
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f_eq[f"E_{n}"].nunique() > 1]
print(f"  v6.5 frame (no DJT): {len(f_eq)} weeks  "
      f"IS={(f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# Merge DJT on the v6.5 frame (inner join — only weeks where DJT exists)
djt_s = raw["DJT"].reindex(f_eq.index)
f_eq["log_DJT"] = np.log(djt_s)
f_eq["DJT_excess"] = residualize(np.log(djt_s), f_eq["log_QQQ"])

# Build zscore and momentum on available DJT data
exc = f_eq["DJT_excess"]
f_eq["DJT_excess_zscore"] = (
    (exc - exc.rolling(52, min_periods=20).mean())
    / exc.rolling(52, min_periods=20).std()
)
mom4 = exc.diff(4)
f_eq["DJT_excess_mom4_z"] = (
    (mom4 - mom4.rolling(52, min_periods=20).mean())
    / mom4.rolling(52, min_periods=20).std()
)

CANDIDATES = ["DJT_excess", "DJT_excess_zscore", "DJT_excess_mom4_z", "log_DJT"]

# Report DJT availability
djt_available = f_eq["DJT_excess"].dropna()
print(f"  DJT data in frame:   {len(djt_available)} weeks  "
      f"first={djt_available.index[0].date()}  last={djt_available.index[-1].date()}")
oos_djt = djt_available[djt_available.index >= pd.Timestamp(OOS_START)]
print(f"  DJT OOS weeks:       {len(oos_djt)}  (out of 69 total OOS)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. v6.5 baseline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Running v6.5 baseline walk-forward...")
pred_base = walk_forward(f_eq, FORCED_V65, active_events)
r2_base, mae_base, n_oos = oos_r2(f_eq, pred_base)
print(f"  v6.5 baseline: OOS R²={r2_base:.4f}  MAE={mae_base:.2f}%  n={n_oos}")

_, _, m_full = select_events(f_eq, FORCED_V65, active_events)
resid_full = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Screen each candidate
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Screening DJT candidates...\n")
results = []

for col in CANDIDATES:
    sub = f_eq.dropna(subset=[col])
    if len(sub) < MIN_TRAIN + 10:
        print(f"  {col:<30} SKIP (n={len(sub)} < {MIN_TRAIN+10})")
        continue
    n_sub_oos = (sub.index >= pd.Timestamp(OOS_START)).sum()
    pred = walk_forward(sub, FORCED_V65 + [col], active_events)
    r2, mae, n = oos_r2(sub, pred)
    delta = (r2 - r2_base) * 100
    vif = compute_vif(f_eq.dropna(subset=[col]), col, FORCED_V65)
    c_oos, p_oos, n_oos_c = oos_subcorr(col, f_eq, resid_full)
    flag = "PASS" if delta >= BARRIER_PP else "----"
    print(f"  {col:<30}  n_sub={len(sub)}(oos={n_sub_oos})  "
          f"OOS R²={r2:.4f}  Δ={delta:+.2f}pp  VIF={vif:.2f}  "
          f"OOS-corr={c_oos:+.3f}(p={p_oos:.2f})  {flag}")
    results.append(dict(col=col, r2=r2, delta=delta, vif=vif,
                        oos_corr=c_oos, oos_p=p_oos, n_sub=len(sub), n_oos=n))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("SUMMARY — DJT signal probe vs v6.5")
print("=" * 68)
print(f"  v6.5 baseline OOS R²: {r2_base:.4f}  (gate: +{BARRIER_PP:.1f}pp)")
print(f"  DJT available from:  {djt_available.index[0].date()} "
      f"({len(djt_available)} weeks in frame, {len(oos_djt)} OOS)")
print()
print(f"  {'factor':<30} {'Δ OOS R²':>10}  {'VIF':>6}  {'OOS-corr':>14}  verdict")
print(f"  {'-'*30} {'-'*10}  {'-'*6}  {'-'*14}  -------")
for r in results:
    verdict = "PASS" if r["delta"] >= BARRIER_PP else "REJECT"
    print(f"  {r['col']:<30} {r['delta']:>+9.2f}pp  {r['vif']:>6.2f}  "
          f"{r['oos_corr']:>+8.3f}(p={r['oos_p']:.2f})  {verdict}")

print()
if results:
    best = max(results, key=lambda r: r["delta"])
    if best["delta"] >= BARRIER_PP:
        print(f"  Best: {best['col']} Δ={best['delta']:+.2f}pp → run robustness")
    else:
        print(f"  Best: {best['col']} Δ={best['delta']:+.2f}pp → REJECT (below +1pp gate)")
print()
print("  NOTE: DJT's shorter history means the walk-forward OOS window may")
print("  be partially affected by missing pre-2023 DJT data. Interpret with care.")

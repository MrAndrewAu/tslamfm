"""
Probe: IWM excess and XLY excess as additive v6.5 factors.

Economic stories:
  IWM_excess: log(IWM) residualized on log(QQQ). Captures small-cap vs
    large-cap rotation — reflects domestic risk appetite, rate sensitivity
    (small-caps are more credit-dependent), and tariff/trade cycles that
    hit small-caps disproportionately. TSLA is large-cap but has retail
    ownership and risk-appetite sensitivity that may track small-cap flows
    better than QQQ-beta alone.

  XLY_excess: log(XLY) residualized on log(QQQ). Consumer discretionary
    sector ETF (XLY includes TSLA as top holding, ~18%, but the residual
    after QQQ-orthogonalization strips out tech-beta). Captures consumer
    spending confidence, consumer credit conditions, and the discretionary
    vs staples rotation that mirrors addressable-market sentiment for TSLA.
    NOTE: Because TSLA is a component of XLY, the excess signal may partly
    reflect TSLA leadership within the sector rather than an independent
    macro signal — this is a key confound to flag.

v6.5 baseline (8 FORCED factors):
  log_QQQ, log_DXY, log_VIX, NVDA_excess, ARKK_excess,
  RBOB_zscore_52w, curve_IEF_SHY_zscore_52w, vix_ts_zscore_52w

Acceptance gate: +1pp walk-forward OOS R² lift vs v6.5.
Robustness: VIF check + OOS sub-period correlation sign preserved.

Transforms tested per candidate:
  <sym>_excess         — residualized on log_QQQ (level)
  <sym>_excess_zscore  — 52w z-score of above
  <sym>_excess_mom4_z  — 4-week momentum z-score of above
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
    X = np.column_stack([np.ones(len(base_s)), base_s.to_numpy()])
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
# 1. Fetch
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("IWM excess + XLY excess probe vs v6.5")
print("=" * 68)
print("\n[1] Fetching weekly closes...")

TICKERS = {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB", "VIX": "^VIX",
    "NVDA": "NVDA", "ARKK": "ARKK", "RBOB": "RB=F",
    "IEF": "IEF", "SHY": "SHY", "VIX3M": "^VIX3M",
    "IWM": "IWM", "XLY": "XLY",
}
raw = {}
for k, v in TICKERS.items():
    try:
        raw[k] = wk(v)
        print(f"  {k:<8} ({v:<12})  {len(raw[k])} rows")
    except Exception as e:
        print(f"  {k:<8} FAILED: {e}")

core_df = (pd.DataFrame({k: raw[k] for k in raw})
           .resample("W-FRI").last().ffill().dropna())
print(f"\n  Core: {len(core_df)} weeks  "
      f"{core_df.index[0].date()} -> {core_df.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. v6.5 factor frame
# ═══════════════════════════════════════════════════════════════════════════════

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
print(f"\n  v6.5 factor frame: {len(f_eq)} weeks  "
      f"IS={(f_eq.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f_eq.index >= pd.Timestamp(OOS_START)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build candidate transforms
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building IWM and XLY transforms...")

for sym in ["IWM", "XLY"]:
    log_sym = np.log(core_df[sym]).reindex(f_eq.index)
    exc = residualize(log_sym, f_eq["log_QQQ"])
    f_eq[f"{sym}_excess"] = exc
    f_eq[f"{sym}_excess_zscore"] = (
        (exc - exc.rolling(52, min_periods=20).mean())
        / exc.rolling(52, min_periods=20).std()
    )
    mom4 = exc.diff(4)
    f_eq[f"{sym}_excess_mom4_z"] = (
        (mom4 - mom4.rolling(52, min_periods=20).mean())
        / mom4.rolling(52, min_periods=20).std()
    )

CANDIDATES = {
    "IWM": ["IWM_excess", "IWM_excess_zscore", "IWM_excess_mom4_z"],
    "XLY": ["XLY_excess", "XLY_excess_zscore", "XLY_excess_mom4_z"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. v6.5 baseline walk-forward
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Running v6.5 baseline walk-forward (this takes ~30s)...")
base_pred = walk_forward(f_eq, FORCED_V65, active_events)
r2_base, mae_base, n_oos = oos_r2(f_eq, base_pred)
print(f"  v6.5 baseline: OOS R²={r2_base:.4f}  MAE={mae_base:.2f}%  n={n_oos}")

# Full-sample fit for residuals
_, _, m_full = select_events(f_eq, FORCED_V65, active_events)
resid_full = m_full["resid"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Screen each candidate
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] Screening candidates...\n")
results = []

for sym, cols in CANDIDATES.items():
    print(f"  ── {sym} ──")
    for col in cols:
        sub = f_eq.dropna(subset=[col])
        if len(sub) < MIN_TRAIN + 10:
            print(f"    {col:<30} SKIP (insufficient data)")
            continue
        pred = walk_forward(sub, FORCED_V65 + [col], active_events)
        r2, mae, n = oos_r2(sub, pred)
        delta = (r2 - r2_base) * 100
        vif = compute_vif(f_eq, col, FORCED_V65)
        c_oos, p_oos, n_oos_c = oos_subcorr(col, f_eq, resid_full)
        flag = "PASS" if delta >= BARRIER_PP else "----"
        print(f"    {col:<30}  OOS R²={r2:.4f}  Δ={delta:+.2f}pp  VIF={vif:.2f}  "
              f"OOS-corr={c_oos:+.3f}(p={p_oos:.2f})  {flag}")
        results.append(dict(sym=sym, col=col, r2=r2, delta=delta,
                            vif=vif, oos_corr=c_oos, oos_p=p_oos))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("SUMMARY — IWM / XLY rotation factor probe vs v6.5")
print("=" * 68)
print(f"  v6.5 baseline OOS R²: {r2_base:.4f}  (gate: +{BARRIER_PP:.1f}pp)")
print()
print(f"  {'factor':<32} {'Δ OOS R²':>10}  {'VIF':>6}  {'OOS-corr':>10}  verdict")
print(f"  {'-'*32} {'-'*10}  {'-'*6}  {'-'*10}  -------")
for r in results:
    verdict = "PASS" if r["delta"] >= BARRIER_PP else "REJECT"
    print(f"  {r['col']:<32} {r['delta']:>+9.2f}pp  {r['vif']:>6.2f}  "
          f"{r['oos_corr']:>+8.3f}(p={r['oos_p']:.2f})  {verdict}")

# Identify best per symbol
print()
for sym in CANDIDATES:
    sub = [r for r in results if r["sym"] == sym]
    if not sub:
        continue
    best = max(sub, key=lambda r: r["delta"])
    print(f"  Best {sym}: {best['col']} Δ={best['delta']:+.2f}pp  → "
          + ("ACCEPT candidate — run robustness" if best["delta"] >= BARRIER_PP
             else "REJECT (below gate)"))

print()
print("Note on XLY: TSLA is ~18% of XLY — if XLY_excess lifts the model,")
print("check whether it is capturing consumer-sector macro or TSLA self-reference.")

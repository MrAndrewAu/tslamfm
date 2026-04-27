"""
TSLL / TSLS as factor candidates for v6.3.

TSLL = Direxion Daily TSLA Bull 2X  (launched 2022-08-09)
TSLS = Direxion Daily TSLA Bear 1X  (launched 2022-08-09)

Four angles tested:

  A. TSLL_excess   = residual of log(TSLL) after regressing on log(QQQ)
                     (same construction as NVDA_excess, ARKK_excess)
  B. TSLS_excess   = residual of log(TSLS) after regressing on log(QQQ)
  C. vol_decay_z52 = z-score-52w of cumulative vol-decay index
                     vd_t = log(TSLL_t) - 2*(log(TSLA_t)-log(TSLA_0)) - log(TSLL_0)
                     Represents the path-dependent compounding shortfall vs theoretical 2x.
                     Equivalent to -(cumulative realized variance), so captures vol-regime.
  D. lev_ratio_z52 = z-score-52w of log(TSLL) + log(TSLS) - (log(TSLA) + const)
                     After removing TSLA direction, captures net vol-decay pace in both legs.

For A/B: test as potential additions to v6.3 (acceptance bar +2pp WF lift).
For C/D: same bar.
Also report lagged-correlation gate results for all four.

NOTE: since TSLL/TSLS start 2022-08-08, joint IS sample is ~126w (vs 146w full v6.3).
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
from itertools import chain
warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
FORCED = ["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess",
          "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"]
EVENTS = ["E_Twitter_close","E_AI_day_2023","E_Trump_election","E_DOGE_brand_damage",
          "E_Musk_exits_DOGE","E_TrillionPay","E_Tariff_shock","E_Robotaxi_Austin"]

# ── helpers ──────────────────────────────────────────────────────────────────
def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s.resample("W-FRI").last()

def zscore_52(s):
    return (s - s.rolling(52, min_periods=20).mean()) / s.rolling(52, min_periods=20).std()

def residualize(t, b):
    mask = t.notna() & b.notna()
    t_v, b_v = t[mask].values, b[mask].values
    X_fit = np.column_stack([np.ones(mask.sum()), b_v])
    c, *_ = np.linalg.lstsq(X_fit, t_v, rcond=None)
    res = pd.Series(np.nan, index=t.index)
    X_full = np.column_stack([np.ones(len(b)), b.to_numpy()])
    res[mask] = t[mask].values - X_fit @ c
    return res

def add_events(f):
    defs = [("Twitter_close","2022-10-27"),("AI_day_2023","2023-07-19"),
            ("Trump_election","2024-11-06"),("DOGE_brand_damage","2025-02-15"),
            ("Musk_exits_DOGE","2025-04-22"),("TrillionPay","2025-09-05"),
            ("Tariff_shock","2026-02-01"),("Robotaxi_Austin","2025-06-22")]
    for name, dt in defs:
        d0 = pd.Timestamp(dt)
        f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

def ols(y, X):
    X2 = np.column_stack([np.ones(len(X)), X])
    # drop near-zero-variance columns (e.g. event dummies absent from trimmed window)
    var = X2.var(axis=0); var[0] = 1.0  # keep intercept
    keep = var > 1e-10
    X2k = X2[:, keep]
    b_keep, *_ = np.linalg.lstsq(X2k, y, rcond=None)
    b = np.zeros(X2.shape[1]); b[keep] = b_keep
    fit = X2k @ b_keep
    res = y - fit
    ss_res = (res**2).sum(); ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot
    n, k = len(y), X2k.shape[1]
    s2 = ss_res / max(n - k, 1)
    cov = s2 * np.linalg.pinv(X2k.T @ X2k)
    se = np.zeros(X2.shape[1]); se[keep] = np.sqrt(np.maximum(cov.diagonal(), 0))
    t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
    p = np.where(se > 0, 2*(1 - stats.t.cdf(np.abs(t), df=max(n-k, 1))), np.nan)
    mae = np.mean(np.abs(res))
    return dict(b=b, r2=r2, p=p, mae=mae)

def wf_oos(df_full, candidate_col):
    """Expanding-window walk-forward. Returns dict with wf_r2, oos_r2, oos_mae."""
    feats = FORCED + EVENTS + [candidate_col]
    d = df_full[["log_TSLA"] + feats].dropna()
    oos_mask = d.index >= OOS_START
    n_oos = oos_mask.sum()
    if n_oos < 20:
        return dict(wf_r2=np.nan, oos_r2=np.nan, oos_mae=np.nan)
    preds = []
    for t in d.index[oos_mask]:
        train = d[d.index < t]
        if len(train) < 40: continue
        res = ols(train["log_TSLA"].values, train[feats].values)
        X_t = np.array([1] + [d.loc[t, c] for c in feats])
        preds.append((t, d.loc[t,"log_TSLA"], X_t @ res["b"]))
    if not preds: return dict(wf_r2=np.nan, oos_r2=np.nan, oos_mae=np.nan)
    p_df = pd.DataFrame(preds, columns=["dt","actual","pred"]).set_index("dt")
    ss_res = ((p_df.actual - p_df.pred)**2).sum()
    ss_tot = ((p_df.actual - p_df.actual.mean())**2).sum()
    wf_r2 = 1 - ss_res/ss_tot
    ss_res2 = ss_res
    p_df2 = p_df  # same
    oos_r2 = 1 - ss_res2/ss_tot
    oos_mae = np.mean(np.abs(p_df.actual - p_df.pred)) * 100
    return dict(wf_r2=wf_r2, oos_r2=oos_r2, oos_mae=oos_mae)

def baseline_wf(df_full):
    feats = FORCED + EVENTS
    d = df_full[["log_TSLA"] + feats].dropna()
    oos_mask = d.index >= OOS_START
    preds = []
    for t in d.index[oos_mask]:
        train = d[d.index < t]
        if len(train) < 40: continue
        res = ols(train["log_TSLA"].values, train[feats].values)
        X_t = np.array([1] + [d.loc[t, c] for c in feats])
        preds.append((t, d.loc[t,"log_TSLA"], X_t @ res["b"]))
    p_df = pd.DataFrame(preds, columns=["dt","actual","pred"]).set_index("dt")
    ss_res = ((p_df.actual - p_df.pred)**2).sum()
    ss_tot = ((p_df.actual - p_df.actual.mean())**2).sum()
    wf_r2 = 1 - ss_res/ss_tot
    oos_mae = np.mean(np.abs(p_df.actual - p_df.pred)) * 100
    return dict(wf_r2=wf_r2, oos_r2=wf_r2, oos_mae=oos_mae)

# ── fetch data ────────────────────────────────────────────────────────────────
print("Fetching data...")
syms = {"TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB","VIX":"^VIX",
        "NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F","IEF":"IEF","SHY":"SHY",
        "TSLL":"TSLL","TSLS":"TSLS"}
raw = {k: wk(v) for k, v in syms.items()}
df_all = pd.DataFrame(raw).ffill()
print(f"  Raw frame: {len(df_all)} weeks  {df_all.index[0].date()} → {df_all.index[-1].date()}")
print(f"  TSLL/TSLS start: {df_all['TSLL'].first_valid_index().date()}  "
      f"({df_all[['TSLL','TSLS']].dropna().shape[0]} wks with both)")

# ── build features ────────────────────────────────────────────────────────────
f = pd.DataFrame(index=df_all.index)
f["log_TSLA"] = np.log(df_all["TSLA"])
f["log_QQQ"]  = np.log(df_all["QQQ"])
f["log_DXY"]  = np.log(df_all["DXY"])
f["log_VIX"]  = np.log(df_all["VIX"])
f["NVDA_excess"]  = residualize(np.log(df_all["NVDA"]),  f["log_QQQ"])
f["ARKK_excess"]  = residualize(np.log(df_all["ARKK"]),  f["log_QQQ"])
f["RBOB_zscore_52w"]       = zscore_52(np.log(df_all["RBOB"]))
f["curve_IEF_SHY_zscore_52w"] = zscore_52(np.log(df_all["IEF"]) - np.log(df_all["SHY"]))
add_events(f)

# ── construct TSLL/TSLS signals ───────────────────────────────────────────────
log_tsll = np.log(df_all["TSLL"])
log_tsls = np.log(df_all["TSLS"])
log_tsla = np.log(df_all["TSLA"])
log_qqq  = f["log_QQQ"]

# A. TSLL_excess: residualize log(TSLL) on log(QQQ)
f["TSLL_excess"] = residualize(log_tsll, log_qqq)

# B. TSLS_excess: residualize log(TSLS) on log(QQQ)
f["TSLS_excess"] = residualize(log_tsls, log_qqq)

# C. vol_decay_z52: cumulative deviation of TSLL from theoretical 2x path
#    vd_t = log(TSLL_t) - [2*(log(TSLA_t) - log(TSLA_0)) + log(TSLL_0)]
#    This accumulates the daily-rebalancing compounding shortfall (= -cumulative realized var)
tsll_valid = df_all["TSLL"].first_valid_index()
tsla_base  = log_tsla[tsll_valid]
tsll_base  = log_tsll[tsll_valid]
theoretical_2x = 2 * (log_tsla - tsla_base) + tsll_base
vol_decay_index = log_tsll - theoretical_2x   # cumulative path shortfall (negative drift)
f["vol_decay_z52"] = zscore_52(vol_decay_index)

# D. lev_ratio_z52: net vol-decay in both legs combined
#    After removing TSLA level: (log_TSLL + log_TSLS) tracks (2x - 1x = +1x) TSLA + both decays
#    Residualize vs log_TSLA level to isolate the combined vol-decay / sentiment residual
lev_sum = log_tsll + log_tsls
f["lev_ratio_z52"] = zscore_52(residualize(lev_sum, log_tsla))

candidates = {
    "TSLL_excess":   "A  TSLL residualized vs QQQ (like NVDA_excess)",
    "TSLS_excess":   "B  TSLS residualized vs QQQ",
    "vol_decay_z52": "C  Vol-decay z52 (TSLL shortfall vs 2x theoretical)",
    "lev_ratio_z52": "D  Lev-ratio z52 (net vol-decay / sentiment residual)",
}

# ── descriptive: correlations between signals and TSLA ───────────────────────
print("\n=== Signal descriptives ===")
for col, label in candidates.items():
    s = f[col].dropna()
    jnt = pd.concat([f["log_TSLA"], s], axis=1).dropna()
    jnt.columns = ["tsla", col]
    if len(jnt) < 2:
        print(f"  {col:20s}  insufficient data"); continue
    corr_tsla, p_tsla = stats.pearsonr(jnt["tsla"].values, jnt[col].values)
    print(f"  {col:20s}  n={len(jnt)}  mean={s.mean():+.4f}  std={s.std():.4f}  "
          f"corr_TSLA={corr_tsla:+.3f} p={p_tsla:.3f}")

# ── gate 1: lagged correlation (1-week ahead) ─────────────────────────────────
print("\n=== Gate 1: lagged correlation with Δlog(TSLA) next week ===")
print(f"  (require |corr| >= 0.15  AND  p < 0.05  AND  n >= 50)\n")
tsla_ret = f["log_TSLA"].diff()
gate1_pass = {}
for col in candidates:
    s = f[col].shift(1)   # use last week's signal to predict this week's return
    jnt = pd.concat([tsla_ret, s], axis=1).dropna()
    jnt.columns = ["ret","sig"]
    if len(jnt) < 50:
        print(f"  {col:20s}  n={len(jnt)}  INSUFFICIENT DATA")
        gate1_pass[col] = False; continue
    r, p = stats.pearsonr(jnt["ret"].values, jnt["sig"].values)
    oos_jnt = jnt[jnt.index >= OOS_START]
    r_oos, p_oos = (stats.pearsonr(oos_jnt["ret"].values, oos_jnt["sig"].values)
                    if len(oos_jnt) >= 10 else (np.nan, np.nan))
    same_sign = np.sign(r) == np.sign(r_oos) if not np.isnan(r_oos) else False
    ok = (abs(r) >= 0.15) and (p < 0.05) and (len(jnt) >= 50)
    gate1_pass[col] = ok
    flag = "PASS" if ok else "FAIL"
    print(f"  {col:20s}  n={len(jnt)}  IS r={r:+.3f} p={p:.3f}  "
          f"OOS r={r_oos:+.3f} p={p_oos:.3f} same_sign={same_sign}  [{flag}]")

# ── IS regression (context) ───────────────────────────────────────────────────
print("\n=== IS regression: v6.3 baseline ===")
feats_base = FORCED + EVENTS
d_base = f[["log_TSLA"] + feats_base].dropna()
print(f"  n_IS={len(d_base[d_base.index < OOS_START])}  n_OOS={len(d_base[d_base.index >= OOS_START])}")
res_base = ols(d_base["log_TSLA"].values, d_base[feats_base].values)
print(f"  v6.3 IS R²={res_base['r2']:.4f}")

print("\n=== IS regression: candidates added to v6.3 ===")
for col, label in candidates.items():
    feats_cand = FORCED + EVENTS + [col]
    d_c = f[["log_TSLA"] + feats_cand].dropna()
    n_is = len(d_c[d_c.index < OOS_START])
    n_oos = len(d_c[d_c.index >= OOS_START])
    res_c = ols(d_c["log_TSLA"].values, d_c[feats_cand].values)
    # coefficient + p-value for the candidate (last in feats, index = len(FORCED+EVENTS)+1 in b)
    idx_c = len(feats_cand)   # 0=intercept, 1..n=feats; candidate is last
    b_c = res_c["b"][idx_c]
    p_c = res_c["p"][idx_c]
    print(f"  {col:20s}  n_IS={n_is}  IS_R²={res_c['r2']:.4f}  "
          f"β={b_c:+.4f}  p={p_c:.4f}")

# ── walk-forward OOS ──────────────────────────────────────────────────────────
print("\n=== Walk-forward OOS comparison ===")
print(f"  {'variant':<35}  {'WF_R²':>7}  {'OOS_MAE':>8}  {'lift_vs_base':>13}  verdict")
print(f"  {'-'*35}  {'-'*7}  {'-'*8}  {'-'*13}  -------")

base = baseline_wf(f)
print(f"  {'v6.3 baseline':35s}  {base['wf_r2']:>7.4f}  {base['oos_mae']:>7.2f}%  {'(ref)':>13}")

for col, label in candidates.items():
    res = wf_oos(f, col)
    lift = res["wf_r2"] - base["wf_r2"] if not np.isnan(res["wf_r2"]) else np.nan
    lift_pp = lift * 100 if not np.isnan(lift) else np.nan
    if np.isnan(lift_pp):
        verdict = "SKIP"
    elif lift_pp >= 2.0:
        verdict = "ACCEPT (+2pp)"
    elif lift_pp >= 0.0:
        verdict = f"WEAK (+{lift_pp:.2f}pp)"
    else:
        verdict = f"REJECT ({lift_pp:+.2f}pp)"
    print(f"  {col:35s}  {res['wf_r2']:>7.4f}  {res['oos_mae']:>7.2f}%  {lift_pp:>+12.2f}pp  {verdict}")

# ── rolling 52w beta stability ────────────────────────────────────────────────
print("\n=== Rolling 52w beta stability: vs log_TSLA ===")
print("  (Lower std = more stable = better persistent signal)\n")
for col in candidates:
    jnt = pd.concat([f["log_TSLA"], f[col]], axis=1).dropna()
    jnt.columns = ["tsla","sig"]
    roll_b = []
    for i in range(52, len(jnt)+1):
        chunk = jnt.iloc[i-52:i]
        if chunk["sig"].std() < 1e-10: continue
        c, _ = np.polyfit(chunk["sig"].values, chunk["tsla"].values, 1)
        roll_b.append(c)
    if roll_b:
        rb = np.array(roll_b)
        print(f"  {col:20s}  rolling β: mean={rb.mean():+.3f}  std={rb.std():.3f}  "
              f"min={rb.min():+.3f}  max={rb.max():+.3f}")
    else:
        print(f"  {col:20s}  insufficient data for rolling β")

"""
#2 Options-skew signal probe vs v6.3 baseline.

True TSLA 25-delta risk reversal is unavailable free. We test the best
accessible proxies in order of TSLA-specificity:

  A. TSLA_rvol_premium  = TSLA 26w realized vol (annualized) / VIX level
                          (TSLA-specific fear premium over broad market IV)
  B. CBOE_SKEW          = ^SKEW  (SPX OTM put demand; tail-risk pricing)
  C. VXN_VIX_spread     = log(VXN) - log(VIX)  (Nasdaq vs S&P vol premium;
                          proxy for tech/growth skew relative to broad market)
  D. TSLA_rvol_z52      = TSLA realized vol z-scored vs trailing 52w
                          (pure TSLA realized vol regime, lookahead-free)

Each candidate is:
  1. Lagged-correlation gate  (lag-1 week, p<0.05 & |corr|>=0.15 & n>=50)
  2. OOS sub-period check     (same sign & p<0.10 in 2025-01-03 onward)
  3. Walk-forward OOS lift    (>=+2pp vs v6.3 baseline; 1pp for event dummies)
     Force-promote if OOS |corr|>=0.20 & p<0.01 even if lagged weak

Acceptance bar: >=+2pp OOS R2 lift vs v6.3 (continuous factor bar).
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings("ignore")

START, END   = "2022-04-26", "2026-04-26"
OOS_START    = "2025-01-03"
BARRIER_PP   = 2.0   # continuous factor bar
FORCE_CORR   = 0.20
FORCE_P      = 0.01

def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty:
        return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s.resample("W-FRI").last()

def residualize(target, base):
    X = np.column_stack([np.ones(len(base)), base.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
    return pd.Series(target.to_numpy() - X @ coef, index=target.index)

def zscore_52(s):
    mu = s.rolling(52, min_periods=20).mean()
    sd = s.rolling(52, min_periods=20).std()
    return (s - mu) / sd

def tsla_rvol_annualized(log_tsla, window=26):
    """Rolling realized vol from weekly log-returns, annualized (x sqrt(52))."""
    ret = log_tsla.diff()
    return ret.rolling(window, min_periods=12).std() * np.sqrt(52)

# ── Download ──────────────────────────────────────────────────────────────────
print("Fetching data...")
tickers = {"TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB",
           "VIX": "^VIX", "NVDA": "NVDA", "ARKK": "ARKK",
           "RBOB": "RB=F", "IEF": "IEF", "SHY": "SHY",
           "SKEW": "^SKEW", "VXN": "^VXN"}
raw = {k: wk(v) for k, v in tickers.items()}
df = pd.DataFrame(raw).ffill().dropna()
print(f"  {len(df)} weeks, {df.index[0].date()} → {df.index[-1].date()}")

# ── Build v6.3 feature frame ──────────────────────────────────────────────────
f = pd.DataFrame(index=df.index)
f["log_TSLA"]  = np.log(df["TSLA"])
f["log_QQQ"]   = np.log(df["QQQ"])
f["log_DXY"]   = np.log(df["DXY"])
f["log_VIX"]   = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])
log_rbob = np.log(df["RBOB"])
f["RBOB_zscore_52w"] = zscore_52(log_rbob)
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
f["curve_IEF_SHY_zscore_52w"] = zscore_52(curve_log)

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

f = f.dropna()
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]

V63 = (["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
        "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_events)

# ── Construct candidates ──────────────────────────────────────────────────────
# A. TSLA realized vol premium over VIX
tsla_rv = tsla_rvol_annualized(f["log_TSLA"], window=26)
vix_lvl = df["VIX"].reindex(f.index) / 100.0   # VIX is in percentage points
f["TSLA_rvol_premium"] = zscore_52(tsla_rv / vix_lvl)

# B. CBOE SKEW (already in df)
f["CBOE_SKEW_z52"] = zscore_52(df["SKEW"].reindex(f.index))

# C. VXN/VIX spread (Nasdaq vol premium over S&P vol)
vxn_vix_spread = np.log(df["VXN"].reindex(f.index)) - np.log(df["VIX"].reindex(f.index))
f["VXN_VIX_spread_z52"] = zscore_52(vxn_vix_spread)

# D. TSLA realized vol z-score (pure TSLA vol regime)
f["TSLA_rvol_z52"] = zscore_52(tsla_rv)

f = f.dropna()
print(f"  After candidate construction: {len(f)} weeks")

CANDIDATES = ["TSLA_rvol_premium", "CBOE_SKEW_z52", "VXN_VIX_spread_z52", "TSLA_rvol_z52"]

# ── OLS helpers ──────────────────────────────────────────────────────────────
def fit_ols(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    r = y - X @ b
    s2 = (r @ r) / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    fitted = pd.Series(X @ b, index=frame.index)
    resid  = pd.Series(r, index=frame.index)
    r2     = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    mae    = np.mean(np.abs(np.exp(y) - np.exp(X @ b))) / np.mean(np.exp(y)) * 100
    return dict(b=b, se=se, t=t, p=p, fitted=fitted, resid=resid, r2=r2, mae=mae,
                factors=factors)

def r2_oos(frame, factors):
    oos = frame[frame.index >= pd.Timestamp(OOS_START)]
    if len(oos) < 10:
        return np.nan, np.nan
    y = oos["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(oos))] + [oos[c].to_numpy() for c in factors])
    # fit on IS, predict OOS
    is_ = frame[frame.index < pd.Timestamp(OOS_START)]
    y_is = is_["log_TSLA"].to_numpy()
    X_is = np.column_stack([np.ones(len(is_))] + [is_[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X_is, y_is, rcond=None)
    pred = X @ b
    r2   = 1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()
    mae  = np.mean(np.abs(np.exp(y) - np.exp(pred))) / np.mean(np.exp(y)) * 100
    return r2, mae

def wf_oos(frame, factors, min_train=104):
    """Expanding-window walk-forward OOS R²."""
    dates = frame.index.tolist()
    preds, actuals = [], []
    for t in range(min_train, len(dates)):
        train = frame.iloc[:t]
        row   = frame.iloc[t:t+1]
        y_tr  = train["log_TSLA"].to_numpy()
        X_tr  = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in factors])
        b, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        X_row = np.column_stack([np.ones(1)] + [row[c].to_numpy() for c in factors])
        preds.append(float((X_row @ b).item()))
        actuals.append(float(row["log_TSLA"].iloc[0]))
    preds, actuals = np.array(preds), np.array(actuals)
    r2  = 1 - ((actuals - preds)**2).sum() / ((actuals - actuals.mean())**2).sum()
    mae = np.mean(np.abs(np.exp(actuals) - np.exp(preds))) / np.mean(np.exp(actuals)) * 100
    return r2, mae

# ── Baseline v6.3 ─────────────────────────────────────────────────────────────
print("\n=== v6.3 baseline ===")
base_is = fit_ols(f, V63)
base_wf_r2, base_wf_mae = wf_oos(f, V63)
print(f"  IS R²={base_is['r2']:.4f}  WF R²={base_wf_r2:.4f}  WF MAE={base_wf_mae:.2f}%")

# ── Descriptive stats on candidates ──────────────────────────────────────────
print("\n=== Candidate descriptive stats ===")
oos_f = f[f.index >= pd.Timestamp(OOS_START)]
hdr = f"  {'candidate':<28} {'n':>4} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}"
print(hdr)
for c in CANDIDATES:
    s = f[c]
    print(f"  {c:<28} {len(s):>4} {s.mean():>7.3f} {s.std():>7.3f} {s.min():>7.3f} {s.max():>7.3f}")

# ── Step 1: Lagged correlation gate ──────────────────────────────────────────
print("\n=== Step 1: Lagged correlation gate (lag-1 week) ===")
print(f"  {'candidate':<28} {'n':>4} {'corr':>7} {'p':>10}  verdict")
passers = []
for cand in CANDIDATES:
    sub = f[["log_TSLA", cand]].dropna()
    # residualize log_TSLA on v6.3 factors, correlate with lagged candidate
    m_base = fit_ols(sub.join(f[V63].dropna(), how="inner", lsuffix="", rsuffix="_r")
                     .dropna()[list(dict.fromkeys(["log_TSLA"] + V63))], V63)
    resid_ser = m_base["resid"]
    cand_lagged = f[cand].shift(1).reindex(resid_ser.index).dropna()
    common = resid_ser.index.intersection(cand_lagged.index)
    r_vec = resid_ser.loc[common]
    c_vec = cand_lagged.loc[common]
    n = len(r_vec)
    if n < 50:
        print(f"  {cand:<28} {n:>4}  (insufficient data)")
        continue
    corr, p_corr = stats.pearsonr(r_vec, c_vec)
    lag_pass = (p_corr < 0.05) and (abs(corr) >= 0.15)
    force_check = (abs(corr) >= 0.05) and (p_corr < 0.10)  # weak signal, track anyway
    verdict = "PASS" if lag_pass else ("WEAK" if force_check else "REJECT")
    print(f"  {cand:<28} {n:>4} {corr:>7.3f} {p_corr:>10.4f}  {verdict}")
    if lag_pass or force_check:
        passers.append((cand, corr, p_corr, verdict))

# ── Step 2: OOS sub-period check ─────────────────────────────────────────────
print("\n=== Step 2: OOS sub-period correlation (2025-01-03 onward) ===")
step2_passers = []
for cand, corr_lag, p_lag, v1 in passers:
    oos_sub = oos_f[["log_TSLA", cand]].dropna()
    if len(oos_sub) < 10:
        print(f"  {cand:<28}  (OOS too short)")
        continue
    # residualize OOS log_TSLA vs v6.3 factors
    oos_all = f[f.index >= pd.Timestamp(OOS_START)].dropna()
    m_oos = fit_ols(oos_all, V63)
    resid_oos = m_oos["resid"]
    cand_lagged_oos = f[cand].shift(1).reindex(resid_oos.index).dropna()
    common_oos = resid_oos.index.intersection(cand_lagged_oos.index)
    r_oos = resid_oos.loc[common_oos]
    c_oos = cand_lagged_oos.loc[common_oos]
    if len(r_oos) < 8:
        print(f"  {cand:<28}  (too few OOS obs)")
        continue
    corr_oos, p_oos = stats.pearsonr(r_oos, c_oos)
    same_sign = (corr_lag * corr_oos) > 0
    oos_pass = same_sign and (p_oos < 0.10)
    force_promote = (abs(corr_oos) >= FORCE_CORR) and (p_oos < FORCE_P) and same_sign
    verdict2 = ("FORCE" if force_promote else ("PASS" if oos_pass else "REJECT"))
    print(f"  {cand:<28}  lag_corr={corr_lag:>+.3f}  oos_corr={corr_oos:>+.3f}  "
          f"p_oos={p_oos:.4f}  same_sign={'Y' if same_sign else 'N'}  {verdict2}")
    if oos_pass or force_promote:
        step2_passers.append((cand, corr_lag, corr_oos, p_oos, verdict2))

# ── Step 3: Walk-forward ──────────────────────────────────────────────────────
if step2_passers:
    print(f"\n=== Step 3: Walk-forward OOS R² lift (bar: +{BARRIER_PP:.1f}pp) ===")
    print(f"  {'variant':<42} {'WF R2':>7} {'WF MAE':>8} {'dR2':>8}")
    print(f"  {'v6.3 baseline':<42} {base_wf_r2:.4f} {base_wf_mae:>7.2f}%  (ref)")
    accepted = []
    for cand, *_ in step2_passers:
        r2_wf, mae_wf = wf_oos(f, V63 + [cand])
        dR2 = (r2_wf - base_wf_r2) * 100
        verdict = "ACCEPT" if dR2 >= BARRIER_PP else "REJECT"
        print(f"  {f'v6.3 + {cand}':<42} {r2_wf:.4f} {mae_wf:>7.2f}%  {dR2:>+.2f}pp  {verdict}")
        if dR2 >= BARRIER_PP:
            accepted.append(cand)
    print(f"\n  Accepted: {accepted if accepted else 'none'}")
else:
    print("\n=== Step 3: No candidates survived gate 2 — skipping walk-forward ===")
    print("  VERDICT: Options-skew candidates rejected at lagged-correlation gate.")

# ── Correlation matrix among candidates + v6.3 residuals ─────────────────────
print("\n=== Correlation matrix: candidates vs v6.3 IS residuals ===")
base_resid = base_is["resid"]
corr_frame = pd.DataFrame({"v6.3_resid": base_resid})
for c in CANDIDATES:
    corr_frame[c] = f[c].reindex(base_resid.index)
corr_frame = corr_frame.dropna()
print(corr_frame.corr().to_string(float_format=lambda x: f"{x:+.3f}"))

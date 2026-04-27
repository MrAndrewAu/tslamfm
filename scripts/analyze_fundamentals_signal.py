"""
#3 Fundamentals probe vs v6.3 baseline.

Candidates:
  A. deliveries_yoy_z  — YoY delivery growth, z-scored vs 4-quarter history
                         (known after delivery report, ~first week of each quarter)
  B. deliveries_qoq    — QoQ delivery change (raw; captures trend breaks)
  C. gross_margin_z    — Tesla total gross margin %, z-scored (from yfinance quarterly)
  D. gross_margin_delta— QoQ change in gross margin (momentum)

Sparse-data strategy:
  - Values known after report date, forward-filled weekly until next report
  - Lagged correlation tested at QUARTERLY frequency (avoids autocorrelation
    inflation from 13 identical weekly values per quarter)
  - Walk-forward run at weekly frequency with forward-filled series
  - Quarterly n is small (~10 IS quarters) → power is limited; we report this
  - Acceptance bar same as continuous factors: +2pp WF OOS R² lift

Delivery data hardcoded (public record, report dates approximate first trading
day of each quarter). Q2 2025 - Q1 2026 marked as estimates where uncertain.
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats
warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
BARRIER_PP  = 2.0

# ── Delivery data (hardcoded; quarterly, aligned to report week) ──────────────
# Format: report_date → vehicles_delivered
# Pre-2025: high confidence. 2025+ marked with * (estimates where uncertain)
DELIVERIES = pd.Series({
    "2022-07-05": 254_695,   # Q2 2022
    "2022-10-04": 343_830,   # Q3 2022
    "2023-01-03": 405_278,   # Q4 2022
    "2023-04-04": 422_875,   # Q1 2023
    "2023-07-04": 466_140,   # Q2 2023
    "2023-10-03": 435_059,   # Q3 2023
    "2024-01-02": 484_507,   # Q4 2023
    "2024-04-02": 386_810,   # Q1 2024
    "2024-07-02": 443_956,   # Q2 2024
    "2024-10-02": 469_796,   # Q3 2024
    "2025-01-03": 495_570,   # Q4 2024
    "2025-04-03": 336_681,   # Q1 2025 (confirmed: 14% miss vs ~390k)
    "2025-07-03": 384_122,   # Q2 2025 *estimate*
    "2025-10-02": 462_890,   # Q3 2025 *estimate*
    "2026-01-05": 495_000,   # Q4 2025 *estimate*
    "2026-04-03": 420_000,   # Q1 2026 *estimate*
}, dtype=float)
DELIVERIES.index = pd.to_datetime(DELIVERIES.index)
print(f"Delivery series: {len(DELIVERIES)} quarters  "
      f"({DELIVERIES.index[0].date()} → {DELIVERIES.index[-1].date()})")

# ── Fetch yfinance quarterly financials ───────────────────────────────────────
print("Fetching TSLA quarterly financials from yfinance...")
tsla = yf.Ticker("TSLA")
try:
    qfin = tsla.quarterly_income_stmt
    # rows = line items, cols = quarter end dates
    gp_row = [r for r in qfin.index if "Gross Profit" in str(r)]
    rev_row = [r for r in qfin.index if "Total Revenue" in str(r)]
    if gp_row and rev_row:
        gp  = qfin.loc[gp_row[0]].sort_index()
        rev = qfin.loc[rev_row[0]].sort_index()
        gm  = (gp / rev * 100).dropna()
        gm.index = pd.to_datetime(gm.index)
        if getattr(gm.index, "tz", None) is not None:
            gm.index = gm.index.tz_localize(None)
        print(f"  Gross margin: {len(gm)} quarters  "
              f"({gm.index[0].date()} → {gm.index[-1].date()})")
        print(f"  Range: {gm.min():.1f}% → {gm.max():.1f}%")
        HAS_GM = True
    else:
        print("  Gross margin line items not found")
        HAS_GM = False
except Exception as e:
    print(f"  yfinance error: {e}")
    HAS_GM = False

# ── Fetch weekly price data ───────────────────────────────────────────────────
def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s.resample("W-FRI").last()

def zscore_52(s): 
    return (s - s.rolling(52,min_periods=20).mean()) / s.rolling(52,min_periods=20).std()

def residualize(t, b):
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    c, *_ = np.linalg.lstsq(X, t.to_numpy(), rcond=None)
    return pd.Series(t.to_numpy() - X @ c, index=t.index)

print("Fetching weekly closes...")
raw = {k: wk(v) for k, v in {"TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB",
      "VIX":"^VIX","NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F",
      "IEF":"IEF","SHY":"SHY"}.items()}
df = pd.DataFrame(raw).ffill().dropna()
print(f"  {len(df)} weeks, {df.index[0].date()} → {df.index[-1].date()}")

fw = pd.DataFrame(index=df.index)
fw["log_TSLA"] = np.log(df["TSLA"])
fw["log_QQQ"]  = np.log(df["QQQ"])
fw["log_DXY"]  = np.log(df["DXY"])
fw["log_VIX"]  = np.log(df["VIX"])
fw["NVDA_excess"] = residualize(np.log(df["NVDA"]), fw["log_QQQ"])
fw["ARKK_excess"] = residualize(np.log(df["ARKK"]), fw["log_QQQ"])
fw["RBOB_zscore_52w"] = zscore_52(np.log(df["RBOB"]))
fw["curve_IEF_SHY_zscore_52w"] = zscore_52(np.log(df["IEF"]) - np.log(df["SHY"]))

EVENT_DEFS = [
    ("Split_squeeze_2020","2020-08-11"),("SP500_inclusion","2020-11-16"),
    ("Hertz_1T_peak","2021-10-25"),    ("Twitter_overhang","2022-04-25"),
    ("Twitter_close","2022-10-27"),    ("AI_day_2023","2023-07-19"),
    ("Trump_election","2024-11-06"),   ("DOGE_brand_damage","2025-02-15"),
    ("Musk_exits_DOGE","2025-04-22"),  ("TrillionPay","2025-09-05"),
    ("Tariff_shock","2026-02-01"),     ("Robotaxi_Austin","2025-06-22"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    fw[f"E_{name}"] = ((fw.index >= d0) & (fw.index < d0 + pd.Timedelta(weeks=8))).astype(int)
# Note: fw.dropna() deferred until after fundamental columns added (below)

# ── Build fundamental series (forward-filled to weekly) ──────────────────────
def forward_fill_quarterly(qseries, weekly_index):
    """Align quarterly series to weekly: value becomes available on report date,
    forward-filled until next report. Returns weekly Series."""
    wkly = pd.Series(np.nan, index=weekly_index, dtype=float)
    for dt, val in qseries.sort_index().items():
        # find first Friday on or after report date
        mask = weekly_index >= dt
        if mask.any():
            wkly.loc[weekly_index[mask][0]] = val
    return wkly.ffill()

# Deliveries
del_wkly  = forward_fill_quarterly(DELIVERIES, fw.index)
del_yoy   = del_wkly.pct_change(4) * 100   # approx YoY using 4 lagged quarters
# YoY properly: need quarterly series, then align
del_q = DELIVERIES.sort_index()
del_yoy_q = del_q.pct_change(4) * 100
del_qoq_q = del_q.pct_change(1) * 100
del_yoy_wkly = forward_fill_quarterly(del_yoy_q.dropna(), fw.index)
del_qoq_wkly = forward_fill_quarterly(del_qoq_q.dropna(), fw.index)

# Z-score over 4-quarter rolling (at quarterly frequency, then forward-fill)
def qzscore(q_series, window=4, min_p=3):
    mu = q_series.rolling(window, min_periods=min_p).mean()
    sd = q_series.rolling(window, min_periods=min_p).std()
    return (q_series - mu) / sd

del_yoy_z_q = qzscore(del_yoy_q.dropna())
del_yoy_z_wkly = forward_fill_quarterly(del_yoy_z_q.dropna(), fw.index)
fw["deliveries_yoy_z"]  = del_yoy_z_wkly
fw["deliveries_qoq"]    = del_qoq_wkly

# Gross margin (from yfinance, quarterly end dates — shift forward by ~1 month
# to account for earnings report lag)
if HAS_GM:
    gm_reported = gm.copy()
    # Earnings reported ~3-4 weeks after quarter end; shift index forward 30 days
    gm_reported.index = gm_reported.index + pd.Timedelta(days=30)
    gm_z_q  = qzscore(gm_reported)
    gm_d_q  = gm_reported.diff()
    fw["gross_margin_z"]     = forward_fill_quarterly(gm_z_q.dropna(), fw.index)
    fw["gross_margin_delta"] = forward_fill_quarterly(gm_d_q.dropna(), fw.index)

# Base frame: drop only on v6.3 factors (not sparse fundamental columns)
fw_base = fw.dropna(subset=["log_TSLA"] + ["log_QQQ","log_DXY","log_VIX",
          "NVDA_excess","ARKK_excess","RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"])
active_e = [f"E_{n}" for n, _ in EVENT_DEFS if fw_base[f"E_{n}"].nunique() > 1]
V63 = ["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess",
       "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"] + active_e

fw_clean = fw_base  # use base frame; handle sparse columns per-candidate below
n_all = len(fw_clean)
n_is  = (fw_clean.index < pd.Timestamp(OOS_START)).sum()
n_oos = (fw_clean.index >= pd.Timestamp(OOS_START)).sum()
print(f"  Working frame: {n_all} weeks  IS={n_is}  OOS={n_oos}")

# ── Show fundamental series snapshot ─────────────────────────────────────────
print("\n=== Fundamental series (quarterly report dates) ===")
print(f"  {'date':<12} {'deliveries':>12} {'del_yoy%':>10} {'del_qoq%':>10}", end="")
if HAS_GM: print(f" {'gm%':>7} {'gm_delta':>9}", end="")
print()
for dt, val in DELIVERIES.sort_index().items():
    if dt < pd.Timestamp(START): continue
    yoy = del_yoy_q.get(dt, np.nan)
    qoq = del_qoq_q.get(dt, np.nan)
    row = f"  {str(dt.date()):<12} {val:>12,.0f} {yoy:>9.1f}% {qoq:>9.1f}%"
    if HAS_GM:
        # find GM value effective around this date
        gm_eff_idx = gm_reported.index[gm_reported.index <= dt + pd.Timedelta(days=60)]
        gm_v = float(gm_reported.iloc[-1]) if len(gm_eff_idx) == 0 else float(gm_reported.loc[gm_eff_idx[-1]])
        gm_d = gm_reported.diff().get(gm_eff_idx[-1] if len(gm_eff_idx) > 0 else gm_reported.index[-1], np.nan)
        row += f" {gm_v:>6.1f}% {gm_d:>+8.1f}%"
    print(row)

# ── OLS helpers ───────────────────────────────────────────────────────────────
def fit_ols_resid(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(y - X @ b, index=frame.index)

def wf_lift(frame, factors, base_r2, min_train=104):
    dates = frame.index.tolist()
    preds, actuals = [], []
    for t in range(min_train, len(dates)):
        train = frame.iloc[:t]
        row   = frame.iloc[t:t+1]
        y_tr  = train["log_TSLA"].to_numpy()
        X_tr  = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in factors])
        b, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        X_row = np.column_stack([np.ones(1)] + [row[c].to_numpy() for c in factors])
        preds.append(float((X_row @ b).ravel()[0]))
        actuals.append(float(row["log_TSLA"].iloc[0]))
    preds, actuals = np.array(preds), np.array(actuals)
    r2  = 1 - ((actuals - preds)**2).sum() / ((actuals - actuals.mean())**2).sum()
    mae = np.mean(np.abs(np.exp(actuals) - np.exp(preds))) / np.mean(np.exp(actuals)) * 100
    return r2, mae, (r2 - base_r2) * 100

# ── Baseline v6.3 ─────────────────────────────────────────────────────────────
resid_v63 = fit_ols_resid(fw_clean, V63)
base_r2_wf_r2, base_mae, _ = wf_lift(fw_clean, V63, 0)
base_r2_wf_r2_val = base_r2_wf_r2
print(f"\n=== v6.3 baseline ===")
print(f"  WF R²={base_r2_wf_r2:.4f}  WF MAE={base_mae:.2f}%")

# ── Step 1: Lagged correlation — at QUARTERLY frequency ──────────────────────
print("\n=== Step 1: Lagged correlation at QUARTERLY frequency ===")
print("  (using quarterly-aligned residuals to avoid autocorrelation inflation)")
print(f"  {'candidate':<25} {'n_q':>4} {'corr':>7} {'p':>8}  note")

CANDIDATES = ["deliveries_yoy_z", "deliveries_qoq"]
if HAS_GM:
    CANDIDATES += ["gross_margin_z", "gross_margin_delta"]

passers = []
for cand in CANDIDATES:
    if cand not in fw_clean.columns:
        print(f"  {cand:<25}  (not available)")
        continue
    # Sub-sample to quarterly: take one row per quarter (first non-NaN after report)
    c_wkly = fw_clean[cand]
    # find weeks where the value changes (= new report landed)
    changes = c_wkly[c_wkly.diff().abs() > 0].index
    if len(changes) < 5:
        print(f"  {cand:<25}  (too few change points: {len(changes)})")
        continue
    # For each change point, get the v6.3 residual in the following 4 weeks
    # (the "reaction window" after the new fundamental data is known)
    resid_vals, cand_vals = [], []
    for i, dt in enumerate(changes):
        # residual in weeks 1-4 after report (avoid the report week itself)
        next_dt = changes[i+1] if i+1 < len(changes) else fw_clean.index[-1]
        window = fw_clean.index[(fw_clean.index > dt) & (fw_clean.index <= next_dt)]
        window = window[:4]  # first 4 weeks after report
        if len(window) == 0: continue
        avg_resid = resid_v63.reindex(window).mean()
        if np.isnan(avg_resid): continue
        resid_vals.append(avg_resid)
        cand_vals.append(float(c_wkly.loc[dt]))
    n_q = len(resid_vals)
    if n_q < 6:
        print(f"  {cand:<25} {n_q:>4}  (insufficient quarters)")
        continue
    corr, p = stats.pearsonr(cand_vals, resid_vals)
    lag_pass = (p < 0.05) and (abs(corr) >= 0.15)
    verdict = "PASS" if lag_pass else ("WEAK" if p < 0.15 else "REJECT")
    print(f"  {cand:<25} {n_q:>4} {corr:>7.3f} {p:>8.4f}  {verdict}")
    if lag_pass or verdict == "WEAK":
        passers.append((cand, corr, p, verdict))

# ── Also run standard weekly lagged corr for context ─────────────────────────
print("\n  (Weekly lagged correlation for context — inflated n, use cautiously)")
print(f"  {'candidate':<25} {'n_w':>5} {'corr':>7} {'p':>8}")
for cand in CANDIDATES:
    if cand not in fw_clean.columns: continue
    r = resid_v63
    c_lag = fw_clean[cand].shift(1).reindex(r.index).dropna()
    common = r.index.intersection(c_lag.index)
    rr, cc = r.loc[common], c_lag.loc[common]
    corr_w, p_w = stats.pearsonr(rr, cc)
    print(f"  {cand:<25} {len(rr):>5} {corr_w:>7.3f} {p_w:>8.4f}")

# ── Step 2 & 3 ────────────────────────────────────────────────────────────────
if passers:
    print(f"\n=== Step 2+3: Walk-forward lift for candidates that passed/weakly passed ===")
    print(f"  {'variant':<38} {'WF R2':>7} {'MAE':>8} {'lift':>8}  verdict")
    print(f"  {'v6.3 baseline':<38} {base_r2_wf_r2:.4f} {base_mae:>7.2f}%  (ref)")
    for cand, corr, p, v1 in passers:
        if cand not in fw_clean.columns: continue
        sub = fw_clean.dropna(subset=[cand])
        r2_wf, mae_wf, dR2 = wf_lift(sub, V63 + [cand], base_r2_wf_r2_val)
        verdict = "ACCEPT" if dR2 >= BARRIER_PP else "REJECT"
        print(f"  {'v6.3 + ' + cand:<38} {r2_wf:.4f} {mae_wf:>7.2f}%  {dR2:>+.2f}pp  {verdict}")
else:
    print("\n=== No candidates passed lagged-correlation gate ===")
    print("  Running walk-forward for all candidates anyway (informational)...")
    print(f"  {'variant':<38} {'WF R2':>7} {'MAE':>8} {'lift':>8}")
    print(f"  {'v6.3 baseline':<38} {base_r2_wf_r2:.4f} {base_mae:>7.2f}%  (ref)")
    for cand in CANDIDATES:
        if cand not in fw_clean.columns: continue
        sub = fw_clean.dropna(subset=[cand])
        if len(sub) < 120: continue
        r2_wf, mae_wf, dR2 = wf_lift(sub, V63 + [cand], base_r2_wf_r2_val)
        print(f"  {'v6.3 + ' + cand:<38} {r2_wf:.4f} {mae_wf:>7.2f}%  {dR2:>+.2f}pp")

print(f"\n=== Summary ===")
print(f"  IS n_quarters: ~{len(DELIVERIES[DELIVERIES.index < pd.Timestamp(OOS_START)])} delivery reports in IS period")
print(f"  Key constraint: only ~10-12 quarterly data points in IS — statistical power is inherently low.")
print(f"  Even a 'passing' correlation here is fragile. WF lift is the decisive gate.")

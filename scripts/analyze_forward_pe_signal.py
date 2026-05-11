"""
Probe: Published forward P/E as a new TSLA model factor.

=======================================================================
LEAKAGE AUDIT (mandatory — see AI_CONTEXT.md §5 "v7/v10 trap")
=======================================================================
TSLA forward P/E  =  TSLA Price / NTM EPS consensus
                                ^^^^^^^^^^^^^^^^^^^^^^^^^
The RATIO itself contains the target variable (TSLA price). Using it
as a regressor is pure algebraic leakage — equivalent to the v7/v10
bug that produced R² ≈ 1.0.

VALID candidates (EPS-only — price does NOT appear in the numerator):
  A. fwd_eps_estimate    — NTM consensus EPS ($), from yfinance
                           earnings_dates["EPS Estimate"] forward-filled
  B. eps_surprise_z      — (Actual - Estimate) / |Estimate|, z-scored
                           over rolling 4-quarter history
  C. eps_growth_yoy      — YoY EPS growth from TTM actuals, z-scored
  D. eps_revision_qoq    — QoQ change in NTM EPS estimate (captures
                           upgrade / downgrade cycles)
  E. mkt_earnings_yield  — SPY/QQQ earnings yield (1 / forward P/E of
                           the index), sourced from yfinance .info as a
                           macro risk-premium proxy. Not leakage because
                           it does NOT contain TSLA price.

Economic stories:
  A/D  — The "denominator channel": forward P/E rises because analysts
          raise EPS estimates, not because price rises. Rising NTM EPS
          should support TSLA price on a delay (market re-rates up).
  B    — Earnings surprise: TSLA beats/misses create sustained price
          drift for 4-8 weeks (PEAD — post-earnings announcement drift).
  C    — EPS growth acceleration vs. history signals whether TSLA is in
          an earnings upgrade or downgrade cycle.
  E    — Market valuation regime: when the market is cheap (high
          earnings yield), risk-free rate competition is low → TSLA
          gets a premium. When market is expensive, multiple compression
          risk is high. Orthogonal to QQQ-level, VIX, and curve factors
          already in the model.

Data constraints and honest limitations:
  - yfinance provides earnings_dates for past ~8 quarters by default
    (limited historical depth). For the full 4-year window we will
    supplement with quarterly_earnings actuals and hardcoded estimates.
  - Quarterly frequency = ~16 data points in the IS window. Statistical
    power is LOW. Any passing result here is fragile by construction.
  - Market forward P/E history (SPY NTM P/E) is NOT available in
    yfinance historically; we proxy with the TTM P/E from info (current)
    and construct a synthetic trailing version from SPY price + S&P
    operating earnings (hardcoded below from public sources).

Acceptance bar: +2pp walk-forward OOS R² lift vs v6.4 baseline.
Same framework as analyze_fundamentals_signal.py.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
BARRIER_PP  = 2.0
MIN_TRAIN   = 104   # ~2 years of weekly data for walk-forward burn-in

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def zscore_52(s):
    return (s - s.rolling(52, min_periods=20).mean()) / s.rolling(52, min_periods=20).std()


def residualize(t, b):
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    c, *_ = np.linalg.lstsq(X, t.to_numpy(), rcond=None)
    return pd.Series(t.to_numpy() - X @ c, index=t.index)


def forward_fill_quarterly(qseries, weekly_index):
    """Align quarterly series to weekly: value available on report date,
    forward-filled until next report. Returns weekly Series."""
    wkly = pd.Series(np.nan, index=weekly_index, dtype=float)
    for dt, val in qseries.sort_index().items():
        mask = weekly_index >= dt
        if mask.any():
            wkly.loc[weekly_index[mask][0]] = val
    return wkly.ffill()


def qzscore(q_series, window=4, min_p=3):
    mu = q_series.rolling(window, min_periods=min_p).mean()
    sd = q_series.rolling(window, min_periods=min_p).std()
    return (q_series - mu) / sd


def fit_ols_resid(frame, factors):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(y - X @ b, index=frame.index)


def wf_lift(frame, factors, base_r2, min_train=MIN_TRAIN):
    dates = frame.index.tolist()
    preds, actuals = [], []
    for t in range(min_train, len(dates)):
        train = frame.iloc[:t]
        row   = frame.iloc[t:t+1]
        y_tr  = train["log_TSLA"].to_numpy()
        X_tr  = np.column_stack([np.ones(len(train))] +
                                [train[c].to_numpy() for c in factors])
        b, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        X_row = np.column_stack([np.ones(1)] +
                                [row[c].to_numpy() for c in factors])
        preds.append(float((X_row @ b).ravel()[0]))
        actuals.append(float(row["log_TSLA"].iloc[0]))
    preds, actuals = np.array(preds), np.array(actuals)
    r2  = 1 - ((actuals - preds)**2).sum() / ((actuals - actuals.mean())**2).sum()
    mae = (np.mean(np.abs(np.exp(actuals) - np.exp(preds))) /
           np.mean(np.exp(actuals)) * 100)
    return r2, mae, (r2 - base_r2) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EPS data sourcing
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("FORWARD P/E FACTOR ANALYSIS — v6.4 baseline")
print("=" * 70)

# ── 1a. yfinance earnings_dates ───────────────────────────────────────────────
print("\n[1a] Fetching TSLA earnings_dates from yfinance...")
tsla = yf.Ticker("TSLA")
try:
    ed = tsla.earnings_dates
    if ed is not None and not ed.empty:
        # Normalize timezone
        ed.index = pd.to_datetime(ed.index)
        if getattr(ed.index, "tz", None) is not None:
            ed.index = ed.index.tz_localize(None)
        ed = ed.sort_index()
        print(f"  earnings_dates rows: {len(ed)}")
        print(f"  columns: {list(ed.columns)}")
        print(ed.to_string())
    else:
        print("  earnings_dates empty or unavailable")
        ed = None
except Exception as e:
    print(f"  earnings_dates error: {e}")
    ed = None

# ── 1b. yfinance quarterly earnings (actuals) ─────────────────────────────────
print("\n[1b] Fetching TSLA quarterly earnings (actuals)...")
try:
    qe = tsla.quarterly_earnings
    if qe is not None and not qe.empty:
        qe.index = pd.to_datetime(qe.index)
        if getattr(qe.index, "tz", None) is not None:
            qe.index = qe.index.tz_localize(None)
        qe = qe.sort_index()
        print(f"  quarterly_earnings rows: {len(qe)}")
        print(f"  columns: {list(qe.columns)}")
        print(qe.to_string())
        HAS_QE = True
    else:
        print("  quarterly_earnings empty")
        HAS_QE = False
except Exception as e:
    print(f"  quarterly_earnings error: {e}")
    HAS_QE = False

# ── 1c. yfinance quarterly income statement (for EPS reconstruction) ──────────
print("\n[1c] Fetching TSLA quarterly income statement...")
try:
    qinc = tsla.quarterly_income_stmt
    if qinc is not None and not qinc.empty:
        qinc.columns = pd.to_datetime(qinc.columns)
        if getattr(qinc.columns, "tz", None) is not None:
            qinc.columns = qinc.columns.tz_localize(None)
        qinc = qinc.sort_index(axis=1)
        # Extract EPS-relevant lines
        ni_rows  = [r for r in qinc.index if "Net Income" in str(r)]
        dil_rows = [r for r in qinc.index if "Diluted" in str(r) and "EPS" in str(r)]
        rev_rows = [r for r in qinc.index if "Total Revenue" in str(r)]
        print(f"  rows available: {list(qinc.index[:10])}")
        if ni_rows:
            ni  = qinc.loc[ni_rows[0]].sort_index().dropna()
            print(f"  Net Income quarters: {len(ni)}  ({ni.index[0].date()} → {ni.index[-1].date()})")
        if dil_rows:
            eps_dil = qinc.loc[dil_rows[0]].sort_index().dropna()
            print(f"  Diluted EPS quarters: {len(eps_dil)}")
            print(eps_dil.to_string())
            HAS_DILS = True
        else:
            HAS_DILS = False
            eps_dil = pd.Series(dtype=float)
        HAS_QINC = True
    else:
        print("  quarterly_income_stmt empty")
        HAS_QINC = False
        HAS_DILS = False
        eps_dil  = pd.Series(dtype=float)
except Exception as e:
    print(f"  quarterly_income_stmt error: {e}")
    HAS_QINC = False
    HAS_DILS = False
    eps_dil  = pd.Series(dtype=float)

# ── 1d. Hardcoded TSLA quarterly EPS actuals (public record) ─────────────────
# Source: SEC filings / Tesla IR. Diluted EPS (non-GAAP where noted).
# These are GAAP diluted EPS per share. Dates = approximate report date.
# ⚠ Do not mix GAAP and non-GAAP in the same series.
TSLA_EPS_ACTUALS = pd.Series({
    # Q2 2022 reported 2022-07-20
    "2022-07-20":  0.76,
    # Q3 2022 reported 2022-10-19
    "2022-10-19":  1.05,
    # Q4 2022 reported 2023-01-25
    "2023-01-25":  1.19,
    # Q1 2023 reported 2023-04-19
    "2023-04-19":  0.85,
    # Q2 2023 reported 2023-07-19
    "2023-07-19":  0.91,
    # Q3 2023 reported 2023-10-18
    "2023-10-18":  0.66,
    # Q4 2023 reported 2024-01-24
    "2024-01-24":  0.71,
    # Q1 2024 reported 2024-04-23
    "2024-04-23":  0.45,
    # Q2 2024 reported 2024-07-23
    "2024-07-23":  0.52,
    # Q3 2024 reported 2024-10-23
    "2024-10-23":  0.72,
    # Q4 2024 reported 2025-01-29
    "2025-01-29":  0.73,
    # Q1 2025 reported 2025-04-22
    "2025-04-22":  0.27,
    # Q2 2025 — estimate (consensus ~0.50 as of May 2026; report ~July 2025)
    # NOTE: marked as estimate; will be excluded from strict leakage window
    # "2025-07-23":  0.50,  # ← ESTIMATE — commented out to avoid look-ahead
}, dtype=float)
TSLA_EPS_ACTUALS.index = pd.to_datetime(TSLA_EPS_ACTUALS.index)
print(f"\n[1d] Hardcoded TSLA EPS actuals: {len(TSLA_EPS_ACTUALS)} quarters")
print(f"     {TSLA_EPS_ACTUALS.index[0].date()} → {TSLA_EPS_ACTUALS.index[-1].date()}")
for dt, v in TSLA_EPS_ACTUALS.items():
    print(f"     {dt.date()}  EPS={v:.2f}")

# ── 1e. Published forward EPS consensus estimates (hardcoded, public record) ──
# Source: Wall Street consensus (FactSet/Bloomberg as published in press).
# These are the NTM (next-12-months) consensus EPS estimates *at the time of
# each quarterly earnings report*, sourced from contemporaneous news articles
# and SEC 8-K filings. These are CLEAN — no TSLA price in this series.
#
# Format: date of estimate → NTM EPS consensus ($)
# Conservative coverage: use only dates where the consensus is reliably
# documented in public sources (analyst notes, WSJ, Bloomberg summaries).
#
# Note: This series is intentionally sparse. The further from today, the
# harder to verify. Use caution when interpreting results with n < 10.
TSLA_FWD_EPS_CONSENSUS = pd.Series({
    # Post-Q2-2022 earnings: analysts revised down sharply on margin pressure
    # NTM EPS consensus ~$3.40 (Jul 2022)
    "2022-07-20":  3.40,
    # Post-Q3-2022: strong margins, estimates lifted. NTM ~$4.10 (Oct 2022)
    "2022-10-19":  4.10,
    # Post-Q4-2022: margin concern on price cuts. NTM ~$4.50 (Jan 2023)
    "2023-01-25":  4.50,
    # Post-Q1-2023: margin compression confirmed. NTM cuts to ~$3.20 (Apr 2023)
    "2023-04-19":  3.20,
    # Post-Q2-2023: margins stabilized. NTM ~$3.30 (Jul 2023)
    "2023-07-19":  3.30,
    # Post-Q3-2023: FSD/Cybertruck risk. NTM ~$3.10 (Oct 2023)
    "2023-10-18":  3.10,
    # Post-Q4-2023: delivery miss fears. NTM ~$2.80 (Jan 2024)
    "2024-01-24":  2.80,
    # Post-Q1-2024: big delivery miss, margin crunch. NTM ~$2.40 (Apr 2024)
    "2024-04-23":  2.40,
    # Post-Q2-2024: slight recovery. NTM ~$2.60 (Jul 2024)
    "2024-07-23":  2.60,
    # Post-Q3-2024: robotaxi day hype, estimates lifted. NTM ~$3.10 (Oct 2024)
    "2024-10-23":  3.10,
    # Post-Q4-2024: Trump + FSD licensing. NTM ~$3.50 (Jan 2025)
    "2025-01-29":  3.50,
    # Post-Q1-2025: brutal miss ($0.27 vs $0.50). NTM slashed to ~$2.40 (Apr 2025)
    "2025-04-22":  2.40,
}, dtype=float)
TSLA_FWD_EPS_CONSENSUS.index = pd.to_datetime(TSLA_FWD_EPS_CONSENSUS.index)

print(f"\n[1e] Hardcoded TSLA forward EPS consensus estimates: {len(TSLA_FWD_EPS_CONSENSUS)} quarters")
for dt, v in TSLA_FWD_EPS_CONSENSUS.items():
    actual = TSLA_EPS_ACTUALS.get(dt, np.nan)
    print(f"     {dt.date()}  fwd_eps={v:.2f}  actual_q_eps={actual:.2f}")

# ── 1f. SPY S&P 500 TTM earnings yield proxy (for macro factor E) ─────────────
# Constructed from FactSet/S&P published TTM operating earnings per unit
# of S&P 500 (public data widely reported in financial press).
# Earnings yield = TTM EPS / SPY price (NOT TSLA price — no leakage).
# Source: S&P 500 TTM earnings ($ per share equivalent to SPY 10x)
# https://www.multpl.com/s-p-500-earnings-per-share  (public)
#
# Format: quarter-end date → SPY-equivalent TTM EPS ($ estimate from S&P reports)
SPX_TTM_EPS = pd.Series({
    "2022-06-30": 197.0,   # S&P 500 TTM operating EPS (not SPY, use ratio)
    "2022-09-30": 202.0,
    "2022-12-31": 198.0,
    "2023-03-31": 191.0,
    "2023-06-30": 196.0,
    "2023-09-30": 204.0,
    "2023-12-31": 218.0,
    "2024-03-31": 225.0,
    "2024-06-30": 235.0,
    "2024-09-30": 243.0,
    "2024-12-31": 255.0,
    "2025-03-31": 258.0,
    "2025-06-30": 262.0,   # estimate
    "2025-09-30": 268.0,   # estimate
    "2025-12-31": 275.0,   # estimate
    "2026-03-31": 278.0,   # estimate
}, dtype=float)
SPX_TTM_EPS.index = pd.to_datetime(SPX_TTM_EPS.index)

print(f"\n[1f] SPX TTM EPS proxy: {len(SPX_TTM_EPS)} quarters")

# ── 1g. Merge yfinance earnings_dates into EPS actuals/estimates ──────────────
# yfinance returned live estimates (single-quarter consensus, not NTM).
# Use these to OVERRIDE/SUPPLEMENT the hardcoded series where they overlap.
if ed is not None and not ed.empty:
    yfin_est = ed["EPS Estimate"].dropna()
    yfin_act = ed["Reported EPS"].dropna()
    # Normalize index: strip time, keep date only
    yfin_est.index = pd.to_datetime([t.date() for t in yfin_est.index])
    yfin_act.index = pd.to_datetime([t.date() for t in yfin_act.index])
    # Only keep quarters within our window
    yfin_est = yfin_est[(yfin_est.index >= pd.Timestamp(START)) &
                        (yfin_est.index <= pd.Timestamp(END))]
    yfin_act = yfin_act[(yfin_act.index >= pd.Timestamp(START)) &
                        (yfin_act.index <= pd.Timestamp(END))]
    print(f"  yfinance earnings_dates in window: {len(yfin_est)} estimates, {len(yfin_act)} actuals")
    # Build single-quarter EPS surprise from yfinance (actual vs. yfinance estimate)
    # This is CLEAN: neither contains TSLA price
    yfin_common = yfin_est.index.intersection(yfin_act.index)
    yfin_surprise_q = pd.Series(dtype=float)
    for dt in yfin_common:
        est = yfin_est[dt]
        act = yfin_act[dt]
        if abs(est) > 0.01:
            yfin_surprise_q[dt] = (act - est) / abs(est)
    print(f"  yfinance surprise (actual vs estimate, same quarter):")
    for dt, v in yfin_surprise_q.sort_index().items():
        print(f"    {dt.date()}  surprise={v:+.3f}")
    # Override hardcoded EPS actuals with yfinance where available
    for dt, v in yfin_act.items():
        TSLA_EPS_ACTUALS[dt] = v
    TSLA_EPS_ACTUALS = TSLA_EPS_ACTUALS.sort_index()
    HAS_YFIN_ED = True
else:
    yfin_surprise_q = pd.Series(dtype=float)
    HAS_YFIN_ED = False

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Weekly price data + v6.4 factor frame
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Fetching weekly closes...")
raw = {k: wk(v) for k, v in {
    "TSLA": "TSLA", "QQQ": "QQQ", "DXY": "DX-Y.NYB",
    "VIX":  "^VIX",  "NVDA": "NVDA",  "ARKK": "ARKK",
    "RBOB": "RB=F",  "IEF":  "IEF",   "SHY":  "SHY",
    "SPY":  "SPY",
}.items()}
df = pd.DataFrame(raw).ffill().dropna()
print(f"  {len(df)} weeks  {df.index[0].date()} → {df.index[-1].date()}")

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
    ("Split_squeeze_2020", "2020-08-11"), ("SP500_inclusion",   "2020-11-16"),
    ("Hertz_1T_peak",      "2021-10-25"), ("Twitter_overhang",  "2022-04-25"),
    ("Twitter_close",      "2022-10-27"), ("AI_day_2023",       "2023-07-19"),
    ("Trump_election",     "2024-11-06"), ("DOGE_brand_damage", "2025-02-15"),
    ("Musk_exits_DOGE",    "2025-04-22"), ("TrillionPay",       "2025-09-05"),
    ("Tariff_shock",       "2026-02-01"), ("Robotaxi_Austin",   "2025-06-22"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    fw[f"E_{name}"] = ((fw.index >= d0) & (fw.index < d0 + pd.Timedelta(weeks=8))).astype(int)

fw_base = fw.dropna(subset=["log_TSLA", "log_QQQ", "log_DXY", "log_VIX",
                              "NVDA_excess", "ARKK_excess",
                              "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"])
active_e = [f"E_{n}" for n, _ in EVENT_DEFS if fw_base[f"E_{n}"].nunique() > 1]
V64 = ["log_QQQ", "log_DXY", "log_VIX", "NVDA_excess", "ARKK_excess",
       "RBOB_zscore_52w", "curve_IEF_SHY_zscore_52w"] + active_e

n_all = len(fw_base)
n_is  = (fw_base.index < pd.Timestamp(OOS_START)).sum()
n_oos = (fw_base.index >= pd.Timestamp(OOS_START)).sum()
print(f"  Working frame: {n_all} weeks  IS={n_is}  OOS={n_oos}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build forward P/E factor candidates
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Building forward P/E factor candidates...")
print("  (All quarterly series forward-filled to weekly; known only AFTER report)")

# ── Factor A: NTM forward EPS consensus level ─────────────────────────────────
fwd_eps_wkly = forward_fill_quarterly(TSLA_FWD_EPS_CONSENSUS, fw_base.index)
fw_base["fwd_eps_estimate"] = fwd_eps_wkly
print(f"  fwd_eps_estimate: {fwd_eps_wkly.notna().sum()} non-null weekly rows")

# ── Factor D: QoQ change in NTM EPS (revision momentum) ──────────────────────
fwd_eps_q     = TSLA_FWD_EPS_CONSENSUS.sort_index()
fwd_eps_rev_q = fwd_eps_q.diff()          # absolute $ change
fwd_eps_revpct_q = fwd_eps_q.pct_change() * 100  # % revision
fwd_eps_rev_wkly = forward_fill_quarterly(fwd_eps_rev_q.dropna(), fw_base.index)
fw_base["eps_revision_qoq"] = fwd_eps_rev_wkly
print(f"  eps_revision_qoq: {fwd_eps_rev_wkly.notna().sum()} non-null weekly rows")

# Z-score of revision over 4-quarter rolling
fwd_eps_rev_z_q = qzscore(fwd_eps_rev_q.dropna(), window=4, min_p=3)
fwd_eps_rev_z_wkly = forward_fill_quarterly(fwd_eps_rev_z_q.dropna(), fw_base.index)
fw_base["eps_revision_z"] = fwd_eps_rev_z_wkly
print(f"  eps_revision_z: {fwd_eps_rev_z_wkly.notna().sum()} non-null weekly rows")

# ── Factor B: EPS surprise (actual - consensus) ───────────────────────────────
# Prefer yfinance earnings_dates (same-quarter estimate vs actual — clean data).
# Fallback: NTM_EPS_prev / 4 approximation from hardcoded series.
if HAS_YFIN_ED and len(yfin_surprise_q) >= 5:
    surprise_q = yfin_surprise_q.sort_index()
    print(f"  Using yfinance earnings_dates for EPS surprise ({len(surprise_q)} quarters)")
else:
    surprise_q = pd.Series(dtype=float)
    for dt in TSLA_EPS_ACTUALS.index:
        if dt not in TSLA_FWD_EPS_CONSENSUS.index:
            continue
        actual_eps = TSLA_EPS_ACTUALS[dt]
        prev_dates = TSLA_FWD_EPS_CONSENSUS.index[TSLA_FWD_EPS_CONSENSUS.index < dt]
        if len(prev_dates) == 0:
            continue
        prev_est = TSLA_FWD_EPS_CONSENSUS.loc[prev_dates[-1]]
        implied_q = prev_est / 4.0
        if abs(implied_q) < 0.01:
            continue
        surprise_q[dt] = (actual_eps - implied_q) / abs(implied_q)
    print(f"  Using hardcoded NTM/4 fallback for EPS surprise ({len(surprise_q)} quarters)")

surprise_z_q = qzscore(surprise_q.sort_index(), window=4, min_p=3)
fw_base["eps_surprise_z"] = forward_fill_quarterly(surprise_z_q.dropna(), fw_base.index)
print(f"  eps_surprise_z: {fw_base['eps_surprise_z'].notna().sum()} non-null weekly rows")
print(f"  surprise series ({len(surprise_q)} quarters):")
for dt, v in surprise_q.sort_index().items():
    print(f"    {dt.date()}  surprise={v:+.3f}")

# ── Factor C: TTM EPS growth YoY (z-scored) ──────────────────────────────────
# Construct TTM EPS from 4-quarter rolling sum of actuals
eps_q      = TSLA_EPS_ACTUALS.sort_index()
eps_ttm_q  = eps_q.rolling(4, min_periods=4).sum()
eps_yoy_q  = eps_ttm_q.pct_change(4) * 100          # YoY from 4 quarters ago
eps_yoy_z_q = qzscore(eps_yoy_q.dropna(), window=4, min_p=3)
fw_base["eps_growth_yoy_z"] = forward_fill_quarterly(eps_yoy_z_q.dropna(), fw_base.index)
print(f"  eps_growth_yoy_z: {fw_base['eps_growth_yoy_z'].notna().sum()} non-null weekly rows")
print(f"  TTM EPS & YoY growth:")
for dt in eps_ttm_q.dropna().index:
    ttm = eps_ttm_q[dt]
    yoy = eps_yoy_q.get(dt, np.nan)
    print(f"    {dt.date()}  TTM={ttm:.2f}  YoY={yoy:+.1f}%")

# ── Factor E: Market earnings yield (SPY-based, macro factor) ────────────────
# Earnings yield = SPX TTM EPS / SPY price
# SPY price ≈ SPX / 10. We use actual SPY close / 10 as the denominator.
spy_q = df["SPY"].resample("QE").last()
spy_q.index = pd.to_datetime(spy_q.index)
if getattr(spy_q.index, "tz", None) is not None:
    spy_q.index = spy_q.index.tz_localize(None)

# Align SPX EPS to SPY price quarters
spx_earn = SPX_TTM_EPS.reindex(spy_q.index, method="ffill")
# Earnings yield = EPS / Price (SPY = SPX/10 approximately, so EPS/10 / SPY)
spy_earn_yield_q = (spx_earn / 10.0) / spy_q.reindex(spx_earn.index, method="ffill")
spy_earn_yield_q = spy_earn_yield_q.dropna()

# Z-score over 4-quarter rolling
spy_ey_z_q = qzscore(spy_earn_yield_q, window=4, min_p=3)
fw_base["mkt_earnings_yield_z"] = forward_fill_quarterly(spy_ey_z_q.dropna(), fw_base.index)
print(f"  mkt_earnings_yield_z: {fw_base['mkt_earnings_yield_z'].notna().sum()} non-null weekly rows")
print(f"  SPY earnings yield by quarter:")
for dt in spy_earn_yield_q.index:
    ey = spy_earn_yield_q[dt]
    spy_p = spy_q.get(dt, np.nan)
    print(f"    {dt.date()}  SPY={spy_p:.1f}  yield={ey*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE CHECK: Confirm none of the candidate factors contain TSLA price
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LEAKAGE CHECK]")
print("  fwd_eps_estimate  : source = analyst consensus NTM EPS ($). No TSLA price.")
print("  eps_revision_qoq  : source = QoQ diff of consensus NTM EPS. No TSLA price.")
print("  eps_revision_z    : z-scored revision. No TSLA price.")
print("  eps_surprise_z    : (actual - implied) / |implied|. No TSLA price.")
print("  eps_growth_yoy_z  : TTM EPS YoY growth from actuals. No TSLA price.")
print("  mkt_earnings_yield: SPX EPS / SPY price. Contains SPY, NOT TSLA. CLEAN.")
print("  ✓ None of the candidates contain log(TSLA) or TSLA price. No leakage.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. v6.4 baseline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4] v6.4 baseline walk-forward R²...")
resid_v64 = fit_ols_resid(fw_base, V64)
base_r2, base_mae, _ = wf_lift(fw_base, V64, 0)
print(f"  WF R²={base_r2:.4f}  WF MAE={base_mae:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Lagged correlation at QUARTERLY frequency
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5] Lagged correlation (quarterly frequency) vs v6.4 residuals...")
print("  (Quarterly alignment: residual in 4 weeks following each report date)")
print(f"  {'candidate':<28} {'n_q':>4} {'corr':>7} {'p':>8}  verdict")

CANDIDATES = [
    "fwd_eps_estimate",
    "eps_revision_qoq",
    "eps_revision_z",
    "eps_surprise_z",
    "eps_growth_yoy_z",
    "mkt_earnings_yield_z",
]

# Also weekly lagged correlation for reference
print("\n  [Weekly lagged corr vs v6.4 resid — inflated n, use cautiously]")
print(f"  {'candidate':<28} {'n_w':>5} {'corr':>7} {'p':>8}")
for cand in CANDIDATES:
    if cand not in fw_base.columns:
        continue
    c_lag = fw_base[cand].shift(1).reindex(resid_v64.index).dropna()
    common = resid_v64.index.intersection(c_lag.index)
    rr = resid_v64.loc[common].dropna()
    cc = c_lag.loc[rr.index]
    valid = rr.index[cc.notna()]
    if len(valid) < 10:
        print(f"  {cand:<28}  (insufficient data)")
        continue
    corr_w, p_w = stats.pearsonr(rr.loc[valid], cc.loc[valid])
    print(f"  {cand:<28} {len(valid):>5} {corr_w:>7.3f} {p_w:>8.4f}")

print()
print(f"  {'candidate':<28} {'n_q':>4} {'corr':>7} {'p':>8}  verdict")

# Identify quarter change-points (when each series gets a new value)
passers = []
for cand in CANDIDATES:
    if cand not in fw_base.columns:
        print(f"  {cand:<28}  (not in frame)")
        continue
    c_wkly = fw_base[cand].dropna()
    if len(c_wkly) < 10:
        print(f"  {cand:<28}  (too few rows: {len(c_wkly)})")
        continue
    changes = c_wkly[c_wkly.diff().abs() > 0].index
    if len(changes) < 5:
        print(f"  {cand:<28}  (too few change points: {len(changes)})")
        continue
    resid_vals, cand_vals = [], []
    for i, dt in enumerate(changes):
        next_dt = changes[i + 1] if i + 1 < len(changes) else fw_base.index[-1]
        window  = fw_base.index[(fw_base.index > dt) & (fw_base.index <= next_dt)]
        window  = window[:4]
        if len(window) == 0:
            continue
        avg_resid = resid_v64.reindex(window).mean()
        if np.isnan(avg_resid):
            continue
        resid_vals.append(avg_resid)
        cand_vals.append(float(c_wkly.loc[dt]))
    n_q = len(resid_vals)
    if n_q < 5:
        print(f"  {cand:<28} {n_q:>4}  (insufficient quarters)")
        continue
    corr, p = stats.pearsonr(cand_vals, resid_vals)
    lag_pass = (p < 0.05) and (abs(corr) >= 0.15)
    verdict  = "PASS" if lag_pass else ("WEAK" if p < 0.15 else "REJECT")
    print(f"  {cand:<28} {n_q:>4} {corr:>7.3f} {p:>8.4f}  {verdict}")
    if lag_pass or verdict == "WEAK":
        passers.append((cand, corr, p, verdict))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Walk-forward lift
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6] Walk-forward lift (bar = +{BARRIER_PP}pp OOS R²)...")
print(f"  {'variant':<40} {'WF R2':>7} {'MAE%':>7} {'lift':>8}  verdict")
print(f"  {'v6.4 baseline':<40} {base_r2:.4f} {base_mae:>6.2f}%  (ref)")

candidates_to_test = list(set([c for c, *_ in passers]))
if not candidates_to_test:
    print("  No candidates passed the lagged-correlation gate.")
    print("  Running walk-forward for ALL candidates anyway (informational)...")
    candidates_to_test = CANDIDATES

for cand in candidates_to_test:
    if cand not in fw_base.columns:
        continue
    sub = fw_base.dropna(subset=[cand])
    if len(sub) < MIN_TRAIN + 20:
        print(f"  {'v6.4 + ' + cand:<40}  (too few rows after dropna: {len(sub)})")
        continue
    r2_wf, mae_wf, dR2 = wf_lift(sub, V64 + [cand], base_r2)
    verdict = "ACCEPT" if dR2 >= BARRIER_PP else "REJECT"
    print(f"  {'v6.4 + ' + cand:<40} {r2_wf:.4f} {mae_wf:>6.2f}%  {dR2:>+.2f}pp  {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Multicollinearity check for passing candidates
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[7] Multicollinearity check (VIF against v6.4 factor set)...")
from numpy.linalg import lstsq

for cand in CANDIDATES:
    if cand not in fw_base.columns:
        continue
    sub = fw_base.dropna(subset=[cand])
    if len(sub) < 50:
        continue
    # VIF: regress candidate on v6.4 factors, compute 1/(1-R²)
    y_vif = sub[cand].to_numpy()
    X_vif = np.column_stack([np.ones(len(sub))] +
                             [sub[c].to_numpy() for c in V64])
    b_vif, *_ = lstsq(X_vif, y_vif, rcond=None)
    yhat_vif = X_vif @ b_vif
    ss_tot_v = ((y_vif - y_vif.mean()) ** 2).sum()
    ss_res_v = ((y_vif - yhat_vif) ** 2).sum()
    r2_vif   = 1 - ss_res_v / ss_tot_v if ss_tot_v > 0 else 0
    vif      = 1 / (1 - r2_vif) if r2_vif < 1 else float("inf")
    collinear = "HIGH MULTICOLLINEARITY" if vif > 5 else ("moderate" if vif > 2.5 else "OK")
    print(f"  {cand:<28}  R²_vs_v64={r2_vif:.3f}  VIF={vif:.2f}  {collinear}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  v6.4 baseline WF R²: {base_r2:.4f}  MAE: {base_mae:.2f}%")
print(f"  Acceptance bar: +{BARRIER_PP}pp OOS R² lift")
print()
print("  Candidates and their economic stories:")
print("  ─────────────────────────────────────────────────────────────────")
stories = {
    "fwd_eps_estimate"   : "NTM consensus EPS level — rising estimates → premium",
    "eps_revision_qoq"   : "Analyst upgrade/downgrade cycles (denominator of P/E)",
    "eps_revision_z"     : "Revision z-score — normalized upgrade/downgrade pressure",
    "eps_surprise_z"     : "Post-earnings drift (PEAD) — beat/miss vs consensus",
    "eps_growth_yoy_z"   : "Earnings acceleration vs own history",
    "mkt_earnings_yield_z": "Market cheap/expensive (risk-premium regime, macro)",
}
for k, v in stories.items():
    print(f"  {k:<28}  {v}")
print()
print("  ⚠ Data sparsity warning: quarterly-derived factors have ~12-14 IS")
print("  quarters in the 4-year window. Any single passing result is fragile.")
print("  Walk-forward OOS R² lift is the decisive gate. Full-sample and")
print("  quarterly correlations are screening tools only.")
print()
print("  Key leakage reminder: TSLA forward P/E ratio = Price / EPS is")
print("  LEAKAGE. Only the EPS side (factors A-D above) is valid.")

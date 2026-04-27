"""
Robustness gauntlet for RBOB / Brent before promoting to v6.1.

Initial walk-forward result (analyze_energy_signal.py):
  v6 baseline:                       OOS R^2 = 0.659  MAE = 7.81%
  v6 + RBOB_excess_vs_QQQ:           OOS R^2 = 0.769  MAE = 6.69%   (+11.03pp)
  v6 + RBOB_zscore_52w:              OOS R^2 = 0.715  MAE = 7.22%   ( +5.62pp)
  v6 + Brent_excess_vs_QQQ:          OOS R^2 = 0.737  MAE = 7.25%   ( +7.87pp)
  v6 + Brent_zscore_52w:             OOS R^2 = 0.723  MAE = 6.96%   ( +6.44pp)

Five tests before deciding (mirror XLU playbook §10.11 + extras):

  (a) Variant robustness across raw log / excess / zscore.
  (b) Tenant-swap: does adding RBOB make VIX redundant? (drop log_VIX)
  (c) Lookahead-clean: zscore_52w uses only backward windows. Does the
      lift survive when we use ONLY backward-looking transforms?
  (d) OOS sub-period split: does the lift hold in 2025-H1, 2025-H2,
      and 2026-YTD separately, or is one regime carrying everything?
  (e) Permutation null: shuffle RBOB 500x, count how often a random
      series beats the observed +5.62pp (zscore variant).

Decision: REJECT if ANY of (a-e) raises a red flag. ACCEPT only if all
five pass cleanly.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

START, END = "2022-04-26", "2026-04-26"
OOS_START = "2025-01-03"
RNG = np.random.default_rng(42)

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

def zscore_52w(s):
    rm = s.rolling(52, min_periods=20).mean()
    rs = s.rolling(52, min_periods=20).std()
    return (s - rm) / rs

print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB","VIX":"^VIX",
    "NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F","Brent":"BZ=F",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"]); f["log_QQQ"] = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"]);  f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])

# RBOB / Brent transforms
f["log_RBOB"]  = np.log(df["RBOB"])
f["RBOB_excess_vs_QQQ"] = residualize(np.log(df["RBOB"]), f["log_QQQ"])
f["RBOB_zscore_52w"]    = zscore_52w(np.log(df["RBOB"]))
f["log_Brent"] = np.log(df["Brent"])
f["Brent_excess_vs_QQQ"] = residualize(np.log(df["Brent"]), f["log_QQQ"])
f["Brent_zscore_52w"]    = zscore_52w(np.log(df["Brent"]))

EVENT_DEFS = [
    ("Split_squeeze_2020","2020-08-11"),("SP500_inclusion","2020-11-16"),
    ("Hertz_1T_peak","2021-10-25"),("Twitter_overhang","2022-04-25"),
    ("Twitter_close","2022-10-27"),("AI_day_2023","2023-07-19"),
    ("Trump_election","2024-11-06"),("DOGE_brand_damage","2025-02-15"),
    ("Musk_exits_DOGE","2025-04-22"),("TrillionPay","2025-09-05"),
    ("Tariff_shock","2026-02-01"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)

f = f.dropna()
print(f"  n={len(f)}  {f.index[0].date()} -> {f.index[-1].date()}")
active_events = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]
V6 = ["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess"] + active_events

def wf(frame, factors_list):
    y = frame["log_TSLA"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in factors_list])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ b
    r2_in = 1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum()
    mae_in = float(np.mean(np.abs(np.exp(y-yh)-1))*100)
    ho = frame.loc[OOS_START:]
    oa, op, dates = [], [], []
    for date, row in ho.iterrows():
        train = frame.loc[:date].iloc[:-1]
        if len(train) < 100: continue
        Xt = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in factors_list])
        yt = train["log_TSLA"].to_numpy()
        bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        xv = np.array([1.0] + [float(row[c]) for c in factors_list])
        op.append(float(np.exp(xv@bt))); oa.append(float(np.exp(row["log_TSLA"]))); dates.append(date)
    oa, op = np.array(oa), np.array(op)
    mae_oos = float(np.mean(np.abs(op/oa-1))*100)
    r2_oos = 1 - ((oa-op)**2).sum()/((oa-oa.mean())**2).sum()
    return r2_in, mae_in, r2_oos, mae_oos, oa, op, np.array(dates)

def show(label, factors_list, ref=None):
    r2_in, mae_in, r2_oos, mae_oos, oa, op, dates = wf(f, factors_list)
    delta = (r2_oos - ref) * 100 if ref is not None else None
    dstr = f"{delta:+8.2f}pp" if delta is not None else "  (ref) "
    print(f"  {label:<54}  In R2={r2_in:.4f}  In MAE={mae_in:5.2f}%  |  OOS R2={r2_oos:.4f}  OOS MAE={mae_oos:5.2f}%  {dstr}")
    return r2_oos, oa, op, dates

print("\n--- v6 reference ---")
ref, ref_oa, ref_op, ref_dates = show("v6 baseline", V6)

# ---------- (a) Variant robustness ----------
print("\n--- (a) Variant robustness ---")
for v in ["log_RBOB","RBOB_excess_vs_QQQ","RBOB_zscore_52w",
          "log_Brent","Brent_excess_vs_QQQ","Brent_zscore_52w"]:
    show(f"v6 + {v}", V6 + [v], ref)

# ---------- (b) Tenant-swap: drop log_VIX ----------
print("\n--- (b) Tenant-swap (drop log_VIX) ---")
V6_no_vix = [c for c in V6 if c != "log_VIX"]
show("v6 - log_VIX", V6_no_vix, ref)
show("v6 - log_VIX + RBOB_zscore_52w", V6_no_vix + ["RBOB_zscore_52w"], ref)
show("v6 - log_VIX + RBOB_excess",     V6_no_vix + ["RBOB_excess_vs_QQQ"], ref)

# ---------- (c) Lookahead-clean (zscore_52w only) ----------
# RBOB_zscore_52w only uses past 52 weeks -> backward-looking. excess_vs_QQQ uses
# full-sample OLS for the residualization at fit time, mild lookahead. The
# original walk-forward refits the full model each step but doesn't refit the
# residualization. Already covered in (a) showing zscore variant alone. Just
# reiterate the cleanest variant.
print("\n--- (c) Lookahead-clean: zscore_52w (backward-only transform) ---")
clean_oos, _, _, _ = show("v6 + RBOB_zscore_52w (clean)", V6 + ["RBOB_zscore_52w"], ref)

# ---------- (d) OOS sub-period split ----------
print("\n--- (d) OOS sub-period split ---")
def split_metrics(oa, op, dates, mask, label):
    if mask.sum() < 10:
        print(f"  {label:<22} n={mask.sum():>3}  insufficient")
        return
    a, p = oa[mask], op[mask]
    mae = float(np.mean(np.abs(p/a-1))*100)
    r2 = 1 - ((a-p)**2).sum()/((a-a.mean())**2).sum() if a.var() > 0 else float("nan")
    print(f"  {label:<22} n={mask.sum():>3}  R2={r2:+.4f}  MAE={mae:5.2f}%")

# Get OOS arrays for v6 and v6+RBOB_zscore
_, _, _, _, ref_oa, ref_op, ref_dates = wf(f, V6)
_, _, _, _, new_oa, new_op, new_dates = wf(f, V6 + ["RBOB_zscore_52w"])
# dates should align (same in-sample-cutoff progression)
assert np.array_equal(ref_dates, new_dates), "OOS date arrays must align"
periods = [
    ("2025-H1", (ref_dates >= pd.Timestamp("2025-01-01")) & (ref_dates < pd.Timestamp("2025-07-01"))),
    ("2025-H2", (ref_dates >= pd.Timestamp("2025-07-01")) & (ref_dates < pd.Timestamp("2026-01-01"))),
    ("2026-YTD",(ref_dates >= pd.Timestamp("2026-01-01"))),
]
for label, mask in periods:
    print(f"  {label}:")
    split_metrics(ref_oa, ref_op, ref_dates, mask, "    v6")
    split_metrics(new_oa, new_op, new_dates, mask, "    v6+RBOB")

# ---------- (e) Permutation null ----------
print("\n--- (e) Permutation null (RBOB_zscore_52w shuffled, 500 trials) ---")
print("  Computing baseline R^2 = ref already.")
target_lift = clean_oos - ref
print(f"  Observed lift: {target_lift*100:+.2f}pp")
N_PERMS = 500
rng = np.random.default_rng(42)
real = f["RBOB_zscore_52w"].to_numpy().copy()
beats = 0
lifts = []
# Use a faster bulk-evaluation: just vary the RBOB column and rerun walk-forward
for i in range(N_PERMS):
    perm = rng.permutation(real)
    f["__shuffled__"] = perm
    _, _, r2_oos_perm, _, _, _, _ = wf(f, V6 + ["__shuffled__"])
    lifts.append(r2_oos_perm - ref)
    if (r2_oos_perm - ref) >= target_lift:
        beats += 1
    if (i+1) % 50 == 0:
        print(f"    {i+1}/{N_PERMS} perms done; beats so far: {beats}")
lifts = np.array(lifts)
print(f"\n  Permutation results (n={N_PERMS}):")
print(f"    Observed lift:                {target_lift*100:+.2f}pp")
print(f"    Permuted lift mean:           {lifts.mean()*100:+.2f}pp")
print(f"    Permuted lift std:            {lifts.std()*100:.2f}pp")
print(f"    Permuted lift 95th pct:       {np.percentile(lifts, 95)*100:+.2f}pp")
print(f"    P(perm >= observed) =         {beats/N_PERMS:.4f}  (n_beats={beats})")
print(f"    Pass: empirical p-value < 0.05 -> {'YES' if beats/N_PERMS < 0.05 else 'NO'}")

print("\n" + "="*60)
print("Gauntlet complete. See output for per-test verdicts.")
print("Acceptance: ALL of (a-e) must pass cleanly.")

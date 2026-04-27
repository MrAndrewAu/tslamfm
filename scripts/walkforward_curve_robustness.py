"""
Robustness gauntlet for curve_IEF_SHY_zscore_52w before promoting to v6.2.

Initial walk-forward result (analyze_vol_rates_signal.py):
  v6.1 baseline:                       OOS R^2 = 0.7148  MAE = 7.22%
  v6.1 + curve_IEF_SHY_zscore_52w:     OOS R^2 = 0.7636  MAE = 6.64%   (+4.88pp)

Plus the conditionality flag from probe_curve_vs_rbob.py:
  v6 baseline:                         OOS R^2 = 0.6586
  v6 + CURVE_z (no RBOB):              OOS R^2 = 0.6055   (-5.31pp)  <-- bad alone
  v6.1 (= v6 + RBOB_z):                OOS R^2 = 0.7148   (+5.62pp)
  v6.1 + CURVE_z:                      OOS R^2 = 0.7636   (+4.88pp)

CURVE only adds value WITH RBOB present. The gauntlet must check whether
that conditional lift is real (joint-regime signal) or a 2026-tariff-shock
coincidence between the two series.

Five tests (mirrors RBOB gauntlet but adapted for v6.1-baseline + interaction):

  (a) Variant robustness across raw / zscore / IEF-SHY vs alternative curve
      proxies (TLT-zscore, log(IEF/SHY) raw).
  (b) Tenant-swap: does CURVE substitute for RBOB? (v6 + CURVE_z vs v6.1)
      -- already known to FAIL but rerun for completeness; also test
      v6.1 - RBOB + CURVE.
  (c) Lookahead-clean: zscore_52w is backward-only; reaffirm.
  (d) OOS sub-period split: 2025-H1 / 2025-H2 / 2026-YTD. Does the lift
      hold across regimes or is it tariff-shock concentrated?
  (e) Permutation null: shuffle CURVE 500x against v6.1 baseline, count
      how often a random series beats the observed +4.88pp.

Decision: REJECT if (b) shows CURVE is a substitute, OR (d) shows lift
concentrated in a single sub-period without economic justification, OR
(e) shows random beats >= 5%. Otherwise tentative ACCEPT.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

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

def zscore_52w(s):
    rm = s.rolling(52, min_periods=20).mean()
    rs = s.rolling(52, min_periods=20).std()
    return (s - rm) / rs

print("Fetching weekly closes...")
S = {k: wk(v) for k, v in {
    "TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB","VIX":"^VIX",
    "NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F",
    "IEF":"IEF","SHY":"SHY","TLT":"TLT","TNX":"^TNX",
}.items()}
df = pd.DataFrame(S).resample("W-FRI").last().ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"]); f["log_QQQ"] = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"]);  f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])
f["RBOB_zscore_52w"] = zscore_52w(np.log(df["RBOB"]))

# CURVE variants
curve_log = np.log(df["IEF"]) - np.log(df["SHY"])
f["curve_IEF_SHY_logratio"]      = curve_log
f["curve_IEF_SHY_zscore_52w"]    = zscore_52w(curve_log)
f["TLT_zscore_52w"]              = zscore_52w(np.log(df["TLT"]))
# Alt curve: TNX z-score (10Y nominal yield, level)
tnx_norm = df["TNX"] / 10.0 if df["TNX"].median() > 10 else df["TNX"]
f["TNX_zscore_52w"]              = zscore_52w(tnx_norm)

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
V6  = ["log_QQQ","log_DXY","log_VIX","NVDA_excess","ARKK_excess"] + active_events
V61 = V6 + ["RBOB_zscore_52w"]

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
    return r2_in, mae_in, np.array(oa), np.array(op), np.array(dates)

def show(label, factors_list, ref=None):
    r2_in, mae_in, oa, op = wf(f, factors_list)[:4]
    mae_oos = float(np.mean(np.abs(op/oa-1))*100)
    r2_oos = 1 - ((oa-op)**2).sum()/((oa-oa.mean())**2).sum()
    delta = (r2_oos - ref) * 100 if ref is not None else None
    dstr = f"{delta:+8.2f}pp" if delta is not None else "  (ref) "
    print(f"  {label:<60}  In R2={r2_in:.4f}  In MAE={mae_in:5.2f}%  |  OOS R2={r2_oos:.4f}  OOS MAE={mae_oos:5.2f}%  {dstr}")
    return r2_oos

# ---- v6 + v6.1 references ----
print("\n--- References ---")
r2_v6  = show("v6 baseline",  V6)
r2_v61 = show("v6.1 baseline (= v6 + RBOB_zscore_52w)", V61)
ref = r2_v61

# ---------- (a) Variant robustness ----------
print("\n--- (a) Variant robustness (vs v6.1 baseline) ---")
for v in ["curve_IEF_SHY_logratio","curve_IEF_SHY_zscore_52w",
          "TLT_zscore_52w","TNX_zscore_52w"]:
    show(f"v6.1 + {v}", V61 + [v], ref)

# ---------- (b) Tenant-swap: substitution test ----------
print("\n--- (b) Tenant-swap (does CURVE substitute for RBOB?) ---")
show("v6 + curve_IEF_SHY_zscore_52w (no RBOB)",  V6 + ["curve_IEF_SHY_zscore_52w"],  r2_v6)
show("v6 - RBOB + CURVE (replace RBOB w/ CURVE)", V6 + ["curve_IEF_SHY_zscore_52w"], ref)
# Both factors together (already known but reiterate vs v6.1)
show("v6.1 + curve_IEF_SHY_zscore_52w (additive)", V61 + ["curve_IEF_SHY_zscore_52w"], ref)

# ---------- (c) Lookahead-clean: zscore_52w only ----------
print("\n--- (c) Lookahead-clean (zscore_52w; backward-only transform) ---")
clean_oos = show("v6.1 + curve_IEF_SHY_zscore_52w (clean)",
                 V61 + ["curve_IEF_SHY_zscore_52w"], ref)

# ---------- (d) OOS sub-period split ----------
print("\n--- (d) OOS sub-period split ---")
_, _, ref_oa, ref_op, ref_dates = wf(f, V61)
_, _, new_oa, new_op, new_dates = wf(f, V61 + ["curve_IEF_SHY_zscore_52w"])
assert np.array_equal(ref_dates, new_dates)

def split_metrics(oa, op, mask, label):
    if mask.sum() < 10:
        print(f"  {label:<22} n={mask.sum():>3}  insufficient"); return
    a, p = oa[mask], op[mask]
    mae = float(np.mean(np.abs(p/a-1))*100)
    r2 = 1 - ((a-p)**2).sum()/((a-a.mean())**2).sum() if a.var() > 0 else float("nan")
    print(f"  {label:<22} n={mask.sum():>3}  R2={r2:+.4f}  MAE={mae:5.2f}%")

periods = [
    ("2025-H1", (ref_dates >= pd.Timestamp("2025-01-01")) & (ref_dates < pd.Timestamp("2025-07-01"))),
    ("2025-H2", (ref_dates >= pd.Timestamp("2025-07-01")) & (ref_dates < pd.Timestamp("2026-01-01"))),
    ("2026-YTD",(ref_dates >= pd.Timestamp("2026-01-01"))),
]
for label, mask in periods:
    print(f"  {label}:")
    split_metrics(ref_oa, ref_op, mask, "    v6.1")
    split_metrics(new_oa, new_op, mask, "    v6.1+CURVE")

# ---------- (e) Permutation null ----------
print("\n--- (e) Permutation null (CURVE shuffled, 500 trials, vs v6.1) ---")
target_lift = clean_oos - ref
print(f"  Observed lift: {target_lift*100:+.2f}pp")
N_PERMS = 500
rng = np.random.default_rng(43)
real = f["curve_IEF_SHY_zscore_52w"].to_numpy().copy()
beats = 0
lifts = []
for i in range(N_PERMS):
    perm = rng.permutation(real)
    f["__shuffled__"] = perm
    _, _, oa_p, op_p, _ = wf(f, V61 + ["__shuffled__"])
    r2_p = 1 - ((oa_p-op_p)**2).sum()/((oa_p-oa_p.mean())**2).sum()
    lifts.append(r2_p - ref)
    if (r2_p - ref) >= target_lift:
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
print("Gauntlet complete. CURVE is a v6.2 candidate ONLY if all 5 pass.")
print("Particular concern: test (b) -- CURVE alone is known to HURT v6.")

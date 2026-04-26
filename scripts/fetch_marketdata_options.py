"""
Fetch weekly TSLA option-chain features from marketdata.app (REST API).

Setup:
  $env:MARKETDATA_TOKEN = "<your token>"   # do NOT commit this
  python scripts/fetch_marketdata_options.py --start 2022-04-26 --end 2026-04-26

Plan: Starter $12/mo (annual). Includes IV + Greeks, 5 years of history.
Cost model: historical chain queries are 1 credit per 1000 contracts returned.
A TSLA full chain (~3-5k contracts) is ~3-5 credits per Friday snapshot.
209 Fridays * ~5 credits = ~1k credits for full backfill (well under daily 10k cap).

Outputs:
  public/data/options_features.csv
  Columns (weekly, Friday close):
    date
    spot
    skew_25d        -- IV(25d put) - IV(25d call), ~30-day expiry  (NaN on Starter historical)
    iv_atm_30d      -- ATM IV at ~30-day expiry                     (NaN on Starter historical)
    pc_oi_ratio     -- total put OI / total call OI in near-term chain
    pc_vol_ratio    -- total put volume / total call volume in near-term chain
    term_slope      -- IV(~90d ATM) - IV(~30d ATM)                  (NaN on Starter historical)

Note: marketdata.app Starter ($12/mo) returns historical bid/ask/OI/volume
but NOT historical IV or Greeks. The IV-dependent columns will be NaN.
Use pc_oi_ratio / pc_vol_ratio for the cheap test; compute IV locally with
a Black-Scholes solver if those fail and you want to test skew.

Then run analyze_options_signal.py to test correlation with model residuals
BEFORE deciding to subscribe annually or integrate into build_model_data.py.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.marketdata.app/v1"
SYMBOL = "TSLA"
OUT = Path(__file__).resolve().parents[1] / "public" / "data" / "options_features.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _token() -> str:
    t = os.environ.get("MARKETDATA_TOKEN", "").strip()
    if not t:
        sys.exit("ERROR: set MARKETDATA_TOKEN env var (do NOT commit it).")
    return t


def _get_chain(asof: date, **filters) -> dict:
    """Hit /options/chain for a given historical date with optional filters."""
    url = f"{API_BASE}/options/chain/{SYMBOL}/"
    params = {"date": asof.strftime("%Y-%m-%d"), **filters}
    headers = {"Authorization": f"Bearer {_token()}"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    # marketdata.app returns:
    #   200 = fresh data
    #   203 = cached/delayed but VALID data (still has s:"ok" body)
    #   204 = no data available
    #   404 = no data for that specific date (e.g. market holiday)
    if r.status_code in (204, 404):
        return {"s": "no_data"}
    if r.status_code not in (200, 203):
        r.raise_for_status()
    return r.json()


def _df(payload: dict) -> pd.DataFrame:
    """Convert columnar response to a DataFrame; empty if no data."""
    if not payload or payload.get("s") != "ok":
        return pd.DataFrame()
    cols = [k for k, v in payload.items() if isinstance(v, list)]
    rows = list(zip(*(payload[c] for c in cols)))
    return pd.DataFrame(rows, columns=cols)


def features_for_week(asof: date) -> dict | None:
    """Compute one row of options features for the given Friday."""
    # Pull a focused slice: dte 7-120 days, OTM-ish strikes only, both sides.
    payload = _get_chain(asof, **{"dte.from": 7, "dte.to": 120})
    df = _df(payload)
    if df.empty:
        print(f"  {asof}: no chain returned")
        return None

    spot = float(df["underlyingPrice"].iloc[0])

    # Coerce numerics (response sometimes has them as strings)
    for c in ("iv", "delta", "openInterest", "volume", "strike", "dte"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Find expiration buckets nearest to 30d and 90d
    exp_unique = sorted(df["dte"].dropna().unique())
    if not exp_unique:
        return None
    e30 = min(exp_unique, key=lambda x: abs(x - 30))
    e90 = min(exp_unique, key=lambda x: abs(x - 90))

    out: dict = {"date": pd.Timestamp(asof), "spot": spot}

    # ATM 30d IV: take put + call IV nearest the spot at expiry e30
    near30 = df[df["dte"] == e30].copy()
    if not near30.empty and near30["iv"].notna().any():
        near30["distance"] = (near30["strike"] - spot).abs()
        atm_rows = near30.nsmallest(2, "distance")  # nearest call + nearest put
        out["iv_atm_30d"] = float(atm_rows["iv"].mean())
    else:
        out["iv_atm_30d"] = np.nan

    # 25-delta skew at expiry e30
    if not near30.empty and near30["delta"].notna().any():
        # Puts have negative delta (~ -0.25 for "25-delta put"), calls positive (~0.25).
        puts = near30[near30["delta"] < 0]
        calls = near30[near30["delta"] > 0]
        if not puts.empty and not calls.empty:
            put25  = puts.iloc[(puts["delta"]   - (-0.25)).abs().argsort()[:1]]
            call25 = calls.iloc[(calls["delta"] - ( 0.25)).abs().argsort()[:1]]
            iv_p = float(put25["iv"].iloc[0])  if pd.notna(put25["iv"].iloc[0])  else np.nan
            iv_c = float(call25["iv"].iloc[0]) if pd.notna(call25["iv"].iloc[0]) else np.nan
            out["skew_25d"] = (iv_p - iv_c) if (np.isfinite(iv_p) and np.isfinite(iv_c)) else np.nan
        else:
            out["skew_25d"] = np.nan
    else:
        out["skew_25d"] = np.nan

    # Term slope: ATM IV at e90 minus ATM IV at e30
    near90 = df[df["dte"] == e90].copy()
    if not near90.empty and near90["iv"].notna().any() and np.isfinite(out.get("iv_atm_30d", np.nan)):
        near90["distance"] = (near90["strike"] - spot).abs()
        atm90 = near90.nsmallest(2, "distance")
        iv90 = float(atm90["iv"].mean())
        out["term_slope"] = iv90 - out["iv_atm_30d"]
    else:
        out["term_slope"] = np.nan

    # P/C ratios across the focused chain (within +/- 25% of spot to filter junk)
    near_band = df[(df["strike"] >= spot * 0.75) & (df["strike"] <= spot * 1.25)]
    puts_n = near_band[near_band.get("side") == "put"]
    calls_n = near_band[near_band.get("side") == "call"]
    pc_oi_p = float(puts_n.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
    pc_oi_c = float(calls_n.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
    pc_v_p  = float(puts_n.get("volume",       pd.Series(dtype=float)).fillna(0).sum())
    pc_v_c  = float(calls_n.get("volume",      pd.Series(dtype=float)).fillna(0).sum())
    out["pc_oi_ratio"]  = pc_oi_p / pc_oi_c if pc_oi_c else np.nan
    out["pc_vol_ratio"] = pc_v_p  / pc_v_c  if pc_v_c  else np.nan

    return out


def fridays(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.date_range(start, end, freq="W-FRI")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only fetch the most recent N Fridays (smoke test).")
    args = ap.parse_args()

    weeks = fridays(pd.to_datetime(args.start).date(), pd.to_datetime(args.end).date())
    if args.limit > 0:
        weeks = weeks[-args.limit:]

    # Resume support: load any existing rows and skip dates we already have.
    existing = pd.DataFrame()
    if OUT.exists():
        existing = pd.read_csv(OUT, index_col="date", parse_dates=["date"])
        have = set(existing.index.date.tolist())
        weeks = [d for d in weeks if d not in have]
        print(f"Resuming: {len(existing)} rows already on disk, {len(weeks)} new weeks to fetch.")

    if not weeks:
        print("Nothing to do — all requested weeks already present.")
        return

    print(f"Fetching {len(weeks)} weekly snapshots {weeks[0]} -> {weeks[-1]}")

    rows = list(existing.reset_index().to_dict("records")) if not existing.empty else []

    def flush():
        if not rows:
            return
        out = pd.DataFrame(rows).set_index("date").sort_index()
        out.to_csv(OUT)

    for i, day in enumerate(weeks, 1):
        try:
            r = features_for_week(day)
            if r:
                rows.append(r)
                print(f"  [{i}/{len(weeks)}] {day} ok  spot={r['spot']:.2f} "
                      f"skew={r.get('skew_25d', float('nan')):.3f} "
                      f"iv30={r.get('iv_atm_30d', float('nan')):.3f}",
                      flush=True)
            else:
                print(f"  [{i}/{len(weeks)}] {day} skipped", flush=True)
        except requests.HTTPError as e:
            print(f"  [{i}/{len(weeks)}] {day} HTTP {e.response.status_code}: {e.response.text[:200]}", flush=True)
        except Exception as ex:
            print(f"  [{i}/{len(weeks)}] {day} FAILED: {ex}", flush=True)
        # Persist after every row so a crash never costs more than one week.
        flush()
        # Be polite — rate limit is generous but no need to hammer.
        time.sleep(0.1)

    if not rows:
        print("No rows fetched — aborting write.")
        sys.exit(1)

    out = pd.DataFrame(rows).set_index("date").sort_index()
    print(f"\nWrote {OUT}  ({len(out)} rows)")
    print(out.tail(8).to_string())


if __name__ == "__main__":
    main()

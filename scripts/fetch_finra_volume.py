"""
Fetch FINRA daily short-sale / off-exchange volume for TSLA.

Source: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
Note:  TotalVolume here is FINRA-TRF reported volume (off-exchange / dark
       pool / wholesaler routes). It is NOT consolidated tape volume.
       To get off_exch_ratio = TotalVolume_FINRA / TotalVolume_consolidated
       we cross with yfinance daily Volume in the analyzer.

Output: public/data/finra_volume.csv
Columns: date, short_vol, short_exempt_vol, finra_total_vol

Usage:
  python scripts/fetch_finra_volume.py --start 2022-04-26 --end 2026-04-26

Resumable: skips dates already on disk. ~1000 trading days, ~10 minutes
with parallel downloads.
"""
from __future__ import annotations
import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests

URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
SYMBOL = "TSLA"
OUT = Path(__file__).resolve().parents[1] / "public" / "data" / "finra_volume.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()


def fetch_day(d: pd.Timestamp) -> dict | None:
    url = URL.format(ymd=d.strftime("%Y%m%d"))
    try:
        r = SESSION.get(url, timeout=15)
    except Exception as ex:
        return {"_error": f"{d.date()} {ex}"}
    if r.status_code == 404:
        return None  # weekend / holiday
    if r.status_code != 200:
        return {"_error": f"{d.date()} HTTP {r.status_code}"}
    # Parse pipe-delimited; filter to TSLA only.
    try:
        df = pd.read_csv(io.StringIO(r.text), sep="|")
    except Exception as ex:
        return {"_error": f"{d.date()} parse: {ex}"}
    row = df[df["Symbol"] == SYMBOL]
    if row.empty:
        return None
    r0 = row.iloc[0]
    return {
        "date": d.normalize(),
        "short_vol": float(r0["ShortVolume"]),
        "short_exempt_vol": float(r0["ShortExemptVolume"]),
        "finra_total_vol": float(r0["TotalVolume"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    days = pd.bdate_range(args.start, args.end)  # Mon-Fri only

    existing = pd.DataFrame()
    if OUT.exists():
        existing = pd.read_csv(OUT, index_col="date", parse_dates=["date"])
        have = set(existing.index.normalize().tolist())
        days = pd.DatetimeIndex([d for d in days if d not in have])
        print(f"Resuming: {len(existing)} rows on disk, {len(days)} new days.")

    if len(days) == 0:
        print("Nothing to do.")
        return

    print(f"Fetching {len(days)} business days {days[0].date()} -> {days[-1].date()}")

    rows = list(existing.reset_index().to_dict("records")) if not existing.empty else []
    errors = 0
    completed = 0

    def flush():
        if not rows:
            return
        df = pd.DataFrame(rows).set_index("date").sort_index()
        df.to_csv(OUT)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_day, d): d for d in days}
        for fut in as_completed(futs):
            completed += 1
            try:
                r = fut.result()
            except Exception as ex:
                errors += 1
                print(f"  EXC: {ex}")
                continue
            if r is None:
                pass  # weekend/holiday/no data
            elif "_error" in r:
                errors += 1
                print(f"  {r['_error']}")
            else:
                rows.append(r)
            if completed % 50 == 0:
                flush()
                print(f"  [{completed}/{len(days)}] rows={len(rows)} errors={errors}")

    flush()
    df = pd.DataFrame(rows).set_index("date").sort_index()
    print(f"\nWrote {OUT}  ({len(df)} rows, {errors} errors)")
    print(df.tail(8).to_string())


if __name__ == "__main__":
    main()

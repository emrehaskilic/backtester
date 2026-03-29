"""
Binance OI indir — data.binance.vision daily metrics.
Her gun 288 kayit (5dk aralıklı OI). 11 ay.
"""
import os, time, requests, zipfile, io
import numpy as np, pandas as pd
from datetime import datetime, timedelta

OUT_OI = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_OI_5m_11mo.parquet")
OUT_5M = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_vol_11mo.parquet")
OUT_MERGED = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_cvd_oi_11mo.parquet")

SYMBOL = "ETHUSDT"
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
START = datetime(2025, 4, 1)
END = datetime(2026, 3, 1)


def main():
    print("=" * 60)
    print("  OI INDIR — Binance Daily Metrics")
    print("=" * 60)

    all_dfs = []
    t0 = time.time()
    d = START
    success = 0
    fail = 0

    while d < END:
        ds = d.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/{SYMBOL}/{SYMBOL}-metrics-{ds}.zip"

        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    names = [n for n in zf.namelist() if n.endswith('.csv')]
                    if names:
                        with zf.open(names[0]) as f:
                            df = pd.read_csv(f)
                            all_dfs.append(df)
                            success += 1
            elif r.status_code == 404:
                fail += 1
            else:
                fail += 1

        except Exception:
            fail += 1

        if (success + fail) % 30 == 0:
            print(f"  {ds} | {success} OK, {fail} FAIL | {len(all_dfs) * 288:,} kayit")

        d += timedelta(days=1)
        time.sleep(0.05)  # hafif rate limit

    if not all_dfs:
        print("  HATA: Hic veri alinamadi!")
        return

    result = pd.concat(all_dfs, ignore_index=True)

    # Timestamp olustur
    result["open_time"] = pd.to_datetime(result["create_time"]).astype(np.int64) // 10**6  # ms
    result["open_interest"] = result["sum_open_interest"].astype(np.float64)
    result = result[["open_time", "open_interest"]].sort_values("open_time").reset_index(drop=True)
    result = result.drop_duplicates(subset="open_time")

    result.to_parquet(OUT_OI, index=False)
    sz = os.path.getsize(OUT_OI) / 1024 / 1024
    print(f"\n  OI: {len(result):,} kayit | {sz:.1f} MB | {success} gun | {time.time()-t0:.0f}s")

    # Merge
    print("\n  Merge yapiliyor...")
    candles = pd.read_parquet(OUT_5M)
    merged = pd.merge_asof(
        candles.sort_values("open_time"),
        result.sort_values("open_time"),
        on="open_time",
        direction="backward"
    )
    merged.to_parquet(OUT_MERGED, index=False)
    cov = merged["open_interest"].notna().sum() / len(merged) * 100
    sz2 = os.path.getsize(OUT_MERGED) / 1024 / 1024
    print(f"  Merged: {len(merged):,} bar | {sz2:.1f} MB | OI coverage: {cov:.0f}%")
    print(f"\n  TAMAMLANDI")


if __name__ == "__main__":
    main()

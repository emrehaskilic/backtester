"""
Binance OI indir — 5dk, 11 ay. Hizli versiyon.
Her request 500 kayit, 200ms bekleme.
"""
import os, time, requests
import numpy as np, pandas as pd
from datetime import datetime

OUT_OI = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_OI_5m_11mo.parquet")
OUT_5M = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_vol_11mo.parquet")
OUT_MERGED = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_cvd_oi_11mo.parquet")

SYMBOL = "ETHUSDT"
URL = "https://fapi.binance.com/futures/data/openInterestHist"

def download_oi():
    start_ms = int(datetime(2025, 4, 1).timestamp() * 1000)
    end_ms = int(datetime(2026, 3, 1).timestamp() * 1000)

    all_data = []
    current = start_ms
    page = 0
    t0 = time.time()

    print("OI indiriliyor...")
    while current < end_ms:
        params = {"symbol": SYMBOL, "period": "5m", "startTime": current, "limit": 500}
        try:
            resp = requests.get(URL, params=params, timeout=15)
            if resp.status_code == 429:
                print("  Rate limit, 5s bekleniyor...")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}, 2s bekleniyor...")
                time.sleep(2)
                continue

            data = resp.json()
            if not data:
                break

            all_data.extend(data)
            current = data[-1]["timestamp"] + 1
            page += 1

            if page % 20 == 0:
                dt = datetime.utcfromtimestamp(data[-1]["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M")
                pct = (current - start_ms) / (end_ms - start_ms) * 100
                print(f"  {len(all_data):>6,} kayit | {dt} | %{pct:.0f} | {time.time()-t0:.0f}s")

            time.sleep(0.2)  # 200ms — daha hizli

        except requests.exceptions.Timeout:
            print("  Timeout, tekrar deneniyor...")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"  Hata: {e}, 3s bekleniyor...")
            time.sleep(3)
            continue

    if not all_data:
        print("HATA: OI verisi alinamadi!")
        return None

    df = pd.DataFrame(all_data)
    df["open_time"] = df["timestamp"].astype(np.int64)
    df["open_interest"] = df["sumOpenInterest"].astype(np.float64)
    df = df[["open_time", "open_interest"]].sort_values("open_time").reset_index(drop=True)
    df = df.drop_duplicates(subset="open_time")

    df.to_parquet(OUT_OI, index=False)
    sz = os.path.getsize(OUT_OI) / 1024 / 1024
    print(f"\nOI: {len(df):,} kayit | {sz:.1f} MB | {time.time()-t0:.0f}s")
    return df


def merge():
    print("\nMerge yapiliyor...")
    candles = pd.read_parquet(OUT_5M)
    oi = pd.read_parquet(OUT_OI)

    merged = pd.merge_asof(
        candles.sort_values("open_time"),
        oi[["open_time", "open_interest"]].sort_values("open_time"),
        on="open_time",
        direction="backward"
    )

    merged.to_parquet(OUT_MERGED, index=False)
    sz = os.path.getsize(OUT_MERGED) / 1024 / 1024
    cov = merged["open_interest"].notna().sum() / len(merged) * 100
    print(f"Merged: {len(merged):,} bar | {sz:.1f} MB | OI coverage: {cov:.0f}%")


def main():
    print("=" * 50)
    # 5dk mum zaten var, sadece OI indir + merge
    if os.path.exists(OUT_5M):
        print(f"5dk mum zaten var: {OUT_5M}")
    else:
        print("HATA: Once build_5m_oi_cache.py calistirin (0a adimi)")
        return

    oi = download_oi()
    if oi is not None:
        merge()
        print("\nTAMAMLANDI")


if __name__ == "__main__":
    main()

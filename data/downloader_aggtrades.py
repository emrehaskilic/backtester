#!/Users/deneme/miniconda/bin/python
"""
Binance aggTrades toplu indirici.
data.binance.vision'dan aylık/günlük aggTrades ZIP dosyalarını indirir.

Kullanım: python aggtrades_downloader.py
"""

import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
import time

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades"

# 6 ay veri indir (tick replay için yeterli, daha fazla çok yer kaplar)
MONTHS_BACK = 6


def get_months(months_back):
    """Son N ayın yıl-ay listesini döndür."""
    months = []
    now = datetime.utcnow()
    for i in range(months_back, 0, -1):
        d = now - timedelta(days=30 * i)
        months.append(d.strftime("%Y-%m"))
    return months


def download_monthly(symbol, year_month):
    """Bir ayın aggTrades ZIP'ini indir ve CSV'ye çıkar."""
    symbol_dir = os.path.join(OUTPUT_DIR, symbol.lower())
    os.makedirs(symbol_dir, exist_ok=True)

    csv_path = os.path.join(symbol_dir, f"aggTrades_{year_month}.csv")

    # Zaten varsa atla
    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        print(f"    {year_month}: zaten mevcut ({size_mb:.0f} MB)")
        return True

    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{year_month}.zip"
    print(f"    {year_month}: indiriliyor...", end="", flush=True)

    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 404:
            print(f" bulunamadi (404)")
            return False
        resp.raise_for_status()

        # ZIP'i aç
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = z.namelist()[0]

        with z.open(csv_name) as f:
            with open(csv_path, "wb") as out:
                out.write(f.read())

        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        print(f" OK ({size_mb:.0f} MB)")
        return True

    except Exception as e:
        print(f" HATA: {e}")
        return False


def download_symbol(symbol, on_progress=None):
    """Bir sembol için tüm ayları indir."""
    print(f"\n  {symbol}:")
    months = get_months(MONTHS_BACK)
    success = 0

    for i, ym in enumerate(months):
        if on_progress:
            on_progress(i, ym)
        if download_monthly(symbol, ym):
            success += 1
        time.sleep(0.5)

    print(f"  {symbol}: {success}/{len(months)} ay indirildi")
    return success


def merge_symbol(symbol):
    """Bir sembolun tum aylik CSV'lerini birlestir (memory-efficient)."""
    symbol_dir = os.path.join(OUTPUT_DIR, symbol.lower())
    if not os.path.exists(symbol_dir):
        return

    csv_files = sorted([f for f in os.listdir(symbol_dir) if f.startswith("aggTrades_")])
    if not csv_files:
        return

    output_path = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_aggtrades.parquet")
    print(f"\n  {symbol}: {len(csv_files)} dosya birlestiriliyor...")

    # Process each CSV one at a time, write individual parquets, then concat
    temp_parquets = []
    total_trades = 0

    for f in csv_files:
        path = os.path.join(symbol_dir, f)
        try:
            df = pd.read_csv(path, dtype={
                "price": "float64",
                "quantity": "float64",
                "transact_time": "int64",
            })
            # Normalize column names
            if "transact_time" in df.columns:
                df = df.rename(columns={"transact_time": "timestamp"})
            df["side"] = df["is_buyer_maker"].map({True: "SELL", False: "BUY"})
            result = df[["timestamp", "price", "quantity", "side"]].copy()
            result = result.sort_values("timestamp").reset_index(drop=True)

            temp_path = os.path.join(symbol_dir, f.replace(".csv", ".parquet"))
            result.to_parquet(temp_path, index=False)
            temp_parquets.append(temp_path)
            total_trades += len(result)
            print(f"    {f}: {len(df):,} trade")
            del df, result
        except Exception as e:
            print(f"    {f}: HATA - {e}")

    if not temp_parquets:
        return

    # Read and concat parquets (much more memory efficient than CSV concat)
    print(f"  {symbol}: parquet dosyalari birlestiriliyor...")
    dfs = [pd.read_parquet(p) for p in temp_parquets]
    merged = pd.concat(dfs, ignore_index=True)
    del dfs
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged.to_parquet(output_path, index=False)

    # Cleanup temp parquets
    for p in temp_parquets:
        os.remove(p)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {symbol}: {total_trades:,} trade -> {output_path} ({size_mb:.0f} MB)")


def main():
    print("=" * 60)
    print("  BINANCE aggTrades INDIRICI")
    print(f"  Son {MONTHS_BACK} ay | {len(SYMBOLS)} sembol")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol in SYMBOLS:
        download_symbol(symbol)

    print("\n" + "=" * 60)
    print("  BIRLESTIRME")
    print("=" * 60)

    for symbol in SYMBOLS:
        merge_symbol(symbol)

    print("\nTamamlandi!")


if __name__ == "__main__":
    main()

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
            print(f" bulunamadı (404)")
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


def download_symbol(symbol):
    """Bir sembol için tüm ayları indir."""
    print(f"\n  {symbol}:")
    months = get_months(MONTHS_BACK)
    success = 0

    for ym in months:
        if download_monthly(symbol, ym):
            success += 1
        time.sleep(0.5)

    print(f"  {symbol}: {success}/{len(months)} ay indirildi")
    return success


def merge_symbol(symbol):
    """Bir sembolün tüm aylık CSV'lerini birleştir."""
    symbol_dir = os.path.join(OUTPUT_DIR, symbol.lower())
    if not os.path.exists(symbol_dir):
        return

    csv_files = sorted([f for f in os.listdir(symbol_dir) if f.startswith("aggTrades_")])
    if not csv_files:
        return

    print(f"\n  {symbol}: {len(csv_files)} dosya birleştiriliyor...")

    columns = ["agg_trade_id", "price", "quantity", "first_trade_id",
               "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]

    dfs = []
    for f in csv_files:
        path = os.path.join(symbol_dir, f)
        try:
            df = pd.read_csv(path, header=None, names=columns)
            dfs.append(df)
            print(f"    {f}: {len(df):,} trade")
        except Exception as e:
            print(f"    {f}: HATA - {e}")

    if not dfs:
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset="agg_trade_id").reset_index(drop=True)

    # Temizlik
    merged["price"] = merged["price"].astype(float)
    merged["quantity"] = merged["quantity"].astype(float)
    merged["timestamp"] = merged["timestamp"].astype(int)
    merged["is_buyer_maker"] = merged["is_buyer_maker"].astype(bool)
    merged["side"] = merged["is_buyer_maker"].map({True: "SELL", False: "BUY"})

    # Sadece gerekli kolonlar
    result = merged[["timestamp", "price", "quantity", "side"]].copy()

    output_path = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_aggtrades.parquet")
    result.to_parquet(output_path, index=False)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {symbol}: {len(result):,} trade → {output_path} ({size_mb:.0f} MB)")


def main():
    print("=" * 60)
    print("  BINANCE aggTrades İNDİRİCİ")
    print(f"  Son {MONTHS_BACK} ay | {len(SYMBOLS)} sembol")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol in SYMBOLS:
        download_symbol(symbol)

    print("\n" + "=" * 60)
    print("  BİRLEŞTİRME")
    print("=" * 60)

    for symbol in SYMBOLS:
        merge_symbol(symbol)

    print("\nTamamlandı!")


if __name__ == "__main__":
    main()

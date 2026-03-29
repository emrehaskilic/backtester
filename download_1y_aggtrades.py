"""
Binance 1 Yillik aggTrades Indirici — ETHUSDT Futures
data.binance.vision'dan aylik ZIP dosyalarini indirir.
Hedef: ~/Desktop/ETHUSDT_1Y_AggTrades/
"""

import os
import requests
import zipfile
import io
import time
from datetime import datetime, timedelta

SYMBOL = "ETHUSDT"
OUTPUT_DIR = os.path.expanduser("~/Desktop/ETHUSDT_1Y_AggTrades")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades"

# 12 ay: Nisan 2025 — Mart 2026
MONTHS = []
for y in [2025]:
    for m in range(4, 13):
        MONTHS.append(f"{y}-{m:02d}")
for y in [2026]:
    for m in range(1, 4):
        MONTHS.append(f"{y}-{m:02d}")


def download_month(symbol, year_month):
    """Bir ayin aggTrades ZIP'ini indir ve CSV'ye cikar."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_filename = f"{symbol}-aggTrades-{year_month}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    # Zaten varsa atla
    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        print(f"  {year_month}: zaten mevcut ({size_mb:.0f} MB) -- ATLANDI")
        return True

    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{year_month}.zip"
    print(f"  {year_month}: indiriliyor... ", end="", flush=True)

    try:
        t0 = time.time()
        resp = requests.get(url, stream=True, timeout=300)

        if resp.status_code == 404:
            print(f"404 NOT FOUND (henuz yayinlanmamis olabilir)")
            return False

        resp.raise_for_status()

        # ZIP icerigini oku
        content = resp.content
        size_mb = len(content) / 1024 / 1024
        elapsed = time.time() - t0

        print(f"ZIP: {size_mb:.0f} MB ({elapsed:.0f}s) | cikariliyor... ", end="", flush=True)

        # ZIP'i ac ve CSV'yi cikar
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # ZIP icindeki CSV dosyasini bul
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                print("HATA: ZIP icinde CSV bulunamadi!")
                return False

            # CSV'yi cikar
            with zf.open(csv_names[0]) as src, open(csv_path, 'wb') as dst:
                dst.write(src.read())

        csv_size_mb = os.path.getsize(csv_path) / 1024 / 1024
        print(f"CSV: {csv_size_mb:.0f} MB -- TAMAM")
        return True

    except requests.exceptions.RequestException as e:
        print(f"HATA: {e}")
        return False


def main():
    print("=" * 60)
    print(f"  BINANCE {SYMBOL} aggTrades INDIRICI")
    print(f"  Periyot: {MONTHS[0]} - {MONTHS[-1]} ({len(MONTHS)} ay)")
    print(f"  Hedef: {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = 0

    for ym in MONTHS:
        ok = download_month(SYMBOL, ym)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n  Sonuc: {success} basarili, {failed} basarisiz")
    print(f"  Veri klasoru: {OUTPUT_DIR}")

    # Toplam boyut
    total_size = 0
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
    print(f"  Toplam boyut: {total_size / 1024 / 1024 / 1024:.1f} GB")


if __name__ == "__main__":
    main()

"""
Adim 0: 5dk mum + OI parquet cache olustur.
0a: aggTrades -> 5dk mum (OHLCV + buy_vol + sell_vol)
0b: Binance OI indir (5dk, 11 ay)
0c: Merge -> tek parquet
"""
import os, glob, time, requests, zipfile, io
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.expanduser("~/Desktop/ETHUSDT_1Y_AggTrades")
OUT_5M = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_vol_11mo.parquet")
OUT_OI = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_OI_5m_11mo.parquet")
OUT_MERGED = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_cvd_oi_11mo.parquet")

SYMBOL = "ETHUSDT"
MONTHS = []
for y in [2025]:
    for m in range(4, 13):
        MONTHS.append(f"{y}-{m:02d}")
for y in [2026]:
    for m in range(1, 3):
        MONTHS.append(f"{y}-{m:02d}")


def build_5m_candles():
    """Adim 0a: aggTrades -> 5dk mum."""
    print("\n  [0a] 5dk mum olusturuluyor...")
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "ETHUSDT-aggTrades-*.csv")))
    all_candles = []
    t0 = time.time()

    for f in csv_files:
        sz = os.path.getsize(f) / 1024 / 1024
        if sz < 1: continue
        bn = os.path.basename(f)
        print(f"    {bn} ({sz:.0f} MB)... ", end="", flush=True)
        t1 = time.time()

        df = pd.read_csv(f, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
        df["price"] = df["price"].astype(np.float64)
        df["quantity"] = df["quantity"].astype(np.float64)
        if df["is_buyer_maker"].dtype == object:
            df["is_buyer_maker"] = df["is_buyer_maker"].str.lower() == "true"
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
        df["buy_qty"] = df["quantity"] * (~df["is_buyer_maker"]).astype(np.float64)
        df["sell_qty"] = df["quantity"] * df["is_buyer_maker"].astype(np.float64)

        period = 300_000  # 5dk = 300s
        df["ts"] = (df["transact_time"] // period) * period
        c = df.groupby("ts").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("quantity", "sum"),
            buy_vol=("buy_qty", "sum"), sell_vol=("sell_qty", "sum"),
            trade_count=("price", "count"),
        ).reset_index().rename(columns={"ts": "open_time"})
        all_candles.append(c)
        print(f"{len(c):,} mum ({time.time()-t1:.0f}s)")
        del df

    result = pd.concat(all_candles, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    result.to_parquet(OUT_5M, index=False)
    sz = os.path.getsize(OUT_5M) / 1024 / 1024
    print(f"  5dk mum: {len(result):,} bar | {sz:.1f} MB | {time.time()-t0:.0f}s")
    return result


def download_oi():
    """Adim 0b: Binance OI indir (5dk)."""
    print("\n  [0b] OI verisi indiriliyor...")

    # Binance data.binance.vision'da OI yok, API ile cekelim
    # /fapi/v1/openInterestHist - 5m period, max 500 per request
    # Alternatif: data.binance.vision/data/futures/um/monthly/metrics/ETHUSDT/

    base_url = "https://data.binance.vision/data/futures/um/monthly/metrics"
    all_oi = []
    t0 = time.time()

    for ym in MONTHS:
        url = f"{base_url}/{SYMBOL}/{SYMBOL}-metrics-{ym}.zip"
        print(f"    {ym}: indiriliyor... ", end="", flush=True)
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 404:
                print("404 — metrics bulunamadi, API ile denenecek")
                break
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                if csv_names:
                    with zf.open(csv_names[0]) as src:
                        df = pd.read_csv(src)
                        print(f"{len(df):,} satir")
                        all_oi.append(df)
                else:
                    print("CSV bulunamadi")
        except Exception as e:
            print(f"HATA: {e}")
            break

    # Eger metrics indiremediyse, API ile dene
    if not all_oi:
        print("  Metrics bulunamadi, Binance API ile OI indiriliyor...")
        return download_oi_api()

    result = pd.concat(all_oi, ignore_index=True)
    # Kolon isimlerini kontrol et
    print(f"  OI kolonlari: {list(result.columns)}")
    result.to_parquet(OUT_OI, index=False)
    print(f"  OI: {len(result):,} satir | {time.time()-t0:.0f}s")
    return result


def download_oi_api():
    """Binance API ile 5dk OI indir — sayfalanmis."""
    import time as t_mod

    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_data = []

    # Baslangic ve bitis (ms)
    start_ms = int(datetime(2025, 4, 1).timestamp() * 1000)
    end_ms = int(datetime(2026, 3, 1).timestamp() * 1000)

    current = start_ms
    page = 0
    t0 = time.time()

    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "period": "5m",
            "startTime": current,
            "limit": 500,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print("  Rate limit, 10s bekleniyor...")
                t_mod.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            last_ts = data[-1]["timestamp"]
            current = last_ts + 1
            page += 1

            if page % 50 == 0:
                dt = datetime.utcfromtimestamp(last_ts / 1000).strftime("%Y-%m-%d")
                print(f"    Sayfa {page}: {len(all_data):,} kayit | {dt}")

            # Rate limit: 500ms bekle
            t_mod.sleep(0.5)

        except Exception as e:
            print(f"  API hatasi: {e}, 5s bekleniyor...")
            t_mod.sleep(5)
            continue

    if not all_data:
        print("  HATA: OI verisi alinamadi!")
        return None

    df = pd.DataFrame(all_data)
    df["open_time"] = df["timestamp"].astype(np.int64)
    df["open_interest"] = df["sumOpenInterest"].astype(np.float64)
    df["open_interest_value"] = df["sumOpenInterestValue"].astype(np.float64)
    df = df[["open_time", "open_interest", "open_interest_value"]].sort_values("open_time").reset_index(drop=True)

    df.to_parquet(OUT_OI, index=False)
    sz = os.path.getsize(OUT_OI) / 1024 / 1024
    print(f"  OI: {len(df):,} kayit | {sz:.1f} MB | {time.time()-t0:.0f}s")
    return df


def merge_data(candles_df, oi_df):
    """Adim 0c: 5dk mum + OI merge."""
    print("\n  [0c] Merge yapiliyor...")

    if oi_df is None:
        print("  OI verisi yok, sadece mum kullanilacak")
        candles_df["open_interest"] = np.nan
        candles_df.to_parquet(OUT_MERGED, index=False)
        return candles_df

    # OI'yi 5dk mum timestamp'lerine hizala (en yakin onceki)
    candles_df = candles_df.sort_values("open_time").reset_index(drop=True)
    oi_df = oi_df.sort_values("open_time").reset_index(drop=True)

    # merge_asof: her mum icin en yakin onceki OI
    merged = pd.merge_asof(
        candles_df,
        oi_df[["open_time", "open_interest"]],
        on="open_time",
        direction="backward"
    )

    merged.to_parquet(OUT_MERGED, index=False)
    sz = os.path.getsize(OUT_MERGED) / 1024 / 1024
    oi_coverage = merged["open_interest"].notna().sum() / len(merged) * 100
    print(f"  Merged: {len(merged):,} bar | {sz:.1f} MB | OI coverage: {oi_coverage:.0f}%")
    return merged


def main():
    print("=" * 60)
    print("  ADIM 0: 5dk Mum + OI Parquet Cache")
    print("=" * 60)

    candles = build_5m_candles()
    oi = download_oi()
    merged = merge_data(candles, oi)

    print(f"\n  TAMAMLANDI")
    print(f"  5dk mum: {OUT_5M}")
    print(f"  OI:      {OUT_OI}")
    print(f"  Merged:  {OUT_MERGED}")


if __name__ == "__main__":
    main()

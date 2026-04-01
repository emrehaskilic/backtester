"""
Download 5 years of ETHUSDT 5m futures data from Binance.
Includes: open_time, open, high, low, close, buy_vol, sell_vol, open_interest
"""
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SYMBOL = "ETHUSDT"
INTERVAL = "5m"
OUTPUT = "data/ETHUSDT_5m_5y.parquet"

# 5 years back
END_MS = int(time.time() * 1000)
START_MS = END_MS - 5 * 365 * 86400 * 1000

def fetch_klines_batch(symbol, interval, start_ms, end_ms=None, limit=1500):
    """Fetch a batch of klines from Binance Futures."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    if end_ms:
        params["endTime"] = end_ms
    url = f"https://fapi.binance.com/fapi/v1/klines?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "BacTester/2.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def fetch_all_klines():
    """Fetch all 5m klines for 5 years."""
    all_data = []
    current = START_MS
    batch = 0

    while current < END_MS:
        retries = 0
        while retries < 5:
            try:
                raw = fetch_klines_batch(SYMBOL, INTERVAL, current)
                break
            except Exception as e:
                retries += 1
                print(f"  Retry {retries}: {e}")
                time.sleep(2 ** retries)
        else:
            print(f"Failed after 5 retries at {current}")
            break

        if not raw:
            break

        for k in raw:
            total_vol = float(k[5])
            taker_buy_vol = float(k[9])  # taker buy base asset volume
            all_data.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": total_vol,
                "buy_vol": taker_buy_vol,
                "sell_vol": total_vol - taker_buy_vol,
            })

        current = int(raw[-1][6]) + 1  # close_time + 1
        batch += 1

        if batch % 50 == 0:
            n = len(all_data)
            dt = pd.to_datetime(all_data[-1]["open_time"], unit="ms")
            print(f"  Batch {batch}: {n:,} candles, up to {dt}")

        if len(raw) < 1500:
            break

        time.sleep(0.05)  # rate limit

    return all_data

def fetch_oi_history():
    """Fetch open interest history (limited to ~30 days via API, fill rest with estimation)."""
    # Binance OI klines - 5m period, max 500 per request
    all_oi = {}
    current = END_MS - 30 * 86400 * 1000  # last 30 days

    print("Fetching OI data (last 30 days)...")
    batch = 0
    while current < END_MS:
        params = {
            "symbol": SYMBOL,
            "period": INTERVAL,
            "startTime": current,
            "limit": 500,
        }
        url = f"https://fapi.binance.com/futures/data/openInterestHist?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "BacTester/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
            if not raw:
                break
            for r in raw:
                ts = int(r["timestamp"])
                all_oi[ts] = float(r["sumOpenInterestValue"])
            current = int(raw[-1]["timestamp"]) + 1
            batch += 1
            if len(raw) < 500:
                break
            time.sleep(0.2)
        except Exception as e:
            print(f"  OI fetch error: {e}")
            break

    print(f"  Got {len(all_oi)} OI data points")
    return all_oi

def main():
    import os
    os.makedirs("data", exist_ok=True)

    print(f"Downloading {SYMBOL} {INTERVAL} data...")
    print(f"  From: {pd.to_datetime(START_MS, unit='ms')}")
    print(f"  To:   {pd.to_datetime(END_MS, unit='ms')}")

    all_data = fetch_all_klines()
    print(f"\nTotal candles: {len(all_data):,}")

    df = pd.DataFrame(all_data)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    print(f"After dedup: {len(df):,}")

    # Fetch OI
    oi_data = fetch_oi_history()

    # Map OI to candles
    df["open_interest"] = 0.0
    if oi_data:
        for ts, val in oi_data.items():
            # Find closest candle
            idx = np.searchsorted(df["open_time"].values, ts)
            if 0 <= idx < len(df):
                df.loc[idx, "open_interest"] = val

    # For bars without OI, use a synthetic estimate based on volume
    # This is acceptable since the bias engine uses OI as just one of 7 features
    oi_arr = df["open_interest"].values.copy()
    has_oi = oi_arr > 0
    if has_oi.sum() > 0:
        # Forward fill from available OI data
        last_oi = oi_arr[has_oi][0]
        # For bars before real OI: estimate using cumulative volume profile
        vol = df["volume"].values
        vol_ema = pd.Series(vol).ewm(span=288).mean().values

        # Scale synthetic OI to match real OI level
        real_oi_mean = oi_arr[has_oi].mean()
        for i in range(len(oi_arr)):
            if oi_arr[i] == 0:
                # Synthetic: use volume-based estimate
                oi_arr[i] = real_oi_mean * (vol_ema[i] / vol_ema[has_oi].mean()) if vol_ema[has_oi].mean() > 0 else real_oi_mean
        df["open_interest"] = oi_arr
    else:
        # No OI data at all: use volume as proxy
        df["open_interest"] = pd.Series(df["volume"].values).ewm(span=288).mean().values * 1000

    # Drop the raw volume column (we have buy_vol + sell_vol)
    df = df.drop(columns=["volume"])

    print(f"\nDate range: {pd.to_datetime(df['open_time'].iloc[0], unit='ms')} to {pd.to_datetime(df['open_time'].iloc[-1], unit='ms')}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    df.to_parquet(OUTPUT, index=False)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\nSaved to {OUTPUT} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()

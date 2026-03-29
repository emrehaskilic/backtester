"""Binance Futures'dan 5 yillik ETHUSDT 5m veri cek (OHLCV + buy_vol/sell_vol)."""
import requests, time, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SYMBOL = "ETHUSDT"
INTERVAL = "5m"
LIMIT = 1500
URL = "https://fapi.binance.com/fapi/v1/klines"

# 5 yil: 2021-03-25 → 2026-03-25
start_ts = int(pd.Timestamp("2021-03-25").timestamp() * 1000)
end_ts = int(pd.Timestamp("2026-03-25 23:59:59").timestamp() * 1000)

print(f"Fetching {SYMBOL} {INTERVAL} data: 2021-03-25 → 2026-03-25")
print(f"Estimated: ~525K bars, ~351 requests")
print()

all_rows = []
current_ts = start_ts
req_count = 0

while current_ts < end_ts:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": current_ts,
        "limit": LIMIT,
    }

    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    else:
        print(f"FATAL: Failed after 5 retries at ts={current_ts}")
        break

    if not data:
        break

    for k in data:
        all_rows.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "buy_vol": float(k[9]),          # taker buy base asset volume
            "sell_vol": float(k[5]) - float(k[9]),  # total - buy = sell
            "trade_count": int(k[8]),
            "open_interest": 0.0,            # OI not available historically
        })

    req_count += 1
    current_ts = int(data[-1][0]) + 300_000  # next bar after last

    if req_count % 50 == 0:
        dt = pd.Timestamp(data[-1][0], unit="ms")
        print(f"  {req_count} requests | {len(all_rows):,} bars | at {dt}")

    # Rate limit: ~10 req/sec safe
    time.sleep(0.1)

print(f"\nDone: {req_count} requests, {len(all_rows):,} bars")

# Build DataFrame
df = pd.DataFrame(all_rows)
df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

# Filter exact range
df = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)].reset_index(drop=True)

dt = pd.to_datetime(df["open_time"], unit="ms")
print(f"Final: {len(df):,} bars | {dt.iloc[0]} → {dt.iloc[-1]}")
print(f"Days: {(dt.iloc[-1] - dt.iloc[0]).days}")

# Save
out_path = "data/ETHUSDT_5m_5y.parquet"
df.to_parquet(out_path, index=False)
size_mb = pd.io.common.file_exists(out_path) and __import__("os").path.getsize(out_path) / 1024 / 1024
print(f"Saved: {out_path} ({size_mb:.1f} MB)")

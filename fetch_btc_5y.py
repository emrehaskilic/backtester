"""BTC spot + perp 5 yillik 5m veri cek."""
import requests, time, pandas as pd, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

LIMIT = 1500
start_ts = int(pd.Timestamp("2021-03-25").timestamp() * 1000)
end_ts = int(pd.Timestamp("2026-03-25 23:59:59").timestamp() * 1000)

def fetch(symbol, interval, url, limit, label):
    print(f"Fetching {label} {symbol} {interval}...")
    all_rows = []
    current_ts = start_ts
    req_count = 0
    while current_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": current_ts, "limit": limit}
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", 60)))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"FATAL at ts={current_ts}")
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
                "buy_vol": float(k[9]),
                "sell_vol": float(k[5]) - float(k[9]),
                "trade_count": int(k[8]),
                "open_interest": 0.0,
            })
        req_count += 1
        current_ts = int(data[-1][0]) + 300_000
        if req_count % 100 == 0:
            dt = pd.Timestamp(data[-1][0], unit="ms")
            print(f"  {req_count} req | {len(all_rows):,} bars | at {dt}")
        time.sleep(0.1)
    print(f"  Done: {req_count} req, {len(all_rows):,} bars")
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    df = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)].reset_index(drop=True)
    return df

# BTC Perp
df_perp = fetch("BTCUSDT", "5m", "https://fapi.binance.com/fapi/v1/klines", LIMIT, "PERP")
out = "data/BTCUSDT_5m_5y_perp.parquet"
df_perp.to_parquet(out, index=False)
dt = pd.to_datetime(df_perp['open_time'], unit='ms')
print(f"  Saved: {out} ({os.path.getsize(out)/1024/1024:.1f} MB) | {len(df_perp):,} bars | {dt.iloc[0]} - {dt.iloc[-1]}")

# BTC Spot
df_spot = fetch("BTCUSDT", "5m", "https://api.binance.com/api/v3/klines", 1000, "SPOT")
out2 = "data/BTCUSDT_5m_5y_spot.parquet"
df_spot.to_parquet(out2, index=False)
dt2 = pd.to_datetime(df_spot['open_time'], unit='ms')
print(f"  Saved: {out2} ({os.path.getsize(out2)/1024/1024:.1f} MB) | {len(df_spot):,} bars | {dt2.iloc[0]} - {dt2.iloc[-1]}")

print("\nDone!")

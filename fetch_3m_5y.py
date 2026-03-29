"""ETHUSDT 3m 5 yillik perp veri cek."""
import requests, time, pandas as pd, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SYMBOL = "ETHUSDT"
INTERVAL = "3m"
LIMIT = 1500
URL = "https://fapi.binance.com/fapi/v1/klines"

start_ts = int(pd.Timestamp("2021-03-25").timestamp() * 1000)
end_ts = int(pd.Timestamp("2026-03-25 23:59:59").timestamp() * 1000)

print(f"Fetching {SYMBOL} {INTERVAL} perp: 2021-03-25 → 2026-03-25")
all_rows = []
current_ts = start_ts
req_count = 0

while current_ts < end_ts:
    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": current_ts, "limit": LIMIT}
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=30)
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
    if not data: break
    for k in data:
        all_rows.append({
            "open_time": int(k[0]),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
            "buy_vol": float(k[9]),
            "sell_vol": float(k[5]) - float(k[9]),
            "trade_count": int(k[8]),
        })
    req_count += 1
    current_ts = int(data[-1][0]) + 180_000
    if req_count % 100 == 0:
        print(f"  {req_count} req | {len(all_rows):,} bars | at {pd.Timestamp(data[-1][0], unit='ms')}")
    time.sleep(0.1)

print(f"Done: {req_count} req, {len(all_rows):,} bars")
df = pd.DataFrame(all_rows)
df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
df = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)].reset_index(drop=True)
dt = pd.to_datetime(df['open_time'], unit='ms')
print(f"Final: {len(df):,} bars | {dt.iloc[0]} → {dt.iloc[-1]}")
out = "data/ETHUSDT_3m_5y_perp.parquet"
df.to_parquet(out, index=False)
print(f"Saved: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")

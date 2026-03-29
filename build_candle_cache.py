"""aggTrades CSV -> 3dk mum parquet cache. Bir kez calistir."""
import os, glob, time
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser("~/Desktop/ETHUSDT_1Y_AggTrades")
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
PERIOD_MS = 180_000  # 3dk

def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "ETHUSDT-aggTrades-*.csv")))
    print(f"CSV dosyalari: {len(csv_files)}")

    all_candles = []
    t0 = time.time()

    for f in csv_files:
        sz = os.path.getsize(f) / 1024 / 1024
        if sz < 1:
            continue
        basename = os.path.basename(f)
        print(f"  {basename} ({sz:.0f} MB)... ", end="", flush=True)
        t1 = time.time()

        df = pd.read_csv(f, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
        df["price"] = df["price"].astype(np.float64)
        df["quantity"] = df["quantity"].astype(np.float64)
        if df["is_buyer_maker"].dtype == object:
            df["is_buyer_maker"] = df["is_buyer_maker"].str.lower() == "true"

        df["candle_ts"] = (df["transact_time"] // PERIOD_MS) * PERIOD_MS

        candles = df.groupby("candle_ts").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("quantity", "sum"),
            trade_count=("price", "count"),
        ).reset_index().rename(columns={"candle_ts": "open_time"})

        all_candles.append(candles)
        print(f"{len(candles):,} mum ({time.time()-t1:.0f}s)")

    result = pd.concat(all_candles, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    result.to_parquet(OUT_PATH, index=False)

    sz_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\nToplam: {len(result):,} mum | {sz_mb:.1f} MB | {OUT_PATH}")
    print(f"Sure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

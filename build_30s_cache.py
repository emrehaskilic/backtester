"""
aggTrades CSV -> 30s mikro-mum parquet cache.
buy_vol ve sell_vol dahil. Bir kez calistir.
Ayrica 3dk ana mum parquet de guncellenir (buy_vol/sell_vol eklenir).
"""
import os, glob, time
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser("~/Desktop/ETHUSDT_1Y_AggTrades")
OUT_30S = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_30s_11mo.parquet")
OUT_3M = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_vol_11mo.parquet")

def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "ETHUSDT-aggTrades-*.csv")))
    print(f"CSV: {len(csv_files)} dosya")

    all_30s = []
    all_3m = []
    t0 = time.time()

    for f in csv_files:
        sz = os.path.getsize(f) / 1024 / 1024
        if sz < 1:
            continue
        bn = os.path.basename(f)
        print(f"  {bn} ({sz:.0f} MB)... ", end="", flush=True)
        t1 = time.time()

        df = pd.read_csv(f, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
        df["price"] = df["price"].astype(np.float64)
        df["quantity"] = df["quantity"].astype(np.float64)
        if df["is_buyer_maker"].dtype == object:
            df["is_buyer_maker"] = df["is_buyer_maker"].str.lower() == "true"
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

        # buy_qty / sell_qty
        df["buy_qty"] = df["quantity"] * (~df["is_buyer_maker"]).astype(np.float64)
        df["sell_qty"] = df["quantity"] * df["is_buyer_maker"].astype(np.float64)

        # 30s mumlar
        period_30s = 30_000
        df["ts_30s"] = (df["transact_time"] // period_30s) * period_30s
        c30 = df.groupby("ts_30s").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("quantity", "sum"),
            buy_vol=("buy_qty", "sum"), sell_vol=("sell_qty", "sum"),
            trade_count=("price", "count"),
        ).reset_index().rename(columns={"ts_30s": "open_time"})
        all_30s.append(c30)

        # 3dk mumlar
        period_3m = 180_000
        df["ts_3m"] = (df["transact_time"] // period_3m) * period_3m
        c3m = df.groupby("ts_3m").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("quantity", "sum"),
            buy_vol=("buy_qty", "sum"), sell_vol=("sell_qty", "sum"),
            trade_count=("price", "count"),
        ).reset_index().rename(columns={"ts_3m": "open_time"})
        all_3m.append(c3m)

        print(f"30s:{len(c30):,} 3m:{len(c3m):,} ({time.time()-t1:.0f}s)")
        del df

    # 30s
    r30 = pd.concat(all_30s, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    r30.to_parquet(OUT_30S, index=False)
    sz30 = os.path.getsize(OUT_30S) / 1024 / 1024
    print(f"\n30s: {len(r30):,} bar | {sz30:.1f} MB | {OUT_30S}")

    # 3m
    r3m = pd.concat(all_3m, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    r3m.to_parquet(OUT_3M, index=False)
    sz3m = os.path.getsize(OUT_3M) / 1024 / 1024
    print(f"3m:  {len(r3m):,} bar | {sz3m:.1f} MB | {OUT_3M}")

    print(f"\nToplam sure: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

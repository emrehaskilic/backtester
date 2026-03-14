"""
Tek pair backtest — hızlı test için.
Kullanım: python run_backtest.py [BTCUSDT]
"""

import sys
import os
import pandas as pd
from tick_engine import TickReplayEngine
from swinginess_strategy import SwingingessStrategy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def run_backtest(symbol, params=None):
    parquet_path = os.path.join(DATA_DIR, f"{symbol.lower()}_aggtrades.parquet")
    if not os.path.exists(parquet_path):
        print(f"Hata: {parquet_path} bulunamadı. Önce download_aggtrades.py çalıştırın.")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} TICK REPLAY BACKTEST")
    print(f"{'='*60}")

    # Veri oku
    print(f"  Veri okunuyor...")
    df = pd.read_parquet(parquet_path)
    n = len(df)
    print(f"  Toplam tick: {n:,}")

    # Engine ve strateji
    rolling_window = (params or {}).get("rolling_window_sec", 3600)
    engine = TickReplayEngine(rolling_window_sec=rolling_window)
    strategy = SwingingessStrategy(params)

    # Tick replay
    ts_arr = df["timestamp"].values
    price_arr = df["price"].values
    qty_arr = df["quantity"].values
    side_arr = df["side"].values

    last_bucket_ts = 0
    processed = 0

    for i in range(n):
        ts_ms = int(ts_arr[i])
        price = float(price_arr[i])
        qty = float(qty_arr[i])
        side = str(side_arr[i])

        engine.process_tick(ts_ms, price, qty, side)

        ts_sec = ts_ms // 1000
        if ts_sec == last_bucket_ts:
            continue
        last_bucket_ts = ts_sec
        processed += 1

        if not engine.warmup_done:
            continue

        strategy.on_second(ts_sec, price, engine)

        if strategy.equity <= 0:
            print(f"  TASFIYE — equity sıfırlandı")
            break

        # Progress
        if processed % 86400 == 0:  # her ~gün
            days = processed / 86400
            print(f"  Gün {days:.0f} | Equity: {strategy.equity:.2f} | İşlem: {len(strategy.trades)}")

    # Açık pozisyonu kapat
    if strategy.position != 0:
        strategy._close_position(float(price_arr[-1]), int(ts_arr[-1]) // 1000, "CLOSE")

    results = strategy.get_results()

    print(f"\n  SONUÇLAR:")
    print(f"  {'─'*40}")
    print(f"  Net P&L:        {results['net_pnl']:+,.2f} USDT ({results['net_pnl_pct']:+.2f}%)")
    print(f"  Profit Factor:  {results['profit_factor']:.3f}")
    print(f"  Win Rate:       {results['win_rate']:.1f}%")
    print(f"  Max Drawdown:   {results['max_drawdown']:.2f}%")
    print(f"  Total Trades:   {results['total_trades']}")
    print(f"  Avg Hold:       {results['avg_hold_sec']:.0f}s ({results['avg_hold_sec']/60:.1f}m)")
    print(f"  Equity Final:   {results['equity_final']:.2f}")
    print(f"  Exit Types:     {results['exit_types']}")

    return results


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    run_backtest(symbol)

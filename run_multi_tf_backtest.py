"""
Multi-TF Backtest Runner
1H sweep direction (5m data) + 3m KC DCA/TP
Weekly reset: 1000 USDT per week, 25x leverage
"""

import numpy as np
import pandas as pd
import rust_engine

# ── Load data ──
print("Loading 5m data...")
df5 = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"  5m bars: {len(df5):,}")

print("Loading 3m data...")
df3 = pd.read_parquet("data/ETHUSDT_3m_vol_11mo.parquet")
print(f"  3m bars: {len(df3):,}")

# ── Check columns ──
print(f"\n5m columns: {list(df5.columns)}")
print(f"3m columns: {list(df3.columns)}")

# ── Prepare 5m arrays ──
# Ensure timestamp is in milliseconds
if "timestamp" in df5.columns:
    ts5 = df5["timestamp"].values.astype(np.uint64)
elif "open_time" in df5.columns:
    ts5 = df5["open_time"].values.astype(np.uint64)
else:
    # Try index
    ts5 = (df5.index.astype(np.int64) // 10**6).values.astype(np.uint64)

closes_5m = np.ascontiguousarray(df5["close"].values, dtype=np.float64)
highs_5m = np.ascontiguousarray(df5["high"].values, dtype=np.float64)
lows_5m = np.ascontiguousarray(df5["low"].values, dtype=np.float64)
buy_vol_5m = np.ascontiguousarray(df5["buy_vol"].values, dtype=np.float64)
sell_vol_5m = np.ascontiguousarray(df5["sell_vol"].values, dtype=np.float64)
oi_5m = np.ascontiguousarray(df5["open_interest"].values, dtype=np.float64)
ts5 = np.ascontiguousarray(ts5)

# ── Prepare 3m arrays ──
if "timestamp" in df3.columns:
    ts3 = df3["timestamp"].values.astype(np.uint64)
elif "open_time" in df3.columns:
    ts3 = df3["open_time"].values.astype(np.uint64)
else:
    ts3 = (df3.index.astype(np.int64) // 10**6).values.astype(np.uint64)

closes_3m = np.ascontiguousarray(df3["close"].values, dtype=np.float64)
highs_3m = np.ascontiguousarray(df3["high"].values, dtype=np.float64)
lows_3m = np.ascontiguousarray(df3["low"].values, dtype=np.float64)
ts3 = np.ascontiguousarray(ts3)

# ── Run ──
print("\nRunning Multi-TF Strategy...")
print("  1H sweep direction from 5m data")
print("  3m KC DCA/TP trading")
print("  Weekly reset: 1000 USDT, 25x leverage")
print()

r = rust_engine.run_multi_tf_strategy_py(
    closes_5m, highs_5m, lows_5m,
    buy_vol_5m, sell_vol_5m, oi_5m, ts5,
    closes_3m, highs_3m, lows_3m, ts3,
)

# ── Results ──
print("=" * 60)
print("  MULTI-TF STRATEGY RESULTS")
print("  1H Sweep Direction + 3m KC Trading")
print("  Weekly Reset: 1000 USDT | Leverage: 25x")
print("=" * 60)

print(f"\n{'GENEL':^60}")
print(f"  Total Weeks:          {r['total_weeks']}")
print(f"  Total Trades:         {r['total_trades']}")
print(f"  Win Rate:             {r['win_rate']:.1f}%")
print(f"  Total PnL:            {r['total_pnl']:.2f} USDT")
print(f"  Total Fees:           {r['total_fees']:.2f} USDT")

print(f"\n{'HAFTALIK PERFORMANS':^60}")
print(f"  Avg Weekly PnL:       {r['avg_weekly_pnl']:.2f} USDT ({r['avg_weekly_pnl_pct']:.2f}%)")
print(f"  Median Weekly PnL:    {r['median_weekly_pnl_pct']:.2f}%")
print(f"  Best Week:            {r['best_week_pct']:.2f}%")
print(f"  Worst Week:           {r['worst_week_pct']:.2f}%")
print(f"  Positive Weeks:       {r['positive_weeks']}/{r['total_weeks']}")
print(f"  Negative Weeks:       {r['negative_weeks']}/{r['total_weeks']}")
print(f"  Max Consec Loss Weeks:{r['max_consecutive_loss_weeks']}")

print(f"\n{'RISK':^60}")
print(f"  Max DD Within Week:   {r['max_drawdown_within_week']:.2f}%")

print(f"\n{'TRADE DETAYLARI':^60}")
print(f"  TP Closes:            {r['total_tp_count']}")
print(f"  Signal Closes:        {r['total_signal_close']}")
print(f"  DCA Count:            {r['total_dca_count']}")
print(f"  Avg Hold (3m bars):   {r['avg_hold_bars_3m']:.1f}")

# Weekly breakdown
print(f"\n{'HAFTALIK KIRILIM':^60}")
print(f"  {'Hafta':>6} {'PnL':>10} {'PnL%':>8} {'Trades':>7} {'Wins':>6} {'MaxDD%':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")

weekly_pnl = np.array(r["weekly_pnl"])
weekly_pnl_pct = np.array(r["weekly_pnl_pct"])
weekly_trades = np.array(r["weekly_trades"])
weekly_wins = np.array(r["weekly_wins"])
weekly_max_dd = np.array(r["weekly_max_dd"])

for i in range(len(weekly_pnl)):
    marker = "+" if weekly_pnl[i] > 0 else "-" if weekly_pnl[i] < 0 else " "
    print(f"  {i+1:>5}{marker} {weekly_pnl[i]:>10.2f} {weekly_pnl_pct[i]:>7.2f}% {weekly_trades[i]:>7.0f} {weekly_wins[i]:>6.0f} {weekly_max_dd[i]:>7.2f}%")

# Summary
cum_pnl = np.cumsum(weekly_pnl)
print(f"\n  Cumulative PnL:       {cum_pnl[-1]:.2f} USDT" if len(cum_pnl) > 0 else "")
print(f"  If compounded (reinvest): N/A (weekly reset)")

# Trade log sample
trades = r.get("trades", [])
if trades:
    print(f"\n{'SON 20 TRADE':^60}")
    print(f"  {'Dir':>5} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'DCA':>4} {'Reason':>12}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*4} {'-'*12}")
    for t in trades[-20:]:
        d = "LONG" if t["direction"] == 1 else "SHORT"
        print(f"  {d:>5} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} {t['pnl']:>10.2f} {t['dca_count']:>4} {t['exit_reason']:>12}")

print(f"\n{'='*60}")

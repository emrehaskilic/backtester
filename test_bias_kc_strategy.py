"""
Bias Engine + KC Strategy Backtest
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# Load 5m data (bias engine input)
df5 = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"5m data: {len(df5)} bars")

# Load 3m data (KC input)
df3 = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
print(f"3m data: {len(df3)} bars")

# Find overlap period
ts5_min, ts5_max = df5["open_time"].iloc[0], df5["open_time"].iloc[-1]
ts3_min, ts3_max = df3["open_time"].iloc[0], df3["open_time"].iloc[-1]
overlap_start = max(ts5_min, ts3_min)
overlap_end = min(ts5_max, ts3_max)
print(f"\nOverlap: {pd.to_datetime(overlap_start, unit='ms')} to {pd.to_datetime(overlap_end, unit='ms')}")

# We use the full 5m data for bias (includes training warmup)
# and filter 3m to overlap period
df3_filtered = df3[(df3["open_time"] >= overlap_start) & (df3["open_time"] <= overlap_end)]
print(f"3m bars in overlap: {len(df3_filtered)}")

# Prepare arrays
ts_5m = df5["open_time"].values.astype(np.uint64)
open_5m = df5["open"].values.astype(np.float64)
high_5m = df5["high"].values.astype(np.float64)
low_5m = df5["low"].values.astype(np.float64)
close_5m = df5["close"].values.astype(np.float64)
buy_vol_5m = df5["buy_vol"].values.astype(np.float64)
sell_vol_5m = df5["sell_vol"].values.astype(np.float64)
oi_5m = df5["open_interest"].values.astype(np.float64)

ts_3m = df3_filtered["open_time"].values.astype(np.uint64)
high_3m = df3_filtered["high"].values.astype(np.float64)
low_3m = df3_filtered["low"].values.astype(np.float64)
close_3m = df3_filtered["close"].values.astype(np.float64)

print(f"\nRunning bias + KC strategy...")
t0 = time.time()
r = rust_engine.run_bias_kc_strategy(
    ts_5m, open_5m, high_5m, low_5m, close_5m, buy_vol_5m, sell_vol_5m, oi_5m,
    ts_3m, high_3m, low_3m, close_3m,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

print(f"\n{'='*60}")
print(f"STRATEGY RESULTS")
print(f"{'='*60}")
print(f"Total PnL: {r['total_pnl']:.2f} USDT")
print(f"Total Trades: {r['total_trades']}")
print(f"Win Rate: {r['win_rate']*100:.1f}%")
print(f"Total Fees: {r['total_fees']:.2f} USDT")
print(f"Max DD: {r['max_dd_pct']:.2f}%")
print(f"Avg Weekly PnL: {r['avg_weekly_pnl']:.2f} USDT")
print(f"Profitable Weeks: {r['profitable_weeks']}/{r['total_weeks']}")

print(f"\n{'='*60}")
print(f"WEEKLY BREAKDOWN")
print(f"{'='*60}")
print(f"{'Week':>4} {'PnL':>10} {'Trades':>7} {'Wins':>5} {'TP':>4} {'Rev':>4} {'MaxDD':>7} {'Balance':>10}")
for w in r['weeks']:
    print(f"{w['week']:>4} {w['pnl']:>10.2f} {w['trades']:>7} {w['wins']:>5} {w['tp']:>4} {w['reversal']:>4} {w['max_dd']:>7.2f} {w['end_balance']:>10.2f}")

# Summary stats
pnls = [w['pnl'] for w in r['weeks']]
if pnls:
    print(f"\nWeekly PnL stats:")
    print(f"  Mean: {np.mean(pnls):.2f}")
    print(f"  Std: {np.std(pnls):.2f}")
    print(f"  Sharpe (weekly): {np.mean(pnls)/np.std(pnls)*np.sqrt(52):.2f}" if np.std(pnls) > 0 else "  Sharpe: N/A")
    print(f"  Best week: {max(pnls):.2f}")
    print(f"  Worst week: {min(pnls):.2f}")

print(f"\n{'='*60}")
print(f"RECENT TRADES (last 20)")
print(f"{'='*60}")
print(f"{'Dir':>5} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Reason':>10} {'DCA':>4} {'Bias':>7} {'Conf':>6}")
for t in r['recent_trades'][:20]:
    print(f"{t['dir']:>5} {t['entry']:>10.2f} {t['exit']:>10.2f} {t['pnl']:>10.2f} {t['reason']:>10} {t['dca']:>4} {t['bias']:>+7.4f} {t['conf']:>6.3f}")

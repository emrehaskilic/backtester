"""
Konsensüs KC parametreleri ile full backtest karşılaştırma
A) Median:   KC(12, 3.9, 12) DCA=1
B) Top-10:   KC(14, 3.8, 15) DCA=1
C) Weighted: KC(18, 3.8, 16) DCA=3
D) Default:  KC(20, 2.0, 14) DCA=unlimited
"""

import numpy as np
import pandas as pd
import rust_engine

print("Loading data...")
df5 = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
df3 = pd.read_parquet("data/ETHUSDT_3m_vol_11mo.parquet")

ts5 = np.ascontiguousarray(df5["open_time"].values.astype(np.uint64))
c5 = np.ascontiguousarray(df5["close"].values, dtype=np.float64)
h5 = np.ascontiguousarray(df5["high"].values, dtype=np.float64)
l5 = np.ascontiguousarray(df5["low"].values, dtype=np.float64)
bv5 = np.ascontiguousarray(df5["buy_vol"].values, dtype=np.float64)
sv5 = np.ascontiguousarray(df5["sell_vol"].values, dtype=np.float64)
oi5 = np.ascontiguousarray(df5["open_interest"].values, dtype=np.float64)

ts3 = np.ascontiguousarray(df3["open_time"].values.astype(np.uint64))
c3 = np.ascontiguousarray(df3["close"].values, dtype=np.float64)
h3 = np.ascontiguousarray(df3["high"].values, dtype=np.float64)
l3 = np.ascontiguousarray(df3["low"].values, dtype=np.float64)

configs = [
    ("A) Median:   KC(12, 3.9, 12) DCA=1", 12, 3.9, 12, 1),
    ("B) Top-10:   KC(14, 3.8, 15) DCA=1", 14, 3.8, 15, 1),
    ("C) Weighted: KC(18, 3.8, 16) DCA=3", 18, 3.8, 16, 3),
    ("D) Default:  KC(20, 2.0, 14) DCA=inf", 20, 2.0, 14, 9999),
]

results = []

for name, kl, km, ka, md in configs:
    if md == 9999:
        r = rust_engine.run_multi_tf_strategy_py(
            c5, h5, l5, bv5, sv5, oi5, ts5, c3, h3, l3, ts3,
        )
    else:
        r = rust_engine.run_multi_tf_params_py(
            c5, h5, l5, bv5, sv5, oi5, ts5, c3, h3, l3, ts3,
            kl, km, ka, md,
        )
    results.append((name, r))

# -- Print each --
for name, r in results:
    wp = np.array(r["weekly_pnl"])
    wpp = np.array(r["weekly_pnl_pct"])
    wt = np.array(r["weekly_trades"])
    ww = np.array(r["weekly_wins"])
    wdd = np.array(r["weekly_max_dd"])
    pos = sum(1 for p in wp if p > 0)
    neg = sum(1 for p in wp if p < 0)
    cum = np.cumsum(wp)

    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  Weeks: {r['total_weeks']}  Trades: {r['total_trades']}  WR: {r['win_rate']:.1f}%")
    print(f"  Avg Weekly:    {r['avg_weekly_pnl']:.2f} USDT ({r['avg_weekly_pnl_pct']:.2f}%)")
    print(f"  Median Weekly: {r['median_weekly_pnl_pct']:.2f}%")
    print(f"  Best/Worst:    {r['best_week_pct']:.2f}% / {r['worst_week_pct']:.2f}%")
    print(f"  Pos/Neg Weeks: {pos}/{neg}")
    print(f"  Max DD Week:   {r['max_drawdown_within_week']:.2f}%")
    print(f"  TP: {r['total_tp_count']}  SigClose: {r['total_signal_close']}  DCA: {r['total_dca_count']}")
    print(f"  Cum PnL:       {cum[-1]:.2f} USDT")
    print(f"  Consec Loss:   {r['max_consecutive_loss_weeks']}")

    print(f"\n  {'Wk':>4} {'PnL':>9} {'PnL%':>7} {'Trd':>5} {'Win':>5} {'DD%':>6}")
    print(f"  {'-'*4} {'-'*9} {'-'*7} {'-'*5} {'-'*5} {'-'*6}")
    for i in range(len(wp)):
        m = "+" if wp[i] > 0 else "-" if wp[i] < 0 else " "
        print(f"  {i+1:>3}{m} {wp[i]:>9.2f} {wpp[i]:>6.2f}% {wt[i]:>5.0f} {ww[i]:>5.0f} {wdd[i]:>5.1f}%")

# -- Comparison table --
print(f"\n\n{'=' * 75}")
print(f"{'KARSILASTIRMA':^75}")
print(f"{'=' * 75}")
print(f"  {'Config':<35} {'AvgW%':>7} {'MedW%':>7} {'Pos/N':>6} {'Worst':>7} {'CumPnL':>9} {'MaxDD':>6}")
print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*9} {'-'*6}")

for name, r in results:
    wp = np.array(r["weekly_pnl"])
    pos = sum(1 for p in wp if p > 0)
    short_name = name.split(")")[0] + ")" + name.split(")")[1][:20]
    print(f"  {short_name:<35} {r['avg_weekly_pnl_pct']:>6.2f}% {r['median_weekly_pnl_pct']:>6.2f}% {pos:>3}/{r['total_weeks']-pos:<2} {r['worst_week_pct']:>6.1f}% {np.sum(wp):>9.1f} {r['max_drawdown_within_week']:>5.1f}%")

print(f"{'=' * 75}")

"""
PMax DCA Backtest Runner — 22-23 Mart 2026 ETHUSDT.P
Dual-loop: tick bazlı limit order + 3dk mum kapanışı PMax/KC.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from candle_builder import CandleBuilder, load_aggtrades
from indicators.adaptive_pmax import AdaptivePMax
from indicators.keltner_channel import KeltnerChannel
from pmax_dca_strategy import PMaxDCAStrategy

# ================================================================
# PARAMETRELER
# ================================================================

PARAMS = {
    # Katman 1: Adaptive PMax (9 parametre)
    "vol_lookback": 260,
    "flip_window": 360,
    "mult_base": 3.25,
    "mult_scale": 2.0,
    "ma_base": 11,
    "ma_scale": 4.5,
    "atr_base": 15,
    "atr_scale": 2.0,
    "update_interval": 55,

    # Katman 2: Keltner Channel (3 parametre)
    "kc_length": 3,
    "kc_multiplier": 0.5,
    "kc_atr_period": 2,

    # Katman 3: Graduated DCA (4 parametre)
    "dca_m1": 0.50,
    "dca_m2": 3.00,
    "dca_m3": 3.75,
    "dca_m4": 3.75,

    # Katman 4: Graduated TP (4 parametre)
    "tp1": 0.50,
    "tp2": 0.80,
    "tp3": 0.85,
    "tp4": 0.90,

    # Sabit parametreler
    "initial_balance": 1000,
    "leverage": 25,
    "maker_fee": 0.0002,
    "taker_fee": 0.0005,
    "margin_ratio": 1.0 / 40.0,  # %2.5
    "max_dca": 4,
    "pmax_atr_period": 10,
    "pmax_atr_multiplier": 3.0,
    "pmax_ma_length": 10,
}

# Veri dosyaları
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES = [
    os.path.join(os.path.dirname(DATA_DIR), "Desktop", "ETHUSDT-aggTrades-2026-03-22.csv"),
    os.path.join(os.path.dirname(DATA_DIR), "Desktop", "ETHUSDT-aggTrades-2026-03-23.csv"),
]

# bactester Desktop'ta olduğu için path düzelt
if not os.path.exists(CSV_FILES[0]):
    CSV_FILES = [
        os.path.expanduser("~/Desktop/ETHUSDT-aggTrades-2026-03-22.csv"),
        os.path.expanduser("~/Desktop/ETHUSDT-aggTrades-2026-03-23.csv"),
    ]


def run_backtest():
    print("=" * 70)
    print("  ADAPTIVE PMAX + KC DCA/TP BACKTEST")
    print("  ETHUSDT.P | 22-23 Mart 2026 | 3dk Mumlar + Tick Limit Orders")
    print("=" * 70)

    # Veri yükle
    print("\n  [1/3] Veri yukleniyor...")
    t0 = time.time()

    for f in CSV_FILES:
        if not os.path.exists(f):
            print(f"  HATA: {f} bulunamadi!")
            return
        print(f"    {os.path.basename(f)}")

    df = load_aggtrades(CSV_FILES)
    n = len(df)
    print(f"  Toplam tick: {n:,} ({time.time()-t0:.1f}s)")

    # Bileşenleri oluştur
    print("\n  [2/3] Engine hazirlaniyor...")
    candle_builder = CandleBuilder(period_sec=180)  # 3dk
    pmax = AdaptivePMax(PARAMS)
    kc = KeltnerChannel(PARAMS)
    strategy = PMaxDCAStrategy(PARAMS)

    # Arrays for speed
    ts_arr = df["timestamp"].values
    price_arr = df["price"].values
    qty_arr = df["quantity"].values
    buyer_maker_arr = df["is_buyer_maker"].values

    # Ana loop
    print("\n  [3/3] Backtest calisiyor...")
    t0 = time.time()
    candle_count = 0
    last_progress = 0

    for i in range(n):
        ts_ms = int(ts_arr[i])
        price = float(price_arr[i])
        qty = float(qty_arr[i])
        is_buyer_maker = bool(buyer_maker_arr[i])

        # 1. TICK LOOP — limit order kontrolü
        strategy.on_tick(ts_ms, price, qty, is_buyer_maker)

        # 2. CANDLE LOOP — mum kapanışı kontrolü
        candle_closed, completed = candle_builder.process_tick(ts_ms, price, qty, is_buyer_maker)

        if candle_closed and completed:
            candle_count += 1

            # İndikatörleri güncelle
            h = completed["high"]
            l = completed["low"]
            c = completed["close"]

            pmax_flipped, pmax_dir, pmax_stop = pmax.update(h, l, c)
            kc_upper, kc_middle, kc_lower, kc_atr = kc.update(h, l, c)

            # Strateji mum kapanışı eventi
            strategy.on_candle_close(
                pmax_flipped=pmax_flipped,
                pmax_direction=pmax_dir,
                pmax_stop=pmax_stop,
                kc_upper=kc_upper,
                kc_middle=kc_middle,
                kc_lower=kc_lower,
                kc_atr=kc_atr,
                candle_close=c,
                candle_ts=completed["timestamp"] // 1000,
            )

            # Progress
            if candle_count % 100 == 0:
                elapsed = time.time() - t0
                pct = i / n * 100
                eq = strategy.equity
                groups = len(strategy.trade_groups)
                pos = "LONG" if strategy.position.side == 1 else ("SHORT" if strategy.position.side == -1 else "FLAT")
                print(f"    Mum #{candle_count:5d} | %{pct:.1f} | Equity: ${eq:.2f} | "
                      f"Gruplar: {groups} | Poz: {pos} | {elapsed:.0f}s")

        # Equity sıfırlandıysa dur
        if strategy.equity <= 0:
            print("  *** TASFIYE — equity sifirlandi ***")
            break

    # Son açık pozisyonu kapat
    if strategy.position.side != 0:
        last_price = float(price_arr[-1])
        last_ts = int(ts_arr[-1]) // 1000
        strategy._close_entire_position(last_price, last_ts, "END_OF_DATA", use_taker=True)

    elapsed = time.time() - t0
    print(f"\n  Backtest tamamlandi: {elapsed:.1f}s | {candle_count} mum | {n:,} tick")

    # ================================================================
    # SONUÇLAR
    # ================================================================
    results = strategy.get_results()

    print(f"\n{'='*70}")
    print(f"  SONUCLAR")
    print(f"{'='*70}")
    print(f"  Baslangic Bakiye  : ${PARAMS['initial_balance']:,.2f}")
    print(f"  Final Equity      : ${results['equity_final']:,.2f}")
    print(f"  Net PnL           : ${results['net_pnl']:,.2f} ({results['net_pnl_pct']:.2f}%)")
    print(f"  {'-'*40}")
    print(f"  Trade Gruplari    : {results['total_groups']}")
    print(f"  Toplam Islem      : {results['total_trades']}")
    print(f"  Win Rate          : {results['win_rate']:.1f}%")
    print(f"  Profit Factor     : {results['profit_factor']:.3f}")
    print(f"  Max Drawdown      : {results['max_drawdown_pct']:.2f}%")
    print(f"  Toplam Komisyon   : ${results['total_fees']:.2f}")
    print(f"  Ort. DCA/Grup     : {results['avg_dca_per_group']:.2f}")
    print(f"  Ort. TP/Grup      : {results['avg_tp_per_group']:.2f}")

    # Trade log
    strategy.print_trade_log()

    # Trade detayları
    print(f"\n{'='*70}")
    print(f"  DETAYLI TRADE LISTESI")
    print(f"{'='*70}")
    for t in strategy.trades:
        ts_str = pd.Timestamp(t["ts"], unit="s").strftime("%Y-%m-%d %H:%M:%S")
        if t["type"].startswith("TP") or t["type"] in ("PMAX_FLIP", "END_OF_DATA"):
            pnl = t.get("pnl", 0)
            icon = "+" if pnl > 0 else ""
            print(f"  {ts_str} | {t['type']:12s} | Price: {t['price']:.2f} | "
                  f"Qty: {t.get('qty',0):.6f} | PnL: {icon}{pnl:.2f} | Eq: ${t['equity_after']:.2f}")
        else:
            print(f"  {ts_str} | {t['type']:12s} | Price: {t['price']:.2f} | "
                  f"Qty: {t.get('qty',0):.6f} | Margin: ${t.get('margin',0):.2f} | Eq: ${t['equity_after']:.2f}")

    return results


if __name__ == "__main__":
    run_backtest()

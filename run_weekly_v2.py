"""
Haftalik Backtest V2 — Rust engine birebir portu, LOOK-AHEAD DUZELTILMIS.
kc[i-1] kullanir. PMax crossover mantigi Rust ile ayni.
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from indicators.adaptive_pmax import AdaptivePMax
from indicators.keltner_channel import KeltnerChannel
from backtest_rust_port import run_backtest_no_lookahead

# ================================================================
# PARAMETRELER
# ================================================================
PARAMS = {
    "vol_lookback": 260, "flip_window": 360,
    "mult_base": 3.25, "mult_scale": 2.0,
    "ma_base": 11, "ma_scale": 4.5,
    "atr_base": 15, "atr_scale": 2.0,
    "update_interval": 55,
    "kc_length": 3, "kc_multiplier": 0.5, "kc_atr_period": 2,
    "pmax_atr_period": 10, "pmax_atr_multiplier": 3.0, "pmax_ma_length": 10,
}

DATA_DIR = os.path.expanduser("~/Desktop/ETHUSDT_1Y_AggTrades")
WARMUP_MONTHS = 2
INITIAL_BALANCE = 1000.0
MARGIN_RATIO = 1.0 / 40.0
CANDLE_SEC = 180


def csv_to_candles(csv_path):
    df = pd.read_csv(csv_path, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    df["price"] = df["price"].astype(np.float64)
    df["quantity"] = df["quantity"].astype(np.float64)
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].str.lower() == "true"
    period_ms = CANDLE_SEC * 1000
    df["candle_ts"] = (df["transact_time"] // period_ms) * period_ms
    candles = df.groupby("candle_ts").agg(
        open=("price", "first"), high=("price", "max"),
        low=("price", "min"), close=("price", "last"),
        volume=("quantity", "sum"),
    ).reset_index().rename(columns={"candle_ts": "timestamp"})
    return candles.sort_values("timestamp").reset_index(drop=True)


def compute_indicators(candles_df, params):
    """Tum mumlar icin PMax ve KC hesapla, numpy array dondur."""
    n = len(candles_df)
    pmax = AdaptivePMax(params)
    kc = KeltnerChannel(params)

    pmax_line_arr = np.full(n, np.nan)
    mavg_arr = np.full(n, np.nan)
    kc_upper_arr = np.full(n, np.nan)
    kc_lower_arr = np.full(n, np.nan)

    for i in range(n):
        h = candles_df["high"].iloc[i]
        l = candles_df["low"].iloc[i]
        c = candles_df["close"].iloc[i]

        flipped, direction, pmax_stop = pmax.update(h, l, c)
        kc_upper, kc_middle, kc_lower, kc_atr = kc.update(h, l, c)

        # PMax line ve mavg (Rust ile ayni crossover mantigi)
        # mavg = MA, pmax_line = pmax_stop
        # crossover: mavg > pmax_line = LONG, mavg < pmax_line = SHORT
        pmax_line_arr[i] = pmax_stop
        # mavg icin MA degerini kullan
        mavg_arr[i] = pmax._calc_ma(pmax.active_ma_length) if len(pmax.closes) >= pmax.active_ma_length else np.nan

        kc_upper_arr[i] = kc_upper
        kc_lower_arr[i] = kc_lower

    return pmax_line_arr, mavg_arr, kc_upper_arr, kc_lower_arr


def get_week_boundaries(start_date, end_date):
    weeks = []
    d = start_date
    while d.weekday() != 0:
        d += timedelta(days=1)
    week_num = 1
    while d < end_date:
        week_start = d
        week_end = min(d + timedelta(days=7), end_date)
        weeks.append({
            "label": f"W{week_num:02d}",
            "date_range": f"{week_start.strftime('%d %b')} - {(week_end - timedelta(days=1)).strftime('%d %b %Y')}",
            "start_ms": int(week_start.timestamp() * 1000),
            "end_ms": int(week_end.timestamp() * 1000),
        })
        d = week_end
        week_num += 1
    return weeks


def run():
    print("=" * 100)
    print("  RUST PORT BACKTEST — LOOK-AHEAD DUZELTILMIS (kc[i-1])")
    print("  ETHUSDT.P | 11 Ay | 2 Ay Warmup + 9 Ay Aktif | Haftalik Kar Cekimi")
    print("=" * 100)

    # CSV dosyalari
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "ETHUSDT-aggTrades-*.csv")))
    months = []
    for f in csv_files:
        parts = os.path.basename(f).replace(".csv", "").split("-")
        y, m = int(parts[2]), int(parts[3])
        sz = os.path.getsize(f) / 1024 / 1024
        if sz < 1:
            continue
        months.append((y, m, f))
        print(f"    {y}-{m:02d} ({sz:.0f} MB)")
    months.sort()

    warmup_months = months[:WARMUP_MONTHS]
    active_months = months[WARMUP_MONTHS:]
    print(f"\n  Warmup: {', '.join(f'{y}-{m:02d}' for y,m,_ in warmup_months)}")
    print(f"  Aktif:  {', '.join(f'{y}-{m:02d}' for y,m,_ in active_months)}")

    # Mumlari olustur
    print(f"\n  Mumlar olusturuluyor...")
    t0 = time.time()
    all_candles = []
    for y, m, path in months:
        print(f"    {y}-{m:02d}: ", end="", flush=True)
        c = csv_to_candles(path)
        print(f"{len(c):,} mum")
        all_candles.append(c)
    candles_df = pd.concat(all_candles, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"  Toplam: {len(candles_df):,} mum ({time.time()-t0:.0f}s)")

    # Indikatorleri hesapla
    print(f"\n  Indikatorler hesaplaniyor...")
    t0 = time.time()
    pmax_line, mavg, kc_upper, kc_lower = compute_indicators(candles_df, PARAMS)
    print(f"  Tamamlandi ({time.time()-t0:.1f}s)")

    # Numpy arrays
    ts_arr = candles_df["timestamp"].values
    closes = candles_df["close"].values.astype(np.float64)
    highs = candles_df["high"].values.astype(np.float64)
    lows = candles_df["low"].values.astype(np.float64)

    # Warmup/aktif ayirimi
    wy, wm, _ = warmup_months[-1]
    if wm == 12:
        active_start_ms = int(datetime(wy + 1, 1, 1).timestamp() * 1000)
    else:
        active_start_ms = int(datetime(wy, wm + 1, 1).timestamp() * 1000)

    warmup_mask = ts_arr < active_start_ms
    active_mask = ts_arr >= active_start_ms
    warmup_count = warmup_mask.sum()
    active_count = active_mask.sum()
    print(f"  Warmup: {warmup_count:,} mum | Aktif: {active_count:,} mum")

    # Hafta sinirlari
    ay, am, _ = active_months[0]
    ly, lm, _ = active_months[-1]
    active_start = datetime(ay, am, 1)
    active_end = datetime(ly, lm + 1, 1) if lm < 12 else datetime(ly + 1, 1, 1)
    weeks = get_week_boundaries(active_start, active_end)
    print(f"  Hafta sayisi: {len(weeks)}\n")

    # Aktif baslangic index
    active_start_idx = int(np.searchsorted(ts_arr, active_start_ms))

    # Haftalik backtest
    weekly_results = []

    for week in weeks:
        # Bu haftanin bar index araligi
        w_start = int(np.searchsorted(ts_arr, week["start_ms"]))
        w_end = int(np.searchsorted(ts_arr, week["end_ms"]))

        if w_end <= w_start:
            continue

        # Bu hafta icin slice (ama indikatorler GLOBAL index ile)
        # Backtest fonksiyonuna tum veriyi verip, sadece bu haftanin
        # barlarini isletmek lazim. Ama basitlik icin:
        # Haftanin barlarini + onceki MIN_BARS bari alalim
        slice_start = max(0, w_start - 201)
        slice_end = w_end

        result = run_backtest_no_lookahead(
            closes=closes[slice_start:slice_end],
            highs=highs[slice_start:slice_end],
            lows=lows[slice_start:slice_end],
            pmax_line=pmax_line[slice_start:slice_end],
            mavg_arr=mavg[slice_start:slice_end],
            kc_upper=kc_upper[slice_start:slice_end],
            kc_lower=kc_lower[slice_start:slice_end],
            initial_balance=INITIAL_BALANCE,
            margin_ratio=MARGIN_RATIO,
            max_dca_steps=4,
            tp_close_pct=0.50,
        )

        profit = max(0, result["balance"] - INITIAL_BALANCE)
        pnl = result["balance"] - INITIAL_BALANCE

        wr = {
            "label": week["label"],
            "date_range": week["date_range"],
            "end_equity": round(result["balance"], 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / INITIAL_BALANCE * 100, 2),
            "withdrawn": round(profit, 2),
            "total_trades": result["total_trades"],
            "win_rate": round(result["win_rate"], 1),
            "profit_factor": 0,
            "max_dd_pct": round(result["max_drawdown"], 2),
            "fees": round(result["total_fees"], 2),
            "tp": result["tp_count"],
            "rev": result["rev_count"],
            "hs": result["hard_stop_count"],
        }
        weekly_results.append(wr)

        sign = "+" if wr["pnl"] >= 0 else ""
        print(f"  {wr['label']} | {wr['date_range']:24s} | "
              f"PnL: {sign}${wr['pnl']:>8.2f} ({sign}{wr['pnl_pct']:>5.1f}%) | "
              f"Trades: {wr['total_trades']:>3d} | WR: {wr['win_rate']:>4.0f}% | "
              f"TP:{wr['tp']:>2d} Rev:{wr['rev']:>2d} HS:{wr['hs']:>2d} | "
              f"DD: {wr['max_dd_pct']:>5.1f}%")

    # Rapor
    print_report(weekly_results)


def print_report(wr_list):
    if not wr_list:
        print("\n  Sonuc yok!")
        return

    print(f"\n\n{'='*105}")
    print(f"  HAFTALIK RAPOR — LOOK-AHEAD DUZELTILMIS (kc[i-1])")
    print(f"  ETHUSDT.P | $1,000 | 25x | Margin: Kasa/40 | Hard Stop: %2.5")
    print(f"{'='*105}")

    print(f"  {'Hafta':5s} | {'Tarih':24s} | {'Bitis':>9s} | {'PnL':>10s} | {'PnL%':>7s} | "
          f"{'Cekilen':>9s} | {'Trd':>3s} | {'WR%':>5s} | {'TP':>3s} | {'Rev':>3s} | {'HS':>3s} | {'DD%':>6s}")
    print(f"  {'-'*103}")

    total_pnl = 0
    total_withdrawn = 0
    total_fees = 0
    total_trades = 0
    win_weeks = 0
    loss_weeks = 0
    max_profit = -1e18
    max_loss = 1e18
    best_label = worst_label = ""
    consec_w = consec_l = max_cw = max_cl = 0

    for wr in wr_list:
        sign = "+" if wr["pnl"] >= 0 else ""
        print(f"  {wr['label']:5s} | {wr['date_range']:24s} | "
              f"${wr['end_equity']:>8,.2f} | {sign}${wr['pnl']:>8,.2f} | {sign}{wr['pnl_pct']:>5.1f}% | "
              f"${wr['withdrawn']:>8,.2f} | {wr['total_trades']:>3d} | {wr['win_rate']:>4.0f}% | "
              f"{wr['tp']:>3d} | {wr['rev']:>3d} | {wr['hs']:>3d} | {wr['max_dd_pct']:>5.1f}%")

        total_pnl += wr["pnl"]
        total_withdrawn += wr["withdrawn"]
        total_fees += wr["fees"]
        total_trades += wr["total_trades"]

        if wr["pnl"] > max_profit: max_profit = wr["pnl"]; best_label = wr["label"]
        if wr["pnl"] < max_loss: max_loss = wr["pnl"]; worst_label = wr["label"]

        if wr["pnl"] > 0:
            win_weeks += 1; consec_w += 1; consec_l = 0
            if consec_w > max_cw: max_cw = consec_w
        elif wr["pnl"] < 0:
            loss_weeks += 1; consec_l += 1; consec_w = 0
            if consec_l > max_cl: max_cl = consec_l
        else:
            consec_w = 0; consec_l = 0

    n = len(wr_list)
    print(f"  {'-'*103}")
    print(f"\n  OZET")
    print(f"  {'-'*55}")
    print(f"  Toplam Hafta            : {n}")
    print(f"  Karli / Zararli         : {win_weeks} ({win_weeks/n*100:.0f}%) / {loss_weeks} ({loss_weeks/n*100:.0f}%)")
    print(f"  Toplam Net PnL          : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar      : ${total_withdrawn:>10,.2f}")
    print(f"  Toplam Komisyon         : ${total_fees:>10,.2f}")
    print(f"  Ort. Haftalik PnL       : ${total_pnl/n:>10,.2f} ({total_pnl/n/INITIAL_BALANCE*100:.2f}%)")
    print(f"  En Iyi Hafta            : {best_label} ${max_profit:>10,.2f}")
    print(f"  En Kotu Hafta           : {worst_label} ${max_loss:>10,.2f}")
    print(f"  Toplam Trade            : {total_trades}")
    print(f"  Maks Ust Uste Karli     : {max_cw} hafta")
    print(f"  Maks Ust Uste Zararli   : {max_cl} hafta")


if __name__ == "__main__":
    run()

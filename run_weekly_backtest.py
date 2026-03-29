"""
Haftalik Backtest Runner — 1 Yillik ETHUSDT.P (OPTIMIZED)
Her hafta $1,000 ile baslar, kar varsa cekilir.
Ilk 2 ay warmup (trade yok), son 9 ay aktif backtest.

Optimizasyon: Tick bazli dongu yerine 3dk mum bazli iterasyon.
Limit order fill: mumun high/low'u ile kontrol (penetrasyon).

Kullanim: python run_weekly_backtest.py
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
    "margin_ratio": 1.0 / 40.0,
    "max_dca": 4,
    "pmax_atr_period": 10,
    "pmax_atr_multiplier": 3.0,
    "pmax_ma_length": 10,
}

DATA_DIR = os.path.expanduser("~/Desktop/ETHUSDT_1Y_AggTrades")
WARMUP_MONTHS = 2
INITIAL_BALANCE = 1000.0
CANDLE_PERIOD_SEC = 180  # 3dk


def csv_to_candles(csv_path):
    """
    aggTrades CSV'yi pandas ile hizlica 3dk OHLCV mumlara cevir.
    Tick bazli dongu yok — tamamen vektorize.
    """
    df = pd.read_csv(csv_path, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])

    df["price"] = df["price"].astype(np.float64)
    df["quantity"] = df["quantity"].astype(np.float64)

    # is_buyer_maker normalize
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].str.lower() == "true"
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

    # 3dk periyoda hizala (ms cinsinden)
    period_ms = CANDLE_PERIOD_SEC * 1000
    df["candle_ts"] = (df["transact_time"] // period_ms) * period_ms

    # Gruplama ile OHLCV
    candles = df.groupby("candle_ts").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        trade_count=("price", "count"),
    ).reset_index()

    candles.rename(columns={"candle_ts": "timestamp"}, inplace=True)
    candles = candles.sort_values("timestamp").reset_index(drop=True)

    return candles


def get_week_boundaries(start_date, end_date):
    """Haftalik sinirlar olustur (Pazartesi 00:00 UTC)."""
    weeks = []
    d = start_date
    while d.weekday() != 0:
        d += timedelta(days=1)

    week_num = 1
    while d < end_date:
        week_start = d
        week_end = d + timedelta(days=7)
        if week_end > end_date:
            week_end = end_date

        start_ms = int(week_start.timestamp() * 1000)
        end_ms = int(week_end.timestamp() * 1000)
        label = f"W{week_num:02d}"
        date_range = f"{week_start.strftime('%d %b')} - {(week_end - timedelta(days=1)).strftime('%d %b %Y')}"

        weeks.append({
            "label": label,
            "date_range": date_range,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

        d = week_end
        week_num += 1

    return weeks


class CandleLevelStrategy:
    """
    Mum bazli strateji wrapper.
    Limit order fill kontrolu: mumun high/low'u ile penetrasyon.
    PMax/KC: mum kapanisinda guncellenir.
    """

    def __init__(self, params):
        self.strategy = PMaxDCAStrategy(params)
        self.pmax = None  # disaridan set edilecek
        self.kc = None

    def set_indicators(self, pmax, kc):
        """Indicator referanslarini ayarla."""
        self.strategy.pmax_direction = pmax.direction
        self.strategy.pmax_ready = pmax.direction != 0
        self.strategy.kc_upper = kc.upper
        self.strategy.kc_lower = kc.lower
        self.strategy.kc_middle = kc.middle
        self.strategy.kc_atr = kc.atr

    def process_candle(self, ts_ms, o, h, l, c, pmax, kc):
        """
        Tek bir 3dk mumu isle.
        1. Mumun high/low ile limit order fill kontrolu (tick simülasyonu)
        2. Mum kapanisinda PMax/KC guncelle
        """
        ts_sec = ts_ms // 1000

        # === 1. LIMIT ORDER FILL KONTROLU (mum ici) ===
        # DCA emirleri: BUY ise low < order.price, SELL ise high > order.price
        # TP emirleri: ayni mantik
        # Penetrasyon filtresi: strict less/greater (touch yetmez)

        filled_dca = []
        for order in self.strategy.pending_dca_orders:
            if order.side == "BUY" and l < order.price:
                filled_dca.append(order)
            elif order.side == "SELL" and h > order.price:
                filled_dca.append(order)

        for order in filled_dca:
            self.strategy._fill_dca_order(order, ts_sec)
            self.strategy.pending_dca_orders.remove(order)

        filled_tp = []
        for order in self.strategy.pending_tp_orders:
            if order.side == "SELL" and h > order.price:
                filled_tp.append(order)
            elif order.side == "BUY" and l < order.price:
                filled_tp.append(order)

        for order in filled_tp:
            self.strategy._fill_tp_order(order, ts_sec)
            self.strategy.pending_tp_orders.remove(order)

        self.strategy._update_equity_tracking()

        # === 2. MUM KAPANISI — INDIKATOR GUNCELLEME ===
        pmax_flipped, pmax_dir, pmax_stop = pmax.update(h, l, c)
        kc_upper, kc_middle, kc_lower, kc_atr = kc.update(h, l, c)

        # Strateji mum kapanisi eventi
        self.strategy.on_candle_close(
            pmax_flipped=pmax_flipped,
            pmax_direction=pmax_dir,
            pmax_stop=pmax_stop,
            kc_upper=kc_upper,
            kc_middle=kc_middle,
            kc_lower=kc_lower,
            kc_atr=kc_atr,
            candle_close=c,
            candle_ts=ts_sec,
        )

        return pmax_flipped, pmax_dir


def run():
    print("=" * 100)
    print("  ADAPTIVE PMAX + KC DCA/TP  --  HAFTALIK BACKTEST (OPTIMIZED)")
    print("  ETHUSDT.P | 11 Ay Data | 2 Ay Warmup + 9 Ay Aktif | Haftalik Kar Cekimi")
    print("=" * 100)

    # ================================================================
    # 1. CSV dosyalarini bul
    # ================================================================
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "ETHUSDT-aggTrades-*.csv")))
    if not csv_files:
        print(f"  HATA: {DATA_DIR} klasorunde CSV bulunamadi!")
        return

    print(f"\n  Bulunan CSV dosyalari: {len(csv_files)}")

    months_available = []
    for f in csv_files:
        basename = os.path.basename(f)
        parts = basename.replace(".csv", "").split("-")
        year = int(parts[2])
        month = int(parts[3])
        size_mb = os.path.getsize(f) / 1024 / 1024
        if size_mb < 1:
            print(f"    {basename} ({size_mb:.0f} MB) -- BOS, ATLANIYOR")
            continue
        print(f"    {basename} ({size_mb:.0f} MB)")
        months_available.append((year, month, f))

    months_available.sort()

    warmup_months = months_available[:WARMUP_MONTHS]
    active_months = months_available[WARMUP_MONTHS:]

    print(f"\n  Warmup: {', '.join(f'{y}-{m:02d}' for y, m, _ in warmup_months)}")
    print(f"  Aktif:  {', '.join(f'{y}-{m:02d}' for y, m, _ in active_months)}")

    # ================================================================
    # 2. TUM VERIYI MUMLARA CEVIR (vektorize, hizli)
    # ================================================================
    print(f"\n  Mumlar olusturuluyor (vektorize)...")
    t_total = time.time()

    all_candles = []
    for year, month, csv_path in months_available:
        t0 = time.time()
        print(f"    {year}-{month:02d}: ", end="", flush=True)
        candles = csv_to_candles(csv_path)
        print(f"{len(candles):,} mum ({time.time()-t0:.1f}s)")
        all_candles.append(candles)

    candles_df = pd.concat(all_candles, ignore_index=True)
    candles_df = candles_df.sort_values("timestamp").reset_index(drop=True)
    total_candles = len(candles_df)
    print(f"  Toplam: {total_candles:,} mum ({time.time()-t_total:.1f}s)")

    # Warmup/aktif ayrilimi
    warmup_end_year, warmup_end_month, _ = warmup_months[-1]
    if warmup_end_month == 12:
        active_start_ms = int(datetime(warmup_end_year + 1, 1, 1).timestamp() * 1000)
    else:
        active_start_ms = int(datetime(warmup_end_year, warmup_end_month + 1, 1).timestamp() * 1000)

    warmup_candles = candles_df[candles_df["timestamp"] < active_start_ms]
    active_candles = candles_df[candles_df["timestamp"] >= active_start_ms]

    print(f"  Warmup mumlar: {len(warmup_candles):,}")
    print(f"  Aktif mumlar:  {len(active_candles):,}")

    # ================================================================
    # 3. WARMUP — indikatorleri isit
    # ================================================================
    print(f"\n  WARMUP basliyor...")
    t0 = time.time()
    pmax = AdaptivePMax(PARAMS)
    kc = KeltnerChannel(PARAMS)

    for _, row in warmup_candles.iterrows():
        pmax.update(row["high"], row["low"], row["close"])
        kc.update(row["high"], row["low"], row["close"])

    print(f"  Warmup tamamlandi: {len(warmup_candles):,} mum ({time.time()-t0:.1f}s)")
    print(f"  PMax: {'LONG' if pmax.direction == 1 else 'SHORT' if pmax.direction == -1 else 'NONE'}")
    print(f"  KC: Upper={kc.upper:.2f} | Middle={kc.middle:.2f} | Lower={kc.lower:.2f}")

    # ================================================================
    # 4. HAFTALIK BACKTEST
    # ================================================================
    first_active = active_months[0]
    last_active = active_months[-1]
    active_start_date = datetime(first_active[0], first_active[1], 1)
    if last_active[1] == 12:
        active_end_date = datetime(last_active[0] + 1, 1, 1)
    else:
        active_end_date = datetime(last_active[0], last_active[1] + 1, 1)

    weeks = get_week_boundaries(active_start_date, active_end_date)
    print(f"\n  Toplam hafta: {len(weeks)}")
    print(f"  Aktif backtest basliyor...\n")

    # Numpy arrays for speed
    ts_arr = active_candles["timestamp"].values
    o_arr = active_candles["open"].values.astype(np.float64)
    h_arr = active_candles["high"].values.astype(np.float64)
    l_arr = active_candles["low"].values.astype(np.float64)
    c_arr = active_candles["close"].values.astype(np.float64)

    weekly_results = []
    candle_idx = 0
    n_candles = len(ts_arr)

    for week in weeks:
        # Yeni strateji ($1,000)
        wrapper = CandleLevelStrategy(PARAMS)
        wrapper.set_indicators(pmax, kc)

        # Bu haftanin mumlari
        week_candle_count = 0

        while candle_idx < n_candles and ts_arr[candle_idx] < week["end_ms"]:
            ts = int(ts_arr[candle_idx])
            o = float(o_arr[candle_idx])
            h = float(h_arr[candle_idx])
            l = float(l_arr[candle_idx])
            c = float(c_arr[candle_idx])

            wrapper.process_candle(ts, o, h, l, c, pmax, kc)
            week_candle_count += 1
            candle_idx += 1

            if wrapper.strategy.equity <= 0:
                break

        # Hafta sonu: acik pozisyonu kapat
        strat = wrapper.strategy
        if strat.position.side != 0 and week_candle_count > 0:
            last_c = float(c_arr[candle_idx - 1])
            last_ts = int(ts_arr[candle_idx - 1]) // 1000
            strat._close_entire_position(last_c, last_ts, "WEEK_END", use_taker=True)

        results = strat.get_results()
        profit = max(0, strat.equity - INITIAL_BALANCE)

        wr = {
            "label": week["label"],
            "date_range": week["date_range"],
            "start_equity": INITIAL_BALANCE,
            "end_equity": round(strat.equity, 2),
            "pnl": round(strat.equity - INITIAL_BALANCE, 2),
            "pnl_pct": round((strat.equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
            "withdrawn": round(profit, 2),
            "trade_groups": results["total_groups"],
            "total_trades": results["total_trades"],
            "win_rate": results["win_rate"],
            "profit_factor": results["profit_factor"],
            "max_dd_pct": results["max_drawdown_pct"],
            "fees": results["total_fees"],
            "candles": week_candle_count,
        }
        weekly_results.append(wr)

        sign = "+" if wr["pnl"] >= 0 else ""
        pf_str = f"{wr['profit_factor']:.2f}" if wr['profit_factor'] < 999 else "INF"
        print(f"  {wr['label']} | {wr['date_range']:24s} | "
              f"PnL: {sign}${wr['pnl']:>8.2f} ({sign}{wr['pnl_pct']:>5.1f}%) | "
              f"Grp: {wr['trade_groups']:>3d} | WR: {wr['win_rate']:>4.0f}% | "
              f"PF: {pf_str:>5s} | DD: {wr['max_dd_pct']:>5.1f}% | "
              f"Mum: {wr['candles']}")

    # ================================================================
    # 5. RAPOR
    # ================================================================
    print_report(weekly_results)


def print_report(weekly_results):
    """Haftalik sonuc tablosu."""
    if not weekly_results:
        print("\n  Sonuc yok!")
        return

    print(f"\n\n{'='*115}")
    print(f"  HAFTALIK BACKTEST RAPORU")
    print(f"  ETHUSDT.P | Adaptive PMax + KC DCA/TP | $1,000 Baslangic | 25x | Margin: Kasa/40")
    print(f"{'='*115}")

    header = (f"  {'Hafta':5s} | {'Tarih Araligi':24s} | {'Baslangic':>10s} | {'Bitis':>10s} | "
              f"{'PnL':>10s} | {'PnL%':>7s} | {'Cekilen':>9s} | {'Grp':>4s} | "
              f"{'WR%':>5s} | {'PF':>6s} | {'MaxDD%':>6s} | {'Fee':>7s}")
    print(header)
    print(f"  {'-'*113}")

    total_withdrawn = 0
    total_pnl = 0
    total_fees = 0
    win_weeks = 0
    loss_weeks = 0
    even_weeks = 0
    total_groups = 0
    max_weekly_profit = -float('inf')
    max_weekly_loss = float('inf')
    best_week_label = ""
    worst_week_label = ""
    consecutive_wins = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0

    for wr in weekly_results:
        sign = "+" if wr["pnl"] >= 0 else ""
        pf_str = f"{wr['profit_factor']:.2f}" if wr['profit_factor'] < 999 else "  INF"

        print(f"  {wr['label']:5s} | {wr['date_range']:24s} | "
              f"${wr['start_equity']:>9,.2f} | ${wr['end_equity']:>9,.2f} | "
              f"{sign}${wr['pnl']:>8,.2f} | {sign}{wr['pnl_pct']:>5.1f}% | "
              f"${wr['withdrawn']:>8,.2f} | {wr['trade_groups']:>4d} | "
              f"{wr['win_rate']:>4.0f}% | {pf_str:>6s} | {wr['max_dd_pct']:>5.1f}% | "
              f"${wr['fees']:>6.2f}")

        total_withdrawn += wr["withdrawn"]
        total_pnl += wr["pnl"]
        total_fees += wr["fees"]
        total_groups += wr["trade_groups"]

        if wr["pnl"] > max_weekly_profit:
            max_weekly_profit = wr["pnl"]
            best_week_label = wr["label"]
        if wr["pnl"] < max_weekly_loss:
            max_weekly_loss = wr["pnl"]
            worst_week_label = wr["label"]

        if wr["pnl"] > 0:
            win_weeks += 1
            consecutive_wins += 1
            consecutive_losses = 0
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins
        elif wr["pnl"] < 0:
            loss_weeks += 1
            consecutive_losses += 1
            consecutive_wins = 0
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses
        else:
            even_weeks += 1
            consecutive_wins = 0
            consecutive_losses = 0

    n_weeks = len(weekly_results)
    avg_pnl = total_pnl / n_weeks if n_weeks > 0 else 0
    win_pct = win_weeks / n_weeks * 100 if n_weeks > 0 else 0

    print(f"  {'-'*113}")
    print(f"\n  GENEL OZET")
    print(f"  {'-'*55}")
    print(f"  Toplam Hafta            : {n_weeks}")
    print(f"  Karli Hafta             : {win_weeks} ({win_pct:.0f}%)")
    print(f"  Zararli Hafta           : {loss_weeks} ({loss_weeks/n_weeks*100:.0f}%)" if n_weeks > 0 else "")
    print(f"  Basabaslik Hafta        : {even_weeks}")
    print(f"  {'-'*55}")
    print(f"  Toplam Net PnL          : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar      : ${total_withdrawn:>10,.2f}")
    print(f"  Toplam Komisyon         : ${total_fees:>10,.2f}")
    print(f"  {'-'*55}")
    print(f"  Ortalama Haftalik PnL   : ${avg_pnl:>10,.2f} ({avg_pnl/INITIAL_BALANCE*100:.2f}%)")
    print(f"  En Iyi Hafta            : {best_week_label} ${max_weekly_profit:>10,.2f}")
    print(f"  En Kotu Hafta           : {worst_week_label} ${max_weekly_loss:>10,.2f}")
    print(f"  {'-'*55}")
    print(f"  Toplam Trade Grubu      : {total_groups}")
    print(f"  Ort. Trade/Hafta        : {total_groups/n_weeks:.1f}" if n_weeks > 0 else "")
    print(f"  Maks Ust Uste Karli     : {max_consecutive_wins} hafta")
    print(f"  Maks Ust Uste Zararli   : {max_consecutive_losses} hafta")

    # Aylik ozet
    print(f"\n  AYLIK OZET")
    print(f"  {'-'*75}")
    print(f"  {'Ay':8s} | {'PnL':>10s} | {'Cekilen':>10s} | {'Hafta':>5s} | {'Grp':>4s} | {'Karli':>5s} | {'Zararli':>7s}")
    print(f"  {'-'*75}")

    monthly = {}
    for wr in weekly_results:
        parts = wr["date_range"].split(" - ")
        # "02 Jun" -> "Jun"
        tokens = parts[0].strip().split(" ")
        month_key = tokens[1] if len(tokens) > 1 else "?"

        # Yil bilgisi: sag taraftan
        right_tokens = parts[1].strip().split(" ") if len(parts) > 1 else []
        year_key = right_tokens[-1] if right_tokens else ""
        full_key = f"{month_key} {year_key}"

        if full_key not in monthly:
            monthly[full_key] = {"pnl": 0, "withdrawn": 0, "weeks": 0, "groups": 0, "wins": 0, "losses": 0}
        monthly[full_key]["pnl"] += wr["pnl"]
        monthly[full_key]["withdrawn"] += wr["withdrawn"]
        monthly[full_key]["weeks"] += 1
        monthly[full_key]["groups"] += wr["trade_groups"]
        if wr["pnl"] > 0:
            monthly[full_key]["wins"] += 1
        elif wr["pnl"] < 0:
            monthly[full_key]["losses"] += 1

    total_monthly_pnl = 0
    total_monthly_withdrawn = 0
    for mk, data in monthly.items():
        sign = "+" if data["pnl"] >= 0 else ""
        total_monthly_pnl += data["pnl"]
        total_monthly_withdrawn += data["withdrawn"]
        print(f"  {mk:8s} | {sign}${data['pnl']:>9,.2f} | ${data['withdrawn']:>9,.2f} | "
              f"{data['weeks']:>5d} | {data['groups']:>4d} | {data['wins']:>5d} | {data['losses']:>7d}")

    print(f"  {'-'*75}")
    tsign = "+" if total_monthly_pnl >= 0 else ""
    print(f"  {'TOPLAM':8s} | {tsign}${total_monthly_pnl:>9,.2f} | ${total_monthly_withdrawn:>9,.2f}")


if __name__ == "__main__":
    run()

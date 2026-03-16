"""Adaptive backtest — sabit PMax vs adaptif PMax karsilastirmasi.

Scalper Bot'un stratejisini birebir uygular ama PMax'i adaptif yapar.
Scalper Bot koduna DOKUNMAZ — sadece import eder.

Kullanim:
    python adaptive_backtest.py BTCUSDT
    python adaptive_backtest.py BTCUSDT --days 90 --mode all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.pmax_kc.config import (
    SCALPER_BOT_PATH, INITIAL_BALANCE, LEVERAGE, MARGIN_PER_TRADE,
    MAKER_FEE, TAKER_FEE, DEFAULT_PARAMS, TRAIN_RATIO,
)
LOOKBACK_DAYS = 180  # inline default
from data.downloader_klines import fetch_klines, split_data
from strategies.pmax_kc.adaptive_pmax import (
    adaptive_pmax_normalized,
    adaptive_pmax_preset,
    adaptive_pmax_continuous,
    adaptive_keltner,
)

if SCALPER_BOT_PATH not in sys.path:
    sys.path.append(SCALPER_BOT_PATH)
from core.strategy.indicators import (
    pmax as original_pmax, atr_rma, rsi, ema, keltner_channel, atr, variant,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("adaptive_backtest")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _get_source(df: pd.DataFrame, source: str) -> pd.Series:
    src = source.lower()
    if src == "hl2":
        return (df["high"] + df["low"]) / 2
    elif src == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3
    elif src == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    return df["close"]


def run_backtest_with_pmax(
    df: pd.DataFrame,
    pmax_line: np.ndarray,
    mavg_arr: np.ndarray,
    direction_arr: np.ndarray,
    params: dict,
    label: str = "fixed",
) -> dict:
    """Run backtest with pre-computed PMax arrays.

    Bu fonksiyon PMax'in NASIL hesaplandigi ile ilgilenmez —
    sadece verilen pmax/mavg/direction ile trade yapar.
    Boylece sabit ve adaptif PMax'i ayni engine'de test edebiliriz.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    closes = close.values
    highs = high.values
    lows = low.values
    times = df["open_time"].values
    n = len(closes)

    kc_length = params.get("kc_length", 20)
    kc_multiplier = params.get("kc_multiplier", 1.5)
    kc_atr_period = params.get("kc_atr_period", 10)
    ema_filter_period = params.get("ema_filter_period", 144)
    rsi_overbought = params.get("rsi_overbought", 65)
    max_dca_steps = params.get("max_dca_steps", 2)
    tp_close_pct = params.get("tp_close_percent", 0.20)
    hard_stop_pct = params.get("hard_stop_pct", 2.5)  # fallback sabit %
    hs_atr_mult = params.get("hs_atr_mult", 0)       # 0 = sabit %, >0 = ATR-based
    hs_atr_period = params.get("hs_atr_period", 14)
    total_fee_rate = MAKER_FEE + TAKER_FEE

    # Pre-compute filters & keltner
    rsi_vals = rsi(close, 28).values
    ema_filter = ema(close, ema_filter_period).values
    rsi_ema_vals = ema(pd.Series(rsi_vals), 10).values
    atr_vol = atr(high, low, close, 50).values
    hs_atr_arr = atr(high, low, close, hs_atr_period).values if hs_atr_mult > 0 else None
    kc_mid, kc_upper, kc_lower = keltner_channel(
        high, low, close, kc_length=kc_length,
        kc_multiplier=kc_multiplier, atr_period=kc_atr_period,
    )
    kc_upper_arr = kc_upper.values
    kc_lower_arr = kc_lower.values

    # Simulation state
    condition = 0.0
    avg_entry_price = 0.0
    entry_idx = 0
    total_notional = 0.0
    dca_fills = 0

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    tp_count = 0
    rev_count = 0
    hard_stop_count = 0
    fake_tp_blocked = 0

    MIN_BARS = 200

    def apply_filters(idx, side):
        c = closes[idx]
        if not np.isnan(ema_filter[idx]):
            if side == "LONG" and c < ema_filter[idx]:
                return False
            if side == "SHORT" and c > ema_filter[idx]:
                return False
        r = rsi_vals[idx] if not np.isnan(rsi_vals[idx]) else 50.0
        r_ema = rsi_ema_vals[idx] if not np.isnan(rsi_ema_vals[idx]) else 50.0
        rsi_os = 100 - rsi_overbought
        if side == "LONG" and r > rsi_overbought and r > r_ema:
            return False
        if side == "SHORT" and r < rsi_os and r < r_ema:
            return False
        if idx >= 200:
            lookback = atr_vol[max(0, idx - 200):idx + 1]
            valid = lookback[~np.isnan(lookback)]
            if len(valid) > 0:
                threshold = np.percentile(valid, 20)
                if not np.isnan(atr_vol[idx]) and atr_vol[idx] < threshold:
                    return False
        return True

    def close_trade(exit_idx, exit_price, reason):
        nonlocal balance, total_pnl, total_fees, condition, total_notional
        nonlocal trade_count, win_count, loss_count, tp_count, rev_count

        if total_notional <= 0:
            return

        side = "LONG" if condition > 0 else "SHORT"
        if side == "LONG":
            pnl_pct = (exit_price - avg_entry_price) / avg_entry_price * 100
        else:
            pnl_pct = (avg_entry_price - exit_price) / avg_entry_price * 100

        pnl = total_notional * pnl_pct / 100
        fee = total_notional * TAKER_FEE
        balance += pnl - fee
        total_pnl += pnl
        total_fees += fee
        trade_count += 1
        if pnl > 0:
            win_count += 1
        else:
            loss_count += 1
        if reason == "REVERSAL":
            rev_count += 1

    # Main loop
    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            continue

        # PMax crossover
        if i > 0 and not np.isnan(mavg_arr[i - 1]) and not np.isnan(pmax_line[i - 1]):
            prev_m = mavg_arr[i - 1]
            prev_p = pmax_line[i - 1]
            curr_m = mavg_arr[i]
            curr_p = pmax_line[i]

            buy_cross = prev_m <= prev_p and curr_m > curr_p
            sell_cross = prev_m >= prev_p and curr_m < curr_p

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                if apply_filters(i, "LONG") and balance >= MARGIN_PER_TRADE:
                    condition = 1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = MARGIN_PER_TRADE * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                if apply_filters(i, "SHORT") and balance >= MARGIN_PER_TRADE:
                    condition = -1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = MARGIN_PER_TRADE * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

        # Keltner DCA/TP
        if condition != 0 and total_notional > 0:
            kc_u = kc_upper_arr[i]
            kc_l = kc_lower_arr[i]
            if np.isnan(kc_u) or np.isnan(kc_l):
                continue

            if condition > 0:  # LONG
                # Hard Stop: DCA full + fiyat ATR veya sabit % kadar uzaklasti
                if dca_fills >= max_dca_steps and avg_entry_price > 0:
                    if hs_atr_mult > 0 and hs_atr_arr is not None and not np.isnan(hs_atr_arr[i]):
                        hs_distance = hs_atr_mult * hs_atr_arr[i]
                        triggered = closes[i] <= avg_entry_price - hs_distance
                    else:
                        loss_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100
                        triggered = loss_pct >= hard_stop_pct
                    if triggered:
                        close_trade(i, closes[i], "HARD_STOP")
                        hard_stop_count += 1
                        condition = 0.0
                        total_notional = 0.0
                        continue

                if dca_fills < max_dca_steps and lows[i] <= kc_l:
                    step = MARGIN_PER_TRADE * LEVERAGE
                    old = total_notional
                    total_notional += step
                    avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional
                    dca_fills += 1
                    fee = step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and highs[i] >= kc_u:
                    # BREAKEVEN FILTRESI: TP sadece kar'da yapilir
                    breakeven_price = avg_entry_price * (1 + total_fee_rate)
                    if kc_u > breakeven_price:
                        closed = total_notional * tp_close_pct
                        pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100
                        pnl = closed * pnl_pct / 100
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1
                        tp_count += 1
                        if pnl > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        if total_notional < 1.0:
                            condition = 0.0
                            total_notional = 0.0
                    else:
                        fake_tp_blocked += 1
            else:  # SHORT
                # Hard Stop: DCA full + fiyat ATR veya sabit % kadar uzaklasti
                if dca_fills >= max_dca_steps and avg_entry_price > 0:
                    if hs_atr_mult > 0 and hs_atr_arr is not None and not np.isnan(hs_atr_arr[i]):
                        hs_distance = hs_atr_mult * hs_atr_arr[i]
                        triggered = closes[i] >= avg_entry_price + hs_distance
                    else:
                        loss_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100
                        triggered = loss_pct >= hard_stop_pct
                    if triggered:
                        close_trade(i, closes[i], "HARD_STOP")
                        hard_stop_count += 1
                        condition = 0.0
                        total_notional = 0.0
                        continue

                if dca_fills < max_dca_steps and highs[i] >= kc_u:
                    step = MARGIN_PER_TRADE * LEVERAGE
                    old = total_notional
                    total_notional += step
                    avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional
                    dca_fills += 1
                    fee = step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and lows[i] <= kc_l:
                    # BREAKEVEN FILTRESI: TP sadece kar'da yapilir
                    breakeven_price = avg_entry_price * (1 - total_fee_rate)
                    if kc_l < breakeven_price:
                        closed = total_notional * tp_close_pct
                        pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100
                        pnl = closed * pnl_pct / 100
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1
                        tp_count += 1
                        if pnl > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        if total_notional < 1.0:
                            condition = 0.0
                            total_notional = 0.0
                    else:
                        fake_tp_blocked += 1

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

    # Close open position
    if condition != 0 and total_notional > 0:
        close_trade(n - 1, closes[n - 1], "END")

    net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    pf = (sum(1 for _ in range(win_count)) * 1.0) if loss_count == 0 else 0  # placeholder
    # Recalculate PF properly from actual PnL
    wr = win_count / trade_count * 100 if trade_count > 0 else 0

    return {
        "label": label,
        "net_pct": round(net_pct, 4),
        "balance": round(balance, 2),
        "total_trades": trade_count,
        "win_rate": round(wr, 2),
        "max_drawdown": round(max_dd, 4),
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "tp_count": tp_count,
        "rev_count": rev_count,
        "hard_stop_count": hard_stop_count,
        "fake_tp_blocked": fake_tp_blocked,
    }


def run_backtest_kelly(
    df: pd.DataFrame,
    pmax_line: np.ndarray,
    mavg_arr: np.ndarray,
    direction_arr: np.ndarray,
    params: dict,
    label: str = "kelly",
    kelly_fraction: float = 0.5,    # 0.5 = half-kelly (muhafazakar)
    kelly_lookback: int = 30,       # son N trade'e bakarak kelly hesapla
    min_margin_pct: float = 2.0,    # bakiyenin min %2'si
    max_margin_pct: float = 25.0,   # bakiyenin max %25'i
    min_trades_for_kelly: int = 10, # kelly hesaplamak icin min trade sayisi
) -> dict:
    """Backtest with Kelly Criterion position sizing.

    Kelly % = W - (1-W)/R
    W = win rate (son kelly_lookback trade)
    R = avg_win / avg_loss

    Pozisyon = balance * kelly% * kelly_fraction
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    closes = close.values
    highs = high.values
    lows = low.values
    times = df["open_time"].values
    n = len(closes)

    kc_length = params.get("kc_length", 5)
    kc_multiplier = params.get("kc_multiplier", 0.7)
    kc_atr_period = params.get("kc_atr_period", 26)
    ema_filter_period = params.get("ema_filter_period", 144)
    rsi_overbought = params.get("rsi_overbought", 65)
    max_dca_steps = params.get("max_dca_steps", 5)
    tp_close_pct = params.get("tp_close_percent", 0.50)

    rsi_vals = rsi(close, 28).values
    ema_filter = ema(close, ema_filter_period).values
    rsi_ema_vals = ema(pd.Series(rsi_vals), 10).values
    atr_vol = atr(high, low, close, 50).values
    kc_mid, kc_upper, kc_lower = keltner_channel(
        high, low, close, kc_length=kc_length,
        kc_multiplier=kc_multiplier, atr_period=kc_atr_period,
    )
    kc_upper_arr = kc_upper.values
    kc_lower_arr = kc_lower.values

    condition = 0.0
    avg_entry_price = 0.0
    entry_idx = 0
    total_notional = 0.0
    dca_fills = 0
    current_margin = 0.0

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    tp_count = 0
    rev_count = 0

    # Trade history for rolling kelly calculation
    trade_pnls: list[float] = []

    MIN_BARS = 200

    def calc_kelly():
        """Calculate Kelly % from recent trade history."""
        if len(trade_pnls) < min_trades_for_kelly:
            return min_margin_pct / 100.0  # default conservative

        recent = trade_pnls[-kelly_lookback:] if len(trade_pnls) >= kelly_lookback else trade_pnls
        wins = [p for p in recent if p > 0]
        losses = [p for p in recent if p <= 0]

        if not wins or not losses:
            return min_margin_pct / 100.0

        w = len(wins) / len(recent)  # win rate
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss == 0:
            return max_margin_pct / 100.0

        r = avg_win / avg_loss  # reward/risk ratio

        kelly_pct = w - (1 - w) / r

        # Apply fraction (half-kelly etc)
        kelly_pct *= kelly_fraction

        # Clamp
        kelly_pct = max(min_margin_pct / 100.0, min(max_margin_pct / 100.0, kelly_pct))

        return kelly_pct

    def calc_margin():
        kelly_pct = calc_kelly()
        margin = balance * kelly_pct
        margin = max(100.0, min(margin, balance * 0.5))  # asla bakiyenin yarisinan fazla
        return margin

    def apply_filters(idx, side):
        c = closes[idx]
        if not np.isnan(ema_filter[idx]):
            if side == "LONG" and c < ema_filter[idx]:
                return False
            if side == "SHORT" and c > ema_filter[idx]:
                return False
        r = rsi_vals[idx] if not np.isnan(rsi_vals[idx]) else 50.0
        r_ema = rsi_ema_vals[idx] if not np.isnan(rsi_ema_vals[idx]) else 50.0
        rsi_os = 100 - rsi_overbought
        if side == "LONG" and r > rsi_overbought and r > r_ema:
            return False
        if side == "SHORT" and r < rsi_os and r < r_ema:
            return False
        if idx >= 200:
            lookback_w = atr_vol[max(0, idx - 200):idx + 1]
            valid = lookback_w[~np.isnan(lookback_w)]
            if len(valid) > 0:
                threshold = np.percentile(valid, 20)
                if not np.isnan(atr_vol[idx]) and atr_vol[idx] < threshold:
                    return False
        return True

    def close_trade(exit_idx, exit_price, reason):
        nonlocal balance, total_pnl, total_fees, condition, total_notional
        nonlocal trade_count, win_count, loss_count, tp_count, rev_count
        if total_notional <= 0:
            return
        side = "LONG" if condition > 0 else "SHORT"
        if side == "LONG":
            pnl_pct = (exit_price - avg_entry_price) / avg_entry_price * 100
        else:
            pnl_pct = (avg_entry_price - exit_price) / avg_entry_price * 100
        pnl = total_notional * pnl_pct / 100
        fee = total_notional * TAKER_FEE
        balance += pnl - fee
        total_pnl += pnl
        total_fees += fee
        trade_count += 1
        trade_pnls.append(pnl - fee)  # net pnl for kelly
        if pnl > 0:
            win_count += 1
        else:
            loss_count += 1
        if reason == "REVERSAL":
            rev_count += 1

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            continue

        if i > 0 and not np.isnan(mavg_arr[i - 1]) and not np.isnan(pmax_line[i - 1]):
            prev_m = mavg_arr[i - 1]
            prev_p = pmax_line[i - 1]
            curr_m = mavg_arr[i]
            curr_p = pmax_line[i]
            buy_cross = prev_m <= prev_p and curr_m > curr_p
            sell_cross = prev_m >= prev_p and curr_m < curr_p

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                current_margin = calc_margin()
                if apply_filters(i, "LONG") and balance >= current_margin:
                    condition = 1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = current_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                current_margin = calc_margin()
                if apply_filters(i, "SHORT") and balance >= current_margin:
                    condition = -1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = current_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

        # Keltner DCA/TP with kelly-sized DCA
        if condition != 0 and total_notional > 0:
            kc_u = kc_upper_arr[i]
            kc_l = kc_lower_arr[i]
            if np.isnan(kc_u) or np.isnan(kc_l):
                continue

            dca_margin = calc_margin()
            dca_step = dca_margin * LEVERAGE

            if condition > 0:
                if dca_fills < max_dca_steps and lows[i] <= kc_l:
                    old = total_notional
                    total_notional += dca_step
                    avg_entry_price = (avg_entry_price * old + kc_l * dca_step) / total_notional
                    dca_fills += 1
                    fee = dca_step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and highs[i] >= kc_u:
                    closed = total_notional * tp_close_pct
                    pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    trade_pnls.append(pnl - fee)
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    trade_count += 1
                    tp_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    if total_notional < 1.0:
                        condition = 0.0
                        total_notional = 0.0
            else:
                if dca_fills < max_dca_steps and highs[i] >= kc_u:
                    old = total_notional
                    total_notional += dca_step
                    avg_entry_price = (avg_entry_price * old + kc_u * dca_step) / total_notional
                    dca_fills += 1
                    fee = dca_step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and lows[i] <= kc_l:
                    closed = total_notional * tp_close_pct
                    pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    trade_pnls.append(pnl - fee)
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    trade_count += 1
                    tp_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    if total_notional < 1.0:
                        condition = 0.0
                        total_notional = 0.0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

        if balance <= 0:
            balance = 0
            break

    if condition != 0 and total_notional > 0:
        close_trade(n - 1, closes[n - 1], "END")

    net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    wr = win_count / trade_count * 100 if trade_count > 0 else 0

    final_kelly = calc_kelly()

    return {
        "label": label,
        "net_pct": round(net_pct, 4),
        "balance": round(balance, 2),
        "total_trades": trade_count,
        "win_rate": round(wr, 2),
        "max_drawdown": round(max_dd, 4),
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "tp_count": tp_count,
        "rev_count": rev_count,
        "peak_balance": round(peak_balance, 2),
        "final_kelly_pct": round(final_kelly * 100, 2),
    }


def run_backtest_adaptive_kc(
    df: pd.DataFrame,
    pmax_line: np.ndarray,
    mavg_arr: np.ndarray,
    direction_arr: np.ndarray,
    params: dict,
    label: str = "adaptive_kc",
    kc_adaptive_params: dict | None = None,
) -> dict:
    """Backtest with adaptive Keltner Channel - KC params change with market."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    closes = close.values
    highs = high.values
    lows = low.values
    times = df["open_time"].values
    n = len(closes)

    ema_filter_period = params.get("ema_filter_period", 144)
    rsi_overbought = params.get("rsi_overbought", 65)

    rsi_vals = rsi(close, 28).values
    ema_filter = ema(close, ema_filter_period).values
    rsi_ema_vals = ema(pd.Series(rsi_vals), 10).values
    atr_vol = atr(high, low, close, 50).values

    # Compute adaptive KC
    akp = kc_adaptive_params or {}
    kc_upper_arr, kc_lower_arr, kc_mid_arr, kc_mult_arr, max_dca_arr, tp_pct_arr = adaptive_keltner(
        high, low, close, direction_arr,
        base_kc_length=akp.get("base_kc_length", 5),
        base_kc_mult=akp.get("base_kc_mult", 0.7),
        base_kc_atr=akp.get("base_kc_atr", 26),
        base_max_dca=akp.get("base_max_dca", 5),
        base_tp_pct=akp.get("base_tp_pct", 0.50),
        kc_vol_lookback=akp.get("kc_vol_lookback", 80),
        kc_update_interval=akp.get("kc_update_interval", 40),
        kc_mult_base=akp.get("kc_mult_base", 0.3),
        kc_mult_scale=akp.get("kc_mult_scale", 0.4),
        kc_dca_base=akp.get("kc_dca_base", 3),
        kc_dca_trend_bonus=akp.get("kc_dca_trend_bonus", 2),
        kc_tp_base=akp.get("kc_tp_base", 0.30),
        kc_tp_scale=akp.get("kc_tp_scale", 0.20),
    )

    condition = 0.0
    avg_entry_price = 0.0
    entry_idx = 0
    total_notional = 0.0
    dca_fills = 0

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    tp_count = 0
    rev_count = 0

    MIN_BARS = 200

    def apply_filters(idx, side):
        c = closes[idx]
        if not np.isnan(ema_filter[idx]):
            if side == "LONG" and c < ema_filter[idx]:
                return False
            if side == "SHORT" and c > ema_filter[idx]:
                return False
        r = rsi_vals[idx] if not np.isnan(rsi_vals[idx]) else 50.0
        r_ema = rsi_ema_vals[idx] if not np.isnan(rsi_ema_vals[idx]) else 50.0
        rsi_os = 100 - rsi_overbought
        if side == "LONG" and r > rsi_overbought and r > r_ema:
            return False
        if side == "SHORT" and r < rsi_os and r < r_ema:
            return False
        if idx >= 200:
            lookback_w = atr_vol[max(0, idx - 200):idx + 1]
            valid = lookback_w[~np.isnan(lookback_w)]
            if len(valid) > 0:
                threshold = np.percentile(valid, 20)
                if not np.isnan(atr_vol[idx]) and atr_vol[idx] < threshold:
                    return False
        return True

    def close_trade(exit_idx, exit_price, reason):
        nonlocal balance, total_pnl, total_fees, condition, total_notional
        nonlocal trade_count, win_count, loss_count, tp_count, rev_count
        if total_notional <= 0:
            return
        side = "LONG" if condition > 0 else "SHORT"
        if side == "LONG":
            pnl_pct = (exit_price - avg_entry_price) / avg_entry_price * 100
        else:
            pnl_pct = (avg_entry_price - exit_price) / avg_entry_price * 100
        pnl = total_notional * pnl_pct / 100
        fee = total_notional * TAKER_FEE
        balance += pnl - fee
        total_pnl += pnl
        total_fees += fee
        trade_count += 1
        if pnl > 0:
            win_count += 1
        else:
            loss_count += 1
        if reason == "REVERSAL":
            rev_count += 1

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            continue

        # PMax crossover
        if i > 0 and not np.isnan(mavg_arr[i - 1]) and not np.isnan(pmax_line[i - 1]):
            prev_m = mavg_arr[i - 1]
            prev_p = pmax_line[i - 1]
            curr_m = mavg_arr[i]
            curr_p = pmax_line[i]
            buy_cross = prev_m <= prev_p and curr_m > curr_p
            sell_cross = prev_m >= prev_p and curr_m < curr_p

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                if apply_filters(i, "LONG") and balance >= MARGIN_PER_TRADE:
                    condition = 1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = MARGIN_PER_TRADE * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                if apply_filters(i, "SHORT") and balance >= MARGIN_PER_TRADE:
                    condition = -1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = MARGIN_PER_TRADE * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

        # Adaptive Keltner DCA/TP
        if condition != 0 and total_notional > 0:
            kc_u = kc_upper_arr[i]
            kc_l = kc_lower_arr[i]
            if np.isnan(kc_u) or np.isnan(kc_l):
                continue

            # Read adaptive DCA/TP values for this bar
            current_max_dca = int(max_dca_arr[i])
            current_tp_pct = tp_pct_arr[i]

            if condition > 0:  # LONG
                if dca_fills < current_max_dca and lows[i] <= kc_l:
                    step = MARGIN_PER_TRADE * LEVERAGE
                    old = total_notional
                    total_notional += step
                    avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional
                    dca_fills += 1
                    fee = step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and highs[i] >= kc_u:
                    closed = total_notional * current_tp_pct
                    pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    trade_count += 1
                    tp_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    if total_notional < 1.0:
                        condition = 0.0
                        total_notional = 0.0
            else:  # SHORT
                if dca_fills < current_max_dca and highs[i] >= kc_u:
                    step = MARGIN_PER_TRADE * LEVERAGE
                    old = total_notional
                    total_notional += step
                    avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional
                    dca_fills += 1
                    fee = step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and lows[i] <= kc_l:
                    closed = total_notional * current_tp_pct
                    pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    trade_count += 1
                    tp_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    if total_notional < 1.0:
                        condition = 0.0
                        total_notional = 0.0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

    if condition != 0 and total_notional > 0:
        close_trade(n - 1, closes[n - 1], "END")

    net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    wr = win_count / trade_count * 100 if trade_count > 0 else 0

    return {
        "label": label,
        "net_pct": round(net_pct, 4),
        "balance": round(balance, 2),
        "total_trades": trade_count,
        "win_rate": round(wr, 2),
        "max_drawdown": round(max_dd, 4),
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "tp_count": tp_count,
        "rev_count": rev_count,
    }


def run_backtest_compounding(
    df: pd.DataFrame,
    pmax_line: np.ndarray,
    mavg_arr: np.ndarray,
    direction_arr: np.ndarray,
    params: dict,
    label: str = "compound",
    # Compounding params
    margin_pct: float = 10.0,       # bakiyenin %X'i margin olarak kullan
    min_margin: float = 100.0,      # minimum margin
    max_margin_pct: float = 30.0,   # bakiyenin max %X'i tek pozisyonda
    # Confidence sizing
    use_confidence: bool = True,
    conf_min_mult: float = 0.5,     # dusuk guven = 0.5x margin
    conf_max_mult: float = 2.0,     # yuksek guven = 2.0x margin
) -> dict:
    """Backtest with compounding + dynamic position sizing.

    Compounding: margin = balance * margin_pct / 100
    Confidence: vol_ratio dusuk + trend_dist yuksek = yuksek guven = buyuk pozisyon
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    closes = close.values
    highs = high.values
    lows = low.values
    times = df["open_time"].values
    n = len(closes)

    kc_length = params.get("kc_length", 20)
    kc_multiplier = params.get("kc_multiplier", 1.5)
    kc_atr_period = params.get("kc_atr_period", 10)
    ema_filter_period = params.get("ema_filter_period", 144)
    rsi_overbought = params.get("rsi_overbought", 65)
    max_dca_steps = params.get("max_dca_steps", 2)
    tp_close_pct = params.get("tp_close_percent", 0.20)

    rsi_vals = rsi(close, 28).values
    ema_filter = ema(close, ema_filter_period).values
    rsi_ema_vals = ema(pd.Series(rsi_vals), 10).values
    atr_vol = atr(high, low, close, 50).values
    base_atr = atr(high, low, close, 10).values
    base_mavg_arr = ema(close, 10).values
    kc_mid, kc_upper, kc_lower = keltner_channel(
        high, low, close, kc_length=kc_length,
        kc_multiplier=kc_multiplier, atr_period=kc_atr_period,
    )
    kc_upper_arr = kc_upper.values
    kc_lower_arr = kc_lower.values

    condition = 0.0
    avg_entry_price = 0.0
    entry_idx = 0
    total_notional = 0.0
    dca_fills = 0
    current_margin = 0.0  # bu pozisyon icin kullanilan margin

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    tp_count = 0
    rev_count = 0

    MIN_BARS = 200

    def calc_confidence(idx):
        """Sinyal guven seviyesi: 0.0 - 1.0"""
        if idx < 480:
            return 0.5
        # Vol ratio (dusuk vol = daha guvenli sinyal)
        atr_window = base_atr[max(0, idx - 480):idx + 1]
        valid = atr_window[~np.isnan(atr_window)]
        if len(valid) < 10:
            return 0.5
        median = np.median(valid)
        curr = base_atr[idx] if not np.isnan(base_atr[idx]) else median
        vol_ratio = curr / median if median > 0 else 1.0

        # Trend dist (yuksek = guclu sinyal)
        mavg_v = base_mavg_arr[idx] if not np.isnan(base_mavg_arr[idx]) else closes[idx]
        atr_v = curr if curr > 0 else 1.0
        trend_dist = abs(closes[idx] - mavg_v) / atr_v

        # Dusuk vol + guclu trend = yuksek guven
        vol_score = max(0, min(1, 1.5 - vol_ratio))  # vol dusukse skor yuksek
        trend_score = max(0, min(1, trend_dist / 2.0))  # trend gucluyse skor yuksek
        confidence = vol_score * 0.6 + trend_score * 0.4
        return max(0.0, min(1.0, confidence))

    def calc_margin(idx):
        """Compounding + confidence-based margin."""
        base_margin = balance * margin_pct / 100.0
        base_margin = max(min_margin, base_margin)
        max_margin = balance * max_margin_pct / 100.0
        base_margin = min(base_margin, max_margin)

        if use_confidence:
            conf = calc_confidence(idx)
            mult = conf_min_mult + conf * (conf_max_mult - conf_min_mult)
            base_margin *= mult

        return max(min_margin, min(base_margin, balance * 0.5))

    def apply_filters(idx, side):
        c = closes[idx]
        if not np.isnan(ema_filter[idx]):
            if side == "LONG" and c < ema_filter[idx]:
                return False
            if side == "SHORT" and c > ema_filter[idx]:
                return False
        r = rsi_vals[idx] if not np.isnan(rsi_vals[idx]) else 50.0
        r_ema = rsi_ema_vals[idx] if not np.isnan(rsi_ema_vals[idx]) else 50.0
        rsi_os = 100 - rsi_overbought
        if side == "LONG" and r > rsi_overbought and r > r_ema:
            return False
        if side == "SHORT" and r < rsi_os and r < r_ema:
            return False
        if idx >= 200:
            lookback_w = atr_vol[max(0, idx - 200):idx + 1]
            valid = lookback_w[~np.isnan(lookback_w)]
            if len(valid) > 0:
                threshold = np.percentile(valid, 20)
                if not np.isnan(atr_vol[idx]) and atr_vol[idx] < threshold:
                    return False
        return True

    def close_trade(exit_idx, exit_price, reason):
        nonlocal balance, total_pnl, total_fees, condition, total_notional
        nonlocal trade_count, win_count, loss_count, tp_count, rev_count

        if total_notional <= 0:
            return

        side = "LONG" if condition > 0 else "SHORT"
        if side == "LONG":
            pnl_pct = (exit_price - avg_entry_price) / avg_entry_price * 100
        else:
            pnl_pct = (avg_entry_price - exit_price) / avg_entry_price * 100

        pnl = total_notional * pnl_pct / 100
        fee = total_notional * TAKER_FEE
        balance += pnl - fee
        total_pnl += pnl
        total_fees += fee
        trade_count += 1
        if pnl > 0:
            win_count += 1
        else:
            loss_count += 1
        if reason == "REVERSAL":
            rev_count += 1

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            continue

        if i > 0 and not np.isnan(mavg_arr[i - 1]) and not np.isnan(pmax_line[i - 1]):
            prev_m = mavg_arr[i - 1]
            prev_p = pmax_line[i - 1]
            curr_m = mavg_arr[i]
            curr_p = pmax_line[i]

            buy_cross = prev_m <= prev_p and curr_m > curr_p
            sell_cross = prev_m >= prev_p and curr_m < curr_p

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                current_margin = calc_margin(i)
                if apply_filters(i, "LONG") and balance >= current_margin:
                    condition = 1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = current_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                current_margin = calc_margin(i)
                if apply_filters(i, "SHORT") and balance >= current_margin:
                    condition = -1.0
                    avg_entry_price = closes[i]
                    entry_idx = i
                    total_notional = current_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

        # Keltner DCA/TP with breakeven filter + hard stop
        total_fee_rate = MAKER_FEE + TAKER_FEE
        if condition != 0 and total_notional > 0:
            kc_u = kc_upper_arr[i]
            kc_l = kc_lower_arr[i]
            if np.isnan(kc_u) or np.isnan(kc_l):
                continue

            dca_margin = calc_margin(i)
            dca_step = dca_margin * LEVERAGE
            hard_stop_pct = params.get("hard_stop_pct", 2.5)
            hs_atr_mult_c = params.get("hs_atr_mult", 0)
            hs_atr_period_c = params.get("hs_atr_period", 14)

            if condition > 0:
                # Hard Stop (ATR-based or fixed %)
                if dca_fills >= max_dca_steps and avg_entry_price > 0:
                    if hs_atr_mult_c > 0:
                        hs_atr_c = atr(high, low, close, hs_atr_period_c).values
                        if not np.isnan(hs_atr_c[i]):
                            hs_dist = hs_atr_mult_c * hs_atr_c[i]
                            triggered = closes[i] <= avg_entry_price - hs_dist
                        else:
                            loss_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100
                            triggered = loss_pct >= hard_stop_pct
                    else:
                        loss_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100
                        triggered = loss_pct >= hard_stop_pct
                    if triggered:
                        close_trade(i, closes[i], "HARD_STOP")
                        condition = 0.0
                        total_notional = 0.0
                        continue

                if dca_fills < max_dca_steps and lows[i] <= kc_l:
                    old = total_notional
                    total_notional += dca_step
                    avg_entry_price = (avg_entry_price * old + kc_l * dca_step) / total_notional
                    dca_fills += 1
                    fee = dca_step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and highs[i] >= kc_u:
                    breakeven_price = avg_entry_price * (1 + total_fee_rate)
                    if kc_u > breakeven_price:
                        closed = total_notional * tp_close_pct
                        pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100
                        pnl = closed * pnl_pct / 100
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1
                        tp_count += 1
                        if pnl > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        if total_notional < 1.0:
                            condition = 0.0
                            total_notional = 0.0
            else:
                # Hard Stop (ATR-based or fixed %)
                if dca_fills >= max_dca_steps and avg_entry_price > 0:
                    if hs_atr_mult_c > 0:
                        hs_atr_c = atr(high, low, close, hs_atr_period_c).values
                        if not np.isnan(hs_atr_c[i]):
                            hs_dist = hs_atr_mult_c * hs_atr_c[i]
                            triggered = closes[i] >= avg_entry_price + hs_dist
                        else:
                            loss_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100
                            triggered = loss_pct >= hard_stop_pct
                    else:
                        loss_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100
                        triggered = loss_pct >= hard_stop_pct
                    if triggered:
                        close_trade(i, closes[i], "HARD_STOP")
                        condition = 0.0
                        total_notional = 0.0
                        continue

                if dca_fills < max_dca_steps and highs[i] >= kc_u:
                    old = total_notional
                    total_notional += dca_step
                    avg_entry_price = (avg_entry_price * old + kc_u * dca_step) / total_notional
                    dca_fills += 1
                    fee = dca_step * MAKER_FEE
                    balance -= fee
                    total_fees += fee
                elif dca_fills > 0 and lows[i] <= kc_l:
                    breakeven_price = avg_entry_price * (1 - total_fee_rate)
                    if kc_l < breakeven_price:
                        closed = total_notional * tp_close_pct
                        pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100
                        pnl = closed * pnl_pct / 100
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1
                        tp_count += 1
                        if pnl > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        if total_notional < 1.0:
                            condition = 0.0
                            total_notional = 0.0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

        # Iflasa karsi koruma
        if balance <= 0:
            balance = 0
            break

    if condition != 0 and total_notional > 0:
        close_trade(n - 1, closes[n - 1], "END")

    net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    wr = win_count / trade_count * 100 if trade_count > 0 else 0

    return {
        "label": label,
        "net_pct": round(net_pct, 4),
        "balance": round(balance, 2),
        "total_trades": trade_count,
        "win_rate": round(wr, 2),
        "max_drawdown": round(max_dd, 4),
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "tp_count": tp_count,
        "rev_count": rev_count,
        "peak_balance": round(peak_balance, 2),
    }


def run_backtest_compounding_v2(
    df: pd.DataFrame,
    pmax_line: np.ndarray,
    mavg_arr: np.ndarray,
    direction_arr: np.ndarray,
    params: dict,
    label: str = "comp_v2",
    margin_pct: float = 15.0,
    max_margin_pct: float = 30.0,
    hs_atr_mult: float = 2.0,
    hs_atr_period: int = 14,
) -> dict:
    """Compounding V2: Cumulative margin fix + free KC TP (no breakeven block).

    Fixes:
    1. margin_pct = TOTAL position risk, divided by (1 + max_dca_steps)
    2. KC TP fires regardless of profit/loss (dynamic de-risking)
    3. ATR-based hard stop as safety valve
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    closes = close.values
    highs = high.values
    lows = low.values
    times = df["open_time"].values
    n = len(closes)

    kc_length = params.get("kc_length", 5)
    kc_multiplier = params.get("kc_multiplier", 0.7)
    kc_atr_period = params.get("kc_atr_period", 26)
    ema_filter_period = params.get("ema_filter_period", 144)
    rsi_overbought = params.get("rsi_overbought", 65)
    max_dca_steps = params.get("max_dca_steps", 5)
    tp_close_pct = params.get("tp_close_percent", 0.50)

    rsi_vals = rsi(close, 28).values
    ema_filter = ema(close, ema_filter_period).values
    rsi_ema_vals = ema(pd.Series(rsi_vals), 10).values
    atr_vol = atr(high, low, close, 50).values
    hs_atr_arr = atr(high, low, close, hs_atr_period).values
    kc_mid, kc_upper, kc_lower = keltner_channel(
        high, low, close, kc_length=kc_length,
        kc_multiplier=kc_multiplier, atr_period=kc_atr_period,
    )
    kc_upper_arr = kc_upper.values
    kc_lower_arr = kc_lower.values

    condition = 0.0
    avg_entry_price = 0.0
    total_notional = 0.0
    dca_fills = 0

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    tp_count = 0
    rev_count = 0
    hard_stop_count = 0
    tp_in_loss_count = 0  # zararda yapilan TP sayisi

    MIN_BARS = 200

    def calc_total_margin():
        """Total allowed margin = balance * margin_pct%."""
        m = balance * margin_pct / 100.0
        m = max(100.0, m)
        m = min(balance * max_margin_pct / 100.0, m)
        return max(100.0, min(m, balance * 0.5))

    def calc_step_margin():
        """Each step margin = total / (1 + max_dca_steps)."""
        total = calc_total_margin()
        return total / (1 + max_dca_steps)

    def apply_filters(idx, side):
        c = closes[idx]
        if not np.isnan(ema_filter[idx]):
            if side == "LONG" and c < ema_filter[idx]:
                return False
            if side == "SHORT" and c > ema_filter[idx]:
                return False
        r = rsi_vals[idx] if not np.isnan(rsi_vals[idx]) else 50.0
        r_ema = rsi_ema_vals[idx] if not np.isnan(rsi_ema_vals[idx]) else 50.0
        rsi_os = 100 - rsi_overbought
        if side == "LONG" and r > rsi_overbought and r > r_ema:
            return False
        if side == "SHORT" and r < rsi_os and r < r_ema:
            return False
        if idx >= 200:
            lookback_w = atr_vol[max(0, idx - 200):idx + 1]
            valid = lookback_w[~np.isnan(lookback_w)]
            if len(valid) > 0:
                threshold = np.percentile(valid, 20)
                if not np.isnan(atr_vol[idx]) and atr_vol[idx] < threshold:
                    return False
        return True

    def close_trade(exit_idx, exit_price, reason):
        nonlocal balance, total_pnl, total_fees, condition, total_notional
        nonlocal trade_count, win_count, loss_count, rev_count, hard_stop_count
        if total_notional <= 0:
            return
        side = "LONG" if condition > 0 else "SHORT"
        if side == "LONG":
            pnl_pct = (exit_price - avg_entry_price) / avg_entry_price * 100
        else:
            pnl_pct = (avg_entry_price - exit_price) / avg_entry_price * 100
        pnl = total_notional * pnl_pct / 100
        fee = total_notional * TAKER_FEE
        balance += pnl - fee
        total_pnl += pnl
        total_fees += fee
        trade_count += 1
        if pnl > 0:
            win_count += 1
        else:
            loss_count += 1
        if reason == "REVERSAL":
            rev_count += 1
        if reason == "HARD_STOP":
            hard_stop_count += 1

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            continue

        if i > 0 and not np.isnan(mavg_arr[i - 1]) and not np.isnan(pmax_line[i - 1]):
            prev_m, prev_p = mavg_arr[i - 1], pmax_line[i - 1]
            curr_m, curr_p = mavg_arr[i], pmax_line[i]
            buy_cross = prev_m <= prev_p and curr_m > curr_p
            sell_cross = prev_m >= prev_p and curr_m < curr_p

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                step_margin = calc_step_margin()
                if apply_filters(i, "LONG") and balance >= step_margin:
                    condition = 1.0
                    avg_entry_price = closes[i]
                    total_notional = step_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    close_trade(i, closes[i], "REVERSAL")
                step_margin = calc_step_margin()
                if apply_filters(i, "SHORT") and balance >= step_margin:
                    condition = -1.0
                    avg_entry_price = closes[i]
                    total_notional = step_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                else:
                    condition = 0.0
                    total_notional = 0.0

        # Keltner DCA/TP + ATR Hard Stop
        if condition != 0 and total_notional > 0:
            kc_u = kc_upper_arr[i]
            kc_l = kc_lower_arr[i]
            if np.isnan(kc_u) or np.isnan(kc_l):
                continue

            step_margin = calc_step_margin()
            dca_step = step_margin * LEVERAGE

            if condition > 0:  # LONG
                # ATR Hard Stop (safety valve)
                if dca_fills >= max_dca_steps and avg_entry_price > 0 and not np.isnan(hs_atr_arr[i]):
                    hs_distance = hs_atr_mult * hs_atr_arr[i]
                    if closes[i] <= avg_entry_price - hs_distance:
                        close_trade(i, closes[i], "HARD_STOP")
                        condition = 0.0
                        total_notional = 0.0
                        continue

                # DCA at KC Lower
                if dca_fills < max_dca_steps and lows[i] <= kc_l:
                    old = total_notional
                    total_notional += dca_step
                    avg_entry_price = (avg_entry_price * old + kc_l * dca_step) / total_notional
                    dca_fills += 1
                    fee = dca_step * MAKER_FEE
                    balance -= fee
                    total_fees += fee

                # TP at KC Upper — NO BREAKEVEN CHECK (free de-risking)
                elif dca_fills > 0 and highs[i] >= kc_u:
                    closed = total_notional * tp_close_pct
                    pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    trade_count += 1
                    tp_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                        tp_in_loss_count += 1
                    if total_notional < 1.0:
                        condition = 0.0
                        total_notional = 0.0

            else:  # SHORT
                # ATR Hard Stop (safety valve)
                if dca_fills >= max_dca_steps and avg_entry_price > 0 and not np.isnan(hs_atr_arr[i]):
                    hs_distance = hs_atr_mult * hs_atr_arr[i]
                    if closes[i] >= avg_entry_price + hs_distance:
                        close_trade(i, closes[i], "HARD_STOP")
                        condition = 0.0
                        total_notional = 0.0
                        continue

                # DCA at KC Upper
                if dca_fills < max_dca_steps and highs[i] >= kc_u:
                    old = total_notional
                    total_notional += dca_step
                    avg_entry_price = (avg_entry_price * old + kc_u * dca_step) / total_notional
                    dca_fills += 1
                    fee = dca_step * MAKER_FEE
                    balance -= fee
                    total_fees += fee

                # TP at KC Lower — NO BREAKEVEN CHECK (free de-risking)
                elif dca_fills > 0 and lows[i] <= kc_l:
                    closed = total_notional * tp_close_pct
                    pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    trade_count += 1
                    tp_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                        tp_in_loss_count += 1
                    if total_notional < 1.0:
                        condition = 0.0
                        total_notional = 0.0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)
        if balance <= 0:
            balance = 0
            break

    if condition != 0 and total_notional > 0:
        close_trade(n - 1, closes[n - 1], "END")

    net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    wr = win_count / trade_count * 100 if trade_count > 0 else 0

    return {
        "label": label,
        "net_pct": round(net_pct, 4),
        "balance": round(balance, 2),
        "total_trades": trade_count,
        "win_rate": round(wr, 2),
        "max_drawdown": round(max_dd, 4),
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "tp_count": tp_count,
        "rev_count": rev_count,
        "hard_stop_count": hard_stop_count,
        "tp_in_loss": tp_in_loss_count,
        "peak_balance": round(peak_balance, 2),
    }


def compare_modes(symbol: str, days: int = 180, continuous_params: dict | None = None):
    """Run all 4 modes and compare results."""
    logger.info("=" * 70)
    logger.info("ADAPTIVE PMAX KARSILASTIRMA | %s | %d gun", symbol, days)
    logger.info("=" * 70)

    # Fetch data
    df = fetch_klines(symbol, "3m", days)
    if len(df) < 500:
        logger.error("Yetersiz veri: %d mum", len(df))
        return

    df_is, df_oos = split_data(df, TRAIN_RATIO)
    logger.info("IS: %d mum | OOS: %d mum", len(df_is), len(df_oos))

    params = DEFAULT_PARAMS.copy()
    source = params.get("source", "hl2")
    ma_type = params.get("ma_type", "EMA")
    atr_period = params.get("atr_period", 10)
    atr_multiplier = params.get("atr_multiplier", 3.0)
    ma_length = params.get("ma_length", 10)

    results = []

    for label, data in [("IS", df_is), ("OOS", df_oos)]:
        logger.info("\n--- %s ---", label)
        high = data["high"]
        low = data["low"]
        close = data["close"]
        src = _get_source(data, source)

        # 1. FIXED (orijinal PMax)
        pmax_line, mavg, direction = original_pmax(
            src, high, low, close,
            atr_period=atr_period, atr_multiplier=atr_multiplier,
            ma_type=ma_type, ma_length=ma_length,
            change_atr=True, normalize_atr=False,
        )
        r = run_backtest_with_pmax(data, pmax_line.values, mavg.values, direction.values, params, "fixed")
        logger.info("FIXED:      Net=%+.2f%% WR=%.1f%% DD=%.2f%% Trades=%d (TP:%d REV:%d)",
                     r["net_pct"], r["win_rate"], r["max_drawdown"], r["total_trades"], r["tp_count"], r["rev_count"])
        r["split"] = label
        results.append(r)

        # 2. ATR-Normalized
        pmax_n, mavg_n, dir_n, mult_n = adaptive_pmax_normalized(
            src, high, low, close,
            base_atr_period=atr_period, base_atr_multiplier=atr_multiplier,
            ma_type=ma_type, ma_length=ma_length,
        )
        r = run_backtest_with_pmax(data, pmax_n.values, mavg_n.values, dir_n.values, params, "normalized")
        logger.info("NORMALIZED: Net=%+.2f%% WR=%.1f%% DD=%.2f%% Trades=%d (TP:%d REV:%d)",
                     r["net_pct"], r["win_rate"], r["max_drawdown"], r["total_trades"], r["tp_count"], r["rev_count"])
        r["split"] = label
        results.append(r)

        # 3. Preset
        pmax_p, mavg_p, dir_p, regime_p = adaptive_pmax_preset(
            src, high, low, close,
            ma_type=ma_type, base_atr_period=atr_period,
        )
        r = run_backtest_with_pmax(data, pmax_p.values, mavg_p.values, dir_p.values, params, "preset")
        logger.info("PRESET:     Net=%+.2f%% WR=%.1f%% DD=%.2f%% Trades=%d (TP:%d REV:%d)",
                     r["net_pct"], r["win_rate"], r["max_drawdown"], r["total_trades"], r["tp_count"], r["rev_count"])
        r["split"] = label
        results.append(r)

        # 4. Continuous
        cp = continuous_params or {}
        pmax_c, mavg_c, dir_c, mult_c, ml_c, ap_c = adaptive_pmax_continuous(
            src, high, low, close,
            ma_type=ma_type, base_atr_period=atr_period,
            base_atr_multiplier=atr_multiplier, base_ma_length=ma_length,
            lookback=cp.get("vol_lookback", 480),
            flip_window=cp.get("flip_window", 120),
            mult_base=cp.get("mult_base", 2.0),
            mult_scale=cp.get("mult_scale", 1.0),
            ma_base=cp.get("ma_base", 8),
            ma_scale=cp.get("ma_scale", 3.0),
            atr_base=cp.get("atr_base", 8),
            atr_scale=cp.get("atr_scale", 1.5),
            update_interval=cp.get("update_interval", 1),
        )
        r = run_backtest_with_pmax(data, pmax_c.values, mavg_c.values, dir_c.values, params, "continuous")
        logger.info("CONTINUOUS: Net=%+.2f%% WR=%.1f%% DD=%.2f%% Trades=%d (TP:%d REV:%d)",
                     r["net_pct"], r["win_rate"], r["max_drawdown"], r["total_trades"], r["tp_count"], r["rev_count"])
        r["split"] = label
        results.append(r)

        # 5. Compounding (continuous PMax + compound sizing)
        r = run_backtest_compounding(data, pmax_c.values, mavg_c.values, dir_c.values, params, "compound")
        logger.info("COMPOUND:   Net=%+.2f%% WR=%.1f%% DD=%.2f%% Trades=%d Balance=$%.0f Peak=$%.0f",
                     r["net_pct"], r["win_rate"], r["max_drawdown"], r["total_trades"], r["balance"], r.get("peak_balance", 0))
        r["split"] = label
        results.append(r)

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("SONUC TABLOSU")
    logger.info("=" * 70)
    logger.info("%-12s %-5s %+8s %7s %7s %6s %5s", "Mode", "Split", "Net%", "WR%", "DD%", "Trades", "TP")
    logger.info("-" * 60)
    for r in results:
        logger.info("%-12s %-5s %+8.2f %6.1f%% %6.2f%% %6d %5d",
                     r["label"], r["split"], r["net_pct"], r["win_rate"],
                     r["max_drawdown"], r["total_trades"], r["tp_count"])

    # Save
    out_path = RESULTS_DIR / f"{symbol}_adaptive_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"symbol": symbol, "days": days, "results": results}, f, indent=2)
    logger.info("\nSonuclar kaydedildi: %s", out_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Adaptive PMax Backtest Karsilastirma")
    parser.add_argument("symbol", help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Lookback days")
    args = parser.parse_args()

    compare_modes(args.symbol, days=args.days)


if __name__ == "__main__":
    main()

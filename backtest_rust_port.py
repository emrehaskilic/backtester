"""
Rust backtest.rs birebir Python portu — LOOK-AHEAD DUZELTILMIS.

Tek fark: kc_upper[i-1] ve kc_lower[i-1] kullanilir (gercekci).
Rust'taki kc_upper[i] / kc_lower[i] look-ahead bias icerir.

Bu dosya dogrudan numpy array'ler uzerinde calisir, sinif yok.
"""

import numpy as np

MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
LEVERAGE = 25.0
MIN_BARS = 200
HARD_STOP_PCT = 2.5


def run_backtest_no_lookahead(
    closes, highs, lows,
    pmax_line, mavg_arr,
    kc_upper, kc_lower,
    initial_balance=1000.0,
    margin_ratio=1.0/40.0,
    max_dca_steps=4,
    tp_close_pct=0.50,
):
    """
    Rust backtest.rs birebir portu, look-ahead duzeltilmis.

    Farklar:
      - kc_upper[i-1], kc_lower[i-1] kullanilir (i degil)
      - Dinamik margin: equity * margin_ratio
      - PMax sinyal: mavg vs pmax_line crossover (Rust ile ayni)
    """
    n = len(closes)
    total_fee_rate = MAKER_FEE + TAKER_FEE

    condition = 0.0       # 1=LONG, -1=SHORT, 0=FLAT
    avg_entry_price = 0.0
    total_notional = 0.0
    dca_fills = 0

    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    tp_count = 0
    rev_count = 0
    hard_stop_count = 0

    # Trade log
    trades = []
    # Haftalik tracking icin equity curve
    equity_curve = np.full(n, np.nan)

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            equity_curve[i] = balance
            continue

        # Dinamik margin
        current_margin = max(0.0, balance * margin_ratio)

        # === PMax crossover ===
        if i > 0 and not np.isnan(mavg_arr[i-1]) and not np.isnan(pmax_line[i-1]):
            prev_m = mavg_arr[i-1]
            prev_p = pmax_line[i-1]
            curr_m = mavg_arr[i]
            curr_p = pmax_line[i]

            buy_cross = prev_m <= prev_p and curr_m > curr_p
            sell_cross = prev_m >= prev_p and curr_m < curr_p

            if buy_cross and condition <= 0.0:
                # Close SHORT (reversal)
                if condition < 0.0 and total_notional > 0.0:
                    pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0
                    pnl = total_notional * pnl_pct / 100.0
                    fee = total_notional * TAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    trade_count += 1
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                    rev_count += 1
                    trades.append({"bar": i, "type": "REV_CLOSE", "side": "SHORT",
                                   "price": closes[i], "pnl": pnl - fee})

                # Open LONG
                current_margin = max(0.0, balance * margin_ratio)
                if balance >= current_margin and current_margin > 0:
                    condition = 1.0
                    avg_entry_price = closes[i]
                    total_notional = current_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                    trades.append({"bar": i, "type": "ENTRY", "side": "LONG",
                                   "price": closes[i], "margin": current_margin})
                else:
                    condition = 0.0
                    total_notional = 0.0

            elif sell_cross and condition >= 0.0:
                # Close LONG (reversal)
                if condition > 0.0 and total_notional > 0.0:
                    pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0
                    pnl = total_notional * pnl_pct / 100.0
                    fee = total_notional * TAKER_FEE
                    balance += pnl - fee
                    total_pnl += pnl
                    total_fees += fee
                    trade_count += 1
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                    rev_count += 1
                    trades.append({"bar": i, "type": "REV_CLOSE", "side": "LONG",
                                   "price": closes[i], "pnl": pnl - fee})

                # Open SHORT
                current_margin = max(0.0, balance * margin_ratio)
                if balance >= current_margin and current_margin > 0:
                    condition = -1.0
                    avg_entry_price = closes[i]
                    total_notional = current_margin * LEVERAGE
                    dca_fills = 0
                    fee = total_notional * TAKER_FEE
                    balance -= fee
                    total_fees += fee
                    trades.append({"bar": i, "type": "ENTRY", "side": "SHORT",
                                   "price": closes[i], "margin": current_margin})
                else:
                    condition = 0.0
                    total_notional = 0.0

        # === Keltner DCA / TP / Hard Stop ===
        # KRITIK: kc[i-1] kullan, kc[i] degil (look-ahead yok)
        if condition != 0.0 and total_notional > 0.0 and i >= MIN_BARS + 1:
            kc_u = kc_upper[i - 1]  # ONCEKI barinki
            kc_l = kc_lower[i - 1]  # ONCEKI barinki

            if np.isnan(kc_u) or np.isnan(kc_l):
                equity_curve[i] = balance
                continue

            if condition > 0.0:
                # LONG Hard Stop
                if dca_fills >= max_dca_steps and avg_entry_price > 0:
                    loss_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0
                    if loss_pct >= HARD_STOP_PCT:
                        pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0
                        pnl = total_notional * pnl_pct / 100.0
                        fee = total_notional * TAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        trade_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        hard_stop_count += 1
                        trades.append({"bar": i, "type": "HARD_STOP", "side": "LONG",
                                       "price": closes[i], "pnl": pnl - fee})
                        condition = 0.0
                        total_notional = 0.0
                        equity_curve[i] = balance
                        continue

                # LONG DCA at KC lower
                if dca_fills < max_dca_steps and lows[i] <= kc_l:
                    current_margin = max(0.0, balance * margin_ratio)
                    if balance >= current_margin and current_margin > 0:
                        step = current_margin * LEVERAGE
                        old = total_notional
                        total_notional += step
                        avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional
                        dca_fills += 1
                        fee = step * MAKER_FEE
                        balance -= fee
                        total_fees += fee
                        trades.append({"bar": i, "type": f"DCA{dca_fills}", "side": "LONG",
                                       "price": kc_l, "margin": current_margin})

                # LONG TP at KC upper
                elif dca_fills > 0 and highs[i] >= kc_u and tp_close_pct > 0:
                    breakeven_price = avg_entry_price * (1.0 + total_fee_rate)
                    if kc_u > breakeven_price:
                        closed = total_notional * tp_close_pct
                        pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0
                        pnl = closed * pnl_pct / 100.0
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1
                        tp_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        trades.append({"bar": i, "type": "TP", "side": "LONG",
                                       "price": kc_u, "pnl": pnl - fee})
                        if total_notional < 1.0:
                            condition = 0.0
                            total_notional = 0.0

            else:  # SHORT
                # SHORT Hard Stop
                if dca_fills >= max_dca_steps and avg_entry_price > 0:
                    loss_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0
                    if loss_pct >= HARD_STOP_PCT:
                        pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0
                        pnl = total_notional * pnl_pct / 100.0
                        fee = total_notional * TAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        trade_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        hard_stop_count += 1
                        trades.append({"bar": i, "type": "HARD_STOP", "side": "SHORT",
                                       "price": closes[i], "pnl": pnl - fee})
                        condition = 0.0
                        total_notional = 0.0
                        equity_curve[i] = balance
                        continue

                # SHORT DCA at KC upper
                if dca_fills < max_dca_steps and highs[i] >= kc_u:
                    current_margin = max(0.0, balance * margin_ratio)
                    if balance >= current_margin and current_margin > 0:
                        step = current_margin * LEVERAGE
                        old = total_notional
                        total_notional += step
                        avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional
                        dca_fills += 1
                        fee = step * MAKER_FEE
                        balance -= fee
                        total_fees += fee
                        trades.append({"bar": i, "type": f"DCA{dca_fills}", "side": "SHORT",
                                       "price": kc_u, "margin": current_margin})

                # SHORT TP at KC lower
                elif dca_fills > 0 and lows[i] <= kc_l and tp_close_pct > 0:
                    breakeven_price = avg_entry_price * (1.0 - total_fee_rate)
                    if kc_l < breakeven_price:
                        closed = total_notional * tp_close_pct
                        pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0
                        pnl = closed * pnl_pct / 100.0
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_pnl += pnl
                        total_fees += fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1
                        tp_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        trades.append({"bar": i, "type": "TP", "side": "SHORT",
                                       "price": kc_l, "pnl": pnl - fee})
                        if total_notional < 1.0:
                            condition = 0.0
                            total_notional = 0.0

        # Drawdown tracking
        if balance > peak_balance:
            peak_balance = balance
        if peak_balance > 0:
            dd = (peak_balance - balance) / peak_balance * 100.0
            if dd > max_dd:
                max_dd = dd

        equity_curve[i] = balance

    # Close remaining position
    if condition != 0.0 and total_notional > 0.0 and n > 0:
        if condition > 0:
            pnl_pct = (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        else:
            pnl_pct = (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        pnl = total_notional * pnl_pct / 100.0
        fee = total_notional * TAKER_FEE
        balance += pnl - fee
        total_pnl += pnl
        total_fees += fee
        trade_count += 1
        if pnl > 0: win_count += 1
        else: loss_count += 1

    if balance > peak_balance:
        peak_balance = balance
    if peak_balance > 0:
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd

    net_pct = (balance - initial_balance) / initial_balance * 100.0
    win_rate = win_count / trade_count * 100.0 if trade_count > 0 else 0.0

    return {
        "net_pct": net_pct,
        "balance": balance,
        "total_trades": trade_count,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "tp_count": tp_count,
        "rev_count": rev_count,
        "hard_stop_count": hard_stop_count,
        "trades": trades,
        "equity_curve": equity_curve,
    }

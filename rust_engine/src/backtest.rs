/// Backtest engine — Python backtest.py::run_backtest_with_pmax() BIREBIR port.

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_PER_TRADE: f64 = 250.0;
const MAKER_FEE: f64 = 0.0002;
const TAKER_FEE: f64 = 0.0005;
const MIN_BARS: usize = 200;

pub struct BacktestResult {
    pub net_pct: f64,
    pub balance: f64,
    pub total_trades: i32,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub tp_count: i32,
    pub rev_count: i32,
    pub hard_stop_count: i32,
}

fn apply_filters(
    idx: usize,
    side: i8,
    closes: &[f64],
    ema_filter: &[f64],
    rsi_vals: &[f64],
    rsi_ema_vals: &[f64],
    atr_vol: &[f64],
    rsi_overbought: f64,
) -> bool {
    let c = closes[idx];

    if !ema_filter[idx].is_nan() {
        if side == 1 && c < ema_filter[idx] { return false; }
        if side == -1 && c > ema_filter[idx] { return false; }
    }

    let r = if !rsi_vals[idx].is_nan() { rsi_vals[idx] } else { 50.0 };
    let r_ema = if !rsi_ema_vals[idx].is_nan() { rsi_ema_vals[idx] } else { 50.0 };
    let rsi_os = 100.0 - rsi_overbought;

    if side == 1 && r > rsi_overbought && r > r_ema { return false; }
    if side == -1 && r < rsi_os && r < r_ema { return false; }

    // ATR volume filter — np.percentile(valid, 20) birebir
    if idx >= 200 {
        let start = idx - 200;
        let window = &atr_vol[start..=idx];
        let mut valid: Vec<f64> = window.iter().copied().filter(|v| !v.is_nan()).collect();
        if !valid.is_empty() {
            valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // np.percentile linear interpolation
            let p = 20.0 / 100.0 * (valid.len() as f64 - 1.0);
            let lo = p.floor() as usize;
            let hi = lo + 1;
            let frac = p - lo as f64;
            let threshold = if hi < valid.len() {
                valid[lo] * (1.0 - frac) + valid[hi] * frac
            } else {
                valid[lo]
            };
            if !atr_vol[idx].is_nan() && atr_vol[idx] < threshold { return false; }
        }
    }
    true
}

pub fn run_backtest_with_pmax(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    pmax_line: &[f64],
    mavg_arr: &[f64],
    _direction_arr: &[f64],
    rsi_vals: &[f64],
    ema_filter: &[f64],
    rsi_ema_vals: &[f64],
    atr_vol: &[f64],
    kc_upper: &[f64],
    kc_lower: &[f64],
    rsi_overbought: f64,
    max_dca_steps: i32,
    tp_close_pct: f64,
) -> BacktestResult {
    let n = closes.len();
    let total_fee_rate = MAKER_FEE + TAKER_FEE;
    let hard_stop_pct = 2.5_f64; // default fallback

    let mut condition: f64 = 0.0;
    let mut avg_entry_price: f64 = 0.0;
    let mut total_notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;

    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;
    let mut tp_count: i32 = 0;
    let mut rev_count: i32 = 0;
    let mut hard_stop_count: i32 = 0;

    for i in MIN_BARS..n {
        if mavg_arr[i].is_nan() || pmax_line[i].is_nan() { continue; }

        // PMax crossover
        if i > 0 && !mavg_arr[i-1].is_nan() && !pmax_line[i-1].is_nan() {
            let prev_m = mavg_arr[i-1];
            let prev_p = pmax_line[i-1];
            let curr_m = mavg_arr[i];
            let curr_p = pmax_line[i];

            let buy_cross = prev_m <= prev_p && curr_m > curr_p;
            let sell_cross = prev_m >= prev_p && curr_m < curr_p;

            if buy_cross && condition <= 0.0 {
                if condition < 0.0 && total_notional > 0.0 {
                    // Close SHORT (reversal)
                    let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                    rev_count += 1;
                }
                if apply_filters(i, 1, closes, ema_filter, rsi_vals, rsi_ema_vals, atr_vol, rsi_overbought)
                    && balance >= MARGIN_PER_TRADE
                {
                    condition = 1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            } else if sell_cross && condition >= 0.0 {
                if condition > 0.0 && total_notional > 0.0 {
                    // Close LONG (reversal)
                    let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                    rev_count += 1;
                }
                if apply_filters(i, -1, closes, ema_filter, rsi_vals, rsi_ema_vals, atr_vol, rsi_overbought)
                    && balance >= MARGIN_PER_TRADE
                {
                    condition = -1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            }
        }

        // Keltner DCA/TP/Hard Stop
        if condition != 0.0 && total_notional > 0.0 {
            let kc_u = kc_upper[i];
            let kc_l = kc_lower[i];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                // LONG Hard Stop
                if dca_fills >= max_dca_steps && avg_entry_price > 0.0 {
                    let loss_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    if loss_pct >= hard_stop_pct {
                        let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = total_notional * pnl_pct / 100.0;
                        let fee = total_notional * TAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        trade_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        hard_stop_count += 1;
                        condition = 0.0;
                        total_notional = 0.0;
                        continue;
                    }
                }
                // LONG DCA at KC lower (lows[i] <= kc_l)
                if dca_fills < max_dca_steps && lows[i] <= kc_l && balance >= MARGIN_PER_TRADE {
                    let step = MARGIN_PER_TRADE * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
                // LONG TP at KC upper (highs[i] >= kc_u)
                else if dca_fills > 0 && highs[i] >= kc_u && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 + total_fee_rate);
                    if kc_u > breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            } else {
                // SHORT Hard Stop
                if dca_fills >= max_dca_steps && avg_entry_price > 0.0 {
                    let loss_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    if loss_pct >= hard_stop_pct {
                        let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                        let pnl = total_notional * pnl_pct / 100.0;
                        let fee = total_notional * TAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        trade_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        hard_stop_count += 1;
                        condition = 0.0;
                        total_notional = 0.0;
                        continue;
                    }
                }
                // SHORT DCA at KC upper (highs[i] >= kc_u)
                if dca_fills < max_dca_steps && highs[i] >= kc_u && balance >= MARGIN_PER_TRADE {
                    let step = MARGIN_PER_TRADE * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
                // SHORT TP at KC lower (lows[i] <= kc_l)
                else if dca_fills > 0 && lows[i] <= kc_l && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 - total_fee_rate);
                    if kc_l < breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            }
        }

        // Drawdown tracking
        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    // Close remaining position
    if condition != 0.0 && total_notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        } else {
            (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        };
        let pnl = total_notional * pnl_pct / 100.0;
        let fee = total_notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    BacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count, hard_stop_count,
    }
}


/// Backtest with KC LAGGED — kc_upper[i-1] / kc_lower[i-1] (NO look-ahead).
/// Birebir run_backtest_with_pmax kopyasi, tek fark: KC index i-1.
/// Filtreler KAPALI (saf PMax + KC testi).
pub fn run_backtest_kc_lagged(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    pmax_line: &[f64],
    mavg_arr: &[f64],
    kc_upper: &[f64],
    kc_lower: &[f64],
    max_dca_steps: i32,
    tp_close_pct: f64,
) -> BacktestResult {
    let n = closes.len();
    let total_fee_rate = MAKER_FEE + TAKER_FEE;

    let mut condition: f64 = 0.0;
    let mut avg_entry_price: f64 = 0.0;
    let mut total_notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;

    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;
    let mut tp_count: i32 = 0;
    let mut rev_count: i32 = 0;
    let mut hard_stop_count: i32 = 0;

    for i in MIN_BARS..n {
        if mavg_arr[i].is_nan() || pmax_line[i].is_nan() { continue; }

        // PMax crossover (ayni mantik, look-ahead yok)
        if i > 0 && !mavg_arr[i-1].is_nan() && !pmax_line[i-1].is_nan() {
            let prev_m = mavg_arr[i-1];
            let prev_p = pmax_line[i-1];
            let curr_m = mavg_arr[i];
            let curr_p = pmax_line[i];

            let buy_cross = prev_m <= prev_p && curr_m > curr_p;
            let sell_cross = prev_m >= prev_p && curr_m < curr_p;

            if buy_cross && condition <= 0.0 {
                if condition < 0.0 && total_notional > 0.0 {
                    let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                    rev_count += 1;
                }
                if balance >= MARGIN_PER_TRADE {
                    condition = 1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            } else if sell_cross && condition >= 0.0 {
                if condition > 0.0 && total_notional > 0.0 {
                    let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                    rev_count += 1;
                }
                if balance >= MARGIN_PER_TRADE {
                    condition = -1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            }
        }

        // === KC DCA/TP — LAGGED: kc[i-1] kullan ===
        if condition != 0.0 && total_notional > 0.0 && i >= MIN_BARS + 1 {
            let kc_u = kc_upper[i - 1];
            let kc_l = kc_lower[i - 1];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                if dca_fills < max_dca_steps && lows[i] <= kc_l && balance >= MARGIN_PER_TRADE {
                    let step = MARGIN_PER_TRADE * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
                else if dca_fills > 0 && highs[i] >= kc_u && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 + total_fee_rate);
                    if kc_u > breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            } else {
                if dca_fills < max_dca_steps && highs[i] >= kc_u && balance >= MARGIN_PER_TRADE {
                    let step = MARGIN_PER_TRADE * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
                else if dca_fills > 0 && lows[i] <= kc_l && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 - total_fee_rate);
                    if kc_l < breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            }
        }

        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    if condition != 0.0 && total_notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        } else {
            (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        };
        let pnl = total_notional * pnl_pct / 100.0;
        let fee = total_notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    BacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count, hard_stop_count,
    }
}


/// Backtest with KC LAGGED + Graduated DCA multipliers.
/// Her DCA adiminda farkli margin carpani: margin * dca_mults[step].
/// kc[i-1] kullanir (look-ahead yok).
pub fn run_backtest_kc_lagged_graduated(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    pmax_line: &[f64],
    mavg_arr: &[f64],
    kc_upper: &[f64],
    kc_lower: &[f64],
    max_dca_steps: i32,
    tp_close_pct: f64,
    dca_m1: f64,
    dca_m2: f64,
    dca_m3: f64,
    dca_m4: f64,
) -> BacktestResult {
    let n = closes.len();
    let total_fee_rate = MAKER_FEE + TAKER_FEE;
    let dca_mults = [dca_m1, dca_m2, dca_m3, dca_m4];

    let mut condition: f64 = 0.0;
    let mut avg_entry_price: f64 = 0.0;
    let mut total_notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;

    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;
    let mut tp_count: i32 = 0;
    let mut rev_count: i32 = 0;
    let hard_stop_count: i32 = 0;

    for i in MIN_BARS..n {
        if mavg_arr[i].is_nan() || pmax_line[i].is_nan() { continue; }

        // PMax crossover
        if i > 0 && !mavg_arr[i-1].is_nan() && !pmax_line[i-1].is_nan() {
            let prev_m = mavg_arr[i-1];
            let prev_p = pmax_line[i-1];
            let curr_m = mavg_arr[i];
            let curr_p = pmax_line[i];

            let buy_cross = prev_m <= prev_p && curr_m > curr_p;
            let sell_cross = prev_m >= prev_p && curr_m < curr_p;

            if buy_cross && condition <= 0.0 {
                if condition < 0.0 && total_notional > 0.0 {
                    let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                }
                if balance >= MARGIN_PER_TRADE {
                    condition = 1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            } else if sell_cross && condition >= 0.0 {
                if condition > 0.0 && total_notional > 0.0 {
                    let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                }
                if balance >= MARGIN_PER_TRADE {
                    condition = -1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            }
        }

        // KC DCA/TP — LAGGED + GRADUATED
        if condition != 0.0 && total_notional > 0.0 && i >= MIN_BARS + 1 {
            let kc_u = kc_upper[i - 1];
            let kc_l = kc_lower[i - 1];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                // LONG DCA — graduated margin
                if dca_fills < max_dca_steps && lows[i] <= kc_l {
                    let mult = dca_mults[dca_fills as usize];
                    let dca_margin = MARGIN_PER_TRADE * mult;
                    if balance >= dca_margin {
                        let step = dca_margin * LEVERAGE;
                        let old = total_notional;
                        total_notional += step;
                        avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                        dca_fills += 1;
                        let fee = step * MAKER_FEE;
                        balance -= fee;
                        total_fees += fee;
                    }
                }
                // LONG TP
                else if dca_fills > 0 && highs[i] >= kc_u && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 + total_fee_rate);
                    if kc_u > breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            } else {
                // SHORT DCA — graduated margin
                if dca_fills < max_dca_steps && highs[i] >= kc_u {
                    let mult = dca_mults[dca_fills as usize];
                    let dca_margin = MARGIN_PER_TRADE * mult;
                    if balance >= dca_margin {
                        let step = dca_margin * LEVERAGE;
                        let old = total_notional;
                        total_notional += step;
                        avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                        dca_fills += 1;
                        let fee = step * MAKER_FEE;
                        balance -= fee;
                        total_fees += fee;
                    }
                }
                // SHORT TP
                else if dca_fills > 0 && lows[i] <= kc_l && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 - total_fee_rate);
                    if kc_l < breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            }
        }

        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    // Close remaining
    if condition != 0.0 && total_notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        } else {
            (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        };
        let pnl = total_notional * pnl_pct / 100.0;
        let fee = total_notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    BacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count: 0, hard_stop_count,
    }
}


/// Backtest with KC LAGGED + Graduated DCA + Graduated TP.
/// Her DCA adiminda farkli margin carpani, her TP'de farkli kapatma yuzdesi.
/// tp_pcts[dca_fills-1] kadar pozisyon kapatilir.
pub fn run_backtest_kc_lagged_graduated_tp(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    pmax_line: &[f64],
    mavg_arr: &[f64],
    kc_upper: &[f64],
    kc_lower: &[f64],
    max_dca_steps: i32,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
    tp1: f64, tp2: f64, tp3: f64, tp4: f64,
) -> BacktestResult {
    let n = closes.len();
    let total_fee_rate = MAKER_FEE + TAKER_FEE;
    let dca_mults = [dca_m1, dca_m2, dca_m3, dca_m4];
    let tp_pcts = [tp1, tp2, tp3, tp4];

    let mut condition: f64 = 0.0;
    let mut avg_entry_price: f64 = 0.0;
    let mut total_notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;

    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;
    let mut tp_count: i32 = 0;
    let hard_stop_count: i32 = 0;

    for i in MIN_BARS..n {
        if mavg_arr[i].is_nan() || pmax_line[i].is_nan() { continue; }

        // PMax crossover
        if i > 0 && !mavg_arr[i-1].is_nan() && !pmax_line[i-1].is_nan() {
            let prev_m = mavg_arr[i-1];
            let prev_p = pmax_line[i-1];
            let curr_m = mavg_arr[i];
            let curr_p = pmax_line[i];

            let buy_cross = prev_m <= prev_p && curr_m > curr_p;
            let sell_cross = prev_m >= prev_p && curr_m < curr_p;

            if buy_cross && condition <= 0.0 {
                if condition < 0.0 && total_notional > 0.0 {
                    let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                }
                if balance >= MARGIN_PER_TRADE {
                    condition = 1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            } else if sell_cross && condition >= 0.0 {
                if condition > 0.0 && total_notional > 0.0 {
                    let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                }
                if balance >= MARGIN_PER_TRADE {
                    condition = -1.0;
                    avg_entry_price = closes[i];
                    total_notional = MARGIN_PER_TRADE * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            }
        }

        // KC DCA/TP — LAGGED + GRADUATED DCA + GRADUATED TP
        if condition != 0.0 && total_notional > 0.0 && i >= MIN_BARS + 1 {
            let kc_u = kc_upper[i - 1];
            let kc_l = kc_lower[i - 1];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                // LONG DCA
                if dca_fills < max_dca_steps && lows[i] <= kc_l {
                    let mult = dca_mults[dca_fills as usize];
                    let dca_margin = MARGIN_PER_TRADE * mult;
                    if balance >= dca_margin {
                        let step = dca_margin * LEVERAGE;
                        let old = total_notional;
                        total_notional += step;
                        avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                        dca_fills += 1;
                        let fee = step * MAKER_FEE;
                        balance -= fee;
                        total_fees += fee;
                    }
                }
                // LONG TP — graduated: tp_pcts[dca_fills-1]
                else if dca_fills > 0 && highs[i] >= kc_u {
                    let tp_idx = ((dca_fills - 1) as usize).min(3);
                    let tp_pct = tp_pcts[tp_idx];
                    if tp_pct > 0.0 {
                        let breakeven_price = avg_entry_price * (1.0 + total_fee_rate);
                        if kc_u > breakeven_price {
                            let closed = total_notional * tp_pct;
                            let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                            let pnl = closed * pnl_pct / 100.0;
                            let fee = closed * MAKER_FEE;
                            balance += pnl - fee;
                            total_pnl += pnl;
                            total_fees += fee;
                            total_notional -= closed;
                            dca_fills = (dca_fills - 1).max(0);
                            trade_count += 1;
                            tp_count += 1;
                            if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                            if total_notional < 1.0 {
                                condition = 0.0;
                                total_notional = 0.0;
                            }
                        }
                    }
                }
            } else {
                // SHORT DCA
                if dca_fills < max_dca_steps && highs[i] >= kc_u {
                    let mult = dca_mults[dca_fills as usize];
                    let dca_margin = MARGIN_PER_TRADE * mult;
                    if balance >= dca_margin {
                        let step = dca_margin * LEVERAGE;
                        let old = total_notional;
                        total_notional += step;
                        avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                        dca_fills += 1;
                        let fee = step * MAKER_FEE;
                        balance -= fee;
                        total_fees += fee;
                    }
                }
                // SHORT TP — graduated
                else if dca_fills > 0 && lows[i] <= kc_l {
                    let tp_idx = ((dca_fills - 1) as usize).min(3);
                    let tp_pct = tp_pcts[tp_idx];
                    if tp_pct > 0.0 {
                        let breakeven_price = avg_entry_price * (1.0 - total_fee_rate);
                        if kc_l < breakeven_price {
                            let closed = total_notional * tp_pct;
                            let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                            let pnl = closed * pnl_pct / 100.0;
                            let fee = closed * MAKER_FEE;
                            balance += pnl - fee;
                            total_pnl += pnl;
                            total_fees += fee;
                            total_notional -= closed;
                            dca_fills = (dca_fills - 1).max(0);
                            trade_count += 1;
                            tp_count += 1;
                            if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                            if total_notional < 1.0 {
                                condition = 0.0;
                                total_notional = 0.0;
                            }
                        }
                    }
                }
            }
        }

        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    if condition != 0.0 && total_notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        } else {
            (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        };
        let pnl = total_notional * pnl_pct / 100.0;
        let fee = total_notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    BacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count: 0, hard_stop_count,
    }
}


/// Dynamic margin hesapla: kasa buyuklugune gore kademeli margin
fn calc_dynamic_margin(
    balance: f64,
    base_pct: f64,
    tier1_threshold: f64,
    tier1_pct: f64,
    tier2_threshold: f64,
    tier2_pct: f64,
) -> f64 {
    let pct = if balance >= tier2_threshold {
        tier2_pct
    } else if balance >= tier1_threshold {
        tier1_pct
    } else {
        base_pct
    };
    let margin = balance * pct / 100.0;
    // Min $50, max %10 of balance
    margin.max(50.0).min(balance * 0.10)
}


/// Backtest with dynamic margin — Kelly/DynComp icin.
/// PMax + KC kilitli, sadece pozisyon buyuklugu degisiyor.
pub fn run_backtest_dynamic_margin(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    pmax_line: &[f64],
    mavg_arr: &[f64],
    _direction_arr: &[f64],
    rsi_vals: &[f64],
    ema_filter: &[f64],
    rsi_ema_vals: &[f64],
    atr_vol: &[f64],
    kc_upper: &[f64],
    kc_lower: &[f64],
    rsi_overbought: f64,
    max_dca_steps: i32,
    tp_close_pct: f64,
    // Dynamic margin params
    base_margin_pct: f64,
    tier1_threshold: f64,
    tier1_pct: f64,
    tier2_threshold: f64,
    tier2_pct: f64,
) -> BacktestResult {
    let n = closes.len();
    let total_fee_rate = MAKER_FEE + TAKER_FEE;
    let hard_stop_pct = 2.5_f64;

    let mut condition: f64 = 0.0;
    let mut avg_entry_price: f64 = 0.0;
    let mut total_notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;
    let mut current_margin: f64 = 0.0;

    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;
    let mut tp_count: i32 = 0;
    let mut rev_count: i32 = 0;
    let mut hard_stop_count: i32 = 0;

    for i in MIN_BARS..n {
        if mavg_arr[i].is_nan() || pmax_line[i].is_nan() { continue; }

        // Dynamic margin hesapla
        current_margin = calc_dynamic_margin(
            balance, base_margin_pct, tier1_threshold, tier1_pct, tier2_threshold, tier2_pct,
        );

        if i > 0 && !mavg_arr[i-1].is_nan() && !pmax_line[i-1].is_nan() {
            let prev_m = mavg_arr[i-1];
            let prev_p = pmax_line[i-1];
            let curr_m = mavg_arr[i];
            let curr_p = pmax_line[i];

            let buy_cross = prev_m <= prev_p && curr_m > curr_p;
            let sell_cross = prev_m >= prev_p && curr_m < curr_p;

            if buy_cross && condition <= 0.0 {
                if condition < 0.0 && total_notional > 0.0 {
                    let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                    rev_count += 1;
                }
                if apply_filters(i, 1, closes, ema_filter, rsi_vals, rsi_ema_vals, atr_vol, rsi_overbought)
                    && balance >= current_margin
                {
                    condition = 1.0;
                    avg_entry_price = closes[i];
                    total_notional = current_margin * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            } else if sell_cross && condition >= 0.0 {
                if condition > 0.0 && total_notional > 0.0 {
                    let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    let pnl = total_notional * pnl_pct / 100.0;
                    let fee = total_notional * TAKER_FEE;
                    balance += pnl - fee;
                    total_pnl += pnl;
                    total_fees += fee;
                    trade_count += 1;
                    if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                    rev_count += 1;
                }
                if apply_filters(i, -1, closes, ema_filter, rsi_vals, rsi_ema_vals, atr_vol, rsi_overbought)
                    && balance >= current_margin
                {
                    condition = -1.0;
                    avg_entry_price = closes[i];
                    total_notional = current_margin * LEVERAGE;
                    dca_fills = 0;
                    let fee = total_notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                } else {
                    condition = 0.0;
                    total_notional = 0.0;
                }
            }
        }

        // Keltner DCA/TP/Hard Stop
        if condition != 0.0 && total_notional > 0.0 {
            let kc_u = kc_upper[i];
            let kc_l = kc_lower[i];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                if dca_fills >= max_dca_steps && avg_entry_price > 0.0 {
                    let loss_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                    if loss_pct >= hard_stop_pct {
                        let pnl_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = total_notional * pnl_pct / 100.0;
                        let fee = total_notional * TAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        trade_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        hard_stop_count += 1;
                        condition = 0.0;
                        total_notional = 0.0;
                        continue;
                    }
                }
                // DCA with dynamic margin
                if dca_fills < max_dca_steps && lows[i] <= kc_l && balance >= current_margin {
                    let step = current_margin * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
                else if dca_fills > 0 && highs[i] >= kc_u && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 + total_fee_rate);
                    if kc_u > breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            } else {
                if dca_fills >= max_dca_steps && avg_entry_price > 0.0 {
                    let loss_pct = (closes[i] - avg_entry_price) / avg_entry_price * 100.0;
                    if loss_pct >= hard_stop_pct {
                        let pnl_pct = (avg_entry_price - closes[i]) / avg_entry_price * 100.0;
                        let pnl = total_notional * pnl_pct / 100.0;
                        let fee = total_notional * TAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        trade_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        hard_stop_count += 1;
                        condition = 0.0;
                        total_notional = 0.0;
                        continue;
                    }
                }
                if dca_fills < max_dca_steps && highs[i] >= kc_u && balance >= current_margin {
                    let step = current_margin * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
                else if dca_fills > 0 && lows[i] <= kc_l && tp_close_pct > 0.0 {
                    let breakeven_price = avg_entry_price * (1.0 - total_fee_rate);
                    if kc_l < breakeven_price {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl;
                        total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1;
                        tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 {
                            condition = 0.0;
                            total_notional = 0.0;
                        }
                    }
                }
            }
        }

        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    if condition != 0.0 && total_notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        } else {
            (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        };
        let pnl = total_notional * pnl_pct / 100.0;
        let fee = total_notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    BacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count, hard_stop_count,
    }
}

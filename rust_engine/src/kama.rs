/// KAMA (Kaufman Adaptive Moving Average) + slope-based backtest.
///
/// KAMA: smoothing adapts to market efficiency.
/// Signal: KAMA slope (normalized by ATR) crosses threshold.
/// No look-ahead, no DCA/TP — pure trend signal with reversal.

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;  // kasa/40 = %2.5
const MAKER_FEE: f64 = 0.0002;
const TAKER_FEE: f64 = 0.0005;
const ATR_PERIOD: usize = 14;
const MIN_BARS: usize = 200;

pub struct KamaBacktestResult {
    pub net_pct: f64,
    pub balance: f64,
    pub total_trades: i32,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub tp_count: i32,
    pub rev_count: i32,
}

/// KAMA hesapla
fn compute_kama(closes: &[f64], period: usize, fast_sc: usize, slow_sc: usize) -> Vec<f64> {
    let n = closes.len();
    let mut kama = vec![f64::NAN; n];
    if n < period + 1 { return kama; }

    let fast_alpha = 2.0 / (fast_sc as f64 + 1.0);
    let slow_alpha = 2.0 / (slow_sc as f64 + 1.0);
    kama[period] = closes[period];

    for i in (period + 1)..n {
        let direction = (closes[i] - closes[i - period]).abs();
        let mut volatility = 0.0;
        for j in (i - period + 1)..=i {
            volatility += (closes[j] - closes[j - 1]).abs();
        }
        let er = if volatility > 0.0 { direction / volatility } else { 0.0 };
        let sc = er * (fast_alpha - slow_alpha) + slow_alpha;
        let sc2 = sc * sc;
        let prev = kama[i - 1];
        kama[i] = if prev.is_nan() { closes[i] } else { prev + sc2 * (closes[i] - prev) };
    }
    kama
}

/// ATR hesapla (RMA/Wilder's)
fn compute_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    let mut atr = vec![f64::NAN; n];
    if n < period + 1 { return atr; }

    let mut sum = 0.0;
    for i in 1..=period {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        sum += tr;
    }
    atr[period] = sum / period as f64;

    for i in (period + 1)..n {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        let prev = atr[i - 1];
        atr[i] = if prev.is_nan() { tr } else { (prev * (period as f64 - 1.0) + tr) / period as f64 };
    }
    atr
}

/// Keltner Channel hesapla
fn compute_kc(highs: &[f64], lows: &[f64], closes: &[f64],
              kc_length: usize, kc_multiplier: f64, kc_atr_period: usize)
    -> (Vec<f64>, Vec<f64>)
{
    let n = closes.len();
    let mut kc_upper = vec![f64::NAN; n];
    let mut kc_lower = vec![f64::NAN; n];

    // EMA
    let k = 2.0 / (kc_length as f64 + 1.0);
    let mut ema = vec![f64::NAN; n];
    ema[0] = closes[0];
    for i in 1..n {
        let prev = if ema[i-1].is_nan() { closes[i] } else { ema[i-1] };
        ema[i] = closes[i] * k + prev * (1.0 - k);
    }

    // KC ATR
    let kc_atr = compute_atr(highs, lows, closes, kc_atr_period);

    for i in 0..n {
        if !ema[i].is_nan() && !kc_atr[i].is_nan() {
            kc_upper[i] = ema[i] + kc_atr[i] * kc_multiplier;
            kc_lower[i] = ema[i] - kc_atr[i] * kc_multiplier;
        }
    }

    (kc_upper, kc_lower)
}

/// Saf KAMA backtest (DCA/TP yok)
pub fn run_kama_backtest(
    closes: &[f64], highs: &[f64], lows: &[f64],
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
) -> KamaBacktestResult {
    let n = closes.len();
    let kama = compute_kama(closes, kama_period, kama_fast, kama_slow);
    let atr = compute_atr(highs, lows, closes, ATR_PERIOD);

    let mut condition: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;

    let start = MIN_BARS.max(kama_period + slope_lookback + 1);

    for i in start..n {
        if kama[i].is_nan() || kama[i - slope_lookback].is_nan() || atr[i].is_nan() || atr[i] <= 0.0 {
            continue;
        }

        let norm_slope = (kama[i] - kama[i - slope_lookback]) / atr[i];
        let new_dir = if norm_slope > slope_threshold { 1.0 }
                      else if norm_slope < -slope_threshold { -1.0 }
                      else { 0.0 };

        if new_dir == 0.0 {
            if balance > peak_balance { peak_balance = balance; }
            if peak_balance > 0.0 {
                let dd = (peak_balance - balance) / peak_balance * 100.0;
                if dd > max_dd { max_dd = dd; }
            }
            continue;
        }

        if new_dir != condition {
            if condition != 0.0 && notional > 0.0 {
                let pnl_pct = if condition > 0.0 {
                    (closes[i] - entry_price) / entry_price * 100.0
                } else {
                    (entry_price - closes[i]) / entry_price * 100.0
                };
                let pnl = notional * pnl_pct / 100.0;
                let fee = notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl; total_fees += fee;
                trade_count += 1;
                if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
            }
            if balance >= (balance * MARGIN_RATIO) {
                condition = new_dir;
                entry_price = closes[i];
                notional = (balance * MARGIN_RATIO) * LEVERAGE;
                let fee = notional * TAKER_FEE;
                balance -= fee; total_fees += fee;
            } else { condition = 0.0; notional = 0.0; }
        }

        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    if condition != 0.0 && notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - entry_price) / entry_price * 100.0
        } else {
            (entry_price - closes[n-1]) / entry_price * 100.0
        };
        let pnl = notional * pnl_pct / 100.0;
        let fee = notional * TAKER_FEE;
        balance += pnl - fee; total_pnl += pnl; total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    KamaBacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count: 0, rev_count: trade_count,
    }
}


/// KAMA + KC lagged backtest — KAMA sinyal + KC DCA/TP (kc[i-1])
pub fn run_kama_kc_backtest(
    closes: &[f64], highs: &[f64], lows: &[f64],
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32, tp_close_pct: f64,
) -> KamaBacktestResult {
    let n = closes.len();
    let kama = compute_kama(closes, kama_period, kama_fast, kama_slow);
    let atr = compute_atr(highs, lows, closes, ATR_PERIOD);
    let (kc_upper, kc_lower) = compute_kc(highs, lows, closes, kc_length, kc_multiplier, kc_atr_period);
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

    let start = MIN_BARS.max(kama_period + slope_lookback + 1);

    for i in start..n {
        if kama[i].is_nan() || kama[i - slope_lookback].is_nan() || atr[i].is_nan() || atr[i] <= 0.0 {
            continue;
        }

        let norm_slope = (kama[i] - kama[i - slope_lookback]) / atr[i];
        let new_dir = if norm_slope > slope_threshold { 1.0 }
                      else if norm_slope < -slope_threshold { -1.0 }
                      else { 0.0 };

        // KAMA reversal
        if new_dir != 0.0 && new_dir != condition {
            // Mevcut pozisyonu kapat
            if condition != 0.0 && total_notional > 0.0 {
                let pnl_pct = if condition > 0.0 {
                    (closes[i] - avg_entry_price) / avg_entry_price * 100.0
                } else {
                    (avg_entry_price - closes[i]) / avg_entry_price * 100.0
                };
                let pnl = total_notional * pnl_pct / 100.0;
                let fee = total_notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl; total_fees += fee;
                trade_count += 1; rev_count += 1;
                if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
            }

            // Yeni pozisyon ac
            if balance >= (balance * MARGIN_RATIO) {
                condition = new_dir;
                avg_entry_price = closes[i];
                total_notional = (balance * MARGIN_RATIO) * LEVERAGE;
                dca_fills = 0;
                let fee = total_notional * TAKER_FEE;
                balance -= fee; total_fees += fee;
            } else {
                condition = 0.0; total_notional = 0.0;
            }
        }

        // KC DCA/TP — LAGGED: kc[i-1]
        if condition != 0.0 && total_notional > 0.0 && i >= start + 1 {
            let kc_u = kc_upper[i - 1];
            let kc_l = kc_lower[i - 1];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                // LONG DCA
                if dca_fills < max_dca_steps && lows[i] <= kc_l && balance >= (balance * MARGIN_RATIO) {
                    let step = (balance * MARGIN_RATIO) * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee; total_fees += fee;
                }
                // LONG TP
                else if dca_fills > 0 && highs[i] >= kc_u && tp_close_pct > 0.0 {
                    let breakeven = avg_entry_price * (1.0 + total_fee_rate);
                    if kc_u > breakeven {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl; total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1; tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 { condition = 0.0; total_notional = 0.0; }
                    }
                }
            } else {
                // SHORT DCA
                if dca_fills < max_dca_steps && highs[i] >= kc_u && balance >= (balance * MARGIN_RATIO) {
                    let step = (balance * MARGIN_RATIO) * LEVERAGE;
                    let old = total_notional;
                    total_notional += step;
                    avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                    dca_fills += 1;
                    let fee = step * MAKER_FEE;
                    balance -= fee; total_fees += fee;
                }
                // SHORT TP
                else if dca_fills > 0 && lows[i] <= kc_l && tp_close_pct > 0.0 {
                    let breakeven = avg_entry_price * (1.0 - total_fee_rate);
                    if kc_l < breakeven {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl; total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1; tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 { condition = 0.0; total_notional = 0.0; }
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

    // Kalan pozisyonu kapat
    if condition != 0.0 && total_notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - avg_entry_price) / avg_entry_price * 100.0
        } else {
            (avg_entry_price - closes[n-1]) / avg_entry_price * 100.0
        };
        let pnl = total_notional * pnl_pct / 100.0;
        let fee = total_notional * TAKER_FEE;
        balance += pnl - fee; total_pnl += pnl; total_fees += fee;
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

    KamaBacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count,
    }
}


/// KAMA + KC lagged + Graduated DCA
pub fn run_kama_kc_grad_dca(
    closes: &[f64], highs: &[f64], lows: &[f64],
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32, tp_close_pct: f64,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
) -> KamaBacktestResult {
    let n = closes.len();
    let kama = compute_kama(closes, kama_period, kama_fast, kama_slow);
    let atr = compute_atr(highs, lows, closes, ATR_PERIOD);
    let (kc_upper, kc_lower) = compute_kc(highs, lows, closes, kc_length, kc_multiplier, kc_atr_period);
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

    let start = MIN_BARS.max(kama_period + slope_lookback + 1);

    for i in start..n {
        if kama[i].is_nan() || kama[i - slope_lookback].is_nan() || atr[i].is_nan() || atr[i] <= 0.0 {
            continue;
        }

        let norm_slope = (kama[i] - kama[i - slope_lookback]) / atr[i];
        let new_dir = if norm_slope > slope_threshold { 1.0 }
                      else if norm_slope < -slope_threshold { -1.0 }
                      else { 0.0 };

        // KAMA reversal
        if new_dir != 0.0 && new_dir != condition {
            if condition != 0.0 && total_notional > 0.0 {
                let pnl_pct = if condition > 0.0 {
                    (closes[i] - avg_entry_price) / avg_entry_price * 100.0
                } else {
                    (avg_entry_price - closes[i]) / avg_entry_price * 100.0
                };
                let pnl = total_notional * pnl_pct / 100.0;
                let fee = total_notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl; total_fees += fee;
                trade_count += 1; rev_count += 1;
                if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
            }
            let margin = balance * MARGIN_RATIO;
            if balance >= margin && margin > 0.0 {
                condition = new_dir;
                avg_entry_price = closes[i];
                total_notional = margin * LEVERAGE;
                dca_fills = 0;
                let fee = total_notional * TAKER_FEE;
                balance -= fee; total_fees += fee;
            } else { condition = 0.0; total_notional = 0.0; }
        }

        // KC DCA/TP — LAGGED + GRADUATED DCA
        if condition != 0.0 && total_notional > 0.0 && i >= start + 1 {
            let kc_u = kc_upper[i - 1];
            let kc_l = kc_lower[i - 1];
            if kc_u.is_nan() || kc_l.is_nan() { continue; }

            if condition > 0.0 {
                // LONG DCA — graduated
                if dca_fills < max_dca_steps && lows[i] <= kc_l {
                    let mult = dca_mults[dca_fills as usize];
                    let dca_margin = balance * MARGIN_RATIO * mult;
                    if balance >= dca_margin && dca_margin > 0.0 {
                        let step = dca_margin * LEVERAGE;
                        let old = total_notional;
                        total_notional += step;
                        avg_entry_price = (avg_entry_price * old + kc_l * step) / total_notional;
                        dca_fills += 1;
                        let fee = step * MAKER_FEE;
                        balance -= fee; total_fees += fee;
                    }
                }
                // LONG TP
                else if dca_fills > 0 && highs[i] >= kc_u && tp_close_pct > 0.0 {
                    let breakeven = avg_entry_price * (1.0 + total_fee_rate);
                    if kc_u > breakeven {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (kc_u - avg_entry_price) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl; total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1; tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 { condition = 0.0; total_notional = 0.0; }
                    }
                }
            } else {
                // SHORT DCA — graduated
                if dca_fills < max_dca_steps && highs[i] >= kc_u {
                    let mult = dca_mults[dca_fills as usize];
                    let dca_margin = balance * MARGIN_RATIO * mult;
                    if balance >= dca_margin && dca_margin > 0.0 {
                        let step = dca_margin * LEVERAGE;
                        let old = total_notional;
                        total_notional += step;
                        avg_entry_price = (avg_entry_price * old + kc_u * step) / total_notional;
                        dca_fills += 1;
                        let fee = step * MAKER_FEE;
                        balance -= fee; total_fees += fee;
                    }
                }
                // SHORT TP
                else if dca_fills > 0 && lows[i] <= kc_l && tp_close_pct > 0.0 {
                    let breakeven = avg_entry_price * (1.0 - total_fee_rate);
                    if kc_l < breakeven {
                        let closed = total_notional * tp_close_pct;
                        let pnl_pct = (avg_entry_price - kc_l) / avg_entry_price * 100.0;
                        let pnl = closed * pnl_pct / 100.0;
                        let fee = closed * MAKER_FEE;
                        balance += pnl - fee;
                        total_pnl += pnl; total_fees += fee;
                        total_notional -= closed;
                        dca_fills = (dca_fills - 1).max(0);
                        trade_count += 1; tp_count += 1;
                        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
                        if total_notional < 1.0 { condition = 0.0; total_notional = 0.0; }
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
        balance += pnl - fee; total_pnl += pnl; total_fees += fee;
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

    KamaBacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, tp_count, rev_count,
    }
}


/// KAMA + KC lagged + Graduated DCA + Graduated TP
pub fn run_kama_kc_grad_dca_tp(
    closes: &[f64], highs: &[f64], lows: &[f64],
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
    tp1: f64, tp2: f64, tp3: f64, tp4: f64,
) -> KamaBacktestResult {
    let n = closes.len();
    let kama_arr = compute_kama(closes, kama_period, kama_fast, kama_slow);
    let atr_arr = compute_atr(highs, lows, closes, ATR_PERIOD);
    let (kc_up, kc_lo) = compute_kc(highs, lows, closes, kc_length, kc_multiplier, kc_atr_period);
    let tfr = MAKER_FEE + TAKER_FEE;
    let dca_m = [dca_m1, dca_m2, dca_m3, dca_m4];
    let tp_p = [tp1, tp2, tp3, tp4];

    let mut cond: f64 = 0.0;
    let mut avg_ep: f64 = 0.0;
    let mut tot_n: f64 = 0.0;
    let mut dca_f: i32 = 0;
    let mut bal = INITIAL_BALANCE;
    let mut peak = INITIAL_BALANCE;
    let mut mdd: f64 = 0.0;
    let mut t_pnl: f64 = 0.0;
    let mut t_fee: f64 = 0.0;
    let mut t_cnt: i32 = 0;
    let mut w_cnt: i32 = 0;
    let mut l_cnt: i32 = 0;
    let mut tp_cnt: i32 = 0;
    let mut rv_cnt: i32 = 0;

    let start_i = MIN_BARS.max(kama_period + slope_lookback + 1);

    for i in start_i..n {
        if kama_arr[i].is_nan() || kama_arr[i-slope_lookback].is_nan() || atr_arr[i].is_nan() || atr_arr[i] <= 0.0 { continue; }
        let ns = (kama_arr[i] - kama_arr[i-slope_lookback]) / atr_arr[i];
        let nd = if ns > slope_threshold { 1.0 } else if ns < -slope_threshold { -1.0 } else { 0.0 };

        if nd != 0.0 && nd != cond {
            if cond != 0.0 && tot_n > 0.0 {
                let pp = if cond > 0.0 { (closes[i]-avg_ep)/avg_ep*100.0 } else { (avg_ep-closes[i])/avg_ep*100.0 };
                let pnl = tot_n * pp / 100.0; let fee = tot_n * TAKER_FEE;
                bal += pnl - fee; t_pnl += pnl; t_fee += fee; t_cnt += 1; rv_cnt += 1;
                if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
            }
            let mg = bal * MARGIN_RATIO;
            if bal >= mg && mg > 0.0 {
                cond = nd; avg_ep = closes[i]; tot_n = mg * LEVERAGE; dca_f = 0;
                let fee = tot_n * TAKER_FEE; bal -= fee; t_fee += fee;
            } else { cond = 0.0; tot_n = 0.0; }
        }

        if cond != 0.0 && tot_n > 0.0 && i >= start_i + 1 {
            let ku = kc_up[i-1]; let kl = kc_lo[i-1];
            if ku.is_nan() || kl.is_nan() { continue; }

            if cond > 0.0 {
                if dca_f < max_dca_steps && lows[i] <= kl {
                    let mult = dca_m[dca_f as usize]; let dm = bal * MARGIN_RATIO * mult;
                    if bal >= dm && dm > 0.0 {
                        let s = dm * LEVERAGE; let o = tot_n; tot_n += s;
                        avg_ep = (avg_ep * o + kl * s) / tot_n; dca_f += 1;
                        let fee = s * MAKER_FEE; bal -= fee; t_fee += fee;
                    }
                } else if dca_f > 0 && highs[i] >= ku {
                    let ti = ((dca_f-1) as usize).min(3); let tpp = tp_p[ti];
                    if tpp > 0.0 { let be = avg_ep * (1.0 + tfr);
                        if ku > be { let cl = tot_n * tpp; let pp = (ku-avg_ep)/avg_ep*100.0;
                            let pnl = cl*pp/100.0; let fee = cl*MAKER_FEE;
                            bal += pnl-fee; t_pnl += pnl; t_fee += fee; tot_n -= cl;
                            dca_f = (dca_f-1).max(0); t_cnt += 1; tp_cnt += 1;
                            if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
                            if tot_n < 1.0 { cond = 0.0; tot_n = 0.0; }
                        }
                    }
                }
            } else {
                if dca_f < max_dca_steps && highs[i] >= ku {
                    let mult = dca_m[dca_f as usize]; let dm = bal * MARGIN_RATIO * mult;
                    if bal >= dm && dm > 0.0 {
                        let s = dm * LEVERAGE; let o = tot_n; tot_n += s;
                        avg_ep = (avg_ep * o + ku * s) / tot_n; dca_f += 1;
                        let fee = s * MAKER_FEE; bal -= fee; t_fee += fee;
                    }
                } else if dca_f > 0 && lows[i] <= kl {
                    let ti = ((dca_f-1) as usize).min(3); let tpp = tp_p[ti];
                    if tpp > 0.0 { let be = avg_ep * (1.0 - tfr);
                        if kl < be { let cl = tot_n * tpp; let pp = (avg_ep-kl)/avg_ep*100.0;
                            let pnl = cl*pp/100.0; let fee = cl*MAKER_FEE;
                            bal += pnl-fee; t_pnl += pnl; t_fee += fee; tot_n -= cl;
                            dca_f = (dca_f-1).max(0); t_cnt += 1; tp_cnt += 1;
                            if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
                            if tot_n < 1.0 { cond = 0.0; tot_n = 0.0; }
                        }
                    }
                }
            }
        }
        if bal > peak { peak = bal; }
        if peak > 0.0 { let dd = (peak-bal)/peak*100.0; if dd > mdd { mdd = dd; } }
    }

    if cond != 0.0 && tot_n > 0.0 && n > 0 {
        let pp = if cond > 0.0 { (closes[n-1]-avg_ep)/avg_ep*100.0 } else { (avg_ep-closes[n-1])/avg_ep*100.0 };
        let pnl = tot_n*pp/100.0; let fee = tot_n*TAKER_FEE;
        bal += pnl-fee; t_pnl += pnl; t_fee += fee; t_cnt += 1;
        if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
    }
    if bal > peak { peak = bal; }
    if peak > 0.0 { let dd = (peak-bal)/peak*100.0; if dd > mdd { mdd = dd; } }

    let np = (bal - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let wr = if t_cnt > 0 { w_cnt as f64 / t_cnt as f64 * 100.0 } else { 0.0 };
    KamaBacktestResult { net_pct: np, balance: bal, total_trades: t_cnt, win_rate: wr,
        max_drawdown: mdd, total_pnl: t_pnl, total_fees: t_fee, tp_count: tp_cnt, rev_count: rv_cnt }
}


// ═══════════════════════════════════════════════════════════════════
// COMBINED: KAMA + PMax + CVD+OI + KC Lagged + Graduated DCA/TP
// ═══════════════════════════════════════════════════════════════════
//
// Entry: Üçlü konfirmasyon — KAMA slope + PMax crossover + CVD+OI
//   - KAMA slope ATR-normalized threshold geçmeli
//   - PMax mavg/pmax crossover aynı yöne bakmalı
//   - CVD sinyali aynı yönde + OI rising
//   Üçü de aynı yöne bakınca giriş.
//
// DCA: KC lagged bantlarında kademeli margin (m1-m4)
// TP: KC lagged bantlarında kademeli kapatma (tp1-tp4)
// Çıkış: Reversal (üçlü sinyal ters döndüğünde) veya TP
// Hard stop yok.

fn ema_vec(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 || period == 0 { return out; }
    let k = 2.0 / (period as f64 + 1.0);
    out[0] = data[0];
    for i in 1..n {
        let prev = if out[i-1].is_nan() { data[i] } else { out[i-1] };
        out[i] = data[i] * k + prev * (1.0 - k);
    }
    out
}

pub fn run_combined_backtest(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    // PMax (pre-computed dışarıdan)
    pmax_line: &[f64], mavg_arr: &[f64],
    // KAMA (sabit)
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    // CVD+OI
    cvd_period: usize, imb_weight: f64, cvd_threshold: f64,
    oi_period: usize, oi_threshold: f64,
    // KC
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    // Graduated DCA + TP
    max_dca_steps: i32,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
    tp1: f64, tp2: f64, tp3: f64, tp4: f64,
) -> KamaBacktestResult {
    let n = closes.len();

    // ── KAMA hesapla ──
    let kama_arr = compute_kama(closes, kama_period, kama_fast, kama_slow);
    let atr_arr = compute_atr(highs, lows, closes, ATR_PERIOD);

    // ── KC hesapla ──
    let (kc_up, kc_lo) = compute_kc(highs, lows, closes, kc_length, kc_multiplier, kc_atr_period);

    // ── CVD hesapla ──
    let mut delta = vec![0.0_f64; n];
    let mut imbalance = vec![0.0_f64; n];
    for i in 0..n {
        delta[i] = buy_vol[i] - sell_vol[i];
        let total = buy_vol[i] + sell_vol[i];
        imbalance[i] = if total > 0.0 { delta[i] / total } else { 0.0 };
    }
    let smooth_imb = ema_vec(&imbalance, cvd_period);

    let mut cvd_signal = vec![0.0_f64; n];
    for i in cvd_period..n {
        let mut rd_sum = 0.0;
        let mut rv_sum = 0.0;
        for k in (i + 1 - cvd_period)..=i {
            rd_sum += delta[k];
            rv_sum += buy_vol[k] + sell_vol[k];
        }
        let norm_cvd = if rv_sum > 0.0 { rd_sum / rv_sum } else { 0.0 };
        let imb_val = if smooth_imb[i].is_nan() { 0.0 } else { smooth_imb[i] };
        cvd_signal[i] = norm_cvd * (1.0 - imb_weight) + imb_val * imb_weight;
    }

    // ── OI change ──
    let mut oi_change = vec![0.0_f64; n];
    for i in oi_period..n {
        if !oi[i].is_nan() && !oi[i - oi_period].is_nan() && oi[i - oi_period] > 0.0 {
            oi_change[i] = (oi[i] - oi[i - oi_period]) / oi[i - oi_period];
        }
    }

    // ── Sabitler ──
    let tfr = MAKER_FEE + TAKER_FEE;
    let dca_m = [dca_m1, dca_m2, dca_m3, dca_m4];
    let tp_p = [tp1, tp2, tp3, tp4];

    // ── State ──
    let mut cond: f64 = 0.0;
    let mut avg_ep: f64 = 0.0;
    let mut tot_n: f64 = 0.0;
    let mut dca_f: i32 = 0;
    let mut bal = INITIAL_BALANCE;
    let mut peak = INITIAL_BALANCE;
    let mut mdd: f64 = 0.0;
    let mut t_pnl: f64 = 0.0;
    let mut t_fee: f64 = 0.0;
    let mut t_cnt: i32 = 0;
    let mut w_cnt: i32 = 0;
    let mut l_cnt: i32 = 0;
    let mut tp_cnt: i32 = 0;
    let mut rv_cnt: i32 = 0;

    let start_i = MIN_BARS.max(kama_period + slope_lookback + 1).max(cvd_period + 1).max(oi_period + 1);

    for i in start_i..n {
        // ── Sinyal 1: KAMA slope ──
        if kama_arr[i].is_nan() || kama_arr[i - slope_lookback].is_nan()
            || atr_arr[i].is_nan() || atr_arr[i] <= 0.0 { continue; }
        let ns = (kama_arr[i] - kama_arr[i - slope_lookback]) / atr_arr[i];
        let kama_dir = if ns > slope_threshold { 1.0 } else if ns < -slope_threshold { -1.0 } else { 0.0 };

        // ── Sinyal 2: PMax crossover ──
        let mut pmax_dir: f64 = 0.0;
        if i > 0 && !mavg_arr[i].is_nan() && !pmax_line[i].is_nan()
            && !mavg_arr[i-1].is_nan() && !pmax_line[i-1].is_nan() {
            let buy_cross = mavg_arr[i-1] <= pmax_line[i-1] && mavg_arr[i] > pmax_line[i];
            let sell_cross = mavg_arr[i-1] >= pmax_line[i-1] && mavg_arr[i] < pmax_line[i];
            // PMax yönü: mavg > pmax ise yukarı, mavg < pmax ise aşağı
            if mavg_arr[i] > pmax_line[i] { pmax_dir = 1.0; }
            else if mavg_arr[i] < pmax_line[i] { pmax_dir = -1.0; }
            // Crossover varsa güçlü sinyal — ama yön zaten yukarıda belirlendi
            let _ = (buy_cross, sell_cross); // kullanılıyor ama entry için üçlü lazım
        }

        // ── Sinyal 3: CVD + OI ──
        let cvd_dir = if cvd_signal[i] > cvd_threshold { 1.0 }
                      else if cvd_signal[i] < -cvd_threshold { -1.0 }
                      else { 0.0 };
        let oi_rising = oi_change[i] > oi_threshold;

        // ── Üçlü Konfirmasyon (OI entry'den çıkarıldı) ──
        // KAMA + PMax + CVD aynı yönde → entry
        let combined_dir = if kama_dir > 0.0 && pmax_dir > 0.0 && cvd_dir > 0.0 { 1.0 }
                           else if kama_dir < 0.0 && pmax_dir < 0.0 && cvd_dir < 0.0 { -1.0 }
                           else { 0.0 };

        // ── Entry / Reversal ──
        if combined_dir != 0.0 && combined_dir != cond {
            // Mevcut pozisyonu kapat
            if cond != 0.0 && tot_n > 0.0 {
                let pp = if cond > 0.0 { (closes[i] - avg_ep) / avg_ep * 100.0 }
                         else { (avg_ep - closes[i]) / avg_ep * 100.0 };
                let pnl = tot_n * pp / 100.0;
                let fee = tot_n * TAKER_FEE;
                bal += pnl - fee; t_pnl += pnl; t_fee += fee; t_cnt += 1; rv_cnt += 1;
                if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
            }
            // Yeni pozisyon aç
            let mg = bal * MARGIN_RATIO;
            if bal >= mg && mg > 0.0 {
                cond = combined_dir; avg_ep = closes[i]; tot_n = mg * LEVERAGE; dca_f = 0;
                let fee = tot_n * TAKER_FEE; bal -= fee; t_fee += fee;
            } else { cond = 0.0; tot_n = 0.0; }
        }

        // KAMA tek başına ters dönse bile pozisyon korunur.
        // Çıkış sadece üçlü ters konfirmasyon veya TP ile olur.

        // ── DCA + TP (KC lagged) ──
        if cond != 0.0 && tot_n > 0.0 && i >= start_i + 1 {
            let ku = kc_up[i - 1];
            let kl = kc_lo[i - 1];
            if ku.is_nan() || kl.is_nan() { continue; }

            if cond > 0.0 {
                // LONG DCA: fiyat KC alt bandına inerse + OI rising
                if dca_f < max_dca_steps && lows[i] <= kl && oi_rising {
                    let mult = dca_m[dca_f as usize];
                    let dm = bal * MARGIN_RATIO * mult;
                    if bal >= dm && dm > 0.0 {
                        let s = dm * LEVERAGE; let o = tot_n; tot_n += s;
                        avg_ep = (avg_ep * o + kl * s) / tot_n; dca_f += 1;
                        let fee = s * MAKER_FEE; bal -= fee; t_fee += fee;
                    }
                }
                // LONG TP: fiyat KC üst bandına çıkarsa
                else if dca_f > 0 && highs[i] >= ku {
                    let ti = ((dca_f - 1) as usize).min(3);
                    let tpp = tp_p[ti];
                    if tpp > 0.0 {
                        let be = avg_ep * (1.0 + tfr);
                        if ku > be {
                            let cl = tot_n * tpp;
                            let pp = (ku - avg_ep) / avg_ep * 100.0;
                            let pnl = cl * pp / 100.0; let fee = cl * MAKER_FEE;
                            bal += pnl - fee; t_pnl += pnl; t_fee += fee; tot_n -= cl;
                            dca_f = (dca_f - 1).max(0); t_cnt += 1; tp_cnt += 1;
                            if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
                            if tot_n < 1.0 { cond = 0.0; tot_n = 0.0; }
                        }
                    }
                }
            } else {
                // SHORT DCA: fiyat KC üst bandına çıkarsa + OI rising
                if dca_f < max_dca_steps && highs[i] >= ku && oi_rising {
                    let mult = dca_m[dca_f as usize];
                    let dm = bal * MARGIN_RATIO * mult;
                    if bal >= dm && dm > 0.0 {
                        let s = dm * LEVERAGE; let o = tot_n; tot_n += s;
                        avg_ep = (avg_ep * o + ku * s) / tot_n; dca_f += 1;
                        let fee = s * MAKER_FEE; bal -= fee; t_fee += fee;
                    }
                }
                // SHORT TP: fiyat KC alt bandına inerse
                else if dca_f > 0 && lows[i] <= kl {
                    let ti = ((dca_f - 1) as usize).min(3);
                    let tpp = tp_p[ti];
                    if tpp > 0.0 {
                        let be = avg_ep * (1.0 - tfr);
                        if kl < be {
                            let cl = tot_n * tpp;
                            let pp = (avg_ep - kl) / avg_ep * 100.0;
                            let pnl = cl * pp / 100.0; let fee = cl * MAKER_FEE;
                            bal += pnl - fee; t_pnl += pnl; t_fee += fee; tot_n -= cl;
                            dca_f = (dca_f - 1).max(0); t_cnt += 1; tp_cnt += 1;
                            if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
                            if tot_n < 1.0 { cond = 0.0; tot_n = 0.0; }
                        }
                    }
                }
            }
        }

        if bal > peak { peak = bal; }
        if peak > 0.0 { let dd = (peak - bal) / peak * 100.0; if dd > mdd { mdd = dd; } }
    }

    // Kalan pozisyonu kapat
    if cond != 0.0 && tot_n > 0.0 && n > 0 {
        let pp = if cond > 0.0 { (closes[n-1] - avg_ep) / avg_ep * 100.0 }
                 else { (avg_ep - closes[n-1]) / avg_ep * 100.0 };
        let pnl = tot_n * pp / 100.0; let fee = tot_n * TAKER_FEE;
        bal += pnl - fee; t_pnl += pnl; t_fee += fee; t_cnt += 1;
        if pnl > 0.0 { w_cnt += 1; } else { l_cnt += 1; }
    }
    if bal > peak { peak = bal; }
    if peak > 0.0 { let dd = (peak - bal) / peak * 100.0; if dd > mdd { mdd = dd; } }

    let np = (bal - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let wr = if t_cnt > 0 { w_cnt as f64 / t_cnt as f64 * 100.0 } else { 0.0 };
    KamaBacktestResult { net_pct: np, balance: bal, total_trades: t_cnt, win_rate: wr,
        max_drawdown: mdd, total_pnl: t_pnl, total_fees: t_fee, tp_count: tp_cnt, rev_count: rv_cnt }
}

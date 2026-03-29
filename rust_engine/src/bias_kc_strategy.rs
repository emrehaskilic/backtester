/// Bias Engine + KC Strategy
///
/// Uses the bias engine's continuous direction signal with Keltner Channel
/// entries/exits on 3m timeframe.
///
/// Logic:
///   Bias BULLISH (> threshold) → KC lower band = LONG DCA, KC upper band = TP
///   Bias BEARISH (< -threshold) → KC upper band = SHORT DCA, KC lower band = TP
///   Bias NEUTRAL → no new entries, can close existing
///   Direction reversal → close position + flip
///
/// Weekly reset: balance → 1000 USDT, force-close open positions.

use crate::bias_engine;
use crate::bias_engine::final_bias::BiasDirection;

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

const INITIAL_BALANCE: f64 = 1000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_PCT: f64 = 0.01;          // 1% of balance per entry
const MAKER_FEE: f64 = 0.0002;         // 0.02%
const TAKER_FEE: f64 = 0.0005;         // 0.05%
const TOTAL_FEE_RATE: f64 = MAKER_FEE + TAKER_FEE; // round-trip per unit

// KC parameters (3m timeframe)
const KC_LENGTH: usize = 20;
const KC_MULT: f64 = 2.0;
const KC_ATR_PERIOD: usize = 14;

// DCA
const MAX_DCA: usize = 3;
const DCA_MULTS: [f64; 3] = [1.0, 0.7, 0.5]; // graduated DCA sizing

// TP
const TP_CLOSE_PCT: f64 = 1.0;  // close 100% on TP (simple)

// Bias thresholds
const BIAS_ENTRY_THRESHOLD: f64 = 0.05;   // |bias| > 0.05 to enter (stronger signal required)
const BIAS_EXIT_THRESHOLD: f64 = 0.00;    // exit when bias flips sign
const MIN_CONFIDENCE: f64 = 0.30;         // minimum confidence to enter

// Weekly reset
const BARS_5M_PER_WEEK: usize = 2016;     // 7 * 24 * 12 * (5/5)
const BARS_3M_PER_WEEK: usize = 3360;     // 7 * 24 * 20 * (3/3)

// ═══════════════════════════════════════════════════════════════
// KC Computation (on 3m data)
// ═══════════════════════════════════════════════════════════════

fn compute_kc(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = closes.len();
    // EMA
    let k = 2.0 / (KC_LENGTH as f64 + 1.0);
    let mut ema = vec![f64::NAN; n];
    ema[0] = closes[0];
    for i in 1..n {
        let prev = if ema[i - 1].is_nan() { closes[i] } else { ema[i - 1] };
        ema[i] = closes[i] * k + prev * (1.0 - k);
    }
    // ATR (Wilder's smoothing)
    let mut tr = vec![0.0f64; n];
    tr[0] = highs[0] - lows[0];
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }
    let mut atr = vec![f64::NAN; n];
    if n > KC_ATR_PERIOD {
        let mut s = 0.0;
        for j in 1..=KC_ATR_PERIOD {
            s += tr[j];
        }
        atr[KC_ATR_PERIOD] = s / KC_ATR_PERIOD as f64;
        for i in (KC_ATR_PERIOD + 1)..n {
            atr[i] = (atr[i - 1] * (KC_ATR_PERIOD as f64 - 1.0) + tr[i]) / KC_ATR_PERIOD as f64;
        }
    }
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if !ema[i].is_nan() && !atr[i].is_nan() {
            upper[i] = ema[i] + atr[i] * KC_MULT;
            lower[i] = ema[i] - atr[i] * KC_MULT;
        }
    }
    (upper, lower)
}

// ═══════════════════════════════════════════════════════════════
// Trade Record
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct TradeRecord {
    pub entry_ts: u64,
    pub exit_ts: u64,
    pub direction: f64,    // +1 long, -1 short
    pub avg_entry: f64,
    pub exit_price: f64,
    pub notional: f64,
    pub pnl: f64,
    pub fees: f64,
    pub dca_count: usize,
    pub exit_reason: &'static str, // "TP", "REVERSAL", "WEEK_RESET"
    pub bias_at_entry: f64,
    pub confidence_at_entry: f64,
}

// ═══════════════════════════════════════════════════════════════
// Weekly Result
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct WeeklyResult {
    pub week_idx: usize,
    pub start_ts: u64,
    pub end_ts: u64,
    pub pnl: f64,
    pub n_trades: usize,
    pub n_wins: usize,
    pub n_tp: usize,
    pub n_reversal: usize,
    pub max_dd_pct: f64,
    pub end_balance: f64,
}

// ═══════════════════════════════════════════════════════════════
// Strategy Result
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct StrategyResult {
    pub total_pnl: f64,
    pub total_trades: usize,
    pub total_wins: usize,
    pub win_rate: f64,
    pub total_fees: f64,
    pub max_dd_pct: f64,
    pub avg_weekly_pnl: f64,
    pub profitable_weeks: usize,
    pub total_weeks: usize,
    pub trades: Vec<TradeRecord>,
    pub weeks: Vec<WeeklyResult>,
}

// ═══════════════════════════════════════════════════════════════
// Main Strategy Function
// ═══════════════════════════════════════════════════════════════

/// Run the bias + KC strategy.
///
/// `ts_5m`, `open_5m`, ... — 5m OHLCV + buy_vol + sell_vol + OI (for bias engine)
/// `ts_3m`, `high_3m`, `low_3m`, `close_3m` — 3m OHLC (for KC)
pub fn run_bias_kc_strategy(
    // 5m data for bias
    ts_5m: &[u64],
    open_5m: &[f64],
    high_5m: &[f64],
    low_5m: &[f64],
    close_5m: &[f64],
    buy_vol_5m: &[f64],
    sell_vol_5m: &[f64],
    oi_5m: &[f64],
    // 3m data for KC
    ts_3m: &[u64],
    high_3m: &[f64],
    low_3m: &[f64],
    close_3m: &[f64],
) -> StrategyResult {
    // ── Step 1: Compute bias series on 5m data ──
    let bias_result = bias_engine::compute_bias_series(
        ts_5m, open_5m, high_5m, low_5m, close_5m, buy_vol_5m, sell_vol_5m, oi_5m,
    );

    // ── Step 2: Compute KC on 3m data ──
    let (kc_upper, kc_lower) = compute_kc(close_3m, high_3m, low_3m);
    let n_3m = close_3m.len();

    // ── Step 3: Build 5m timestamp → bias lookup ──
    // For each 3m bar, find the most recent 5m bias
    // 3m bars fit inside 5m bars: each 5m bar covers ~1.67 3m bars
    // Strategy: binary search for the latest 5m bar that closed before this 3m bar
    let n_5m = ts_5m.len();
    let bar_outputs = &bias_result.bar_outputs;

    // ── Step 4: Run strategy on 3m bars ──
    let mut balance = INITIAL_BALANCE;
    let mut peak = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;

    // Position state
    let mut cond: f64 = 0.0;       // +1 long, -1 short, 0 flat
    let mut avg_entry: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut dca_fills: usize = 0;
    let mut entry_ts: u64 = 0;
    let mut entry_bias: f64 = 0.0;
    let mut entry_conf: f64 = 0.0;

    let mut trades: Vec<TradeRecord> = Vec::new();
    let mut weeks: Vec<WeeklyResult> = Vec::new();

    // Weekly tracking
    let mut week_start_idx: usize = 0;
    let mut week_idx: usize = 0;
    let mut week_pnl: f64 = 0.0;
    let mut week_trades: usize = 0;
    let mut week_wins: usize = 0;
    let mut week_tp: usize = 0;
    let mut week_reversal: usize = 0;
    let mut week_max_dd: f64 = 0.0;

    // 5m index tracker (for bias lookup)
    let mut last_5m_idx: usize = 0;

    for i in 1..n_3m {
        // Skip if KC not ready
        if kc_upper[i - 1].is_nan() || kc_lower[i - 1].is_nan() {
            continue;
        }

        // Find the latest 5m bar bias for this 3m bar
        let cur_ts = ts_3m[i];
        while last_5m_idx + 1 < n_5m && ts_5m[last_5m_idx + 1] <= cur_ts {
            last_5m_idx += 1;
        }

        // Get bias and confidence
        let (bias, confidence, direction) = if last_5m_idx < bar_outputs.len() {
            let bo = &bar_outputs[last_5m_idx];
            (bo.final_bias, bo.confidence, bo.direction)
        } else {
            (0.0, 0.0, BiasDirection::Neutral)
        };

        // ── Weekly reset check ──
        let bars_in_week = i - week_start_idx;
        if bars_in_week >= BARS_3M_PER_WEEK {
            // Force close any open position
            if cond != 0.0 {
                let exit_price = close_3m[i];
                let pp = if cond > 0.0 {
                    (exit_price - avg_entry) / avg_entry
                } else {
                    (avg_entry - exit_price) / avg_entry
                };
                let pnl = notional * pp;
                let fee = notional * MAKER_FEE;
                balance += pnl - fee;
                week_pnl += pnl - fee;

                trades.push(TradeRecord {
                    entry_ts,
                    exit_ts: cur_ts,
                    direction: cond,
                    avg_entry,
                    exit_price,
                    notional,
                    pnl: pnl - fee,
                    fees: fee + notional * TAKER_FEE, // entry fee was already deducted
                    dca_count: dca_fills,
                    exit_reason: "WEEK_RESET",
                    bias_at_entry: entry_bias,
                    confidence_at_entry: entry_conf,
                });

                week_trades += 1;
                if pnl > 0.0 { week_wins += 1; }

                cond = 0.0;
                notional = 0.0;
                avg_entry = 0.0;
                dca_fills = 0;
            }

            // Save weekly result
            weeks.push(WeeklyResult {
                week_idx,
                start_ts: ts_3m[week_start_idx],
                end_ts: cur_ts,
                pnl: week_pnl,
                n_trades: week_trades,
                n_wins: week_wins,
                n_tp: week_tp,
                n_reversal: week_reversal,
                max_dd_pct: week_max_dd,
                end_balance: balance,
            });

            // Reset for new week
            balance = INITIAL_BALANCE;
            peak = INITIAL_BALANCE;
            week_start_idx = i;
            week_idx += 1;
            week_pnl = 0.0;
            week_trades = 0;
            week_wins = 0;
            week_tp = 0;
            week_reversal = 0;
            week_max_dd = 0.0;
        }

        // Use lagged KC (i-1) for entry/exit decisions
        let kc_u = kc_upper[i - 1];
        let kc_l = kc_lower[i - 1];

        // ── Determine desired direction from bias ──
        let desired_dir: f64 = if bias > BIAS_ENTRY_THRESHOLD && confidence >= MIN_CONFIDENCE {
            1.0 // long
        } else if bias < -BIAS_ENTRY_THRESHOLD && confidence >= MIN_CONFIDENCE {
            -1.0 // short
        } else {
            0.0 // neutral
        };

        // ── Direction reversal → close position ──
        if cond != 0.0 && desired_dir != 0.0 && desired_dir != cond {
            let exit_price = close_3m[i];
            let pp = if cond > 0.0 {
                (exit_price - avg_entry) / avg_entry
            } else {
                (avg_entry - exit_price) / avg_entry
            };
            let pnl = notional * pp;
            let fee = notional * MAKER_FEE;
            balance += pnl - fee;
            week_pnl += pnl - fee;

            trades.push(TradeRecord {
                entry_ts,
                exit_ts: cur_ts,
                direction: cond,
                avg_entry,
                exit_price,
                notional,
                pnl: pnl - fee,
                fees: fee + notional * TAKER_FEE,
                dca_count: dca_fills,
                exit_reason: "REVERSAL",
                bias_at_entry: entry_bias,
                confidence_at_entry: entry_conf,
            });

            week_trades += 1;
            week_reversal += 1;
            if pnl > 0.0 { week_wins += 1; }

            cond = 0.0;
            notional = 0.0;
            avg_entry = 0.0;
            dca_fills = 0;
        }

        // ── Open new position ──
        if cond == 0.0 && desired_dir != 0.0 {
            // Long: enter at KC lower band
            if desired_dir > 0.0 && low_3m[i] <= kc_l {
                let margin = balance * MARGIN_PCT;
                if margin > 1.0 {
                    cond = 1.0;
                    avg_entry = kc_l;
                    notional = margin * LEVERAGE;
                    dca_fills = 0;
                    entry_ts = cur_ts;
                    entry_bias = bias;
                    entry_conf = confidence;
                    let fee = notional * TAKER_FEE;
                    balance -= fee;
                }
            }
            // Short: enter at KC upper band
            else if desired_dir < 0.0 && high_3m[i] >= kc_u {
                let margin = balance * MARGIN_PCT;
                if margin > 1.0 {
                    cond = -1.0;
                    avg_entry = kc_u;
                    notional = margin * LEVERAGE;
                    dca_fills = 0;
                    entry_ts = cur_ts;
                    entry_bias = bias;
                    entry_conf = confidence;
                    let fee = notional * TAKER_FEE;
                    balance -= fee;
                }
            }
        }

        // ── DCA (add to position at KC band) ──
        if cond > 0.0 && dca_fills < MAX_DCA && low_3m[i] <= kc_l {
            let mult = DCA_MULTS[dca_fills];
            let dm = balance * MARGIN_PCT * mult;
            if dm > 1.0 {
                let step = dm * LEVERAGE;
                let old_notional = notional;
                avg_entry = (avg_entry * old_notional + kc_l * step) / (old_notional + step);
                notional += step;
                dca_fills += 1;
                let fee = step * TAKER_FEE;
                balance -= fee;
            }
        }
        if cond < 0.0 && dca_fills < MAX_DCA && high_3m[i] >= kc_u {
            let mult = DCA_MULTS[dca_fills];
            let dm = balance * MARGIN_PCT * mult;
            if dm > 1.0 {
                let step = dm * LEVERAGE;
                let old_notional = notional;
                avg_entry = (avg_entry * old_notional + kc_u * step) / (old_notional + step);
                notional += step;
                dca_fills += 1;
                let fee = step * TAKER_FEE;
                balance -= fee;
            }
        }

        // ── TP (take profit at opposite KC band) ──
        if cond > 0.0 && high_3m[i] >= kc_u {
            let be = avg_entry * (1.0 + TOTAL_FEE_RATE);
            if kc_u > be {
                let close_amt = notional * TP_CLOSE_PCT;
                let pp = (kc_u - avg_entry) / avg_entry;
                let pnl = close_amt * pp;
                let fee = close_amt * MAKER_FEE;
                balance += pnl - fee;
                week_pnl += pnl - fee;

                trades.push(TradeRecord {
                    entry_ts,
                    exit_ts: cur_ts,
                    direction: cond,
                    avg_entry,
                    exit_price: kc_u,
                    notional: close_amt,
                    pnl: pnl - fee,
                    fees: fee + close_amt * TAKER_FEE,
                    dca_count: dca_fills,
                    exit_reason: "TP",
                    bias_at_entry: entry_bias,
                    confidence_at_entry: entry_conf,
                });

                week_trades += 1;
                week_tp += 1;
                if pnl > 0.0 { week_wins += 1; }

                notional -= close_amt;
                if notional < 1.0 {
                    cond = 0.0;
                    notional = 0.0;
                    avg_entry = 0.0;
                    dca_fills = 0;
                }
            }
        }
        if cond < 0.0 && low_3m[i] <= kc_l {
            let be = avg_entry * (1.0 - TOTAL_FEE_RATE);
            if kc_l < be {
                let close_amt = notional * TP_CLOSE_PCT;
                let pp = (avg_entry - kc_l) / avg_entry;
                let pnl = close_amt * pp;
                let fee = close_amt * MAKER_FEE;
                balance += pnl - fee;
                week_pnl += pnl - fee;

                trades.push(TradeRecord {
                    entry_ts,
                    exit_ts: cur_ts,
                    direction: cond,
                    avg_entry,
                    exit_price: kc_l,
                    notional: close_amt,
                    pnl: pnl - fee,
                    fees: fee + close_amt * TAKER_FEE,
                    dca_count: dca_fills,
                    exit_reason: "TP",
                    bias_at_entry: entry_bias,
                    confidence_at_entry: entry_conf,
                });

                week_trades += 1;
                week_tp += 1;
                if pnl > 0.0 { week_wins += 1; }

                notional -= close_amt;
                if notional < 1.0 {
                    cond = 0.0;
                    notional = 0.0;
                    avg_entry = 0.0;
                    dca_fills = 0;
                }
            }
        }

        // ── Update drawdown ──
        if balance > peak {
            peak = balance;
        }
        let dd = (peak - balance) / peak * 100.0;
        if dd > max_dd { max_dd = dd; }
        if dd > week_max_dd { week_max_dd = dd; }
    }

    // Close final week
    if week_start_idx < n_3m {
        // Force close if open
        if cond != 0.0 {
            let exit_price = close_3m[n_3m - 1];
            let pp = if cond > 0.0 {
                (exit_price - avg_entry) / avg_entry
            } else {
                (avg_entry - exit_price) / avg_entry
            };
            let pnl = notional * pp;
            let fee = notional * MAKER_FEE;
            balance += pnl - fee;
            week_pnl += pnl - fee;

            trades.push(TradeRecord {
                entry_ts,
                exit_ts: ts_3m[n_3m - 1],
                direction: cond,
                avg_entry,
                exit_price,
                notional,
                pnl: pnl - fee,
                fees: fee + notional * TAKER_FEE,
                dca_count: dca_fills,
                exit_reason: "WEEK_RESET",
                bias_at_entry: entry_bias,
                confidence_at_entry: entry_conf,
            });

            week_trades += 1;
            if pnl > 0.0 { week_wins += 1; }
        }

        weeks.push(WeeklyResult {
            week_idx,
            start_ts: ts_3m[week_start_idx],
            end_ts: ts_3m[n_3m - 1],
            pnl: week_pnl,
            n_trades: week_trades,
            n_wins: week_wins,
            n_tp: week_tp,
            n_reversal: week_reversal,
            max_dd_pct: week_max_dd,
            end_balance: balance,
        });
    }

    // Aggregate
    let total_trades = trades.len();
    let total_wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    let total_pnl: f64 = weeks.iter().map(|w| w.pnl).sum();
    let total_fees: f64 = trades.iter().map(|t| t.fees).sum();
    let profitable_weeks = weeks.iter().filter(|w| w.pnl > 0.0).count();
    let total_weeks = weeks.len();
    let avg_weekly_pnl = if total_weeks > 0 { total_pnl / total_weeks as f64 } else { 0.0 };
    let global_max_dd = weeks.iter().map(|w| w.max_dd_pct).fold(0.0f64, f64::max);

    StrategyResult {
        total_pnl,
        total_trades,
        total_wins,
        win_rate: if total_trades > 0 { total_wins as f64 / total_trades as f64 } else { 0.0 },
        total_fees,
        max_dd_pct: global_max_dd,
        avg_weekly_pnl,
        profitable_weeks,
        total_weeks,
        trades,
        weeks,
    }
}

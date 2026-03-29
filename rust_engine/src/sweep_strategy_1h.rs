/// 1H Sweep Strategy — Triple barrier, pattern-based entry.
///
/// 1H high sweep + pattern match → LONG
/// 1H low sweep + pattern match → SHORT
/// TP/SL = 0.5 × Daily ATR, Timeout = 72 bar (6h)
/// Tek pozisyon, sweep pattern'inin dogasiyla uyumlu exit.

use crate::sweep_miner::compute_5m_features;

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const TAKER_FEE: f64 = 0.0005;
const MIN_BARS: usize = 300;

const BP_1H: usize = 12;
const TIMEOUT: usize = 72; // 6h
const TP_SL_MULT: f64 = 0.5;

// ── Top WF-validated 1H patterns ──

struct PatternDef {
    features: Vec<usize>,
    quantiles: Vec<u8>,
    is_high_sweep: bool, // true=high sweep→LONG, false=low sweep→SHORT
}

fn get_1h_patterns() -> Vec<PatternDef> {
    vec![
        // HIGH sweep → LONG (continuation = yukari)
        // OI_change(3)=Q4 AND Vol_micro(4)=Q3 → 88.1% cont, WF 6/7
        PatternDef { features: vec![3, 4], quantiles: vec![3, 2], is_high_sweep: true },
        // OI_change(3)=Q4 AND ATR(6)=Q5 → 85.3% cont, WF 3/4
        PatternDef { features: vec![3, 6], quantiles: vec![3, 4], is_high_sweep: true },

        // LOW sweep → SHORT (continuation = asagi)
        // CVD_micro(0)=Q1 AND Imbalance(5)=Q4 → 97.2% cont, WF 4/4
        PatternDef { features: vec![0, 5], quantiles: vec![0, 3], is_high_sweep: false },
        // CVD_macro(1)=Q3 AND Imbalance(5)=Q4 → 88.3% cont, WF 6/7
        PatternDef { features: vec![1, 5], quantiles: vec![2, 3], is_high_sweep: false },
    ]
}

// ── Quantile helpers ──

fn compute_quantile_thresholds(features: &[[f64; 7]], end: usize) -> Vec<[f64; 4]> {
    let mut thresholds = vec![[0.0_f64; 4]; 7];
    for fi in 0..7 {
        let mut vals: Vec<f64> = (MIN_BARS..end).filter_map(|i| {
            let v = features[i][fi];
            if v.is_nan() { None } else { Some(v) }
        }).collect();
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for q in 0..4 {
            let idx = ((q + 1) as f64 / 5.0 * vals.len() as f64) as usize;
            thresholds[fi][q] = vals[idx.min(vals.len() - 1)];
        }
    }
    thresholds
}

fn get_quantile(val: f64, th: &[f64; 4]) -> u8 {
    let mut q = 0u8;
    for &t in th { if val > t { q += 1; } }
    q
}

// ── Result ──

pub struct Strategy1HResult {
    pub net_pct: f64,
    pub balance: f64,
    pub total_trades: i32,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub long_trades: i32,
    pub short_trades: i32,
    pub long_wins: i32,
    pub short_wins: i32,
    pub tp_count: i32,
    pub sl_count: i32,
    pub timeout_count: i32,
    pub avg_trade_bars: f64,
    pub max_consecutive_loss: i32,
    pub weekly_returns: Vec<f64>,
    pub trade_log: Vec<TradeRecord>,
}

#[derive(Clone)]
pub struct TradeRecord {
    pub bar: usize,
    pub direction: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_type: String, // "TP", "SL", "TIMEOUT"
    pub pnl: f64,
    pub balance_after: f64,
}

// ── Main strategy ──

pub fn run_1h_strategy(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> Strategy1HResult {
    let n = closes.len();

    // Compute features + daily ATR
    let (features, daily_atr) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);

    // Quantile thresholds (rolling monthly update)
    let warmup = (288 * 30).min(n);
    let mut q_th = compute_quantile_thresholds(&features, warmup);

    // Patterns
    let patterns = get_1h_patterns();

    // 1H candle tracking
    let mut prev_1h_high = f64::NEG_INFINITY;
    let mut prev_1h_low = f64::INFINITY;
    let mut high_active = false;
    let mut low_active = false;
    let mut high_cooldown = 0usize;
    let mut low_cooldown = 0usize;
    let mut bars_below = 0usize;
    let mut bars_above = 0usize;

    // Position state
    let mut in_position = false;
    let mut position_dir: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut entry_bar: usize = 0;
    let mut tp_price: f64 = 0.0;
    let mut sl_price: f64 = 0.0;

    // Accounting
    let mut balance = INITIAL_BALANCE;
    let mut peak = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut total_trades: i32 = 0;
    let mut wins: i32 = 0;
    let mut long_trades: i32 = 0;
    let mut short_trades: i32 = 0;
    let mut long_wins: i32 = 0;
    let mut short_wins: i32 = 0;
    let mut tp_count: i32 = 0;
    let mut sl_count: i32 = 0;
    let mut timeout_count: i32 = 0;
    let mut total_trade_bars: usize = 0;
    let mut consec_loss: i32 = 0;
    let mut max_consec: i32 = 0;

    let mut trade_log: Vec<TradeRecord> = Vec::new();
    let mut weekly_returns: Vec<f64> = Vec::new();
    let mut week_start_bal = INITIAL_BALANCE;
    let bpw = 288 * 7;

    let start = MIN_BARS.max(warmup);

    for i in start..n {
        // Monthly quantile update
        if i > warmup && i % (288 * 30) == 0 {
            q_th = compute_quantile_thresholds(&features, i);
        }

        if features[i][0].is_nan() || daily_atr[i].is_nan() || daily_atr[i] <= 0.0 {
            continue;
        }

        let reset_dist = 0.5 * daily_atr[i];

        // ── Update 1H candle at boundaries ──
        if i >= BP_1H && i % BP_1H == 0 {
            let ps = i - BP_1H;
            prev_1h_high = f64::NEG_INFINITY;
            prev_1h_low = f64::INFINITY;
            for j in ps..i {
                if highs[j] > prev_1h_high { prev_1h_high = highs[j]; }
                if lows[j] < prev_1h_low { prev_1h_low = lows[j]; }
            }
            high_active = true;
            low_active = true;
            high_cooldown = 0;
            low_cooldown = 0;
            bars_below = 0;
            bars_above = 0;
        }

        if prev_1h_high == f64::NEG_INFINITY { continue; }

        // ── Check existing position ──
        if in_position {
            let bars_held = i - entry_bar;

            // TP/SL/Timeout check
            let mut exit_type = "";
            let mut exit_price: f64 = 0.0;

            if position_dir > 0.0 {
                // LONG
                if highs[i] >= tp_price {
                    exit_type = "TP";
                    exit_price = tp_price;
                } else if lows[i] <= sl_price {
                    exit_type = "SL";
                    exit_price = sl_price;
                } else if bars_held >= TIMEOUT {
                    exit_type = "TIMEOUT";
                    exit_price = closes[i];
                }
            } else {
                // SHORT
                if lows[i] <= tp_price {
                    exit_type = "TP";
                    exit_price = tp_price;
                } else if highs[i] >= sl_price {
                    exit_type = "SL";
                    exit_price = sl_price;
                } else if bars_held >= TIMEOUT {
                    exit_type = "TIMEOUT";
                    exit_price = closes[i];
                }
            }

            if !exit_type.is_empty() {
                let pnl_pct = if position_dir > 0.0 {
                    (exit_price - entry_price) / entry_price * 100.0
                } else {
                    (entry_price - exit_price) / entry_price * 100.0
                };
                let pnl = notional * pnl_pct / 100.0;
                let fee = notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl;
                total_fees += fee;
                total_trades += 1;
                total_trade_bars += bars_held;

                if position_dir > 0.0 { long_trades += 1; } else { short_trades += 1; }

                match exit_type {
                    "TP" => { tp_count += 1; },
                    "SL" => { sl_count += 1; },
                    _ => { timeout_count += 1; },
                }

                if pnl > 0.0 {
                    wins += 1;
                    if position_dir > 0.0 { long_wins += 1; } else { short_wins += 1; }
                    consec_loss = 0;
                } else {
                    consec_loss += 1;
                    if consec_loss > max_consec { max_consec = consec_loss; }
                }

                trade_log.push(TradeRecord {
                    bar: entry_bar,
                    direction: position_dir,
                    entry_price,
                    exit_price,
                    exit_type: exit_type.to_string(),
                    pnl,
                    balance_after: balance,
                });

                in_position = false;
                position_dir = 0.0;
                notional = 0.0;
            }
        }

        // ── Sweep detection + pattern match (only if not in position) ──
        if !in_position {
            // High sweep → LONG
            if high_active && i > 0 && highs[i] > prev_1h_high && highs[i-1] <= prev_1h_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let feat = &features[i-1];
                    for pat in patterns.iter().filter(|p| p.is_high_sweep) {
                        let matched = pat.features.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| {
                            get_quantile(feat[fi], &q_th[fi]) == q
                        });
                        if matched {
                            // LONG entry
                            let margin = balance * MARGIN_RATIO;
                            if balance >= margin && margin > 0.0 {
                                let atr = daily_atr[i];
                                in_position = true;
                                position_dir = 1.0;
                                entry_price = prev_1h_high; // entry at level
                                notional = margin * LEVERAGE;
                                entry_bar = i;
                                tp_price = entry_price + TP_SL_MULT * atr;
                                sl_price = entry_price - TP_SL_MULT * atr;
                                let fee = notional * TAKER_FEE;
                                balance -= fee;
                                total_fees += fee;
                            }
                            break;
                        }
                    }
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            // Low sweep → SHORT
            if !in_position && low_active && i > 0 && lows[i] < prev_1h_low && lows[i-1] >= prev_1h_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let feat = &features[i-1];
                    for pat in patterns.iter().filter(|p| !p.is_high_sweep) {
                        let matched = pat.features.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| {
                            get_quantile(feat[fi], &q_th[fi]) == q
                        });
                        if matched {
                            let margin = balance * MARGIN_RATIO;
                            if balance >= margin && margin > 0.0 {
                                let atr = daily_atr[i];
                                in_position = true;
                                position_dir = -1.0;
                                entry_price = prev_1h_low;
                                notional = margin * LEVERAGE;
                                entry_bar = i;
                                tp_price = entry_price - TP_SL_MULT * atr;
                                sl_price = entry_price + TP_SL_MULT * atr;
                                let fee = notional * TAKER_FEE;
                                balance -= fee;
                                total_fees += fee;
                            }
                            break;
                        }
                    }
                    low_active = false;
                    low_cooldown = 0;
                }
            }
        }

        // Reset logic
        if !high_active {
            if closes[i] < prev_1h_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
            high_cooldown += 1;
            if bars_below >= 12 { high_active = true; bars_below = 0; }
        }
        if !low_active {
            if closes[i] > prev_1h_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
            low_cooldown += 1;
            if bars_above >= 12 { low_active = true; bars_above = 0; }
        }

        // DD
        if balance > peak { peak = balance; }
        if peak > 0.0 {
            let dd = (peak - balance) / peak * 100.0;
            if dd > max_dd { max_dd = dd; }
        }

        // Weekly
        if i > start && (i - start) % bpw == 0 {
            let wr = (balance - week_start_bal) / week_start_bal * 100.0;
            weekly_returns.push(wr);
            week_start_bal = balance;
        }
    }

    // Close remaining
    if in_position && n > 0 {
        let pnl_pct = if position_dir > 0.0 {
            (closes[n-1] - entry_price) / entry_price * 100.0
        } else {
            (entry_price - closes[n-1]) / entry_price * 100.0
        };
        let pnl = notional * pnl_pct / 100.0;
        let fee = notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        total_trades += 1;
        if position_dir > 0.0 { long_trades += 1; } else { short_trades += 1; }
        if pnl > 0.0 { wins += 1; if position_dir > 0.0 { long_wins += 1; } else { short_wins += 1; } }
    }

    if balance != week_start_bal {
        let wr = (balance - week_start_bal) / week_start_bal * 100.0;
        weekly_returns.push(wr);
    }

    if balance > peak { peak = balance; }
    if peak > 0.0 { let dd = (peak-balance)/peak*100.0; if dd > max_dd { max_dd = dd; } }

    let net = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let wr = if total_trades > 0 { wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };
    let avg_bars = if total_trades > 0 { total_trade_bars as f64 / total_trades as f64 } else { 0.0 };

    Strategy1HResult {
        net_pct: net, balance, total_trades, win_rate: wr, max_drawdown: max_dd,
        total_pnl, total_fees, long_trades, short_trades, long_wins, short_wins,
        tp_count, sl_count, timeout_count, avg_trade_bars: avg_bars,
        max_consecutive_loss: max_consec, weekly_returns, trade_log,
    }
}

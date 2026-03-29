/// Sweep Strategy — Multi-timeframe weighted voting, always-in-market.
///
/// Mimari:
/// 1. Her bar'da tum TF'lerdeki aktif sweep sinyallerini tara
/// 2. Pattern match + feature filtresi uygula
/// 3. Fibonacci agirlikli skor hesapla (1,2,5,13,34)
/// 4. Skor isaret degistirince reversal
/// 5. Skor 0'da: karda trailing stop, zararda pozisyon kapat
/// 6. Signal expiry: her TF'nin optimal timeout'u kadar yasar

use crate::sweep_miner::{compute_5m_features, FEATURE_NAMES};

// ── Constants ──

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const MAKER_FEE: f64 = 0.0002;
const TAKER_FEE: f64 = 0.0005;
const MIN_BARS: usize = 300;

// Fibonacci weights per TF
const W_1H: f64 = 1.0;
const W_4H: f64 = 2.0;
const W_8H: f64 = 5.0;
const W_DAILY: f64 = 13.0;
const W_WEEKLY: f64 = 34.0;

// Signal expiry (bars)
const EXP_1H: usize = 72;    // 6h
const EXP_4H: usize = 144;   // 12h
const EXP_8H: usize = 240;   // 20h
const EXP_DAILY: usize = 288; // 24h
const EXP_WEEKLY: usize = 720; // 60h

// Trailing stop
const TRAILING_ATR_MULT: f64 = 0.5;

// TF candle sizes (in 5m bars)
const BP_1H: usize = 12;
const BP_4H: usize = 48;
const BP_8H: usize = 96;
const BP_DAILY: usize = 288;
const BP_WEEKLY: usize = 2016; // 7 * 288

// ── Pattern definitions (from sweep miner results) ──
// Top WF-validated patterns per sweep type

#[derive(Clone)]
struct PatternDef {
    feature_indices: Vec<usize>,
    quantiles: Vec<u8>,       // required quantile (0-4)
    direction: f64,           // +1 = continuation is UP (high sweep), -1 = continuation is DOWN (low sweep)
}

fn get_patterns() -> Vec<(usize, f64, usize, Vec<PatternDef>)> {
    // (candle_size, weight, expiry, patterns)
    // Each pattern: if matched after sweep → score += weight * direction

    vec![
        // ── 1H patterns ──
        (BP_1H, W_1H, EXP_1H, vec![
            // 1H High sweep → continuation UP
            // OI_change(3)=Q4 AND Vol_micro(4)=Q3 → 88.1% cont, WF 6/7
            PatternDef { feature_indices: vec![3, 4], quantiles: vec![3, 2], direction: 1.0 },
            // OI_change(3)=Q4 AND ATR(6)=Q5 → 85.3% cont, WF 3/4
            PatternDef { feature_indices: vec![3, 6], quantiles: vec![3, 4], direction: 1.0 },
            // 1H Low sweep → continuation DOWN
            // CVD_micro(0)=Q1 AND Imbalance(5)=Q4 → 97.2% cont, WF 4/4
            PatternDef { feature_indices: vec![0, 5], quantiles: vec![0, 3], direction: -1.0 },
            // CVD_macro(1)=Q3 AND Imbalance(5)=Q4 → 88.3% cont, WF 6/7
            PatternDef { feature_indices: vec![1, 5], quantiles: vec![2, 3], direction: -1.0 },
        ]),

        // ── 4H patterns ──
        (BP_4H, W_4H, EXP_4H, vec![
            // 4H High sweep → continuation UP
            // CVD_macro(1)=Q5 AND Vol_micro(4)=Q3 → 85.8% cont, WF 4/4
            PatternDef { feature_indices: vec![1, 4], quantiles: vec![4, 2], direction: 1.0 },
            // Imbalance(5)=Q4 AND ATR(6)=Q4 → 85.2% cont, WF 5/5
            PatternDef { feature_indices: vec![5, 6], quantiles: vec![3, 3], direction: 1.0 },
            // 4H Low sweep → continuation DOWN
            // CVD_micro(0)=Q1 AND ATR(6)=Q4 → 84.5% cont, WF 7/7
            PatternDef { feature_indices: vec![0, 6], quantiles: vec![0, 3], direction: -1.0 },
            // CVD_micro(0)=Q2 AND ATR(6)=Q5 → 77.8% cont, WF 5/6
            PatternDef { feature_indices: vec![0, 6], quantiles: vec![1, 4], direction: -1.0 },
        ]),

        // ── 8H patterns ──
        (BP_8H, W_8H, EXP_8H, vec![
            // 8H High sweep → continuation UP
            // Imbalance(5)=Q4 AND ATR(6)=Q4 → 86.4% cont, WF 5/5
            PatternDef { feature_indices: vec![5, 6], quantiles: vec![3, 3], direction: 1.0 },
            // OI_change(3)=Q4 AND ATR(6)=Q5 → 80.8% cont, WF 3/3
            PatternDef { feature_indices: vec![3, 6], quantiles: vec![3, 4], direction: 1.0 },
            // 8H Low sweep → continuation DOWN
            // OI_change(3)=Q5 AND Imbalance(5)=Q4 → 84.8% cont, WF 3/3
            PatternDef { feature_indices: vec![3, 5], quantiles: vec![4, 3], direction: -1.0 },
            // OI_change(3)=Q5 AND Vol_micro(4)=Q1 → 86.7% cont, WF 4/4
            PatternDef { feature_indices: vec![3, 4], quantiles: vec![4, 0], direction: -1.0 },
        ]),

        // ── Daily patterns ──
        (BP_DAILY, W_DAILY, EXP_DAILY, vec![
            // Daily High sweep → continuation UP
            // CVD_macro(1)=Q5 AND ATR(6)=Q5 → 82.7% cont, WF 3/3
            PatternDef { feature_indices: vec![1, 6], quantiles: vec![4, 4], direction: 1.0 },
            // CVD_micro(0)=Q1 AND ATR(6)=Q5 → 81.0% cont, WF 4/4
            PatternDef { feature_indices: vec![0, 6], quantiles: vec![0, 4], direction: 1.0 },
            // Daily Low sweep → continuation DOWN
            // OI_change(3)=Q5 AND ATR(6)=Q3 → 88.9% cont, WF 3/3
            PatternDef { feature_indices: vec![3, 6], quantiles: vec![4, 2], direction: -1.0 },
            // CVD_micro(0)=Q1 AND OI_change(3)=Q5 → 83.0% cont, WF 3/3
            PatternDef { feature_indices: vec![0, 3], quantiles: vec![0, 4], direction: -1.0 },
        ]),

        // ── Weekly patterns ──
        // No FDR-validated patterns, use base rate bias
        (BP_WEEKLY, W_WEEKLY, EXP_WEEKLY, vec![
            // Weekly high sweep base rate 71.4% continuation → strong UP bias
            // Use ATR(6)=Q5 as minimal filter (p=0.034)
            PatternDef { feature_indices: vec![6], quantiles: vec![4], direction: 1.0 },
            // Weekly low sweep base rate 74.2% continuation → strong DOWN bias
            // Use OI_change(3) any (p=0.001) as filter
            PatternDef { feature_indices: vec![3], quantiles: vec![3], direction: -1.0 },
        ]),
    ]
}

// ── Quantile computation ──

fn compute_quantile_thresholds(features: &[[f64; 7]], end: usize) -> Vec<[f64; 4]> {
    // For each of 7 features, compute Q20/Q40/Q60/Q80 thresholds from data[0..end]
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

fn get_quantile(val: f64, thresholds: &[f64; 4]) -> u8 {
    let mut q = 0u8;
    for &t in thresholds {
        if val > t { q += 1; }
    }
    q
}

// ── Active signal tracker ──

#[derive(Clone)]
struct ActiveSignal {
    bar_created: usize,
    expiry: usize,
    weight: f64,
    direction: f64, // +1 or -1
}

impl ActiveSignal {
    fn is_alive(&self, current_bar: usize) -> bool {
        current_bar < self.bar_created + self.expiry
    }

    fn score(&self, current_bar: usize) -> f64 {
        if self.is_alive(current_bar) {
            self.weight * self.direction
        } else {
            0.0
        }
    }
}

// ── Sweep detection (inline, per bar) ──

struct TFState {
    candle_size: usize,
    prev_high: f64,
    prev_low: f64,
    high_active: bool,
    low_active: bool,
    high_cooldown: usize,
    low_cooldown: usize,
    bars_below: usize,
    bars_above: usize,
}

impl TFState {
    fn new(candle_size: usize) -> Self {
        TFState {
            candle_size,
            prev_high: f64::NEG_INFINITY,
            prev_low: f64::INFINITY,
            high_active: false,
            low_active: false,
            high_cooldown: 0,
            low_cooldown: 0,
            bars_below: 0,
            bars_above: 0,
        }
    }
}

// ── Strategy result ──

pub struct StrategyResult {
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
    pub avg_trade_bars: f64,
    pub max_consecutive_loss: i32,
    pub weekly_returns: Vec<f64>,
}

// ── Main strategy ──

pub fn run_sweep_strategy(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> StrategyResult {
    let n = closes.len();

    // Compute features
    let (features, daily_atr) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);

    // Quantile thresholds (rolling: use first 30 days as warmup, then update monthly)
    let warmup = 288 * 30; // 30 days
    let mut q_thresholds = compute_quantile_thresholds(&features, warmup.min(n));

    // Pattern definitions
    let tf_patterns = get_patterns();

    // TF states for sweep detection
    let candle_sizes = [BP_1H, BP_4H, BP_8H, BP_DAILY, BP_WEEKLY];
    let mut tf_states: Vec<TFState> = candle_sizes.iter().map(|&cs| TFState::new(cs)).collect();

    // Active signals
    let mut signals: Vec<ActiveSignal> = Vec::new();

    // Position state
    let mut condition: f64 = 0.0; // +1=long, -1=short, 0=flat
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut entry_bar: usize = 0;

    // Trailing stop state
    let mut trailing_active = false;
    let mut trailing_price: f64 = 0.0;

    // Accounting
    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut total_trades: i32 = 0;
    let mut wins: i32 = 0;
    let mut long_trades: i32 = 0;
    let mut short_trades: i32 = 0;
    let mut long_wins: i32 = 0;
    let mut short_wins: i32 = 0;
    let mut total_trade_bars: usize = 0;
    let mut consecutive_losses: i32 = 0;
    let mut max_consec_loss: i32 = 0;

    // Weekly tracking
    let mut weekly_returns: Vec<f64> = Vec::new();
    let mut week_start_balance = INITIAL_BALANCE;
    let bpw = 288 * 7;

    let start = MIN_BARS.max(warmup);

    for i in start..n {
        // Update quantile thresholds monthly
        if i > warmup && i % (288 * 30) == 0 {
            q_thresholds = compute_quantile_thresholds(&features, i);
        }

        if features[i][0].is_nan() || daily_atr[i].is_nan() || daily_atr[i] <= 0.0 {
            continue;
        }

        let reset_dist = 0.5 * daily_atr[i];

        // ── Sweep detection per TF ──
        for (tf_idx, (candle_size, weight, expiry, patterns)) in tf_patterns.iter().enumerate() {
            let cs = *candle_size;
            let state = &mut tf_states[tf_idx];

            // Update prev candle high/low at candle boundaries
            if i >= cs && i % cs == 0 {
                let prev_start = i - cs;
                let mut ph = f64::NEG_INFINITY;
                let mut pl = f64::INFINITY;
                for j in prev_start..i {
                    if highs[j] > ph { ph = highs[j]; }
                    if lows[j] < pl { pl = lows[j]; }
                }
                state.prev_high = ph;
                state.prev_low = pl;
                state.high_active = true;
                state.low_active = true;
                state.high_cooldown = 0;
                state.low_cooldown = 0;
                state.bars_below = 0;
                state.bars_above = 0;
            }

            if state.prev_high == f64::NEG_INFINITY { continue; }

            // High sweep detection
            if state.high_active && i > 0 && highs[i] > state.prev_high && highs[i-1] <= state.prev_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    // Check patterns
                    let feat = &features[i-1];
                    for pat in patterns.iter().filter(|p| p.direction > 0.0) {
                        let matched = pat.feature_indices.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| {
                            get_quantile(feat[fi], &q_thresholds[fi]) == q
                        });
                        if matched {
                            signals.push(ActiveSignal {
                                bar_created: i, expiry: *expiry,
                                weight: *weight, direction: pat.direction,
                            });
                        }
                    }
                    state.high_active = false;
                    state.high_cooldown = 0;
                }
            }

            // Low sweep detection
            if state.low_active && i > 0 && lows[i] < state.prev_low && lows[i-1] >= state.prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let feat = &features[i-1];
                    for pat in patterns.iter().filter(|p| p.direction < 0.0) {
                        let matched = pat.feature_indices.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| {
                            get_quantile(feat[fi], &q_thresholds[fi]) == q
                        });
                        if matched {
                            signals.push(ActiveSignal {
                                bar_created: i, expiry: *expiry,
                                weight: *weight, direction: pat.direction,
                            });
                        }
                    }
                    state.low_active = false;
                    state.low_cooldown = 0;
                }
            }

            // Reset logic
            if !state.high_active {
                if closes[i] < state.prev_high - reset_dist { state.bars_below += 1; } else { state.bars_below = 0; }
                state.high_cooldown += 1;
                if state.bars_below >= 12 { state.high_active = true; state.bars_below = 0; }
            }
            if !state.low_active {
                if closes[i] > state.prev_low + reset_dist { state.bars_above += 1; } else { state.bars_above = 0; }
                state.low_cooldown += 1;
                if state.bars_above >= 12 { state.low_active = true; state.bars_above = 0; }
            }
        }

        // ── Compute aggregate score ──
        let score: f64 = signals.iter().map(|s| s.score(i)).sum();

        // Clean expired signals periodically
        if i % 100 == 0 {
            signals.retain(|s| s.is_alive(i));
        }

        // ── Determine target direction ──
        let target_dir = if score > 0.0 { 1.0 } else if score < 0.0 { -1.0 } else { 0.0 };

        // ── Trailing stop check (when score == 0) ──
        if trailing_active && condition != 0.0 && notional > 0.0 {
            let hit = if condition > 0.0 {
                lows[i] <= trailing_price
            } else {
                highs[i] >= trailing_price
            };

            if hit {
                // Close position via trailing stop
                let exit_price = trailing_price;
                let pnl_pct = if condition > 0.0 {
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
                total_trade_bars += i - entry_bar;
                if condition > 0.0 { long_trades += 1; } else { short_trades += 1; }
                if pnl > 0.0 {
                    wins += 1;
                    if condition > 0.0 { long_wins += 1; } else { short_wins += 1; }
                    consecutive_losses = 0;
                } else {
                    consecutive_losses += 1;
                    if consecutive_losses > max_consec_loss { max_consec_loss = consecutive_losses; }
                }
                condition = 0.0;
                notional = 0.0;
                trailing_active = false;
            } else {
                // Update trailing stop
                if condition > 0.0 {
                    let new_trail = highs[i] - daily_atr[i] * TRAILING_ATR_MULT;
                    if new_trail > trailing_price { trailing_price = new_trail; }
                } else {
                    let new_trail = lows[i] + daily_atr[i] * TRAILING_ATR_MULT;
                    if new_trail < trailing_price { trailing_price = new_trail; }
                }
            }
        }

        // ── Position management ──
        if target_dir != 0.0 && target_dir != condition {
            // Deactivate trailing
            trailing_active = false;

            // Close existing position
            if condition != 0.0 && notional > 0.0 {
                let pnl_pct = if condition > 0.0 {
                    (closes[i] - entry_price) / entry_price * 100.0
                } else {
                    (entry_price - closes[i]) / entry_price * 100.0
                };
                let pnl = notional * pnl_pct / 100.0;
                let fee = notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl;
                total_fees += fee;
                total_trades += 1;
                total_trade_bars += i - entry_bar;
                if condition > 0.0 { long_trades += 1; } else { short_trades += 1; }
                if pnl > 0.0 {
                    wins += 1;
                    if condition > 0.0 { long_wins += 1; } else { short_wins += 1; }
                    consecutive_losses = 0;
                } else {
                    consecutive_losses += 1;
                    if consecutive_losses > max_consec_loss { max_consec_loss = consecutive_losses; }
                }
            }

            // Open new position
            let margin = balance * MARGIN_RATIO;
            if balance >= margin && margin > 0.0 {
                condition = target_dir;
                entry_price = closes[i];
                notional = margin * LEVERAGE;
                entry_bar = i;
                let fee = notional * TAKER_FEE;
                balance -= fee;
                total_fees += fee;
            } else {
                condition = 0.0;
                notional = 0.0;
            }
        }

        // ── Score == 0: activate trailing or close if losing ──
        if target_dir == 0.0 && condition != 0.0 && notional > 0.0 && !trailing_active {
            let current_pnl = if condition > 0.0 {
                (closes[i] - entry_price) / entry_price * 100.0
            } else {
                (entry_price - closes[i]) / entry_price * 100.0
            };

            if current_pnl > 0.0 {
                // In profit: activate trailing stop
                trailing_active = true;
                if condition > 0.0 {
                    trailing_price = closes[i] - daily_atr[i] * TRAILING_ATR_MULT;
                } else {
                    trailing_price = closes[i] + daily_atr[i] * TRAILING_ATR_MULT;
                }
            } else {
                // Losing: close immediately
                let pnl = notional * current_pnl / 100.0;
                let fee = notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl;
                total_fees += fee;
                total_trades += 1;
                total_trade_bars += i - entry_bar;
                if condition > 0.0 { long_trades += 1; } else { short_trades += 1; }
                consecutive_losses += 1;
                if consecutive_losses > max_consec_loss { max_consec_loss = consecutive_losses; }
                condition = 0.0;
                notional = 0.0;
                trailing_active = false;
            }
        }

        // ── DD tracking ──
        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }

        // ── Weekly return tracking ──
        if i > start && (i - start) % bpw == 0 {
            let wr = (balance - week_start_balance) / week_start_balance * 100.0;
            weekly_returns.push(wr);
            week_start_balance = balance;
        }
    }

    // Close remaining position
    if condition != 0.0 && notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
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
        if condition > 0.0 { long_trades += 1; } else { short_trades += 1; }
        if pnl > 0.0 {
            wins += 1;
            if condition > 0.0 { long_wins += 1; } else { short_wins += 1; }
        }
    }

    // Final weekly
    if balance != week_start_balance {
        let wr = (balance - week_start_balance) / week_start_balance * 100.0;
        weekly_returns.push(wr);
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if total_trades > 0 { wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };
    let avg_bars = if total_trades > 0 { total_trade_bars as f64 / total_trades as f64 } else { 0.0 };

    StrategyResult {
        net_pct, balance, total_trades, win_rate, max_drawdown: max_dd,
        total_pnl, total_fees, long_trades, short_trades, long_wins, short_wins,
        avg_trade_bars: avg_bars, max_consecutive_loss: max_consec_loss,
        weekly_returns,
    }
}

/// Sweep Candle Strategy — Mum kapanisi bazli, feature filtreli
/// TP/SL/Timeout yok. Sadece ters sinyal ile cikis.
/// Karar sadece 1H mum kapanisinda.

use crate::sweep_candle_analysis::FEATURE_NAMES;
use crate::sweep_miner::compute_5m_features;

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const TAKER_FEE: f64 = 0.0005;
const BP_1H: usize = 12;
const MIN_BARS: usize = 300;

// ── Quantile ──

fn quantile_thresholds(vals: &[f64]) -> [f64; 4] {
    let mut sorted: Vec<f64> = vals.iter().filter(|v| !v.is_nan()).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut th = [0.0_f64; 4];
    if sorted.is_empty() { return th; }
    for q in 0..4 {
        let idx = ((q + 1) as f64 / 5.0 * sorted.len() as f64) as usize;
        th[q] = sorted[idx.min(sorted.len() - 1)];
    }
    th
}

fn get_q(val: f64, th: &[f64; 4]) -> u8 {
    let mut q = 0u8;
    for &t in th { if val > t { q += 1; } }
    q
}

// ── Pattern match ──

struct Pattern {
    features: Vec<usize>,
    quantiles: Vec<u8>,
}

// LONG patterns (high continuation + low reversal)
fn long_patterns() -> Vec<Pattern> {
    vec![
        // High cont: CVD_micro(0)=Q5 AND Imbalance(5)=Q5 -> 97.8%, WF 8/8
        Pattern { features: vec![0, 5], quantiles: vec![4, 4] },
        // High cont: CVD_micro(0)=Q5 AND Vol_micro(3)=Q5 -> 98.4%, WF 8/8
        Pattern { features: vec![0, 3], quantiles: vec![4, 4] },
        // High cont: CVD_micro(0)=Q4 AND Imbalance(5)=Q4 -> 92.2%, WF 8/8
        Pattern { features: vec![0, 5], quantiles: vec![3, 3] },
        // Low rev: Vol_micro(3)=Q1 AND Imbalance(5)=Q5 -> 88.0%, WF 8/8
        Pattern { features: vec![3, 5], quantiles: vec![0, 4] },
        // Low rev: CVD_micro(0)=Q5 AND Imbalance(5)=Q5 -> 77.0%, WF 8/8
        Pattern { features: vec![0, 5], quantiles: vec![4, 4] },
    ]
}

// SHORT patterns (low continuation + high reversal)
fn short_patterns() -> Vec<Pattern> {
    vec![
        // Low cont: CVD_micro(0)=Q1 AND Vol_micro(3)=Q5 -> 99.6%, WF 8/8
        Pattern { features: vec![0, 3], quantiles: vec![0, 4] },
        // Low cont: CVD_micro(0)=Q1 AND Imbalance(5)=Q1 -> 97.5%, WF 8/8
        Pattern { features: vec![0, 5], quantiles: vec![0, 0] },
        // Low cont: CVD_micro(0)=Q2 AND Vol_micro(3)=Q4 -> 90.8%, WF 8/8
        Pattern { features: vec![0, 3], quantiles: vec![1, 3] },
        // High rev: CVD_micro(0)=Q1 AND Imbalance(5)=Q1 -> 74.3%, WF 8/8
        Pattern { features: vec![0, 5], quantiles: vec![0, 0] },
        // High rev: Vol_micro(3)=Q1 AND Imbalance(5)=Q1 -> 75.2%, WF 8/8
        Pattern { features: vec![3, 5], quantiles: vec![0, 0] },
    ]
}

fn check_patterns(feat: &[f64; 7], patterns: &[Pattern], q_th: &[[f64; 4]; 7]) -> bool {
    for p in patterns {
        let matched = p.features.iter().zip(p.quantiles.iter()).all(|(&fi, &q)| {
            !feat[fi].is_nan() && get_q(feat[fi], &q_th[fi]) == q
        });
        if matched { return true; }
    }
    false
}

// ── Result ──

pub struct CandleStrategyResult {
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
    pub avg_hold_hours: f64,
    pub max_consecutive_loss: i32,
    pub weekly_returns: Vec<f64>,
    pub trade_log: Vec<TradeRecord>,
}

#[derive(Clone)]
pub struct TradeRecord {
    pub candle_idx: usize,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_reason: String,
    pub pnl: f64,
    pub balance_after: f64,
    pub hours_held: f64,
}

// ── Main ──

pub fn run_candle_strategy(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> CandleStrategyResult {
    let n = closes.len();

    let (features, _) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);

    // Build 1H candles
    let n_candles = n / BP_1H;
    struct Candle { start: usize, open: f64, high: f64, low: f64, close: f64 }
    let mut candles: Vec<Candle> = Vec::with_capacity(n_candles);
    for c in 0..n_candles {
        let s = c * BP_1H;
        let e = s + BP_1H;
        let mut h = f64::NEG_INFINITY;
        let mut l = f64::INFINITY;
        for i in s..e { if highs[i] > h { h = highs[i]; } if lows[i] < l { l = lows[i]; } }
        candles.push(Candle { start: s, open: closes[s], high: h, low: l, close: closes[e - 1] });
    }

    // Rolling quantile thresholds (update monthly)
    let warmup_candles = 24 * 30; // 30 days of 1H candles
    let warmup_bars = warmup_candles * BP_1H;

    let mut q_th = [[0.0_f64; 4]; 7];
    {
        let end = warmup_bars.min(n);
        for fi in 0..7 {
            let vals: Vec<f64> = (MIN_BARS..end).filter_map(|i| {
                if features[i][fi].is_nan() { None } else { Some(features[i][fi]) }
            }).collect();
            q_th[fi] = quantile_thresholds(&vals);
        }
    }

    let l_pats = long_patterns();
    let s_pats = short_patterns();

    // State
    let mut cond: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut entry_candle: usize = 0;

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
    let mut total_hold: f64 = 0.0;
    let mut consec_loss: i32 = 0;
    let mut max_consec: i32 = 0;

    let mut trade_log: Vec<TradeRecord> = Vec::new();
    let mut weekly_returns: Vec<f64> = Vec::new();
    let mut week_start_bal = INITIAL_BALANCE;
    let candles_per_week = 24 * 7;

    let start_candle = warmup_candles.max(2);

    for ci in start_candle..candles.len() {
        // Monthly quantile update
        if ci % (24 * 30) == 0 && ci > warmup_candles {
            let end = ci * BP_1H;
            for fi in 0..7 {
                let vals: Vec<f64> = (MIN_BARS..end).filter_map(|i| {
                    if features[i][fi].is_nan() { None } else { Some(features[i][fi]) }
                }).collect();
                q_th[fi] = quantile_thresholds(&vals);
            }
        }

        let prev = &candles[ci - 1];
        let curr = &candles[ci];

        // Feature at candle close
        let feat_idx = curr.start + BP_1H - 1;
        if feat_idx >= n || features[feat_idx][0].is_nan() { continue; }
        let feat = &features[feat_idx];

        let is_green = curr.close > curr.open;
        let is_red = curr.close < curr.open;
        let swept_high = curr.high > prev.high;
        let swept_low = curr.low < prev.low;

        // Determine signal
        let mut long_signal = false;
        let mut short_signal = false;

        if swept_high && !swept_low {
            if curr.close > prev.high {
                // High continuation → check LONG patterns
                long_signal = check_patterns(feat, &l_pats, &q_th);
            } else if is_red {
                // High reversal → check SHORT patterns
                short_signal = check_patterns(feat, &s_pats, &q_th);
            }
        } else if swept_low && !swept_high {
            if curr.close < prev.low {
                // Low continuation → check SHORT patterns
                short_signal = check_patterns(feat, &s_pats, &q_th);
            } else if is_green {
                // Low reversal → check LONG patterns
                long_signal = check_patterns(feat, &l_pats, &q_th);
            }
        } else if swept_high && swept_low {
            // Outside bar
            if curr.close > prev.high {
                long_signal = check_patterns(feat, &l_pats, &q_th);
            } else if curr.close < prev.low {
                short_signal = check_patterns(feat, &s_pats, &q_th);
            } else if is_red {
                short_signal = check_patterns(feat, &s_pats, &q_th);
            } else if is_green {
                long_signal = check_patterns(feat, &l_pats, &q_th);
            }
        }

        // ── Position management ──
        let new_dir = if long_signal && !short_signal { 1.0 }
                      else if short_signal && !long_signal { -1.0 }
                      else { 0.0 };

        // Close if opposite signal
        if new_dir != 0.0 && new_dir != cond && cond != 0.0 && notional > 0.0 {
            let exit = curr.close;
            let pnl_pct = if cond > 0.0 {
                (exit - entry_price) / entry_price * 100.0
            } else {
                (entry_price - exit) / entry_price * 100.0
            };
            let pnl = notional * pnl_pct / 100.0;
            let fee = notional * TAKER_FEE;
            balance += pnl - fee;
            total_pnl += pnl;
            total_fees += fee;
            total_trades += 1;
            let hours = (ci - entry_candle) as f64;
            total_hold += hours;

            if cond > 0.0 { long_trades += 1; } else { short_trades += 1; }
            if pnl > 0.0 {
                wins += 1;
                if cond > 0.0 { long_wins += 1; } else { short_wins += 1; }
                consec_loss = 0;
            } else {
                consec_loss += 1;
                if consec_loss > max_consec { max_consec = consec_loss; }
            }

            let reason = if new_dir > 0.0 { "LONG sinyal" } else { "SHORT sinyal" };
            trade_log.push(TradeRecord {
                candle_idx: entry_candle, direction: if cond > 0.0 { "LONG".to_string() } else { "SHORT".to_string() },
                entry_price, exit_price: exit, exit_reason: reason.to_string(),
                pnl, balance_after: balance, hours_held: hours,
            });

            cond = 0.0;
            notional = 0.0;
        }

        // Open new position
        if new_dir != 0.0 && cond == 0.0 {
            let margin = balance * MARGIN_RATIO;
            if balance >= margin && margin > 0.0 {
                cond = new_dir;
                entry_price = curr.close;
                notional = margin * LEVERAGE;
                entry_candle = ci;
                let fee = notional * TAKER_FEE;
                balance -= fee;
                total_fees += fee;
            }
        }

        // If same direction signal while in position → hold (do nothing)

        // DD
        if balance > peak { peak = balance; }
        if peak > 0.0 { let dd = (peak - balance) / peak * 100.0; if dd > max_dd { max_dd = dd; } }

        // Weekly
        if ci > start_candle && (ci - start_candle) % candles_per_week == 0 {
            let wr = (balance - week_start_bal) / week_start_bal * 100.0;
            weekly_returns.push(wr);
            week_start_bal = balance;
        }
    }

    // Close remaining
    if cond != 0.0 && notional > 0.0 && !candles.is_empty() {
        let exit = candles.last().unwrap().close;
        let pnl_pct = if cond > 0.0 { (exit - entry_price) / entry_price * 100.0 }
                       else { (entry_price - exit) / entry_price * 100.0 };
        let pnl = notional * pnl_pct / 100.0;
        let fee = notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        total_trades += 1;
        if cond > 0.0 { long_trades += 1; } else { short_trades += 1; }
        if pnl > 0.0 { wins += 1; if cond > 0.0 { long_wins += 1; } else { short_wins += 1; } }
        trade_log.push(TradeRecord {
            candle_idx: entry_candle, direction: if cond > 0.0 { "LONG".to_string() } else { "SHORT".to_string() },
            entry_price, exit_price: exit, exit_reason: "END".to_string(),
            pnl, balance_after: balance, hours_held: (candles.len() - entry_candle) as f64,
        });
    }

    if balance != week_start_bal {
        weekly_returns.push((balance - week_start_bal) / week_start_bal * 100.0);
    }

    if balance > peak { peak = balance; }
    if peak > 0.0 { let dd = (peak-balance)/peak*100.0; if dd > max_dd { max_dd = dd; } }

    let net = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let wr = if total_trades > 0 { wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };
    let avg_h = if total_trades > 0 { total_hold / total_trades as f64 } else { 0.0 };

    CandleStrategyResult {
        net_pct: net, balance, total_trades, win_rate: wr, max_drawdown: max_dd,
        total_pnl, total_fees, long_trades, short_trades, long_wins, short_wins,
        avg_hold_hours: avg_h, max_consecutive_loss: max_consec,
        weekly_returns, trade_log,
    }
}

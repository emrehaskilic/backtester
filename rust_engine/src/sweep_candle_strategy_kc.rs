/// Sweep Candle Strategy + KC DCA/TP
///
/// Giris: 1H mum kapanisi + sweep + feature pattern match
/// TP: KC ust/alt bandina ulasinca kar al (kademeli)
/// DCA: KC ters bandina dusunce pozisyon buyut
/// Cikis: TP ile kapatma VEYA ters sweep sinyali
/// KC lagged (i-1) — look-ahead yok

use crate::sweep_candle_analysis::FEATURE_NAMES;
use crate::sweep_miner::compute_5m_features;

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const MAKER_FEE: f64 = 0.0002;
const TAKER_FEE: f64 = 0.0005;
const BP_1H: usize = 12;
const MIN_BARS: usize = 300;

// KC params
const KC_LENGTH: usize = 20;
const KC_MULT: f64 = 2.0;
const KC_ATR_PERIOD: usize = 14;

// DCA
const MAX_DCA: i32 = 3;
const DCA_MULTS: [f64; 3] = [1.0, 0.7, 0.5]; // kademeli margin

// TP
const TP_CLOSE_PCT: f64 = 0.5; // pozisyonun %50'sini kapat

// ── KC hesapla ──

fn compute_kc(closes: &[f64], highs: &[f64], lows: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = closes.len();
    // EMA
    let k = 2.0 / (KC_LENGTH as f64 + 1.0);
    let mut ema = vec![f64::NAN; n];
    ema[0] = closes[0];
    for i in 1..n {
        let prev = if ema[i-1].is_nan() { closes[i] } else { ema[i-1] };
        ema[i] = closes[i] * k + prev * (1.0 - k);
    }
    // ATR
    let mut tr = vec![0.0_f64; n];
    tr[0] = highs[0] - lows[0];
    for i in 1..n {
        tr[i] = (highs[i]-lows[i]).max((highs[i]-closes[i-1]).abs()).max((lows[i]-closes[i-1]).abs());
    }
    let mut atr = vec![f64::NAN; n];
    if n > KC_ATR_PERIOD {
        let mut s = 0.0;
        for i in 1..=KC_ATR_PERIOD { s += tr[i]; }
        atr[KC_ATR_PERIOD] = s / KC_ATR_PERIOD as f64;
        for i in (KC_ATR_PERIOD+1)..n {
            atr[i] = (atr[i-1]*(KC_ATR_PERIOD as f64-1.0)+tr[i])/KC_ATR_PERIOD as f64;
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

// ── Quantile ──

fn quantile_thresholds(vals: &[f64]) -> [f64; 4] {
    let mut sorted: Vec<f64> = vals.iter().filter(|v| !v.is_nan()).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut th = [0.0_f64; 4];
    if sorted.is_empty() { return th; }
    for q in 0..4 {
        let idx = ((q+1) as f64 / 5.0 * sorted.len() as f64) as usize;
        th[q] = sorted[idx.min(sorted.len()-1)];
    }
    th
}

fn get_q(val: f64, th: &[f64; 4]) -> u8 {
    let mut q = 0u8;
    for &t in th { if val > t { q += 1; } }
    q
}

// ── Patterns ──

struct Pattern { features: Vec<usize>, quantiles: Vec<u8> }

fn long_patterns() -> Vec<Pattern> {
    vec![
        Pattern { features: vec![0, 5], quantiles: vec![4, 4] }, // CVD Q5 + Imb Q5
        Pattern { features: vec![0, 3], quantiles: vec![4, 4] }, // CVD Q5 + Vol Q5
        Pattern { features: vec![0, 5], quantiles: vec![3, 3] }, // CVD Q4 + Imb Q4
        Pattern { features: vec![3, 5], quantiles: vec![0, 4] }, // Vol Q1 + Imb Q5 (low rev)
    ]
}

fn short_patterns() -> Vec<Pattern> {
    vec![
        Pattern { features: vec![0, 3], quantiles: vec![0, 4] }, // CVD Q1 + Vol Q5
        Pattern { features: vec![0, 5], quantiles: vec![0, 0] }, // CVD Q1 + Imb Q1
        Pattern { features: vec![0, 3], quantiles: vec![1, 3] }, // CVD Q2 + Vol Q4
        Pattern { features: vec![3, 5], quantiles: vec![0, 0] }, // Vol Q1 + Imb Q1 (high rev)
    ]
}

fn check_patterns(feat: &[f64; 7], pats: &[Pattern], q_th: &[[f64; 4]; 7]) -> bool {
    for p in pats {
        if p.features.iter().zip(p.quantiles.iter()).all(|(&fi, &q)| {
            !feat[fi].is_nan() && get_q(feat[fi], &q_th[fi]) == q
        }) { return true; }
    }
    false
}

// ── Result ──

pub struct KCStrategyResult {
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
    pub signal_close_count: i32,
    pub dca_count: i32,
    pub avg_hold_hours: f64,
    pub max_consecutive_loss: i32,
    pub weekly_returns: Vec<f64>,
}

// ── Main ──

pub fn run_kc_strategy(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> KCStrategyResult {
    let n = closes.len();

    let (features, _) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);
    let (kc_upper, kc_lower) = compute_kc(closes, highs, lows);

    // 1H candles
    let n_candles = n / BP_1H;
    struct Candle { start: usize, open: f64, high: f64, low: f64, close: f64 }
    let mut candles: Vec<Candle> = Vec::with_capacity(n_candles);
    for c in 0..n_candles {
        let s = c * BP_1H;
        let e = s + BP_1H;
        let mut ch = f64::NEG_INFINITY;
        let mut cl = f64::INFINITY;
        for i in s..e { if highs[i] > ch { ch = highs[i]; } if lows[i] < cl { cl = lows[i]; } }
        candles.push(Candle { start: s, open: closes[s], high: ch, low: cl, close: closes[e-1] });
    }

    // Rolling quantiles
    let warmup_candles = 24 * 30;
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

    // Position state
    let mut cond: f64 = 0.0;
    let mut avg_entry: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;
    let mut entry_candle: usize = 0;

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
    let mut signal_close: i32 = 0;
    let mut dca_total: i32 = 0;
    let mut total_hold: f64 = 0.0;
    let mut consec_loss: i32 = 0;
    let mut max_consec: i32 = 0;

    let mut weekly_returns: Vec<f64> = Vec::new();
    let mut week_start_bal = INITIAL_BALANCE;
    let candles_per_week = 24 * 7;

    let start_candle = warmup_candles.max(2);
    let tfr = MAKER_FEE + TAKER_FEE;

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

        // ── KC DCA/TP check on 5m bars (bar-by-bar within candle) ──
        if cond != 0.0 && notional > 0.0 {
            let bar_start = candles[ci].start;
            let bar_end = bar_start + BP_1H;

            for bi in bar_start..bar_end {
                if bi < 1 || kc_upper[bi-1].is_nan() || kc_lower[bi-1].is_nan() { continue; }
                let ku = kc_upper[bi-1]; // lagged
                let kl = kc_lower[bi-1];

                if cond > 0.0 {
                    // LONG: DCA at KC lower
                    if dca_fills < MAX_DCA && lows[bi] <= kl {
                        let mult = DCA_MULTS[dca_fills as usize];
                        let dm = balance * MARGIN_RATIO * mult;
                        if balance >= dm && dm > 0.0 {
                            let step = dm * LEVERAGE;
                            let old = notional;
                            notional += step;
                            avg_entry = (avg_entry * old + kl * step) / notional;
                            dca_fills += 1;
                            dca_total += 1;
                            let fee = step * MAKER_FEE;
                            balance -= fee;
                            total_fees += fee;
                        }
                    }
                    // LONG: TP at KC upper
                    else if dca_fills > 0 && highs[bi] >= ku {
                        let be = avg_entry * (1.0 + tfr);
                        if ku > be {
                            let close_amt = notional * TP_CLOSE_PCT;
                            let pp = (ku - avg_entry) / avg_entry * 100.0;
                            let pnl = close_amt * pp / 100.0;
                            let fee = close_amt * MAKER_FEE;
                            balance += pnl - fee;
                            total_pnl += pnl;
                            total_fees += fee;
                            notional -= close_amt;
                            dca_fills = (dca_fills - 1).max(0);
                            total_trades += 1;
                            tp_count += 1;
                            long_trades += 1;
                            if pnl > 0.0 { wins += 1; long_wins += 1; consec_loss = 0; }
                            else { consec_loss += 1; if consec_loss > max_consec { max_consec = consec_loss; } }
                            if notional < 1.0 { cond = 0.0; notional = 0.0; }
                        }
                    }
                } else {
                    // SHORT: DCA at KC upper
                    if dca_fills < MAX_DCA && highs[bi] >= ku {
                        let mult = DCA_MULTS[dca_fills as usize];
                        let dm = balance * MARGIN_RATIO * mult;
                        if balance >= dm && dm > 0.0 {
                            let step = dm * LEVERAGE;
                            let old = notional;
                            notional += step;
                            avg_entry = (avg_entry * old + ku * step) / notional;
                            dca_fills += 1;
                            dca_total += 1;
                            let fee = step * MAKER_FEE;
                            balance -= fee;
                            total_fees += fee;
                        }
                    }
                    // SHORT: TP at KC lower
                    else if dca_fills > 0 && lows[bi] <= kl {
                        let be = avg_entry * (1.0 - tfr);
                        if kl < be {
                            let close_amt = notional * TP_CLOSE_PCT;
                            let pp = (avg_entry - kl) / avg_entry * 100.0;
                            let pnl = close_amt * pp / 100.0;
                            let fee = close_amt * MAKER_FEE;
                            balance += pnl - fee;
                            total_pnl += pnl;
                            total_fees += fee;
                            notional -= close_amt;
                            dca_fills = (dca_fills - 1).max(0);
                            total_trades += 1;
                            tp_count += 1;
                            short_trades += 1;
                            if pnl > 0.0 { wins += 1; short_wins += 1; consec_loss = 0; }
                            else { consec_loss += 1; if consec_loss > max_consec { max_consec = consec_loss; } }
                            if notional < 1.0 { cond = 0.0; notional = 0.0; }
                        }
                    }
                }
            }
        }

        // ── 1H candle signal check ──
        if ci < 1 { continue; }
        let prev = &candles[ci-1];
        let curr = &candles[ci];

        let feat_idx = curr.start + BP_1H - 1;
        if feat_idx >= n || features[feat_idx][0].is_nan() { continue; }
        let feat = &features[feat_idx];

        let is_green = curr.close > curr.open;
        let is_red = curr.close < curr.open;
        let swept_high = curr.high > prev.high;
        let swept_low = curr.low < prev.low;

        let mut long_signal = false;
        let mut short_signal = false;

        if swept_high && !swept_low {
            if curr.close > prev.high { long_signal = check_patterns(feat, &l_pats, &q_th); }
            else if is_red { short_signal = check_patterns(feat, &s_pats, &q_th); }
        } else if swept_low && !swept_high {
            if curr.close < prev.low { short_signal = check_patterns(feat, &s_pats, &q_th); }
            else if is_green { long_signal = check_patterns(feat, &l_pats, &q_th); }
        } else if swept_high && swept_low {
            if curr.close > prev.high { long_signal = check_patterns(feat, &l_pats, &q_th); }
            else if curr.close < prev.low { short_signal = check_patterns(feat, &s_pats, &q_th); }
            else if is_red { short_signal = check_patterns(feat, &s_pats, &q_th); }
            else if is_green { long_signal = check_patterns(feat, &l_pats, &q_th); }
        }

        let new_dir = if long_signal && !short_signal { 1.0 }
                      else if short_signal && !long_signal { -1.0 }
                      else { 0.0 };

        // Close on opposite signal (reversal)
        if new_dir != 0.0 && new_dir != cond && cond != 0.0 && notional > 0.0 {
            let exit = curr.close;
            let pp = if cond > 0.0 { (exit - avg_entry) / avg_entry * 100.0 }
                     else { (avg_entry - exit) / avg_entry * 100.0 };
            let pnl = notional * pp / 100.0;
            let fee = notional * TAKER_FEE;
            balance += pnl - fee;
            total_pnl += pnl;
            total_fees += fee;
            total_trades += 1;
            signal_close += 1;
            total_hold += (ci - entry_candle) as f64;
            if cond > 0.0 { long_trades += 1; } else { short_trades += 1; }
            if pnl > 0.0 { wins += 1; if cond > 0.0 { long_wins += 1; } else { short_wins += 1; } consec_loss = 0; }
            else { consec_loss += 1; if consec_loss > max_consec { max_consec = consec_loss; } }
            cond = 0.0; notional = 0.0;
        }

        // Open new position
        if new_dir != 0.0 && cond == 0.0 {
            let margin = balance * MARGIN_RATIO;
            if balance >= margin && margin > 0.0 {
                cond = new_dir;
                avg_entry = curr.close;
                notional = margin * LEVERAGE;
                dca_fills = 0;
                entry_candle = ci;
                let fee = notional * TAKER_FEE;
                balance -= fee;
                total_fees += fee;
            }
        }

        // DD
        if balance > peak { peak = balance; }
        if peak > 0.0 { let dd = (peak-balance)/peak*100.0; if dd > max_dd { max_dd = dd; } }

        // Weekly
        if ci > start_candle && (ci - start_candle) % candles_per_week == 0 {
            weekly_returns.push((balance - week_start_bal) / week_start_bal * 100.0);
            week_start_bal = balance;
        }
    }

    // Close remaining
    if cond != 0.0 && notional > 0.0 && n > 0 {
        let exit = closes[n-1];
        let pp = if cond > 0.0 { (exit-avg_entry)/avg_entry*100.0 } else { (avg_entry-exit)/avg_entry*100.0 };
        let pnl = notional * pp / 100.0;
        let fee = notional * TAKER_FEE;
        balance += pnl - fee; total_pnl += pnl; total_fees += fee; total_trades += 1;
        if cond > 0.0 { long_trades += 1; } else { short_trades += 1; }
        if pnl > 0.0 { wins += 1; if cond > 0.0 { long_wins += 1; } else { short_wins += 1; } }
    }

    if balance != week_start_bal { weekly_returns.push((balance-week_start_bal)/week_start_bal*100.0); }
    if balance > peak { peak = balance; }
    if peak > 0.0 { let dd=(peak-balance)/peak*100.0; if dd>max_dd { max_dd=dd; } }

    let net = (balance-INITIAL_BALANCE)/INITIAL_BALANCE*100.0;
    let wr = if total_trades > 0 { wins as f64/total_trades as f64*100.0 } else { 0.0 };
    let avg_h = if (total_trades - tp_count) > 0 { total_hold / (total_trades - tp_count) as f64 } else { 0.0 };

    KCStrategyResult {
        net_pct: net, balance, total_trades, win_rate: wr, max_drawdown: max_dd,
        total_pnl, total_fees, long_trades, short_trades, long_wins, short_wins,
        tp_count, signal_close_count: signal_close, dca_count: dca_total,
        avg_hold_hours: avg_h, max_consecutive_loss: max_consec, weekly_returns,
    }
}

/// Sweep KC Multi-TF Strategy (Parametric)
///
/// 1H sweep + candle close → yön belirleme (5m data'dan 1H aggregate)
/// 3m KC → DCA/TP işlem (3m data)
/// Haftalık kasa sıfırlama (her hafta 1000 USDT)
///
/// Feature pattern filtresi: 8/8 WF geçen pattern'ler (sabit, domain prior)
/// KC lagged (i-1) — look-ahead yok
/// KC params dışarıdan geçilebilir (optimizer için)

use crate::sweep_miner::compute_5m_features;

// ── Constants ──

const INITIAL_BALANCE: f64 = 1000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 0.10; // kasa %10
const TAKER_FEE: f64 = 0.0004; // 0.04% taker
const MAKER_FEE: f64 = 0.0002; // 0.02% maker
const BP_1H: usize = 12; // 5m bars per 1H
const MIN_BARS: usize = 300;
pub const CANDLES_PER_WEEK: usize = 24 * 7; // 168 1H candles

// ── KC Params (configurable) ──

#[derive(Clone, Debug)]
pub struct KCParams {
    pub kc_length: usize,
    pub kc_mult: f64,
    pub kc_atr_period: usize,
    pub max_dca: i32,
    pub dca_scale: f64,      // geometric: DCA_k margin = base * scale^k
    pub tp_levels: i32,      // 1=full close, 2-3=graduated
    pub tp_first_pct: f64,   // first TP closes this % of position (0.25-1.0)
}

impl Default for KCParams {
    fn default() -> Self {
        Self {
            kc_length: 20, kc_mult: 2.0, kc_atr_period: 14,
            max_dca: i32::MAX, dca_scale: 1.0,
            tp_levels: 1, tp_first_pct: 1.0,
        }
    }
}

// ── KC on 3m data (parametric) ──

pub fn compute_kc_3m(
    closes: &[f64], highs: &[f64], lows: &[f64],
    kc_length: usize, kc_mult: f64, kc_atr_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = closes.len();
    let k = 2.0 / (kc_length as f64 + 1.0);
    let mut ema = vec![f64::NAN; n];
    ema[0] = closes[0];
    for i in 1..n {
        let prev = if ema[i - 1].is_nan() { closes[i] } else { ema[i - 1] };
        ema[i] = closes[i] * k + prev * (1.0 - k);
    }

    let mut tr = vec![0.0_f64; n];
    tr[0] = highs[0] - lows[0];
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }

    let mut atr = vec![f64::NAN; n];
    if n > kc_atr_period {
        let mut s = 0.0;
        for i in 1..=kc_atr_period { s += tr[i]; }
        atr[kc_atr_period] = s / kc_atr_period as f64;
        for i in (kc_atr_period + 1)..n {
            atr[i] = (atr[i - 1] * (kc_atr_period as f64 - 1.0) + tr[i]) / kc_atr_period as f64;
        }
    }

    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if !ema[i].is_nan() && !atr[i].is_nan() {
            upper[i] = ema[i] + atr[i] * kc_mult;
            lower[i] = ema[i] - atr[i] * kc_mult;
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

// ── Patterns (8/8 WF validated, domain prior — sabit) ──

struct Pattern { features: [usize; 2], quantiles: [u8; 2] }

fn long_patterns() -> Vec<Pattern> {
    vec![
        Pattern { features: [0, 5], quantiles: [4, 4] },
        Pattern { features: [0, 3], quantiles: [4, 4] },
        Pattern { features: [0, 5], quantiles: [3, 3] },
        Pattern { features: [3, 5], quantiles: [0, 4] },
    ]
}

fn short_patterns() -> Vec<Pattern> {
    vec![
        Pattern { features: [0, 3], quantiles: [0, 4] },
        Pattern { features: [0, 5], quantiles: [0, 0] },
        Pattern { features: [0, 3], quantiles: [1, 3] },
        Pattern { features: [3, 5], quantiles: [0, 0] },
    ]
}

fn check_patterns(feat: &[f64; 7], pats: &[Pattern], q_th: &[[f64; 4]; 7]) -> bool {
    for p in pats {
        let (f0, f1) = (p.features[0], p.features[1]);
        if !feat[f0].is_nan() && !feat[f1].is_nan()
            && get_q(feat[f0], &q_th[f0]) == p.quantiles[0]
            && get_q(feat[f1], &q_th[f1]) == p.quantiles[1]
        { return true; }
    }
    false
}

// ── 5m → 1H candle ──

pub struct Candle1H {
    pub start_5m: usize,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub timestamp: u64,
}

pub fn build_1h_candles(
    closes: &[f64], highs: &[f64], lows: &[f64], timestamps: &[u64],
) -> Vec<Candle1H> {
    let n = closes.len();
    let n_candles = n / BP_1H;
    let mut candles = Vec::with_capacity(n_candles);
    for c in 0..n_candles {
        let s = c * BP_1H;
        let e = s + BP_1H;
        let mut ch = f64::NEG_INFINITY;
        let mut cl = f64::INFINITY;
        for i in s..e {
            if highs[i] > ch { ch = highs[i]; }
            if lows[i] < cl { cl = lows[i]; }
        }
        candles.push(Candle1H {
            start_5m: s, open: closes[s], high: ch, low: cl,
            close: closes[e - 1], timestamp: timestamps[e - 1],
        });
    }
    candles
}

// ── 3m timestamp → bar index ──

pub fn find_3m_bar_at_time(ts_3m: &[u64], target_ts: u64) -> usize {
    match ts_3m.binary_search(&target_ts) {
        Ok(i) => i,
        Err(i) => if i == 0 { 0 } else { i - 1 },
    }
}

// ── Direction ──

#[derive(Clone, Copy, PartialEq)]
enum Direction { Long, Short, Flat }

// ── Trade log ──

#[derive(Clone, Debug)]
pub struct TradeLog {
    pub entry_price: f64,
    pub exit_price: f64,
    pub direction: i8,
    pub pnl: f64,
    pub fee: f64,
    pub entry_ts: u64,
    pub exit_ts: u64,
    pub exit_reason: String,
    pub dca_count: i32,
}

// ── Week Result (tek hafta sonucu) ──

#[derive(Clone, Debug)]
pub struct WeekResult {
    pub pnl: f64,
    pub pnl_pct: f64,
    pub trades: i32,
    pub wins: i32,
    pub max_dd: f64,
    pub tp_count: i32,
    pub signal_close_count: i32,
    pub dca_count: i32,
}

// ── Full Result ──

pub struct MultiTFResult {
    pub total_weeks: usize,
    pub weekly_pnl: Vec<f64>,
    pub weekly_pnl_pct: Vec<f64>,
    pub weekly_trades: Vec<i32>,
    pub weekly_wins: Vec<i32>,
    pub weekly_max_dd: Vec<f64>,
    pub total_trades: i32,
    pub total_wins: i32,
    pub win_rate: f64,
    pub avg_weekly_pnl: f64,
    pub avg_weekly_pnl_pct: f64,
    pub median_weekly_pnl_pct: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub positive_weeks: usize,
    pub negative_weeks: usize,
    pub best_week_pct: f64,
    pub worst_week_pct: f64,
    pub max_drawdown_within_week: f64,
    pub total_dca_count: i32,
    pub total_tp_count: i32,
    pub total_signal_close: i32,
    pub avg_hold_bars_3m: f64,
    pub max_consecutive_loss_weeks: i32,
    pub trades: Vec<TradeLog>,
}

// ── Precomputed data (optimizer'ın tekrar tekrar hesaplamaması için) ──

pub struct PrecomputedData {
    pub features: Vec<[f64; 7]>,
    pub candles: Vec<Candle1H>,
    pub n5: usize,
    pub n3: usize,
    pub closes_3m: Vec<f64>,
    pub highs_3m: Vec<f64>,
    pub lows_3m: Vec<f64>,
    pub timestamps_3m: Vec<u64>,
    pub closes_5m: Vec<f64>,
}

impl PrecomputedData {
    pub fn new(
        closes_5m: &[f64], highs_5m: &[f64], lows_5m: &[f64],
        buy_vol_5m: &[f64], sell_vol_5m: &[f64], oi_5m: &[f64],
        timestamps_5m: &[u64],
        closes_3m: &[f64], highs_3m: &[f64], lows_3m: &[f64],
        timestamps_3m: &[u64],
    ) -> Self {
        let (features, _) = compute_5m_features(closes_5m, highs_5m, lows_5m, buy_vol_5m, sell_vol_5m, oi_5m);
        let candles = build_1h_candles(closes_5m, highs_5m, lows_5m, timestamps_5m);
        Self {
            features, candles,
            n5: closes_5m.len(), n3: closes_3m.len(),
            closes_3m: closes_3m.to_vec(),
            highs_3m: highs_3m.to_vec(),
            lows_3m: lows_3m.to_vec(),
            timestamps_3m: timestamps_3m.to_vec(),
            closes_5m: closes_5m.to_vec(),
        }
    }
}

// ── Run strategy on a range of 1H candles, return week results ──
/// candle_start..candle_end aralığında strateji çalıştır
/// KC pre-computed olarak gelir (tüm 3m data üzerinden, causal)
/// Quantile thresholds candle_start'a kadar olan data'dan hesaplanır

pub fn run_candle_range(
    data: &PrecomputedData,
    kc_upper: &[f64],
    kc_lower: &[f64],
    params: &KCParams,
    candle_start: usize,
    candle_end: usize,
) -> Vec<WeekResult> {
    let n5 = data.n5;
    let n3 = data.n3;
    let features = &data.features;
    let candles = &data.candles;

    let l_pats = long_patterns();
    let s_pats = short_patterns();

    // Quantile: candle_start'a kadar olan data'dan (look-ahead yok)
    let warmup_bars = candle_start * BP_1H;
    let mut q_th = [[0.0_f64; 4]; 7];
    {
        let end = warmup_bars.min(n5);
        for fi in 0..7 {
            let vals: Vec<f64> = (MIN_BARS..end)
                .filter_map(|i| if !features[i][fi].is_nan() { Some(features[i][fi]) } else { None })
                .collect();
            q_th[fi] = quantile_thresholds(&vals);
        }
    }

    let mut direction = Direction::Flat;
    let mut avg_entry: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut dca_fills: i32 = 0;
    let mut tp_fills: i32 = 0;
    let mut entry_bar_3m: usize = 0;

    let mut week_balance = INITIAL_BALANCE;
    let mut week_peak = INITIAL_BALANCE;
    let mut week_max_dd: f64 = 0.0;
    let mut week_trades: i32 = 0;
    let mut week_wins: i32 = 0;
    let mut week_tp: i32 = 0;
    let mut week_signal_close: i32 = 0;
    let mut week_dca: i32 = 0;
    let mut week_start_candle = candle_start;

    let mut weeks: Vec<WeekResult> = Vec::new();

    for ci in candle_start..candle_end {
        // Monthly quantile update (rolling, causal)
        if ci > 0 && ci % (24 * 30) == 0 {
            let end = (ci * BP_1H).min(n5);
            for fi in 0..7 {
                let vals: Vec<f64> = (MIN_BARS..end)
                    .filter_map(|i| if !features[i][fi].is_nan() { Some(features[i][fi]) } else { None })
                    .collect();
                q_th[fi] = quantile_thresholds(&vals);
            }
        }

        // ── Weekly reset ──
        if ci > week_start_candle && (ci - week_start_candle) >= CANDLES_PER_WEEK {
            // Close open position
            if direction != Direction::Flat && notional > 0.0 {
                let exit_3m = find_3m_bar_at_time(&data.timestamps_3m, candles[ci].timestamp);
                let exit_price = if exit_3m < n3 { data.closes_3m[exit_3m] } else { data.closes_5m[candles[ci].start_5m + BP_1H - 1] };
                let pp = if direction == Direction::Long {
                    (exit_price - avg_entry) / avg_entry * 100.0
                } else {
                    (avg_entry - exit_price) / avg_entry * 100.0
                };
                let pnl = notional * pp / 100.0;
                let fee = notional * TAKER_FEE;
                week_balance += pnl - fee;
                week_trades += 1;
                if pnl > 0.0 { week_wins += 1; }
                direction = Direction::Flat;
                notional = 0.0;
                dca_fills = 0;
                tp_fills = 0;
            }

            weeks.push(WeekResult {
                pnl: week_balance - INITIAL_BALANCE,
                pnl_pct: (week_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0,
                trades: week_trades, wins: week_wins, max_dd: week_max_dd,
                tp_count: week_tp, signal_close_count: week_signal_close, dca_count: week_dca,
            });

            week_balance = INITIAL_BALANCE;
            week_peak = INITIAL_BALANCE;
            week_max_dd = 0.0;
            week_trades = 0;
            week_wins = 0;
            week_tp = 0;
            week_signal_close = 0;
            week_dca = 0;
            week_start_candle = ci;
        }

        // ── 3m KC DCA/TP ──
        if direction != Direction::Flat && notional > 0.0 {
            let candle_ts_start = candles[ci].timestamp.saturating_sub(3_600_000);
            let candle_ts_end = candles[ci].timestamp;
            let bar3_start = find_3m_bar_at_time(&data.timestamps_3m, candle_ts_start);
            let bar3_end = find_3m_bar_at_time(&data.timestamps_3m, candle_ts_end);

            for bi in bar3_start..=bar3_end.min(n3 - 1) {
                if bi < 1 || kc_upper[bi - 1].is_nan() || kc_lower[bi - 1].is_nan() { continue; }
                let ku = kc_upper[bi - 1];
                let kl = kc_lower[bi - 1];

                // DCA entry price = KC band price
                // TP exit price = KC band price
                let is_long = direction == Direction::Long;
                let dca_hit = if is_long { data.lows_3m[bi] <= kl } else { data.highs_3m[bi] >= ku };
                let tp_hit = if is_long { data.highs_3m[bi] >= ku } else { data.lows_3m[bi] <= kl };
                let dca_price = if is_long { kl } else { ku };
                let tp_price = if is_long { ku } else { kl };

                if dca_hit && dca_fills < params.max_dca {
                    // Graduated DCA: margin = base * scale^dca_fills
                    let scale_mult = params.dca_scale.powi(dca_fills);
                    let dm = week_balance * MARGIN_RATIO * scale_mult;
                    if week_balance >= dm && dm > 0.0 {
                        let step = dm * LEVERAGE;
                        let old = notional;
                        notional += step;
                        avg_entry = (avg_entry * old + dca_price * step) / notional;
                        dca_fills += 1;
                        week_dca += 1;
                        let fee = step * MAKER_FEE;
                        week_balance -= fee;
                    }
                } else if tp_hit && notional > 0.0 {
                    let be = if is_long {
                        avg_entry * (1.0 + TAKER_FEE + MAKER_FEE)
                    } else {
                        avg_entry * (1.0 - TAKER_FEE - MAKER_FEE)
                    };
                    let in_profit = if is_long { tp_price > be } else { tp_price < be };

                    if in_profit {
                        // Graduated TP: determine close amount
                        let close_pct = if tp_fills + 1 >= params.tp_levels {
                            1.0 // last TP level = close everything
                        } else {
                            params.tp_first_pct // partial close
                        };
                        let close_amt = notional * close_pct;

                        let pp = if is_long {
                            (tp_price - avg_entry) / avg_entry * 100.0
                        } else {
                            (avg_entry - tp_price) / avg_entry * 100.0
                        };
                        let pnl = close_amt * pp / 100.0;
                        let fee = close_amt * MAKER_FEE;
                        week_balance += pnl - fee;
                        week_trades += 1;
                        week_tp += 1;
                        if pnl > 0.0 { week_wins += 1; }

                        notional -= close_amt;
                        tp_fills += 1;

                        if notional < 1.0 || tp_fills >= params.tp_levels {
                            // Fully closed
                            direction = Direction::Flat;
                            notional = 0.0;
                            dca_fills = 0;
                            tp_fills = 0;
                            break;
                        }
                        // Partial close — position still open, reset DCA counter
                        // so new DCA can fire from updated avg_entry
                    }
                }

                if direction == Direction::Flat { break; }

                if week_balance > week_peak { week_peak = week_balance; }
                if week_peak > 0.0 {
                    let dd = (week_peak - week_balance) / week_peak * 100.0;
                    if dd > week_max_dd { week_max_dd = dd; }
                }
            }
        }

        // ── 1H signal ──
        if ci < 1 { continue; }
        let prev = &candles[ci - 1];
        let curr = &candles[ci];

        let feat_idx = curr.start_5m + BP_1H - 1;
        if feat_idx >= n5 || features[feat_idx][0].is_nan() { continue; }
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

        let new_dir = if long_signal && !short_signal { Direction::Long }
                      else if short_signal && !long_signal { Direction::Short }
                      else { Direction::Flat };

        // Close on reversal
        if new_dir != Direction::Flat && new_dir != direction && direction != Direction::Flat && notional > 0.0 {
            let exit_3m_idx = find_3m_bar_at_time(&data.timestamps_3m, curr.timestamp);
            let exit_price = if exit_3m_idx < n3 { data.closes_3m[exit_3m_idx] } else { curr.close };
            let pp = if direction == Direction::Long {
                (exit_price - avg_entry) / avg_entry * 100.0
            } else {
                (avg_entry - exit_price) / avg_entry * 100.0
            };
            let pnl = notional * pp / 100.0;
            let fee = notional * TAKER_FEE;
            week_balance += pnl - fee;
            week_trades += 1;
            week_signal_close += 1;
            if pnl > 0.0 { week_wins += 1; }
            direction = Direction::Flat;
            notional = 0.0;
            dca_fills = 0;
            tp_fills = 0;
        }

        // Open new
        if new_dir != Direction::Flat && direction == Direction::Flat {
            let margin = week_balance * MARGIN_RATIO;
            if week_balance >= margin && margin > 0.0 {
                let entry_3m_idx = find_3m_bar_at_time(&data.timestamps_3m, curr.timestamp);
                let entry_price = if entry_3m_idx < n3 { data.closes_3m[entry_3m_idx] } else { curr.close };
                direction = new_dir;
                avg_entry = entry_price;
                notional = margin * LEVERAGE;
                dca_fills = 0;
                tp_fills = 0;
                entry_bar_3m = entry_3m_idx;
                let fee = notional * TAKER_FEE;
                week_balance -= fee;
            }
        }

        if week_balance > week_peak { week_peak = week_balance; }
        if week_peak > 0.0 {
            let dd = (week_peak - week_balance) / week_peak * 100.0;
            if dd > week_max_dd { week_max_dd = dd; }
        }
    }

    // Final partial week
    if direction != Direction::Flat && notional > 0.0 && n3 > 0 {
        let exit_price = data.closes_3m[n3 - 1];
        let pp = if direction == Direction::Long {
            (exit_price - avg_entry) / avg_entry * 100.0
        } else {
            (avg_entry - exit_price) / avg_entry * 100.0
        };
        let pnl = notional * pp / 100.0;
        let fee = notional * TAKER_FEE;
        week_balance += pnl - fee;
        week_trades += 1;
        if pnl > 0.0 { week_wins += 1; }
    }

    if week_balance > week_peak { week_peak = week_balance; }
    if week_peak > 0.0 {
        let dd = (week_peak - week_balance) / week_peak * 100.0;
        if dd > week_max_dd { week_max_dd = dd; }
    }

    weeks.push(WeekResult {
        pnl: week_balance - INITIAL_BALANCE,
        pnl_pct: (week_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0,
        trades: week_trades, wins: week_wins, max_dd: week_max_dd,
        tp_count: week_tp, signal_close_count: week_signal_close, dca_count: week_dca,
    });

    weeks
}

// ── Convenience: full run with default params (backward compat) ──

pub fn run_multi_tf_strategy(
    closes_5m: &[f64], highs_5m: &[f64], lows_5m: &[f64],
    buy_vol_5m: &[f64], sell_vol_5m: &[f64], oi_5m: &[f64], timestamps_5m: &[u64],
    closes_3m: &[f64], highs_3m: &[f64], lows_3m: &[f64], timestamps_3m: &[u64],
) -> MultiTFResult {
    run_multi_tf_with_params(
        closes_5m, highs_5m, lows_5m, buy_vol_5m, sell_vol_5m, oi_5m, timestamps_5m,
        closes_3m, highs_3m, lows_3m, timestamps_3m,
        &KCParams::default(),
    )
}

pub fn run_multi_tf_with_params(
    closes_5m: &[f64], highs_5m: &[f64], lows_5m: &[f64],
    buy_vol_5m: &[f64], sell_vol_5m: &[f64], oi_5m: &[f64], timestamps_5m: &[u64],
    closes_3m: &[f64], highs_3m: &[f64], lows_3m: &[f64], timestamps_3m: &[u64],
    params: &KCParams,
) -> MultiTFResult {
    let data = PrecomputedData::new(
        closes_5m, highs_5m, lows_5m, buy_vol_5m, sell_vol_5m, oi_5m, timestamps_5m,
        closes_3m, highs_3m, lows_3m, timestamps_3m,
    );

    let (kc_upper, kc_lower) = compute_kc_3m(
        closes_3m, highs_3m, lows_3m,
        params.kc_length, params.kc_mult, params.kc_atr_period,
    );

    let warmup_candles = (24 * 30).max(2);
    let n_candles = data.candles.len();
    let weeks = run_candle_range(&data, &kc_upper, &kc_lower, params, warmup_candles, n_candles);

    aggregate_weeks(weeks)
}

// ── Aggregate week results ──

pub fn aggregate_weeks(weeks: Vec<WeekResult>) -> MultiTFResult {
    let total_weeks = weeks.len();
    let weekly_pnl: Vec<f64> = weeks.iter().map(|w| w.pnl).collect();
    let weekly_pnl_pct: Vec<f64> = weeks.iter().map(|w| w.pnl_pct).collect();
    let weekly_trades: Vec<i32> = weeks.iter().map(|w| w.trades).collect();
    let weekly_wins: Vec<i32> = weeks.iter().map(|w| w.wins).collect();
    let weekly_max_dd: Vec<f64> = weeks.iter().map(|w| w.max_dd).collect();

    let total_trades: i32 = weekly_trades.iter().sum();
    let total_wins: i32 = weekly_wins.iter().sum();
    let win_rate = if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };
    let avg_weekly_pnl = if total_weeks > 0 { weekly_pnl.iter().sum::<f64>() / total_weeks as f64 } else { 0.0 };
    let avg_weekly_pnl_pct = if total_weeks > 0 { weekly_pnl_pct.iter().sum::<f64>() / total_weeks as f64 } else { 0.0 };
    let positive_weeks = weekly_pnl.iter().filter(|&&p| p > 0.0).count();
    let negative_weeks = weekly_pnl.iter().filter(|&&p| p < 0.0).count();
    let best_week_pct = weekly_pnl_pct.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let worst_week_pct = weekly_pnl_pct.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_dd_within = weekly_max_dd.iter().cloned().fold(0.0_f64, f64::max);

    let total_dca: i32 = weeks.iter().map(|w| w.dca_count).sum();
    let total_tp: i32 = weeks.iter().map(|w| w.tp_count).sum();
    let total_signal_close: i32 = weeks.iter().map(|w| w.signal_close_count).sum();

    let mut sorted_pnl_pct = weekly_pnl_pct.clone();
    sorted_pnl_pct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_weekly = if sorted_pnl_pct.is_empty() { 0.0 }
        else if sorted_pnl_pct.len() % 2 == 0 {
            (sorted_pnl_pct[sorted_pnl_pct.len() / 2 - 1] + sorted_pnl_pct[sorted_pnl_pct.len() / 2]) / 2.0
        } else { sorted_pnl_pct[sorted_pnl_pct.len() / 2] };

    let mut max_consec_neg: i32 = 0;
    let mut consec_neg: i32 = 0;
    for &p in &weekly_pnl {
        if p < 0.0 { consec_neg += 1; if consec_neg > max_consec_neg { max_consec_neg = consec_neg; } }
        else { consec_neg = 0; }
    }

    MultiTFResult {
        total_pnl: weekly_pnl.iter().sum(),
        total_weeks, weekly_pnl, weekly_pnl_pct, weekly_trades, weekly_wins, weekly_max_dd,
        total_trades, total_wins, win_rate, avg_weekly_pnl, avg_weekly_pnl_pct,
        median_weekly_pnl_pct: median_weekly,
        total_fees: 0.0, // not tracked in week results for speed
        positive_weeks, negative_weeks, best_week_pct, worst_week_pct,
        max_drawdown_within_week: max_dd_within,
        total_dca_count: total_dca, total_tp_count: total_tp, total_signal_close,
        avg_hold_bars_3m: 0.0, max_consecutive_loss_weeks: max_consec_neg,
        trades: Vec::new(), // trade log disabled for aggregated runs
    }
}

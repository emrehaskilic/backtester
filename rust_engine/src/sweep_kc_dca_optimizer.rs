/// DCA Walk-Forward Optimizer
/// KC sabit (14, 3.8, 15), sadece max_dca optimize
/// 40-fold WF, her fold'da DCA=0..10 grid search (11 trial)

use crate::sweep_kc_multi_tf::{
    PrecomputedData, KCParams, WeekResult,
    compute_kc_3m, run_candle_range, CANDLES_PER_WEEK,
};

const KC_LENGTH: usize = 14;
const KC_MULT: f64 = 3.8;
const KC_ATR_PERIOD: usize = 15;
const MIN_TRAIN_WEEKS: usize = 4;
const MAX_DCA_RANGE: i32 = 10; // 0..=10

fn compute_train_score(weeks: &[WeekResult]) -> f64 {
    if weeks.is_empty() { return -999.0; }
    let total_trades: i32 = weeks.iter().map(|w| w.trades).sum();
    if total_trades < 5 { return -999.0; }

    let mut sorted: Vec<f64> = weeks.iter().map(|w| w.pnl_pct).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let worst = sorted[0];
    let worst_penalty = if worst < -20.0 { 0.3 }
                        else if worst < -10.0 { 0.6 }
                        else if worst < -5.0 { 0.8 }
                        else { 1.0 };

    let pos_ratio = weeks.iter().filter(|w| w.pnl > 0.0).count() as f64 / weeks.len() as f64;
    let max_dd = weeks.iter().map(|w| w.max_dd).fold(0.0_f64, f64::max);
    let dd_penalty = if max_dd > 50.0 { 0.3 }
                     else if max_dd > 30.0 { 0.6 }
                     else if max_dd > 15.0 { 0.8 }
                     else { 1.0 };

    let score = median * worst_penalty * dd_penalty * (0.5 + pos_ratio);
    if score.is_nan() { -999.0 } else { score }
}

#[derive(Clone, Debug)]
pub struct DCAFoldResult {
    pub fold_idx: usize,
    pub best_dca: i32,
    pub train_score: f64,
    pub all_dca_train_scores: Vec<f64>, // score for DCA=0..10
    pub test_pnl: f64,
    pub test_pnl_pct: f64,
    pub test_trades: i32,
    pub test_wins: i32,
    pub test_max_dd: f64,
    pub test_tp: i32,
    pub test_signal_close: i32,
    pub test_dca: i32,
}

pub struct DCAOptResult {
    pub folds: Vec<DCAFoldResult>,
    pub total_folds: usize,
    pub positive_folds: usize,
    pub negative_folds: usize,
    pub avg_oos_pnl_pct: f64,
    pub median_oos_pnl_pct: f64,
    pub best_fold_pct: f64,
    pub worst_fold_pct: f64,
    pub total_oos_pnl: f64,
    pub max_consec_neg: i32,
    // DCA distribution: how many folds chose each DCA value
    pub dca_choice_counts: Vec<i32>, // index = dca value, value = count
}

pub fn run_dca_wf_optimization(
    closes_5m: &[f64], highs_5m: &[f64], lows_5m: &[f64],
    buy_vol_5m: &[f64], sell_vol_5m: &[f64], oi_5m: &[f64], timestamps_5m: &[u64],
    closes_3m: &[f64], highs_3m: &[f64], lows_3m: &[f64], timestamps_3m: &[u64],
) -> DCAOptResult {
    let data = PrecomputedData::new(
        closes_5m, highs_5m, lows_5m, buy_vol_5m, sell_vol_5m, oi_5m, timestamps_5m,
        closes_3m, highs_3m, lows_3m, timestamps_3m,
    );

    // KC sabit — bir kere hesapla
    let (kc_upper, kc_lower) = compute_kc_3m(
        closes_3m, highs_3m, lows_3m, KC_LENGTH, KC_MULT, KC_ATR_PERIOD,
    );

    let n_candles = data.candles.len();
    let warmup_candles = (24 * 30).max(2);

    let mut week_boundaries: Vec<usize> = Vec::new();
    week_boundaries.push(warmup_candles);
    let mut ci = warmup_candles;
    while ci + CANDLES_PER_WEEK <= n_candles {
        ci += CANDLES_PER_WEEK;
        week_boundaries.push(ci);
    }
    if *week_boundaries.last().unwrap_or(&0) < n_candles {
        week_boundaries.push(n_candles);
    }

    let total_weeks = week_boundaries.len() - 1;
    let start_fold = MIN_TRAIN_WEEKS;
    let n_folds = total_weeks - start_fold;

    eprintln!("[DCA-WF] {} folds, DCA grid 0..{}", n_folds, MAX_DCA_RANGE);

    let mut folds: Vec<DCAFoldResult> = Vec::with_capacity(n_folds);
    let mut dca_choice_counts = vec![0i32; (MAX_DCA_RANGE + 1) as usize];

    for f in 0..n_folds {
        let test_week_idx = start_fold + f;
        let train_candle_start = warmup_candles;
        let train_candle_end = week_boundaries[test_week_idx];
        let test_candle_start = week_boundaries[test_week_idx];
        let test_candle_end = week_boundaries[test_week_idx + 1];

        // Grid search: DCA=0..10
        let mut best_dca = 1i32;
        let mut best_score = f64::NEG_INFINITY;
        let mut all_scores = Vec::with_capacity((MAX_DCA_RANGE + 1) as usize);

        for dca in 0..=MAX_DCA_RANGE {
            let params = KCParams {
                kc_length: KC_LENGTH, kc_mult: KC_MULT,
                kc_atr_period: KC_ATR_PERIOD, max_dca: dca,
                dca_scale: 1.0, tp_levels: 1, tp_first_pct: 1.0,
            };
            let weeks = run_candle_range(&data, &kc_upper, &kc_lower, &params, train_candle_start, train_candle_end);
            let score = compute_train_score(&weeks);
            all_scores.push(score);
            if score > best_score {
                best_score = score;
                best_dca = dca;
            }
        }

        // Test with best DCA
        let test_params = KCParams {
            kc_length: KC_LENGTH, kc_mult: KC_MULT,
            kc_atr_period: KC_ATR_PERIOD, max_dca: best_dca,
            dca_scale: 1.0, tp_levels: 1, tp_first_pct: 1.0,
        };
        let test_result = run_candle_range(&data, &kc_upper, &kc_lower, &test_params, test_candle_start, test_candle_end);

        let test_week = if test_result.is_empty() {
            WeekResult { pnl: 0.0, pnl_pct: 0.0, trades: 0, wins: 0, max_dd: 0.0, tp_count: 0, signal_close_count: 0, dca_count: 0 }
        } else {
            let mut combined = WeekResult { pnl: 0.0, pnl_pct: 0.0, trades: 0, wins: 0, max_dd: 0.0, tp_count: 0, signal_close_count: 0, dca_count: 0 };
            for w in &test_result {
                combined.pnl += w.pnl;
                combined.trades += w.trades;
                combined.wins += w.wins;
                combined.tp_count += w.tp_count;
                combined.signal_close_count += w.signal_close_count;
                combined.dca_count += w.dca_count;
                if w.max_dd > combined.max_dd { combined.max_dd = w.max_dd; }
            }
            combined.pnl_pct = combined.pnl / 1000.0 * 100.0;
            combined
        };

        dca_choice_counts[best_dca as usize] += 1;

        eprintln!("[DCA-WF] Fold {}/{}: best_dca={} score={:.3} | OOS: {:.2}% ({:.2} USDT)",
            f + 1, n_folds, best_dca, best_score, test_week.pnl_pct, test_week.pnl);

        folds.push(DCAFoldResult {
            fold_idx: f, best_dca, train_score: best_score,
            all_dca_train_scores: all_scores,
            test_pnl: test_week.pnl, test_pnl_pct: test_week.pnl_pct,
            test_trades: test_week.trades, test_wins: test_week.wins,
            test_max_dd: test_week.max_dd, test_tp: test_week.tp_count,
            test_signal_close: test_week.signal_close_count,
            test_dca: test_week.dca_count,
        });
    }

    // Aggregate
    let total_folds = folds.len();
    let positive_folds = folds.iter().filter(|f| f.test_pnl > 0.0).count();
    let negative_folds = folds.iter().filter(|f| f.test_pnl < 0.0).count();
    let total_oos_pnl: f64 = folds.iter().map(|f| f.test_pnl).sum();
    let avg_oos_pnl_pct = if total_folds > 0 { folds.iter().map(|f| f.test_pnl_pct).sum::<f64>() / total_folds as f64 } else { 0.0 };

    let mut sorted_pcts: Vec<f64> = folds.iter().map(|f| f.test_pnl_pct).collect();
    sorted_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_oos = if sorted_pcts.is_empty() { 0.0 }
        else if sorted_pcts.len() % 2 == 0 {
            (sorted_pcts[sorted_pcts.len() / 2 - 1] + sorted_pcts[sorted_pcts.len() / 2]) / 2.0
        } else { sorted_pcts[sorted_pcts.len() / 2] };
    let best_fold = sorted_pcts.last().cloned().unwrap_or(0.0);
    let worst_fold = sorted_pcts.first().cloned().unwrap_or(0.0);

    let mut max_consec: i32 = 0;
    let mut consec: i32 = 0;
    for f in &folds {
        if f.test_pnl < 0.0 { consec += 1; if consec > max_consec { max_consec = consec; } }
        else { consec = 0; }
    }

    DCAOptResult {
        folds, total_folds, positive_folds, negative_folds,
        avg_oos_pnl_pct, median_oos_pnl_pct: median_oos,
        best_fold_pct: best_fold, worst_fold_pct: worst_fold,
        total_oos_pnl, max_consec_neg: max_consec,
        dca_choice_counts,
    }
}

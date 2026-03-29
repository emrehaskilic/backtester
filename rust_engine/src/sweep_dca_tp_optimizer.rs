/// DCA/TP Graduated Walk-Forward Optimizer
/// KC sabit (14, 3.8, 15), 4 parametre brute force grid
/// 40-fold WF, her fold'da 4608 kombinasyon

use rayon::prelude::*;

use crate::sweep_kc_multi_tf::{
    PrecomputedData, KCParams, WeekResult,
    compute_kc_3m, run_candle_range, CANDLES_PER_WEEK,
};

const KC_LENGTH: usize = 14;
const KC_MULT: f64 = 3.8;
const KC_ATR_PERIOD: usize = 15;
const MIN_TRAIN_WEEKS: usize = 4;
const INITIAL_BALANCE: f64 = 1000.0;

// ── Grid ──

struct GridPoint {
    max_dca: i32,
    dca_scale: f64,
    tp_levels: i32,
    tp_first_pct: f64,
}

fn build_grid() -> Vec<GridPoint> {
    let mut grid = Vec::with_capacity(6 * 16 * 3 * 16);
    for max_dca in 0..=5i32 {
        for dca_s in 0..=15u32 { // 0.5 to 2.0, step 0.1
            let dca_scale = 0.5 + dca_s as f64 * 0.1;
            for tp_levels in 1..=3i32 {
                for tp_p in 0..=15u32 { // 0.25 to 1.0, step 0.05
                    let tp_first_pct = 0.25 + tp_p as f64 * 0.05;
                    grid.push(GridPoint { max_dca, dca_scale, tp_levels, tp_first_pct });
                }
            }
        }
    }
    grid
}

fn grid_to_params(g: &GridPoint) -> KCParams {
    KCParams {
        kc_length: KC_LENGTH, kc_mult: KC_MULT, kc_atr_period: KC_ATR_PERIOD,
        max_dca: g.max_dca, dca_scale: g.dca_scale,
        tp_levels: g.tp_levels, tp_first_pct: g.tp_first_pct,
    }
}

// ── Score ──

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

// ── Fold result ──

#[derive(Clone, Debug)]
pub struct GradFoldResult {
    pub fold_idx: usize,
    pub best_max_dca: i32,
    pub best_dca_scale: f64,
    pub best_tp_levels: i32,
    pub best_tp_first_pct: f64,
    pub train_score: f64,
    pub train_weeks: usize,
    pub test_pnl: f64,
    pub test_pnl_pct: f64,
    pub test_trades: i32,
    pub test_wins: i32,
    pub test_max_dd: f64,
    pub test_tp: i32,
    pub test_signal_close: i32,
    pub test_dca: i32,
}

pub struct GradOptResult {
    pub folds: Vec<GradFoldResult>,
    pub total_folds: usize,
    pub positive_folds: usize,
    pub negative_folds: usize,
    pub avg_oos_pnl_pct: f64,
    pub median_oos_pnl_pct: f64,
    pub best_fold_pct: f64,
    pub worst_fold_pct: f64,
    pub total_oos_pnl: f64,
    pub max_consec_neg: i32,
    pub grid_size: usize,
}

// ── Main ──

pub fn run_grad_wf_optimization(
    closes_5m: &[f64], highs_5m: &[f64], lows_5m: &[f64],
    buy_vol_5m: &[f64], sell_vol_5m: &[f64], oi_5m: &[f64], timestamps_5m: &[u64],
    closes_3m: &[f64], highs_3m: &[f64], lows_3m: &[f64], timestamps_3m: &[u64],
) -> GradOptResult {
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

    let grid = build_grid();
    let grid_size = grid.len();
    eprintln!("[GRAD-WF] {} folds, grid size: {}", n_folds, grid_size);

    let mut folds: Vec<GradFoldResult> = Vec::with_capacity(n_folds);

    for f in 0..n_folds {
        let test_week_idx = start_fold + f;
        let train_candle_start = warmup_candles;
        let train_candle_end = week_boundaries[test_week_idx];
        let test_candle_start = week_boundaries[test_week_idx];
        let test_candle_end = week_boundaries[test_week_idx + 1];
        let train_weeks = test_week_idx;

        // Grid search in parallel
        let scores: Vec<f64> = grid.par_iter().map(|g| {
            let params = grid_to_params(g);
            let weeks = run_candle_range(&data, &kc_upper, &kc_lower, &params, train_candle_start, train_candle_end);
            compute_train_score(&weeks)
        }).collect();

        // Find best
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &s) in scores.iter().enumerate() {
            if s > best_score { best_score = s; best_idx = i; }
        }

        let best = &grid[best_idx];
        let best_params = grid_to_params(best);

        // Test
        let test_result = run_candle_range(&data, &kc_upper, &kc_lower, &best_params, test_candle_start, test_candle_end);
        let test_week = if test_result.is_empty() {
            WeekResult { pnl: 0.0, pnl_pct: 0.0, trades: 0, wins: 0, max_dd: 0.0, tp_count: 0, signal_close_count: 0, dca_count: 0 }
        } else {
            let mut c = WeekResult { pnl: 0.0, pnl_pct: 0.0, trades: 0, wins: 0, max_dd: 0.0, tp_count: 0, signal_close_count: 0, dca_count: 0 };
            for w in &test_result {
                c.pnl += w.pnl; c.trades += w.trades; c.wins += w.wins;
                c.tp_count += w.tp_count; c.signal_close_count += w.signal_close_count;
                c.dca_count += w.dca_count;
                if w.max_dd > c.max_dd { c.max_dd = w.max_dd; }
            }
            c.pnl_pct = c.pnl / INITIAL_BALANCE * 100.0;
            c
        };

        eprintln!("[GRAD-WF] Fold {}/{}: DCA({},{:.1}) TP({},{:.2}) | train={:.3} | OOS: {:.2}% ({:.2}$) {}trd",
            f + 1, n_folds,
            best.max_dca, best.dca_scale, best.tp_levels, best.tp_first_pct,
            best_score, test_week.pnl_pct, test_week.pnl, test_week.trades);

        folds.push(GradFoldResult {
            fold_idx: f,
            best_max_dca: best.max_dca,
            best_dca_scale: best.dca_scale,
            best_tp_levels: best.tp_levels,
            best_tp_first_pct: best.tp_first_pct,
            train_score: best_score,
            train_weeks: train_weeks,
            test_pnl: test_week.pnl,
            test_pnl_pct: test_week.pnl_pct,
            test_trades: test_week.trades,
            test_wins: test_week.wins,
            test_max_dd: test_week.max_dd,
            test_tp: test_week.tp_count,
            test_signal_close: test_week.signal_close_count,
            test_dca: test_week.dca_count,
        });
    }

    // Aggregate
    let total_folds = folds.len();
    let positive_folds = folds.iter().filter(|f| f.test_pnl > 0.0).count();
    let negative_folds = folds.iter().filter(|f| f.test_pnl < 0.0).count();
    let total_oos_pnl: f64 = folds.iter().map(|f| f.test_pnl).sum();
    let avg_oos = if total_folds > 0 { folds.iter().map(|f| f.test_pnl_pct).sum::<f64>() / total_folds as f64 } else { 0.0 };

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

    GradOptResult {
        folds, total_folds, positive_folds, negative_folds,
        avg_oos_pnl_pct: avg_oos, median_oos_pnl_pct: median_oos,
        best_fold_pct: best_fold, worst_fold_pct: worst_fold,
        total_oos_pnl, max_consec_neg: max_consec, grid_size,
    }
}

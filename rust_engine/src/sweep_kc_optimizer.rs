/// KC Walk-Forward Optimizer — 35-fold, her hafta bağımsız test
///
/// Her fold w:
///   Train: hafta warmup..w-1 (expanding window)
///   Test:  hafta w (tek hafta)
///   TPE ile KC params optimize → test haftasında OOS sonuç
///
/// Look-ahead koruması:
///   - KC causal (EMA/ATR bar-by-bar, lagged kullanım)
///   - Quantile thresholds: candle_start'a kadar olan data'dan
///   - Optimizer SADECE train haftaların PnL'ine bakıyor
///   - Test haftası hiçbir şekilde train'e sızmıyor

use rayon::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::sweep_kc_multi_tf::{
    PrecomputedData, KCParams, WeekResult,
    compute_kc_3m, run_candle_range, CANDLES_PER_WEEK,
};

// ── Optimizer config ──

const N_TRIALS: usize = 500;
const N_STARTUP: usize = 100;   // random exploration
const GAMMA: f64 = 0.25;        // top 25% = good
const BATCH_SIZE: usize = 64;
const N_REFINE_RATIO: f64 = 0.2; // %20 budget local refinement
const MIN_TRAIN_WEEKS: usize = 4; // minimum train hafta sayısı

// ── Parameter space ──

#[derive(Clone)]
struct ParamSpec {
    min: f64,
    max: f64,
    step: f64,
    is_int: bool,
}

fn param_specs() -> [ParamSpec; 4] {
    [
        ParamSpec { min: 10.0, max: 50.0, step: 1.0, is_int: true },   // kc_length
        ParamSpec { min: 1.0,  max: 4.0,  step: 0.1, is_int: false },  // kc_mult
        ParamSpec { min: 7.0,  max: 28.0, step: 1.0, is_int: true },   // kc_atr_period
        ParamSpec { min: 0.0,  max: 10.0, step: 1.0, is_int: true },   // max_dca
    ]
}

impl ParamSpec {
    fn sample_random(&self, rng: &mut ChaCha8Rng) -> f64 {
        let steps = ((self.max - self.min) / self.step).round() as i32;
        let s = rng.gen_range(0..=steps);
        self.min + s as f64 * self.step
    }

    fn sample_tpe(&self, good_vals: &[f64], rng: &mut ChaCha8Rng) -> f64 {
        if good_vals.is_empty() { return self.sample_random(rng); }
        let steps = ((self.max - self.min) / self.step).round() as usize;
        let mut counts = vec![1u32; steps + 1]; // Laplace smoothing
        for &v in good_vals {
            let idx = ((v - self.min) / self.step).round() as usize;
            if idx <= steps { counts[idx] += 10; }
        }
        let total: u32 = counts.iter().sum();
        let r = rng.gen_range(0..total);
        let mut cum = 0u32;
        for (i, &c) in counts.iter().enumerate() {
            cum += c;
            if r < cum { return self.min + i as f64 * self.step; }
        }
        self.max
    }

    fn perturb(&self, val: f64, scale: f64, rng: &mut ChaCha8Rng) -> f64 {
        let range = (self.max - self.min) * scale;
        let delta = rng.gen_range(-range..=range);
        let raw = (val + delta).clamp(self.min, self.max);
        let steps = ((raw - self.min) / self.step).round();
        (self.min + steps * self.step).clamp(self.min, self.max)
    }
}

fn params_to_kc(p: &[f64; 4]) -> KCParams {
    KCParams {
        kc_length: p[0].round() as usize,
        kc_mult: (p[1] * 10.0).round() / 10.0,
        kc_atr_period: p[2].round() as usize,
        max_dca: p[3].round() as i32,
        dca_scale: 1.0, tp_levels: 1, tp_first_pct: 1.0,
    }
}

// ── Score function (train weeks) ──

fn compute_train_score(weeks: &[WeekResult]) -> f64 {
    if weeks.is_empty() { return -999.0; }

    let total_trades: i32 = weeks.iter().map(|w| w.trades).sum();
    if total_trades < 10 { return -999.0; } // çok az trade

    let pnl_pcts: Vec<f64> = weeks.iter().map(|w| w.pnl_pct).collect();

    // Median weekly PnL %
    let mut sorted = pnl_pcts.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Worst week penalty
    let worst = sorted[0]; // en kötü hafta
    let worst_penalty = if worst < -20.0 { 0.3 }
                        else if worst < -10.0 { 0.6 }
                        else if worst < -5.0 { 0.8 }
                        else { 1.0 };

    // Positive week ratio bonus
    let pos_ratio = weeks.iter().filter(|w| w.pnl > 0.0).count() as f64 / weeks.len() as f64;

    // Max DD penalty
    let max_dd = weeks.iter().map(|w| w.max_dd).fold(0.0_f64, f64::max);
    let dd_penalty = if max_dd > 50.0 { 0.3 }
                     else if max_dd > 30.0 { 0.6 }
                     else if max_dd > 15.0 { 0.8 }
                     else { 1.0 };

    // Score = median * worst_penalty * dd_penalty * pos_ratio_boost
    let score = median * worst_penalty * dd_penalty * (0.5 + pos_ratio);

    if score.is_nan() { -999.0 } else { score }
}

// ── Trial result ──

#[derive(Clone, Debug)]
struct TrialResult {
    params: [f64; 4],
    score: f64,
}

// ── Single fold optimizer ──

fn optimize_single_fold(
    data: &PrecomputedData,
    train_candle_start: usize,
    train_candle_end: usize,
    seed: u64,
) -> [f64; 4] {
    let specs = param_specs();
    let mut all_results: Vec<TrialResult> = Vec::with_capacity(N_TRIALS);

    // Phase 1: Random exploration
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let random_params: Vec<[f64; 4]> = (0..N_STARTUP).map(|_| {
        [
            specs[0].sample_random(&mut rng),
            specs[1].sample_random(&mut rng),
            specs[2].sample_random(&mut rng),
            specs[3].sample_random(&mut rng),
        ]
    }).collect();

    let random_results: Vec<TrialResult> = random_params.par_iter().map(|p| {
        let kc = params_to_kc(p);
        let (ku, kl) = compute_kc_3m(
            &data.closes_3m, &data.highs_3m, &data.lows_3m,
            kc.kc_length, kc.kc_mult, kc.kc_atr_period,
        );
        let weeks = run_candle_range(data, &ku, &kl, &kc, train_candle_start, train_candle_end);
        let score = compute_train_score(&weeks);
        TrialResult { params: *p, score }
    }).collect();

    all_results.extend(random_results);

    // Phase 2: TPE-guided search
    let n_tpe = N_TRIALS - N_STARTUP - (N_TRIALS as f64 * N_REFINE_RATIO) as usize;

    for batch_start in (0..n_tpe).step_by(BATCH_SIZE) {
        let actual_batch = BATCH_SIZE.min(n_tpe - batch_start);

        let mut valid: Vec<&TrialResult> = all_results.iter()
            .filter(|r| r.score > -999.0)
            .collect();
        valid.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let n_good = (valid.len() as f64 * GAMMA).ceil() as usize;

        let tpe_params: Vec<[f64; 4]> = if n_good >= 5 {
            let good = &valid[..n_good];
            (0..actual_batch).map(|j| {
                let mut trial_rng = ChaCha8Rng::seed_from_u64(seed + 1000 + batch_start as u64 + j as u64);
                [
                    specs[0].sample_tpe(&good.iter().map(|r| r.params[0]).collect::<Vec<_>>(), &mut trial_rng),
                    specs[1].sample_tpe(&good.iter().map(|r| r.params[1]).collect::<Vec<_>>(), &mut trial_rng),
                    specs[2].sample_tpe(&good.iter().map(|r| r.params[2]).collect::<Vec<_>>(), &mut trial_rng),
                    specs[3].sample_tpe(&good.iter().map(|r| r.params[3]).collect::<Vec<_>>(), &mut trial_rng),
                ]
            }).collect()
        } else {
            (0..actual_batch).map(|j| {
                let mut trial_rng = ChaCha8Rng::seed_from_u64(seed + 2000 + batch_start as u64 + j as u64);
                [
                    specs[0].sample_random(&mut trial_rng),
                    specs[1].sample_random(&mut trial_rng),
                    specs[2].sample_random(&mut trial_rng),
                    specs[3].sample_random(&mut trial_rng),
                ]
            }).collect()
        };

        let batch_results: Vec<TrialResult> = tpe_params.par_iter().map(|p| {
            let kc = params_to_kc(p);
            let (ku, kl) = compute_kc_3m(
                &data.closes_3m, &data.highs_3m, &data.lows_3m,
                kc.kc_length, kc.kc_mult, kc.kc_atr_period,
            );
            let weeks = run_candle_range(data, &ku, &kl, &kc, train_candle_start, train_candle_end);
            let score = compute_train_score(&weeks);
            TrialResult { params: *p, score }
        }).collect();

        all_results.extend(batch_results);
    }

    // Phase 3: Local refinement around best
    let n_refine = (N_TRIALS as f64 * N_REFINE_RATIO) as usize;
    let mut valid: Vec<&TrialResult> = all_results.iter()
        .filter(|r| r.score > -999.0)
        .collect();
    valid.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if valid.len() >= 3 {
        let top_k = valid.len().min(10);
        let top_params: Vec<[f64; 4]> = valid[..top_k].iter().map(|r| r.params).collect();

        let refine_params: Vec<[f64; 4]> = (0..n_refine).map(|j| {
            let mut trial_rng = ChaCha8Rng::seed_from_u64(seed + 5000 + j as u64);
            let base = &top_params[j % top_k];
            [
                specs[0].perturb(base[0], 0.15, &mut trial_rng),
                specs[1].perturb(base[1], 0.15, &mut trial_rng),
                specs[2].perturb(base[2], 0.15, &mut trial_rng),
                specs[3].perturb(base[3], 0.15, &mut trial_rng),
            ]
        }).collect();

        let refine_results: Vec<TrialResult> = refine_params.par_iter().map(|p| {
            let kc = params_to_kc(p);
            let (ku, kl) = compute_kc_3m(
                &data.closes_3m, &data.highs_3m, &data.lows_3m,
                kc.kc_length, kc.kc_mult, kc.kc_atr_period,
            );
            let weeks = run_candle_range(data, &ku, &kl, &kc, train_candle_start, train_candle_end);
            let score = compute_train_score(&weeks);
            TrialResult { params: *p, score }
        }).collect();

        all_results.extend(refine_results);
    }

    // Return best
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    if all_results.is_empty() || all_results[0].score <= -999.0 {
        [20.0, 2.0, 14.0, 3.0] // fallback default
    } else {
        all_results[0].params
    }
}

// ── Fold result ──

#[derive(Clone, Debug)]
pub struct FoldResult {
    pub fold_idx: usize,
    pub kc_length: usize,
    pub kc_mult: f64,
    pub kc_atr_period: usize,
    pub max_dca: i32,
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

// ── Full WF result ──

pub struct WFOptResult {
    pub folds: Vec<FoldResult>,
    pub total_folds: usize,
    pub positive_folds: usize,
    pub negative_folds: usize,
    pub avg_oos_pnl_pct: f64,
    pub median_oos_pnl_pct: f64,
    pub best_fold_pct: f64,
    pub worst_fold_pct: f64,
    pub total_oos_pnl: f64,
    pub max_consec_neg: i32,
}

// ── Main: 35-fold walk-forward ──

pub fn run_wf_optimization(
    closes_5m: &[f64], highs_5m: &[f64], lows_5m: &[f64],
    buy_vol_5m: &[f64], sell_vol_5m: &[f64], oi_5m: &[f64], timestamps_5m: &[u64],
    closes_3m: &[f64], highs_3m: &[f64], lows_3m: &[f64], timestamps_3m: &[u64],
    seed: u64,
) -> WFOptResult {
    // Precompute once
    let data = PrecomputedData::new(
        closes_5m, highs_5m, lows_5m, buy_vol_5m, sell_vol_5m, oi_5m, timestamps_5m,
        closes_3m, highs_3m, lows_3m, timestamps_3m,
    );

    let n_candles = data.candles.len();
    let warmup_candles = (24 * 30).max(2); // 30 days warmup

    // Build week boundaries: candle indices where each week starts
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
    eprintln!("[WF] Total weeks: {}, Warmup candles: {}, N candles: {}", total_weeks, warmup_candles, n_candles);

    // 35-fold walk-forward: each fold tests one week
    // Need at least MIN_TRAIN_WEEKS weeks for training
    let start_fold = MIN_TRAIN_WEEKS;
    let n_folds = total_weeks - start_fold;

    eprintln!("[WF] Running {} folds (week {} to {})", n_folds, start_fold + 1, total_weeks);

    // Sequential folds (each fold's train window is different, cannot parallelize fold-level
    // because KC recomputation per trial is already parallelized via rayon)
    let mut folds: Vec<FoldResult> = Vec::with_capacity(n_folds);

    for f in 0..n_folds {
        let test_week_idx = start_fold + f; // which week to test
        let train_candle_start = warmup_candles;
        let train_candle_end = week_boundaries[test_week_idx];
        let test_candle_start = week_boundaries[test_week_idx];
        let test_candle_end = week_boundaries[test_week_idx + 1];
        let train_weeks = test_week_idx; // how many weeks in train

        eprintln!("[WF] Fold {}/{}: train candles [{}, {}), test candles [{}, {}), train_weeks={}",
            f + 1, n_folds,
            train_candle_start, train_candle_end,
            test_candle_start, test_candle_end,
            train_weeks);

        // Optimize on train
        let best_params = optimize_single_fold(
            &data,
            train_candle_start,
            train_candle_end,
            seed + f as u64 * 1000,
        );
        let best_kc = params_to_kc(&best_params);

        // Compute train score for reporting
        let (ku_train, kl_train) = compute_kc_3m(
            &data.closes_3m, &data.highs_3m, &data.lows_3m,
            best_kc.kc_length, best_kc.kc_mult, best_kc.kc_atr_period,
        );
        let train_result = run_candle_range(&data, &ku_train, &kl_train, &best_kc, train_candle_start, train_candle_end);
        let train_score = compute_train_score(&train_result);

        // Test: run ONLY the test week with best params
        // KC is computed on ALL data up to test_candle_end (causal, no look-ahead)
        let (ku_test, kl_test) = compute_kc_3m(
            &data.closes_3m, &data.highs_3m, &data.lows_3m,
            best_kc.kc_length, best_kc.kc_mult, best_kc.kc_atr_period,
        );
        let test_result = run_candle_range(&data, &ku_test, &kl_test, &best_kc, test_candle_start, test_candle_end);

        let test_week = if test_result.is_empty() {
            WeekResult { pnl: 0.0, pnl_pct: 0.0, trades: 0, wins: 0, max_dd: 0.0, tp_count: 0, signal_close_count: 0, dca_count: 0 }
        } else {
            // May contain >1 week if partial, take the test-relevant one
            // Since we pass exact week boundaries, should be exactly 1 week
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
            combined.pnl_pct = combined.pnl / INITIAL_BALANCE * 100.0;
            combined
        };

        eprintln!("[WF] Fold {}: KC({},{:.1},{}) maxDCA={} | train_score={:.3} | OOS: {:.2}% ({:.2} USDT), {} trades",
            f + 1,
            best_kc.kc_length, best_kc.kc_mult, best_kc.kc_atr_period, best_kc.max_dca,
            train_score,
            test_week.pnl_pct, test_week.pnl, test_week.trades);

        folds.push(FoldResult {
            fold_idx: f,
            kc_length: best_kc.kc_length,
            kc_mult: best_kc.kc_mult,
            kc_atr_period: best_kc.kc_atr_period,
            max_dca: best_kc.max_dca,
            train_score,
            train_weeks,
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

    WFOptResult {
        folds, total_folds, positive_folds, negative_folds,
        avg_oos_pnl_pct, median_oos_pnl_pct: median_oos,
        best_fold_pct: best_fold, worst_fold_pct: worst_fold,
        total_oos_pnl, max_consec_neg: max_consec,
    }
}

const INITIAL_BALANCE: f64 = 1000.0;

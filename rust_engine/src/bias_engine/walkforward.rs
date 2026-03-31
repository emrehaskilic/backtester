/// Bias Engine — Step 12: Walk-Forward Evaluation (Spec Section 17)
///
/// Walk-forward with rolling windows:
///   train   = 40,320 bars (4.67 months)
///   val     = 8,640 bars (1 month)
///   test    = 8,640 bars (1 month)
///   step    = 8,640 bars (1 month)
///
/// Each fold:
///   1. Train: mine states, compute probabilities, robustness
///   2. Val: fit calibration
///   3. Test: produce bias, measure performance
///
/// WF-validated: a state passes significance in ≥75% of folds AND edge sign is consistent.

use crate::bias_engine::calibration::IsotonicCalibrator;
use crate::bias_engine::fallback::{self, FallbackResult, ValidatedState};
use crate::bias_engine::features::{self, FeatureArrays, N_FEATURES};
use crate::bias_engine::probability::{self, StateStats};
use crate::bias_engine::quantize;
use crate::bias_engine::regime;
use crate::bias_engine::robustness;
use crate::bias_engine::state::{self, StateKey};
use crate::bias_engine::sweep;
use crate::bias_engine::final_bias::{self, FinalBiasOutput};
use crate::bias_engine::htf;
use crate::bias_engine::confidence;
use std::collections::{HashMap, HashSet};

/// Walk-forward window sizes (in 5m bars).
pub const TRAIN_WINDOW: usize = 48_960;   // ~5.67 months (balance: regime diversity + enough folds)
pub const VAL_WINDOW: usize = 8_640;      // ~1 month
pub const TEST_WINDOW: usize = 8_640;     // ~1 month
pub const STEP_SIZE: usize = 8_640;       // ~1 month

/// Per-fold evaluation result.
#[derive(Clone, Debug)]
pub struct FoldResult {
    pub fold_idx: usize,
    /// Ranges
    pub train_start: usize,
    pub train_end: usize,
    pub val_start: usize,
    pub val_end: usize,
    pub test_start: usize,
    pub test_end: usize,

    /// Number of validated states in this fold's train set
    pub n_validated: usize,
    /// Test set metrics
    pub test_accuracy: f64,         // direction accuracy
    pub test_strong_accuracy: f64,  // accuracy when |bias| > 0.15
    pub test_n_bars: usize,
    pub test_n_strong: usize,       // bars with |bias| > 0.15
    pub test_mean_bias: f64,
    pub test_mean_abs_bias: f64,
    /// States that were validated in this fold
    pub validated_states: HashSet<StateKey>,
    /// Edge signs per state in this fold (true = bullish)
    pub state_edge_signs: HashMap<StateKey, bool>,
    /// Brier score (calibrated vs uncalibrated)
    pub brier_uncalibrated: f64,
    pub brier_calibrated: f64,
}

/// Full walk-forward result.
#[derive(Clone, Debug)]
pub struct WalkForwardResult {
    pub folds: Vec<FoldResult>,
    /// States validated in ≥75% of folds with consistent edge sign
    pub wf_validated_states: HashSet<StateKey>,
    /// Overall test accuracy across all folds
    pub overall_accuracy: f64,
    /// Overall strong signal accuracy
    pub overall_strong_accuracy: f64,
    /// Accuracy std across folds
    pub accuracy_std: f64,
    /// Total test bars evaluated
    pub total_test_bars: usize,
}

/// Run the full walk-forward evaluation on the entire dataset.
///
/// All raw arrays must be the same length.
pub fn run_walk_forward(
    timestamps: &[u64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
) -> WalkForwardResult {
    let n = close.len();
    let min_required = TRAIN_WINDOW + VAL_WINDOW + TEST_WINDOW;
    assert!(n >= min_required, "Need at least {} bars for WF, got {}", min_required, n);

    // Pre-compute features and quantiles for the whole dataset (shared)
    let feats = features::compute_features(timestamps, high, low, close, buy_vol, sell_vol, oi);
    let quant_results = quantize::quantize_all_full(&feats);
    let quintiles: Vec<Vec<u8>> = quant_results.iter().map(|qr| qr.quintiles.clone()).collect();
    let boundaries: Vec<Vec<[f64; quantize::MAX_BOUNDARIES]>> = quant_results.iter().map(|qr| qr.boundaries.clone()).collect();

    // Pre-compute HTF series (for sweep)
    let htf_series = htf::build_all_htf(timestamps, open, high, low, close, buy_vol, sell_vol);

    // Pre-compute ATR percentile for regime detection
    let atr_pct = feats.get(6); // feature #6 = atr_percentile

    // Generate folds
    let mut folds = Vec::new();
    let mut fold_idx = 0;
    let mut train_start = 0;

    // Track state appearances across folds
    let mut state_fold_count: HashMap<StateKey, usize> = HashMap::new();
    let mut state_edge_signs_all: HashMap<StateKey, Vec<bool>> = HashMap::new();

    while train_start + TRAIN_WINDOW + VAL_WINDOW + TEST_WINDOW <= n {
        let train_end = train_start + TRAIN_WINDOW;
        let val_start = train_end;
        let val_end = val_start + VAL_WINDOW;
        let test_start = val_end;
        let test_end = (test_start + TEST_WINDOW).min(n);

        // ── TRAIN: mine states, compute probabilities, robustness ──
        let train_prob = probability::compute_probabilities(
            &close[..train_end],
            &quintiles.iter().map(|q| q[..train_end].to_vec()).collect::<Vec<_>>(),
            train_end,
        );

        let significant_states: HashMap<StateKey, StateStats> = train_prob
            .state_stats
            .iter()
            .filter(|(_, s)| s.significant)
            .map(|(&k, s)| (k, s.clone()))
            .collect();

        let feature_vecs: Vec<Vec<f64>> = (0..N_FEATURES)
            .map(|i| feats.get(i)[..train_end].to_vec())
            .collect();

        let train_boundaries: Vec<Vec<[f64; quantize::MAX_BOUNDARIES]>> = boundaries
            .iter()
            .map(|b| b[..train_end].to_vec())
            .collect();

        let train_quintiles: Vec<Vec<u8>> = quintiles
            .iter()
            .map(|q| q[..train_end].to_vec())
            .collect();

        let rob_result = robustness::run_robustness(
            &close[..train_end],
            &train_quintiles,
            &train_boundaries,
            &feature_vecs,
            &train_prob.outcomes,
            train_prob.baseline_bull_rate,
            &significant_states,
            train_end,
        );

        // Build validated state map
        let mut validated: HashMap<StateKey, ValidatedState> = HashMap::new();
        let mut validated_set = HashSet::new();
        let mut state_edges = HashMap::new();

        for (&key, rob) in &rob_result.results {
            if rob.validated {
                if let Some(stats) = train_prob.state_stats.get(&key) {
                    validated.insert(key, ValidatedState {
                        stats: stats.clone(),
                        robustness: rob.clone(),
                    });
                    validated_set.insert(key);
                    let is_bullish = stats.bias > 0.0;
                    state_edges.insert(key, is_bullish);

                    *state_fold_count.entry(key).or_insert(0) += 1;
                    state_edge_signs_all.entry(key).or_default().push(is_bullish);
                }
            }
        }

        // ── VALIDATION: fit calibration ──
        let val_fallback = compute_fallback_for_range(
            &quintiles, &train_prob.outcomes, &validated, &train_prob.state_stats,
            val_start, val_end,
        );
        let val_probs: Vec<f64> = val_fallback.iter().map(|f| f.smoothed_prob).collect();
        // Use outcomes from the val range (need to compute fresh)
        let val_outcomes = probability::compute_outcomes(&close[val_start..val_end], probability::K_HORIZON);
        let calibrator = IsotonicCalibrator::fit(&val_probs, &val_outcomes);

        // ── TEST: produce bias, measure performance ──
        let test_outcomes = probability::compute_outcomes(&close[test_start..test_end], probability::K_HORIZON);
        let test_fallback = compute_fallback_for_range(
            &quintiles, &train_prob.outcomes, &validated, &train_prob.state_stats,
            test_start, test_end,
        );

        // Compute regime for test range
        let test_regimes = regime::detect_regimes(
            &close[test_start..test_end],
            &high[test_start..test_end],
            &low[test_start..test_end],
            &atr_pct[test_start..test_end],
        );

        // Compute sweep bias for test range (using global HTF series mapped to test range)
        let test_sweep = compute_sweep_for_range(&htf_series, test_start, test_end);

        // Combine into final bias
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut strong_correct = 0usize;
        let mut strong_total = 0usize;
        let mut sum_bias = 0.0f64;
        let mut sum_abs_bias = 0.0f64;
        let mut brier_uncal = 0.0f64;
        let mut brier_cal = 0.0f64;
        let mut brier_n = 0usize;

        let test_len = test_end - test_start;
        for i in 0..test_len {
            let fb = &test_fallback[i];
            let sweep_bias = test_sweep[i];
            let regime_shift = test_regimes[i].regime_shift;

            // Confidence
            let conf = if fb.fallback_used {
                confidence::compute_fallback_confidence()
            } else {
                // Use first matched state's stats for confidence
                let ci_width = if let Some(&key) = fb.matched_states.first() {
                    if let Some(vs) = validated.get(&key) {
                        vs.stats.ci_95_half * 2.0
                    } else { 0.20 }
                } else { 0.20 };
                let ns = if let Some(&key) = fb.matched_states.first() {
                    if let Some(vs) = validated.get(&key) {
                        vs.robustness.noise_stability
                    } else { 0.0 }
                } else { 0.0 };
                let n_tot = if let Some(&key) = fb.matched_states.first() {
                    if let Some(vs) = validated.get(&key) {
                        vs.stats.n_total
                    } else { 0 }
                } else { 0 };
                confidence::compute_confidence(n_tot, ci_width, ns, true)
            };

            // Calibrate
            let cal_prob = calibrator.transform(fb.smoothed_prob);
            let cal_bias = cal_prob - 0.50;

            let final_out = final_bias::compute_final_bias(cal_bias, sweep_bias, regime_shift, conf);

            if test_outcomes[i] != 255 {
                let actual_bull = test_outcomes[i] == 1;
                let pred_bull = final_out.final_bias > 0.0;

                total += 1;
                if pred_bull == actual_bull { correct += 1; }

                if final_out.final_bias.abs() > 0.15 {
                    strong_total += 1;
                    if pred_bull == actual_bull { strong_correct += 1; }
                }

                sum_bias += final_out.final_bias;
                sum_abs_bias += final_out.final_bias.abs();

                // Brier score
                let actual_f = if actual_bull { 1.0 } else { 0.0 };
                brier_uncal += (fb.smoothed_prob - actual_f).powi(2);
                brier_cal += (cal_prob - actual_f).powi(2);
                brier_n += 1;
            }
        }

        let test_accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.50 };
        let test_strong_acc = if strong_total > 0 { strong_correct as f64 / strong_total as f64 } else { 0.50 };

        folds.push(FoldResult {
            fold_idx,
            train_start,
            train_end,
            val_start,
            val_end,
            test_start,
            test_end,
            n_validated: validated_set.len(),
            test_accuracy,
            test_strong_accuracy: test_strong_acc,
            test_n_bars: total,
            test_n_strong: strong_total,
            test_mean_bias: if total > 0 { sum_bias / total as f64 } else { 0.0 },
            test_mean_abs_bias: if total > 0 { sum_abs_bias / total as f64 } else { 0.0 },
            validated_states: validated_set,
            state_edge_signs: state_edges,
            brier_uncalibrated: if brier_n > 0 { brier_uncal / brier_n as f64 } else { 0.0 },
            brier_calibrated: if brier_n > 0 { brier_cal / brier_n as f64 } else { 0.0 },
        });

        fold_idx += 1;
        train_start += STEP_SIZE;
    }

    // Determine WF-validated states: ≥75% of folds + consistent edge sign
    let n_folds = folds.len();
    let min_folds = ((n_folds as f64 * 0.75).ceil() as usize).max(1);

    let mut wf_validated = HashSet::new();
    for (&key, &count) in &state_fold_count {
        if count >= min_folds {
            // Check edge sign consistency
            if let Some(signs) = state_edge_signs_all.get(&key) {
                let n_bull = signs.iter().filter(|&&s| s).count();
                let n_bear = signs.len() - n_bull;
                // All the same sign
                if n_bull == signs.len() || n_bear == signs.len() {
                    wf_validated.insert(key);
                }
            }
        }
    }

    // Aggregate metrics
    let accuracies: Vec<f64> = folds.iter().map(|f| f.test_accuracy).collect();
    let mean_acc = if accuracies.is_empty() {
        0.50
    } else {
        accuracies.iter().sum::<f64>() / accuracies.len() as f64
    };
    let acc_std = if accuracies.len() > 1 {
        let var = accuracies.iter().map(|&a| (a - mean_acc).powi(2)).sum::<f64>()
            / (accuracies.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let total_test_bars: usize = folds.iter().map(|f| f.test_n_bars).sum();
    let total_correct: usize = folds.iter().map(|f| (f.test_accuracy * f.test_n_bars as f64).round() as usize).sum();
    let overall_acc = if total_test_bars > 0 { total_correct as f64 / total_test_bars as f64 } else { 0.50 };

    let total_strong_bars: usize = folds.iter().map(|f| f.test_n_strong).sum();
    let total_strong_correct: usize = folds.iter().map(|f| (f.test_strong_accuracy * f.test_n_strong as f64).round() as usize).sum();
    let overall_strong_acc = if total_strong_bars > 0 { total_strong_correct as f64 / total_strong_bars as f64 } else { 0.50 };

    WalkForwardResult {
        folds,
        wf_validated_states: wf_validated,
        overall_accuracy: overall_acc,
        overall_strong_accuracy: overall_strong_acc,
        accuracy_std: acc_std,
        total_test_bars,
    }
}

/// Helper: compute fallback bias series for a sub-range of the full dataset.
fn compute_fallback_for_range(
    quintiles: &[Vec<u8>],
    outcomes: &[u8],
    validated: &HashMap<StateKey, ValidatedState>,
    all_stats: &HashMap<StateKey, StateStats>,
    start: usize,
    end: usize,
) -> Vec<FallbackResult> {
    let mut results = Vec::with_capacity(end - start);

    for i in start..end {
        let baseline = fallback::compute_baseline_bias(outcomes, i.min(outcomes.len()));

        let mut q = [0u8; N_FEATURES];
        for f in 0..N_FEATURES {
            q[f] = if i < quintiles[f].len() { quintiles[f][i] } else { 0 };
        }

        results.push(fallback::compute_bar_bias(&q, validated, all_stats, baseline));
    }

    results
}

/// Helper: extract sweep bias for a range of 5m bars.
fn compute_sweep_for_range(
    htf_series: &[htf::HtfSeries],
    start: usize,
    end: usize,
) -> Vec<f64> {
    // Compute full sweep series (this is fast)
    // We need the full series because sweep recency tracks from beginning
    let full_len = htf_series[0].bars.last().map(|b| b.end_idx + 1).unwrap_or(0);
    let n = full_len.max(end);
    let full_sweep = sweep::compute_sweep_bias_series(htf_series, n);

    // Extract the test range
    (start..end)
        .map(|i| {
            if i < full_sweep.len() {
                full_sweep[i].sweep_bias
            } else {
                0.0
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// Scoring-based Walk-Forward (uses combined scoring with BTC)
// ═══════════════════════════════════════════════════════════════

use crate::bias_engine::scoring;
use crate::bias_engine::params::{GroupAParams, GroupBParams, BiasEngineParams};

/// Scoring-based walk-forward result
#[derive(Clone, Debug)]
pub struct ScoringWfResult {
    pub n_chunks: usize,
    pub chunk_accuracies: Vec<f64>,
    pub overall_accuracy: f64,
    pub accuracy_std: f64,
    pub total_bars: usize,
    pub total_correct: usize,
}

/// Run scoring-based walk-forward with given parameters.
/// Train/test windows adapt to data size: train=40% of data, chunk=10% of data.
/// Falls back to TRAIN_1H/CHUNK_1H constants when data is large enough.
const TRAIN_1H: usize = 17_520;  // ~2 years at 1H
const CHUNK_1H: usize = 4_380;   // ~6 months at 1H

pub fn run_scoring_walkforward(
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
    high: &[f64],
    low: &[f64],
    btc_close: Option<&[f64]>,
    btc_buy_vol: Option<&[f64]>,
    btc_sell_vol: Option<&[f64]>,
    params_a: &GroupAParams,
    params_b: &GroupBParams,
) -> ScoringWfResult {
    let n = close.len();

    // Adaptive window sizing: use fixed constants for 1H+ data,
    // proportional sizing for smaller datasets (4H, 8H, etc.)
    let (train_size, chunk_size) = if n >= TRAIN_1H + CHUNK_1H * 2 {
        (TRAIN_1H, CHUNK_1H)
    } else {
        // Train on ~50% of data, test in ~10% chunks (minimum 3 chunks)
        let train = (n * 50) / 100;
        let chunk = ((n - train) / 3).max(100);
        (train, chunk)
    };

    // Generate synthetic hourly timestamps (1H bars starting from epoch)
    // Used for hour-of-day feature computation
    let timestamps: Vec<u64> = (0..n).map(|i| (i as u64) * 3_600_000).collect();

    // Build test chunks
    let mut test_chunks: Vec<(usize, usize)> = Vec::new();
    let mut idx = train_size;
    while idx + chunk_size <= n {
        test_chunks.push((idx, idx + chunk_size));
        idx += chunk_size;
    }

    // Compute bias engine pipeline (train on first train_size bars)
    let train_n = train_size.min(n);
    let feats = features::compute_features_with_params(
        &timestamps[..train_n],
        &high[..train_n], &low[..train_n], &close[..train_n],
        &buy_vol[..train_n], &sell_vol[..train_n], &oi[..train_n], params_a,
    );
    let quant = quantize::quantize_all_with_params(&feats, params_a.quant_window, params_a.quantile_count);
    let quintiles_train: Vec<Vec<u8>> = quant.iter().map(|qr| qr.quintiles.clone()).collect();
    let boundaries_train: Vec<Vec<[f64; quantize::MAX_BOUNDARIES]>> = quant.iter().map(|qr| qr.boundaries.clone()).collect();

    let prob = probability::compute_probabilities_with_params(
        &close[..train_n], &quintiles_train, train_n,
        params_a.k_horizon, params_a.min_sample_size, params_a.min_edge, params_a.prior_strength,
    );

    let significant: HashMap<StateKey, probability::StateStats> = prob.state_stats
        .iter().filter(|(_, s)| s.significant).map(|(&k, s)| (k, s.clone())).collect();

    let feature_vecs: Vec<Vec<f64>> = (0..N_FEATURES).map(|i| feats.get(i).to_vec()).collect();

    let rob = robustness::run_robustness_with_params(
        &close[..train_n], &quintiles_train, &boundaries_train, &feature_vecs,
        &prob.outcomes, prob.baseline_bull_rate, &significant, train_n,
        params_a.fdr_alpha, params_a.temporal_min_segments,
        params_a.temporal_max_reversals, params_a.min_noise_stability,
        params_a.quantile_count,
    );

    let mut validated: HashMap<StateKey, fallback::ValidatedState> = HashMap::new();
    for (&key, r) in &rob.results {
        if r.validated {
            if let Some(stats) = prob.state_stats.get(&key) {
                validated.insert(key, fallback::ValidatedState { stats: stats.clone(), robustness: r.clone() });
            }
        }
    }

    // Full-range features + quintiles + bias values
    let feats_full = features::compute_features_with_params(&timestamps, high, low, close, buy_vol, sell_vol, oi, params_a);
    let quant_full = quantize::quantize_all_with_params(&feats_full, params_a.quant_window, params_a.quantile_count);
    let quintiles_full: Vec<Vec<u8>> = quant_full.iter().map(|qr| qr.quintiles.clone()).collect();
    let outcomes_full = probability::compute_outcomes(close, params_a.k_horizon);

    let mut bias_values = vec![0.0f64; n];
    for i in 0..n {
        let baseline = fallback::compute_baseline_bias(&outcomes_full, i);
        let mut q = [0u8; N_FEATURES];
        let mut valid = true;
        for f in 0..N_FEATURES {
            if i < quintiles_full[f].len() { q[f] = quintiles_full[f][i]; }
            if q[f] == 0 { valid = false; }
        }
        if valid {
            let fb = fallback::compute_bar_bias(&q, &validated, &prob.state_stats, baseline);
            bias_values[i] = fb.bias;
        } else {
            bias_values[i] = baseline;
        }
    }

    // Compute combined scores (with BTC)
    let scored = scoring::compute_combined_scores(
        &bias_values, close, buy_vol, sell_vol,
        btc_close, btc_buy_vol, btc_sell_vol,
        params_b,
    );

    // Evaluate on test chunks
    let mut chunk_accs: Vec<f64> = Vec::new();
    let mut total_correct = 0usize;
    let mut total_bars = 0usize;

    for &(start, end) in &test_chunks {
        let mut correct = 0usize;
        let mut total = 0usize;
        for i in start..end.min(n) {
            if outcomes_full[i] > 1 { continue; }
            let dir = scored[i].direction;
            if dir == 0 { continue; }
            let actual_bull = outcomes_full[i] == 1;
            let pred_bull = dir > 0;
            total += 1;
            if pred_bull == actual_bull { correct += 1; }
        }
        if total >= 100 {
            chunk_accs.push(correct as f64 / total as f64);
            total_correct += correct;
            total_bars += total;
        }
    }

    let overall = if total_bars > 0 { total_correct as f64 / total_bars as f64 } else { 0.50 };
    let mean = if chunk_accs.is_empty() { 0.50 } else { chunk_accs.iter().sum::<f64>() / chunk_accs.len() as f64 };
    let std = if chunk_accs.len() > 1 {
        let var = chunk_accs.iter().map(|&a| (a - mean).powi(2)).sum::<f64>() / (chunk_accs.len() - 1) as f64;
        var.sqrt()
    } else { 0.0 };

    ScoringWfResult {
        n_chunks: chunk_accs.len(),
        chunk_accuracies: chunk_accs,
        overall_accuracy: overall,
        accuracy_std: std,
        total_bars,
        total_correct,
    }
}

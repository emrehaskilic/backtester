/// Bias Engine — Continuous Bias System (Steps 1–12)
///
/// Step 1:  Data pipeline  — feature computation + HTF aggregation
/// Step 2:  Quantization   — rolling quintile + state enumeration / matching
/// Step 3:  Probability    — outcome computation + Bayesian smoothing + significance
/// Step 4:  Robustness     — permutation + noise injection + temporal subsample + BH FDR
/// Step 5:  Fallback       — hierarchy (depth-3 > 2 > 1 > baseline) + continuous output
/// Step 6:  Calibration    — isotonic regression
/// Step 7:  Regime         — trending / mean-reverting / high-vol detection
/// Step 8:  Sweep overlay  — multi-TF sweep continuation/reversal bias
/// Step 9:  Final bias     — state + sweep combination + confidence score
/// Step 10: Decay          — exponential time decay + online update
/// Step 11: Monitoring     — edge decay monitoring + state health
/// Step 12: Walk-forward   — rolling train/val/test evaluation

pub mod features;
pub mod htf;
pub mod probability;
pub mod quantize;
pub mod robustness;
pub mod state;
pub mod fallback;
pub mod calibration;
pub mod regime;
pub mod sweep;
pub mod confidence;
pub mod final_bias;
pub mod decay;
pub mod monitoring;
pub mod walkforward;
pub mod params;
pub mod scoring;
pub mod optimizer;

use features::FeatureArrays;
use probability::StateStats;
use robustness::RobustnessResult;
use state::StateKey;
use std::collections::HashMap;

/// Result of the Step 1–2 analysis.
pub struct AnalysisResult {
    pub n_bars: usize,
    /// Bars where all 7 features are non-NaN
    pub n_valid_bars: usize,
    /// Bars where all 7 quintiles are valid (non-zero)
    pub n_quantized_bars: usize,
    /// Warmup bars before quantization starts
    pub warmup_bars: usize,

    /// 7 feature arrays, each length n_bars
    pub features: FeatureArrays,

    /// 7 quintile arrays (0 = invalid, 1–5)
    pub quintiles: Vec<Vec<u8>>,

    /// HTF bar counts per timeframe: [(name, count)]
    pub htf_counts: Vec<(String, usize)>,

    /// State occurrence counts
    pub state_counts: HashMap<StateKey, u32>,
    pub depth1_active: usize,
    pub depth2_active: usize,
    pub depth3_active: usize,
}

/// Result of the full Step 1–4 pipeline.
pub struct FullAnalysisResult {
    // ── Step 1-2 ──
    pub n_bars: usize,
    pub n_valid_bars: usize,
    pub n_quantized_bars: usize,
    pub warmup_bars: usize,
    pub features: FeatureArrays,
    pub quintiles: Vec<Vec<u8>>,
    pub htf_counts: Vec<(String, usize)>,
    pub state_counts: HashMap<StateKey, u32>,
    pub depth1_active: usize,
    pub depth2_active: usize,
    pub depth3_active: usize,

    // ── Step 3 ──
    pub outcomes: Vec<u8>,
    pub baseline_bull_rate: f64,
    pub state_stats: HashMap<StateKey, StateStats>,
    pub n_significant: usize,
    pub sig_depth1: usize,
    pub sig_depth2: usize,
    pub sig_depth3: usize,

    // ── Step 4 ──
    pub robustness: HashMap<StateKey, RobustnessResult>,
    pub n_validated: usize,
    pub val_depth1: usize,
    pub val_depth2: usize,
    pub val_depth3: usize,
}

/// Per-bar output from the full bias pipeline (Steps 5–9).
#[derive(Clone, Debug)]
pub struct BarBiasOutput {
    pub final_bias: f64,
    pub direction: final_bias::BiasDirection,
    pub strength: final_bias::BiasStrength,
    pub confidence: f64,

    // Components
    pub state_bias: f64,
    pub calibrated_state_bias: f64,
    pub sweep_bias: f64,
    pub alignment: final_bias::Alignment,
    pub regime_shift_penalty: bool,

    // State info
    pub matched_depth: u8,
    pub fallback_used: bool,
    pub matched_state_count: usize,

    // Regime
    pub regime: regime::Regime,
}

/// Result of the full Steps 1–9 pipeline (batch bias computation).
pub struct BiasSeriesResult {
    // ── Steps 1-4 summary ──
    pub n_bars: usize,
    pub n_validated: usize,
    pub baseline_bull_rate: f64,

    // ── Steps 5-9: per-bar bias ──
    pub bar_outputs: Vec<BarBiasOutput>,

    // ── Coverage stats ──
    pub coverage_pct: f64,      // should be 100%
    pub fallback_pct: f64,      // % of bars using baseline fallback
    pub depth3_pct: f64,        // % matching depth-3
    pub depth2_pct: f64,
    pub depth1_pct: f64,

    // ── Accuracy (in-sample, informational) ──
    pub direction_accuracy: f64,
    pub strong_signal_accuracy: f64,
    pub n_strong_bars: usize,

    // ── Regime distribution ──
    pub pct_trending: f64,
    pub pct_mean_reverting: f64,
    pub pct_high_vol: f64,

    // ── Calibration ──
    pub brier_uncalibrated: f64,
    pub brier_calibrated: f64,
}

/// Run the full Step 1–2 pipeline.
pub fn analyze(
    timestamps: &[u64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
) -> AnalysisResult {
    let n = close.len();

    // ── Step 1a: Compute 7 features ──
    let feats = features::compute_features(timestamps, high, low, close, buy_vol, sell_vol, oi);

    // ── Step 1b: HTF aggregation ──
    let htf_series = htf::build_all_htf(timestamps, open, high, low, close, buy_vol, sell_vol);
    let htf_counts: Vec<(String, usize)> = htf_series
        .iter()
        .map(|s| (s.period_name.to_string(), s.bars.len()))
        .collect();

    // Count bars with all features valid
    let n_valid = (0..n).filter(|&i| feats.all_valid(i)).count();

    // ── Step 2a: Quantize all features (parallel via rayon) ──
    let quintiles = quantize::quantize_all(&feats);

    // Count bars with all quintiles valid
    let n_quantized = (0..n)
        .filter(|&i| (0..features::N_FEATURES).all(|f| quintiles[f][i] > 0))
        .count();

    // ── Step 2b: State counting ──
    let state_counts = state::count_states(&quintiles, n);

    let depth1_active = state_counts
        .keys()
        .filter(|&&k| state::state_depth(k) == 1)
        .count();
    let depth2_active = state_counts
        .keys()
        .filter(|&&k| state::state_depth(k) == 2)
        .count();
    let depth3_active = state_counts
        .keys()
        .filter(|&&k| state::state_depth(k) == 3)
        .count();

    AnalysisResult {
        n_bars: n,
        n_valid_bars: n_valid,
        n_quantized_bars: n_quantized,
        warmup_bars: quantize::QUANT_WINDOW,
        features: feats,
        quintiles,
        htf_counts,
        state_counts,
        depth1_active,
        depth2_active,
        depth3_active,
    }
}

/// Run the full Step 1–4 pipeline (analyze + probability + robustness).
pub fn analyze_full(
    timestamps: &[u64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
) -> FullAnalysisResult {
    let n = close.len();

    // ── Step 1–2 ──
    let feats = features::compute_features(timestamps, high, low, close, buy_vol, sell_vol, oi);
    let htf_series = htf::build_all_htf(timestamps, open, high, low, close, buy_vol, sell_vol);
    let htf_counts: Vec<(String, usize)> = htf_series
        .iter()
        .map(|s| (s.period_name.to_string(), s.bars.len()))
        .collect();
    let n_valid = (0..n).filter(|&i| feats.all_valid(i)).count();

    // Use quantize_all_full to get both quintiles AND boundaries (for noise injection)
    let quant_results = quantize::quantize_all_full(&feats);
    let quintiles: Vec<Vec<u8>> = quant_results.iter().map(|qr| qr.quintiles.clone()).collect();
    let boundaries: Vec<Vec<[f64; quantize::MAX_BOUNDARIES]>> = quant_results.iter().map(|qr| qr.boundaries.clone()).collect();

    let n_quantized = (0..n)
        .filter(|&i| (0..features::N_FEATURES).all(|f| quintiles[f][i] > 0))
        .count();
    let state_counts = state::count_states(&quintiles, n);

    let depth1_active = state_counts
        .keys()
        .filter(|&&k| state::state_depth(k) == 1)
        .count();
    let depth2_active = state_counts
        .keys()
        .filter(|&&k| state::state_depth(k) == 2)
        .count();
    let depth3_active = state_counts
        .keys()
        .filter(|&&k| state::state_depth(k) == 3)
        .count();

    // ── Step 3: Probability ──
    let prob_result = probability::compute_probabilities(close, &quintiles, n);

    // ── Step 4: Robustness ──
    // Collect significant states only
    let significant_states: HashMap<StateKey, StateStats> = prob_result
        .state_stats
        .iter()
        .filter(|(_, s)| s.significant)
        .map(|(&k, s)| (k, s.clone()))
        .collect();

    // Prepare feature arrays as Vec<Vec<f64>> for robustness
    let feature_vecs: Vec<Vec<f64>> = (0..features::N_FEATURES)
        .map(|i| feats.get(i).to_vec())
        .collect();

    let rob_result = robustness::run_robustness(
        close,
        &quintiles,
        &boundaries,
        &feature_vecs,
        &prob_result.outcomes,
        prob_result.baseline_bull_rate,
        &significant_states,
        n,
    );

    FullAnalysisResult {
        n_bars: n,
        n_valid_bars: n_valid,
        n_quantized_bars: n_quantized,
        warmup_bars: quantize::QUANT_WINDOW,
        features: feats,
        quintiles,
        htf_counts,
        state_counts,
        depth1_active,
        depth2_active,
        depth3_active,

        outcomes: prob_result.outcomes,
        baseline_bull_rate: prob_result.baseline_bull_rate,
        state_stats: prob_result.state_stats,
        n_significant: prob_result.n_significant,
        sig_depth1: prob_result.sig_depth1,
        sig_depth2: prob_result.sig_depth2,
        sig_depth3: prob_result.sig_depth3,

        robustness: rob_result.results,
        n_validated: rob_result.n_validated,
        val_depth1: rob_result.val_depth1,
        val_depth2: rob_result.val_depth2,
        val_depth3: rob_result.val_depth3,
    }
}

/// Run the full Steps 1–9 pipeline: batch bias computation.
///
/// This is the main entry point for computing per-bar bias values.
/// It runs Steps 1–4 (analysis), then Steps 5–9 (fallback, calibration,
/// regime, sweep, final combination).
pub fn compute_bias_series(
    timestamps: &[u64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
) -> BiasSeriesResult {
    let n = close.len();

    // ── Steps 1–4 ──
    let analysis = analyze_full(timestamps, open, high, low, close, buy_vol, sell_vol, oi);

    // Build validated state map for fallback
    let mut validated: HashMap<StateKey, fallback::ValidatedState> = HashMap::new();
    for (&key, rob) in &analysis.robustness {
        if rob.validated {
            if let Some(stats) = analysis.state_stats.get(&key) {
                validated.insert(key, fallback::ValidatedState {
                    stats: stats.clone(),
                    robustness: rob.clone(),
                });
            }
        }
    }

    // ── Step 5: Fallback — compute per-bar state bias ──
    let fallback_results = fallback::compute_bias_series(
        &analysis.quintiles,
        &analysis.outcomes,
        &validated,
        &analysis.state_stats,
        n,
    );

    // ── Step 6: Calibration ──
    // Split: first 80% for calibration fit, apply to all
    let cal_end = (n * 4) / 5;
    let cal_probs: Vec<f64> = fallback_results[..cal_end]
        .iter()
        .map(|f| f.smoothed_prob)
        .collect();
    let calibrator = calibration::IsotonicCalibrator::fit(&cal_probs, &analysis.outcomes[..cal_end]);

    // ── Step 7: Regime detection ──
    let atr_pct = analysis.features.get(6); // feature #6 = atr_percentile
    let regimes = regime::detect_regimes(close, high, low, atr_pct);

    // ── Step 8: Sweep overlay ──
    let htf_series = htf::build_all_htf(timestamps, open, high, low, close, buy_vol, sell_vol);
    let sweep_results = sweep::compute_sweep_bias_series(&htf_series, n);

    // ── Step 9: Final bias computation ──
    let mut bar_outputs = Vec::with_capacity(n);
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut strong_correct = 0usize;
    let mut strong_total = 0usize;
    let mut brier_uncal = 0.0f64;
    let mut brier_cal = 0.0f64;
    let mut brier_n = 0usize;
    let mut n_fallback = 0usize;
    let mut n_d3 = 0usize;
    let mut n_d2 = 0usize;
    let mut n_d1 = 0usize;
    let mut n_trending = 0usize;
    let mut n_mr = 0usize;
    let mut n_hv = 0usize;

    for i in 0..n {
        let fb = &fallback_results[i];
        let sweep_bias = sweep_results[i].sweep_bias;
        let regime_info = &regimes[i];

        // Calibrate the state probability
        let cal_prob = calibrator.transform(fb.smoothed_prob);
        let cal_bias = cal_prob - 0.50;

        // Confidence
        let conf = if fb.fallback_used {
            confidence::compute_fallback_confidence()
        } else {
            let (n_tot, ci_w, ns) = if let Some(&key) = fb.matched_states.first() {
                if let Some(vs) = validated.get(&key) {
                    (vs.stats.n_total, vs.stats.ci_95_half * 2.0, vs.robustness.noise_stability)
                } else {
                    (0, 0.20, 0.0)
                }
            } else {
                (0, 0.20, 0.0)
            };
            confidence::compute_confidence(n_tot, ci_w, ns, true)
        };

        // Final combination
        let final_out = final_bias::compute_final_bias(
            cal_bias,
            sweep_bias,
            regime_info.regime_shift,
            conf,
        );

        // Track metrics
        match fb.matched_depth {
            0 => n_fallback += 1,
            1 => n_d1 += 1,
            2 => n_d2 += 1,
            3 => n_d3 += 1,
            _ => {}
        }
        match regime_info.regime {
            regime::Regime::Trending => n_trending += 1,
            regime::Regime::MeanReverting => n_mr += 1,
            regime::Regime::HighVolatility => n_hv += 1,
        }

        if analysis.outcomes[i] != 255 {
            let actual_bull = analysis.outcomes[i] == 1;
            let pred_bull = final_out.final_bias > 0.0;
            total += 1;
            if pred_bull == actual_bull { correct += 1; }
            if final_out.final_bias.abs() > 0.15 {
                strong_total += 1;
                if pred_bull == actual_bull { strong_correct += 1; }
            }

            let actual_f = if actual_bull { 1.0 } else { 0.0 };
            brier_uncal += (fb.smoothed_prob - actual_f).powi(2);
            brier_cal += (cal_prob - actual_f).powi(2);
            brier_n += 1;
        }

        bar_outputs.push(BarBiasOutput {
            final_bias: final_out.final_bias,
            direction: final_out.direction,
            strength: final_out.strength,
            confidence: final_out.confidence,
            state_bias: fb.bias,
            calibrated_state_bias: cal_bias,
            sweep_bias,
            alignment: final_out.alignment,
            regime_shift_penalty: final_out.regime_shift_penalty,
            matched_depth: fb.matched_depth,
            fallback_used: fb.fallback_used,
            matched_state_count: fb.matched_states.len(),
            regime: regime_info.regime,
        });
    }

    let n_f = n as f64;
    BiasSeriesResult {
        n_bars: n,
        n_validated: analysis.n_validated,
        baseline_bull_rate: analysis.baseline_bull_rate,
        bar_outputs,

        coverage_pct: 100.0, // guaranteed by fallback
        fallback_pct: n_fallback as f64 / n_f * 100.0,
        depth3_pct: n_d3 as f64 / n_f * 100.0,
        depth2_pct: n_d2 as f64 / n_f * 100.0,
        depth1_pct: n_d1 as f64 / n_f * 100.0,

        direction_accuracy: if total > 0 { correct as f64 / total as f64 } else { 0.50 },
        strong_signal_accuracy: if strong_total > 0 { strong_correct as f64 / strong_total as f64 } else { 0.50 },
        n_strong_bars: strong_total,

        pct_trending: n_trending as f64 / n_f * 100.0,
        pct_mean_reverting: n_mr as f64 / n_f * 100.0,
        pct_high_vol: n_hv as f64 / n_f * 100.0,

        brier_uncalibrated: if brier_n > 0 { brier_uncal / brier_n as f64 } else { 0.0 },
        brier_calibrated: if brier_n > 0 { brier_cal / brier_n as f64 } else { 0.0 },
    }
}

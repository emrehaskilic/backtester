/// Bias Engine — TPE Optimizer (2-Phase Nested)
///
/// Outer loop: Group A (18 params) — full bias engine recompute per trial
/// Inner loop: Group B (20 params) — fast scoring-only optimization
///
/// Walk-forward objective: train first 2 years, test remaining 3 years in 6-month chunks
/// Score = mean(chunk_accuracies) - 0.5 * std(chunk_accuracies)

use rayon::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::io::Write;

use super::features::{self, N_FEATURES};
use super::params::*;
use super::probability::{self, StateStats};
use super::quantize;
use super::robustness;
use super::state::{self, StateKey};
use super::fallback::{self, ValidatedState};
use super::scoring;

// ═══════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════

const GAMMA: f64 = 0.25;
const INNER_TRIALS: usize = 50;
const INNER_STARTUP: usize = 15;

const TRAIN_BARS: usize = 17_520;  // ~2 years 1H
const CHUNK_SIZE: usize = 4_380;   // ~6 months 1H

// ═══════════════════════════════════════════════════════════════
// TPE Sampling (reused from sweep_kc_optimizer pattern)
// ═══════════════════════════════════════════════════════════════

fn sample_random(spec: &ParamSpec, rng: &mut ChaCha8Rng) -> f64 {
    let steps = ((spec.max - spec.min) / spec.step).round() as i32;
    if steps <= 0 { return spec.min; }
    let s = rng.gen_range(0..=steps);
    let val = spec.min + s as f64 * spec.step;
    if spec.is_int { val.round() } else { val }
}

fn sample_tpe(spec: &ParamSpec, good_vals: &[f64], rng: &mut ChaCha8Rng) -> f64 {
    if good_vals.is_empty() { return sample_random(spec, rng); }
    let steps = ((spec.max - spec.min) / spec.step).round() as usize;
    let mut counts = vec![1u32; steps + 1];
    for &v in good_vals {
        let idx = ((v - spec.min) / spec.step).round() as usize;
        if idx <= steps { counts[idx] += 10; }
    }
    let total: u32 = counts.iter().sum();
    let r = rng.gen_range(0..total);
    let mut cum = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        cum += c;
        if r < cum {
            let val = spec.min + i as f64 * spec.step;
            return if spec.is_int { val.round() } else { val };
        }
    }
    spec.max
}

fn perturb(spec: &ParamSpec, val: f64, scale: f64, rng: &mut ChaCha8Rng) -> f64 {
    let range = (spec.max - spec.min) * scale;
    let delta = rng.gen_range(-range..=range);
    let raw = (val + delta).clamp(spec.min, spec.max);
    let steps = ((raw - spec.min) / spec.step).round();
    let val = spec.min + steps * spec.step;
    if spec.is_int { val.round() } else { val }
}

fn sample_param_vec(specs: &[ParamSpec], rng: &mut ChaCha8Rng) -> Vec<f64> {
    specs.iter().map(|s| sample_random(s, rng)).collect()
}

fn sample_tpe_vec(specs: &[ParamSpec], good_vecs: &[Vec<f64>], rng: &mut ChaCha8Rng) -> Vec<f64> {
    specs.iter().enumerate().map(|(i, spec)| {
        let good_vals: Vec<f64> = good_vecs.iter().map(|v| v[i]).collect();
        sample_tpe(spec, &good_vals, rng)
    }).collect()
}

fn perturb_vec(specs: &[ParamSpec], base: &[f64], scale: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
    specs.iter().enumerate().map(|(i, spec)| perturb(spec, base[i], scale, rng)).collect()
}

// ═══════════════════════════════════════════════════════════════
// Pre-computed cache for fast inner loop
// ═══════════════════════════════════════════════════════════════

/// Cache of values that depend on Group A but not Group B.
/// Computed once per outer trial, reused for all 200 inner trials.
struct InnerCache {
    bias_values: Vec<f64>,      // per-bar bias from engine
    outcomes: Vec<u8>,          // per-bar outcomes (based on k_horizon)
    cvd_zscore: Vec<f64>,       // CVD z-score for scoring
    close: Vec<f64>,            // close prices (reference)
    buy_vol: Vec<f64>,
    sell_vol: Vec<f64>,
    // BTC data (optional — None if not provided)
    btc_close: Option<Vec<f64>>,
    btc_buy_vol: Option<Vec<f64>>,
    btc_sell_vol: Option<Vec<f64>>,
    test_chunks: Vec<(usize, usize)>,
}

/// Fast inner evaluation using cached values.
/// Only computes EMA + RSI + weights — no bias engine, no outcomes recompute.
fn evaluate_inner_fast(cache: &InnerCache, params_b: &GroupBParams) -> f64 {
    let n = cache.close.len();

    // EMA (fast O(n) computation)
    let ema1 = scoring::compute_ema(&cache.close, params_b.mr_ema_span1);
    let ema2 = scoring::compute_ema(&cache.close, params_b.mr_ema_span2);
    let rsi = scoring::compute_rsi(&cache.close, params_b.rsi_period);

    // BTC features (pre-compute per inner trial — window params change)
    let btc_mom = cache.btc_close.as_ref()
        .map(|bc| scoring::compute_btc_momentum(bc, params_b.btc_mom_window));
    let btc_lead = cache.btc_close.as_ref()
        .map(|bc| scoring::compute_btc_lead(bc, &cache.close, params_b.btc_lead_window));
    let btc_cvd_z = match (&cache.btc_buy_vol, &cache.btc_sell_vol) {
        (Some(bbv), Some(bsv)) => Some(scoring::compute_btc_cvd_zscore(bbv, bsv, 24)),
        _ => None,
    };

    // Build scores
    let mut scores = vec![0.0f64; n];
    for i in 0..n {
        let mut score = 0.0f64;

        // Bias engine
        score += cache.bias_values[i] * params_b.w_bias;

        // MR primary
        let mr1 = if cache.close[i] > ema1[i] { -1.0f64 } else { 1.0 };
        score += mr1 * params_b.w_mr1;

        // MR secondary (agree only)
        let mr2 = if cache.close[i] > ema2[i] { -1.0f64 } else { 1.0 };
        if mr1 == mr2 { score += mr1 * params_b.w_mr2; }

        // RSI
        let rsi_os = rsi[i] < (50.0 - params_b.rsi_threshold);
        let rsi_ob = rsi[i] > (50.0 + params_b.rsi_threshold);
        if (rsi_os && mr1 > 0.0) || (rsi_ob && mr1 < 0.0) {
            score += mr1 * params_b.w_rsi;
        }

        // Agreement
        if (cache.bias_values[i] > 0.0 && mr1 > 0.0) || (cache.bias_values[i] < 0.0 && mr1 < 0.0) {
            score += mr1.copysign(cache.bias_values[i]) * params_b.w_agree;
        }

        // CVD
        if cache.cvd_zscore[i] > 0.5 { score += params_b.w_cvd; }
        else if cache.cvd_zscore[i] < -0.5 { score -= params_b.w_cvd; }

        // BTC momentum
        if let Some(ref bm) = btc_mom {
            if bm[i] > 0.5 { score += params_b.w_btc_mom; }
            else if bm[i] < -0.5 { score -= params_b.w_btc_mom; }
        }

        // BTC lead
        if let Some(ref bl) = btc_lead {
            if bl[i] > 0.005 { score += params_b.w_btc_lead; }
            else if bl[i] < -0.005 { score -= params_b.w_btc_lead; }
        }

        // BTC CVD
        if let Some(ref bcvd) = btc_cvd_z {
            if bcvd[i] > 0.5 { score += params_b.w_btc_cvd; }
            else if bcvd[i] < -0.5 { score -= params_b.w_btc_cvd; }
        }

        // Override
        if cache.bias_values[i].abs() >= params_b.bias_override_threshold {
            score = cache.bias_values[i] * params_b.w_bias * params_b.bias_override_mult;
        }

        scores[i] = score;
    }

    // Evaluate on test chunks
    let mut chunk_accs: Vec<f64> = Vec::new();
    for &(start, end) in &cache.test_chunks {
        let mut correct = 0usize;
        let mut total = 0usize;
        for i in start..end.min(n) {
            if cache.outcomes[i] > 1 { continue; }
            let dir = if scores[i] > 0.001 { 1 } else if scores[i] < -0.001 { -1 } else { 0 };
            if dir == 0 { continue; }
            let actual_bull = cache.outcomes[i] == 1;
            let pred_bull = dir > 0;
            total += 1;
            if pred_bull == actual_bull { correct += 1; }
        }
        if total >= 100 {
            chunk_accs.push(correct as f64 / total as f64);
        }
    }

    if chunk_accs.is_empty() { return 0.0; }
    let mean = chunk_accs.iter().sum::<f64>() / chunk_accs.len() as f64;
    let std = if chunk_accs.len() > 1 {
        let var = chunk_accs.iter().map(|&a| (a - mean).powi(2)).sum::<f64>() / (chunk_accs.len() - 1) as f64;
        var.sqrt()
    } else { 0.0 };
    mean - 0.5 * std
}

// ═══════════════════════════════════════════════════════════════
// Bias Pipeline (Group A)
// ═══════════════════════════════════════════════════════════════

fn run_bias_pipeline_and_cache(
    high: &[f64], low: &[f64], close: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    timestamps: &[u64],
    params: &GroupAParams,
    test_chunks: &[(usize, usize)],
    btc_close: Option<&[f64]>,
    btc_buy_vol: Option<&[f64]>,
    btc_sell_vol: Option<&[f64]>,
) -> (InnerCache, usize) {
    let n = close.len();
    let train_n = TRAIN_BARS.min(n);

    // Steps 1-4 on train set
    let feats = features::compute_features_with_params(
        &high[..train_n], &low[..train_n], &close[..train_n],
        &buy_vol[..train_n], &sell_vol[..train_n], &oi[..train_n], params,
    );
    let quant = quantize::quantize_all_with_params(&feats, params.quant_window, params.quantile_count);
    let quintiles: Vec<Vec<u8>> = quant.iter().map(|qr| qr.quintiles.clone()).collect();
    let boundaries: Vec<Vec<[f64; 4]>> = quant.iter().map(|qr| qr.boundaries.clone()).collect();

    let prob = probability::compute_probabilities_with_params(
        &close[..train_n], &quintiles, train_n,
        params.k_horizon, params.min_sample_size, params.min_edge, params.prior_strength,
    );

    let significant: HashMap<StateKey, StateStats> = prob.state_stats
        .iter().filter(|(_, s)| s.significant).map(|(&k, s)| (k, s.clone())).collect();

    let feature_vecs: Vec<Vec<f64>> = (0..N_FEATURES).map(|i| feats.get(i).to_vec()).collect();

    let rob = robustness::run_robustness_with_params(
        &close[..train_n], &quintiles, &boundaries, &feature_vecs,
        &prob.outcomes, prob.baseline_bull_rate, &significant, train_n,
        params.fdr_alpha, params.temporal_min_segments,
        params.temporal_max_reversals, params.min_noise_stability,
    );

    let mut validated: HashMap<StateKey, ValidatedState> = HashMap::new();
    for (&key, r) in &rob.results {
        if r.validated {
            if let Some(stats) = prob.state_stats.get(&key) {
                validated.insert(key, ValidatedState { stats: stats.clone(), robustness: r.clone() });
            }
        }
    }
    let n_validated = validated.len();

    // Compute full-range features + quintiles for scoring
    let feats_full = features::compute_features_with_params(high, low, close, buy_vol, sell_vol, oi, params);
    let quant_full = quantize::quantize_all_with_params(&feats_full, params.quant_window, params.quantile_count);
    let quintiles_full: Vec<Vec<u8>> = quant_full.iter().map(|qr| qr.quintiles.clone()).collect();
    let outcomes_full = probability::compute_outcomes(close, params.k_horizon);

    // Compute bias values for full range
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

    // CVD z-score (precompute once)
    let cvd_z = scoring::compute_cvd_zscore(buy_vol, sell_vol, 24);

    let cache = InnerCache {
        bias_values,
        outcomes: outcomes_full,
        cvd_zscore: cvd_z,
        close: close.to_vec(),
        buy_vol: buy_vol.to_vec(),
        sell_vol: sell_vol.to_vec(),
        btc_close: btc_close.map(|s| s.to_vec()),
        btc_buy_vol: btc_buy_vol.map(|s| s.to_vec()),
        btc_sell_vol: btc_sell_vol.map(|s| s.to_vec()),
        test_chunks: test_chunks.to_vec(),
    };

    (cache, n_validated)
}

// ═══════════════════════════════════════════════════════════════
// Inner TPE: Optimize Group B (fast)
// ═══════════════════════════════════════════════════════════════

fn optimize_group_b(cache: &InnerCache, seed: u64) -> (GroupBParams, f64) {
    let specs = group_b_specs();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut trials: Vec<(Vec<f64>, f64)> = Vec::with_capacity(INNER_TRIALS);

    // Phase 1: Random
    for _ in 0..INNER_STARTUP {
        let vals = sample_param_vec(&specs, &mut rng);
        let params_b = vec_to_group_b(&vals);
        let score = evaluate_inner_fast(cache, &params_b);
        trials.push((vals, score));
    }

    // Phase 2: TPE
    let n_tpe = INNER_TRIALS - INNER_STARTUP - (INNER_TRIALS / 5);
    for _ in 0..n_tpe {
        trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_good = (trials.len() as f64 * GAMMA).ceil() as usize;
        let good: Vec<Vec<f64>> = trials[..n_good].iter().map(|(v, _)| v.clone()).collect();
        let vals = sample_tpe_vec(&specs, &good, &mut rng);
        let params_b = vec_to_group_b(&vals);
        let score = evaluate_inner_fast(cache, &params_b);
        trials.push((vals, score));
    }

    // Phase 3: Refine
    trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let n_refine = INNER_TRIALS / 5;
    for r in 0..n_refine {
        let base_idx = r % 5;
        let scale = 0.15 * (1.0 - r as f64 / n_refine as f64);
        let vals = perturb_vec(&specs, &trials[base_idx].0, scale, &mut rng);
        let params_b = vec_to_group_b(&vals);
        let score = evaluate_inner_fast(cache, &params_b);
        trials.push((vals, score));
    }

    trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    (vec_to_group_b(&trials[0].0), trials[0].1)
}

// ═══════════════════════════════════════════════════════════════
// Result
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub best_params: BiasEngineParams,
    pub best_score: f64,
    pub chunk_accuracies: Vec<f64>,
    pub n_validated_states: usize,
    pub trials_evaluated: usize,
}

// ═══════════════════════════════════════════════════════════════
// Main: 2-phase nested TPE
// ═══════════════════════════════════════════════════════════════

pub fn run_optimization(
    timestamps: &[u64],
    high: &[f64], low: &[f64], close: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    btc_close: Option<&[f64]>,
    btc_buy_vol: Option<&[f64]>,
    btc_sell_vol: Option<&[f64]>,
    n_outer_trials: usize,
    seed: u64,
) -> OptimizationResult {
    let n = close.len();
    let specs_a = group_a_specs();

    let mut test_chunks: Vec<(usize, usize)> = Vec::new();
    let mut idx = TRAIN_BARS;
    while idx + CHUNK_SIZE <= n {
        test_chunks.push((idx, idx + CHUNK_SIZE));
        idx += CHUNK_SIZE;
    }

    let n_startup = n_outer_trials / 5;
    let n_tpe = n_outer_trials * 3 / 5;
    let n_refine = n_outer_trials - n_startup - n_tpe;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut all_trials: Vec<(Vec<f64>, f64, GroupBParams, usize)> = Vec::with_capacity(n_outer_trials);

    // Helper: evaluate one outer config
    let eval_outer = |vals: &[f64], inner_seed: u64| -> (f64, GroupBParams, usize) {
        let params_a = vec_to_group_a(vals);
        let (cache, n_val) = run_bias_pipeline_and_cache(
            high, low, close, buy_vol, sell_vol, oi, timestamps,
            &params_a, &test_chunks,
            btc_close, btc_buy_vol, btc_sell_vol,
        );
        let (best_b, score) = optimize_group_b(&cache, inner_seed);
        (score, best_b, n_val)
    };

    // Progress logging helper
    let log_progress = |phase: &str, trial_num: usize, total: usize, best_score: f64, n_val: usize| {
        let msg = format!(
            "[{}] trial {}/{} | best_score={:.4} | validated={}",
            phase, trial_num, total, best_score, n_val
        );
        eprintln!("{}", msg);
        // Also write to file
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true).append(true)
            .open("bias_optimizer_progress.log")
        {
            let _ = writeln!(f, "{}", msg);
        }
    };

    // Phase 1: Random (parallel)
    eprintln!("[STARTUP] Running {} parallel trials...", n_startup);
    let startup_configs: Vec<(Vec<f64>, u64)> = (0..n_startup)
        .map(|_| (sample_param_vec(&specs_a, &mut rng), rng.gen::<u64>()))
        .collect();

    let startup_results: Vec<(Vec<f64>, f64, GroupBParams, usize)> = startup_configs
        .par_iter()
        .map(|(vals, iseed)| {
            let (score, best_b, n_val) = eval_outer(vals, *iseed);
            (vals.clone(), score, best_b, n_val)
        })
        .collect();
    all_trials.extend(startup_results);

    // Log startup result
    all_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    log_progress("STARTUP", n_startup, n_outer_trials, all_trials[0].1, all_trials[0].3);

    // Phase 2: TPE (sequential — needs sorted results for sampling)
    eprintln!("[TPE] Running {} sequential trials...", n_tpe);
    for t in 0..n_tpe {
        all_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_good = (all_trials.len() as f64 * GAMMA).ceil() as usize;
        let good: Vec<Vec<f64>> = all_trials[..n_good].iter().map(|(v, _, _, _)| v.clone()).collect();
        let vals = sample_tpe_vec(&specs_a, &good, &mut rng);
        let iseed = rng.gen::<u64>();
        let (score, best_b, n_val) = eval_outer(&vals, iseed);
        all_trials.push((vals, score, best_b, n_val));

        // Log every 10 trials
        if (t + 1) % 10 == 0 {
            all_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            log_progress("TPE", n_startup + t + 1, n_outer_trials, all_trials[0].1, all_trials[0].3);
        }
    }

    // Phase 3: Refine
    eprintln!("[REFINE] Running {} refinement trials...", n_refine);
    all_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for r in 0..n_refine {
        let base_idx = r % 10;
        let scale = 0.10 * (1.0 - r as f64 / n_refine.max(1) as f64);
        let vals = perturb_vec(&specs_a, &all_trials[base_idx].0, scale, &mut rng);
        let iseed = rng.gen::<u64>();
        let (score, best_b, n_val) = eval_outer(&vals, iseed);
        all_trials.push((vals, score, best_b, n_val));

        if (r + 1) % 10 == 0 {
            all_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            log_progress("REFINE", n_startup + n_tpe + r + 1, n_outer_trials, all_trials[0].1, all_trials[0].3);
        }
    }

    all_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (best_a_vec, best_score, best_b, n_val) = &all_trials[0];

    OptimizationResult {
        best_params: BiasEngineParams { a: vec_to_group_a(best_a_vec), b: best_b.clone() },
        best_score: *best_score,
        chunk_accuracies: Vec::new(),
        n_validated_states: *n_val,
        trials_evaluated: all_trials.len(),
    }
}

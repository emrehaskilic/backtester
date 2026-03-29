/// Bias Engine — Outcome Computation + Bayesian Probability + Significance Filter
/// (Sections 4, 6, 7 of spec)
///
/// Step 3a: Compute outcome[i] = close[i+K] > close[i] → 1, else 0
/// Step 3b: Per-state stats (N_total, N_bull, raw_prob)
/// Step 3c: Bayesian smoothing (Beta-Binomial, prior_strength=50)
/// Step 3d: Significance filter (N>=100, edge>=0.03, CI rule)

use std::collections::HashMap;

use super::features::N_FEATURES;
use super::state::{self, StateKey};

/// Prediction horizon: 12 bars = 1 hour at 5m resolution
pub const K_HORIZON: usize = 12;

/// Minimum sample size for a state to be significant
pub const MIN_SAMPLE_SIZE: u32 = 100;

/// Minimum edge (|smoothed_prob - 0.50|) for a state to matter
pub const MIN_EDGE: f64 = 0.02;

/// Bayesian prior strength (pseudo-observation count)
pub const PRIOR_STRENGTH: f64 = 30.0;

/// Statistics for a single state.
#[derive(Debug, Clone)]
pub struct StateStats {
    pub n_total: u32,
    pub n_bull: u32,
    pub raw_prob: f64,
    pub smoothed_prob: f64,
    pub bias: f64,         // smoothed_prob - 0.50
    pub ci_95_half: f64,   // half-width of 95% CI
    pub significant: bool, // passes all 3 significance tests
}

/// Result of outcome + probability computation.
pub struct ProbabilityResult {
    /// outcome[i] = 1 if close[i+K] > close[i], 0 otherwise. NaN-like sentinel = u8::MAX for last K bars.
    pub outcomes: Vec<u8>,
    /// Baseline bull rate across entire dataset
    pub baseline_bull_rate: f64,
    /// Per-state statistics
    pub state_stats: HashMap<StateKey, StateStats>,
    /// Number of states passing significance filter
    pub n_significant: usize,
    /// Significant states per depth
    pub sig_depth1: usize,
    pub sig_depth2: usize,
    pub sig_depth3: usize,
}

/// Compute binary outcomes: 1 = bullish (close goes up in K bars), 0 = bearish.
/// Last K bars get sentinel value 255 (no outcome available).
pub fn compute_outcomes(close: &[f64], k: usize) -> Vec<u8> {
    let n = close.len();
    let mut outcomes = vec![255u8; n];

    if n <= k {
        return outcomes;
    }

    for i in 0..(n - k) {
        outcomes[i] = if close[i + k] > close[i] { 1 } else { 0 };
    }

    outcomes
}

/// Run the full probability pipeline:
/// 1. Compute outcomes
/// 2. Accumulate per-state stats
/// 3. Bayesian smooth
/// 4. Apply significance filter
pub fn compute_probabilities(
    close: &[f64],
    quintiles: &[Vec<u8>],
    n_bars: usize,
) -> ProbabilityResult {
    let outcomes = compute_outcomes(close, K_HORIZON);

    // ── Baseline bull rate ──
    let (total_valid_outcomes, total_bull) = outcomes
        .iter()
        .fold((0u64, 0u64), |(t, b), &o| {
            if o <= 1 {
                (t + 1, b + o as u64)
            } else {
                (t, b)
            }
        });
    let baseline_bull_rate = if total_valid_outcomes > 0 {
        total_bull as f64 / total_valid_outcomes as f64
    } else {
        0.50
    };

    // ── Per-state accumulation ──
    let mut raw_counts: HashMap<StateKey, (u32, u32)> = HashMap::with_capacity(state::TOTAL_STATES);

    for i in 0..n_bars {
        // Need valid outcome
        if outcomes[i] > 1 {
            continue;
        }
        let outcome = outcomes[i] as u32;

        // Need all quintiles valid
        let mut q = [0u8; N_FEATURES];
        let mut valid = true;
        for f in 0..N_FEATURES {
            q[f] = quintiles[f][i];
            if q[f] == 0 {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }

        // Match all 63 states for this bar
        let keys = state::match_bar_states(&q);
        for key in keys {
            let entry = raw_counts.entry(key).or_insert((0, 0));
            entry.0 += 1; // n_total
            entry.1 += outcome; // n_bull
        }
    }

    // ── Bayesian smoothing + significance ──
    let prior_alpha = baseline_bull_rate * PRIOR_STRENGTH;
    let prior_beta = (1.0 - baseline_bull_rate) * PRIOR_STRENGTH;

    let mut state_stats: HashMap<StateKey, StateStats> = HashMap::with_capacity(raw_counts.len());
    let mut n_significant = 0usize;
    let mut sig_depth1 = 0usize;
    let mut sig_depth2 = 0usize;
    let mut sig_depth3 = 0usize;

    for (&key, &(n_total, n_bull)) in &raw_counts {
        let raw_prob = if n_total > 0 {
            n_bull as f64 / n_total as f64
        } else {
            0.50
        };

        let smoothed_prob =
            (n_bull as f64 + prior_alpha) / (n_total as f64 + PRIOR_STRENGTH);

        let bias = smoothed_prob - 0.50;

        // 95% CI half-width
        let ci_95_half = if n_total > 0 {
            1.96 * (smoothed_prob * (1.0 - smoothed_prob) / n_total as f64).sqrt()
        } else {
            1.0 // huge CI → not significant
        };

        // Significance tests:
        // 1. N >= 100
        // 2. |excess edge over baseline| >= 0.03  (not over 0.50)
        // 3. CI doesn't cross baseline  (not 0.50)
        // This filters out states that merely ride the market trend.
        let excess_edge = smoothed_prob - baseline_bull_rate;
        let sig_n = n_total >= MIN_SAMPLE_SIZE;
        let sig_edge = excess_edge.abs() >= MIN_EDGE;
        let sig_ci = if excess_edge > 0.0 {
            smoothed_prob - ci_95_half > baseline_bull_rate
        } else {
            smoothed_prob + ci_95_half < baseline_bull_rate
        };
        let significant = sig_n && sig_edge && sig_ci;

        if significant {
            n_significant += 1;
            match state::state_depth(key) {
                1 => sig_depth1 += 1,
                2 => sig_depth2 += 1,
                3 => sig_depth3 += 1,
                _ => {}
            }
        }

        state_stats.insert(
            key,
            StateStats {
                n_total,
                n_bull,
                raw_prob,
                smoothed_prob,
                bias,
                ci_95_half,
                significant,
            },
        );
    }

    ProbabilityResult {
        outcomes,
        baseline_bull_rate,
        state_stats,
        n_significant,
        sig_depth1,
        sig_depth2,
        sig_depth3,
    }
}

/// Parametric version: custom K horizon, min_sample, min_edge, prior_strength
pub fn compute_probabilities_with_params(
    close: &[f64],
    quintiles: &[Vec<u8>],
    n_bars: usize,
    k_horizon: usize,
    min_sample_size: u32,
    min_edge: f64,
    prior_strength: f64,
) -> ProbabilityResult {
    let outcomes = compute_outcomes(close, k_horizon);

    let (total_valid_outcomes, total_bull) = outcomes
        .iter()
        .fold((0u64, 0u64), |(t, b), &o| {
            if o <= 1 { (t + 1, b + o as u64) } else { (t, b) }
        });
    let baseline_bull_rate = if total_valid_outcomes > 0 {
        total_bull as f64 / total_valid_outcomes as f64
    } else { 0.50 };

    let mut raw_counts: HashMap<StateKey, (u32, u32)> = HashMap::with_capacity(state::TOTAL_STATES);

    for i in 0..n_bars {
        if outcomes[i] > 1 { continue; }
        let outcome = outcomes[i] as u32;

        let n_feat = quintiles.len();
        let mut q = [0u8; N_FEATURES];
        let mut valid = true;
        for f in 0..n_feat.min(N_FEATURES) {
            q[f] = quintiles[f][i];
            if q[f] == 0 { valid = false; break; }
        }
        if !valid { continue; }

        let keys = state::match_bar_states(&q);
        for key in keys {
            let entry = raw_counts.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += outcome;
        }
    }

    let prior_alpha = baseline_bull_rate * prior_strength;

    let mut state_stats: HashMap<StateKey, StateStats> = HashMap::with_capacity(raw_counts.len());
    let mut n_significant = 0usize;
    let mut sig_depth1 = 0usize;
    let mut sig_depth2 = 0usize;
    let mut sig_depth3 = 0usize;

    for (&key, &(n_total, n_bull)) in &raw_counts {
        let raw_prob = if n_total > 0 { n_bull as f64 / n_total as f64 } else { 0.50 };
        let smoothed_prob = (n_bull as f64 + prior_alpha) / (n_total as f64 + prior_strength);
        let bias = smoothed_prob - 0.50;
        let ci_95_half = if n_total > 0 {
            1.96 * (smoothed_prob * (1.0 - smoothed_prob) / n_total as f64).sqrt()
        } else { 1.0 };

        let excess_edge = smoothed_prob - baseline_bull_rate;
        let sig_n = n_total >= min_sample_size;
        let sig_edge = excess_edge.abs() >= min_edge;
        let sig_ci = if excess_edge > 0.0 {
            smoothed_prob - ci_95_half > baseline_bull_rate
        } else {
            smoothed_prob + ci_95_half < baseline_bull_rate
        };
        let significant = sig_n && sig_edge && sig_ci;

        if significant {
            n_significant += 1;
            match state::state_depth(key) {
                1 => sig_depth1 += 1,
                2 => sig_depth2 += 1,
                3 => sig_depth3 += 1,
                _ => {}
            }
        }

        state_stats.insert(key, StateStats {
            n_total, n_bull, raw_prob, smoothed_prob, bias, ci_95_half, significant,
        });
    }

    ProbabilityResult {
        outcomes, baseline_bull_rate, state_stats,
        n_significant, sig_depth1, sig_depth2, sig_depth3,
    }
}

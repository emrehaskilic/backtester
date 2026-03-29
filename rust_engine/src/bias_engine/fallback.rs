/// Bias Engine — Step 5: Fallback Hierarchy + Continuous Output (Spec Section 9)
///
/// Ensures every bar has a bias value. Priority:
///   Level 1: Depth-3 validated state(s)
///   Level 2: Depth-2 validated state(s)
///   Level 3: Depth-1 validated state(s)
///   Level 4: Baseline bias (rolling outcome mean − 0.50)
///
/// Multiple validated states at same depth → confidence-weighted average.

use crate::bias_engine::features::N_FEATURES;
use crate::bias_engine::probability::StateStats;
use crate::bias_engine::robustness::RobustnessResult;
use crate::bias_engine::state::{self, StateKey};
use std::collections::HashMap;

/// Per-bar bias result from the fallback hierarchy.
#[derive(Clone, Debug)]
pub struct FallbackResult {
    /// The bias value for this bar (range roughly ±0.30)
    pub bias: f64,
    /// Smoothed probability (0–1) before converting to bias
    pub smoothed_prob: f64,
    /// Which depth matched (0 = baseline fallback, 1/2/3 = state depth)
    pub matched_depth: u8,
    /// Which validated states matched at the winning depth
    pub matched_states: Vec<StateKey>,
    /// True if no validated state matched (baseline used)
    pub fallback_used: bool,
}

/// Validated state info needed for fallback resolution.
#[derive(Clone, Debug)]
pub struct ValidatedState {
    pub stats: StateStats,
    pub robustness: RobustnessResult,
}

/// Compute confidence weight for a validated state (Section 9.4).
/// confidence = N_total × (1 − p_value) × noise_stability
fn state_confidence(vs: &ValidatedState) -> f64 {
    vs.stats.n_total as f64 * (1.0 - vs.robustness.perm_p_value) * vs.robustness.noise_stability
}

/// Compute the baseline bias for a given bar using a rolling window of outcomes.
/// Uses dual window: short (168 = 1 week on 1H) for trend, long (720 = 1 month) for stability.
/// Final baseline = weighted blend favoring the short window.
pub fn compute_baseline_bias(outcomes: &[u8], bar_idx: usize) -> f64 {
    const SHORT_WINDOW: usize = 168;  // ~1 week of 1H bars
    const LONG_WINDOW: usize = 720;   // ~1 month of 1H bars

    let compute_window = |window: usize| -> f64 {
        let start = if bar_idx >= window { bar_idx - window } else { 0 };
        let mut n = 0u32;
        let mut n_bull = 0u32;
        for i in start..bar_idx {
            if outcomes[i] != 255 {
                n += 1;
                if outcomes[i] == 1 { n_bull += 1; }
            }
        }
        if n < 10 { 0.0 } else { (n_bull as f64 / n as f64) - 0.50 }
    };

    let short_bias = compute_window(SHORT_WINDOW);
    let long_bias = compute_window(LONG_WINDOW);

    // Blend: 70% short (responsive) + 30% long (stable)
    short_bias * 0.70 + long_bias * 0.30
}

/// Compute bias for a single bar using the fallback hierarchy.
/// `all_stats` — all state stats (significant or not) for ensemble fallback.
pub fn compute_bar_bias(
    quintiles_at_bar: &[u8; N_FEATURES],
    validated: &HashMap<StateKey, ValidatedState>,
    all_stats: &HashMap<StateKey, crate::bias_engine::probability::StateStats>,
    baseline_bias: f64,
) -> FallbackResult {
    // Check if all quintiles are valid (non-zero)
    let all_valid = quintiles_at_bar.iter().all(|&q| q > 0);

    if !all_valid {
        return FallbackResult {
            bias: baseline_bias,
            smoothed_prob: baseline_bias + 0.50,
            matched_depth: 0,
            matched_states: Vec::new(),
            fallback_used: true,
        };
    }

    // Get all matching states for this bar
    let matching = state::match_bar_states(quintiles_at_bar);

    // Try each depth level: 3 → 2 → 1
    for target_depth in (1u32..=3).rev() {
        let mut matches: Vec<(StateKey, &ValidatedState)> = Vec::new();

        for &key in &matching {
            if state::state_depth(key) == target_depth {
                if let Some(vs) = validated.get(&key) {
                    matches.push((key, vs));
                }
            }
        }

        if !matches.is_empty() {
            // Confidence-weighted average of matched states at this depth
            let mut sum_weighted_bias = 0.0f64;
            let mut sum_confidence = 0.0f64;
            let mut sum_weighted_prob = 0.0f64;
            let mut matched_keys = Vec::with_capacity(matches.len());

            for (key, vs) in &matches {
                let conf = state_confidence(vs);
                sum_weighted_bias += vs.stats.bias * conf;
                sum_weighted_prob += vs.stats.smoothed_prob * conf;
                sum_confidence += conf;
                matched_keys.push(*key);
            }

            let combined_bias = if sum_confidence > 0.0 {
                sum_weighted_bias / sum_confidence
            } else {
                matches[0].1.stats.bias
            };

            let combined_prob = if sum_confidence > 0.0 {
                sum_weighted_prob / sum_confidence
            } else {
                matches[0].1.stats.smoothed_prob
            };

            return FallbackResult {
                bias: combined_bias,
                smoothed_prob: combined_prob,
                matched_depth: target_depth as u8,
                matched_states: matched_keys,
                fallback_used: false,
            };
        }
    }

    // No validated state matched → use ensemble of ALL matching states (N>=50)
    // Weight by sample size, gives a weak but continuous signal
    let mut ens_weighted_bias = 0.0f64;
    let mut ens_weight_sum = 0.0f64;
    let mut ens_weighted_prob = 0.0f64;

    for &key in &matching {
        if let Some(stats) = all_stats.get(&key) {
            if stats.n_total >= 50 {
                let w = stats.n_total as f64; // weight by sample size
                ens_weighted_bias += stats.bias * w;
                ens_weighted_prob += stats.smoothed_prob * w;
                ens_weight_sum += w;
            }
        }
    }

    if ens_weight_sum > 0.0 {
        let ens_bias = ens_weighted_bias / ens_weight_sum;
        let ens_prob = ens_weighted_prob / ens_weight_sum;
        // No dampening — always give a direction
        let dampened_bias = ens_bias;
        FallbackResult {
            bias: dampened_bias,
            smoothed_prob: ens_prob,
            matched_depth: 0,
            matched_states: Vec::new(),
            fallback_used: true,
        }
    } else {
        // Truly no data → baseline
        FallbackResult {
            bias: baseline_bias,
            smoothed_prob: baseline_bias + 0.50,
            matched_depth: 0,
            matched_states: Vec::new(),
            fallback_used: true,
        }
    }
}

/// Compute the full bias series for all bars.
pub fn compute_bias_series(
    quintiles: &[Vec<u8>],
    outcomes: &[u8],
    validated: &HashMap<StateKey, ValidatedState>,
    all_stats: &HashMap<StateKey, crate::bias_engine::probability::StateStats>,
    n_bars: usize,
) -> Vec<FallbackResult> {
    let mut results = Vec::with_capacity(n_bars);

    for i in 0..n_bars {
        let baseline = compute_baseline_bias(outcomes, i);

        let mut q = [0u8; N_FEATURES];
        for f in 0..N_FEATURES {
            q[f] = quintiles[f][i];
        }

        results.push(compute_bar_bias(&q, validated, all_stats, baseline));
    }

    results
}

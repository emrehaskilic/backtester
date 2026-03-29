/// Bias Engine — Step 10: Time Decay + Online Update (Spec Section 12)
///
/// Exponential decay on state counts so older observations fade.
/// decay_rate = 0.9999 per 5m bar:
///   1 day  (288 bars): 0.972 retention
///   1 week (2016 bars): 0.817 retention
///   1 month (8640 bars): 0.421 retention
///
/// Incremental update: each new bar updates K-bar-old outcomes.

use crate::bias_engine::features::N_FEATURES;
use crate::bias_engine::probability::{self, StateStats, PRIOR_STRENGTH};
use crate::bias_engine::state::{self, StateKey};
use std::collections::HashMap;

/// Per-bar decay rate (Section 12.1).
pub const DECAY_RATE: f64 = 0.9999;

/// Revalidation interval in bars (7 days at 5m).
pub const REVALIDATION_INTERVAL: usize = 2016;

/// Decayed state counts — floating point for smooth decay.
#[derive(Clone, Debug)]
pub struct DecayedStateStats {
    pub n_total: f64,
    pub n_bull: f64,
    /// Baseline bull rate for Bayesian smoothing
    pub baseline_bull_rate: f64,
}

impl DecayedStateStats {
    pub fn new(baseline_bull_rate: f64) -> Self {
        Self {
            n_total: 0.0,
            n_bull: 0.0,
            baseline_bull_rate,
        }
    }

    /// Initialize from discrete state stats.
    pub fn from_stats(stats: &StateStats, baseline_bull_rate: f64) -> Self {
        Self {
            n_total: stats.n_total as f64,
            n_bull: stats.n_bull as f64,
            baseline_bull_rate,
        }
    }

    /// Apply one step of exponential decay.
    pub fn decay(&mut self) {
        self.n_total *= DECAY_RATE;
        self.n_bull *= DECAY_RATE;
    }

    /// Add a new observation.
    pub fn add_observation(&mut self, is_bull: bool) {
        self.n_total += 1.0;
        if is_bull {
            self.n_bull += 1.0;
        }
    }

    /// Compute Bayesian smoothed probability.
    pub fn smoothed_prob(&self) -> f64 {
        let alpha = self.baseline_bull_rate * PRIOR_STRENGTH;
        (self.n_bull + alpha) / (self.n_total + PRIOR_STRENGTH)
    }

    /// Compute bias.
    pub fn bias(&self) -> f64 {
        self.smoothed_prob() - 0.50
    }
}

/// Online state tracker with exponential decay.
#[derive(Clone, Debug)]
pub struct OnlineStateTracker {
    /// Decayed statistics per state
    pub states: HashMap<StateKey, DecayedStateStats>,
    /// Global baseline bull rate (periodically updated)
    pub baseline_bull_rate: f64,
    /// Bar counter for revalidation scheduling
    pub bar_count: usize,
    /// Total decayed outcome counts for baseline
    pub total_n: f64,
    pub total_bull: f64,
}

impl OnlineStateTracker {
    /// Create from the validated states produced by batch analysis.
    pub fn from_batch(
        state_stats: &HashMap<StateKey, StateStats>,
        baseline_bull_rate: f64,
    ) -> Self {
        let states = state_stats
            .iter()
            .map(|(&key, stats)| {
                (key, DecayedStateStats::from_stats(stats, baseline_bull_rate))
            })
            .collect();

        Self {
            states,
            baseline_bull_rate,
            bar_count: 0,
            total_n: 0.0,
            total_bull: 0.0,
        }
    }

    /// Process a new bar: apply decay + update K-bar-old outcome.
    ///
    /// `quintiles_at_k_ago` — quintile tuple of the bar K bars ago (whose outcome we now know)
    /// `outcome_k_ago`      — the outcome of that bar (1=bull, 0=bear)
    /// `valid`              — whether the K-ago bar had valid quintiles
    pub fn update(
        &mut self,
        quintiles_at_k_ago: Option<&[u8; N_FEATURES]>,
        outcome_k_ago: Option<bool>,
    ) {
        self.bar_count += 1;

        // Step 1: Apply decay to all states
        for stats in self.states.values_mut() {
            stats.decay();
        }
        self.total_n *= DECAY_RATE;
        self.total_bull *= DECAY_RATE;

        // Step 2: If we have a resolved outcome from K bars ago, update
        if let (Some(quints), Some(is_bull)) = (quintiles_at_k_ago, outcome_k_ago) {
            // Update baseline
            self.total_n += 1.0;
            if is_bull {
                self.total_bull += 1.0;
            }
            if self.total_n > 0.0 {
                self.baseline_bull_rate = self.total_bull / self.total_n;
            }

            // Find matching states and update
            let matching = state::match_bar_states(quints);
            for key in matching {
                let entry = self.states
                    .entry(key)
                    .or_insert_with(|| DecayedStateStats::new(self.baseline_bull_rate));
                entry.add_observation(is_bull);
                entry.baseline_bull_rate = self.baseline_bull_rate;
            }
        }
    }

    /// Check if it's time for revalidation.
    pub fn needs_revalidation(&self) -> bool {
        self.bar_count > 0 && self.bar_count % REVALIDATION_INTERVAL == 0
    }

    /// Get the current smoothed probability for a state.
    pub fn get_bias(&self, key: StateKey) -> Option<f64> {
        self.states.get(&key).map(|s| s.bias())
    }
}

/// Apply batch decay to state stats for simulation purposes.
/// Simulates `n_bars` of decay on the counts.
pub fn apply_batch_decay(
    state_stats: &HashMap<StateKey, StateStats>,
    n_bars: usize,
    baseline_bull_rate: f64,
) -> HashMap<StateKey, DecayedStateStats> {
    let decay_factor = DECAY_RATE.powi(n_bars as i32);
    state_stats
        .iter()
        .map(|(&key, stats)| {
            let ds = DecayedStateStats {
                n_total: stats.n_total as f64 * decay_factor,
                n_bull: stats.n_bull as f64 * decay_factor,
                baseline_bull_rate,
            };
            (key, ds)
        })
        .collect()
}

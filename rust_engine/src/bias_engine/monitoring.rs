/// Bias Engine — Step 11: Edge Decay Monitoring (Spec Section 16)
///
/// Rolling metrics per state + global system health.
///
/// Per-state:
///   WARNING:  rolling edge sign reversal → weight × 0.50
///   DISABLE:  rolling_accuracy < 0.45 for 3 consecutive windows
///   RECOVER:  rolling_accuracy > 0.52 for 2 consecutive windows after disable
///
/// Global:
///   global_rolling_accuracy < 0.48 for 3 consecutive windows → global damping × 0.50

use crate::bias_engine::state::StateKey;
use std::collections::HashMap;

/// Rolling window size: 504 bars ≈ 42 hours ≈ 2.5 days.
pub const ROLLING_WINDOW: usize = 504;

/// Global monitoring window: 2016 bars = 7 days.
pub const GLOBAL_WINDOW: usize = 2016;

/// Per-state health status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateHealth {
    Healthy,
    Warning,
    Disabled,
}

/// Per-state monitoring metrics.
#[derive(Clone, Debug)]
pub struct StateMonitor {
    /// Rolling accuracy over last ROLLING_WINDOW bars where this state was active
    pub rolling_accuracy: f64,
    /// Rolling edge: mean(outcome) − 0.50 when state was active
    pub rolling_edge: f64,
    /// Trained (original) edge sign: true = bullish, false = bearish
    pub trained_edge_sign: bool,
    /// Current health status
    pub health: StateHealth,
    /// Consecutive windows with accuracy < 0.45
    pub consecutive_bad_windows: u32,
    /// Consecutive windows with accuracy > 0.52 (after disable)
    pub consecutive_good_windows: u32,
    /// Weight modifier (1.0 = full, 0.50 = halved due to warning)
    pub weight_modifier: f64,
}

/// Global system monitoring.
#[derive(Clone, Debug)]
pub struct GlobalMonitor {
    /// Rolling accuracy across all bias predictions
    pub rolling_accuracy: f64,
    /// Consecutive windows below 0.48
    pub consecutive_bad_windows: u32,
    /// Global damping factor (1.0 = normal, 0.50 = degraded)
    pub damping: f64,
}

/// Complete monitoring result for all states + global.
#[derive(Clone, Debug)]
pub struct MonitoringResult {
    pub states: HashMap<StateKey, StateMonitor>,
    pub global: GlobalMonitor,
}

/// Compute per-state monitoring metrics from a rolling window of observations.
///
/// `state_observations` — for each state: Vec of (bar_idx, predicted_bullish, actual_outcome)
/// `trained_edges`      — original trained bias sign per state
pub fn compute_state_monitors(
    state_observations: &HashMap<StateKey, Vec<(usize, bool, bool)>>,
    trained_edges: &HashMap<StateKey, bool>,
) -> HashMap<StateKey, StateMonitor> {
    let mut monitors = HashMap::new();

    for (&key, obs) in state_observations {
        if obs.is_empty() {
            continue;
        }

        let n = obs.len();
        let correct = obs.iter().filter(|&&(_, pred, actual)| pred == actual).count();
        let n_bull = obs.iter().filter(|&&(_, _, actual)| actual).count();

        let accuracy = if n > 0 { correct as f64 / n as f64 } else { 0.5 };
        let edge = if n > 0 { n_bull as f64 / n as f64 - 0.50 } else { 0.0 };

        let trained_sign = trained_edges.get(&key).copied().unwrap_or(true);
        let current_sign = edge >= 0.0;
        let sign_reversed = current_sign != trained_sign;

        let health = StateHealth::Healthy;
        let weight_modifier = if sign_reversed { 0.50 } else { 1.0 };

        monitors.insert(key, StateMonitor {
            rolling_accuracy: accuracy,
            rolling_edge: edge,
            trained_edge_sign: trained_sign,
            health,
            consecutive_bad_windows: if accuracy < 0.45 { 1 } else { 0 },
            consecutive_good_windows: if accuracy > 0.52 { 1 } else { 0 },
            weight_modifier,
        });
    }

    monitors
}

/// Update state health based on consecutive window counts.
pub fn update_state_health(monitor: &mut StateMonitor, window_accuracy: f64) {
    match monitor.health {
        StateHealth::Healthy | StateHealth::Warning => {
            // Check for edge sign reversal
            let current_sign = monitor.rolling_edge >= 0.0;
            if current_sign != monitor.trained_edge_sign {
                monitor.health = StateHealth::Warning;
                monitor.weight_modifier = 0.50;
            }

            // Check for persistent underperformance
            if window_accuracy < 0.45 {
                monitor.consecutive_bad_windows += 1;
                if monitor.consecutive_bad_windows >= 3 {
                    monitor.health = StateHealth::Disabled;
                    monitor.weight_modifier = 0.0;
                    monitor.consecutive_bad_windows = 0;
                }
            } else {
                monitor.consecutive_bad_windows = 0;
            }
        }
        StateHealth::Disabled => {
            // Check for recovery
            if window_accuracy > 0.52 {
                monitor.consecutive_good_windows += 1;
                if monitor.consecutive_good_windows >= 2 {
                    monitor.health = StateHealth::Warning; // re-enable at half weight
                    monitor.weight_modifier = 0.50;
                    monitor.consecutive_good_windows = 0;
                }
            } else {
                monitor.consecutive_good_windows = 0;
            }
        }
    }
}

/// Compute global monitoring metrics.
///
/// `predictions` — Vec of (predicted_bullish, actual_outcome) over global window
pub fn compute_global_monitor(
    predictions: &[(bool, bool)],
    prev_consecutive_bad: u32,
) -> GlobalMonitor {
    let n = predictions.len();
    let correct = predictions.iter().filter(|&&(pred, actual)| pred == actual).count();
    let accuracy = if n > 0 { correct as f64 / n as f64 } else { 0.50 };

    let consecutive_bad = if accuracy < 0.48 {
        prev_consecutive_bad + 1
    } else {
        0
    };

    let damping = if consecutive_bad >= 3 { 0.50 } else { 1.0 };

    GlobalMonitor {
        rolling_accuracy: accuracy,
        consecutive_bad_windows: consecutive_bad,
        damping,
    }
}

/// Compute rolling monitoring for the full bar series.
/// Used in batch analysis to evaluate state health across the dataset.
///
/// `final_biases` — per-bar final bias values
/// `outcomes`     — per-bar outcomes (1=bull, 0=bear, 255=skip)
/// `state_matches` — per-bar: which states matched (with their bias signs)
/// `trained_edges` — trained bias sign per state
pub fn compute_batch_monitoring(
    final_biases: &[f64],
    outcomes: &[u8],
    state_matches: &[Vec<(StateKey, bool)>], // (state, predicted_bullish)
    trained_edges: &HashMap<StateKey, bool>,
    n_bars: usize,
) -> MonitoringResult {
    // Collect per-state observations from the most recent ROLLING_WINDOW
    let window_start = if n_bars > ROLLING_WINDOW {
        n_bars - ROLLING_WINDOW
    } else {
        0
    };

    let mut state_obs: HashMap<StateKey, Vec<(usize, bool, bool)>> = HashMap::new();
    let mut global_preds: Vec<(bool, bool)> = Vec::new();

    for i in window_start..n_bars {
        if outcomes[i] == 255 {
            continue;
        }
        let actual_bull = outcomes[i] == 1;
        let pred_bull = final_biases[i] > 0.0;

        global_preds.push((pred_bull, actual_bull));

        for &(key, state_pred_bull) in &state_matches[i] {
            state_obs
                .entry(key)
                .or_default()
                .push((i, state_pred_bull, actual_bull));
        }
    }

    let state_monitors = compute_state_monitors(&state_obs, trained_edges);
    let global = compute_global_monitor(&global_preds, 0);

    MonitoringResult {
        states: state_monitors,
        global,
    }
}

/// Bias Engine — Step 8: Sweep Event Overlay (Spec Section 10)
///
/// Multi-TF sweep detection on HTF bars (1H/2H/4H/8H/1D).
/// Each TF independently detects sweep → continuation/reversal/ambiguous.
/// Recency decay + TF weighting → combined sweep bias (max ±0.30).

use crate::bias_engine::htf::HtfSeries;

/// Sweep type on an HTF bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepType {
    High,
    Low,
    Both,
    None,
}

/// Sweep result classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepResult {
    Continuation,
    Reversal,
    Ambiguous,
    None,
}

/// Per-TF sweep signal with recency info.
#[derive(Clone, Debug)]
pub struct TfSweepSignal {
    pub tf_name: &'static str,
    pub sweep_type: SweepType,
    pub result: SweepResult,
    /// Bars (5m) since this sweep event's HTF bar closed
    pub recency: usize,
    /// Decay factor: exp(−recency / half_life)
    pub decay: f64,
    /// Raw signal bias: +1.0 (bull), −1.0 (bear), 0.0 (none/ambiguous)
    pub signal_bias: f64,
}

/// Per-bar sweep bias output.
#[derive(Clone, Debug)]
pub struct SweepBiasResult {
    /// Combined, weighted, decayed, normalized sweep bias (range ±0.30)
    pub sweep_bias: f64,
    /// Per-TF breakdown
    pub signals: Vec<TfSweepSignal>,
}

/// TF weights (bigger TF → more weight).
const TF_WEIGHTS: [f64; 5] = [0.08, 0.12, 0.20, 0.25, 0.35]; // 1H, 2H, 4H, 8H, 1D
/// Half-life values in 5m bars per TF.
const HALF_LIVES: [f64; 5] = [12.0, 24.0, 48.0, 96.0, 288.0]; // 1H, 2H, 4H, 8H, 1D
/// Final scaling factor — reduced from 0.30 to 0.15 because sweep alone
/// shows no directional edge in backtesting; it should assist, not dominate.
const SWEEP_SCALE: f64 = 0.15;

/// Detect sweep on an HTF bar relative to its predecessor.
fn detect_sweep(prev_high: f64, prev_low: f64, curr: &HtfBarInfo) -> (SweepType, SweepResult) {
    let high_sweep = curr.high > prev_high && curr.low <= prev_high;
    let low_sweep = curr.low < prev_low && curr.high >= prev_low;

    if high_sweep && low_sweep {
        return (SweepType::Both, SweepResult::None); // ambiguous double sweep → pass
    }

    if high_sweep {
        let result = if curr.close > prev_high {
            SweepResult::Continuation // bullish: broke above and held
        } else if curr.close < curr.open {
            SweepResult::Reversal // bearish: red candle, failed breakout
        } else {
            SweepResult::Ambiguous
        };
        return (SweepType::High, result);
    }

    if low_sweep {
        let result = if curr.close < prev_low {
            SweepResult::Continuation // bearish: broke below and held
        } else if curr.close > curr.open {
            SweepResult::Reversal // bullish: green candle, dip bought
        } else {
            SweepResult::Ambiguous
        };
        return (SweepType::Low, result);
    }

    (SweepType::None, SweepResult::None)
}

/// Convert sweep type + result into a directional signal bias.
fn signal_bias(sweep_type: SweepType, result: SweepResult) -> f64 {
    match (sweep_type, result) {
        (SweepType::High, SweepResult::Continuation) => 1.0,   // bull: breakout held
        (SweepType::High, SweepResult::Reversal) => -1.0,      // bear: rejected at high
        (SweepType::Low, SweepResult::Continuation) => -1.0,   // bear: breakdown held
        (SweepType::Low, SweepResult::Reversal) => 1.0,        // bull: dip bought
        _ => 0.0,                                                // ambiguous, both, none
    }
}

/// Minimal bar info extracted from HtfSeries for sweep detection.
struct HtfBarInfo {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    /// Index of the last 5m bar in this HTF bar
    end_idx: usize,
}

/// Per-TF sweep event tracking.
struct TfSweepTracker {
    tf_idx: usize,
    tf_name: &'static str,
    /// The most recent sweep event info
    last_sweep_type: SweepType,
    last_sweep_result: SweepResult,
    last_sweep_signal: f64,
    /// 5m bar index when the last sweep's HTF bar closed
    last_sweep_close_bar: usize,
}

/// Compute sweep bias for every 5m bar.
///
/// `htf_series` — 5 HTF series in order: [1H, 2H, 4H, 8H, 1D]
/// `n_bars`     — total 5m bars
pub fn compute_sweep_bias_series(
    htf_series: &[HtfSeries],
    n_bars: usize,
) -> Vec<SweepBiasResult> {
    assert!(htf_series.len() == 5, "Expected 5 HTF series");

    // Pre-extract HTF bar info per TF
    let tf_names: [&str; 5] = ["1H", "2H", "4H", "8H", "1D"];
    let tf_bars: Vec<Vec<HtfBarInfo>> = htf_series
        .iter()
        .map(|series| {
            series
                .bars
                .iter()
                .map(|b| HtfBarInfo {
                    open: b.open,
                    high: b.high,
                    low: b.low,
                    close: b.close,
                    end_idx: b.end_idx,
                })
                .collect()
        })
        .collect();

    // For each TF, pre-compute all sweep events
    // sweep_events[tf_idx] = Vec of (sweep_type, sweep_result, signal, htf_bar_close_5m_idx)
    let mut sweep_events: Vec<Vec<(SweepType, SweepResult, f64, usize)>> = Vec::with_capacity(5);

    for tf_idx in 0..5 {
        let bars = &tf_bars[tf_idx];
        let mut events = Vec::new();

        for j in 1..bars.len() {
            let (st, sr) = detect_sweep(bars[j - 1].high, bars[j - 1].low, &bars[j]);
            let sb = signal_bias(st, sr);
            if st != SweepType::None {
                events.push((st, sr, sb, bars[j].end_idx));
            }
        }
        sweep_events.push(events);
    }

    // For each 5m bar, find the most recent sweep per TF and compute decayed bias
    let mut results = Vec::with_capacity(n_bars);

    // Track current event index per TF (events are sorted by end_idx)
    let mut event_ptrs = [0usize; 5];

    for bar_i in 0..n_bars {
        let mut signals = Vec::with_capacity(5);
        let mut weighted_sum = 0.0f64;

        for tf_idx in 0..5 {
            let events = &sweep_events[tf_idx];

            // Advance event pointer to latest event that has closed by bar_i
            while event_ptrs[tf_idx] < events.len()
                && events[event_ptrs[tf_idx]].3 <= bar_i
            {
                event_ptrs[tf_idx] += 1;
            }

            // The most recent event is at event_ptrs[tf_idx] - 1
            let sig = if event_ptrs[tf_idx] > 0 {
                let &(st, sr, sb, close_bar) = &events[event_ptrs[tf_idx] - 1];
                let recency = bar_i.saturating_sub(close_bar);
                let decay = (-(recency as f64) / HALF_LIVES[tf_idx]).exp();
                let decayed_bias = sb * decay;

                TfSweepSignal {
                    tf_name: tf_names[tf_idx],
                    sweep_type: st,
                    result: sr,
                    recency,
                    decay,
                    signal_bias: sb,
                }
            } else {
                TfSweepSignal {
                    tf_name: tf_names[tf_idx],
                    sweep_type: SweepType::None,
                    result: SweepResult::None,
                    recency: 0,
                    decay: 0.0,
                    signal_bias: 0.0,
                }
            };

            let tf_contribution = TF_WEIGHTS[tf_idx] * sig.signal_bias * sig.decay;
            weighted_sum += tf_contribution;
            signals.push(sig);
        }

        let sweep_bias = weighted_sum * SWEEP_SCALE;

        results.push(SweepBiasResult {
            sweep_bias,
            signals,
        });
    }

    // Reset event pointers for sequential access (they were consumed above)
    // Actually the above loop already handles this correctly since we iterate bar_i in order

    results
}

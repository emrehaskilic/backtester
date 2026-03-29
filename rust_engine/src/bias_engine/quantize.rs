/// Bias Engine — Rolling Quintile Quantization (Section 3 of spec)
///
/// Each feature is quantized into 5 groups (Q1–Q5) using a rolling window
/// of 2016 bars (7 days at 5m resolution).
///
/// Q1 = bottom 20%, Q5 = top 20%.
/// 0 = invalid (NaN or insufficient warmup).

use rayon::prelude::*;

use super::features::{FeatureArrays, N_FEATURES};

/// Rolling window for quantile computation (7 days of 5m bars)
pub const QUANT_WINDOW: usize = 2016;

/// Result of rolling quintile computation: quintile values + boundary thresholds.
pub struct QuantileResult {
    /// Quintile per bar: 0 = invalid, 1–5
    pub quintiles: Vec<u8>,
    /// Per-bar boundaries: [p20, p40, p60, p80].
    /// NAN if bar is invalid. Used for fast noise-injection re-quantization.
    pub boundaries: Vec<[f64; 4]>,
}

/// Compute rolling quintiles AND store per-bar boundaries in a single pass.
pub fn rolling_quintile_full(data: &[f64], window: usize) -> QuantileResult {
    let n = data.len();
    let nan4 = [f64::NAN; 4];
    let mut quintiles = vec![0u8; n];
    let mut boundaries = vec![nan4; n];

    if n < window {
        return QuantileResult {
            quintiles,
            boundaries,
        };
    }

    let mut buf: Vec<f64> = Vec::with_capacity(window);

    for i in (window - 1)..n {
        let val = data[i];
        if val.is_nan() {
            continue;
        }

        buf.clear();
        let start = i + 1 - window;
        for j in start..=i {
            if !data[j].is_nan() {
                buf.push(data[j]);
            }
        }

        let nv = buf.len();
        if nv < window / 2 {
            continue;
        }

        buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p20 = buf[nv / 5];
        let p40 = buf[nv * 2 / 5];
        let p60 = buf[nv * 3 / 5];
        let p80 = buf[nv * 4 / 5];

        boundaries[i] = [p20, p40, p60, p80];

        quintiles[i] = if val <= p20 {
            1
        } else if val <= p40 {
            2
        } else if val <= p60 {
            3
        } else if val <= p80 {
            4
        } else {
            5
        };
    }

    QuantileResult {
        quintiles,
        boundaries,
    }
}

/// Quantize a single value using pre-computed boundaries. O(1).
///
/// Used by noise injection to avoid re-sorting the entire window.
#[inline]
pub fn quantize_with_boundaries(val: f64, bounds: &[f64; 4]) -> u8 {
    if val.is_nan() || bounds[0].is_nan() {
        return 0;
    }
    if val <= bounds[0] {
        1
    } else if val <= bounds[1] {
        2
    } else if val <= bounds[2] {
        3
    } else if val <= bounds[3] {
        4
    } else {
        5
    }
}

/// Legacy: just quintiles, no boundaries.
pub fn rolling_quintile(data: &[f64], window: usize) -> Vec<u8> {
    rolling_quintile_full(data, window).quintiles
}

/// Quantize all 7 features in parallel, returning quintiles + boundaries.
pub fn quantize_all_full(features: &FeatureArrays) -> Vec<QuantileResult> {
    (0..N_FEATURES)
        .into_par_iter()
        .map(|i| rolling_quintile_full(features.get(i), QUANT_WINDOW))
        .collect()
}

/// Quantize all 7 features in parallel (legacy, quintiles only).
pub fn quantize_all(features: &FeatureArrays) -> Vec<Vec<u8>> {
    (0..N_FEATURES)
        .into_par_iter()
        .map(|i| rolling_quintile(features.get(i), QUANT_WINDOW))
        .collect()
}

/// Variable-Q quantization: split into `q_count` equal-probability bins.
/// Returns quintile values 1..=q_count (0 = invalid).
pub fn rolling_quantile_variable(data: &[f64], window: usize, q_count: usize) -> QuantileResult {
    let n = data.len();
    let nan4 = [f64::NAN; 4];
    let mut quintiles = vec![0u8; n];
    let mut boundaries = vec![nan4; n]; // always store up to 4 boundaries for compat

    if n < window || q_count < 2 || q_count > 7 {
        return QuantileResult { quintiles, boundaries };
    }

    let mut buf: Vec<f64> = Vec::with_capacity(window);

    for i in (window - 1)..n {
        let val = data[i];
        if val.is_nan() { continue; }

        buf.clear();
        let start = i + 1 - window;
        for j in start..=i {
            if !data[j].is_nan() { buf.push(data[j]); }
        }

        let nv = buf.len();
        if nv < window / 2 { continue; }

        buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute q_count-1 boundary percentiles
        let mut bounds = [f64::NAN; 4];
        for k in 1..q_count {
            let idx = nv * k / q_count;
            if k - 1 < 4 {
                bounds[k - 1] = buf[idx.min(nv - 1)];
            }
        }
        boundaries[i] = bounds;

        // Classify into bin 1..=q_count
        let mut bin = q_count as u8;
        for k in (1..q_count).rev() {
            if k - 1 < 4 && !bounds[k - 1].is_nan() && val <= bounds[k - 1] {
                bin = k as u8;
            }
        }
        quintiles[i] = bin;
    }

    QuantileResult { quintiles, boundaries }
}

/// Quantize all features with custom window and Q count (parallel).
pub fn quantize_all_with_params(
    features: &FeatureArrays,
    window: usize,
    q_count: usize,
) -> Vec<QuantileResult> {
    if q_count == 5 {
        // Use optimized original path
        (0..N_FEATURES)
            .into_par_iter()
            .map(|i| rolling_quintile_full(features.get(i), window))
            .collect()
    } else {
        (0..N_FEATURES)
            .into_par_iter()
            .map(|i| rolling_quantile_variable(features.get(i), window, q_count))
            .collect()
    }
}

/// Bias Engine — Feature Computation (Section 2 of spec)
///
/// 8 features computed from 5m OHLCV + buy_vol/sell_vol + OI:
///   0. CVD Z-Score Micro  (window=12,  1H)
///   1. CVD Z-Score Macro  (window=288, 1D)
///   2. OI Change ATR-Norm (window=288)
///   3. Volume Z-Score Micro (window=12)
///   4. Volume Z-Score Macro (window=288)
///   5. Imbalance Smoothed (EMA span=12)
///   6. ATR Percentile     (window=288)
///   7. VWAP Distance      (window=48)

const EPSILON: f64 = 1e-10;

pub const N_FEATURES: usize = 8;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "cvd_micro",
    "cvd_macro",
    "oi_change",
    "vol_micro",
    "vol_macro",
    "imbalance_smooth",
    "atr_percentile",
    "vwap_distance",
];

pub struct FeatureArrays {
    pub data: [Vec<f64>; N_FEATURES],
}

impl FeatureArrays {
    #[inline]
    pub fn get(&self, index: usize) -> &[f64] {
        &self.data[index]
    }

    /// True if all 7 features are non-NaN at bar i
    pub fn all_valid(&self, i: usize) -> bool {
        self.data.iter().all(|f| !f[i].is_nan())
    }
}

// ── Public API ──

/// Compute all 7 features from raw 5m data
pub fn compute_features(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
) -> FeatureArrays {
    let n = close.len();

    // Pre-compute common derived arrays
    let cvd_bar: Vec<f64> = (0..n).map(|i| buy_vol[i] - sell_vol[i]).collect();
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();
    let atr_bar: Vec<f64> = (0..n).map(|i| high[i] - low[i]).collect();

    // 0. CVD Z-Score Micro (window=12)
    let cvd_micro = rolling_zscore(&cvd_bar, 12);

    // 1. CVD Z-Score Macro (window=288)
    let cvd_macro = rolling_zscore(&cvd_bar, 288);

    // 2. OI Change ATR-Normalized (window=288)
    let oi_change = compute_oi_change(oi, &atr_bar, 288);

    // 3. Volume Z-Score Micro (window=12)
    let vol_micro = rolling_zscore(&total_vol, 12);

    // 4. Volume Z-Score Macro (window=288)
    let vol_macro = rolling_zscore(&total_vol, 288);

    // 5. Imbalance Smoothed (EMA span=12)
    let raw_imbalance: Vec<f64> = (0..n)
        .map(|i| {
            let total = total_vol[i];
            if total < EPSILON {
                0.0
            } else {
                cvd_bar[i] / total
            }
        })
        .collect();
    let imbalance_smooth = ema(&raw_imbalance, 12);

    // 6. ATR Percentile (window=288)
    let atr_percentile = rolling_rank(&atr_bar, 288);

    // 7. VWAP Distance (window=48)
    let vwap_dist = compute_vwap_distance(high, low, close, &total_vol, 48);

    FeatureArrays {
        data: [
            cvd_micro,
            cvd_macro,
            oi_change,
            vol_micro,
            vol_macro,
            imbalance_smooth,
            atr_percentile,
            vwap_dist,
        ],
    }
}

/// Compute all 7 features with custom window parameters
pub fn compute_features_with_params(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
    params: &super::params::GroupAParams,
) -> FeatureArrays {
    let n = close.len();
    let cvd_bar: Vec<f64> = (0..n).map(|i| buy_vol[i] - sell_vol[i]).collect();
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();
    let atr_bar: Vec<f64> = (0..n).map(|i| high[i] - low[i]).collect();

    let cvd_micro = rolling_zscore(&cvd_bar, params.cvd_micro_window);
    let cvd_macro = rolling_zscore(&cvd_bar, params.cvd_macro_window);
    let oi_change = compute_oi_change(oi, &atr_bar, params.oi_change_window);
    let vol_micro = rolling_zscore(&total_vol, params.vol_micro_window);
    let vol_macro = rolling_zscore(&total_vol, params.vol_macro_window);

    let raw_imbalance: Vec<f64> = (0..n)
        .map(|i| {
            let total = total_vol[i];
            if total < EPSILON { 0.0 } else { cvd_bar[i] / total }
        })
        .collect();
    let imbalance_smooth = ema(&raw_imbalance, params.imbalance_ema_span);
    let atr_percentile = rolling_rank(&atr_bar, params.atr_pct_window);
    let vwap_dist = compute_vwap_distance(high, low, close, &total_vol, params.vwap_window);

    FeatureArrays {
        data: [cvd_micro, cvd_macro, oi_change, vol_micro, vol_macro, imbalance_smooth, atr_percentile, vwap_dist],
    }
}

// ── Helpers ──

/// Rolling z-score: (value - rolling_mean) / rolling_std
fn rolling_zscore(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n < window {
        return result;
    }

    let wf = window as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    // Seed the first window
    for j in 0..window {
        sum += data[j];
        sum_sq += data[j] * data[j];
    }

    // First complete window -> index = window-1
    {
        let mean = sum / wf;
        let var = (sum_sq / wf) - mean * mean;
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        result[window - 1] = if std > EPSILON {
            (data[window - 1] - mean) / std
        } else {
            0.0
        };
    }

    // Slide
    for i in window..n {
        sum += data[i] - data[i - window];
        sum_sq += data[i] * data[i] - data[i - window] * data[i - window];

        let mean = sum / wf;
        let var = (sum_sq / wf) - mean * mean;
        // Guard against floating-point rounding producing tiny negatives
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        result[i] = if std > EPSILON {
            (data[i] - mean) / std
        } else {
            0.0
        };
    }

    result
}

/// OI change normalized by sum-of-ATR-bars (daily range proxy)
fn compute_oi_change(oi: &[f64], atr_bar: &[f64], window: usize) -> Vec<f64> {
    let n = oi.len();
    let mut result = vec![f64::NAN; n];
    if n <= window {
        return result;
    }

    // Rolling sum of atr_bar over `window` bars
    let mut atr_sum = 0.0_f64;
    for j in 0..window {
        atr_sum += atr_bar[j];
    }

    // First valid: i = window  (need oi[i] and oi[i - window])
    for i in window..n {
        // Slide atr_sum
        atr_sum += atr_bar[i] - atr_bar[i - window];

        if !oi[i].is_nan() && !oi[i - window].is_nan() {
            let oi_delta = oi[i] - oi[i - window];
            result[i] = if atr_sum.abs() > EPSILON {
                oi_delta / atr_sum
            } else {
                0.0
            };
        }
    }

    result
}

/// Exponential Moving Average (span-based α = 2/(span+1))
fn ema(data: &[f64], span: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }

    let alpha = 2.0 / (span as f64 + 1.0);

    // Find first non-NaN to seed
    let mut start = 0;
    while start < n && data[start].is_nan() {
        start += 1;
    }
    if start >= n {
        return result;
    }

    result[start] = data[start];
    for i in (start + 1)..n {
        let prev = result[i - 1];
        if data[i].is_nan() {
            result[i] = prev; // carry forward
        } else {
            result[i] = alpha * data[i] + (1.0 - alpha) * prev;
        }
    }

    result
}

/// Rolling VWAP distance: z-score of (close - vwap) / vwap
/// VWAP = sum(typical_price * volume) / sum(volume) over rolling window
fn compute_vwap_distance(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    total_vol: &[f64],
    window: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if n < window || window == 0 {
        return result;
    }

    // typical_price * volume for each bar
    let tp_vol: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0 * total_vol[i])
        .collect();

    // Rolling sums
    let mut sum_tpv = 0.0_f64;
    let mut sum_vol = 0.0_f64;

    // Seed first window
    for j in 0..window {
        sum_tpv += tp_vol[j];
        sum_vol += total_vol[j];
    }

    // First VWAP at window-1
    if sum_vol > EPSILON {
        let vwap = sum_tpv / sum_vol;
        result[window - 1] = (close[window - 1] - vwap) / vwap.max(EPSILON);
    }

    // Slide
    for i in window..n {
        sum_tpv += tp_vol[i] - tp_vol[i - window];
        sum_vol += total_vol[i] - total_vol[i - window];

        if sum_vol > EPSILON {
            let vwap = sum_tpv / sum_vol;
            result[i] = (close[i] - vwap) / vwap.max(EPSILON);
        }
    }

    // Now z-score the raw distances for quantization compatibility
    rolling_zscore_skipnan(&result, window)
}

/// Rolling z-score that skips NaN values in input
fn rolling_zscore_skipnan(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n < window {
        return result;
    }

    for i in (window - 1)..n {
        if data[i].is_nan() {
            continue;
        }
        let start = i + 1 - window;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count = 0u32;
        for j in start..=i {
            if !data[j].is_nan() {
                sum += data[j];
                sum_sq += data[j] * data[j];
                count += 1;
            }
        }
        if count < 2 {
            result[i] = 0.0;
            continue;
        }
        let cf = count as f64;
        let mean = sum / cf;
        let var = (sum_sq / cf) - mean * mean;
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        result[i] = if std > EPSILON {
            (data[i] - mean) / std
        } else {
            0.0
        };
    }
    result
}

/// Rolling rank: fraction of window values <= current value (0.0 – 1.0)
fn rolling_rank(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n < window {
        return result;
    }

    for i in (window - 1)..n {
        let val = data[i];
        if val.is_nan() {
            continue;
        }

        let start = i + 1 - window;
        let mut rank = 0u32;
        let mut count = 0u32;
        for j in start..=i {
            if !data[j].is_nan() {
                count += 1;
                if data[j] <= val {
                    rank += 1;
                }
            }
        }

        if count > 0 {
            result[i] = rank as f64 / count as f64;
        }
    }

    result
}

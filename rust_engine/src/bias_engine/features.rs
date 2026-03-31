/// Bias Engine — Feature Computation
///
/// 13 features computed from OHLCV + buy_vol/sell_vol + OI:
///   0.  CVD Z-Score Micro       (short-term buy/sell pressure)
///   1.  CVD Z-Score Macro       (long-term buy/sell pressure)
///   2.  OI Change ATR-Norm      (position building/unwinding)
///   3.  Volume Z-Score Micro    (short-term volume spike)
///   4.  Volume Z-Score Macro    (long-term volume trend)
///   5.  Imbalance Smoothed      (directional volume bias)
///   6.  ATR Percentile          (volatility regime)
///   7.  VWAP Distance           (price-volume equilibrium)
///   8.  Price Momentum          (ROC z-score, trend strength)
///   9.  Bar Structure           (wick ratio, rejection pattern)
///   10. Volume-Price Divergence (price up + vol down = weakness)
///   11. OI-Volume Interaction   (new positions vs liquidations)
///   12. Return Autocorrelation  (momentum vs mean-reversion regime)

const EPSILON: f64 = 1e-10;

pub const N_FEATURES: usize = 13;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "cvd_micro",
    "cvd_macro",
    "oi_change",
    "vol_micro",
    "vol_macro",
    "imbalance_smooth",
    "atr_percentile",
    "vwap_distance",
    "price_momentum",
    "bar_structure",
    "vol_price_divergence",
    "oi_vol_interaction",
    "return_autocorr",
];

pub struct FeatureArrays {
    pub data: [Vec<f64>; N_FEATURES],
}

impl FeatureArrays {
    #[inline]
    pub fn get(&self, index: usize) -> &[f64] {
        &self.data[index]
    }

    /// True if all features are non-NaN at bar i
    pub fn all_valid(&self, i: usize) -> bool {
        self.data.iter().all(|f| !f[i].is_nan())
    }
}

// ── Public API ──

/// Compute all 13 features from raw data
pub fn compute_features(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
) -> FeatureArrays {
    compute_features_inner(high, low, close, buy_vol, sell_vol, oi,
        12, 288, 12, 288, 12, 288, 288, 48, 24, 24, 48, 48, 24)
}

/// Compute all 13 features with custom window parameters
pub fn compute_features_with_params(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    oi: &[f64],
    params: &super::params::GroupAParams,
) -> FeatureArrays {
    compute_features_inner(high, low, close, buy_vol, sell_vol, oi,
        params.cvd_micro_window, params.cvd_macro_window,
        params.vol_micro_window, params.vol_macro_window,
        params.imbalance_ema_span, params.atr_pct_window,
        params.oi_change_window, params.vwap_window,
        params.momentum_window, params.wick_window,
        params.divergence_window, params.oi_vol_window,
        params.autocorr_window)
}

#[allow(clippy::too_many_arguments)]
fn compute_features_inner(
    high: &[f64], low: &[f64], close: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    cvd_micro_w: usize, cvd_macro_w: usize,
    vol_micro_w: usize, vol_macro_w: usize,
    imb_ema_span: usize, atr_pct_w: usize,
    oi_change_w: usize, vwap_w: usize,
    momentum_w: usize, wick_w: usize,
    divergence_w: usize, oi_vol_w: usize,
    autocorr_w: usize,
) -> FeatureArrays {
    let n = close.len();

    // Pre-compute common derived arrays
    let cvd_bar: Vec<f64> = (0..n).map(|i| buy_vol[i] - sell_vol[i]).collect();
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();
    let atr_bar: Vec<f64> = (0..n).map(|i| high[i] - low[i]).collect();
    let returns: Vec<f64> = {
        let mut r = vec![0.0f64; n];
        for i in 1..n {
            if close[i - 1].abs() > EPSILON {
                r[i] = (close[i] - close[i - 1]) / close[i - 1];
            }
        }
        r
    };

    // 0. CVD Z-Score Micro
    let cvd_micro = rolling_zscore(&cvd_bar, cvd_micro_w);

    // 1. CVD Z-Score Macro
    let cvd_macro = rolling_zscore(&cvd_bar, cvd_macro_w);

    // 2. OI Change ATR-Normalized
    let oi_change = compute_oi_change(oi, &atr_bar, oi_change_w);

    // 3. Volume Z-Score Micro
    let vol_micro = rolling_zscore(&total_vol, vol_micro_w);

    // 4. Volume Z-Score Macro
    let vol_macro = rolling_zscore(&total_vol, vol_macro_w);

    // 5. Imbalance Smoothed
    let raw_imbalance: Vec<f64> = (0..n)
        .map(|i| {
            let total = total_vol[i];
            if total < EPSILON { 0.0 } else { cvd_bar[i] / total }
        })
        .collect();
    let imbalance_smooth = ema(&raw_imbalance, imb_ema_span);

    // 6. ATR Percentile
    let atr_percentile = rolling_rank(&atr_bar, atr_pct_w);

    // 7. VWAP Distance
    let vwap_dist = compute_vwap_distance(high, low, close, &total_vol, vwap_w);

    // 8. Price Momentum (ROC z-score)
    let price_momentum = compute_price_momentum(&returns, momentum_w);

    // 9. Bar Structure (wick ratio)
    let bar_structure = compute_bar_structure(high, low, close, &atr_bar, wick_w);

    // 10. Volume-Price Divergence
    let vol_price_div = compute_vol_price_divergence(&returns, &total_vol, divergence_w);

    // 11. OI-Volume Interaction
    let oi_vol_interact = compute_oi_vol_interaction(oi, &total_vol, &atr_bar, oi_vol_w);

    // 12. Return Autocorrelation
    let ret_autocorr = compute_return_autocorr(&returns, autocorr_w);

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
            price_momentum,
            bar_structure,
            vol_price_div,
            oi_vol_interact,
            ret_autocorr,
        ],
    }
}

// ══════════════════════════════════════════════════════════════
// NEW FEATURE COMPUTATIONS
// ══════════════════════════════════════════════════════════════

/// Feature 8: Price Momentum — rolling z-score of cumulative returns
/// Positive = uptrend, negative = downtrend
fn compute_price_momentum(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut cum_ret = vec![0.0f64; n];
    for i in 0..n {
        if i >= window {
            // Rolling sum of returns over window
            let mut s = 0.0;
            for j in (i + 1 - window)..=i {
                s += returns[j];
            }
            cum_ret[i] = s;
        }
    }
    rolling_zscore(&cum_ret, window)
}

/// Feature 9: Bar Structure — signed wick ratio
/// Measures rejection: long upper wick = bearish rejection, long lower wick = bullish rejection
/// Output: z-scored for quantization
fn compute_bar_structure(
    high: &[f64], low: &[f64], close: &[f64],
    atr_bar: &[f64], window: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut wick_ratio = vec![0.0f64; n];

    for i in 1..n {
        let range = atr_bar[i];
        if range < EPSILON {
            continue;
        }
        let open_est = close[i - 1]; // approximate open as previous close
        let body_top = close[i].max(open_est);
        let body_bot = close[i].min(open_est);

        let upper_wick = high[i] - body_top;
        let lower_wick = body_bot - low[i];

        // Signed: positive = more lower wick (bullish rejection)
        //         negative = more upper wick (bearish rejection)
        wick_ratio[i] = (lower_wick - upper_wick) / range;
    }

    rolling_zscore(&wick_ratio, window)
}

/// Feature 10: Volume-Price Divergence
/// When price trends up but volume trends down (or vice versa) = divergence
/// Computed as: -correlation(returns, volume_change) over rolling window
/// Positive = divergence (price up, vol down), negative = confirmation
fn compute_vol_price_divergence(
    returns: &[f64], total_vol: &[f64], window: usize,
) -> Vec<f64> {
    let n = returns.len();
    let mut result = vec![f64::NAN; n];
    if n < window + 1 {
        return result;
    }

    // Volume changes
    let mut vol_change = vec![0.0f64; n];
    for i in 1..n {
        if total_vol[i - 1].abs() > EPSILON {
            vol_change[i] = (total_vol[i] - total_vol[i - 1]) / total_vol[i - 1];
        }
    }

    // Rolling correlation between returns and vol_change
    for i in (window - 1)..n {
        let start = i + 1 - window;
        let mut sum_r = 0.0;
        let mut sum_v = 0.0;
        let mut sum_rr = 0.0;
        let mut sum_vv = 0.0;
        let mut sum_rv = 0.0;
        let wf = window as f64;

        for j in start..=i {
            let r = returns[j];
            let v = vol_change[j];
            sum_r += r;
            sum_v += v;
            sum_rr += r * r;
            sum_vv += v * v;
            sum_rv += r * v;
        }

        let mean_r = sum_r / wf;
        let mean_v = sum_v / wf;
        let var_r = (sum_rr / wf) - mean_r * mean_r;
        let var_v = (sum_vv / wf) - mean_v * mean_v;
        let cov = (sum_rv / wf) - mean_r * mean_v;

        let std_r = if var_r > 0.0 { var_r.sqrt() } else { 0.0 };
        let std_v = if var_v > 0.0 { var_v.sqrt() } else { 0.0 };

        if std_r > EPSILON && std_v > EPSILON {
            let corr = cov / (std_r * std_v);
            // Negate: positive divergence = price and vol moving opposite
            result[i] = -corr;
        } else {
            result[i] = 0.0;
        }
    }

    result
}

/// Feature 11: OI-Volume Interaction
/// OI_up + high_vol = new positions opening (directional conviction)
/// OI_down + high_vol = positions closing/liquidations
/// Output: signed by OI direction, magnitude by volume
fn compute_oi_vol_interaction(
    oi: &[f64], total_vol: &[f64], atr_bar: &[f64], window: usize,
) -> Vec<f64> {
    let n = oi.len();
    let mut raw = vec![0.0f64; n];

    // Rolling ATR sum for normalization
    let mut atr_sum = 0.0f64;
    if n > window {
        for j in 0..window {
            atr_sum += atr_bar[j];
        }
    }

    for i in window..n {
        atr_sum += atr_bar[i] - atr_bar[i - window];

        if oi[i].is_nan() || oi[i - 1].is_nan() {
            continue;
        }

        let oi_change = oi[i] - oi[i - 1];
        let oi_dir = if oi_change > 0.0 { 1.0 } else if oi_change < 0.0 { -1.0 } else { 0.0 };

        // Volume relative to recent average
        let mut vol_sum = 0.0;
        for j in (i + 1 - window)..=i {
            vol_sum += total_vol[j];
        }
        let vol_avg = vol_sum / window as f64;
        let vol_ratio = if vol_avg > EPSILON {
            total_vol[i] / vol_avg
        } else {
            1.0
        };

        // Interaction: OI direction * volume intensity
        // High positive = new longs/shorts opening with conviction
        // High negative = liquidations/closes with high volume
        raw[i] = oi_dir * (vol_ratio - 1.0);  // centered around 0
    }

    rolling_zscore(&raw, window)
}

/// Feature 12: Return Autocorrelation
/// Rolling autocorrelation of returns at lag 1
/// Positive = momentum regime (trending), negative = mean-reversion regime
fn compute_return_autocorr(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    let mut result = vec![f64::NAN; n];
    if n < window + 1 {
        return result;
    }

    for i in window..n {
        let start = i + 1 - window;

        // Mean of returns in window
        let mut sum = 0.0;
        for j in start..=i {
            sum += returns[j];
        }
        let mean = sum / window as f64;

        // Autocovariance at lag 1 and variance
        let mut autocov = 0.0;
        let mut var = 0.0;
        for j in (start + 1)..=i {
            let r_t = returns[j] - mean;
            let r_t1 = returns[j - 1] - mean;
            autocov += r_t * r_t1;
            var += r_t * r_t;
        }

        if var.abs() > EPSILON {
            result[i] = autocov / var;  // autocorrelation coefficient
        } else {
            result[i] = 0.0;
        }
    }

    result
}

// ══════════════════════════════════════════════════════════════
// EXISTING HELPERS
// ══════════════════════════════════════════════════════════════

/// Rolling z-score: (value - rolling_mean) / rolling_std
fn rolling_zscore(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    if n < window || window == 0 {
        return result;
    }

    let wf = window as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for j in 0..window {
        sum += data[j];
        sum_sq += data[j] * data[j];
    }

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

    for i in window..n {
        sum += data[i] - data[i - window];
        sum_sq += data[i] * data[i] - data[i - window] * data[i - window];

        let mean = sum / wf;
        let var = (sum_sq / wf) - mean * mean;
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        result[i] = if std > EPSILON {
            (data[i] - mean) / std
        } else {
            0.0
        };
    }

    result
}

/// OI change normalized by sum-of-ATR-bars
fn compute_oi_change(oi: &[f64], atr_bar: &[f64], window: usize) -> Vec<f64> {
    let n = oi.len();
    let mut result = vec![f64::NAN; n];
    if n <= window {
        return result;
    }

    let mut atr_sum = 0.0_f64;
    for j in 0..window {
        atr_sum += atr_bar[j];
    }

    for i in window..n {
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
            result[i] = prev;
        } else {
            result[i] = alpha * data[i] + (1.0 - alpha) * prev;
        }
    }

    result
}

/// Rolling VWAP distance: z-score of (close - vwap) / vwap
fn compute_vwap_distance(
    high: &[f64], low: &[f64], close: &[f64],
    total_vol: &[f64], window: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if n < window || window == 0 {
        return result;
    }

    let tp_vol: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0 * total_vol[i])
        .collect();

    let mut sum_tpv = 0.0_f64;
    let mut sum_vol = 0.0_f64;

    for j in 0..window {
        sum_tpv += tp_vol[j];
        sum_vol += total_vol[j];
    }

    if sum_vol > EPSILON {
        let vwap = sum_tpv / sum_vol;
        result[window - 1] = (close[window - 1] - vwap) / vwap.max(EPSILON);
    }

    for i in window..n {
        sum_tpv += tp_vol[i] - tp_vol[i - window];
        sum_vol += total_vol[i] - total_vol[i - window];

        if sum_vol > EPSILON {
            let vwap = sum_tpv / sum_vol;
            result[i] = (close[i] - vwap) / vwap.max(EPSILON);
        }
    }

    rolling_zscore_skipnan(&result, window)
}

/// Rolling z-score that skips NaN values
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

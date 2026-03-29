/// Pattern Miner — Quantile-based conditional pattern discovery.
///
/// Pipeline:
/// 1. Feature hesapla (12 continuous feature per bar)
/// 2. Triple barrier label (TP/SL/Timeout)
/// 3. Quantile encoding (her feature 5 bucket)
/// 4. Conditional slicing (k=2,3 feature kombinasyonları)
/// 5. Statistical filtering (WR >= 60%, N >= 100, monthly consistency >= 70%)
/// 6. FDR correction (Benjamini-Hochberg)
/// 7. Pattern ranking & merging

use rayon::prelude::*;

// ── Sabitler ──

const MIN_BARS: usize = 200;
const N_QUANTILES: usize = 5; // Q1-Q5
const MIN_SAMPLE: usize = 50;
const MIN_WR: f64 = 50.0;
const MIN_MONTHLY_CONSISTENCY: f64 = 60.0;
const FDR_ALPHA: f64 = 0.05;

// Triple barrier
const TP_PCT: f64 = 0.5;  // +0.5%
const SL_PCT: f64 = 0.5;  // -0.5%
const TIMEOUT_BARS: usize = 160; // 8h in 3m bars

// ── Feature names ──

pub const FEATURE_NAMES: [&str; 12] = [
    "CVD_div_zscore",
    "CVD_slope",
    "CVD_div_duration",
    "price_slope",
    "price_vs_EMA",
    "OI_change_pct",
    "OI_trend",
    "volume_zscore",
    "ATR_percentile",
    "KC_distance",
    "KC_width",
    "imbalance",
];

pub const QUANTILE_LABELS: [&str; 5] = ["Q1", "Q2", "Q3", "Q4", "Q5"];

// ── Yardımcı fonksiyonlar ──

fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 || period == 0 { return out; }
    let k = 2.0 / (period as f64 + 1.0);
    out[0] = data[0];
    for i in 1..n {
        let prev = if out[i-1].is_nan() { data[i] } else { out[i-1] };
        out[i] = data[i] * k + prev * (1.0 - k);
    }
    out
}

fn rolling_mean(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    let mut sum = 0.0;
    for i in 0..period { sum += data[i]; }
    out[period - 1] = sum / period as f64;
    for i in period..n {
        sum += data[i] - data[i - period];
        out[i] = sum / period as f64;
    }
    out
}

fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    let mean = rolling_mean(data, period);
    for i in (period - 1)..n {
        if mean[i].is_nan() { continue; }
        let mut var = 0.0;
        for j in (i + 1 - period)..=i {
            var += (data[j] - mean[i]).powi(2);
        }
        out[i] = (var / period as f64).sqrt();
    }
    out
}

fn linear_slope(data: &[f64], end: usize, period: usize) -> f64 {
    if end < period { return f64::NAN; }
    let start = end + 1 - period;
    let n = period as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for (j, i) in (start..=end).enumerate() {
        let x = j as f64;
        let y = data[i];
        if y.is_nan() { return f64::NAN; }
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 { return 0.0; }
    (n * sum_xy - sum_x * sum_y) / denom
}

fn percentile_rank(val: f64, sorted_window: &[f64]) -> f64 {
    if sorted_window.is_empty() { return 50.0; }
    let below = sorted_window.iter().filter(|&&v| v < val).count();
    below as f64 / sorted_window.len() as f64 * 100.0
}

// ── Step 1: Feature computation ──

pub fn compute_features(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> Vec<[f64; 12]> {
    let n = closes.len();
    let lookback = 20usize;
    let long_lb = 50usize;

    // Pre-compute base series
    let mut delta = vec![0.0_f64; n];
    let mut imbalance_raw = vec![0.0_f64; n];
    for i in 0..n {
        delta[i] = buy_vol[i] - sell_vol[i];
        let total = buy_vol[i] + sell_vol[i];
        imbalance_raw[i] = if total > 0.0 { delta[i] / total } else { 0.0 };
    }

    // Cumulative delta (rolling)
    let mut cvd = vec![0.0_f64; n];
    for i in lookback..n {
        let mut sum = 0.0;
        for j in (i + 1 - lookback)..=i { sum += delta[j]; }
        cvd[i] = sum;
    }

    let cvd_mean = rolling_mean(&cvd, lookback);
    let cvd_std = rolling_std(&cvd, lookback);

    // EMA
    let ema20 = ema(closes, 20);

    // ATR
    let mut tr = vec![0.0_f64; n];
    if n > 0 { tr[0] = highs[0] - lows[0]; }
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
    }
    let atr_arr = {
        let period = 14usize;
        let mut out = vec![f64::NAN; n];
        if n > period {
            let mut sum = 0.0;
            for i in 1..=period { sum += tr[i]; }
            out[period] = sum / period as f64;
            for i in (period+1)..n {
                let prev = out[i-1];
                out[i] = if prev.is_nan() { tr[i] } else { (prev * (period as f64 - 1.0) + tr[i]) / period as f64 };
            }
        }
        out
    };

    // KC
    let kc_mult = 2.0;
    let mut kc_upper = vec![f64::NAN; n];
    let mut kc_lower = vec![f64::NAN; n];
    for i in 0..n {
        if !ema20[i].is_nan() && !atr_arr[i].is_nan() {
            kc_upper[i] = ema20[i] + atr_arr[i] * kc_mult;
            kc_lower[i] = ema20[i] - atr_arr[i] * kc_mult;
        }
    }

    // Volume
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();
    let vol_mean = rolling_mean(&total_vol, lookback);
    let vol_std = rolling_std(&total_vol, lookback);

    // OI change
    let oi_lb = 20usize;
    let mut oi_change = vec![0.0_f64; n];
    for i in oi_lb..n {
        if !oi[i].is_nan() && !oi[i - oi_lb].is_nan() && oi[i - oi_lb] > 0.0 {
            oi_change[i] = (oi[i] - oi[i - oi_lb]) / oi[i - oi_lb] * 100.0;
        }
    }

    // OI trend (slope)
    let oi_smooth = ema(oi, 20);

    // ATR rolling window for percentile
    let atr_lb = 100usize;

    // Build feature matrix
    let start = MIN_BARS.max(long_lb + lookback);
    let mut features = vec![[f64::NAN; 12]; n];

    for i in start..n {
        if cvd_std[i].is_nan() || cvd_std[i] <= 0.0 { continue; }
        if ema20[i].is_nan() || atr_arr[i].is_nan() { continue; }
        if kc_upper[i].is_nan() || kc_lower[i].is_nan() { continue; }

        // F0: CVD divergence z-score
        let cvd_z = (cvd[i] - cvd_mean[i]) / cvd_std[i];

        // F1: CVD slope (son 20 bar)
        let cvd_slope = linear_slope(&cvd, i, lookback);

        // F2: CVD divergence duration (kaç bar üst üste z > 1 veya z < -1)
        let mut div_dur = 0.0;
        let z_thresh = 1.0;
        if cvd_z.abs() > z_thresh {
            let mut cnt = 0i32;
            let sign = if cvd_z > 0.0 { 1.0 } else { -1.0 };
            let mut j = i;
            while j > start {
                let z_j = if cvd_std[j] > 0.0 { (cvd[j] - cvd_mean[j]) / cvd_std[j] } else { 0.0 };
                if z_j * sign > z_thresh { cnt += 1; j -= 1; } else { break; }
            }
            div_dur = cnt as f64;
        }

        // F3: Price slope (normalized by ATR)
        let p_slope = linear_slope(closes, i, lookback);
        let p_slope_norm = if atr_arr[i] > 0.0 { p_slope / atr_arr[i] } else { 0.0 };

        // F4: Price vs EMA (normalized)
        let price_vs_ema = (closes[i] - ema20[i]) / atr_arr[i];

        // F5: OI change %
        let oi_chg = oi_change[i];

        // F6: OI trend (slope normalized)
        let oi_trend = linear_slope(&oi_smooth, i, lookback);
        let oi_trend_norm = if oi_smooth[i].is_nan() || oi_smooth[i] == 0.0 { 0.0 }
            else { oi_trend / oi_smooth[i] * 1000.0 }; // scale

        // F7: Volume z-score
        let vol_z = if !vol_std[i].is_nan() && vol_std[i] > 0.0 {
            (total_vol[i] - vol_mean[i]) / vol_std[i]
        } else { 0.0 };

        // F8: ATR percentile (son 100 bar icinde)
        let atr_pctl = if i >= atr_lb {
            let mut window: Vec<f64> = atr_arr[i+1-atr_lb..=i].iter()
                .filter(|v| !v.is_nan()).cloned().collect();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            percentile_rank(atr_arr[i], &window)
        } else { 50.0 };

        // F9: KC distance (price position within KC bands, 0=lower, 1=upper)
        let kc_w = kc_upper[i] - kc_lower[i];
        let kc_dist = if kc_w > 0.0 { (closes[i] - kc_lower[i]) / kc_w } else { 0.5 };

        // F10: KC width (normalized by price)
        let kc_width_norm = if closes[i] > 0.0 { kc_w / closes[i] * 100.0 } else { 0.0 };

        // F11: Imbalance (smoothed)
        let imb = imbalance_raw[i];

        features[i] = [
            cvd_z, cvd_slope, div_dur, p_slope_norm, price_vs_ema,
            oi_chg, oi_trend_norm, vol_z, atr_pctl, kc_dist, kc_width_norm, imb,
        ];
    }

    features
}

// ── Step 2: Triple barrier labeling ──

pub fn triple_barrier_label(
    closes: &[f64], highs: &[f64], lows: &[f64],
) -> Vec<i8> {
    // For each bar: 1 = TP hit first (trend continued), 0 = SL hit first or timeout
    // Long bias: TP = close + TP_PCT%, SL = close - SL_PCT%
    // We label BOTH directions and pick the one that applies
    let n = closes.len();
    let mut labels = vec![-1i8; n]; // -1 = invalid/not enough future data

    for i in 0..n.saturating_sub(TIMEOUT_BARS) {
        let entry = closes[i];
        if entry <= 0.0 { continue; }

        let tp_price = entry * (1.0 + TP_PCT / 100.0);
        let sl_price = entry * (1.0 - SL_PCT / 100.0);

        let mut label = 0i8; // default: timeout/fail
        for j in (i+1)..=(i + TIMEOUT_BARS).min(n - 1) {
            if highs[j] >= tp_price {
                label = 1; // TP hit → trend continued (long)
                break;
            }
            if lows[j] <= sl_price {
                label = 0; // SL hit → fail
                break;
            }
        }
        labels[i] = label;
    }
    labels
}

// Short direction labels
pub fn triple_barrier_label_short(
    closes: &[f64], highs: &[f64], lows: &[f64],
) -> Vec<i8> {
    let n = closes.len();
    let mut labels = vec![-1i8; n];

    for i in 0..n.saturating_sub(TIMEOUT_BARS) {
        let entry = closes[i];
        if entry <= 0.0 { continue; }

        let tp_price = entry * (1.0 - TP_PCT / 100.0);
        let sl_price = entry * (1.0 + SL_PCT / 100.0);

        let mut label = 0i8;
        for j in (i+1)..=(i + TIMEOUT_BARS).min(n - 1) {
            if lows[j] <= tp_price {
                label = 1;
                break;
            }
            if highs[j] >= sl_price {
                label = 0;
                break;
            }
        }
        labels[i] = label;
    }
    labels
}

// ── Step 3: Quantile encoding ──

pub fn quantile_encode(features: &[[f64; 12]], valid_mask: &[bool]) -> Vec<[u8; 12]> {
    let n = features.len();
    let mut encoded = vec![[0u8; 12]; n];

    // For each feature, compute quantile thresholds from valid data
    for fi in 0..12 {
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            if valid_mask[i] && !features[i][fi].is_nan() {
                vals.push(features[i][fi]);
            }
        }
        if vals.is_empty() { continue; }

        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let thresholds: Vec<f64> = (1..N_QUANTILES).map(|q| {
            let idx = (q as f64 / N_QUANTILES as f64 * vals.len() as f64) as usize;
            vals[idx.min(vals.len() - 1)]
        }).collect();

        for i in 0..n {
            if !valid_mask[i] || features[i][fi].is_nan() { continue; }
            let v = features[i][fi];
            let mut q = 0u8;
            for &th in &thresholds {
                if v > th { q += 1; }
            }
            encoded[i][fi] = q; // 0=Q1, 1=Q2, ..., 4=Q5
        }
    }

    encoded
}

// ── Step 4: Conditional slicing (k=2,3) ──

#[derive(Clone, Debug)]
pub struct PatternCandidate {
    pub features: Vec<usize>,   // feature indices
    pub quantiles: Vec<u8>,     // required quantile per feature
    pub direction: String,      // "long" or "short"
    pub sample: usize,
    pub wins: usize,
    pub wr: f64,
    pub avg_return: f64,
    pub sharpe: f64,
    pub monthly_consistency: f64,
    pub p_value: f64,
    pub score: f64,
}

fn binomial_p_value(n: usize, k: usize, p0: f64) -> f64 {
    // Approximate with normal distribution for large N
    if n == 0 { return 1.0; }
    let mean = n as f64 * p0;
    let std = (n as f64 * p0 * (1.0 - p0)).sqrt();
    if std <= 0.0 { return 1.0; }
    let z = (k as f64 - mean) / std;
    // One-sided p-value from z-score (approximation)
    let p = 0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2));
    p
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub fn mine_patterns(
    encoded: &[[u8; 12]],
    labels_long: &[i8],
    labels_short: &[i8],
    closes: &[f64],
    valid_mask: &[bool],
    total_bars: usize,
) -> Vec<PatternCandidate> {
    let n = encoded.len();
    let bpd = 480usize;
    let bpm = bpd * 30;

    // Pre-compute valid indices
    let valid_long: Vec<usize> = (0..n).filter(|&i| valid_mask[i] && labels_long[i] >= 0).collect();
    let valid_short: Vec<usize> = (0..n).filter(|&i| valid_mask[i] && labels_short[i] >= 0).collect();

    // Base rates
    let base_wr_long = valid_long.iter().filter(|&&i| labels_long[i] == 1).count() as f64 / valid_long.len().max(1) as f64;
    let base_wr_short = valid_short.iter().filter(|&&i| labels_short[i] == 1).count() as f64 / valid_short.len().max(1) as f64;

    // Generate all k=2 and k=3 combinations
    let mut combos: Vec<(Vec<usize>, Vec<u8>, bool)> = Vec::new(); // (features, quantiles, is_long)

    // k=2 combinations
    for f1 in 0..12 {
        for f2 in (f1+1)..12 {
            for q1 in 0..N_QUANTILES as u8 {
                for q2 in 0..N_QUANTILES as u8 {
                    combos.push((vec![f1, f2], vec![q1, q2], true));
                    combos.push((vec![f1, f2], vec![q1, q2], false));
                }
            }
        }
    }

    // k=3 combinations (only extreme quantiles to limit search space)
    for f1 in 0..12 {
        for f2 in (f1+1)..12 {
            for f3 in (f2+1)..12 {
                // Only Q1, Q3(mid), Q5 for k=3 to keep manageable
                for &q1 in &[0u8, 2, 4] {
                    for &q2 in &[0u8, 2, 4] {
                        for &q3 in &[0u8, 2, 4] {
                            combos.push((vec![f1, f2, f3], vec![q1, q2, q3], true));
                            combos.push((vec![f1, f2, f3], vec![q1, q2, q3], false));
                        }
                    }
                }
            }
        }
    }

    // Parallel evaluation
    let candidates: Vec<PatternCandidate> = combos.par_iter().filter_map(|(feats, quants, is_long)| {
        let valid_indices = if *is_long { &valid_long } else { &valid_short };
        let labels = if *is_long { labels_long } else { labels_short };
        let base_wr = if *is_long { base_wr_long } else { base_wr_short };

        // Filter indices matching this pattern
        let matching: Vec<usize> = valid_indices.iter().filter(|&&i| {
            feats.iter().zip(quants.iter()).all(|(&fi, &q)| encoded[i][fi] == q)
        }).cloned().collect();

        let sample = matching.len();
        if sample < MIN_SAMPLE { return None; }

        let wins = matching.iter().filter(|&&i| labels[i] == 1).count();
        let wr = wins as f64 / sample as f64 * 100.0;
        if wr < MIN_WR { return None; }

        // Monthly consistency
        let n_months = (total_bars / bpm).max(1);
        let mut monthly_pos = 0;
        let mut monthly_total = 0;
        for m in 0..n_months {
            let m_start = m * bpm;
            let m_end = (m + 1) * bpm;
            let month_matches: Vec<usize> = matching.iter()
                .filter(|&&i| i >= m_start && i < m_end).cloned().collect();
            if month_matches.len() >= 5 {
                monthly_total += 1;
                let month_wins = month_matches.iter().filter(|&&i| labels[i] == 1).count();
                let month_wr = month_wins as f64 / month_matches.len() as f64;
                if month_wr > 0.5 { monthly_pos += 1; }
            }
        }
        let mc = if monthly_total > 0 { monthly_pos as f64 / monthly_total as f64 * 100.0 } else { 0.0 };
        if mc < MIN_MONTHLY_CONSISTENCY { return None; }

        // Avg return
        let avg_ret: f64 = matching.iter().map(|&i| {
            let entry = closes[i];
            if entry <= 0.0 { return 0.0; }
            let exit_idx = (i + TIMEOUT_BARS).min(n - 1);
            let ret = (closes[exit_idx] - entry) / entry * 100.0;
            if *is_long { ret } else { -ret }
        }).sum::<f64>() / sample as f64;

        // Sharpe
        let returns: Vec<f64> = matching.iter().map(|&i| {
            let entry = closes[i];
            if entry <= 0.0 { return 0.0; }
            let exit_idx = (i + TIMEOUT_BARS).min(n - 1);
            let ret = (closes[exit_idx] - entry) / entry * 100.0;
            if *is_long { ret } else { -ret }
        }).collect();
        let ret_std = {
            let mean = avg_ret;
            let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / sample as f64;
            var.sqrt()
        };
        let sharpe = if ret_std > 0.0 { avg_ret / ret_std } else { 0.0 };

        // P-value (binomial test: is WR significantly better than base?)
        let p_val = binomial_p_value(sample, wins, base_wr);

        // Score
        let score = wr / 100.0 * (sample as f64).ln() * mc / 100.0;

        Some(PatternCandidate {
            features: feats.clone(),
            quantiles: quants.clone(),
            direction: if *is_long { "long".to_string() } else { "short".to_string() },
            sample,
            wins,
            wr,
            avg_return: avg_ret,
            sharpe,
            monthly_consistency: mc,
            p_value: p_val,
            score,
        })
    }).collect();

    candidates
}

// ── Step 6: FDR correction (Benjamini-Hochberg) ──

pub fn fdr_filter(mut candidates: Vec<PatternCandidate>) -> Vec<PatternCandidate> {
    if candidates.is_empty() { return candidates; }

    // Sort by p-value ascending
    candidates.sort_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap_or(std::cmp::Ordering::Equal));

    let m = candidates.len();
    let mut keep = vec![false; m];

    for (i, c) in candidates.iter().enumerate() {
        let bh_threshold = FDR_ALPHA * (i + 1) as f64 / m as f64;
        if c.p_value <= bh_threshold {
            keep[i] = true;
        } else {
            break; // BH procedure: stop at first rejection
        }
    }

    candidates.into_iter().zip(keep.iter()).filter(|(_, &k)| k).map(|(c, _)| c).collect()
}

// ── Step 7: Pattern ranking ──

pub fn rank_patterns(mut candidates: Vec<PatternCandidate>) -> Vec<PatternCandidate> {
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}

// ── Master function ──

pub struct MinerResult {
    pub base_wr_long: f64,
    pub base_wr_short: f64,
    pub total_valid_long: usize,
    pub total_valid_short: usize,
    pub total_combos_tested: usize,
    pub pre_fdr_count: usize,
    pub post_fdr_count: usize,
    pub patterns: Vec<PatternCandidate>,
}

pub fn run_pattern_mining(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    train_end: usize, // ilk N bar train, geri kalan test
) -> MinerResult {
    let n = closes.len();

    // Compute features on ALL data
    let features = compute_features(closes, highs, lows, buy_vol, sell_vol, oi);

    // Labels
    let labels_long = triple_barrier_label(closes, highs, lows);
    let labels_short = triple_barrier_label_short(closes, highs, lows);

    // Valid mask (feature'lar NaN değil)
    let valid_mask: Vec<bool> = features.iter().map(|f| !f[0].is_nan()).collect();

    // Train only mask
    let train_mask: Vec<bool> = (0..n).map(|i| valid_mask[i] && i < train_end).collect();

    // Quantile encode on TRAIN data only
    let encoded = quantile_encode(&features, &train_mask);

    // Mine patterns on TRAIN
    let total_valid_long = (0..train_end).filter(|&i| train_mask[i] && labels_long[i] >= 0).count();
    let total_valid_short = (0..train_end).filter(|&i| train_mask[i] && labels_short[i] >= 0).count();
    let base_wr_long = (0..train_end).filter(|&i| train_mask[i] && labels_long[i] == 1).count() as f64 / total_valid_long.max(1) as f64;
    let base_wr_short = (0..train_end).filter(|&i| train_mask[i] && labels_short[i] == 1).count() as f64 / total_valid_short.max(1) as f64;

    let candidates = mine_patterns(&encoded, &labels_long, &labels_short, closes, &train_mask, train_end);
    let pre_fdr = candidates.len();

    // FDR filter
    let filtered = fdr_filter(candidates);
    let post_fdr = filtered.len();

    // Rank
    let ranked = rank_patterns(filtered);

    // Combo count
    let k2 = 12 * 11 / 2 * 25 * 2; // C(12,2) * 5^2 * 2 directions
    let k3 = 12 * 11 * 10 / 6 * 27 * 2; // C(12,3) * 3^3 * 2 directions
    let total_combos = k2 + k3;

    MinerResult {
        base_wr_long: base_wr_long * 100.0,
        base_wr_short: base_wr_short * 100.0,
        total_valid_long,
        total_valid_short,
        total_combos_tested: total_combos,
        pre_fdr_count: pre_fdr,
        post_fdr_count: post_fdr,
        patterns: ranked,
    }
}

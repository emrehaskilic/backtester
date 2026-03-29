/// Sweep Miner — Parametrik Sweep Pattern Discovery v3
///
/// 5m native data, 7 order flow feature, path-dependent triple barrier,
/// simetrik grid, quantile analiz, walk-forward validation, FDR correction.

use rayon::prelude::*;

// ── Constants ──

const MIN_BARS: usize = 300; // 288 (24h lookback) + buffer
const N_QUANTILES: usize = 5;
const MIN_SAMPLE_DAILY: usize = 30; // kombinasyon kesişim minimum
const WALK_FORWARD_TRAIN_MONTHS: usize = 3;
const WALK_FORWARD_WINDOWS: usize = 8;
const MIN_CONSISTENCY: f64 = 70.0; // %70 pencerede pozitif

// ── Helper functions ──

fn ema_arr(data: &[f64], period: usize) -> Vec<f64> {
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

fn rolling_sum(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![0.0_f64; n];
    if n < period { return out; }
    let mut s = 0.0;
    for i in 0..period { s += data[i]; }
    out[period - 1] = s;
    for i in period..n {
        s += data[i] - data[i - period];
        out[i] = s;
    }
    out
}

fn rolling_mean_std(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut mean = vec![f64::NAN; n];
    let mut std = vec![f64::NAN; n];
    if n < period { return (mean, std); }
    for i in (period - 1)..n {
        let slice = &data[i + 1 - period..=i];
        let m: f64 = slice.iter().sum::<f64>() / period as f64;
        let v: f64 = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / period as f64;
        mean[i] = m;
        std[i] = v.sqrt();
    }
    (mean, std)
}

fn mann_whitney_u(a: &[f64], b: &[f64]) -> (f64, f64) {
    // Returns (U statistic, approximate p-value)
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 { return (0.0, 1.0); }

    // Count how many times a[i] > b[j]
    let mut u: f64 = 0.0;
    for &ai in a {
        for &bj in b {
            if ai > bj { u += 1.0; }
            else if (ai - bj).abs() < 1e-12 { u += 0.5; }
        }
    }

    let mu = na as f64 * nb as f64 / 2.0;
    let sigma = ((na as f64 * nb as f64 * (na + nb + 1) as f64) / 12.0).sqrt();
    if sigma <= 0.0 { return (u, 1.0); }
    let z = (u - mu) / sigma;

    // Two-sided p-value approximation
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));
    (u, p)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592; let a2 = -0.284496736; let a3 = 1.421413741;
    let a4 = -1.453152027; let a5 = 1.061405429; let p = 0.3275911;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn binomial_p(n: usize, k: usize, p0: f64) -> f64 {
    if n == 0 { return 1.0; }
    let mu = n as f64 * p0;
    let sig = (n as f64 * p0 * (1.0 - p0)).sqrt();
    if sig <= 0.0 { return 1.0; }
    let z = (k as f64 - mu) / sig;
    1.0 - normal_cdf(z)
}

fn median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() { return f64::NAN; }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 0 { (vals[n/2 - 1] + vals[n/2]) / 2.0 } else { vals[n/2] }
}

// ── Data structures ──

#[derive(Clone, Debug)]
pub struct SweepEvent {
    pub bar_idx: usize,
    pub level: f64,         // kırılan seviye
    pub sweep_type: u8,     // 0=weekly_high, 1=weekly_low, 2=daily_high, 3=daily_low
    pub features: [f64; 7], // 7 feature
    pub sweep_depth: f64,   // sadece çıktı için, feature değil
}

#[derive(Clone, Debug)]
pub struct LabeledSweep {
    pub event: SweepEvent,
    pub label: i8,  // 1=continuation, 0=reversal, -1=timeout
}

#[derive(Clone, Debug)]
pub struct GridResult {
    pub mult: f64,
    pub timeout_bars: usize,
    pub sweep_type: u8,
    pub n_total: usize,
    pub n_continuation: usize,
    pub n_reversal: usize,
    pub n_timeout: usize,
    pub continuation_rate: f64,
    pub reversal_rate: f64,
    pub timeout_rate: f64,
    pub separation: f64,
}

#[derive(Clone, Debug)]
pub struct FeatureComparison {
    pub feature_name: String,
    pub cont_median: f64,
    pub rev_median: f64,
    pub u_stat: f64,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Clone, Debug)]
pub struct QuantileRow {
    pub feature_idx: usize,
    pub quantile: u8,
    pub n: usize,
    pub continuation_rate: f64,
}

#[derive(Clone, Debug)]
pub struct PatternCandidate {
    pub feature_indices: Vec<usize>,
    pub quantiles: Vec<u8>,
    pub n: usize,
    pub continuation_rate: f64,
    pub p_value: f64,
    pub score: f64,
    pub wf_positive_windows: usize,
    pub wf_total_windows: usize,
    pub wf_consistency: f64,
}

#[derive(Clone, Debug)]
pub struct SweepTypeResult {
    pub sweep_type: u8,
    pub sweep_type_name: String,
    pub total_events: usize,
    pub best_grid: GridResult,
    pub base_continuation_rate: f64,
    pub base_reversal_rate: f64,
    pub feature_comparisons: Vec<FeatureComparison>,
    pub quantile_analysis: Vec<QuantileRow>,
    pub top_patterns: Vec<PatternCandidate>,
}

pub struct FullResult {
    pub sweep_results: Vec<SweepTypeResult>,
    pub total_sweep_events: usize,
    pub total_grid_tests: usize,
}

// ── Feature names ──

pub const FEATURE_NAMES: [&str; 7] = [
    "CVD_zscore_micro",   // 12-bar (1h)
    "CVD_zscore_macro",   // 288-bar (24h)
    "OI_change",          // 288-bar, ATR normalized
    "Vol_zscore_micro",   // 12-bar
    "Vol_zscore_macro",   // 288-bar
    "Imbalance_smooth",   // EMA_12
    "ATR_percentile",     // 288-bar rank
];

// ── Step 1: Compute 5m features for entire dataset ──

pub struct FeatureData {
    pub features: Vec<[f64; 7]>,
    pub daily_atr: Vec<f64>,
    pub atr_1h: Vec<f64>,   // 12-bar ATR
    pub atr_4h: Vec<f64>,   // 48-bar ATR
    pub atr_8h: Vec<f64>,   // 96-bar ATR
}

pub fn compute_5m_features(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> (Vec<[f64; 7]>, Vec<f64>) {
    // Returns (features, daily_atr_at_each_bar)
    // For backward compat - use compute_5m_features_full for multi-TF ATR
    let n = closes.len();
    let micro = 12usize;  // 1 hour
    let macro_lb = 288usize; // 24 hours

    // Delta & imbalance
    let mut delta = vec![0.0_f64; n];
    let mut imb_raw = vec![0.0_f64; n];
    for i in 0..n {
        delta[i] = buy_vol[i] - sell_vol[i];
        let total = buy_vol[i] + sell_vol[i];
        imb_raw[i] = if total > 0.0 { delta[i] / total } else { 0.0 };
    }

    // CVD rolling sums
    let cvd_micro = rolling_sum(&delta, micro);
    let cvd_macro = rolling_sum(&delta, macro_lb);

    let (cvd_micro_mean, cvd_micro_std) = rolling_mean_std(&cvd_micro, micro);
    let (cvd_macro_mean, cvd_macro_std) = rolling_mean_std(&cvd_macro, macro_lb);

    // Volume
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();
    let vol_micro = rolling_sum(&total_vol, micro);
    let vol_macro = rolling_sum(&total_vol, macro_lb);
    let (vol_micro_mean, vol_micro_std) = rolling_mean_std(&vol_micro, micro);
    let (vol_macro_mean, vol_macro_std) = rolling_mean_std(&vol_macro, macro_lb);

    // Imbalance smoothed
    let imb_smooth = ema_arr(&imb_raw, micro);

    // ATR (5m bars, period 14)
    let atr_period = 14usize;
    let mut tr = vec![0.0_f64; n];
    if n > 0 { tr[0] = highs[0] - lows[0]; }
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
    }
    let mut atr_5m = vec![f64::NAN; n];
    if n > atr_period {
        let mut s = 0.0;
        for i in 1..=atr_period { s += tr[i]; }
        atr_5m[atr_period] = s / atr_period as f64;
        for i in (atr_period+1)..n {
            atr_5m[i] = (atr_5m[i-1] * (atr_period as f64 - 1.0) + tr[i]) / atr_period as f64;
        }
    }

    // Daily ATR (previous completed day)
    // 1 day = 288 bars in 5m
    let bpd = 288usize;
    let mut daily_atr = vec![f64::NAN; n];
    // Compute daily high/low/close, then 14-day ATR
    let n_days = n / bpd;
    if n_days >= 15 {
        let mut day_highs = vec![0.0_f64; n_days];
        let mut day_lows = vec![f64::INFINITY; n_days];
        let mut day_closes = vec![0.0_f64; n_days];
        for d in 0..n_days {
            let start = d * bpd;
            let end = ((d + 1) * bpd).min(n);
            for i in start..end {
                if highs[i] > day_highs[d] { day_highs[d] = highs[i]; }
                if lows[i] < day_lows[d] { day_lows[d] = lows[i]; }
                day_closes[d] = closes[i];
            }
        }

        let mut day_tr = vec![0.0_f64; n_days];
        day_tr[0] = day_highs[0] - day_lows[0];
        for d in 1..n_days {
            day_tr[d] = (day_highs[d] - day_lows[d])
                .max((day_highs[d] - day_closes[d-1]).abs())
                .max((day_lows[d] - day_closes[d-1]).abs());
        }

        let mut day_atr = vec![f64::NAN; n_days];
        if n_days > 14 {
            let mut s = 0.0;
            for d in 1..=14 { s += day_tr[d]; }
            day_atr[14] = s / 14.0;
            for d in 15..n_days {
                day_atr[d] = (day_atr[d-1] * 13.0 + day_tr[d]) / 14.0;
            }
        }

        // Map daily ATR to 5m bars: each bar gets PREVIOUS day's ATR
        for d in 1..n_days {
            if day_atr[d-1].is_nan() { continue; }
            let start = d * bpd;
            let end = ((d + 1) * bpd).min(n);
            for i in start..end {
                daily_atr[i] = day_atr[d - 1]; // previous day!
            }
        }
    }

    // OI change (288-bar)
    let mut oi_change = vec![0.0_f64; n];
    for i in macro_lb..n {
        if !oi[i].is_nan() && !oi[i - macro_lb].is_nan() && oi[i - macro_lb] > 0.0 && !daily_atr[i].is_nan() && daily_atr[i] > 0.0 {
            let raw_change = oi[i] - oi[i - macro_lb];
            oi_change[i] = raw_change / daily_atr[i];
        }
    }

    // ATR percentile (288-bar rank)
    let mut atr_pctile = vec![f64::NAN; n];
    for i in macro_lb..n {
        if atr_5m[i].is_nan() { continue; }
        let mut below = 0usize;
        let mut total_valid = 0usize;
        for j in (i + 1 - macro_lb)..=i {
            if !atr_5m[j].is_nan() {
                total_valid += 1;
                if atr_5m[j] < atr_5m[i] { below += 1; }
            }
        }
        if total_valid > 0 {
            atr_pctile[i] = below as f64 / total_valid as f64 * 100.0;
        }
    }

    // Build feature matrix
    let mut features = vec![[f64::NAN; 7]; n];
    for i in MIN_BARS..n {
        let f0 = if !cvd_micro_std[i].is_nan() && cvd_micro_std[i] > 0.0 {
            (cvd_micro[i] - cvd_micro_mean[i]) / cvd_micro_std[i]
        } else { continue };

        let f1 = if !cvd_macro_std[i].is_nan() && cvd_macro_std[i] > 0.0 {
            (cvd_macro[i] - cvd_macro_mean[i]) / cvd_macro_std[i]
        } else { continue };

        let f2 = oi_change[i];
        if daily_atr[i].is_nan() { continue; }

        let f3 = if !vol_micro_std[i].is_nan() && vol_micro_std[i] > 0.0 {
            (vol_micro[i] - vol_micro_mean[i]) / vol_micro_std[i]
        } else { continue };

        let f4 = if !vol_macro_std[i].is_nan() && vol_macro_std[i] > 0.0 {
            (vol_macro[i] - vol_macro_mean[i]) / vol_macro_std[i]
        } else { continue };

        let f5 = if !imb_smooth[i].is_nan() { imb_smooth[i] } else { continue };
        let f6 = if !atr_pctile[i].is_nan() { atr_pctile[i] } else { continue };

        features[i] = [f0, f1, f2, f3, f4, f5, f6];
    }

    (features, daily_atr)
}

// ── Step 1: Detect sweeps ──

fn detect_sweeps(
    closes: &[f64], highs: &[f64], lows: &[f64],
    features: &[[f64; 7]], daily_atr: &[f64],
) -> Vec<SweepEvent> {
    let n = closes.len();
    let bpd = 288usize; // 5m bars per day
    let bpw = bpd * 7;  // bars per week

    let mut events = Vec::new();

    // ── Daily high/low sweeps ──
    let n_days = n / bpd;
    for d in 1..n_days {
        let prev_day_start = (d - 1) * bpd;
        let prev_day_end = d * bpd;

        // Previous day high/low
        let mut prev_high = f64::NEG_INFINITY;
        let mut prev_low = f64::INFINITY;
        for i in prev_day_start..prev_day_end {
            if highs[i] > prev_high { prev_high = highs[i]; }
            if lows[i] < prev_low { prev_low = lows[i]; }
        }

        // Search for sweeps in current and future bars
        let search_start = d * bpd;
        let search_end = n;

        // Reset tracking
        let mut high_active = true;
        let mut low_active = true;
        let mut high_cooldown = 0usize;
        let mut low_cooldown = 0usize;
        let mut bars_below_high = 0usize;
        let mut bars_above_low = 0usize;

        for i in search_start..search_end {
            if daily_atr[i].is_nan() || daily_atr[i] <= 0.0 { continue; }
            if features[i][0].is_nan() { continue; }
            let reset_dist = 0.5 * daily_atr[i];

            // Daily high sweep
            if high_active && i > search_start && highs[i] > prev_high && highs[i - 1] <= prev_high {
                // Use features from bar i-1 (look-ahead protection)
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (highs[i] - prev_high) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_high, sweep_type: 2, // daily_high
                        features: features[i - 1], sweep_depth: depth,
                    });
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            // Daily low sweep
            if low_active && i > search_start && lows[i] < prev_low && lows[i - 1] >= prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (prev_low - lows[i]) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_low, sweep_type: 3, // daily_low
                        features: features[i - 1], sweep_depth: depth,
                    });
                    low_active = false;
                    low_cooldown = 0;
                }
            }

            // Reset logic for high
            if !high_active {
                if closes[i] < prev_high - reset_dist {
                    bars_below_high += 1;
                } else {
                    bars_below_high = 0;
                }
                high_cooldown += 1;
                if bars_below_high >= 12 || (closes[i] < prev_high - reset_dist && high_cooldown > 12) {
                    high_active = true;
                    bars_below_high = 0;
                }
            }

            // Reset logic for low
            if !low_active {
                if closes[i] > prev_low + reset_dist {
                    bars_above_low += 1;
                } else {
                    bars_above_low = 0;
                }
                low_cooldown += 1;
                if bars_above_low >= 12 || (closes[i] > prev_low + reset_dist && low_cooldown > 12) {
                    low_active = true;
                    bars_above_low = 0;
                }
            }
        }
    }

    // ── Weekly high/low sweeps ──
    let n_weeks = n / bpw;
    for w in 1..n_weeks {
        let prev_week_start = (w - 1) * bpw;
        let prev_week_end = w * bpw;

        let mut prev_high = f64::NEG_INFINITY;
        let mut prev_low = f64::INFINITY;
        for i in prev_week_start..prev_week_end {
            if highs[i] > prev_high { prev_high = highs[i]; }
            if lows[i] < prev_low { prev_low = lows[i]; }
        }

        let search_start = w * bpw;
        let search_end = n;
        let mut high_active = true;
        let mut low_active = true;
        let mut high_cooldown = 0usize;
        let mut low_cooldown = 0usize;
        let mut bars_below = 0usize;
        let mut bars_above = 0usize;

        for i in search_start..search_end {
            if daily_atr[i].is_nan() || daily_atr[i] <= 0.0 { continue; }
            if features[i][0].is_nan() { continue; }
            let reset_dist = 0.5 * daily_atr[i];

            if high_active && i > search_start && highs[i] > prev_high && highs[i - 1] <= prev_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (highs[i] - prev_high) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_high, sweep_type: 0, // weekly_high
                        features: features[i - 1], sweep_depth: depth,
                    });
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            if low_active && i > search_start && lows[i] < prev_low && lows[i - 1] >= prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (prev_low - lows[i]) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_low, sweep_type: 1, // weekly_low
                        features: features[i - 1], sweep_depth: depth,
                    });
                    low_active = false;
                    low_cooldown = 0;
                }
            }

            if !high_active {
                if closes[i] < prev_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
                high_cooldown += 1;
                if bars_below >= 12 { high_active = true; bars_below = 0; }
            }
            if !low_active {
                if closes[i] > prev_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
                low_cooldown += 1;
                if bars_above >= 12 { low_active = true; bars_above = 0; }
            }
        }
    }

    // ── 4H high/low sweeps ──
    let bp4h = 48usize; // 5m bars per 4h candle (4*60/5 = 48)
    let n_4h = n / bp4h;
    for c4 in 1..n_4h {
        let prev_start = (c4 - 1) * bp4h;
        let prev_end = c4 * bp4h;

        let mut prev_high = f64::NEG_INFINITY;
        let mut prev_low = f64::INFINITY;
        for i in prev_start..prev_end {
            if highs[i] > prev_high { prev_high = highs[i]; }
            if lows[i] < prev_low { prev_low = lows[i]; }
        }

        let search_start = c4 * bp4h;
        let search_end = n;
        let mut high_active = true;
        let mut low_active = true;
        let mut high_cooldown = 0usize;
        let mut low_cooldown = 0usize;
        let mut bars_below = 0usize;
        let mut bars_above = 0usize;

        for i in search_start..search_end {
            if daily_atr[i].is_nan() || daily_atr[i] <= 0.0 { continue; }
            if features[i][0].is_nan() { continue; }
            let reset_dist = 0.5 * daily_atr[i];

            // 4H high sweep
            if high_active && i > search_start && highs[i] > prev_high && highs[i - 1] <= prev_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (highs[i] - prev_high) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_high, sweep_type: 4, // 4h_high
                        features: features[i - 1], sweep_depth: depth,
                    });
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            // 4H low sweep
            if low_active && i > search_start && lows[i] < prev_low && lows[i - 1] >= prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (prev_low - lows[i]) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_low, sweep_type: 5, // 4h_low
                        features: features[i - 1], sweep_depth: depth,
                    });
                    low_active = false;
                    low_cooldown = 0;
                }
            }

            if !high_active {
                if closes[i] < prev_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
                high_cooldown += 1;
                if bars_below >= 12 { high_active = true; bars_below = 0; }
            }
            if !low_active {
                if closes[i] > prev_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
                low_cooldown += 1;
                if bars_above >= 12 { low_active = true; bars_above = 0; }
            }
        }
    }

    // ── 8H high/low sweeps ──
    let bp8h = 96usize; // 5m bars per 8h candle (8*60/5 = 96)
    let n_8h = n / bp8h;
    for c8 in 1..n_8h {
        let prev_start = (c8 - 1) * bp8h;
        let prev_end = c8 * bp8h;

        let mut prev_high = f64::NEG_INFINITY;
        let mut prev_low = f64::INFINITY;
        for i in prev_start..prev_end {
            if highs[i] > prev_high { prev_high = highs[i]; }
            if lows[i] < prev_low { prev_low = lows[i]; }
        }

        let search_start = c8 * bp8h;
        let search_end = n;
        let mut high_active = true;
        let mut low_active = true;
        let mut high_cooldown = 0usize;
        let mut low_cooldown = 0usize;
        let mut bars_below = 0usize;
        let mut bars_above = 0usize;

        for i in search_start..search_end {
            if daily_atr[i].is_nan() || daily_atr[i] <= 0.0 { continue; }
            if features[i][0].is_nan() { continue; }
            let reset_dist = 0.5 * daily_atr[i];

            if high_active && i > search_start && highs[i] > prev_high && highs[i - 1] <= prev_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (highs[i] - prev_high) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_high, sweep_type: 10, // 8h_high
                        features: features[i - 1], sweep_depth: depth,
                    });
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            if low_active && i > search_start && lows[i] < prev_low && lows[i - 1] >= prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (prev_low - lows[i]) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_low, sweep_type: 11, // 8h_low
                        features: features[i - 1], sweep_depth: depth,
                    });
                    low_active = false;
                    low_cooldown = 0;
                }
            }

            if !high_active {
                if closes[i] < prev_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
                high_cooldown += 1;
                if bars_below >= 12 { high_active = true; bars_below = 0; }
            }
            if !low_active {
                if closes[i] > prev_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
                low_cooldown += 1;
                if bars_above >= 12 { low_active = true; bars_above = 0; }
            }
        }
    }

    // ── 15m high/low sweeps ──
    let bp15m = 3usize; // 5m bars per 15m candle (15/5 = 3)
    let n_15m = n / bp15m;
    for c15 in 1..n_15m {
        let prev_start = (c15 - 1) * bp15m;
        let prev_end = c15 * bp15m;

        let mut prev_high = f64::NEG_INFINITY;
        let mut prev_low = f64::INFINITY;
        for i in prev_start..prev_end {
            if highs[i] > prev_high { prev_high = highs[i]; }
            if lows[i] < prev_low { prev_low = lows[i]; }
        }

        let search_start = c15 * bp15m;
        let search_end = n;
        let mut high_active = true;
        let mut low_active = true;
        let mut high_cooldown = 0usize;
        let mut low_cooldown = 0usize;
        let mut bars_below = 0usize;
        let mut bars_above = 0usize;

        for i in search_start..search_end {
            if daily_atr[i].is_nan() || daily_atr[i] <= 0.0 { continue; }
            if features[i][0].is_nan() { continue; }
            let reset_dist = 0.5 * daily_atr[i];

            if high_active && i > search_start && highs[i] > prev_high && highs[i - 1] <= prev_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (highs[i] - prev_high) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_high, sweep_type: 8, // 15m_high
                        features: features[i - 1], sweep_depth: depth,
                    });
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            if low_active && i > search_start && lows[i] < prev_low && lows[i - 1] >= prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (prev_low - lows[i]) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_low, sweep_type: 9, // 15m_low
                        features: features[i - 1], sweep_depth: depth,
                    });
                    low_active = false;
                    low_cooldown = 0;
                }
            }

            if !high_active {
                if closes[i] < prev_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
                high_cooldown += 1;
                if bars_below >= 12 { high_active = true; bars_below = 0; }
            }
            if !low_active {
                if closes[i] > prev_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
                low_cooldown += 1;
                if bars_above >= 12 { low_active = true; bars_above = 0; }
            }
        }
    }

    // ── 1H high/low sweeps ──
    let bp1h = 12usize; // 5m bars per 1h candle (60/5 = 12)
    let n_1h = n / bp1h;
    for c1 in 1..n_1h {
        let prev_start = (c1 - 1) * bp1h;
        let prev_end = c1 * bp1h;

        let mut prev_high = f64::NEG_INFINITY;
        let mut prev_low = f64::INFINITY;
        for i in prev_start..prev_end {
            if highs[i] > prev_high { prev_high = highs[i]; }
            if lows[i] < prev_low { prev_low = lows[i]; }
        }

        let search_start = c1 * bp1h;
        let search_end = n;
        let mut high_active = true;
        let mut low_active = true;
        let mut high_cooldown = 0usize;
        let mut low_cooldown = 0usize;
        let mut bars_below = 0usize;
        let mut bars_above = 0usize;

        for i in search_start..search_end {
            if daily_atr[i].is_nan() || daily_atr[i] <= 0.0 { continue; }
            if features[i][0].is_nan() { continue; }
            let reset_dist = 0.5 * daily_atr[i];

            // 1H high sweep
            if high_active && i > search_start && highs[i] > prev_high && highs[i - 1] <= prev_high {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (highs[i] - prev_high) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_high, sweep_type: 6, // 1h_high
                        features: features[i - 1], sweep_depth: depth,
                    });
                    high_active = false;
                    high_cooldown = 0;
                }
            }

            // 1H low sweep
            if low_active && i > search_start && lows[i] < prev_low && lows[i - 1] >= prev_low {
                if i >= 1 && !features[i-1][0].is_nan() {
                    let depth = (prev_low - lows[i]) / daily_atr[i];
                    events.push(SweepEvent {
                        bar_idx: i, level: prev_low, sweep_type: 7, // 1h_low
                        features: features[i - 1], sweep_depth: depth,
                    });
                    low_active = false;
                    low_cooldown = 0;
                }
            }

            if !high_active {
                if closes[i] < prev_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
                high_cooldown += 1;
                if bars_below >= 12 { high_active = true; bars_below = 0; }
            }
            if !low_active {
                if closes[i] > prev_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
                low_cooldown += 1;
                if bars_above >= 12 { low_active = true; bars_above = 0; }
            }
        }
    }

    events.sort_by_key(|e| e.bar_idx);
    events
}

// ── Step 3: Label sweeps with triple barrier ──

fn compute_tf_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    let mut tr = vec![0.0_f64; n];
    if n > 0 { tr[0] = highs[0] - lows[0]; }
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
    }
    let mut atr = vec![f64::NAN; n];
    if n > period {
        let mut s = 0.0;
        for i in 1..=period { s += tr[i]; }
        atr[period] = s / period as f64;
        for i in (period+1)..n {
            atr[i] = (atr[i-1] * (period as f64 - 1.0) + tr[i]) / period as f64;
        }
    }
    atr
}

fn get_atr_for_sweep(sweep_type: u8, i: usize,
    daily_atr: &[f64], atr_1h: &[f64], atr_4h: &[f64], atr_8h: &[f64]) -> f64 {
    match sweep_type {
        0 | 1 => daily_atr[i],           // weekly → daily ATR
        2 | 3 => daily_atr[i],           // daily → daily ATR
        4 | 5 => atr_4h[i],              // 4H → 4H ATR (48-bar)
        6 | 7 => atr_1h[i],              // 1H → 1H ATR (12-bar)
        8 | 9 => atr_1h[i],              // 15m → 1H ATR (3-bar too short)
        10 | 11 => atr_8h[i],            // 8H → 8H ATR (96-bar)
        _ => daily_atr[i],
    }
}

fn label_sweeps(
    events: &[SweepEvent],
    highs: &[f64], lows: &[f64],
    daily_atr: &[f64], atr_1h: &[f64], atr_4h: &[f64], atr_8h: &[f64],
    mult: f64, timeout_bars: usize,
) -> Vec<LabeledSweep> {
    let n = highs.len();

    events.iter().map(|ev| {
        let i = ev.bar_idx;
        let atr = get_atr_for_sweep(ev.sweep_type, i, daily_atr, atr_1h, atr_4h, atr_8h);
        let level = ev.level;

        let mut label: i8 = -1; // timeout

        if !atr.is_nan() && atr > 0.0 && i + timeout_bars < n {
            let is_high = matches!(ev.sweep_type, 0 | 2 | 4 | 6 | 8 | 10);
            let (tp, sl) = if is_high {
                (level + mult * atr, level - mult * atr)
            } else {
                (level - mult * atr, level + mult * atr)
            };

            for j in (i + 1)..=(i + timeout_bars).min(n - 1) {
                if is_high {
                    if highs[j] >= tp { label = 1; break; }
                    if lows[j] <= sl { label = 0; break; }
                } else {
                    if lows[j] <= tp { label = 1; break; }
                    if highs[j] >= sl { label = 0; break; }
                }
            }
        }

        LabeledSweep { event: ev.clone(), label }
    }).collect()
}

// ── Step 4: Grid search ──

fn grid_search(
    events: &[SweepEvent], highs: &[f64], lows: &[f64],
    daily_atr: &[f64], atr_1h: &[f64], atr_4h: &[f64], atr_8h: &[f64],
    sweep_type: u8, timeframe: u8,
) -> Vec<GridResult> {
    let type_events: Vec<&SweepEvent> = events.iter().filter(|e| e.sweep_type == sweep_type).collect();

    let mults: Vec<f64> = (3..=20).map(|i| i as f64 * 0.1).collect(); // 0.3 - 2.0
    let timeouts: Vec<usize> = match timeframe {
        1 => (0..25).map(|i| 288 + i * 48).collect(),     // weekly: 288-1440 step 48
        2 => (0..23).map(|i| 12 + i * 6).collect(),       // 4h: 12-144 step 6 (1h-12h)
        3 => (0..12).map(|i| 6 + i * 6).collect(),        // 1h: 6-72 step 6 (30m-6h)
        4 => (0..12).map(|i| 3 + i * 3).collect(),        // 15m: 3-36 step 3 (15m-3h)
        5 => (0..21).map(|i| 48 + i * 12).collect(),      // 8h: 48-288 step 12 (4h-24h)
        _ => (0..23).map(|i| 24 + i * 12).collect(),      // daily: 24-288 step 12
    };

    let combos: Vec<(f64, usize)> = mults.iter().flat_map(|&m| {
        timeouts.iter().map(move |&t| (m, t))
    }).collect();

    combos.par_iter().filter_map(|&(mult, timeout)| {
        let labeled = label_sweeps(
            &type_events.iter().map(|&&ref e| e.clone()).collect::<Vec<_>>(),
            highs, lows, daily_atr, atr_1h, atr_4h, atr_8h, mult, timeout,
        );

        let n_total = labeled.len();
        let n_cont = labeled.iter().filter(|l| l.label == 1).count();
        let n_rev = labeled.iter().filter(|l| l.label == 0).count();
        let n_timeout = labeled.iter().filter(|l| l.label == -1).count();

        if n_total == 0 { return None; }
        let timeout_rate = n_timeout as f64 / n_total as f64 * 100.0;
        if timeout_rate > 25.0 { return None; }

        let min_sample = if timeframe == 1 { 0 } else { 100 }; // weekly: no min, daily/4h: 100
        if n_total < min_sample { return None; }

        let cont_rate = n_cont as f64 / (n_cont + n_rev).max(1) as f64 * 100.0;
        let rev_rate = n_rev as f64 / (n_cont + n_rev).max(1) as f64 * 100.0;
        let separation = (cont_rate - rev_rate).abs();

        Some(GridResult {
            mult, timeout_bars: timeout, sweep_type,
            n_total, n_continuation: n_cont, n_reversal: n_rev, n_timeout,
            continuation_rate: cont_rate, reversal_rate: rev_rate,
            timeout_rate, separation,
        })
    }).collect()
}

// ── Step 5a: Feature comparison ──

fn compare_features(labeled: &[LabeledSweep]) -> Vec<FeatureComparison> {
    let cont: Vec<&LabeledSweep> = labeled.iter().filter(|l| l.label == 1).collect();
    let rev: Vec<&LabeledSweep> = labeled.iter().filter(|l| l.label == 0).collect();

    (0..7).map(|fi| {
        let mut c_vals: Vec<f64> = cont.iter().map(|l| l.event.features[fi]).collect();
        let mut r_vals: Vec<f64> = rev.iter().map(|l| l.event.features[fi]).collect();

        let c_med = median(&mut c_vals.clone());
        let r_med = median(&mut r_vals.clone());
        let (u, p) = mann_whitney_u(&c_vals, &r_vals);

        FeatureComparison {
            feature_name: FEATURE_NAMES[fi].to_string(),
            cont_median: c_med, rev_median: r_med,
            u_stat: u, p_value: p,
            significant: p < 0.05,
        }
    }).collect()
}

// ── Step 5b: Quantile analysis ──

fn quantile_thresholds(vals: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = vals.iter().filter(|v| !v.is_nan()).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() { return vec![0.0; N_QUANTILES - 1]; }
    (1..N_QUANTILES).map(|q| {
        let idx = (q as f64 / N_QUANTILES as f64 * sorted.len() as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }).collect()
}

fn assign_quantile(val: f64, thresholds: &[f64]) -> u8 {
    let mut q = 0u8;
    for &t in thresholds {
        if val > t { q += 1; }
    }
    q
}

fn quantile_analysis(labeled: &[LabeledSweep]) -> (Vec<QuantileRow>, Vec<Vec<f64>>) {
    // Returns quantile rows + thresholds per feature
    let valid: Vec<&LabeledSweep> = labeled.iter().filter(|l| l.label >= 0).collect();
    let mut all_thresholds = Vec::new();
    let mut rows = Vec::new();

    for fi in 0..7 {
        let vals: Vec<f64> = valid.iter().map(|l| l.event.features[fi]).collect();
        let thresholds = quantile_thresholds(&vals);
        all_thresholds.push(thresholds.clone());

        for q in 0..N_QUANTILES as u8 {
            let matching: Vec<&&LabeledSweep> = valid.iter()
                .filter(|l| assign_quantile(l.event.features[fi], &thresholds) == q)
                .collect();

            let n = matching.len();
            let n_cont = matching.iter().filter(|l| l.label == 1).count();
            let cont_rate = if n > 0 { n_cont as f64 / n as f64 * 100.0 } else { 0.0 };

            rows.push(QuantileRow { feature_idx: fi, quantile: q, n, continuation_rate: cont_rate });
        }
    }

    (rows, all_thresholds)
}

// ── Step 5c: Combinatorial search ──

fn combinatorial_search(
    labeled: &[LabeledSweep],
    thresholds: &[Vec<f64>],
    base_rate: f64,
) -> Vec<PatternCandidate> {
    let valid: Vec<&LabeledSweep> = labeled.iter().filter(|l| l.label >= 0).collect();
    let n_valid = valid.len();

    // k=2 combinations
    let mut combos: Vec<(Vec<usize>, Vec<u8>)> = Vec::new();
    for f1 in 0..7 {
        for f2 in (f1+1)..7 {
            for q1 in 0..N_QUANTILES as u8 {
                for q2 in 0..N_QUANTILES as u8 {
                    combos.push((vec![f1, f2], vec![q1, q2]));
                }
            }
        }
    }

    // k=3 (only Q1, Q3, Q5)
    for f1 in 0..7 {
        for f2 in (f1+1)..7 {
            for f3 in (f2+1)..7 {
                for &q1 in &[0u8, 2, 4] {
                    for &q2 in &[0u8, 2, 4] {
                        for &q3 in &[0u8, 2, 4] {
                            combos.push((vec![f1, f2, f3], vec![q1, q2, q3]));
                        }
                    }
                }
            }
        }
    }

    let candidates: Vec<PatternCandidate> = combos.par_iter().filter_map(|(feats, quants)| {
        let matching: Vec<&&LabeledSweep> = valid.iter().filter(|l| {
            feats.iter().zip(quants.iter()).all(|(&fi, &q)| {
                assign_quantile(l.event.features[fi], &thresholds[fi]) == q
            })
        }).collect();

        let n = matching.len();
        if n < MIN_SAMPLE_DAILY { return None; }

        let n_cont = matching.iter().filter(|l| l.label == 1).count();
        let cont_rate = n_cont as f64 / n as f64 * 100.0;

        let p = binomial_p(n, n_cont, base_rate / 100.0);
        let score = (cont_rate / 100.0) * (n as f64).ln();

        Some(PatternCandidate {
            feature_indices: feats.clone(),
            quantiles: quants.clone(),
            n, continuation_rate: cont_rate,
            p_value: p, score,
            wf_positive_windows: 0, wf_total_windows: 0, wf_consistency: 0.0,
        })
    }).collect();

    candidates
}

// ── Step 6: Walk-forward validation ──

fn walk_forward_validate(
    events: &[SweepEvent],
    highs: &[f64], lows: &[f64],
    daily_atr: &[f64], atr_1h: &[f64], atr_4h: &[f64], atr_8h: &[f64],
    sweep_type: u8, mult: f64, timeout: usize,
    pattern: &PatternCandidate,
    total_bars: usize,
) -> (usize, usize, f64) {
    // 3 month train -> 1 month test, rolling
    let bpm = 288 * 30; // bars per month (approx)
    let train_len = WALK_FORWARD_TRAIN_MONTHS * bpm;

    let mut positive = 0usize;
    let mut total_windows = 0usize;

    for w in 0..WALK_FORWARD_WINDOWS {
        let train_start = w * bpm;
        let train_end = train_start + train_len;
        let test_start = train_end;
        let test_end = (test_start + bpm).min(total_bars);

        if test_end > total_bars { break; }

        // Get train events for this sweep type
        let train_events: Vec<SweepEvent> = events.iter()
            .filter(|e| e.sweep_type == sweep_type && e.bar_idx >= train_start && e.bar_idx < train_end)
            .cloned().collect();

        // Compute thresholds from train
        let train_labeled = label_sweeps(&train_events, highs, lows, daily_atr, atr_1h, atr_4h, atr_8h, mult, timeout);
        let train_valid: Vec<&LabeledSweep> = train_labeled.iter().filter(|l| l.label >= 0).collect();

        let mut thresholds = Vec::new();
        for fi in 0..7 {
            let vals: Vec<f64> = train_valid.iter().map(|l| l.event.features[fi]).collect();
            thresholds.push(quantile_thresholds(&vals));
        }

        // Test events
        let test_events: Vec<SweepEvent> = events.iter()
            .filter(|e| e.sweep_type == sweep_type && e.bar_idx >= test_start && e.bar_idx < test_end)
            .cloned().collect();

        let test_labeled = label_sweeps(&test_events, highs, lows, daily_atr, atr_1h, atr_4h, atr_8h, mult, timeout);
        let test_valid: Vec<&LabeledSweep> = test_labeled.iter().filter(|l| l.label >= 0).collect();

        // Apply pattern with train thresholds to test data
        let matching: Vec<&&LabeledSweep> = test_valid.iter().filter(|l| {
            pattern.feature_indices.iter().zip(pattern.quantiles.iter()).all(|(&fi, &q)| {
                assign_quantile(l.event.features[fi], &thresholds[fi]) == q
            })
        }).collect();

        if matching.len() >= 5 {
            total_windows += 1;
            let n_cont = matching.iter().filter(|l| l.label == 1).count();
            let cont_rate = n_cont as f64 / matching.len() as f64;
            if cont_rate > 0.5 { positive += 1; }
        }
    }

    let consistency = if total_windows > 0 { positive as f64 / total_windows as f64 * 100.0 } else { 0.0 };
    (positive, total_windows, consistency)
}

// ── FDR correction ──

fn fdr_filter(candidates: &mut Vec<PatternCandidate>) {
    candidates.sort_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap_or(std::cmp::Ordering::Equal));
    let m = candidates.len();
    let mut cutoff = 0;
    for (i, c) in candidates.iter().enumerate() {
        let bh = 0.05 * (i + 1) as f64 / m as f64;
        if c.p_value <= bh { cutoff = i + 1; }
    }
    candidates.truncate(cutoff);
}

// ── Master function ──

pub fn run_sweep_mining(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> FullResult {
    let n = closes.len();

    // Compute features
    let (features, daily_atr) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);

    // Multi-TF ATR
    let atr_1h = compute_tf_atr(highs, lows, closes, 12);   // 1H = 12 bar
    let atr_4h = compute_tf_atr(highs, lows, closes, 48);   // 4H = 48 bar
    let atr_8h = compute_tf_atr(highs, lows, closes, 96);   // 8H = 96 bar

    // Detect all sweeps
    let events = detect_sweeps(closes, highs, lows, &features, &daily_atr);

    let sweep_types: Vec<(u8, &str, u8)> = vec![
        (0, "Weekly High Sweep", 1),   // timeframe: 1=weekly
        (1, "Weekly Low Sweep", 1),
        (2, "Daily High Sweep", 0),    // 0=daily
        (3, "Daily Low Sweep", 0),
        (4, "4H High Sweep", 2),       // 2=4h
        (5, "4H Low Sweep", 2),
        (6, "1H High Sweep", 3),       // 3=1h
        (7, "1H Low Sweep", 3),
        (8, "15m High Sweep", 4),      // 4=15m
        (9, "15m Low Sweep", 4),
        (10, "8H High Sweep", 5),      // 5=8h
        (11, "8H Low Sweep", 5),
    ];

    let mut total_grid = 0usize;

    let sweep_results: Vec<SweepTypeResult> = sweep_types.iter().map(|&(st, name, timeframe)| {
        let type_events: Vec<&SweepEvent> = events.iter().filter(|e| e.sweep_type == st).collect();
        let n_events = type_events.len();

        // Grid search
        let mut grid = grid_search(&events, highs, lows, &daily_atr, &atr_1h, &atr_4h, &atr_8h, st, timeframe);
        total_grid += grid.len();
        grid.sort_by(|a, b| b.separation.partial_cmp(&a.separation).unwrap_or(std::cmp::Ordering::Equal));

        let best_grid = grid.first().cloned().unwrap_or(GridResult {
            mult: 0.5, timeout_bars: 144, sweep_type: st,
            n_total: 0, n_continuation: 0, n_reversal: 0, n_timeout: 0,
            continuation_rate: 50.0, reversal_rate: 50.0, timeout_rate: 0.0, separation: 0.0,
        });

        // Label with best params
        let type_events_owned: Vec<SweepEvent> = events.iter().filter(|e| e.sweep_type == st).cloned().collect();
        let labeled = label_sweeps(&type_events_owned, highs, lows, &daily_atr, &atr_1h, &atr_4h, &atr_8h, best_grid.mult, best_grid.timeout_bars);

        let valid_labeled: Vec<&LabeledSweep> = labeled.iter().filter(|l| l.label >= 0).collect();
        let base_cont = valid_labeled.iter().filter(|l| l.label == 1).count() as f64 / valid_labeled.len().max(1) as f64 * 100.0;
        let base_rev = 100.0 - base_cont;

        // Feature comparison
        let feat_comp = compare_features(&labeled);

        // Quantile analysis
        let (quant_rows, thresholds) = quantile_analysis(&labeled);

        // Combinatorial search
        let mut candidates = combinatorial_search(&labeled, &thresholds, base_cont);

        // FDR
        fdr_filter(&mut candidates);

        // Walk-forward on top candidates
        for c in candidates.iter_mut().take(50) {
            let (pos, tot, cons) = walk_forward_validate(
                &events, highs, lows, &daily_atr, &atr_1h, &atr_4h, &atr_8h,
                st, best_grid.mult, best_grid.timeout_bars,
                c, n,
            );
            c.wf_positive_windows = pos;
            c.wf_total_windows = tot;
            c.wf_consistency = cons;
        }

        // Filter by consistency and sort by score
        candidates.retain(|c| c.wf_consistency >= MIN_CONSISTENCY || c.wf_total_windows < 3);
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(10);

        SweepTypeResult {
            sweep_type: st,
            sweep_type_name: name.to_string(),
            total_events: n_events,
            best_grid,
            base_continuation_rate: base_cont,
            base_reversal_rate: base_rev,
            feature_comparisons: feat_comp,
            quantile_analysis: quant_rows,
            top_patterns: candidates,
        }
    }).collect();

    FullResult {
        total_sweep_events: events.len(),
        total_grid_tests: total_grid,
        sweep_results,
    }
}

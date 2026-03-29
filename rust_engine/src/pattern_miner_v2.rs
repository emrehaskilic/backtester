/// Pattern Miner V2 — Comprehensive 7-task analysis.
///
/// Task 1: Symmetry (long vs short)
/// Task 2: Regime isolation (uptrend/downtrend/sideways)
/// Task 3: Volatility dependency
/// Task 4: Pattern generalization (relaxed quantiles)
/// Task 5: Signal strength filtering
/// Task 6: Pullback continuation discovery
/// Task 7: Final robust patterns

use rayon::prelude::*;
use crate::pattern_miner::{
    compute_features, triple_barrier_label, triple_barrier_label_short,
    quantile_encode, mine_patterns, fdr_filter, rank_patterns,
    PatternCandidate, FEATURE_NAMES, QUANTILE_LABELS,
};

const MIN_BARS: usize = 200;
const TIMEOUT_BARS: usize = 160;

// ── Regime detection ──

#[derive(Clone, Debug, PartialEq)]
pub enum Regime { Uptrend, Downtrend, Sideways }

fn detect_regimes(closes: &[f64], period: usize) -> Vec<Regime> {
    let n = closes.len();
    let mut regimes = vec![Regime::Sideways; n];

    // HTF EMA
    let mut ema = vec![f64::NAN; n];
    if n == 0 { return regimes; }
    let k = 2.0 / (period as f64 + 1.0);
    ema[0] = closes[0];
    for i in 1..n {
        let prev = if ema[i-1].is_nan() { closes[i] } else { ema[i-1] };
        ema[i] = closes[i] * k + prev * (1.0 - k);
    }

    let slope_lb = 100; // slope lookback
    for i in slope_lb..n {
        if ema[i].is_nan() || ema[i - slope_lb].is_nan() { continue; }
        let slope = (ema[i] - ema[i - slope_lb]) / ema[i - slope_lb] * 100.0;
        regimes[i] = if slope > 0.5 { Regime::Uptrend }
                     else if slope < -0.5 { Regime::Downtrend }
                     else { Regime::Sideways };
    }
    regimes
}

// ── Volatility buckets ──

#[derive(Clone, Debug, PartialEq)]
pub enum VolBucket { Low, Mid, High }

fn volatility_buckets(features: &[[f64; 12]], valid: &[bool]) -> Vec<VolBucket> {
    // ATR_percentile is feature index 8
    features.iter().zip(valid.iter()).map(|(f, &v)| {
        if !v || f[8].is_nan() { return VolBucket::Mid; }
        let atr_p = f[8];
        if atr_p < 30.0 { VolBucket::Low }
        else if atr_p > 70.0 { VolBucket::High }
        else { VolBucket::Mid }
    }).collect()
}

// ── Pattern WR per subset ──

fn pattern_wr_on_subset(
    encoded: &[[u8; 12]], labels: &[i8], mask: &[bool],
    features: &[usize], quantiles: &[u8],
) -> (usize, usize, f64) {
    let mut total = 0usize;
    let mut wins = 0usize;
    for i in 0..encoded.len() {
        if !mask[i] || labels[i] < 0 { continue; }
        let matches = features.iter().zip(quantiles.iter()).all(|(&fi, &q)| encoded[i][fi] == q);
        if matches {
            total += 1;
            if labels[i] == 1 { wins += 1; }
        }
    }
    let wr = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };
    (total, wins, wr)
}

// Relaxed: accept q-1 to q+1 range
fn pattern_wr_relaxed(
    encoded: &[[u8; 12]], labels: &[i8], mask: &[bool],
    features: &[usize], quantiles: &[u8],
) -> (usize, usize, f64) {
    let mut total = 0usize;
    let mut wins = 0usize;
    for i in 0..encoded.len() {
        if !mask[i] || labels[i] < 0 { continue; }
        let matches = features.iter().zip(quantiles.iter()).all(|(&fi, &q)| {
            let actual = encoded[i][fi];
            let lo = if q > 0 { q - 1 } else { 0 };
            let hi = if q < 4 { q + 1 } else { 4 };
            actual >= lo && actual <= hi
        });
        if matches {
            total += 1;
            if labels[i] == 1 { wins += 1; }
        }
    }
    let wr = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };
    (total, wins, wr)
}

// ── Signal strength ──

fn signal_strength(features: &[[f64; 12]], valid: &[bool]) -> Vec<f64> {
    // strength = |price_slope| * |CVD_slope| * ATR_percentile
    features.iter().zip(valid.iter()).map(|(f, &v)| {
        if !v || f[3].is_nan() || f[1].is_nan() || f[8].is_nan() { return 0.0; }
        f[3].abs() * f[1].abs() * f[8] / 100.0
    }).collect()
}

// ── Pullback patterns ──

fn find_pullback_patterns(
    features: &[[f64; 12]], encoded: &[[u8; 12]],
    labels_long: &[i8], labels_short: &[i8],
    valid: &[bool], train_end: usize,
) -> Vec<PullbackPattern> {
    let n = features.len().min(train_end);
    let mut results = Vec::new();

    // Pullback long: trend up + temporary dip + CVD not reversing
    // price_slope (f3) > 0 (trend up)
    // price_vs_EMA (f4) < 0 (pulled back below EMA)
    // CVD_slope (f1) >= 0 (CVD not reversing)
    // + optional: ATR elevated, OI stable

    let pullback_configs: Vec<(&str, &str, Vec<(usize, Box<dyn Fn(f64) -> bool + Send + Sync>)>)> = vec![
        // Long pullbacks
        ("Pullback Long: Trend + Dip + CVD holds", "long", vec![
            (3, Box::new(|v: f64| v > 0.0)),    // price_slope positive (uptrend)
            (4, Box::new(|v: f64| v < -0.5)),    // price below EMA (pulled back)
            (1, Box::new(|v: f64| v > -0.5)),    // CVD slope not strongly negative
        ]),
        ("Pullback Long: Trend + KC Lower + Volume", "long", vec![
            (3, Box::new(|v: f64| v > 0.0)),    // uptrend
            (9, Box::new(|v: f64| v < 0.3)),     // KC distance low (near lower band)
            (7, Box::new(|v: f64| v < 0.5)),     // volume not spiking (quiet pullback)
        ]),
        ("Pullback Long: Trend + OI stable + ATR elevated", "long", vec![
            (3, Box::new(|v: f64| v > 0.0)),    // uptrend
            (4, Box::new(|v: f64| v < 0.0)),     // below EMA
            (5, Box::new(|v: f64| v > -0.5)),    // OI not dropping much
            (8, Box::new(|v: f64| v > 40.0)),    // ATR still elevated
        ]),
        // Short pullbacks
        ("Pullback Short: Downtrend + Rally + CVD holds", "short", vec![
            (3, Box::new(|v: f64| v < 0.0)),    // price_slope negative (downtrend)
            (4, Box::new(|v: f64| v > 0.5)),     // price above EMA (rallied)
            (1, Box::new(|v: f64| v < 0.5)),     // CVD slope not strongly positive
        ]),
        ("Pullback Short: Downtrend + KC Upper + Volume", "short", vec![
            (3, Box::new(|v: f64| v < 0.0)),    // downtrend
            (9, Box::new(|v: f64| v > 0.7)),     // KC distance high (near upper band)
            (7, Box::new(|v: f64| v < 0.5)),     // quiet pullback
        ]),
        ("Pullback Short: Downtrend + OI stable + ATR", "short", vec![
            (3, Box::new(|v: f64| v < 0.0)),    // downtrend
            (4, Box::new(|v: f64| v > 0.0)),     // above EMA
            (5, Box::new(|v: f64| v > -0.5)),    // OI not dropping
            (8, Box::new(|v: f64| v > 40.0)),    // ATR elevated
        ]),
    ];

    for (name, dir, conditions) in &pullback_configs {
        let labels = if *dir == "long" { labels_long } else { labels_short };

        let mut total = 0usize;
        let mut wins = 0usize;
        let mut returns = Vec::new();

        for i in MIN_BARS..n {
            if !valid[i] || labels[i] < 0 { continue; }

            let matches = conditions.iter().all(|(fi, check)| {
                !features[i][*fi].is_nan() && check(features[i][*fi])
            });

            if matches {
                total += 1;
                if labels[i] == 1 { wins += 1; }
                // Compute return
                let exit_idx = (i + TIMEOUT_BARS).min(features.len() - 1);
                // We need closes but don't have direct access here
                // Store index for return computation later
                returns.push(i);
            }
        }

        let wr = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };

        results.push(PullbackPattern {
            name: name.to_string(),
            direction: dir.to_string(),
            sample: total,
            wins,
            wr,
            conditions_desc: conditions.iter().map(|(fi, _)| {
                FEATURE_NAMES[*fi].to_string()
            }).collect(),
        });
    }

    results
}

#[derive(Clone, Debug)]
pub struct PullbackPattern {
    pub name: String,
    pub direction: String,
    pub sample: usize,
    pub wins: usize,
    pub wr: f64,
    pub conditions_desc: Vec<String>,
}

// ── Enriched pattern with regime/vol analysis ──

#[derive(Clone, Debug)]
pub struct EnrichedPattern {
    pub base: PatternCandidate,
    // Regime WR
    pub wr_uptrend: f64,
    pub n_uptrend: usize,
    pub wr_downtrend: f64,
    pub n_downtrend: usize,
    pub wr_sideways: f64,
    pub n_sideways: usize,
    pub regime_count: usize, // how many regimes WR > 50%
    // Volatility WR
    pub wr_vol_low: f64,
    pub n_vol_low: usize,
    pub wr_vol_mid: f64,
    pub n_vol_mid: usize,
    pub wr_vol_high: f64,
    pub n_vol_high: usize,
    // Generalized
    pub gen_wr: f64,
    pub gen_n: usize,
    pub gen_wr_drop: f64,
    // Signal strength top 30% WR
    pub strength_top30_wr: f64,
    pub strength_top30_n: usize,
}

// ── Master analysis ──

pub struct AnalysisResult {
    pub base_wr_long: f64,
    pub base_wr_short: f64,
    // Task 1: symmetry
    pub long_patterns: Vec<PatternCandidate>,
    pub short_patterns: Vec<PatternCandidate>,
    // Task 2-5: enriched top patterns
    pub enriched: Vec<EnrichedPattern>,
    // Task 6: pullback
    pub pullback_patterns: Vec<PullbackPattern>,
    // Stats
    pub total_combos: usize,
    pub regime_distribution: [usize; 3], // up, down, sideways bar counts
}

pub fn run_comprehensive_analysis(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    train_end: usize,
) -> AnalysisResult {
    let n = closes.len();

    // ── Compute features ──
    let features = compute_features(closes, highs, lows, buy_vol, sell_vol, oi);
    let labels_long = triple_barrier_label(closes, highs, lows);
    let labels_short = triple_barrier_label_short(closes, highs, lows);
    let valid_mask: Vec<bool> = features.iter().map(|f| !f[0].is_nan()).collect();
    let train_mask: Vec<bool> = (0..n).map(|i| valid_mask[i] && i < train_end).collect();
    let encoded = quantile_encode(&features, &train_mask);

    // Base WR
    let vl: Vec<usize> = (0..train_end).filter(|&i| train_mask[i] && labels_long[i] >= 0).collect();
    let vs: Vec<usize> = (0..train_end).filter(|&i| train_mask[i] && labels_short[i] >= 0).collect();
    let base_wr_long = vl.iter().filter(|&&i| labels_long[i] == 1).count() as f64 / vl.len().max(1) as f64 * 100.0;
    let base_wr_short = vs.iter().filter(|&&i| labels_short[i] == 1).count() as f64 / vs.len().max(1) as f64 * 100.0;

    // ── Task 1: Mine long and short separately ──
    let all_candidates = mine_patterns(&encoded, &labels_long, &labels_short, closes, &train_mask, train_end);
    let filtered = fdr_filter(all_candidates);
    let ranked = rank_patterns(filtered);

    let long_patterns: Vec<PatternCandidate> = ranked.iter().filter(|p| p.direction == "long").cloned().collect();
    let short_patterns: Vec<PatternCandidate> = ranked.iter().filter(|p| p.direction == "short").cloned().collect();

    // ── Regime detection ──
    let regimes = detect_regimes(closes, 200); // 200-bar EMA
    let regime_dist = [
        (0..train_end).filter(|&i| regimes[i] == Regime::Uptrend).count(),
        (0..train_end).filter(|&i| regimes[i] == Regime::Downtrend).count(),
        (0..train_end).filter(|&i| regimes[i] == Regime::Sideways).count(),
    ];

    // Regime masks
    let up_mask: Vec<bool> = (0..n).map(|i| train_mask[i] && regimes[i] == Regime::Uptrend).collect();
    let down_mask: Vec<bool> = (0..n).map(|i| train_mask[i] && regimes[i] == Regime::Downtrend).collect();
    let side_mask: Vec<bool> = (0..n).map(|i| train_mask[i] && regimes[i] == Regime::Sideways).collect();

    // ── Volatility buckets ──
    let vol_buckets = volatility_buckets(&features, &valid_mask);
    let vol_low_mask: Vec<bool> = (0..n).map(|i| train_mask[i] && vol_buckets[i] == VolBucket::Low).collect();
    let vol_mid_mask: Vec<bool> = (0..n).map(|i| train_mask[i] && vol_buckets[i] == VolBucket::Mid).collect();
    let vol_high_mask: Vec<bool> = (0..n).map(|i| train_mask[i] && vol_buckets[i] == VolBucket::High).collect();

    // ── Signal strength ──
    let strengths = signal_strength(&features, &valid_mask);

    // ── Task 2-5: Enrich top patterns (parallel) ──
    let top_n = ranked.len().min(50);
    let enriched: Vec<EnrichedPattern> = ranked[..top_n].par_iter().map(|pat| {
        let labels = if pat.direction == "long" { &labels_long } else { &labels_short };

        // Task 2: Regime WR
        let (n_up, _, wr_up) = pattern_wr_on_subset(&encoded, labels, &up_mask, &pat.features, &pat.quantiles);
        let (n_down, _, wr_down) = pattern_wr_on_subset(&encoded, labels, &down_mask, &pat.features, &pat.quantiles);
        let (n_side, _, wr_side) = pattern_wr_on_subset(&encoded, labels, &side_mask, &pat.features, &pat.quantiles);

        let regime_count = [wr_up, wr_down, wr_side].iter().filter(|&&wr| wr > 50.0).count();

        // Task 3: Volatility WR
        let (n_vl, _, wr_vl) = pattern_wr_on_subset(&encoded, labels, &vol_low_mask, &pat.features, &pat.quantiles);
        let (n_vm, _, wr_vm) = pattern_wr_on_subset(&encoded, labels, &vol_mid_mask, &pat.features, &pat.quantiles);
        let (n_vh, _, wr_vh) = pattern_wr_on_subset(&encoded, labels, &vol_high_mask, &pat.features, &pat.quantiles);

        // Task 4: Generalized
        let (gen_n, gen_wins, gen_wr) = pattern_wr_relaxed(&encoded, labels, &train_mask, &pat.features, &pat.quantiles);
        let gen_wr_drop = pat.wr - gen_wr;

        // Task 5: Signal strength top 30%
        // Find strength threshold for top 30%
        let matching_strengths: Vec<f64> = (0..train_end).filter(|&i| {
            train_mask[i] && labels[i] >= 0 &&
            pat.features.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| encoded[i][fi] == q)
        }).map(|i| strengths[i]).collect();

        let (str_top30_n, str_top30_wr) = if matching_strengths.len() >= 10 {
            let mut sorted_str = matching_strengths.clone();
            sorted_str.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top30_thresh = sorted_str[(sorted_str.len() as f64 * 0.3) as usize];

            let top30_indices: Vec<usize> = (0..train_end).filter(|&i| {
                train_mask[i] && labels[i] >= 0 &&
                pat.features.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| encoded[i][fi] == q) &&
                strengths[i] >= top30_thresh
            }).collect();

            let wins = top30_indices.iter().filter(|&&i| labels[i] == 1).count();
            let wr = if !top30_indices.is_empty() { wins as f64 / top30_indices.len() as f64 * 100.0 } else { 0.0 };
            (top30_indices.len(), wr)
        } else {
            (0, 0.0)
        };

        EnrichedPattern {
            base: pat.clone(),
            wr_uptrend: wr_up, n_uptrend: n_up,
            wr_downtrend: wr_down, n_downtrend: n_down,
            wr_sideways: wr_side, n_sideways: n_side,
            regime_count,
            wr_vol_low: wr_vl, n_vol_low: n_vl,
            wr_vol_mid: wr_vm, n_vol_mid: n_vm,
            wr_vol_high: wr_vh, n_vol_high: n_vh,
            gen_wr, gen_n, gen_wr_drop,
            strength_top30_wr: str_top30_wr, strength_top30_n: str_top30_n,
        }
    }).collect();

    // ── Task 6: Pullback patterns ──
    let pullback_patterns = find_pullback_patterns(&features, &encoded, &labels_long, &labels_short, &train_mask, train_end);

    // Combo count
    let k2 = 12 * 11 / 2 * 25 * 2;
    let k3 = 12 * 11 * 10 / 6 * 27 * 2;

    AnalysisResult {
        base_wr_long,
        base_wr_short,
        long_patterns,
        short_patterns,
        enriched,
        pullback_patterns,
        total_combos: k2 + k3,
        regime_distribution: regime_dist,
    }
}

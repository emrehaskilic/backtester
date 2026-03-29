/// Sweep Candle Pattern Miner — Quantile bazli pattern discovery
/// Mum kapanisi bazli continuation/reversal uzerinde.

use rayon::prelude::*;
use crate::sweep_candle_analysis::{run_candle_analysis, run_candle_analysis_offset, run_candle_analysis_full, run_candle_analysis_cross, CandleEvent, CandleSignal, FEATURE_NAMES};

const N_QUANTILES: usize = 5;
const MIN_SAMPLE: usize = 30;
const MIN_CONSISTENCY: f64 = 70.0;

// ── Quantile helpers ──

fn quantile_thresholds(vals: &[f64]) -> [f64; 4] {
    let mut sorted: Vec<f64> = vals.iter().filter(|v| !v.is_nan()).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut th = [0.0_f64; 4];
    if sorted.is_empty() { return th; }
    for q in 0..4 {
        let idx = ((q + 1) as f64 / 5.0 * sorted.len() as f64) as usize;
        th[q] = sorted[idx.min(sorted.len() - 1)];
    }
    th
}

fn get_q(val: f64, th: &[f64; 4]) -> u8 {
    let mut q = 0u8;
    for &t in th { if val > t { q += 1; } }
    q
}

fn binomial_p(n: usize, k: usize, p0: f64) -> f64 {
    if n == 0 { return 1.0; }
    let mu = n as f64 * p0;
    let sig = (n as f64 * p0 * (1.0 - p0)).sqrt();
    if sig <= 0.0 { return 1.0; }
    let z = (k as f64 - mu) / sig;
    1.0 - 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
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

// ── Pattern candidate ──

#[derive(Clone, Debug)]
pub struct CandlePattern {
    pub sweep_type: String,       // "high" or "low"
    pub target: String,           // "continuation" or "reversal"
    pub feature_indices: Vec<usize>,
    pub quantiles: Vec<u8>,
    pub n: usize,
    pub target_count: usize,
    pub target_rate: f64,
    pub p_value: f64,
    pub score: f64,
    pub wf_positive: usize,
    pub wf_total: usize,
    pub wf_consistency: f64,
}

// ── Result ──

pub struct CandleMinerResult {
    pub high_base_cont_rate: f64,
    pub low_base_cont_rate: f64,
    pub high_total: usize,
    pub low_total: usize,

    // Quantile single feature analysis
    pub quantile_rows: Vec<QuantileRow>,

    // Top patterns
    pub high_cont_patterns: Vec<CandlePattern>,
    pub high_rev_patterns: Vec<CandlePattern>,
    pub low_cont_patterns: Vec<CandlePattern>,
    pub low_rev_patterns: Vec<CandlePattern>,
}

#[derive(Clone, Debug)]
pub struct QuantileRow {
    pub sweep_type: String,
    pub feature_idx: usize,
    pub feature_name: String,
    pub quantile: u8,
    pub n: usize,
    pub cont_count: usize,
    pub cont_rate: f64,
}

// ── Walk-forward ──

fn walk_forward(
    events: &[(bool, [u8; 7])], // (is_continuation, quantile_encoded)
    pattern_feats: &[usize], pattern_quants: &[u8],
    base_rate: f64,
) -> (usize, usize, f64) {
    let n = events.len();
    let window = n / 11; // ~1 month
    let train_windows = 3;

    let mut positive = 0usize;
    let mut total = 0usize;

    for w in 0..8 {
        let train_start = w * window;
        let train_end = train_start + train_windows * window;
        let test_start = train_end;
        let test_end = (test_start + window).min(n);

        if test_end > n { break; }

        // Check pattern in test
        let test_matches: Vec<bool> = events[test_start..test_end].iter().filter(|(_, eq)| {
            pattern_feats.iter().zip(pattern_quants.iter()).all(|(&fi, &q)| eq[fi] == q)
        }).map(|(is_cont, _)| *is_cont).collect();

        if test_matches.len() >= 5 {
            total += 1;
            let cont_count = test_matches.iter().filter(|&&c| c).count();
            let rate = cont_count as f64 / test_matches.len() as f64;
            if rate > base_rate { positive += 1; }
        }
    }

    let consistency = if total > 0 { positive as f64 / total as f64 * 100.0 } else { 0.0 };
    (positive, total, consistency)
}

// ── Mine patterns for one sweep type ──

fn mine_one_side(
    events: &[&CandleEvent],
    is_cont_fn: &dyn Fn(&CandleSignal) -> bool,
    is_target_fn: &dyn Fn(&CandleSignal) -> bool,
    sweep_type: &str,
    target_name: &str,
) -> (f64, usize, Vec<QuantileRow>, Vec<CandlePattern>) {
    if events.is_empty() { return (0.0, 0, Vec::new(), Vec::new()); }

    let n = events.len();

    // Compute quantile thresholds
    let mut thresholds = [[0.0_f64; 4]; 7];
    for fi in 0..7 {
        let vals: Vec<f64> = events.iter().map(|e| e.features[fi]).collect();
        thresholds[fi] = quantile_thresholds(&vals);
    }

    // Encode
    let encoded: Vec<(bool, [u8; 7])> = events.iter().map(|e| {
        let is_target = is_target_fn(&e.signal);
        let mut eq = [0u8; 7];
        for fi in 0..7 {
            eq[fi] = get_q(e.features[fi], &thresholds[fi]);
        }
        (is_target, eq)
    }).collect();

    let target_count = encoded.iter().filter(|(t, _)| *t).count();
    let base_rate = target_count as f64 / n as f64;

    // Single feature quantile analysis
    let mut q_rows: Vec<QuantileRow> = Vec::new();
    for fi in 0..7 {
        for q in 0..N_QUANTILES as u8 {
            let matching: Vec<&(bool, [u8; 7])> = encoded.iter().filter(|(_, eq)| eq[fi] == q).collect();
            let cnt = matching.len();
            let tgt = matching.iter().filter(|(t, _)| *t).count();
            let rate = if cnt > 0 { tgt as f64 / cnt as f64 * 100.0 } else { 0.0 };
            q_rows.push(QuantileRow {
                sweep_type: sweep_type.to_string(),
                feature_idx: fi,
                feature_name: FEATURE_NAMES[fi].to_string(),
                quantile: q, n: cnt, cont_count: tgt, cont_rate: rate,
            });
        }
    }

    // Combinatorial search
    let mut combos: Vec<(Vec<usize>, Vec<u8>)> = Vec::new();

    // k=2
    for f1 in 0..7 {
        for f2 in (f1+1)..7 {
            for q1 in 0..N_QUANTILES as u8 {
                for q2 in 0..N_QUANTILES as u8 {
                    combos.push((vec![f1, f2], vec![q1, q2]));
                }
            }
        }
    }

    // k=3 (Q1, Q3, Q5 only)
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

    let mut candidates: Vec<CandlePattern> = combos.par_iter().filter_map(|(feats, quants)| {
        let matching: Vec<&(bool, [u8; 7])> = encoded.iter().filter(|(_, eq)| {
            feats.iter().zip(quants.iter()).all(|(&fi, &q)| eq[fi] == q)
        }).collect();

        let sample = matching.len();
        if sample < MIN_SAMPLE { return None; }

        let tgt = matching.iter().filter(|(t, _)| *t).count();
        let rate = tgt as f64 / sample as f64 * 100.0;

        // Must be significantly better than base rate
        if rate <= base_rate * 100.0 + 5.0 { return None; } // at least 5% above base

        let p = binomial_p(sample, tgt, base_rate);
        if p > 0.01 { return None; } // strict

        let score = (rate / 100.0) * (sample as f64).ln();

        Some(CandlePattern {
            sweep_type: sweep_type.to_string(),
            target: target_name.to_string(),
            feature_indices: feats.clone(),
            quantiles: quants.clone(),
            n: sample, target_count: tgt, target_rate: rate,
            p_value: p, score,
            wf_positive: 0, wf_total: 0, wf_consistency: 0.0,
        })
    }).collect();

    // FDR
    candidates.sort_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap_or(std::cmp::Ordering::Equal));
    let m = candidates.len();
    let mut cutoff = 0;
    for (i, c) in candidates.iter().enumerate() {
        let bh = 0.05 * (i + 1) as f64 / m.max(1) as f64;
        if c.p_value <= bh { cutoff = i + 1; }
    }
    candidates.truncate(cutoff);

    // Walk-forward on top 50
    for c in candidates.iter_mut().take(50) {
        let (pos, tot, cons) = walk_forward(&encoded, &c.feature_indices, &c.quantiles, base_rate);
        c.wf_positive = pos;
        c.wf_total = tot;
        c.wf_consistency = cons;
    }

    // Filter by consistency and sort
    candidates.retain(|c| c.wf_consistency >= MIN_CONSISTENCY || c.wf_total < 3);
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(15);

    (base_rate * 100.0, n, q_rows, candidates)
}

// ── Master function ──

pub fn run_candle_mining(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> CandleMinerResult {
    run_candle_mining_full(closes, highs, lows, buy_vol, sell_vol, oi, 0, 12)
}

pub fn run_candle_mining_offset(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    feat_bar_offset: usize,
) -> CandleMinerResult {
    run_candle_mining_full(closes, highs, lows, buy_vol, sell_vol, oi, feat_bar_offset, 12)
}

pub fn run_candle_mining_full(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> CandleMinerResult {
    // Run candle analysis first
    let analysis = run_candle_analysis_full(closes, highs, lows, buy_vol, sell_vol, oi, feat_bar_offset, bars_per_candle);

    // Split events by sweep type
    let high_events: Vec<&CandleEvent> = analysis.events.iter().filter(|e| {
        matches!(e.signal, CandleSignal::HighContinuation | CandleSignal::HighReversal)
    }).collect();

    let low_events: Vec<&CandleEvent> = analysis.events.iter().filter(|e| {
        matches!(e.signal, CandleSignal::LowContinuation | CandleSignal::LowReversal)
    }).collect();

    // Mine high sweep: continuation patterns
    let (hc_base, hc_total, hc_qrows, hc_patterns) = mine_one_side(
        &high_events,
        &|s| matches!(s, CandleSignal::HighContinuation),
        &|s| matches!(s, CandleSignal::HighContinuation),
        "high", "continuation",
    );

    // Mine high sweep: reversal patterns
    let (_, _, _, hr_patterns) = mine_one_side(
        &high_events,
        &|s| matches!(s, CandleSignal::HighReversal),
        &|s| matches!(s, CandleSignal::HighReversal),
        "high", "reversal",
    );

    // Mine low sweep: continuation patterns
    let (lc_base, lc_total, lc_qrows, lc_patterns) = mine_one_side(
        &low_events,
        &|s| matches!(s, CandleSignal::LowContinuation),
        &|s| matches!(s, CandleSignal::LowContinuation),
        "low", "continuation",
    );

    // Mine low sweep: reversal patterns
    let (_, _, _, lr_patterns) = mine_one_side(
        &low_events,
        &|s| matches!(s, CandleSignal::LowReversal),
        &|s| matches!(s, CandleSignal::LowReversal),
        "low", "reversal",
    );

    let mut all_qrows = hc_qrows;
    all_qrows.extend(lc_qrows);

    CandleMinerResult {
        high_base_cont_rate: hc_base,
        low_base_cont_rate: lc_base,
        high_total: hc_total,
        low_total: lc_total,
        quantile_rows: all_qrows,
        high_cont_patterns: hc_patterns,
        high_rev_patterns: hr_patterns,
        low_cont_patterns: lc_patterns,
        low_rev_patterns: lr_patterns,
    }
}

/// Cross-asset miner: ETH mumlari + BTC feature'lari
pub fn run_candle_mining_cross(
    candle_closes: &[f64], candle_highs: &[f64], candle_lows: &[f64],
    feat_closes: &[f64], feat_highs: &[f64], feat_lows: &[f64],
    feat_buy_vol: &[f64], feat_sell_vol: &[f64], feat_oi: &[f64],
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> CandleMinerResult {
    let analysis = run_candle_analysis_cross(
        candle_closes, candle_highs, candle_lows,
        feat_closes, feat_highs, feat_lows,
        feat_buy_vol, feat_sell_vol, feat_oi,
        feat_bar_offset, bars_per_candle,
    );

    let high_events: Vec<&CandleEvent> = analysis.events.iter().filter(|e| {
        matches!(e.signal, CandleSignal::HighContinuation | CandleSignal::HighReversal)
    }).collect();

    let low_events: Vec<&CandleEvent> = analysis.events.iter().filter(|e| {
        matches!(e.signal, CandleSignal::LowContinuation | CandleSignal::LowReversal)
    }).collect();

    let (hc_base, hc_total, hc_qrows, hc_patterns) = mine_one_side(
        &high_events,
        &|s| matches!(s, CandleSignal::HighContinuation),
        &|s| matches!(s, CandleSignal::HighContinuation),
        "high", "continuation",
    );

    let (_, _, _, hr_patterns) = mine_one_side(
        &high_events,
        &|s| matches!(s, CandleSignal::HighReversal),
        &|s| matches!(s, CandleSignal::HighReversal),
        "high", "reversal",
    );

    let (lc_base, lc_total, lc_qrows, lc_patterns) = mine_one_side(
        &low_events,
        &|s| matches!(s, CandleSignal::LowContinuation),
        &|s| matches!(s, CandleSignal::LowContinuation),
        "low", "continuation",
    );

    let (_, _, _, lr_patterns) = mine_one_side(
        &low_events,
        &|s| matches!(s, CandleSignal::LowReversal),
        &|s| matches!(s, CandleSignal::LowReversal),
        "low", "reversal",
    );

    let mut all_qrows = hc_qrows;
    all_qrows.extend(lc_qrows);

    CandleMinerResult {
        high_base_cont_rate: hc_base,
        low_base_cont_rate: lc_base,
        high_total: hc_total,
        low_total: lc_total,
        quantile_rows: all_qrows,
        high_cont_patterns: hc_patterns,
        high_rev_patterns: hr_patterns,
        low_cont_patterns: lc_patterns,
        low_rev_patterns: lr_patterns,
    }
}

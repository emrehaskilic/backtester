/// Sweep Analysis v2 — 28 feature ile kapsamli analiz + miner
/// Ayni candle siniflandirma mantigi, genisletilmis feature seti

use rayon::prelude::*;
use crate::features_v2::{compute_features_v2, N_FEATURES_V2, FEATURE_NAMES_V2};

const N_QUANTILES: usize = 5;
const MIN_SAMPLE: usize = 30;
const MIN_CONSISTENCY: f64 = 70.0;

// ── Sinyal tipleri (ayni) ──

#[derive(Clone, Debug, PartialEq)]
pub enum Signal {
    HighCont, HighRev, HighAmbig,
    LowCont, LowRev, LowAmbig,
    InsideBar,
}

#[derive(Clone, Debug)]
pub struct SweepEvent {
    pub bar_idx: usize,
    pub signal: Signal,
    pub features: [f64; N_FEATURES_V2],
}

// ── Pattern ──

#[derive(Clone, Debug)]
pub struct PatternV2 {
    pub sweep_type: String,
    pub target: String,
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

// ── Quantile row ──

#[derive(Clone, Debug)]
pub struct QRow {
    pub sweep_type: String,
    pub feature_idx: usize,
    pub feature_name: String,
    pub quantile: u8,
    pub n: usize,
    pub cont_count: usize,
    pub cont_rate: f64,
}

// ── Result ──

pub struct MinerResultV2 {
    pub high_base_cont_rate: f64,
    pub low_base_cont_rate: f64,
    pub high_total: usize,
    pub low_total: usize,
    pub quantile_rows: Vec<QRow>,
    pub high_cont_patterns: Vec<PatternV2>,
    pub high_rev_patterns: Vec<PatternV2>,
    pub low_cont_patterns: Vec<PatternV2>,
    pub low_rev_patterns: Vec<PatternV2>,
}

// ── Helpers ──

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

// ── Walk-forward ──

fn walk_forward(
    events: &[(bool, Vec<u8>)],
    pattern_feats: &[usize], pattern_quants: &[u8],
    base_rate: f64,
) -> (usize, usize, f64) {
    let n = events.len();
    if n < 44 { return (0, 0, 0.0); }
    let window = n / 11;
    let train_windows = 3;

    let mut positive = 0usize;
    let mut total = 0usize;

    for w in 0..8 {
        let train_start = w * window;
        let train_end = train_start + train_windows * window;
        let test_start = train_end;
        let test_end = (test_start + window).min(n);
        if test_end > n { break; }

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

// ── Mine one side ──

fn mine_one_side(
    events: &[&SweepEvent],
    is_target_fn: &dyn Fn(&Signal) -> bool,
    sweep_type: &str,
    target_name: &str,
) -> (f64, usize, Vec<QRow>, Vec<PatternV2>) {
    if events.is_empty() { return (0.0, 0, Vec::new(), Vec::new()); }

    let n = events.len();
    let nf = N_FEATURES_V2;

    // Quantile thresholds per feature
    let mut thresholds = vec![[0.0_f64; 4]; nf];
    for fi in 0..nf {
        let vals: Vec<f64> = events.iter().map(|e| e.features[fi]).collect();
        thresholds[fi] = quantile_thresholds(&vals);
    }

    // Encode
    let encoded: Vec<(bool, Vec<u8>)> = events.iter().map(|e| {
        let is_target = is_target_fn(&e.signal);
        let eq: Vec<u8> = (0..nf).map(|fi| get_q(e.features[fi], &thresholds[fi])).collect();
        (is_target, eq)
    }).collect();

    let target_count = encoded.iter().filter(|(t, _)| *t).count();
    let base_rate = target_count as f64 / n as f64;

    // Single feature quantile analysis
    let mut q_rows: Vec<QRow> = Vec::new();
    for fi in 0..nf {
        for q in 0..N_QUANTILES as u8 {
            let matching: Vec<&(bool, Vec<u8>)> = encoded.iter().filter(|(_, eq)| eq[fi] == q).collect();
            let cnt = matching.len();
            let tgt = matching.iter().filter(|(t, _)| *t).count();
            let rate = if cnt > 0 { tgt as f64 / cnt as f64 * 100.0 } else { 0.0 };
            q_rows.push(QRow {
                sweep_type: sweep_type.to_string(),
                feature_idx: fi,
                feature_name: FEATURE_NAMES_V2[fi].to_string(),
                quantile: q, n: cnt, cont_count: tgt, cont_rate: rate,
            });
        }
    }

    // Combinatorial search (k=2, all features)
    let mut combos: Vec<(Vec<usize>, Vec<u8>)> = Vec::new();
    for f1 in 0..nf {
        for f2 in (f1+1)..nf {
            for q1 in 0..N_QUANTILES as u8 {
                for q2 in 0..N_QUANTILES as u8 {
                    combos.push((vec![f1, f2], vec![q1, q2]));
                }
            }
        }
    }

    // k=3 sadece Q1, Q3, Q5 (cok fazla combo olmasin diye top 10 feature ile sinirla)
    // Top features by spread
    let mut feat_spreads: Vec<(usize, f64)> = (0..nf).map(|fi| {
        let vals: Vec<f64> = (0..5).map(|q| {
            let rows: Vec<&QRow> = q_rows.iter().filter(|r| r.feature_idx == fi && r.quantile == q as u8).collect();
            if rows.is_empty() { 0.0 } else { rows[0].cont_rate }
        }).collect();
        let spread = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                   - vals.iter().cloned().fold(f64::INFINITY, f64::min);
        (fi, spread)
    }).collect();
    feat_spreads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_feats: Vec<usize> = feat_spreads.iter().take(10).map(|&(fi, _)| fi).collect();

    for i in 0..top_feats.len() {
        for j in (i+1)..top_feats.len() {
            for k in (j+1)..top_feats.len() {
                let f1 = top_feats[i]; let f2 = top_feats[j]; let f3 = top_feats[k];
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

    let mut candidates: Vec<PatternV2> = combos.par_iter().filter_map(|(feats, quants)| {
        let matching: Vec<&(bool, Vec<u8>)> = encoded.iter().filter(|(_, eq)| {
            feats.iter().zip(quants.iter()).all(|(&fi, &q)| eq[fi] == q)
        }).collect();

        let sample = matching.len();
        if sample < MIN_SAMPLE { return None; }

        let tgt = matching.iter().filter(|(t, _)| *t).count();
        let rate = tgt as f64 / sample as f64 * 100.0;

        if rate <= base_rate * 100.0 + 5.0 { return None; }

        let p = binomial_p(sample, tgt, base_rate);
        if p > 0.01 { return None; }

        let score = (rate / 100.0) * (sample as f64).ln();

        Some(PatternV2 {
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

    candidates.retain(|c| c.wf_consistency >= MIN_CONSISTENCY || c.wf_total < 3);
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(20);

    (base_rate * 100.0, n, q_rows, candidates)
}

// ── Master function ──

pub fn run_sweep_miner_v2(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64],
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> MinerResultV2 {
    let bp = bars_per_candle;
    let n = closes.len();

    // Compute v2 features
    let features = compute_features_v2(closes, highs, lows, buy_vol, sell_vol);

    // Build candles
    let n_candles = n / bp;
    let mut candles: Vec<(usize, f64, f64, f64, f64)> = Vec::with_capacity(n_candles);
    for c_idx in 0..n_candles {
        let start = c_idx * bp;
        let end = start + bp;
        if end > n { break; }
        let mut ch = f64::NEG_INFINITY;
        let mut cl = f64::INFINITY;
        for i in start..end {
            if highs[i] > ch { ch = highs[i]; }
            if lows[i] < cl { cl = lows[i]; }
        }
        candles.push((start, closes[start], ch, cl, closes[end - 1]));
    }

    // Classify + attach features
    let mut events: Vec<SweepEvent> = Vec::new();
    for i in 1..candles.len() {
        let (_, _, prev_high, prev_low, _) = candles[i - 1];
        let (curr_start, curr_open, curr_high, curr_low, curr_close) = candles[i];

        let feat_idx = curr_start + feat_bar_offset.min(bp - 1);
        if feat_idx >= n || features[feat_idx][0].is_nan() { continue; }

        let swept_high = curr_high > prev_high;
        let swept_low = curr_low < prev_low;
        let is_green = curr_close > curr_open;
        let is_red = curr_close < curr_open;

        let signal = if swept_high && !swept_low {
            if curr_close > prev_high { Signal::HighCont }
            else if is_red { Signal::HighRev }
            else { Signal::HighAmbig }
        } else if swept_low && !swept_high {
            if curr_close < prev_low { Signal::LowCont }
            else if is_green { Signal::LowRev }
            else { Signal::LowAmbig }
        } else if swept_high && swept_low {
            if curr_close > prev_high { Signal::HighCont }
            else if curr_close < prev_low { Signal::LowCont }
            else if is_red { Signal::HighRev }
            else if is_green { Signal::LowRev }
            else { Signal::InsideBar }
        } else {
            Signal::InsideBar
        };

        if signal == Signal::InsideBar || signal == Signal::HighAmbig || signal == Signal::LowAmbig {
            continue;
        }

        events.push(SweepEvent {
            bar_idx: curr_start,
            signal,
            features: features[feat_idx],
        });
    }

    // Split by sweep type
    let high_events: Vec<&SweepEvent> = events.iter().filter(|e| {
        matches!(e.signal, Signal::HighCont | Signal::HighRev)
    }).collect();

    let low_events: Vec<&SweepEvent> = events.iter().filter(|e| {
        matches!(e.signal, Signal::LowCont | Signal::LowRev)
    }).collect();

    let (hc_base, hc_total, hc_qrows, hc_patterns) = mine_one_side(
        &high_events, &|s| matches!(s, Signal::HighCont), "high", "continuation");
    let (_, _, _, hr_patterns) = mine_one_side(
        &high_events, &|s| matches!(s, Signal::HighRev), "high", "reversal");
    let (lc_base, lc_total, lc_qrows, lc_patterns) = mine_one_side(
        &low_events, &|s| matches!(s, Signal::LowCont), "low", "continuation");
    let (_, _, _, lr_patterns) = mine_one_side(
        &low_events, &|s| matches!(s, Signal::LowRev), "low", "reversal");

    let mut all_qrows = hc_qrows;
    all_qrows.extend(lc_qrows);

    MinerResultV2 {
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

/// Rust-native TPE Optimizer with Rayon parallelism.
///
/// Histogram-based TPE:
/// 1. Random exploration phase (n_startup trials)
/// 2. Split results into good/bad by quantile
/// 3. Build histogram per parameter from good group
/// 4. Sample from good histogram, evaluate in parallel
/// 5. Repeat until n_trials done
///
/// 2-stage: exploration → exploitation with local refinement

use rayon::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::kama::{run_combined_backtest, KamaBacktestResult};
use crate::adaptive_pmax::adaptive_pmax_continuous;

// ── Parameter bounds ──

#[derive(Clone, Debug)]
pub enum ParamType {
    IntRange { min: i32, max: i32 },
    FloatRange { min: f64, max: f64, step: f64 },
}

#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub name: String,
    pub ptype: ParamType,
}

impl ParamSpec {
    fn sample_random(&self, rng: &mut ChaCha8Rng) -> f64 {
        match &self.ptype {
            ParamType::IntRange { min, max } => {
                rng.gen_range(*min..=*max) as f64
            }
            ParamType::FloatRange { min, max, step } => {
                let steps = ((*max - *min) / *step).round() as i32;
                let s = rng.gen_range(0..=steps);
                *min + s as f64 * *step
            }
        }
    }

    fn sample_tpe(&self, good_vals: &[f64], rng: &mut ChaCha8Rng) -> f64 {
        if good_vals.is_empty() {
            return self.sample_random(rng);
        }

        match &self.ptype {
            ParamType::IntRange { min, max } => {
                // Histogram-based: count good vals in bins, sample proportionally
                let range = (*max - *min + 1) as usize;
                let mut counts = vec![1u32; range]; // Laplace smoothing
                for &v in good_vals {
                    let idx = (v.round() as i32 - *min) as usize;
                    if idx < range {
                        counts[idx] += 10; // Weight good observations
                    }
                }
                let total: u32 = counts.iter().sum();
                let r = rng.gen_range(0..total);
                let mut cum = 0u32;
                for (i, &c) in counts.iter().enumerate() {
                    cum += c;
                    if r < cum {
                        return (*min + i as i32) as f64;
                    }
                }
                *max as f64
            }
            ParamType::FloatRange { min, max, step } => {
                let steps = ((*max - *min) / *step).round() as usize;
                let mut counts = vec![1u32; steps + 1];
                for &v in good_vals {
                    let idx = ((v - *min) / *step).round() as usize;
                    if idx <= steps {
                        counts[idx] += 10;
                    }
                }
                let total: u32 = counts.iter().sum();
                let r = rng.gen_range(0..total);
                let mut cum = 0u32;
                for (i, &c) in counts.iter().enumerate() {
                    cum += c;
                    if r < cum {
                        return *min + i as f64 * *step;
                    }
                }
                *max
            }
        }
    }

    fn perturb(&self, val: f64, scale: f64, rng: &mut ChaCha8Rng) -> f64 {
        match &self.ptype {
            ParamType::IntRange { min, max } => {
                let range = (*max - *min) as f64 * scale;
                let delta = rng.gen_range(-range..=range);
                let new_val = (val + delta).round().clamp(*min as f64, *max as f64);
                new_val
            }
            ParamType::FloatRange { min, max, step } => {
                let range = (*max - *min) * scale;
                let delta = rng.gen_range(-range..=range);
                let raw = (val + delta).clamp(*min, *max);
                // Snap to step
                let steps = ((raw - *min) / *step).round();
                (*min + steps * *step).clamp(*min, *max)
            }
        }
    }
}

// ── Trial result ──

#[derive(Clone, Debug)]
pub struct TrialResult {
    pub params: Vec<f64>,
    pub score: f64,
    pub net_pct: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub total_trades: i32,
    pub tp_count: i32,
    pub rev_count: i32,
    pub total_fees: f64,
}

// ── Score function ──

fn compute_score(r: &KamaBacktestResult, min_trades: i32) -> f64 {
    if r.total_trades < min_trades { return -999.0; }
    if r.net_pct < 0.0 { return -999.0; }
    let score = (r.net_pct / r.max_drawdown.max(0.5)) * r.win_rate * 0.01;
    if r.net_pct > 500.0 { score * 0.8 } else { score }
}

// ── Main optimizer ──

pub struct CombinedOptimizer {
    pub param_specs: Vec<ParamSpec>,
    pub n_trials: usize,
    pub n_startup: usize,
    pub gamma: f64, // quantile for good/bad split (0.25 = top 25%)
}

impl CombinedOptimizer {
    pub fn new(n_trials: usize) -> Self {
        let param_specs = build_param_specs();
        Self {
            param_specs,
            n_trials,
            n_startup: (n_trials / 5).max(50),
            gamma: 0.25,
        }
    }

    /// Run optimization for one fold.
    /// Returns top-K results sorted by score descending.
    pub fn optimize_fold(
        &self,
        // Train data
        tr_closes: &[f64], tr_highs: &[f64], tr_lows: &[f64],
        tr_buy_vol: &[f64], tr_sell_vol: &[f64], tr_oi: &[f64],
        // Test data
        te_closes: &[f64], te_highs: &[f64], te_lows: &[f64],
        te_buy_vol: &[f64], te_sell_vol: &[f64], te_oi: &[f64],
        // KAMA fixed
        kama_period: usize, kama_fast: usize, kama_slow: usize,
        slope_lookback: usize, slope_threshold: f64,
        // Warm-start params from previous fold
        warm_start: &[Vec<f64>],
        seed: u64,
    ) -> Vec<TrialResult> {

        let n_params = self.param_specs.len();
        let mut all_results: Vec<TrialResult> = Vec::with_capacity(self.n_trials);

        // ── Phase 1: Random exploration + warm-start ──
        let n_random = self.n_startup;
        let mut random_params: Vec<Vec<f64>> = Vec::with_capacity(n_random);

        // Add warm-start params first
        for ws in warm_start {
            if ws.len() == n_params {
                random_params.push(ws.clone());
            }
        }

        // Fill rest with random
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        while random_params.len() < n_random {
            let p: Vec<f64> = self.param_specs.iter()
                .map(|s| s.sample_random(&mut rng))
                .collect();
            random_params.push(p);
        }

        // Evaluate random phase in parallel
        let random_results: Vec<TrialResult> = random_params.par_iter()
            .map(|params| {
                evaluate_trial(
                    params, &self.param_specs,
                    tr_closes, tr_highs, tr_lows, tr_buy_vol, tr_sell_vol, tr_oi,
                    te_closes, te_highs, te_lows, te_buy_vol, te_sell_vol, te_oi,
                    kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
                )
            })
            .collect();

        all_results.extend(random_results);

        // ── Phase 2: TPE-guided search ──
        let n_tpe = self.n_trials - n_random;
        let batch_size = 64; // Evaluate 64 at a time in parallel

        for batch_start in (0..n_tpe).step_by(batch_size) {
            let actual_batch = batch_size.min(n_tpe - batch_start);

            // Split into good/bad
            let mut valid: Vec<&TrialResult> = all_results.iter()
                .filter(|r| r.score > -999.0)
                .collect();
            valid.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

            let n_good = (valid.len() as f64 * self.gamma).ceil() as usize;

            // Generate TPE-guided params
            let tpe_params: Vec<Vec<f64>> = if n_good >= 5 {
                let good_results = &valid[..n_good];
                (0..actual_batch).map(|j| {
                    let mut trial_rng = ChaCha8Rng::seed_from_u64(seed + 1000 + batch_start as u64 + j as u64);
                    self.param_specs.iter().enumerate().map(|(pi, spec)| {
                        let good_vals: Vec<f64> = good_results.iter().map(|r| r.params[pi]).collect();
                        spec.sample_tpe(&good_vals, &mut trial_rng)
                    }).collect()
                }).collect()
            } else {
                // Not enough good results, keep random
                (0..actual_batch).map(|j| {
                    let mut trial_rng = ChaCha8Rng::seed_from_u64(seed + 2000 + batch_start as u64 + j as u64);
                    self.param_specs.iter().map(|s| s.sample_random(&mut trial_rng)).collect()
                }).collect()
            };

            // Evaluate batch in parallel
            let batch_results: Vec<TrialResult> = tpe_params.par_iter()
                .map(|params| {
                    evaluate_trial(
                        params, &self.param_specs,
                        tr_closes, tr_highs, tr_lows, tr_buy_vol, tr_sell_vol, tr_oi,
                        te_closes, te_highs, te_lows, te_buy_vol, te_sell_vol, te_oi,
                        kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
                    )
                })
                .collect();

            all_results.extend(batch_results);
        }

        // ── Phase 3: Local refinement around best ──
        let n_refine = self.n_trials / 5; // 20% of budget for refinement
        let mut valid: Vec<&TrialResult> = all_results.iter()
            .filter(|r| r.score > -999.0)
            .collect();
        valid.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        if valid.len() >= 3 {
            let top_k = valid.len().min(10);
            let top_params: Vec<Vec<f64>> = valid[..top_k].iter().map(|r| r.params.clone()).collect();

            let refine_params: Vec<Vec<f64>> = (0..n_refine).map(|j| {
                let mut trial_rng = ChaCha8Rng::seed_from_u64(seed + 5000 + j as u64);
                let base = &top_params[j % top_k];
                self.param_specs.iter().enumerate().map(|(pi, spec)| {
                    spec.perturb(base[pi], 0.15, &mut trial_rng)
                }).collect()
            }).collect();

            let refine_results: Vec<TrialResult> = refine_params.par_iter()
                .map(|params| {
                    evaluate_trial(
                        params, &self.param_specs,
                        tr_closes, tr_highs, tr_lows, tr_buy_vol, tr_sell_vol, tr_oi,
                        te_closes, te_highs, te_lows, te_buy_vol, te_sell_vol, te_oi,
                        kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
                    )
                })
                .collect();

            all_results.extend(refine_results);
        }

        // Sort by score, return top results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(20);
        all_results
    }
}

// ── Evaluate a single trial ──

fn evaluate_trial(
    params: &[f64],
    specs: &[ParamSpec],
    tr_closes: &[f64], tr_highs: &[f64], tr_lows: &[f64],
    tr_buy_vol: &[f64], tr_sell_vol: &[f64], tr_oi: &[f64],
    te_closes: &[f64], te_highs: &[f64], te_lows: &[f64],
    te_buy_vol: &[f64], te_sell_vol: &[f64], te_oi: &[f64],
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
) -> TrialResult {
    let fail = TrialResult {
        params: params.to_vec(), score: -999.0,
        net_pct: 0.0, max_drawdown: 0.0, win_rate: 0.0,
        total_trades: 0, tp_count: 0, rev_count: 0, total_fees: 0.0,
    };

    // Extract params
    let p = extract_params(params);

    // ── Train: compute PMax + backtest ──
    let tr_pmax = std::panic::catch_unwind(|| {
        adaptive_pmax_continuous(
            tr_closes, tr_highs, tr_lows, tr_closes,
            p.pmax_atr_period, p.pmax_atr_mult, p.pmax_ma_length,
            p.pmax_lookback, p.pmax_flip_window,
            p.pmax_mult_base, p.pmax_mult_scale,
            p.pmax_ma_base, p.pmax_ma_scale,
            p.pmax_atr_base, p.pmax_atr_scale,
            p.pmax_update_interval,
        )
    });
    let tr_pmax = match tr_pmax {
        Ok(r) => r,
        Err(_) => return fail,
    };

    let tr_r = std::panic::catch_unwind(|| {
        run_combined_backtest(
            tr_closes, tr_highs, tr_lows, tr_buy_vol, tr_sell_vol, tr_oi,
            &tr_pmax.pmax_line, &tr_pmax.mavg,
            kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
            p.cvd_period, p.imb_weight, p.cvd_threshold, p.oi_period, p.oi_threshold,
            p.kc_length, p.kc_multiplier, p.kc_atr_period,
            p.max_dca_steps,
            p.dca_m1, p.dca_m2, p.dca_m3, p.dca_m4,
            p.tp1, p.tp2, p.tp3, p.tp4,
        )
    });
    let tr_r = match tr_r {
        Ok(r) => r,
        Err(_) => return fail,
    };

    // Train filter
    if tr_r.total_trades < 5 { return fail; }

    // ── Test: compute PMax + backtest ──
    let te_pmax = std::panic::catch_unwind(|| {
        adaptive_pmax_continuous(
            te_closes, te_highs, te_lows, te_closes,
            p.pmax_atr_period, p.pmax_atr_mult, p.pmax_ma_length,
            p.pmax_lookback, p.pmax_flip_window,
            p.pmax_mult_base, p.pmax_mult_scale,
            p.pmax_ma_base, p.pmax_ma_scale,
            p.pmax_atr_base, p.pmax_atr_scale,
            p.pmax_update_interval,
        )
    });
    let te_pmax = match te_pmax {
        Ok(r) => r,
        Err(_) => return fail,
    };

    let te_r = std::panic::catch_unwind(|| {
        run_combined_backtest(
            te_closes, te_highs, te_lows, te_buy_vol, te_sell_vol, te_oi,
            &te_pmax.pmax_line, &te_pmax.mavg,
            kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
            p.cvd_period, p.imb_weight, p.cvd_threshold, p.oi_period, p.oi_threshold,
            p.kc_length, p.kc_multiplier, p.kc_atr_period,
            p.max_dca_steps,
            p.dca_m1, p.dca_m2, p.dca_m3, p.dca_m4,
            p.tp1, p.tp2, p.tp3, p.tp4,
        )
    });
    let te_r = match te_r {
        Ok(r) => r,
        Err(_) => return fail,
    };

    let score = compute_score(&te_r, 3);

    TrialResult {
        params: params.to_vec(),
        score,
        net_pct: te_r.net_pct,
        max_drawdown: te_r.max_drawdown,
        win_rate: te_r.win_rate,
        total_trades: te_r.total_trades,
        tp_count: te_r.tp_count,
        rev_count: te_r.rev_count,
        total_fees: te_r.total_fees,
    }
}

// ── Extracted params struct ──

struct ExtractedParams {
    pmax_atr_period: usize,
    pmax_atr_mult: f64,
    pmax_ma_length: usize,
    pmax_lookback: usize,
    pmax_flip_window: usize,
    pmax_mult_base: f64,
    pmax_mult_scale: f64,
    pmax_ma_base: usize,
    pmax_ma_scale: f64,
    pmax_atr_base: usize,
    pmax_atr_scale: f64,
    pmax_update_interval: usize,
    cvd_period: usize,
    imb_weight: f64,
    cvd_threshold: f64,
    oi_period: usize,
    oi_threshold: f64,
    kc_length: usize,
    kc_multiplier: f64,
    kc_atr_period: usize,
    max_dca_steps: i32,
    dca_m1: f64,
    dca_m2: f64,
    dca_m3: f64,
    dca_m4: f64,
    tp1: f64,
    tp2: f64,
    tp3: f64,
    tp4: f64,
}

fn extract_params(p: &[f64]) -> ExtractedParams {
    ExtractedParams {
        pmax_atr_period: p[0] as usize,
        pmax_atr_mult: p[1],
        pmax_ma_length: p[2] as usize,
        pmax_lookback: p[3] as usize,
        pmax_flip_window: p[4] as usize,
        pmax_mult_base: p[5],
        pmax_mult_scale: p[6],
        pmax_ma_base: p[7] as usize,
        pmax_ma_scale: p[8],
        pmax_atr_base: p[9] as usize,
        pmax_atr_scale: p[10],
        pmax_update_interval: p[11] as usize,
        cvd_period: p[12] as usize,
        imb_weight: p[13],
        cvd_threshold: p[14],
        oi_period: p[15] as usize,
        oi_threshold: p[16],
        kc_length: p[17] as usize,
        kc_multiplier: p[18],
        kc_atr_period: p[19] as usize,
        max_dca_steps: p[20] as i32,
        dca_m1: p[21],
        dca_m2: p[22],
        dca_m3: p[23],
        dca_m4: p[24],
        tp1: p[25],
        tp2: p[26],
        tp3: p[27],
        tp4: p[28],
    }
}

// ── Parameter specs (must match Python order) ──

pub fn build_param_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec { name: "pmax_atr_period".into(), ptype: ParamType::IntRange { min: 8, max: 24 } },
        ParamSpec { name: "pmax_atr_mult".into(), ptype: ParamType::FloatRange { min: 1.0, max: 4.0, step: 0.1 } },
        ParamSpec { name: "pmax_ma_length".into(), ptype: ParamType::IntRange { min: 5, max: 20 } },
        ParamSpec { name: "pmax_lookback".into(), ptype: ParamType::IntRange { min: 20, max: 100 } },
        ParamSpec { name: "pmax_flip_window".into(), ptype: ParamType::IntRange { min: 10, max: 50 } },
        ParamSpec { name: "pmax_mult_base".into(), ptype: ParamType::FloatRange { min: 0.5, max: 3.0, step: 0.1 } },
        ParamSpec { name: "pmax_mult_scale".into(), ptype: ParamType::FloatRange { min: 0.1, max: 1.5, step: 0.1 } },
        ParamSpec { name: "pmax_ma_base".into(), ptype: ParamType::IntRange { min: 5, max: 12 } },
        ParamSpec { name: "pmax_ma_scale".into(), ptype: ParamType::FloatRange { min: 0.1, max: 0.5, step: 0.1 } },
        ParamSpec { name: "pmax_atr_base".into(), ptype: ParamType::IntRange { min: 5, max: 15 } },
        ParamSpec { name: "pmax_atr_scale".into(), ptype: ParamType::FloatRange { min: 0.1, max: 1.0, step: 0.1 } },
        ParamSpec { name: "pmax_update_interval".into(), ptype: ParamType::IntRange { min: 1, max: 10 } },
        ParamSpec { name: "cvd_period".into(), ptype: ParamType::IntRange { min: 5, max: 100 } },
        ParamSpec { name: "imb_weight".into(), ptype: ParamType::FloatRange { min: 0.0, max: 1.0, step: 0.05 } },
        ParamSpec { name: "cvd_threshold".into(), ptype: ParamType::FloatRange { min: 0.001, max: 0.1, step: 0.001 } },
        ParamSpec { name: "oi_period".into(), ptype: ParamType::IntRange { min: 5, max: 100 } },
        ParamSpec { name: "oi_threshold".into(), ptype: ParamType::FloatRange { min: 0.0, max: 0.05, step: 0.001 } },
        ParamSpec { name: "kc_length".into(), ptype: ParamType::IntRange { min: 2, max: 50 } },
        ParamSpec { name: "kc_multiplier".into(), ptype: ParamType::FloatRange { min: 0.5, max: 5.0, step: 0.1 } },
        ParamSpec { name: "kc_atr_period".into(), ptype: ParamType::IntRange { min: 2, max: 50 } },
        ParamSpec { name: "max_dca".into(), ptype: ParamType::IntRange { min: 1, max: 4 } },
        ParamSpec { name: "dca_m1".into(), ptype: ParamType::FloatRange { min: 0.3, max: 2.0, step: 0.1 } },
        ParamSpec { name: "dca_m2".into(), ptype: ParamType::FloatRange { min: 0.2, max: 1.5, step: 0.1 } },
        ParamSpec { name: "dca_m3".into(), ptype: ParamType::FloatRange { min: 0.1, max: 1.2, step: 0.1 } },
        ParamSpec { name: "dca_m4".into(), ptype: ParamType::FloatRange { min: 0.1, max: 1.0, step: 0.1 } },
        ParamSpec { name: "tp1".into(), ptype: ParamType::FloatRange { min: 0.05, max: 0.8, step: 0.05 } },
        ParamSpec { name: "tp2".into(), ptype: ParamType::FloatRange { min: 0.05, max: 0.8, step: 0.05 } },
        ParamSpec { name: "tp3".into(), ptype: ParamType::FloatRange { min: 0.05, max: 0.8, step: 0.05 } },
        ParamSpec { name: "tp4".into(), ptype: ParamType::FloatRange { min: 0.05, max: 0.8, step: 0.05 } },
    ]
}

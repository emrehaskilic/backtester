/// Saf PMax Backtest — sadece adaptive PMax crossover, reversal ile çıkış.
/// KC/DCA/TP/CVD/OI yok. Sürekli işlemde.

use crate::adaptive_pmax::adaptive_pmax_continuous;

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const TAKER_FEE: f64 = 0.0005;
const MIN_BARS: usize = 200;

pub struct PmaxResult {
    pub net_pct: f64,
    pub balance: f64,
    pub total_trades: i32,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
}

pub fn run_pure_pmax_backtest(
    closes: &[f64], highs: &[f64], lows: &[f64],
    pmax_line: &[f64], mavg_arr: &[f64],
) -> PmaxResult {
    let n = closes.len();

    let mut cond: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut bal = INITIAL_BALANCE;
    let mut peak = INITIAL_BALANCE;
    let mut mdd: f64 = 0.0;
    let mut t_pnl: f64 = 0.0;
    let mut t_fee: f64 = 0.0;
    let mut t_cnt: i32 = 0;
    let mut w_cnt: i32 = 0;

    for i in MIN_BARS..n {
        if mavg_arr[i].is_nan() || pmax_line[i].is_nan() { continue; }
        if i == 0 || mavg_arr[i-1].is_nan() || pmax_line[i-1].is_nan() { continue; }

        let buy_cross = mavg_arr[i-1] <= pmax_line[i-1] && mavg_arr[i] > pmax_line[i];
        let sell_cross = mavg_arr[i-1] >= pmax_line[i-1] && mavg_arr[i] < pmax_line[i];

        if buy_cross && cond <= 0.0 {
            // Mevcut SHORT kapat
            if cond < 0.0 && notional > 0.0 {
                let pp = (entry_price - closes[i]) / entry_price * 100.0;
                let pnl = notional * pp / 100.0;
                let fee = notional * TAKER_FEE;
                bal += pnl - fee; t_pnl += pnl; t_fee += fee; t_cnt += 1;
                if pnl > 0.0 { w_cnt += 1; }
            }
            // LONG aç
            let mg = bal * MARGIN_RATIO;
            if bal >= mg && mg > 0.0 {
                cond = 1.0; entry_price = closes[i]; notional = mg * LEVERAGE;
                let fee = notional * TAKER_FEE; bal -= fee; t_fee += fee;
            } else { cond = 0.0; notional = 0.0; }
        } else if sell_cross && cond >= 0.0 {
            // Mevcut LONG kapat
            if cond > 0.0 && notional > 0.0 {
                let pp = (closes[i] - entry_price) / entry_price * 100.0;
                let pnl = notional * pp / 100.0;
                let fee = notional * TAKER_FEE;
                bal += pnl - fee; t_pnl += pnl; t_fee += fee; t_cnt += 1;
                if pnl > 0.0 { w_cnt += 1; }
            }
            // SHORT aç
            let mg = bal * MARGIN_RATIO;
            if bal >= mg && mg > 0.0 {
                cond = -1.0; entry_price = closes[i]; notional = mg * LEVERAGE;
                let fee = notional * TAKER_FEE; bal -= fee; t_fee += fee;
            } else { cond = 0.0; notional = 0.0; }
        }

        if bal > peak { peak = bal; }
        if peak > 0.0 { let dd = (peak - bal) / peak * 100.0; if dd > mdd { mdd = dd; } }
    }

    // Kalan pozisyonu kapat
    if cond != 0.0 && notional > 0.0 && n > 0 {
        let pp = if cond > 0.0 { (closes[n-1] - entry_price) / entry_price * 100.0 }
                 else { (entry_price - closes[n-1]) / entry_price * 100.0 };
        let pnl = notional * pp / 100.0;
        let fee = notional * TAKER_FEE;
        bal += pnl - fee; t_pnl += pnl; t_fee += fee; t_cnt += 1;
        if pnl > 0.0 { w_cnt += 1; }
    }
    if bal > peak { peak = bal; }
    if peak > 0.0 { let dd = (peak - bal) / peak * 100.0; if dd > mdd { mdd = dd; } }

    let np = (bal - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let wr = if t_cnt > 0 { w_cnt as f64 / t_cnt as f64 * 100.0 } else { 0.0 };
    PmaxResult { net_pct: np, balance: bal, total_trades: t_cnt, win_rate: wr,
        max_drawdown: mdd, total_pnl: t_pnl, total_fees: t_fee }
}

// ── Optimizer: 12 PMax param ──

use rayon::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
pub struct PmaxTrialResult {
    pub params: Vec<f64>,
    pub score: f64,
    pub net_pct: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub total_trades: i32,
    pub total_fees: f64,
}

struct PmaxParams {
    atr_period: usize,
    atr_mult: f64,
    ma_length: usize,
    lookback: usize,
    flip_window: usize,
    mult_base: f64,
    mult_scale: f64,
    ma_base: usize,
    ma_scale: f64,
    atr_base: usize,
    atr_scale: f64,
    update_interval: usize,
}

fn extract(p: &[f64]) -> PmaxParams {
    PmaxParams {
        atr_period: p[0] as usize,
        atr_mult: p[1],
        ma_length: p[2] as usize,
        lookback: p[3] as usize,
        flip_window: p[4] as usize,
        mult_base: p[5],
        mult_scale: p[6],
        ma_base: p[7] as usize,
        ma_scale: p[8],
        atr_base: p[9] as usize,
        atr_scale: p[10],
        update_interval: p[11] as usize,
    }
}

// Parametre araliklari: [min, max, step(0=int)]
const BOUNDS: [(f64, f64, f64); 12] = [
    (8.0, 24.0, 0.0),    // atr_period (int)
    (1.0, 4.0, 0.1),     // atr_mult
    (5.0, 20.0, 0.0),    // ma_length (int)
    (20.0, 100.0, 0.0),  // lookback (int)
    (10.0, 50.0, 0.0),   // flip_window (int)
    (0.5, 3.0, 0.1),     // mult_base
    (0.1, 1.5, 0.1),     // mult_scale
    (5.0, 12.0, 0.0),    // ma_base (int)
    (0.1, 0.5, 0.1),     // ma_scale
    (5.0, 15.0, 0.0),    // atr_base (int)
    (0.1, 1.0, 0.1),     // atr_scale
    (1.0, 10.0, 0.0),    // update_interval (int)
];

fn sample_random(rng: &mut ChaCha8Rng) -> Vec<f64> {
    BOUNDS.iter().map(|&(min, max, step)| {
        if step == 0.0 {
            rng.gen_range(min as i32..=max as i32) as f64
        } else {
            let steps = ((max - min) / step).round() as i32;
            min + rng.gen_range(0..=steps) as f64 * step
        }
    }).collect()
}

fn sample_tpe(good: &[Vec<f64>], rng: &mut ChaCha8Rng) -> Vec<f64> {
    BOUNDS.iter().enumerate().map(|(pi, &(min, max, step))| {
        let good_vals: Vec<f64> = good.iter().map(|p| p[pi]).collect();
        if step == 0.0 {
            // Int histogram
            let range = (max - min + 1.0) as usize;
            let mut counts = vec![1u32; range];
            for &v in &good_vals { let idx = (v - min) as usize; if idx < range { counts[idx] += 10; } }
            let total: u32 = counts.iter().sum();
            let r = rng.gen_range(0..total);
            let mut cum = 0u32;
            for (i, &c) in counts.iter().enumerate() { cum += c; if r < cum { return min + i as f64; } }
            max
        } else {
            let steps = ((max - min) / step).round() as usize;
            let mut counts = vec![1u32; steps + 1];
            for &v in &good_vals { let idx = ((v - min) / step).round() as usize; if idx <= steps { counts[idx] += 10; } }
            let total: u32 = counts.iter().sum();
            let r = rng.gen_range(0..total);
            let mut cum = 0u32;
            for (i, &c) in counts.iter().enumerate() { cum += c; if r < cum { return min + i as f64 * step; } }
            max
        }
    }).collect()
}

fn perturb(base: &[f64], scale: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
    BOUNDS.iter().enumerate().map(|(pi, &(min, max, step))| {
        let range = (max - min) * scale;
        let delta = rng.gen_range(-range..=range);
        let raw = (base[pi] + delta).clamp(min, max);
        if step == 0.0 { raw.round() } else { (((raw - min) / step).round() * step + min).clamp(min, max) }
    }).collect()
}

fn eval_trial(
    params: &[f64],
    tr_c: &[f64], tr_h: &[f64], tr_l: &[f64],
    te_c: &[f64], te_h: &[f64], te_l: &[f64],
) -> PmaxTrialResult {
    let fail = PmaxTrialResult {
        params: params.to_vec(), score: -999.0,
        net_pct: 0.0, max_drawdown: 0.0, win_rate: 0.0, total_trades: 0, total_fees: 0.0,
    };

    let p = extract(params);

    // Train PMax
    let tr_pm = std::panic::catch_unwind(|| {
        adaptive_pmax_continuous(tr_c, tr_h, tr_l, tr_c,
            p.atr_period, p.atr_mult, p.ma_length, p.lookback, p.flip_window,
            p.mult_base, p.mult_scale, p.ma_base, p.ma_scale,
            p.atr_base, p.atr_scale, p.update_interval)
    });
    let tr_pm = match tr_pm { Ok(r) => r, Err(_) => return fail };

    let tr_r = run_pure_pmax_backtest(tr_c, tr_h, tr_l, &tr_pm.pmax_line, &tr_pm.mavg);
    if tr_r.total_trades < 5 { return fail; }

    // Test PMax
    let te_pm = std::panic::catch_unwind(|| {
        adaptive_pmax_continuous(te_c, te_h, te_l, te_c,
            p.atr_period, p.atr_mult, p.ma_length, p.lookback, p.flip_window,
            p.mult_base, p.mult_scale, p.ma_base, p.ma_scale,
            p.atr_base, p.atr_scale, p.update_interval)
    });
    let te_pm = match te_pm { Ok(r) => r, Err(_) => return fail };

    let te_r = run_pure_pmax_backtest(te_c, te_h, te_l, &te_pm.pmax_line, &te_pm.mavg);
    if te_r.total_trades < 3 || te_r.net_pct < 0.0 { return fail; }

    let score = (te_r.net_pct / te_r.max_drawdown.max(0.5)) * te_r.win_rate * 0.01;
    let score = if te_r.net_pct > 500.0 { score * 0.8 } else { score };

    PmaxTrialResult {
        params: params.to_vec(), score,
        net_pct: te_r.net_pct, max_drawdown: te_r.max_drawdown,
        win_rate: te_r.win_rate, total_trades: te_r.total_trades, total_fees: te_r.total_fees,
    }
}

pub fn optimize_pmax_fold(
    tr_c: &[f64], tr_h: &[f64], tr_l: &[f64],
    te_c: &[f64], te_h: &[f64], te_l: &[f64],
    n_trials: usize, seed: u64,
    warm_start: &[Vec<f64>],
) -> Vec<PmaxTrialResult> {
    let n_startup = (n_trials / 5).max(50);
    let gamma = 0.25;

    let mut all: Vec<PmaxTrialResult> = Vec::with_capacity(n_trials * 2);

    // Phase 1: Random + warm-start
    let mut random_params: Vec<Vec<f64>> = Vec::with_capacity(n_startup);
    for ws in warm_start { if ws.len() == 12 { random_params.push(ws.clone()); } }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    while random_params.len() < n_startup { random_params.push(sample_random(&mut rng)); }

    let results: Vec<PmaxTrialResult> = random_params.par_iter()
        .map(|p| eval_trial(p, tr_c, tr_h, tr_l, te_c, te_h, te_l))
        .collect();
    all.extend(results);

    // Phase 2: TPE
    let n_tpe = n_trials - n_startup;
    let batch = 128;
    for bs in (0..n_tpe).step_by(batch) {
        let actual = batch.min(n_tpe - bs);
        let mut valid: Vec<&PmaxTrialResult> = all.iter().filter(|r| r.score > -999.0).collect();
        valid.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let n_good = (valid.len() as f64 * gamma).ceil() as usize;

        let params: Vec<Vec<f64>> = if n_good >= 5 {
            let good: Vec<Vec<f64>> = valid[..n_good].iter().map(|r| r.params.clone()).collect();
            (0..actual).map(|j| {
                let mut r = ChaCha8Rng::seed_from_u64(seed + 1000 + bs as u64 + j as u64);
                sample_tpe(&good, &mut r)
            }).collect()
        } else {
            (0..actual).map(|j| {
                let mut r = ChaCha8Rng::seed_from_u64(seed + 2000 + bs as u64 + j as u64);
                sample_random(&mut r)
            }).collect()
        };

        let results: Vec<PmaxTrialResult> = params.par_iter()
            .map(|p| eval_trial(p, tr_c, tr_h, tr_l, te_c, te_h, te_l))
            .collect();
        all.extend(results);
    }

    // Phase 3: Refinement
    let n_refine = n_trials / 5;
    let mut valid: Vec<&PmaxTrialResult> = all.iter().filter(|r| r.score > -999.0).collect();
    valid.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if valid.len() >= 3 {
        let top_k = valid.len().min(10);
        let tops: Vec<Vec<f64>> = valid[..top_k].iter().map(|r| r.params.clone()).collect();

        let refine: Vec<Vec<f64>> = (0..n_refine).map(|j| {
            let mut r = ChaCha8Rng::seed_from_u64(seed + 5000 + j as u64);
            perturb(&tops[j % top_k], 0.15, &mut r)
        }).collect();

        let results: Vec<PmaxTrialResult> = refine.par_iter()
            .map(|p| eval_trial(p, tr_c, tr_h, tr_l, te_c, te_h, te_l))
            .collect();
        all.extend(results);
    }

    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(20);
    all
}

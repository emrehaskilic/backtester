/// Bias Engine — Combined Scoring (MR + RSI + CVD + Bias Engine)
///
/// Produces continuous direction signal by combining:
/// 1. Bias engine state bias (from validated states)
/// 2. Mean reversion (EMA distance)
/// 3. RSI (oversold/overbought)
/// 4. CVD momentum
/// 5. Bias + MR agreement bonus
///
/// Output: per-bar score (positive=bullish, negative=bearish), direction, confidence

use super::params::GroupBParams;

/// Per-bar combined score output
#[derive(Clone, Debug)]
pub struct ScoredBar {
    pub score: f64,
    pub direction: i8,  // +1, -1, 0
}

/// Compute EMA of a series
pub fn compute_ema(data: &[f64], span: usize) -> Vec<f64> {
    let n = data.len();
    let k = 2.0 / (span as f64 + 1.0);
    let mut out = vec![0.0f64; n];
    out[0] = data[0];
    for i in 1..n {
        out[i] = data[i] * k + out[i - 1] * (1.0 - k);
    }
    out
}

/// Compute RSI
pub fn compute_rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut rsi = vec![50.0f64; n];
    if n < 2 || period == 0 {
        return rsi;
    }

    let mut gains = vec![0.0f64; n];
    let mut losses = vec![0.0f64; n];
    for i in 1..n {
        let d = close[i] - close[i - 1];
        if d > 0.0 {
            gains[i] = d;
        } else {
            losses[i] = -d;
        }
    }

    let avg_g = compute_ema(&gains, period);
    let avg_l = compute_ema(&losses, period);

    for i in period..n {
        if avg_l[i] > 1e-15 {
            let rs = avg_g[i] / avg_l[i];
            rsi[i] = 100.0 - 100.0 / (1.0 + rs);
        }
    }
    rsi
}

/// Compute BTC momentum z-score: rolling z-score of BTC bar returns.
/// Positive = BTC trending up, negative = BTC trending down.
pub fn compute_btc_momentum(btc_close: &[f64], window: usize) -> Vec<f64> {
    let n = btc_close.len();
    let mut returns = vec![0.0f64; n];
    for i in 1..n {
        if btc_close[i - 1] > 1e-10 {
            returns[i] = (btc_close[i] - btc_close[i - 1]) / btc_close[i - 1];
        }
    }
    // Rolling z-score of returns
    let ema_ret = compute_ema(&returns, window);
    let mut std_arr = vec![0.0f64; n];
    for i in window..n {
        let mut sum_sq = 0.0;
        for j in (i + 1 - window)..=i {
            let diff = returns[j] - ema_ret[j];
            sum_sq += diff * diff;
        }
        std_arr[i] = (sum_sq / window as f64).sqrt();
    }
    let mut z = vec![0.0f64; n];
    for i in window..n {
        if std_arr[i] > 1e-15 {
            z[i] = (returns[i] - ema_ret[i]) / std_arr[i];
        }
    }
    z
}

/// Compute BTC lead signal: cumulative BTC return - cumulative ETH return over a window.
/// Positive = BTC outperformed ETH recently → ETH likely to catch up (bullish ETH).
/// Negative = BTC underperformed ETH → ETH may follow down (bearish ETH).
pub fn compute_btc_lead(btc_close: &[f64], eth_close: &[f64], window: usize) -> Vec<f64> {
    let n = btc_close.len().min(eth_close.len());
    let mut lead = vec![0.0f64; n];
    for i in window..n {
        let btc_ret = if btc_close[i - window] > 1e-10 {
            (btc_close[i] - btc_close[i - window]) / btc_close[i - window]
        } else { 0.0 };
        let eth_ret = if eth_close[i - window] > 1e-10 {
            (eth_close[i] - eth_close[i - window]) / eth_close[i - window]
        } else { 0.0 };
        lead[i] = btc_ret - eth_ret;
    }
    lead
}

/// Compute BTC CVD z-score (BTC buying/selling pressure).
pub fn compute_btc_cvd_zscore(btc_buy_vol: &[f64], btc_sell_vol: &[f64], window: usize) -> Vec<f64> {
    compute_cvd_zscore(btc_buy_vol, btc_sell_vol, window)
}

/// Compute CVD z-score for scoring (separate from bias engine features)
pub fn compute_cvd_zscore(buy_vol: &[f64], sell_vol: &[f64], window: usize) -> Vec<f64> {
    let n = buy_vol.len();
    let mut cvd_bar = vec![0.0f64; n];
    for i in 0..n {
        cvd_bar[i] = buy_vol[i] - sell_vol[i];
    }

    let ema_cvd = compute_ema(&cvd_bar, window);
    let mut std_arr = vec![0.0f64; n];
    for i in window..n {
        let mut sum_sq = 0.0;
        for j in (i + 1 - window)..=i {
            let diff = cvd_bar[j] - ema_cvd[j];
            sum_sq += diff * diff;
        }
        std_arr[i] = (sum_sq / window as f64).sqrt();
    }

    let mut z = vec![0.0f64; n];
    for i in window..n {
        if std_arr[i] > 1e-15 {
            z[i] = (cvd_bar[i] - ema_cvd[i]) / std_arr[i];
        }
    }
    z
}

/// Compute combined scores for all bars.
///
/// `bias_values` — per-bar bias from bias engine (Steps 1-9)
/// `close` — 1H close prices (ETH)
/// `buy_vol`, `sell_vol` — 1H volumes (ETH)
/// `btc_close` — 1H BTC close prices (None if unavailable)
/// `btc_buy_vol`, `btc_sell_vol` — 1H BTC volumes (None if unavailable)
/// `params` — Group B parameters
pub fn compute_combined_scores(
    bias_values: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    btc_close: Option<&[f64]>,
    btc_buy_vol: Option<&[f64]>,
    btc_sell_vol: Option<&[f64]>,
    params: &GroupBParams,
) -> Vec<ScoredBar> {
    let n = close.len();

    // Mean Reversion: EMA distances
    let ema1 = compute_ema(close, params.mr_ema_span1);
    let ema2 = compute_ema(close, params.mr_ema_span2);

    // MR signals: negative when price above EMA (expect pullback)
    let mut mr1_sign = vec![0.0f64; n];
    let mut mr2_sign = vec![0.0f64; n];
    for i in 0..n {
        if ema1[i] > 1e-10 {
            mr1_sign[i] = if close[i] > ema1[i] { -1.0 } else { 1.0 };
        }
        if ema2[i] > 1e-10 {
            mr2_sign[i] = if close[i] > ema2[i] { -1.0 } else { 1.0 };
        }
    }

    // RSI
    let rsi = compute_rsi(close, params.rsi_period);

    // CVD z-score (24-bar window for scoring)
    let cvd_z = compute_cvd_zscore(buy_vol, sell_vol, 24);

    // BTC features (pre-compute if available)
    let btc_mom = btc_close.map(|bc| compute_btc_momentum(bc, params.btc_mom_window));
    let btc_lead = btc_close.map(|bc| compute_btc_lead(bc, close, params.btc_lead_window));
    let btc_cvd_z = match (btc_buy_vol, btc_sell_vol) {
        (Some(bbv), Some(bsv)) => Some(compute_btc_cvd_zscore(bbv, bsv, 24)),
        _ => None,
    };

    // Compute scores
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let mut score = 0.0f64;

        // 1. Bias engine contribution
        score += bias_values[i] * params.w_bias;

        // 2. MR primary
        score += mr1_sign[i] * params.w_mr1;

        // 3. MR secondary (only when agrees with primary)
        if mr1_sign[i] != 0.0 && mr2_sign[i] != 0.0 && mr1_sign[i] == mr2_sign[i] {
            score += mr1_sign[i] * params.w_mr2;
        }

        // 4. RSI confirmation (only when RSI is extreme AND agrees with MR)
        let rsi_oversold = rsi[i] < (50.0 - params.rsi_threshold);
        let rsi_overbought = rsi[i] > (50.0 + params.rsi_threshold);
        if (rsi_oversold && mr1_sign[i] > 0.0) || (rsi_overbought && mr1_sign[i] < 0.0) {
            score += mr1_sign[i] * params.w_rsi;
        }

        // 5. Bias + MR agreement bonus
        if bias_values[i] > 0.0 && mr1_sign[i] > 0.0 {
            score += params.w_agree;
        } else if bias_values[i] < 0.0 && mr1_sign[i] < 0.0 {
            score -= params.w_agree;  // negative score for bearish agreement
        }

        // 6. CVD contribution
        if cvd_z[i] > 0.5 {
            score += params.w_cvd;
        } else if cvd_z[i] < -0.5 {
            score -= params.w_cvd;
        }

        // 7. BTC momentum: if BTC is pumping/dumping, ETH follows
        if let Some(ref bm) = btc_mom {
            if bm[i] > 0.5 {
                score += params.w_btc_mom;
            } else if bm[i] < -0.5 {
                score -= params.w_btc_mom;
            }
        }

        // 8. BTC lead: BTC outperformed ETH → ETH catches up (bullish)
        if let Some(ref bl) = btc_lead {
            if bl[i] > 0.005 {
                score += params.w_btc_lead;
            } else if bl[i] < -0.005 {
                score -= params.w_btc_lead;
            }
        }

        // 9. BTC CVD: BTC buying pressure → bullish market
        if let Some(ref bcvd) = btc_cvd_z {
            if bcvd[i] > 0.5 {
                score += params.w_btc_cvd;
            } else if bcvd[i] < -0.5 {
                score -= params.w_btc_cvd;
            }
        }

        // 10. Strong bias engine override
        if bias_values[i].abs() >= params.bias_override_threshold {
            score = bias_values[i] * params.w_bias * params.bias_override_mult;
        }

        let direction = if score > 0.001 {
            1i8
        } else if score < -0.001 {
            -1i8
        } else {
            0i8
        };

        results.push(ScoredBar { score, direction });
    }

    results
}

/// Bias Engine — All 38 Optimizable Parameters
///
/// Group A (18 params): Require full bias engine recompute
/// Group B (20 params): Only scoring formula, <1ms per evaluation

/// Group A: Feature computation, quantization, significance, robustness, fallback
#[derive(Clone, Debug)]
pub struct GroupAParams {
    // Feature windows
    pub cvd_micro_window: usize,      // default 12
    pub cvd_macro_window: usize,      // default 288
    pub vol_micro_window: usize,      // default 12
    pub vol_macro_window: usize,      // default 288
    pub imbalance_ema_span: usize,    // default 12
    pub atr_pct_window: usize,        // default 288
    pub oi_change_window: usize,      // default 288

    // Quantization
    pub quant_window: usize,          // default 2016
    pub quantile_count: usize,        // default 5

    // Outcome
    pub k_horizon: usize,             // default 12

    // Significance
    pub min_sample_size: u32,         // default 100
    pub min_edge: f64,                // default 0.02
    pub prior_strength: f64,          // default 30.0

    // Robustness
    pub fdr_alpha: f64,               // default 0.05
    pub temporal_min_segments: usize, // default 3
    pub temporal_max_reversals: usize,// default 1
    pub min_noise_stability: f64,     // default 0.80

    // Fallback
    pub ensemble_min_n: u32,          // default 50

    // VWAP
    pub vwap_window: usize,           // default 48

    // New features (batch 1)
    pub momentum_window: usize,       // default 24
    pub wick_window: usize,           // default 24
    pub divergence_window: usize,     // default 48
    pub oi_vol_window: usize,         // default 48
    pub autocorr_window: usize,       // default 24

    // New features (batch 2)
    pub mtf_4h_window: usize,         // default 4
    pub mtf_daily_window: usize,      // default 24
}

/// Group B: Scoring formula, MR, RSI, combination weights, regime, BTC correlation
#[derive(Clone, Debug)]
pub struct GroupBParams {
    // Mean Reversion
    pub mr_ema_span1: usize,          // default 20
    pub mr_ema_span2: usize,          // default 120

    // RSI
    pub rsi_period: usize,            // default 28
    pub rsi_threshold: f64,           // default 20.0

    // Combination weights
    pub w_bias: f64,                  // default 0.3
    pub w_mr1: f64,                   // default 0.5
    pub w_mr2: f64,                   // default 0.5
    pub w_rsi: f64,                   // default 0.7
    pub w_agree: f64,                 // default 0.3
    pub w_cvd: f64,                   // default 0.5
    pub bias_override_threshold: f64, // default 0.07
    pub bias_override_mult: f64,      // default 3.0

    // Sweep overlay
    pub sweep_scale: f64,             // default 0.15
    pub sweep_aligned_weight: f64,    // default 0.30
    pub sweep_conflict_mult: f64,     // default 0.70

    // Regime
    pub regime_dir_lookback: usize,   // default 72
    pub trending_threshold: f64,      // default 1.5
    pub high_vol_threshold: f64,      // default 0.90
    pub regime_shift_lookback: usize, // default 48
    pub regime_shift_penalty: f64,    // default 0.50

    // BTC correlation
    pub btc_mom_window: usize,        // default 24 (BTC momentum z-score window)
    pub w_btc_mom: f64,               // default 0.5 (BTC momentum weight)
    pub btc_lead_window: usize,       // default 12 (BTC-ETH lead-lag window)
    pub w_btc_lead: f64,              // default 0.5 (BTC lead weight)
    pub w_btc_cvd: f64,               // default 0.3 (BTC CVD weight)
}

/// All 38 parameters combined
#[derive(Clone, Debug)]
pub struct BiasEngineParams {
    pub a: GroupAParams,
    pub b: GroupBParams,
}

impl Default for GroupAParams {
    fn default() -> Self {
        Self {
            cvd_micro_window: 12,
            cvd_macro_window: 288,
            vol_micro_window: 12,
            vol_macro_window: 288,
            imbalance_ema_span: 12,
            atr_pct_window: 288,
            oi_change_window: 288,
            quant_window: 2016,
            quantile_count: 7,
            k_horizon: 12,
            min_sample_size: 100,
            min_edge: 0.02,
            prior_strength: 30.0,
            fdr_alpha: 0.05,
            temporal_min_segments: 3,
            temporal_max_reversals: 1,
            min_noise_stability: 0.80,
            ensemble_min_n: 50,
            vwap_window: 48,
            momentum_window: 24,
            wick_window: 24,
            divergence_window: 48,
            oi_vol_window: 48,
            autocorr_window: 24,
            mtf_4h_window: 4,
            mtf_daily_window: 24,
        }
    }
}

impl Default for GroupBParams {
    fn default() -> Self {
        Self {
            mr_ema_span1: 20,
            mr_ema_span2: 120,
            rsi_period: 28,
            rsi_threshold: 20.0,
            w_bias: 0.3,
            w_mr1: 0.5,
            w_mr2: 0.5,
            w_rsi: 0.7,
            w_agree: 0.3,
            w_cvd: 0.5,
            bias_override_threshold: 0.07,
            bias_override_mult: 3.0,
            sweep_scale: 0.15,
            sweep_aligned_weight: 0.30,
            sweep_conflict_mult: 0.70,
            regime_dir_lookback: 72,
            trending_threshold: 1.5,
            high_vol_threshold: 0.90,
            regime_shift_lookback: 48,
            regime_shift_penalty: 0.50,
            btc_mom_window: 24,
            w_btc_mom: 0.5,
            btc_lead_window: 12,
            w_btc_lead: 0.5,
            w_btc_cvd: 0.3,
        }
    }
}

impl Default for BiasEngineParams {
    fn default() -> Self {
        Self {
            a: GroupAParams::default(),
            b: GroupBParams::default(),
        }
    }
}

/// Parameter specification for TPE optimizer
#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub is_int: bool,
}

/// Get all Group A parameter specs (18 params)
pub fn group_a_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec { name: "cvd_micro_window", min: 6.0, max: 48.0, step: 2.0, is_int: true },
        ParamSpec { name: "cvd_macro_window", min: 72.0, max: 720.0, step: 24.0, is_int: true },
        ParamSpec { name: "vol_micro_window", min: 6.0, max: 48.0, step: 2.0, is_int: true },
        ParamSpec { name: "vol_macro_window", min: 72.0, max: 720.0, step: 24.0, is_int: true },
        ParamSpec { name: "imbalance_ema_span", min: 4.0, max: 48.0, step: 2.0, is_int: true },
        ParamSpec { name: "atr_pct_window", min: 72.0, max: 720.0, step: 24.0, is_int: true },
        ParamSpec { name: "oi_change_window", min: 72.0, max: 720.0, step: 24.0, is_int: true },
        ParamSpec { name: "quant_window", min: 500.0, max: 5000.0, step: 100.0, is_int: true },
        ParamSpec { name: "quantile_count", min: 3.0, max: 7.0, step: 1.0, is_int: true },
        ParamSpec { name: "k_horizon", min: 1.0, max: 48.0, step: 1.0, is_int: true },
        ParamSpec { name: "min_sample_size", min: 30.0, max: 500.0, step: 10.0, is_int: true },
        ParamSpec { name: "min_edge", min: 0.005, max: 0.10, step: 0.005, is_int: false },
        ParamSpec { name: "prior_strength", min: 5.0, max: 200.0, step: 5.0, is_int: false },
        ParamSpec { name: "fdr_alpha", min: 0.01, max: 0.20, step: 0.01, is_int: false },
        ParamSpec { name: "temporal_min_segments", min: 2.0, max: 4.0, step: 1.0, is_int: true },
        ParamSpec { name: "temporal_max_reversals", min: 0.0, max: 2.0, step: 1.0, is_int: true },
        ParamSpec { name: "min_noise_stability", min: 0.50, max: 0.95, step: 0.05, is_int: false },
        ParamSpec { name: "ensemble_min_n", min: 20.0, max: 200.0, step: 10.0, is_int: true },
        ParamSpec { name: "vwap_window", min: 12.0, max: 288.0, step: 12.0, is_int: true },
        ParamSpec { name: "momentum_window", min: 6.0, max: 96.0, step: 6.0, is_int: true },
        ParamSpec { name: "wick_window", min: 6.0, max: 96.0, step: 6.0, is_int: true },
        ParamSpec { name: "divergence_window", min: 12.0, max: 144.0, step: 12.0, is_int: true },
        ParamSpec { name: "oi_vol_window", min: 12.0, max: 144.0, step: 12.0, is_int: true },
        ParamSpec { name: "autocorr_window", min: 6.0, max: 96.0, step: 6.0, is_int: true },
        ParamSpec { name: "mtf_4h_window", min: 2.0, max: 12.0, step: 1.0, is_int: true },
        ParamSpec { name: "mtf_daily_window", min: 12.0, max: 72.0, step: 6.0, is_int: true },
    ]
}

/// Get all Group B parameter specs (25 params)
pub fn group_b_specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec { name: "mr_ema_span1", min: 8.0, max: 72.0, step: 4.0, is_int: true },
        ParamSpec { name: "mr_ema_span2", min: 24.0, max: 168.0, step: 8.0, is_int: true },
        ParamSpec { name: "rsi_period", min: 6.0, max: 28.0, step: 2.0, is_int: true },
        ParamSpec { name: "rsi_threshold", min: 5.0, max: 25.0, step: 1.0, is_int: false },
        ParamSpec { name: "w_bias", min: 0.0, max: 3.0, step: 0.1, is_int: false },
        ParamSpec { name: "w_mr1", min: 0.0, max: 3.0, step: 0.1, is_int: false },
        ParamSpec { name: "w_mr2", min: 0.0, max: 2.0, step: 0.1, is_int: false },
        ParamSpec { name: "w_rsi", min: 0.0, max: 2.0, step: 0.1, is_int: false },
        ParamSpec { name: "w_agree", min: 0.0, max: 2.0, step: 0.1, is_int: false },
        ParamSpec { name: "w_cvd", min: 0.0, max: 1.0, step: 0.1, is_int: false },
        ParamSpec { name: "bias_override_threshold", min: 0.03, max: 0.15, step: 0.01, is_int: false },
        ParamSpec { name: "bias_override_mult", min: 1.5, max: 5.0, step: 0.5, is_int: false },
        ParamSpec { name: "sweep_scale", min: 0.0, max: 0.30, step: 0.05, is_int: false },
        ParamSpec { name: "sweep_aligned_weight", min: 0.0, max: 0.60, step: 0.05, is_int: false },
        ParamSpec { name: "sweep_conflict_mult", min: 0.3, max: 1.0, step: 0.1, is_int: false },
        ParamSpec { name: "regime_dir_lookback", min: 24.0, max: 288.0, step: 24.0, is_int: true },
        ParamSpec { name: "trending_threshold", min: 0.5, max: 4.0, step: 0.25, is_int: false },
        ParamSpec { name: "high_vol_threshold", min: 0.75, max: 0.98, step: 0.01, is_int: false },
        ParamSpec { name: "regime_shift_lookback", min: 12.0, max: 168.0, step: 12.0, is_int: true },
        ParamSpec { name: "regime_shift_penalty", min: 0.0, max: 1.0, step: 0.1, is_int: false },
        // BTC correlation
        ParamSpec { name: "btc_mom_window", min: 6.0, max: 72.0, step: 6.0, is_int: true },
        ParamSpec { name: "w_btc_mom", min: 0.0, max: 2.0, step: 0.1, is_int: false },
        ParamSpec { name: "btc_lead_window", min: 3.0, max: 48.0, step: 3.0, is_int: true },
        ParamSpec { name: "w_btc_lead", min: 0.0, max: 2.0, step: 0.1, is_int: false },
        ParamSpec { name: "w_btc_cvd", min: 0.0, max: 1.0, step: 0.1, is_int: false },
    ]
}

/// Convert Group A f64 vector back to params struct
pub fn vec_to_group_a(vals: &[f64]) -> GroupAParams {
    GroupAParams {
        cvd_micro_window: vals[0] as usize,
        cvd_macro_window: vals[1] as usize,
        vol_micro_window: vals[2] as usize,
        vol_macro_window: vals[3] as usize,
        imbalance_ema_span: vals[4] as usize,
        atr_pct_window: vals[5] as usize,
        oi_change_window: vals[6] as usize,
        quant_window: vals[7] as usize,
        quantile_count: vals[8] as usize,
        k_horizon: vals[9] as usize,
        min_sample_size: vals[10] as u32,
        min_edge: vals[11],
        prior_strength: vals[12],
        fdr_alpha: vals[13],
        temporal_min_segments: vals[14] as usize,
        temporal_max_reversals: vals[15] as usize,
        min_noise_stability: vals[16],
        ensemble_min_n: vals[17] as u32,
        vwap_window: if vals.len() > 18 { vals[18] as usize } else { 48 },
        momentum_window: if vals.len() > 19 { vals[19] as usize } else { 24 },
        wick_window: if vals.len() > 20 { vals[20] as usize } else { 24 },
        divergence_window: if vals.len() > 21 { vals[21] as usize } else { 48 },
        oi_vol_window: if vals.len() > 22 { vals[22] as usize } else { 48 },
        autocorr_window: if vals.len() > 23 { vals[23] as usize } else { 24 },
        mtf_4h_window: if vals.len() > 24 { vals[24] as usize } else { 4 },
        mtf_daily_window: if vals.len() > 25 { vals[25] as usize } else { 24 },
    }
}

/// Convert Group B f64 vector back to params struct
pub fn vec_to_group_b(vals: &[f64]) -> GroupBParams {
    GroupBParams {
        mr_ema_span1: vals[0] as usize,
        mr_ema_span2: vals[1] as usize,
        rsi_period: vals[2] as usize,
        rsi_threshold: vals[3],
        w_bias: vals[4],
        w_mr1: vals[5],
        w_mr2: vals[6],
        w_rsi: vals[7],
        w_agree: vals[8],
        w_cvd: vals[9],
        bias_override_threshold: vals[10],
        bias_override_mult: vals[11],
        sweep_scale: vals[12],
        sweep_aligned_weight: vals[13],
        sweep_conflict_mult: vals[14],
        regime_dir_lookback: vals[15] as usize,
        trending_threshold: vals[16],
        high_vol_threshold: vals[17],
        regime_shift_lookback: vals[18] as usize,
        regime_shift_penalty: vals[19],
        btc_mom_window: vals[20] as usize,
        w_btc_mom: vals[21],
        btc_lead_window: vals[22] as usize,
        w_btc_lead: vals[23],
        w_btc_cvd: vals[24],
    }
}

/// Convert Group A params to f64 vector
pub fn group_a_to_vec(p: &GroupAParams) -> Vec<f64> {
    vec![
        p.cvd_micro_window as f64, p.cvd_macro_window as f64,
        p.vol_micro_window as f64, p.vol_macro_window as f64,
        p.imbalance_ema_span as f64, p.atr_pct_window as f64,
        p.oi_change_window as f64, p.quant_window as f64,
        p.quantile_count as f64, p.k_horizon as f64,
        p.min_sample_size as f64, p.min_edge, p.prior_strength,
        p.fdr_alpha, p.temporal_min_segments as f64,
        p.temporal_max_reversals as f64, p.min_noise_stability,
        p.ensemble_min_n as f64,
        p.vwap_window as f64,
        p.momentum_window as f64,
        p.wick_window as f64,
        p.divergence_window as f64,
        p.oi_vol_window as f64,
        p.autocorr_window as f64,
        p.mtf_4h_window as f64,
        p.mtf_daily_window as f64,
    ]
}

/// Convert Group B params to f64 vector
pub fn group_b_to_vec(p: &GroupBParams) -> Vec<f64> {
    vec![
        p.mr_ema_span1 as f64, p.mr_ema_span2 as f64,
        p.rsi_period as f64, p.rsi_threshold,
        p.w_bias, p.w_mr1, p.w_mr2, p.w_rsi, p.w_agree, p.w_cvd,
        p.bias_override_threshold, p.bias_override_mult,
        p.sweep_scale, p.sweep_aligned_weight, p.sweep_conflict_mult,
        p.regime_dir_lookback as f64, p.trending_threshold,
        p.high_vol_threshold, p.regime_shift_lookback as f64,
        p.regime_shift_penalty,
        p.btc_mom_window as f64, p.w_btc_mom,
        p.btc_lead_window as f64, p.w_btc_lead,
        p.w_btc_cvd,
    ]
}

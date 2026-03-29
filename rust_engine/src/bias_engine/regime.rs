/// Bias Engine — Step 7: Regime Detection (Spec Section 11)
///
/// Three regimes:
///   1. HIGH_VOLATILITY — ATR percentile > 0.90
///   2. TRENDING        — directional move > 1.5 ATR (72-bar)
///   3. MEAN_REVERTING  — everything else
///
/// Regime shift: regime changed within last 48 bars → bias penalty × 0.50

/// Market regime classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Regime {
    Trending = 0,
    MeanReverting = 1,
    HighVolatility = 2,
}

/// Per-bar regime information.
#[derive(Clone, Debug)]
pub struct RegimeInfo {
    pub regime: Regime,
    /// True if regime changed within the last 48 bars
    pub regime_shift: bool,
    /// abs(close[i] - close[i-72]) / atr_72
    pub directional_move: f64,
    /// ATR percentile at this bar (0–1)
    pub vol_regime: f64,
}

/// Classify a single bar's regime.
fn classify_regime(directional_move: f64, vol_regime: f64) -> Regime {
    if vol_regime > 0.90 {
        Regime::HighVolatility
    } else if directional_move > 1.5 {
        Regime::Trending
    } else {
        Regime::MeanReverting
    }
}

/// Compute rolling ATR (simple mean of high−low over `window`).
fn rolling_atr(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
    let n = high.len();
    let mut atr = vec![f64::NAN; n];
    let mut sum = 0.0;

    for i in 0..n {
        let range = high[i] - low[i];
        sum += range;
        if i >= window {
            sum -= high[i - window] - low[i - window];
        }
        if i + 1 >= window {
            atr[i] = sum / window as f64;
        }
    }
    atr
}

/// Detect regime with custom parameters.
pub fn detect_regimes_with_params(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    atr_pct: &[f64],
    dir_lookback: usize,
    trending_threshold: f64,
    high_vol_threshold: f64,
    shift_lookback: usize,
) -> Vec<RegimeInfo> {
    let n = close.len();
    let atr_w = rolling_atr(high, low, dir_lookback);

    let classify = |dm: f64, vr: f64| -> Regime {
        if vr > high_vol_threshold { Regime::HighVolatility }
        else if dm > trending_threshold { Regime::Trending }
        else { Regime::MeanReverting }
    };

    let mut raw_regimes = Vec::with_capacity(n);
    let mut dir_moves = vec![0.0f64; n];

    for i in 0..n {
        let (dm, regime) = if i < dir_lookback || atr_w[i].is_nan() || atr_w[i] < 1e-15 {
            (0.0, if atr_pct[i].is_nan() { Regime::MeanReverting } else { classify(0.0, atr_pct[i]) })
        } else {
            let dm = (close[i] - close[i - dir_lookback]).abs() / atr_w[i];
            let vr = if atr_pct[i].is_nan() { 0.5 } else { atr_pct[i] };
            (dm, classify(dm, vr))
        };
        dir_moves[i] = dm;
        raw_regimes.push(regime);
    }

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let shift = if i >= shift_lookback {
            raw_regimes[i] != raw_regimes[i - shift_lookback]
        } else { false };
        results.push(RegimeInfo {
            regime: raw_regimes[i], regime_shift: shift,
            directional_move: dir_moves[i],
            vol_regime: if atr_pct[i].is_nan() { 0.5 } else { atr_pct[i] },
        });
    }
    results
}

/// Detect regime for every bar.
///
/// `close`   — 5m close prices
/// `high`    — 5m high prices
/// `low`     — 5m low prices
/// `atr_pct` — pre-computed ATR percentile (feature #6, 0–1 range)
pub fn detect_regimes(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    atr_pct: &[f64],
) -> Vec<RegimeInfo> {
    let n = close.len();
    const DIR_LOOKBACK: usize = 72; // 6 hours at 5m
    const ATR_WINDOW: usize = 72;
    const SHIFT_LOOKBACK: usize = 48;

    let atr_72 = rolling_atr(high, low, ATR_WINDOW);

    // First pass: compute raw regime per bar
    let mut raw_regimes = Vec::with_capacity(n);
    let mut dir_moves = vec![0.0f64; n];

    for i in 0..n {
        let (dm, regime) = if i < DIR_LOOKBACK || atr_72[i].is_nan() || atr_72[i] < 1e-15 {
            (0.0, if atr_pct[i].is_nan() {
                Regime::MeanReverting
            } else {
                classify_regime(0.0, atr_pct[i])
            })
        } else {
            let dm = (close[i] - close[i - DIR_LOOKBACK]).abs() / atr_72[i];
            let vr = if atr_pct[i].is_nan() { 0.5 } else { atr_pct[i] };
            (dm, classify_regime(dm, vr))
        };
        dir_moves[i] = dm;
        raw_regimes.push(regime);
    }

    // Second pass: detect regime shifts
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let shift = if i >= SHIFT_LOOKBACK {
            raw_regimes[i] != raw_regimes[i - SHIFT_LOOKBACK]
        } else {
            false
        };

        results.push(RegimeInfo {
            regime: raw_regimes[i],
            regime_shift: shift,
            directional_move: dir_moves[i],
            vol_regime: if atr_pct[i].is_nan() { 0.5 } else { atr_pct[i] },
        });
    }

    results
}

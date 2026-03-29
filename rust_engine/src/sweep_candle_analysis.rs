/// Sweep Candle Analysis — Mum kapanisi bazli continuation/reversal analizi
///
/// 1H mumlar uzerinde:
/// - High sweep + close > prev_high = Continuation (LONG)
/// - High sweep + close < prev_high + kirmizi mum = Reversal (SHORT)
/// - High sweep + close < prev_high + yesil mum = Belirsiz (pas)
/// - Low sweep + close < prev_low = Continuation (SHORT)
/// - Low sweep + close > prev_low + yesil mum = Reversal (LONG)
/// - Low sweep + close > prev_low + kirmizi mum = Belirsiz (pas)
/// - Inside bar = pas

use crate::sweep_miner::compute_5m_features;

const BP_1H: usize = 12; // 5m bars per 1h

// ── Event types ──

#[derive(Clone, Debug)]
pub enum CandleSignal {
    HighContinuation,  // LONG
    HighReversal,      // SHORT
    HighAmbiguous,     // pas
    LowContinuation,   // SHORT
    LowReversal,       // LONG
    LowAmbiguous,      // pas
    InsideBar,         // pas
}

#[derive(Clone, Debug)]
pub struct CandleEvent {
    pub bar_idx: usize,       // 1H mum'un 5m baslangic index'i
    pub signal: CandleSignal,
    pub prev_high: f64,
    pub prev_low: f64,
    pub prev_open: f64,
    pub prev_close: f64,
    pub curr_open: f64,
    pub curr_high: f64,
    pub curr_low: f64,
    pub curr_close: f64,
    pub features: [f64; 7],   // mum kapanisindaki feature'lar
}

// ── Aftermath: sonraki N mum ne olmus ──

#[derive(Clone, Debug)]
pub struct Aftermath {
    pub horizon_bars: usize,  // kac 1H mum sonra
    pub avg_return: f64,
    pub median_return: f64,
    pub win_rate: f64,        // yonde kar eden orani
    pub sample: usize,
    pub max_favorable: f64,
    pub max_adverse: f64,
}

// ── Full result ──

pub struct CandleAnalysisResult {
    pub total_1h_candles: usize,
    pub high_cont_count: usize,
    pub high_rev_count: usize,
    pub high_ambig_count: usize,
    pub low_cont_count: usize,
    pub low_rev_count: usize,
    pub low_ambig_count: usize,
    pub inside_bar_count: usize,

    // Aftermath per signal type per horizon
    pub high_cont_aftermath: Vec<Aftermath>,
    pub high_rev_aftermath: Vec<Aftermath>,
    pub low_cont_aftermath: Vec<Aftermath>,
    pub low_rev_aftermath: Vec<Aftermath>,

    // Feature comparison
    pub feature_comparisons: Vec<FeatureComp>,

    // Events for external use
    pub events: Vec<CandleEvent>,
}

#[derive(Clone, Debug)]
pub struct FeatureComp {
    pub sweep_type: String,    // "high" or "low"
    pub feature_name: String,
    pub cont_median: f64,
    pub rev_median: f64,
    pub ambig_median: f64,
    pub p_value: f64,
    pub significant: bool,
}

// ── Helpers ──

fn median_of(vals: &mut Vec<f64>) -> f64 {
    if vals.is_empty() { return f64::NAN; }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 0 { (vals[n/2 - 1] + vals[n/2]) / 2.0 } else { vals[n/2] }
}

fn mann_whitney_p(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len();
    let nb = b.len();
    if na < 5 || nb < 5 { return 1.0; }
    let mut u: f64 = 0.0;
    for &ai in a {
        for &bj in b {
            if ai > bj { u += 1.0; }
            else if (ai - bj).abs() < 1e-12 { u += 0.5; }
        }
    }
    let mu = na as f64 * nb as f64 / 2.0;
    let sigma = ((na as f64 * nb as f64 * (na + nb + 1) as f64) / 12.0).sqrt();
    if sigma <= 0.0 { return 1.0; }
    let z = (u - mu) / sigma;
    2.0 * (1.0 - normal_cdf(z.abs()))
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

pub const FEATURE_NAMES: [&str; 7] = [
    "CVD_zscore_micro", "CVD_zscore_macro", "OI_change",
    "Vol_zscore_micro", "Vol_zscore_macro", "Imbalance_smooth", "ATR_percentile",
];

// ── Main analysis ──

pub fn run_candle_analysis(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> CandleAnalysisResult {
    run_candle_analysis_full(closes, highs, lows, buy_vol, sell_vol, oi, 0, BP_1H)
}

pub fn run_candle_analysis_offset(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    feat_bar_offset: usize,
) -> CandleAnalysisResult {
    run_candle_analysis_full(closes, highs, lows, buy_vol, sell_vol, oi, feat_bar_offset, BP_1H)
}

/// feat_bar_offset: 0 = t mumunun ilk 5m bar'i, ...
/// bars_per_candle: 12 = 1H, 48 = 4H, 3 = 15M, etc.
pub fn run_candle_analysis_full(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> CandleAnalysisResult {
    let bp = bars_per_candle;
    let n = closes.len();

    // Features
    let (features, _daily_atr) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);

    // Build candles
    let n_candles = n / bp;
    let mut candles: Vec<(usize, f64, f64, f64, f64)> = Vec::with_capacity(n_candles);
    // (start_bar, open, high, low, close)
    for c in 0..n_candles {
        let start = c * bp;
        let end = start + bp;
        let o = closes[start]; // open = first bar close (approx, better: open of first bar)
        // Actually use proper OHLC
        let mut h = f64::NEG_INFINITY;
        let mut l = f64::INFINITY;
        for i in start..end {
            if highs[i] > h { h = highs[i]; }
            if lows[i] < l { l = lows[i]; }
        }
        let c_price = closes[end - 1];
        let o_price = closes[start]; // approximate open
        candles.push((start, o_price, h, l, c_price));
    }

    // Classify each candle
    let mut events: Vec<CandleEvent> = Vec::new();

    for i in 1..candles.len() {
        let (prev_start, prev_open, prev_high, prev_low, prev_close) = candles[i - 1];
        let (curr_start, curr_open, curr_high, curr_low, curr_close) = candles[i];

        // Get features at t mumunun feat_bar_offset'inci 5m bar'i
        let feat_idx = curr_start + feat_bar_offset.min(bp - 1);
        if feat_idx >= n || features[feat_idx][0].is_nan() { continue; }
        let feat = features[feat_idx];

        let is_green = curr_close > curr_open;
        let is_red = curr_close < curr_open;

        let swept_high = curr_high > prev_high;
        let swept_low = curr_low < prev_low;

        let signal = if swept_high && !swept_low {
            // High sweep only
            if curr_close > prev_high {
                CandleSignal::HighContinuation
            } else if is_red {
                CandleSignal::HighReversal
            } else {
                CandleSignal::HighAmbiguous
            }
        } else if swept_low && !swept_high {
            // Low sweep only
            if curr_close < prev_low {
                CandleSignal::LowContinuation
            } else if is_green {
                CandleSignal::LowReversal
            } else {
                CandleSignal::LowAmbiguous
            }
        } else if swept_high && swept_low {
            // Both swept — outside bar, check close
            if curr_close > prev_high {
                CandleSignal::HighContinuation
            } else if curr_close < prev_low {
                CandleSignal::LowContinuation
            } else if is_red {
                CandleSignal::HighReversal // swept both but closed red inside
            } else if is_green {
                CandleSignal::LowReversal // swept both but closed green inside
            } else {
                CandleSignal::InsideBar
            }
        } else {
            // Inside bar
            CandleSignal::InsideBar
        };

        events.push(CandleEvent {
            bar_idx: curr_start,
            signal,
            prev_high, prev_low, prev_open, prev_close,
            curr_open, curr_high, curr_low, curr_close,
            features: feat,
        });
    }

    // Count
    let mut hc = 0usize; let mut hr = 0; let mut ha = 0;
    let mut lc = 0; let mut lr = 0; let mut la = 0;
    let mut ib = 0;
    for e in &events {
        match e.signal {
            CandleSignal::HighContinuation => hc += 1,
            CandleSignal::HighReversal => hr += 1,
            CandleSignal::HighAmbiguous => ha += 1,
            CandleSignal::LowContinuation => lc += 1,
            CandleSignal::LowReversal => lr += 1,
            CandleSignal::LowAmbiguous => la += 1,
            CandleSignal::InsideBar => ib += 1,
        }
    }

    // ── Aftermath analysis ──
    // After each signal, what happens in next 1, 2, 4, 8, 24 candles?
    let horizons = [1usize, 2, 4, 8, 24]; // 1H candles

    fn compute_aftermath(
        events: &[CandleEvent], candles: &[(usize, f64, f64, f64, f64)],
        filter: &dyn Fn(&CandleSignal) -> bool,
        direction: f64, // +1 for long expectation, -1 for short
        horizons: &[usize],
        bp: usize,
    ) -> Vec<Aftermath> {
        let n_candles = candles.len();
        horizons.iter().map(|&h| {
            let mut returns: Vec<f64> = Vec::new();
            let mut max_fav: Vec<f64> = Vec::new();
            let mut max_adv: Vec<f64> = Vec::new();

            for e in events {
                if !filter(&e.signal) { continue; }
                let candle_idx = e.bar_idx / bp;
                if candle_idx + h >= n_candles { continue; }

                let entry = e.curr_close;
                if entry <= 0.0 { continue; }

                // Return at horizon
                let exit = candles[candle_idx + h].4; // close of horizon candle
                let ret = (exit - entry) / entry * 100.0 * direction;
                returns.push(ret);

                // MFE/MAE
                let mut best = 0.0_f64;
                let mut worst = 0.0_f64;
                for j in 1..=h {
                    let ci = candle_idx + j;
                    if ci >= n_candles { break; }
                    let (_, _, ch, cl, _) = candles[ci];
                    let fav = if direction > 0.0 { (ch - entry) / entry * 100.0 } else { (entry - cl) / entry * 100.0 };
                    let adv = if direction > 0.0 { (entry - cl) / entry * 100.0 } else { (ch - entry) / entry * 100.0 };
                    if fav > best { best = fav; }
                    if adv > worst { worst = adv; }
                }
                max_fav.push(best);
                max_adv.push(worst);
            }

            let sample = returns.len();
            let wins = returns.iter().filter(|&&r| r > 0.0).count();
            let wr = if sample > 0 { wins as f64 / sample as f64 * 100.0 } else { 0.0 };
            let avg = if sample > 0 { returns.iter().sum::<f64>() / sample as f64 } else { 0.0 };
            let med = median_of(&mut returns.clone());
            let mfe = if max_fav.is_empty() { 0.0 } else { median_of(&mut max_fav) };
            let mae = if max_adv.is_empty() { 0.0 } else { median_of(&mut max_adv) };

            Aftermath {
                horizon_bars: h, avg_return: avg, median_return: med,
                win_rate: wr, sample, max_favorable: mfe, max_adverse: mae,
            }
        }).collect()
    }

    let hc_aft = compute_aftermath(&events, &candles,
        &|s| matches!(s, CandleSignal::HighContinuation), 1.0, &horizons, bp);
    let hr_aft = compute_aftermath(&events, &candles,
        &|s| matches!(s, CandleSignal::HighReversal), -1.0, &horizons, bp);
    let lc_aft = compute_aftermath(&events, &candles,
        &|s| matches!(s, CandleSignal::LowContinuation), -1.0, &horizons, bp);
    let lr_aft = compute_aftermath(&events, &candles,
        &|s| matches!(s, CandleSignal::LowReversal), 1.0, &horizons, bp);

    // ── Feature comparison ──
    let mut feat_comps: Vec<FeatureComp> = Vec::new();

    // High sweep: continuation vs reversal
    let hc_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::HighContinuation)).collect();
    let hr_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::HighReversal)).collect();
    let ha_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::HighAmbiguous)).collect();

    for fi in 0..7 {
        let c_vals: Vec<f64> = hc_events.iter().map(|e| e.features[fi]).collect();
        let r_vals: Vec<f64> = hr_events.iter().map(|e| e.features[fi]).collect();
        let a_vals: Vec<f64> = ha_events.iter().map(|e| e.features[fi]).collect();

        let c_med = median_of(&mut c_vals.clone());
        let r_med = median_of(&mut r_vals.clone());
        let a_med = median_of(&mut a_vals.clone());
        let p = mann_whitney_p(&c_vals, &r_vals);

        feat_comps.push(FeatureComp {
            sweep_type: "high".to_string(),
            feature_name: FEATURE_NAMES[fi].to_string(),
            cont_median: c_med, rev_median: r_med, ambig_median: a_med,
            p_value: p, significant: p < 0.05,
        });
    }

    // Low sweep: continuation vs reversal
    let lc_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::LowContinuation)).collect();
    let lr_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::LowReversal)).collect();
    let la_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::LowAmbiguous)).collect();

    for fi in 0..7 {
        let c_vals: Vec<f64> = lc_events.iter().map(|e| e.features[fi]).collect();
        let r_vals: Vec<f64> = lr_events.iter().map(|e| e.features[fi]).collect();
        let a_vals: Vec<f64> = la_events.iter().map(|e| e.features[fi]).collect();

        let c_med = median_of(&mut c_vals.clone());
        let r_med = median_of(&mut r_vals.clone());
        let a_med = median_of(&mut a_vals.clone());
        let p = mann_whitney_p(&c_vals, &r_vals);

        feat_comps.push(FeatureComp {
            sweep_type: "low".to_string(),
            feature_name: FEATURE_NAMES[fi].to_string(),
            cont_median: c_med, rev_median: r_med, ambig_median: a_med,
            p_value: p, significant: p < 0.05,
        });
    }

    CandleAnalysisResult {
        total_1h_candles: candles.len(),
        high_cont_count: hc, high_rev_count: hr, high_ambig_count: ha,
        low_cont_count: lc, low_rev_count: lr, low_ambig_count: la,
        inside_bar_count: ib,
        high_cont_aftermath: hc_aft, high_rev_aftermath: hr_aft,
        low_cont_aftermath: lc_aft, low_rev_aftermath: lr_aft,
        feature_comparisons: feat_comps,
        events,
    }
}

/// Cross-asset analysis: ETH mumlari + baska varligin (BTC) feature'lari
/// candle_* = ETH (mum yapisi + sinyal siniflandirmasi icin)
/// feat_* = BTC (feature hesaplama icin)
pub fn run_candle_analysis_cross(
    candle_closes: &[f64], candle_highs: &[f64], candle_lows: &[f64],
    feat_closes: &[f64], feat_highs: &[f64], feat_lows: &[f64],
    feat_buy_vol: &[f64], feat_sell_vol: &[f64], feat_oi: &[f64],
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> CandleAnalysisResult {
    let bp = bars_per_candle;
    let n = candle_closes.len().min(feat_closes.len());

    // Features from external asset (BTC)
    let (features, _daily_atr) = compute_5m_features(feat_closes, feat_highs, feat_lows, feat_buy_vol, feat_sell_vol, feat_oi);

    // Build candles from primary asset (ETH)
    let n_candles = n / bp;
    let mut candles: Vec<(usize, f64, f64, f64, f64)> = Vec::with_capacity(n_candles);
    for c in 0..n_candles {
        let start = c * bp;
        let end = start + bp;
        if end > n { break; }
        let mut h = f64::NEG_INFINITY;
        let mut l = f64::INFINITY;
        for i in start..end {
            if candle_highs[i] > h { h = candle_highs[i]; }
            if candle_lows[i] < l { l = candle_lows[i]; }
        }
        let c_price = candle_closes[end - 1];
        let o_price = candle_closes[start];
        candles.push((start, o_price, h, l, c_price));
    }

    // Classify each candle (using ETH candle data)
    let mut events: Vec<CandleEvent> = Vec::new();

    for i in 1..candles.len() {
        let (_prev_start, prev_open, prev_high, prev_low, prev_close) = candles[i - 1];
        let (curr_start, curr_open, curr_high, curr_low, curr_close) = candles[i];

        // Features from BTC at specified offset
        let feat_idx = curr_start + feat_bar_offset.min(bp - 1);
        if feat_idx >= n || feat_idx >= features.len() || features[feat_idx][0].is_nan() { continue; }
        let feat = features[feat_idx];

        let is_green = curr_close > curr_open;
        let is_red = curr_close < curr_open;

        let swept_high = curr_high > prev_high;
        let swept_low = curr_low < prev_low;

        let signal = if swept_high && !swept_low {
            if curr_close > prev_high {
                CandleSignal::HighContinuation
            } else if is_red {
                CandleSignal::HighReversal
            } else {
                CandleSignal::HighAmbiguous
            }
        } else if swept_low && !swept_high {
            if curr_close < prev_low {
                CandleSignal::LowContinuation
            } else if is_green {
                CandleSignal::LowReversal
            } else {
                CandleSignal::LowAmbiguous
            }
        } else if swept_high && swept_low {
            if curr_close > prev_high {
                CandleSignal::HighContinuation
            } else if curr_close < prev_low {
                CandleSignal::LowContinuation
            } else if is_red {
                CandleSignal::HighReversal
            } else if is_green {
                CandleSignal::LowReversal
            } else {
                CandleSignal::InsideBar
            }
        } else {
            CandleSignal::InsideBar
        };

        events.push(CandleEvent {
            bar_idx: curr_start,
            signal,
            prev_high, prev_low, prev_open, prev_close,
            curr_open, curr_high, curr_low, curr_close,
            features: feat,
        });
    }

    // Count
    let mut hc = 0usize; let mut hr = 0; let mut ha = 0;
    let mut lc = 0; let mut lr = 0; let mut la = 0;
    let mut ib = 0;
    for e in &events {
        match e.signal {
            CandleSignal::HighContinuation => hc += 1,
            CandleSignal::HighReversal => hr += 1,
            CandleSignal::HighAmbiguous => ha += 1,
            CandleSignal::LowContinuation => lc += 1,
            CandleSignal::LowReversal => lr += 1,
            CandleSignal::LowAmbiguous => la += 1,
            CandleSignal::InsideBar => ib += 1,
        }
    }

    // Aftermath (using ETH candle data)
    let horizons = [1usize, 2, 4, 8, 24];

    fn compute_aftermath_cross(
        events: &[CandleEvent], candles: &[(usize, f64, f64, f64, f64)],
        filter: &dyn Fn(&CandleSignal) -> bool,
        direction: f64,
        horizons: &[usize],
        bp: usize,
    ) -> Vec<Aftermath> {
        let n_candles = candles.len();
        horizons.iter().map(|&h| {
            let mut returns: Vec<f64> = Vec::new();
            let mut max_fav: Vec<f64> = Vec::new();
            let mut max_adv: Vec<f64> = Vec::new();
            for e in events {
                if !filter(&e.signal) { continue; }
                let candle_idx = e.bar_idx / bp;
                if candle_idx + h >= n_candles { continue; }
                let entry = e.curr_close;
                if entry <= 0.0 { continue; }
                let exit = candles[candle_idx + h].4;
                let ret = (exit - entry) / entry * 100.0 * direction;
                returns.push(ret);
                let mut best = 0.0_f64;
                let mut worst = 0.0_f64;
                for j in 1..=h {
                    let ci = candle_idx + j;
                    if ci >= n_candles { break; }
                    let (_, _, ch, cl, _) = candles[ci];
                    let fav = if direction > 0.0 { (ch - entry) / entry * 100.0 } else { (entry - cl) / entry * 100.0 };
                    let adv = if direction > 0.0 { (entry - cl) / entry * 100.0 } else { (ch - entry) / entry * 100.0 };
                    if fav > best { best = fav; }
                    if adv > worst { worst = adv; }
                }
                max_fav.push(best);
                max_adv.push(worst);
            }
            let sample = returns.len();
            let wins = returns.iter().filter(|&&r| r > 0.0).count();
            let wr = if sample > 0 { wins as f64 / sample as f64 * 100.0 } else { 0.0 };
            let avg = if sample > 0 { returns.iter().sum::<f64>() / sample as f64 } else { 0.0 };
            let med = median_of(&mut returns.clone());
            let mfe = if max_fav.is_empty() { 0.0 } else { median_of(&mut max_fav) };
            let mae = if max_adv.is_empty() { 0.0 } else { median_of(&mut max_adv) };
            Aftermath {
                horizon_bars: h, avg_return: avg, median_return: med,
                win_rate: wr, sample, max_favorable: mfe, max_adverse: mae,
            }
        }).collect()
    }

    let hc_aft = compute_aftermath_cross(&events, &candles,
        &|s| matches!(s, CandleSignal::HighContinuation), 1.0, &horizons, bp);
    let hr_aft = compute_aftermath_cross(&events, &candles,
        &|s| matches!(s, CandleSignal::HighReversal), -1.0, &horizons, bp);
    let lc_aft = compute_aftermath_cross(&events, &candles,
        &|s| matches!(s, CandleSignal::LowContinuation), -1.0, &horizons, bp);
    let lr_aft = compute_aftermath_cross(&events, &candles,
        &|s| matches!(s, CandleSignal::LowReversal), 1.0, &horizons, bp);

    // Feature comparison
    let mut feat_comps: Vec<FeatureComp> = Vec::new();

    let hc_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::HighContinuation)).collect();
    let hr_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::HighReversal)).collect();
    let ha_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::HighAmbiguous)).collect();

    for fi in 0..7 {
        let c_vals: Vec<f64> = hc_events.iter().map(|e| e.features[fi]).collect();
        let r_vals: Vec<f64> = hr_events.iter().map(|e| e.features[fi]).collect();
        let a_vals: Vec<f64> = ha_events.iter().map(|e| e.features[fi]).collect();
        let c_med = median_of(&mut c_vals.clone());
        let r_med = median_of(&mut r_vals.clone());
        let a_med = median_of(&mut a_vals.clone());
        let p = mann_whitney_p(&c_vals, &r_vals);
        feat_comps.push(FeatureComp {
            sweep_type: "high".to_string(),
            feature_name: FEATURE_NAMES[fi].to_string(),
            cont_median: c_med, rev_median: r_med, ambig_median: a_med,
            p_value: p, significant: p < 0.05,
        });
    }

    let lc_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::LowContinuation)).collect();
    let lr_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::LowReversal)).collect();
    let la_events: Vec<&CandleEvent> = events.iter().filter(|e| matches!(e.signal, CandleSignal::LowAmbiguous)).collect();

    for fi in 0..7 {
        let c_vals: Vec<f64> = lc_events.iter().map(|e| e.features[fi]).collect();
        let r_vals: Vec<f64> = lr_events.iter().map(|e| e.features[fi]).collect();
        let a_vals: Vec<f64> = la_events.iter().map(|e| e.features[fi]).collect();
        let c_med = median_of(&mut c_vals.clone());
        let r_med = median_of(&mut r_vals.clone());
        let a_med = median_of(&mut a_vals.clone());
        let p = mann_whitney_p(&c_vals, &r_vals);
        feat_comps.push(FeatureComp {
            sweep_type: "low".to_string(),
            feature_name: FEATURE_NAMES[fi].to_string(),
            cont_median: c_med, rev_median: r_med, ambig_median: a_med,
            p_value: p, significant: p < 0.05,
        });
    }

    CandleAnalysisResult {
        total_1h_candles: candles.len(),
        high_cont_count: hc, high_rev_count: hr, high_ambig_count: ha,
        low_cont_count: lc, low_rev_count: lr, low_ambig_count: la,
        inside_bar_count: ib,
        high_cont_aftermath: hc_aft, high_rev_aftermath: hr_aft,
        low_cont_aftermath: lc_aft, low_rev_aftermath: lr_aft,
        feature_comparisons: feat_comps,
        events,
    }
}

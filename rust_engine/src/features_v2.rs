/// Feature Set v2 — Kapsamli piyasa ozellikleri
///
/// 5m OHLCV + buy_vol/sell_vol'dan turetilen genis feature seti.
/// Kategoriler:
///   1. Order Flow (CVD, Imbalance) — mevcut
///   2. Fiyat Yapisi (trend, momentum, mean-reversion)
///   3. Mum Anatomisi (body/wick oranlari, pattern'ler)
///   4. Volatilite (ATR, squeeze, expansion)
///   5. Volume Profili (volume anomalileri)
///   6. Zaman (session, gun ici)
///   7. Multi-timeframe (ust TF context)
///   8. Pre-sweep CVD (sweep oncesi momentum tukenmisligi)

/// Feature sayisi
pub const N_FEATURES_V2: usize = 28;

pub const FEATURE_NAMES_V2: [&str; N_FEATURES_V2] = [
    // Order Flow (0-4)
    "CVD_zscore_micro",      // 0: 1h CVD z-score
    "CVD_zscore_macro",      // 1: 24h CVD z-score
    "Imbalance_smooth",      // 2: EMA smoothed buy/sell imbalance
    "CVD_accel",             // 3: CVD ivmelenme (2. turev)
    "Imbalance_divergence",  // 4: fiyat-imbalance divergence

    // Fiyat Yapisi (5-10)
    "Price_trend_1h",        // 5: son 1h lineer regresyon egimi
    "Price_trend_4h",        // 6: son 4h lineer regresyon egimi
    "Price_trend_24h",       // 7: son 24h lineer regresyon egimi
    "Momentum_roc_1h",       // 8: 1h rate of change
    "Momentum_roc_4h",       // 9: 4h rate of change
    "Mean_rev_zscore",       // 10: 24h ortalamadan sapma z-score

    // Mum Anatomisi (11-15)
    "Body_ratio",            // 11: |close-open| / (high-low) — gövde/range oranı
    "Upper_wick_ratio",      // 12: üst fitil / range
    "Lower_wick_ratio",      // 13: alt fitil / range
    "Candle_direction_streak", // 14: ardışık aynı yön mum sayısı
    "Range_vs_avg",          // 15: mum range / ortalama range

    // Volatilite (16-19)
    "ATR_percentile",        // 16: 24h ATR percentile
    "Volatility_ratio",      // 17: kısa vadeli vol / uzun vadeli vol (squeeze/expansion)
    "BB_width_pctile",       // 18: Bollinger Band genişliği percentile
    "Range_expansion",       // 19: son 3 mumun range'i / önceki 3 mumun range'i

    // Volume (20-22)
    "Vol_zscore_micro",      // 20: 1h volume z-score
    "Vol_spike",             // 21: anlık volume / 24h ortalama
    "Vol_trend",             // 22: volume artıyor mu azalıyor mu (regresyon)

    // Multi-TF (23-25)
    "Higher_high_count",     // 23: son 6 mumda kaç higher high var (trend gücü)
    "Lower_low_count",       // 24: son 6 mumda kaç lower low var
    "Swing_position",        // 25: fiyatın son 24h range'ındaki yeri (0-100)

    // Pre-sweep (26-27)
    "Pre_CVD_2h",            // 26: 2h öncesi kümülatif CVD değişimi (z-scored)
    "Pre_CVD_4h",            // 27: 4h öncesi kümülatif CVD değişimi (z-scored)
];

// ── Helpers ──

fn linreg_slope(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 { return 0.0; }
    let nf = n as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut sx2 = 0.0_f64;
    for i in 0..n {
        let x = i as f64;
        sx += x;
        sy += data[i];
        sxy += x * data[i];
        sx2 += x * x;
    }
    let denom = nf * sx2 - sx * sx;
    if denom.abs() < 1e-12 { return 0.0; }
    (nf * sxy - sx * sy) / denom
}

fn rolling_percentile(val: f64, window: &[f64]) -> f64 {
    if window.is_empty() { return 50.0; }
    let below = window.iter().filter(|&&x| x < val && !x.is_nan()).count();
    let valid = window.iter().filter(|&&x| !x.is_nan()).count();
    if valid == 0 { return 50.0; }
    below as f64 / valid as f64 * 100.0
}

/// Ana feature hesaplama fonksiyonu
pub fn compute_features_v2(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64],
) -> Vec<[f64; N_FEATURES_V2]> {
    let n = closes.len();
    let mut features = vec![[f64::NAN; N_FEATURES_V2]; n];

    if n < 300 { return features; }

    let micro = 12usize;   // 1h in 5m bars
    let mid = 48usize;     // 4h
    let macro_lb = 288usize; // 24h

    // ── Pre-compute arrays ──

    // Delta & CVD
    let delta: Vec<f64> = (0..n).map(|i| buy_vol[i] - sell_vol[i]).collect();
    let mut cvd = vec![0.0_f64; n];
    cvd[0] = delta[0];
    for i in 1..n { cvd[i] = cvd[i-1] + delta[i]; }

    // Total volume
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();

    // Imbalance raw
    let imb_raw: Vec<f64> = (0..n).map(|i| {
        let t = total_vol[i];
        if t > 0.0 { delta[i] / t } else { 0.0 }
    }).collect();

    // Rolling sums
    let cvd_sum_micro = rolling_sum_f(&delta, micro);
    let cvd_sum_macro = rolling_sum_f(&delta, macro_lb);
    let vol_sum_micro = rolling_sum_f(&total_vol, micro);

    // Rolling mean/std
    let (cvd_micro_mean, cvd_micro_std) = rolling_mean_std_f(&cvd_sum_micro, micro);
    let (cvd_macro_mean, cvd_macro_std) = rolling_mean_std_f(&cvd_sum_macro, macro_lb);
    let (vol_micro_mean, vol_micro_std) = rolling_mean_std_f(&vol_sum_micro, micro);

    // Imbalance EMA
    let imb_smooth = ema_f(&imb_raw, micro);

    // Price EMA for trend
    let price_ema_short = ema_f(closes, micro);
    let price_ema_mid = ema_f(closes, mid);
    let price_ema_long = ema_f(closes, macro_lb);

    // Price rolling mean/std for mean-reversion
    let (price_mean_24h, price_std_24h) = rolling_mean_std_f(closes, macro_lb);

    // ATR (5m, 14-period)
    let mut tr = vec![0.0_f64; n];
    if n > 0 { tr[0] = highs[0] - lows[0]; }
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
    }
    let atr_short = ema_f(&tr, 14);
    let atr_long = ema_f(&tr, macro_lb);

    // Bollinger Band width
    let mut bb_width = vec![f64::NAN; n];
    for i in macro_lb..n {
        if !price_std_24h[i].is_nan() && !price_mean_24h[i].is_nan() && price_mean_24h[i] > 0.0 {
            bb_width[i] = price_std_24h[i] * 2.0 / price_mean_24h[i] * 100.0;
        }
    }

    // Candle properties (per bar)
    let mut body_ratio = vec![f64::NAN; n];
    let mut upper_wick = vec![f64::NAN; n];
    let mut lower_wick = vec![f64::NAN; n];
    let mut candle_range = vec![0.0_f64; n];
    let mut candle_dir = vec![0i8; n]; // 1=green, -1=red, 0=doji

    for i in 0..n {
        let range = highs[i] - lows[i];
        candle_range[i] = range;
        if range > 0.0 {
            let body = (closes[i] - if i > 0 { closes[i-1] } else { closes[i] }).abs();
            body_ratio[i] = body / range;
            if closes[i] >= (if i > 0 { closes[i-1] } else { closes[i] }) {
                // Green-ish
                upper_wick[i] = (highs[i] - closes[i]) / range;
                lower_wick[i] = ((if i > 0 { closes[i-1] } else { closes[i] }) - lows[i]).max(0.0) / range;
            } else {
                upper_wick[i] = (highs[i] - (if i > 0 { closes[i-1] } else { closes[i] })).max(0.0) / range;
                lower_wick[i] = (closes[i] - lows[i]) / range;
            }
            candle_dir[i] = if closes[i] > (if i > 0 { closes[i-1] } else { closes[i] }) { 1 } else { -1 };
        }
    }

    // Average candle range (rolling)
    let (avg_range, _) = rolling_mean_std_f(&candle_range, mid);

    // CVD rolling for pre-sweep
    let cvd_change_2h: Vec<f64> = (0..n).map(|i| {
        if i >= 24 { cvd[i] - cvd[i - 24] } else { 0.0 }
    }).collect();
    let cvd_change_4h: Vec<f64> = (0..n).map(|i| {
        if i >= mid { cvd[i] - cvd[i - mid] } else { 0.0 }
    }).collect();
    let (cvd_2h_mean, cvd_2h_std) = rolling_mean_std_f(&cvd_change_2h, macro_lb);
    let (cvd_4h_mean, cvd_4h_std) = rolling_mean_std_f(&cvd_change_4h, macro_lb);

    // Volume 24h average for spike detection
    let vol_sum_24h = rolling_sum_f(&total_vol, macro_lb);

    // ── Build features ──

    for i in macro_lb..n {
        let mut f = [f64::NAN; N_FEATURES_V2];

        // 0: CVD zscore micro
        if cvd_micro_std[i] > 0.0 && !cvd_micro_std[i].is_nan() {
            f[0] = (cvd_sum_micro[i] - cvd_micro_mean[i]) / cvd_micro_std[i];
        } else { continue; }

        // 1: CVD zscore macro
        if cvd_macro_std[i] > 0.0 && !cvd_macro_std[i].is_nan() {
            f[1] = (cvd_sum_macro[i] - cvd_macro_mean[i]) / cvd_macro_std[i];
        } else { continue; }

        // 2: Imbalance smooth
        if !imb_smooth[i].is_nan() { f[2] = imb_smooth[i]; } else { continue; }

        // 3: CVD acceleration — CVD_micro change over last 12 bars
        if i >= micro * 2 {
            let cvd_now = cvd_sum_micro[i];
            let cvd_prev = cvd_sum_micro[i - micro];
            let cvd_prev2 = cvd_sum_micro[i - micro * 2];
            let vel_now = cvd_now - cvd_prev;
            let vel_prev = cvd_prev - cvd_prev2;
            f[3] = vel_now - vel_prev; // acceleration
            // Normalize by std
            if cvd_micro_std[i] > 0.0 {
                f[3] /= cvd_micro_std[i];
            }
        } else { f[3] = 0.0; }

        // 4: Imbalance-Price divergence
        // Fiyat yukari gidiyor ama imbalance dusuyor = negatif divergence
        if i >= micro {
            let price_dir = closes[i] - closes[i - micro];
            let imb_dir = imb_smooth[i] - imb_smooth[i - micro];
            // Normalize
            let price_norm = if price_std_24h[i] > 0.0 && !price_std_24h[i].is_nan() {
                price_dir / price_std_24h[i]
            } else { 0.0 };
            f[4] = price_norm * imb_dir.signum(); // pozitif = confirmation, negatif = divergence
        } else { f[4] = 0.0; }

        // 5: Price trend 1h (linreg slope)
        if i >= micro {
            let slice = &closes[i + 1 - micro..=i];
            f[5] = linreg_slope(slice) / closes[i] * 10000.0; // bps per bar
        } else { f[5] = 0.0; }

        // 6: Price trend 4h
        if i >= mid {
            let slice = &closes[i + 1 - mid..=i];
            f[6] = linreg_slope(slice) / closes[i] * 10000.0;
        } else { f[6] = 0.0; }

        // 7: Price trend 24h
        if i >= macro_lb {
            let slice = &closes[i + 1 - macro_lb..=i];
            f[7] = linreg_slope(slice) / closes[i] * 10000.0;
        } else { f[7] = 0.0; }

        // 8: Momentum ROC 1h
        if i >= micro && closes[i - micro] > 0.0 {
            f[8] = (closes[i] - closes[i - micro]) / closes[i - micro] * 100.0;
        } else { f[8] = 0.0; }

        // 9: Momentum ROC 4h
        if i >= mid && closes[i - mid] > 0.0 {
            f[9] = (closes[i] - closes[i - mid]) / closes[i - mid] * 100.0;
        } else { f[9] = 0.0; }

        // 10: Mean reversion z-score
        if price_std_24h[i] > 0.0 && !price_std_24h[i].is_nan() && !price_mean_24h[i].is_nan() {
            f[10] = (closes[i] - price_mean_24h[i]) / price_std_24h[i];
        } else { f[10] = 0.0; }

        // 11: Body ratio (son mum)
        f[11] = if !body_ratio[i].is_nan() { body_ratio[i] } else { 0.5 };

        // 12: Upper wick ratio
        f[12] = if !upper_wick[i].is_nan() { upper_wick[i] } else { 0.25 };

        // 13: Lower wick ratio
        f[13] = if !lower_wick[i].is_nan() { lower_wick[i] } else { 0.25 };

        // 14: Candle direction streak
        let mut streak = 0i32;
        let dir = candle_dir[i];
        if dir != 0 {
            streak = 1;
            let mut j = i as i64 - 1;
            while j >= 0 && candle_dir[j as usize] == dir {
                streak += 1;
                j -= 1;
                if streak >= 20 { break; }
            }
            streak *= dir as i32; // negative streak for red
        }
        f[14] = streak as f64;

        // 15: Range vs average
        if !avg_range[i].is_nan() && avg_range[i] > 0.0 {
            f[15] = candle_range[i] / avg_range[i];
        } else { f[15] = 1.0; }

        // 16: ATR percentile
        if i >= macro_lb && !atr_short[i].is_nan() {
            let window: Vec<f64> = atr_short[i + 1 - macro_lb..=i].to_vec();
            f[16] = rolling_percentile(atr_short[i], &window);
        } else { f[16] = 50.0; }

        // 17: Volatility ratio (short/long ATR)
        if !atr_short[i].is_nan() && !atr_long[i].is_nan() && atr_long[i] > 0.0 {
            f[17] = atr_short[i] / atr_long[i];
        } else { f[17] = 1.0; }

        // 18: BB width percentile
        if i >= macro_lb && !bb_width[i].is_nan() {
            let window: Vec<f64> = bb_width[i + 1 - macro_lb..=i].iter().filter(|x| !x.is_nan()).cloned().collect();
            f[18] = rolling_percentile(bb_width[i], &window);
        } else { f[18] = 50.0; }

        // 19: Range expansion (son 3 mum / onceki 3 mum)
        if i >= 6 {
            let recent: f64 = candle_range[i-2..=i].iter().sum();
            let prev: f64 = candle_range[i-5..=i-3].iter().sum();
            f[19] = if prev > 0.0 { recent / prev } else { 1.0 };
        } else { f[19] = 1.0; }

        // 20: Vol zscore micro
        if vol_micro_std[i] > 0.0 && !vol_micro_std[i].is_nan() {
            f[20] = (vol_sum_micro[i] - vol_micro_mean[i]) / vol_micro_std[i];
        } else { f[20] = 0.0; }

        // 21: Vol spike (current bar vol / 24h avg bar vol)
        if vol_sum_24h[i] > 0.0 {
            let avg_bar_vol = vol_sum_24h[i] / macro_lb as f64;
            f[21] = total_vol[i] / avg_bar_vol;
        } else { f[21] = 1.0; }

        // 22: Vol trend (12-bar volume linreg slope)
        if i >= micro {
            let vol_slice = &total_vol[i + 1 - micro..=i];
            let avg_vol = vol_slice.iter().sum::<f64>() / micro as f64;
            f[22] = if avg_vol > 0.0 { linreg_slope(vol_slice) / avg_vol } else { 0.0 };
        } else { f[22] = 0.0; }

        // 23: Higher high count (son 6 mum)
        let lookback_hh = 6usize.min(i);
        let mut hh_count = 0u32;
        for j in 1..=lookback_hh {
            if i >= j && highs[i - j + 1] > highs[i - j] { hh_count += 1; }
        }
        f[23] = hh_count as f64;

        // 24: Lower low count (son 6 mum)
        let mut ll_count = 0u32;
        for j in 1..=lookback_hh {
            if i >= j && lows[i - j + 1] < lows[i - j] { ll_count += 1; }
        }
        f[24] = ll_count as f64;

        // 25: Swing position (fiyatin 24h range icindeki yeri 0-100)
        if i >= macro_lb {
            let window_high = highs[i + 1 - macro_lb..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let window_low = lows[i + 1 - macro_lb..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = window_high - window_low;
            f[25] = if range > 0.0 { (closes[i] - window_low) / range * 100.0 } else { 50.0 };
        } else { f[25] = 50.0; }

        // 26: Pre-CVD 2h (z-scored)
        if !cvd_2h_std[i].is_nan() && cvd_2h_std[i] > 0.0 && !cvd_2h_mean[i].is_nan() {
            f[26] = (cvd_change_2h[i] - cvd_2h_mean[i]) / cvd_2h_std[i];
        } else { f[26] = 0.0; }

        // 27: Pre-CVD 4h (z-scored)
        if !cvd_4h_std[i].is_nan() && cvd_4h_std[i] > 0.0 && !cvd_4h_mean[i].is_nan() {
            f[27] = (cvd_change_4h[i] - cvd_4h_mean[i]) / cvd_4h_std[i];
        } else { f[27] = 0.0; }

        features[i] = f;
    }

    features
}

// ── Helper fonksiyonlar (standalone versiyonlar) ──

fn rolling_sum_f(data: &[f64], period: usize) -> Vec<f64> {
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

fn rolling_mean_std_f(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
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

fn ema_f(data: &[f64], period: usize) -> Vec<f64> {
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

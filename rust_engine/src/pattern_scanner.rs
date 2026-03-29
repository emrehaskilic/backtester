/// Pattern Scanner — istatistiksel pattern tarama.
/// Parametre optimizasyonu yok. Pattern ya var ya yok.
/// Her pattern için: koşul oluştuğunda sonraki N bar fiyat değişimi istatistiği.

use rayon::prelude::*;

const MIN_BARS: usize = 200;

// ── Horizon'lar: 1h, 2h, 4h, 8h, 24h (3m bar cinsinden) ──
const HORIZONS: [usize; 5] = [20, 40, 80, 160, 480];
const HORIZON_NAMES: [&str; 5] = ["1h", "2h", "4h", "8h", "24h"];

// ── Yardımcı fonksiyonlar ──

fn ema(data: &[f64], period: usize) -> Vec<f64> {
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

fn rolling_mean(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    let mut sum = 0.0;
    for i in 0..period { sum += data[i]; }
    out[period - 1] = sum / period as f64;
    for i in period..n {
        sum += data[i] - data[i - period];
        out[i] = sum / period as f64;
    }
    out
}

fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n < period { return out; }
    for i in (period - 1)..n {
        let slice = &data[i + 1 - period..=i];
        let mean: f64 = slice.iter().sum::<f64>() / period as f64;
        let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        out[i] = var.sqrt();
    }
    out
}

fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    let mut tr = vec![0.0_f64; n];
    tr[0] = highs[0] - lows[0];
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
    }
    let mut out = vec![f64::NAN; n];
    if n < period + 1 { return out; }
    let mut sum = 0.0;
    for i in 1..=period { sum += tr[i]; }
    out[period] = sum / period as f64;
    for i in (period+1)..n {
        let prev = out[i-1];
        out[i] = if prev.is_nan() { tr[i] } else { (prev * (period as f64 - 1.0) + tr[i]) / period as f64 };
    }
    out
}

fn keltner_channel(highs: &[f64], lows: &[f64], closes: &[f64], kc_len: usize, kc_mult: f64, atr_period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = closes.len();
    let ema_arr = ema(closes, kc_len);
    let atr_arr = atr(highs, lows, closes, atr_period);
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if !ema_arr[i].is_nan() && !atr_arr[i].is_nan() {
            upper[i] = ema_arr[i] + atr_arr[i] * kc_mult;
            lower[i] = ema_arr[i] - atr_arr[i] * kc_mult;
        }
    }
    (upper, lower)
}

// ── Pattern sonuç yapısı ──

#[derive(Clone, Debug)]
pub struct HorizonStats {
    pub horizon_name: String,
    pub horizon_bars: usize,
    pub count: usize,
    pub win_rate: f64,       // yön tutma oranı (%)
    pub avg_return: f64,     // ortalama getiri (%)
    pub median_return: f64,
    pub std_return: f64,
    pub sharpe: f64,
    pub max_favorable: f64,  // en iyi sonuç (%)
    pub max_adverse: f64,    // en kötü sonuç (%)
    pub monthly_consistency: f64, // kaç ayda pozitif (%)
}

#[derive(Clone, Debug)]
pub struct PatternResult {
    pub name: String,
    pub direction: String,   // "long" veya "short"
    pub occurrences: usize,
    pub horizons: Vec<HorizonStats>,
}

// ── İstatistik hesapla ──

fn compute_stats(returns: &[f64], direction: f64, timestamps: &[usize], total_bars: usize, horizon_bars: usize, horizon_name: &str) -> HorizonStats {
    let n = returns.len();
    if n == 0 {
        return HorizonStats {
            horizon_name: horizon_name.to_string(), horizon_bars, count: 0,
            win_rate: 0.0, avg_return: 0.0, median_return: 0.0, std_return: 0.0,
            sharpe: 0.0, max_favorable: 0.0, max_adverse: 0.0, monthly_consistency: 0.0,
        };
    }

    let directed: Vec<f64> = returns.iter().map(|r| r * direction).collect();
    let wins = directed.iter().filter(|&&r| r > 0.0).count();
    let wr = wins as f64 / n as f64 * 100.0;

    let avg = directed.iter().sum::<f64>() / n as f64;
    let mut sorted = directed.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] };

    let variance = directed.iter().map(|r| (r - avg).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    let sharpe = if std > 0.0 { avg / std } else { 0.0 };

    let max_fav = directed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_adv = directed.iter().cloned().fold(f64::INFINITY, f64::min);

    // Aylık tutarlılık: her 30 günlük pencerede pozitif mi?
    let bpd = 480usize; // bars per day
    let bpm = bpd * 30;
    let mut monthly_pos = 0;
    let mut monthly_total = 0;
    let n_months = (total_bars / bpm).max(1);
    for m in 0..n_months {
        let m_start = m * bpm;
        let m_end = (m + 1) * bpm;
        let month_returns: Vec<f64> = timestamps.iter().zip(directed.iter())
            .filter(|(&ts, _)| ts >= m_start && ts < m_end)
            .map(|(_, &r)| r)
            .collect();
        if !month_returns.is_empty() {
            monthly_total += 1;
            let month_avg: f64 = month_returns.iter().sum::<f64>() / month_returns.len() as f64;
            if month_avg > 0.0 { monthly_pos += 1; }
        }
    }
    let monthly_consistency = if monthly_total > 0 { monthly_pos as f64 / monthly_total as f64 * 100.0 } else { 0.0 };

    HorizonStats {
        horizon_name: horizon_name.to_string(), horizon_bars, count: n,
        win_rate: wr, avg_return: avg, median_return: median, std_return: std,
        sharpe, max_favorable: max_fav, max_adverse: max_adv, monthly_consistency,
    }
}

// ── Ana tarama fonksiyonu ──

pub fn scan_all_patterns(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> Vec<PatternResult> {
    let n = closes.len();
    let lookback = 20usize;

    // ── Pre-compute seriler ──
    // CVD
    let mut delta = vec![0.0_f64; n];
    let mut imbalance = vec![0.0_f64; n];
    for i in 0..n {
        delta[i] = buy_vol[i] - sell_vol[i];
        let total = buy_vol[i] + sell_vol[i];
        imbalance[i] = if total > 0.0 { delta[i] / total } else { 0.0 };
    }

    // Rolling CVD (cumulative delta over window)
    let mut cvd = vec![0.0_f64; n];
    for i in lookback..n {
        let mut sum = 0.0;
        for j in (i + 1 - lookback)..=i { sum += delta[j]; }
        cvd[i] = sum;
    }
    let cvd_mean = rolling_mean(&cvd, lookback);
    let cvd_std = rolling_std(&cvd, lookback);

    // Volume ratio
    let mut vol_ratio = vec![0.0_f64; n];
    for i in 0..n {
        let sv = sell_vol[i];
        vol_ratio[i] = if sv > 0.0 { buy_vol[i] / sv } else { 1.0 };
    }
    let vol_ratio_smooth = rolling_mean(&vol_ratio, lookback);

    // OI change
    let oi_lookback = 20usize; // ~1 saat
    let mut oi_change = vec![0.0_f64; n];
    for i in oi_lookback..n {
        if !oi[i].is_nan() && !oi[i - oi_lookback].is_nan() && oi[i - oi_lookback] > 0.0 {
            oi_change[i] = (oi[i] - oi[i - oi_lookback]) / oi[i - oi_lookback] * 100.0;
        }
    }

    // ATR
    let atr_arr = atr(highs, lows, closes, 14);

    // KC (default params)
    let (kc_upper, kc_lower) = keltner_channel(highs, lows, closes, 20, 2.0, 14);

    // Volume smooth
    let total_vol: Vec<f64> = (0..n).map(|i| buy_vol[i] + sell_vol[i]).collect();
    let vol_mean = rolling_mean(&total_vol, lookback);
    let vol_min_20: Vec<f64> = {
        let mut out = vec![f64::NAN; n];
        for i in (lookback - 1)..n {
            let mut min_v = f64::INFINITY;
            for j in (i + 1 - lookback)..=i {
                if total_vol[j] < min_v { min_v = total_vol[j]; }
            }
            out[i] = min_v;
        }
        out
    };

    // Price change (close-to-close over lookback)
    let mut price_chg = vec![0.0_f64; n];
    for i in lookback..n {
        if closes[i - lookback] > 0.0 {
            price_chg[i] = (closes[i] - closes[i - lookback]) / closes[i - lookback] * 100.0;
        }
    }

    // ── Pattern tanımları ve tarama ──
    let start = MIN_BARS.max(lookback * 2);
    let max_horizon = *HORIZONS.last().unwrap();

    // Her pattern için: (trigger_indices, direction)
    let mut patterns: Vec<(&str, &str, Vec<usize>)> = Vec::new();

    // Pattern 1 & 2: CVD spike
    let mut cvd_spike_long = Vec::new();
    let mut cvd_spike_short = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        if cvd_std[i].is_nan() || cvd_std[i] <= 0.0 || cvd_mean[i].is_nan() { continue; }
        let z = (cvd[i] - cvd_mean[i]) / cvd_std[i];
        if z > 2.0 { cvd_spike_long.push(i); }
        if z < -2.0 { cvd_spike_short.push(i); }
    }
    patterns.push(("CVD Spike", "long", cvd_spike_long));
    patterns.push(("CVD Spike", "short", cvd_spike_short));

    // Pattern 3 & 4: OI surge + price direction
    let mut oi_surge_up = Vec::new();
    let mut oi_surge_down = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        if oi_change[i] > 1.0 { // OI %1+ artış
            if closes[i] > closes[i - lookback] { oi_surge_up.push(i); }
            else { oi_surge_down.push(i); }
        }
    }
    patterns.push(("OI Surge + Price Up", "long", oi_surge_up));
    patterns.push(("OI Surge + Price Down", "short", oi_surge_down));

    // Pattern 5 & 6: Volume imbalance
    let mut vol_imb_long = Vec::new();
    let mut vol_imb_short = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        if vol_ratio_smooth[i].is_nan() { continue; }
        if vol_ratio_smooth[i] > 1.5 { vol_imb_long.push(i); }
        if vol_ratio_smooth[i] < 0.667 { vol_imb_short.push(i); }
    }
    patterns.push(("Volume Imbalance", "long", vol_imb_long));
    patterns.push(("Volume Imbalance", "short", vol_imb_short));

    // Pattern 7 & 8: KC band touch
    let mut kc_lower_touch = Vec::new();
    let mut kc_upper_touch = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        if kc_lower[i].is_nan() || kc_upper[i].is_nan() { continue; }
        if lows[i] <= kc_lower[i] && closes[i] > kc_lower[i] { kc_lower_touch.push(i); }
        if highs[i] >= kc_upper[i] && closes[i] < kc_upper[i] { kc_upper_touch.push(i); }
    }
    patterns.push(("KC Lower Touch", "long", kc_lower_touch));
    patterns.push(("KC Upper Touch", "short", kc_upper_touch));

    // Pattern 9 & 10: CVD divergence
    let mut cvd_div_bull = Vec::new();
    let mut cvd_div_bear = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        let price_down = price_chg[i] < -1.0; // fiyat %1+ düşmüş
        let price_up = price_chg[i] > 1.0;
        let cvd_up = cvd[i] > cvd[i - lookback]; // CVD yükseliyor
        let cvd_down = cvd[i] < cvd[i - lookback];

        if price_down && cvd_up { cvd_div_bull.push(i); }
        if price_up && cvd_down { cvd_div_bear.push(i); }
    }
    patterns.push(("CVD Divergence", "long", cvd_div_bull));
    patterns.push(("CVD Divergence", "short", cvd_div_bear));

    // Pattern 11: OI drop + volatility spike (likidasyonlar)
    let mut oi_drop_vol_spike = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        if oi_change[i] < -0.5 && !atr_arr[i].is_nan() && !atr_arr[i - lookback].is_nan() {
            if atr_arr[i] > atr_arr[i - lookback] * 1.5 {
                oi_drop_vol_spike.push(i);
            }
        }
    }
    // Bu pattern yönsüz — mean reversion varsayımıyla long
    patterns.push(("OI Drop + Vol Spike", "long", oi_drop_vol_spike));

    // Pattern 12: Volume dry-up → breakout
    let mut vol_dryup_breakout = Vec::new();
    for i in start..n.saturating_sub(max_horizon) {
        if vol_min_20[i].is_nan() || vol_mean[i].is_nan() { continue; }
        // Son 20 bar'ın min volume'u çok düşük ve şu anki volume 2x ortalama
        if total_vol[i] > vol_mean[i] * 2.0 && vol_min_20[i] < vol_mean[i] * 0.3 {
            if closes[i] > closes[i-1] { vol_dryup_breakout.push(i); }
        }
    }
    patterns.push(("Vol Dry-up Breakout", "long", vol_dryup_breakout));

    // ── Paralel istatistik hesapla ──
    let results: Vec<PatternResult> = patterns.par_iter().map(|(name, dir, triggers)| {
        let direction = if *dir == "long" { 1.0 } else { -1.0 };

        let horizons: Vec<HorizonStats> = HORIZONS.iter().zip(HORIZON_NAMES.iter()).map(|(&h, &hname)| {
            let mut returns = Vec::with_capacity(triggers.len());
            let mut timestamps = Vec::with_capacity(triggers.len());
            for &idx in triggers {
                if idx + h < n {
                    let ret = (closes[idx + h] - closes[idx]) / closes[idx] * 100.0;
                    returns.push(ret);
                    timestamps.push(idx);
                }
            }
            compute_stats(&returns, direction, &timestamps, n, h, hname)
        }).collect();

        PatternResult {
            name: name.to_string(),
            direction: dir.to_string(),
            occurrences: triggers.len(),
            horizons,
        }
    }).collect();

    results
}

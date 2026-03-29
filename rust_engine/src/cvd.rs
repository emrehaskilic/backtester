/// CVD Order Flow Strategy — 30s mikro-mum + 3dk ana mum.
///
/// Katman 1: CVD Osilatör (30s, yön belirler)
///   - Rolling CVD normalized by volume (-1 to +1)
///   - EMA(Imbalance) (-1 to +1)
///   - Signal = weighted combination
///
/// Saf sinyal testi: DCA/TP yok, reversal ile kapanır.

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const TAKER_FEE: f64 = 0.0005;
const MIN_BARS: usize = 200;

pub struct CvdBacktestResult {
    pub net_pct: f64,
    pub balance: f64,
    pub total_trades: i32,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
}

/// EMA hesapla
fn ema_arr(data: &[f64], period: usize) -> Vec<f64> {
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

/// Saf CVD sinyal backtest (30s bar uzerinde)
pub fn run_cvd_backtest(
    closes: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    cvd_period: usize,
    imb_weight: f64,
    signal_threshold: f64,
) -> CvdBacktestResult {
    let n = closes.len();

    // Delta ve Imbalance hesapla
    let mut delta = vec![0.0_f64; n];
    let mut imbalance = vec![0.0_f64; n];

    for i in 0..n {
        delta[i] = buy_vol[i] - sell_vol[i];
        let total = buy_vol[i] + sell_vol[i];
        imbalance[i] = if total > 0.0 { delta[i] / total } else { 0.0 };
    }

    // EMA(Imbalance, cvd_period)
    let smooth_imb = ema_arr(&imbalance, cvd_period);

    // Signal hesapla
    let mut signal = vec![0.0_f64; n];

    for i in cvd_period..n {
        // Rolling CVD (osilatör — sonsuz drift yok)
        let mut rolling_delta_sum = 0.0;
        let mut rolling_vol_sum = 0.0;
        for k in (i + 1 - cvd_period)..=i {
            rolling_delta_sum += delta[k];
            rolling_vol_sum += buy_vol[k] + sell_vol[k];
        }
        let norm_cvd = if rolling_vol_sum > 0.0 {
            rolling_delta_sum / rolling_vol_sum
        } else {
            0.0
        };

        let imb_val = if smooth_imb[i].is_nan() { 0.0 } else { smooth_imb[i] };

        signal[i] = norm_cvd * (1.0 - imb_weight) + imb_val * imb_weight;
    }

    // Backtest
    let mut condition: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;

    let mut balance = INITIAL_BALANCE;
    let mut peak_balance = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut trade_count: i32 = 0;
    let mut win_count: i32 = 0;
    let mut loss_count: i32 = 0;

    let start = MIN_BARS.max(cvd_period + 1);

    for i in start..n {
        let sig = signal[i];

        let new_dir = if sig > signal_threshold { 1.0 }
                      else if sig < -signal_threshold { -1.0 }
                      else { 0.0 };

        // Dead zone
        if new_dir == 0.0 {
            if balance > peak_balance { peak_balance = balance; }
            if peak_balance > 0.0 {
                let dd = (peak_balance - balance) / peak_balance * 100.0;
                if dd > max_dd { max_dd = dd; }
            }
            continue;
        }

        // Reversal
        if new_dir != condition {
            // Mevcut pozisyonu kapat
            if condition != 0.0 && notional > 0.0 {
                let pnl_pct = if condition > 0.0 {
                    (closes[i] - entry_price) / entry_price * 100.0
                } else {
                    (entry_price - closes[i]) / entry_price * 100.0
                };
                let pnl = notional * pnl_pct / 100.0;
                let fee = notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl;
                total_fees += fee;
                trade_count += 1;
                if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
            }

            // Yeni pozisyon aç
            let margin = balance * MARGIN_RATIO;
            if balance >= margin && margin > 0.0 {
                condition = new_dir;
                entry_price = closes[i];
                notional = margin * LEVERAGE;
                let fee = notional * TAKER_FEE;
                balance -= fee;
                total_fees += fee;
            } else {
                condition = 0.0;
                notional = 0.0;
            }
        }

        if balance > peak_balance { peak_balance = balance; }
        if peak_balance > 0.0 {
            let dd = (peak_balance - balance) / peak_balance * 100.0;
            if dd > max_dd { max_dd = dd; }
        }
    }

    // Kalan pozisyonu kapat
    if condition != 0.0 && notional > 0.0 && n > 0 {
        let pnl_pct = if condition > 0.0 {
            (closes[n-1] - entry_price) / entry_price * 100.0
        } else {
            (entry_price - closes[n-1]) / entry_price * 100.0
        };
        let pnl = notional * pnl_pct / 100.0;
        let fee = notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        trade_count += 1;
        if pnl > 0.0 { win_count += 1; } else { loss_count += 1; }
    }

    if balance > peak_balance { peak_balance = balance; }
    if peak_balance > 0.0 {
        let dd = (peak_balance - balance) / peak_balance * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    let net_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 * 100.0 } else { 0.0 };

    CvdBacktestResult {
        net_pct, balance, total_trades: trade_count, win_rate,
        max_drawdown: max_dd, total_pnl, total_fees,
    }
}

/// 1H Sweep Continuation Strategy v2
///
/// TP/SL/Timeout yok. Pozisyonu sadece ters sweep sinyali kapatir.
/// Continuation = trend devam, Reversal = trend donus.
///
/// | Pozisyon | Sweep | Pattern | Aksiyon |
/// |----------|-------|---------|---------|
/// | YOK | High | Cont | LONG ac |
/// | YOK | High | Rev | SHORT ac |
/// | YOK | Low | Cont | SHORT ac |
/// | YOK | Low | Rev | LONG ac |
/// | LONG | High | Cont | Tut |
/// | LONG | High | Rev | Kapat |
/// | LONG | Low | Cont | Kapat |
/// | LONG | Low | Rev | Tut |
/// | SHORT | High | Cont | Kapat |
/// | SHORT | High | Rev | Tut |
/// | SHORT | Low | Cont | Tut |
/// | SHORT | Low | Rev | Kapat |

use crate::sweep_miner::compute_5m_features;

const INITIAL_BALANCE: f64 = 10000.0;
const LEVERAGE: f64 = 25.0;
const MARGIN_RATIO: f64 = 1.0 / 40.0;
const TAKER_FEE: f64 = 0.0005;
const MIN_BARS: usize = 300;
const BP_1H: usize = 12;

// ── Pattern definitions ──
// Continuation patterns: sweep yonunde devam
// Reversal patterns: sweep yonunun tersi

struct PatternDef {
    features: Vec<usize>,
    quantiles: Vec<u8>,
}

// High sweep continuation patterns (yukari devam)
fn high_cont_patterns() -> Vec<PatternDef> {
    vec![
        // OI_change(3)=Q4 AND Vol_micro(4)=Q3 -> 88.1% cont, WF 6/7
        PatternDef { features: vec![3, 4], quantiles: vec![3, 2] },
        // OI_change(3)=Q4 AND ATR(6)=Q5 -> 85.3% cont, WF 3/4
        PatternDef { features: vec![3, 6], quantiles: vec![3, 4] },
    ]
}

// High sweep reversal patterns (asagi donus)
fn high_rev_patterns() -> Vec<PatternDef> {
    // Reversal = continuation'in tersi. Miner'dan:
    // High sweep base cont %50.9, rev %49.1
    // Reversal kosuller: OI dusuk, ATR dusuk, CVD zayif
    vec![
        // OI_change(3)=Q1 AND ATR(6)=Q1 -> dusuk OI + dusuk vol = fake breakout
        PatternDef { features: vec![3, 6], quantiles: vec![0, 0] },
        // Imbalance(5)=Q1 AND ATR(6)=Q1 -> satis baskisi + dusuk vol
        PatternDef { features: vec![5, 6], quantiles: vec![0, 0] },
    ]
}

// Low sweep continuation patterns (asagi devam)
fn low_cont_patterns() -> Vec<PatternDef> {
    vec![
        // CVD_micro(0)=Q1 AND Imbalance(5)=Q4 -> 97.2% cont, WF 4/4
        PatternDef { features: vec![0, 5], quantiles: vec![0, 3] },
        // CVD_macro(1)=Q3 AND Imbalance(5)=Q4 -> 88.3% cont, WF 6/7
        PatternDef { features: vec![1, 5], quantiles: vec![2, 3] },
    ]
}

// Low sweep reversal patterns (yukari donus)
fn low_rev_patterns() -> Vec<PatternDef> {
    // Reversal = yukari donus. OI artiyor, ATR yuksek
    vec![
        // OI_change(3)=Q5 AND ATR(6)=Q5 -> OI artiyor + yuksek vol = gercek donus
        PatternDef { features: vec![3, 6], quantiles: vec![4, 4] },
        // CVD_micro(0)=Q5 AND ATR(6)=Q4 -> alicilar guclu + vol yuksek
        PatternDef { features: vec![0, 6], quantiles: vec![4, 3] },
    ]
}

// ── Quantile helpers ──

fn compute_quantile_thresholds(features: &[[f64; 7]], end: usize) -> Vec<[f64; 4]> {
    let mut thresholds = vec![[0.0_f64; 4]; 7];
    for fi in 0..7 {
        let mut vals: Vec<f64> = (MIN_BARS..end).filter_map(|i| {
            let v = features[i][fi];
            if v.is_nan() { None } else { Some(v) }
        }).collect();
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for q in 0..4 {
            let idx = ((q + 1) as f64 / 5.0 * vals.len() as f64) as usize;
            thresholds[fi][q] = vals[idx.min(vals.len() - 1)];
        }
    }
    thresholds
}

fn get_quantile(val: f64, th: &[f64; 4]) -> u8 {
    let mut q = 0u8;
    for &t in th { if val > t { q += 1; } }
    q
}

fn check_patterns(feat: &[f64; 7], patterns: &[PatternDef], q_th: &[[f64; 4]; 7]) -> bool {
    for pat in patterns {
        let matched = pat.features.iter().zip(pat.quantiles.iter()).all(|(&fi, &q)| {
            !feat[fi].is_nan() && get_quantile(feat[fi], &q_th[fi]) == q
        });
        if matched { return true; }
    }
    false
}

// ── Signal type ──

#[derive(Clone, Copy, PartialEq)]
enum SweepSignal {
    None,
    HighCont,   // high sweep + continuation = yukari
    HighRev,    // high sweep + reversal = asagi
    LowCont,    // low sweep + continuation = asagi
    LowRev,     // low sweep + reversal = yukari
}

// ── Result ──

pub struct Strategy1Hv2Result {
    pub net_pct: f64,
    pub balance: f64,
    pub total_trades: i32,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub long_trades: i32,
    pub short_trades: i32,
    pub long_wins: i32,
    pub short_wins: i32,
    pub avg_trade_bars: f64,
    pub max_consecutive_loss: i32,
    pub weekly_returns: Vec<f64>,
    pub trade_log: Vec<TradeRecord>,
}

#[derive(Clone)]
pub struct TradeRecord {
    pub bar: usize,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_reason: String,
    pub pnl: f64,
    pub balance_after: f64,
    pub bars_held: usize,
}

// ── Main strategy ──

pub fn run_1h_v2_strategy(
    closes: &[f64], highs: &[f64], lows: &[f64],
    buy_vol: &[f64], sell_vol: &[f64], oi: &[f64],
) -> Strategy1Hv2Result {
    let n = closes.len();

    let (features, daily_atr) = compute_5m_features(closes, highs, lows, buy_vol, sell_vol, oi);

    let warmup = (288 * 30).min(n);
    let mut q_th = compute_quantile_thresholds(&features, warmup);
    // Convert to fixed array
    let mut q_arr = [[0.0_f64; 4]; 7];
    for fi in 0..7 { q_arr[fi] = q_th[fi]; }

    let h_cont = high_cont_patterns();
    let h_rev = high_rev_patterns();
    let l_cont = low_cont_patterns();
    let l_rev = low_rev_patterns();

    // 1H candle state
    let mut prev_high = f64::NEG_INFINITY;
    let mut prev_low = f64::INFINITY;
    let mut high_active = false;
    let mut low_active = false;
    let mut high_cooldown = 0usize;
    let mut low_cooldown = 0usize;
    let mut bars_below = 0usize;
    let mut bars_above = 0usize;

    // Position
    let mut cond: f64 = 0.0; // +1=long, -1=short, 0=flat
    let mut entry_price: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut entry_bar: usize = 0;

    // Accounting
    let mut balance = INITIAL_BALANCE;
    let mut peak = INITIAL_BALANCE;
    let mut max_dd: f64 = 0.0;
    let mut total_pnl: f64 = 0.0;
    let mut total_fees: f64 = 0.0;
    let mut total_trades: i32 = 0;
    let mut wins: i32 = 0;
    let mut long_trades: i32 = 0;
    let mut short_trades: i32 = 0;
    let mut long_wins: i32 = 0;
    let mut short_wins: i32 = 0;
    let mut total_trade_bars: usize = 0;
    let mut consec_loss: i32 = 0;
    let mut max_consec: i32 = 0;

    let mut trade_log: Vec<TradeRecord> = Vec::new();
    let mut weekly_returns: Vec<f64> = Vec::new();
    let mut week_start_bal = INITIAL_BALANCE;
    let bpw = 288 * 7;

    let start = MIN_BARS.max(warmup);

    for i in start..n {
        // Monthly quantile update
        if i > warmup && i % (288 * 30) == 0 {
            q_th = compute_quantile_thresholds(&features, i);
            for fi in 0..7 { q_arr[fi] = q_th[fi]; }
        }

        if features[i][0].is_nan() || daily_atr[i].is_nan() || daily_atr[i] <= 0.0 {
            continue;
        }

        let reset_dist = 0.5 * daily_atr[i];

        // ── Update 1H candle ──
        if i >= BP_1H && i % BP_1H == 0 {
            let ps = i - BP_1H;
            prev_high = f64::NEG_INFINITY;
            prev_low = f64::INFINITY;
            for j in ps..i {
                if highs[j] > prev_high { prev_high = highs[j]; }
                if lows[j] < prev_low { prev_low = lows[j]; }
            }
            high_active = true;
            low_active = true;
            high_cooldown = 0;
            low_cooldown = 0;
            bars_below = 0;
            bars_above = 0;
        }

        if prev_high == f64::NEG_INFINITY { continue; }

        // ── Detect sweep + pattern ──
        let mut signal = SweepSignal::None;

        // High sweep
        if high_active && i > 0 && highs[i] > prev_high && highs[i-1] <= prev_high {
            if i >= 1 && !features[i-1][0].is_nan() {
                let feat = &features[i-1];
                if check_patterns(feat, &h_cont, &q_arr) {
                    signal = SweepSignal::HighCont;
                } else if check_patterns(feat, &h_rev, &q_arr) {
                    signal = SweepSignal::HighRev;
                }
                high_active = false;
                high_cooldown = 0;
            }
        }

        // Low sweep (only if no high sweep signal this bar)
        if signal == SweepSignal::None && low_active && i > 0 && lows[i] < prev_low && lows[i-1] >= prev_low {
            if i >= 1 && !features[i-1][0].is_nan() {
                let feat = &features[i-1];
                if check_patterns(feat, &l_cont, &q_arr) {
                    signal = SweepSignal::LowCont;
                } else if check_patterns(feat, &l_rev, &q_arr) {
                    signal = SweepSignal::LowRev;
                }
                low_active = false;
                low_cooldown = 0;
            }
        }

        // ── Apply signal to position ──
        if signal != SweepSignal::None {
            // Determine action
            let should_close = match (cond as i32, &signal) {
                (1, SweepSignal::HighRev) => true,   // long + high rev = kapat
                (1, SweepSignal::LowCont) => true,   // long + low cont = kapat
                (-1, SweepSignal::HighCont) => true,  // short + high cont = kapat
                (-1, SweepSignal::LowRev) => true,    // short + low rev = kapat
                _ => false,
            };

            let new_dir = match &signal {
                SweepSignal::HighCont => 1.0,  // yukari devam = long
                SweepSignal::LowRev => 1.0,    // donus yukari = long
                SweepSignal::LowCont => -1.0,  // asagi devam = short
                SweepSignal::HighRev => -1.0,  // donus asagi = short
                _ => 0.0,
            };

            // Close if needed
            if should_close && cond != 0.0 && notional > 0.0 {
                let pnl_pct = if cond > 0.0 {
                    (closes[i] - entry_price) / entry_price * 100.0
                } else {
                    (entry_price - closes[i]) / entry_price * 100.0
                };
                let pnl = notional * pnl_pct / 100.0;
                let fee = notional * TAKER_FEE;
                balance += pnl - fee;
                total_pnl += pnl;
                total_fees += fee;
                total_trades += 1;
                total_trade_bars += i - entry_bar;

                if cond > 0.0 { long_trades += 1; } else { short_trades += 1; }
                if pnl > 0.0 {
                    wins += 1;
                    if cond > 0.0 { long_wins += 1; } else { short_wins += 1; }
                    consec_loss = 0;
                } else {
                    consec_loss += 1;
                    if consec_loss > max_consec { max_consec = consec_loss; }
                }

                let reason = match &signal {
                    SweepSignal::HighRev => "High Rev",
                    SweepSignal::HighCont => "High Cont",
                    SweepSignal::LowCont => "Low Cont",
                    SweepSignal::LowRev => "Low Rev",
                    _ => "Unknown",
                };

                trade_log.push(TradeRecord {
                    bar: entry_bar,
                    direction: if cond > 0.0 { "LONG".to_string() } else { "SHORT".to_string() },
                    entry_price, exit_price: closes[i],
                    exit_reason: reason.to_string(),
                    pnl, balance_after: balance,
                    bars_held: i - entry_bar,
                });

                cond = 0.0;
                notional = 0.0;
            }

            // Open new position if flat and signal gives direction
            if cond == 0.0 && new_dir != 0.0 {
                let margin = balance * MARGIN_RATIO;
                if balance >= margin && margin > 0.0 {
                    cond = new_dir;
                    entry_price = closes[i];
                    notional = margin * LEVERAGE;
                    entry_bar = i;
                    let fee = notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
            }

            // If closed and new signal is opposite direction, open immediately
            if cond == 0.0 && should_close && new_dir != 0.0 {
                let margin = balance * MARGIN_RATIO;
                if balance >= margin && margin > 0.0 {
                    cond = new_dir;
                    entry_price = closes[i];
                    notional = margin * LEVERAGE;
                    entry_bar = i;
                    let fee = notional * TAKER_FEE;
                    balance -= fee;
                    total_fees += fee;
                }
            }
        }

        // Reset logic
        if !high_active {
            if closes[i] < prev_high - reset_dist { bars_below += 1; } else { bars_below = 0; }
            high_cooldown += 1;
            if bars_below >= 12 { high_active = true; bars_below = 0; }
        }
        if !low_active {
            if closes[i] > prev_low + reset_dist { bars_above += 1; } else { bars_above = 0; }
            low_cooldown += 1;
            if bars_above >= 12 { low_active = true; bars_above = 0; }
        }

        // DD
        if balance > peak { peak = balance; }
        if peak > 0.0 {
            let dd = (peak - balance) / peak * 100.0;
            if dd > max_dd { max_dd = dd; }
        }

        // Weekly
        if i > start && (i - start) % bpw == 0 {
            let wr = (balance - week_start_bal) / week_start_bal * 100.0;
            weekly_returns.push(wr);
            week_start_bal = balance;
        }
    }

    // Close remaining
    if cond != 0.0 && notional > 0.0 && n > 0 {
        let pnl_pct = if cond > 0.0 {
            (closes[n-1] - entry_price) / entry_price * 100.0
        } else {
            (entry_price - closes[n-1]) / entry_price * 100.0
        };
        let pnl = notional * pnl_pct / 100.0;
        let fee = notional * TAKER_FEE;
        balance += pnl - fee;
        total_pnl += pnl;
        total_fees += fee;
        total_trades += 1;
        if cond > 0.0 { long_trades += 1; } else { short_trades += 1; }
        if pnl > 0.0 { wins += 1; if cond > 0.0 { long_wins += 1; } else { short_wins += 1; } }

        trade_log.push(TradeRecord {
            bar: entry_bar, direction: if cond > 0.0 { "LONG".to_string() } else { "SHORT".to_string() },
            entry_price, exit_price: closes[n-1], exit_reason: "END".to_string(),
            pnl, balance_after: balance, bars_held: n - 1 - entry_bar,
        });
    }

    if balance != week_start_bal {
        weekly_returns.push((balance - week_start_bal) / week_start_bal * 100.0);
    }

    if balance > peak { peak = balance; }
    if peak > 0.0 { let dd = (peak-balance)/peak*100.0; if dd > max_dd { max_dd = dd; } }

    let net = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100.0;
    let wr = if total_trades > 0 { wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };
    let avg = if total_trades > 0 { total_trade_bars as f64 / total_trades as f64 } else { 0.0 };

    Strategy1Hv2Result {
        net_pct: net, balance, total_trades, win_rate: wr, max_drawdown: max_dd,
        total_pnl, total_fees, long_trades, short_trades, long_wins, short_wins,
        avg_trade_bars: avg, max_consecutive_loss: max_consec, weekly_returns, trade_log,
    }
}

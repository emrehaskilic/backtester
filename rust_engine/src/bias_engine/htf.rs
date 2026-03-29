/// Bias Engine — Higher Timeframe Aggregation (Section 1.3 of spec)
///
/// Aggregates 5m bars into 1H, 2H, 4H, 8H, 1D bars.
/// Only fully closed HTF bars are emitted.

/// A single aggregated higher-timeframe bar.
pub struct HtfBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub buy_vol: f64,
    pub sell_vol: f64,
    /// First 5m bar index (inclusive)
    pub start_idx: usize,
    /// Last 5m bar index (inclusive)
    pub end_idx: usize,
}

/// A complete series for one timeframe.
pub struct HtfSeries {
    pub bars: Vec<HtfBar>,
    pub period_name: &'static str,
    /// Expected number of 5m bars per HTF bar
    pub period_bars: usize,
}

/// Aggregate 5m bars into one HTF series.
///
/// `period_ms` — duration of one HTF bar in milliseconds.
/// Bars are aligned to calendar boundaries via floor(timestamp / period_ms).
/// The last bar is dropped if it contains fewer than `period_bars` 5m bars
/// (i.e. it is incomplete).
pub fn aggregate_htf(
    timestamps: &[u64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
    period_ms: u64,
    period_name: &'static str,
    period_bars: usize,
) -> HtfSeries {
    let n = timestamps.len();
    let mut bars: Vec<HtfBar> = Vec::new();

    if n == 0 {
        return HtfSeries {
            bars,
            period_name,
            period_bars,
        };
    }

    let mut cur_period = timestamps[0] / period_ms;
    let mut b_open = open[0];
    let mut b_high = high[0];
    let mut b_low = low[0];
    let mut b_close = close[0];
    let mut b_bv = buy_vol[0];
    let mut b_sv = sell_vol[0];
    let mut b_start: usize = 0;

    for i in 1..n {
        let period = timestamps[i] / period_ms;

        if period != cur_period {
            // Emit completed bar
            bars.push(HtfBar {
                open: b_open,
                high: b_high,
                low: b_low,
                close: b_close,
                buy_vol: b_bv,
                sell_vol: b_sv,
                start_idx: b_start,
                end_idx: i - 1,
            });

            // Reset for new bar
            cur_period = period;
            b_open = open[i];
            b_high = high[i];
            b_low = low[i];
            b_close = close[i];
            b_bv = buy_vol[i];
            b_sv = sell_vol[i];
            b_start = i;
        } else {
            // Update current bar
            if high[i] > b_high {
                b_high = high[i];
            }
            if low[i] < b_low {
                b_low = low[i];
            }
            b_close = close[i];
            b_bv += buy_vol[i];
            b_sv += sell_vol[i];
        }
    }

    // Last bar — only push if it has the expected number of 5m bars (complete)
    let last_bar_count = n - b_start;
    if last_bar_count >= period_bars {
        bars.push(HtfBar {
            open: b_open,
            high: b_high,
            low: b_low,
            close: b_close,
            buy_vol: b_bv,
            sell_vol: b_sv,
            start_idx: b_start,
            end_idx: n - 1,
        });
    }

    HtfSeries {
        bars,
        period_name,
        period_bars,
    }
}

/// Build all 5 HTF series from 5m data.
///
/// Returns: \[1H, 2H, 4H, 8H, 1D\]
pub fn build_all_htf(
    timestamps: &[u64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    buy_vol: &[f64],
    sell_vol: &[f64],
) -> Vec<HtfSeries> {
    const MS_5M: u64 = 5 * 60 * 1000;

    let configs: [(u64, &str, usize); 5] = [
        (12 * MS_5M, "1H", 12),
        (24 * MS_5M, "2H", 24),
        (48 * MS_5M, "4H", 48),
        (96 * MS_5M, "8H", 96),
        (288 * MS_5M, "1D", 288),
    ];

    configs
        .iter()
        .map(|&(ms, name, bars)| {
            aggregate_htf(timestamps, open, high, low, close, buy_vol, sell_vol, ms, name, bars)
        })
        .collect()
}

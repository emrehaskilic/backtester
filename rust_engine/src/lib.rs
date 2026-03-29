/// rust_engine — PyO3 Python module for high-performance trading.
///
/// Modules:
///   indicators     — EMA, ATR, RSI, Keltner Channel
///   adaptive_pmax  — Adaptive PMax continuous
///   backtest       — Array-scan backtest (optimizer use)
///   engine         — Stateful TradingEngine (dry-run & live)

mod indicators;
mod adaptive_pmax;
mod backtest;
mod engine;
mod kama;
mod cvd;
mod cvd_oi;
mod optimizer;
mod pmax_pure;
mod pattern_scanner;
mod pattern_miner;
mod pattern_miner_v2;
mod sweep_miner;
mod sweep_strategy;
mod sweep_strategy_1h;
mod sweep_strategy_1h_v2;
mod features_v2;
mod sweep_analysis_v2;
mod sweep_candle_analysis;
mod sweep_candle_analysis_15m;
mod sweep_candle_miner;
mod sweep_candle_miner_15m;
mod sweep_candle_strategy;
mod sweep_candle_strategy_kc;
mod sweep_kc_multi_tf;
mod sweep_kc_optimizer;
mod sweep_kc_dca_optimizer;
mod sweep_dca_tp_optimizer;
mod bias_engine;
mod bias_kc_strategy;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Helper: PyReadonlyArray1 -> &[f64] (zero-copy, contiguous)
fn as_slice<'py>(arr: &'py PyReadonlyArray1<'py, f64>) -> &'py [f64] {
    arr.as_slice().expect("Array must be contiguous (use np.ascontiguousarray)")
}

// ═══════════════════════════════════════════════════════════════════
// Existing functions (backward compatible — optimizer uses these)
// ═══════════════════════════════════════════════════════════════════

/// Precompute fixed indicators for a fold (called once per fold).
#[pyfunction]
fn precompute_indicators<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    ema_filter_period: usize,
    kc_length: usize,
    kc_multiplier: f64,
    kc_atr_period: usize,
) -> PyResult<Py<PyDict>> {
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);

    let result = indicators::precompute_indicators(
        h, l, c, ema_filter_period, kc_length, kc_multiplier, kc_atr_period,
    );

    let dict = PyDict::new(py);
    dict.set_item("rsi_vals", PyArray1::from_vec(py, result.rsi_vals))?;
    dict.set_item("ema_filter", PyArray1::from_vec(py, result.ema_filter))?;
    dict.set_item("rsi_ema_vals", PyArray1::from_vec(py, result.rsi_ema_vals))?;
    dict.set_item("atr_vol", PyArray1::from_vec(py, result.atr_vol))?;
    dict.set_item("kc_upper_arr", PyArray1::from_vec(py, result.kc_upper))?;
    dict.set_item("kc_lower_arr", PyArray1::from_vec(py, result.kc_lower))?;

    Ok(dict.unbind())
}

/// Compute adaptive PMax continuous.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_adaptive_pmax<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    base_atr_period: usize,
    base_atr_multiplier: f64,
    base_ma_length: usize,
    lookback: usize,
    flip_window: usize,
    mult_base: f64,
    mult_scale: f64,
    ma_base: usize,
    ma_scale: f64,
    atr_base: usize,
    atr_scale: f64,
    update_interval: usize,
) -> PyResult<Py<PyDict>> {
    let s = as_slice(&src);
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);

    let result = adaptive_pmax::adaptive_pmax_continuous(
        s, h, l, c,
        base_atr_period, base_atr_multiplier, base_ma_length,
        lookback, flip_window,
        mult_base, mult_scale,
        ma_base, ma_scale,
        atr_base, atr_scale,
        update_interval,
    );

    let dict = PyDict::new(py);
    dict.set_item("pmax_line", PyArray1::from_vec(py, result.pmax_line))?;
    dict.set_item("mavg", PyArray1::from_vec(py, result.mavg))?;
    dict.set_item("direction", PyArray1::from_vec(py, result.direction))?;

    Ok(dict.unbind())
}

/// Run backtest with pre-computed PMax + indicators (fixed margin).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_backtest<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    pmax_line: PyReadonlyArray1<'py, f64>,
    mavg_arr: PyReadonlyArray1<'py, f64>,
    direction_arr: PyReadonlyArray1<'py, f64>,
    rsi_vals: PyReadonlyArray1<'py, f64>,
    ema_filter: PyReadonlyArray1<'py, f64>,
    rsi_ema_vals: PyReadonlyArray1<'py, f64>,
    atr_vol: PyReadonlyArray1<'py, f64>,
    kc_upper: PyReadonlyArray1<'py, f64>,
    kc_lower: PyReadonlyArray1<'py, f64>,
    rsi_overbought: f64,
    max_dca_steps: i32,
    tp_close_percent: f64,
) -> PyResult<Py<PyDict>> {
    let result = backtest::run_backtest_with_pmax(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&pmax_line), as_slice(&mavg_arr), as_slice(&direction_arr),
        as_slice(&rsi_vals), as_slice(&ema_filter), as_slice(&rsi_ema_vals),
        as_slice(&atr_vol), as_slice(&kc_upper), as_slice(&kc_lower),
        rsi_overbought, max_dca_steps, tp_close_percent,
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    dict.set_item("hard_stop_count", result.hard_stop_count)?;

    Ok(dict.unbind())
}

/// Run backtest with dynamic margin (Kelly/DynComp).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_backtest_dynamic<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    pmax_line: PyReadonlyArray1<'py, f64>,
    mavg_arr: PyReadonlyArray1<'py, f64>,
    direction_arr: PyReadonlyArray1<'py, f64>,
    rsi_vals: PyReadonlyArray1<'py, f64>,
    ema_filter: PyReadonlyArray1<'py, f64>,
    rsi_ema_vals: PyReadonlyArray1<'py, f64>,
    atr_vol: PyReadonlyArray1<'py, f64>,
    kc_upper: PyReadonlyArray1<'py, f64>,
    kc_lower: PyReadonlyArray1<'py, f64>,
    rsi_overbought: f64,
    max_dca_steps: i32,
    tp_close_percent: f64,
    base_margin_pct: f64,
    tier1_threshold: f64,
    tier1_pct: f64,
    tier2_threshold: f64,
    tier2_pct: f64,
) -> PyResult<Py<PyDict>> {
    let result = backtest::run_backtest_dynamic_margin(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&pmax_line), as_slice(&mavg_arr), as_slice(&direction_arr),
        as_slice(&rsi_vals), as_slice(&ema_filter), as_slice(&rsi_ema_vals),
        as_slice(&atr_vol), as_slice(&kc_upper), as_slice(&kc_lower),
        rsi_overbought, max_dca_steps, tp_close_percent,
        base_margin_pct, tier1_threshold, tier1_pct, tier2_threshold, tier2_pct,
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    dict.set_item("hard_stop_count", result.hard_stop_count)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Multi-TF Strategy: 1H Sweep Direction + 3m KC Trading
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_multi_tf_strategy_py<'py>(
    py: Python<'py>,
    // 5m data
    closes_5m: PyReadonlyArray1<'py, f64>,
    highs_5m: PyReadonlyArray1<'py, f64>,
    lows_5m: PyReadonlyArray1<'py, f64>,
    buy_vol_5m: PyReadonlyArray1<'py, f64>,
    sell_vol_5m: PyReadonlyArray1<'py, f64>,
    oi_5m: PyReadonlyArray1<'py, f64>,
    timestamps_5m: PyReadonlyArray1<'py, u64>,
    // 3m data
    closes_3m: PyReadonlyArray1<'py, f64>,
    highs_3m: PyReadonlyArray1<'py, f64>,
    lows_3m: PyReadonlyArray1<'py, f64>,
    timestamps_3m: PyReadonlyArray1<'py, u64>,
) -> PyResult<Py<PyDict>> {
    let ts5 = timestamps_5m.as_slice().expect("timestamps_5m must be contiguous");
    let ts3 = timestamps_3m.as_slice().expect("timestamps_3m must be contiguous");

    let r = sweep_kc_multi_tf::run_multi_tf_strategy(
        as_slice(&closes_5m), as_slice(&highs_5m), as_slice(&lows_5m),
        as_slice(&buy_vol_5m), as_slice(&sell_vol_5m), as_slice(&oi_5m), ts5,
        as_slice(&closes_3m), as_slice(&highs_3m), as_slice(&lows_3m), ts3,
    );

    let dict = PyDict::new(py);
    dict.set_item("total_weeks", r.total_weeks)?;
    dict.set_item("weekly_pnl", PyArray1::from_vec(py, r.weekly_pnl))?;
    dict.set_item("weekly_pnl_pct", PyArray1::from_vec(py, r.weekly_pnl_pct))?;
    dict.set_item("weekly_trades", PyArray1::from_vec(py, r.weekly_trades.iter().map(|&x| x as f64).collect()))?;
    dict.set_item("weekly_wins", PyArray1::from_vec(py, r.weekly_wins.iter().map(|&x| x as f64).collect()))?;
    dict.set_item("weekly_max_dd", PyArray1::from_vec(py, r.weekly_max_dd))?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("total_wins", r.total_wins)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("avg_weekly_pnl", r.avg_weekly_pnl)?;
    dict.set_item("avg_weekly_pnl_pct", r.avg_weekly_pnl_pct)?;
    dict.set_item("median_weekly_pnl_pct", r.median_weekly_pnl_pct)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("positive_weeks", r.positive_weeks)?;
    dict.set_item("negative_weeks", r.negative_weeks)?;
    dict.set_item("best_week_pct", r.best_week_pct)?;
    dict.set_item("worst_week_pct", r.worst_week_pct)?;
    dict.set_item("max_drawdown_within_week", r.max_drawdown_within_week)?;
    dict.set_item("total_dca_count", r.total_dca_count)?;
    dict.set_item("total_tp_count", r.total_tp_count)?;
    dict.set_item("total_signal_close", r.total_signal_close)?;
    dict.set_item("avg_hold_bars_3m", r.avg_hold_bars_3m)?;
    dict.set_item("max_consecutive_loss_weeks", r.max_consecutive_loss_weeks)?;

    // Trade log as list of dicts
    let trade_list: Vec<Py<PyDict>> = r.trades.iter().map(|t| {
        let td = PyDict::new(py);
        td.set_item("entry_price", t.entry_price).unwrap();
        td.set_item("exit_price", t.exit_price).unwrap();
        td.set_item("direction", t.direction).unwrap();
        td.set_item("pnl", t.pnl).unwrap();
        td.set_item("fee", t.fee).unwrap();
        td.set_item("entry_ts", t.entry_ts).unwrap();
        td.set_item("exit_ts", t.exit_ts).unwrap();
        td.set_item("exit_reason", &t.exit_reason).unwrap();
        td.set_item("dca_count", t.dca_count).unwrap();
        td.unbind()
    }).collect();
    dict.set_item("trades", trade_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Multi-TF Strategy with custom KC params
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_multi_tf_params_py<'py>(
    py: Python<'py>,
    closes_5m: PyReadonlyArray1<'py, f64>,
    highs_5m: PyReadonlyArray1<'py, f64>,
    lows_5m: PyReadonlyArray1<'py, f64>,
    buy_vol_5m: PyReadonlyArray1<'py, f64>,
    sell_vol_5m: PyReadonlyArray1<'py, f64>,
    oi_5m: PyReadonlyArray1<'py, f64>,
    timestamps_5m: PyReadonlyArray1<'py, u64>,
    closes_3m: PyReadonlyArray1<'py, f64>,
    highs_3m: PyReadonlyArray1<'py, f64>,
    lows_3m: PyReadonlyArray1<'py, f64>,
    timestamps_3m: PyReadonlyArray1<'py, u64>,
    kc_length: usize,
    kc_mult: f64,
    kc_atr_period: usize,
    max_dca: i32,
    dca_scale: f64,
    tp_levels: i32,
    tp_first_pct: f64,
) -> PyResult<Py<PyDict>> {
    let ts5 = timestamps_5m.as_slice().expect("contiguous");
    let ts3 = timestamps_3m.as_slice().expect("contiguous");

    let params = sweep_kc_multi_tf::KCParams {
        kc_length, kc_mult, kc_atr_period, max_dca,
        dca_scale, tp_levels, tp_first_pct,
    };

    let r = sweep_kc_multi_tf::run_multi_tf_with_params(
        as_slice(&closes_5m), as_slice(&highs_5m), as_slice(&lows_5m),
        as_slice(&buy_vol_5m), as_slice(&sell_vol_5m), as_slice(&oi_5m), ts5,
        as_slice(&closes_3m), as_slice(&highs_3m), as_slice(&lows_3m), ts3,
        &params,
    );

    let dict = PyDict::new(py);
    dict.set_item("total_weeks", r.total_weeks)?;
    dict.set_item("weekly_pnl", PyArray1::from_vec(py, r.weekly_pnl))?;
    dict.set_item("weekly_pnl_pct", PyArray1::from_vec(py, r.weekly_pnl_pct))?;
    dict.set_item("weekly_trades", PyArray1::from_vec(py, r.weekly_trades.iter().map(|&x| x as f64).collect()))?;
    dict.set_item("weekly_wins", PyArray1::from_vec(py, r.weekly_wins.iter().map(|&x| x as f64).collect()))?;
    dict.set_item("weekly_max_dd", PyArray1::from_vec(py, r.weekly_max_dd))?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("total_wins", r.total_wins)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("avg_weekly_pnl", r.avg_weekly_pnl)?;
    dict.set_item("avg_weekly_pnl_pct", r.avg_weekly_pnl_pct)?;
    dict.set_item("median_weekly_pnl_pct", r.median_weekly_pnl_pct)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("positive_weeks", r.positive_weeks)?;
    dict.set_item("negative_weeks", r.negative_weeks)?;
    dict.set_item("best_week_pct", r.best_week_pct)?;
    dict.set_item("worst_week_pct", r.worst_week_pct)?;
    dict.set_item("max_drawdown_within_week", r.max_drawdown_within_week)?;
    dict.set_item("total_dca_count", r.total_dca_count)?;
    dict.set_item("total_tp_count", r.total_tp_count)?;
    dict.set_item("total_signal_close", r.total_signal_close)?;
    dict.set_item("avg_hold_bars_3m", r.avg_hold_bars_3m)?;
    dict.set_item("max_consecutive_loss_weeks", r.max_consecutive_loss_weeks)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KC Walk-Forward Optimizer (35-fold)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_kc_wf_optimization_py<'py>(
    py: Python<'py>,
    closes_5m: PyReadonlyArray1<'py, f64>,
    highs_5m: PyReadonlyArray1<'py, f64>,
    lows_5m: PyReadonlyArray1<'py, f64>,
    buy_vol_5m: PyReadonlyArray1<'py, f64>,
    sell_vol_5m: PyReadonlyArray1<'py, f64>,
    oi_5m: PyReadonlyArray1<'py, f64>,
    timestamps_5m: PyReadonlyArray1<'py, u64>,
    closes_3m: PyReadonlyArray1<'py, f64>,
    highs_3m: PyReadonlyArray1<'py, f64>,
    lows_3m: PyReadonlyArray1<'py, f64>,
    timestamps_3m: PyReadonlyArray1<'py, u64>,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let ts5 = timestamps_5m.as_slice().expect("timestamps_5m contiguous");
    let ts3 = timestamps_3m.as_slice().expect("timestamps_3m contiguous");

    let r = sweep_kc_optimizer::run_wf_optimization(
        as_slice(&closes_5m), as_slice(&highs_5m), as_slice(&lows_5m),
        as_slice(&buy_vol_5m), as_slice(&sell_vol_5m), as_slice(&oi_5m), ts5,
        as_slice(&closes_3m), as_slice(&highs_3m), as_slice(&lows_3m), ts3,
        seed,
    );

    let dict = PyDict::new(py);
    dict.set_item("total_folds", r.total_folds)?;
    dict.set_item("positive_folds", r.positive_folds)?;
    dict.set_item("negative_folds", r.negative_folds)?;
    dict.set_item("avg_oos_pnl_pct", r.avg_oos_pnl_pct)?;
    dict.set_item("median_oos_pnl_pct", r.median_oos_pnl_pct)?;
    dict.set_item("best_fold_pct", r.best_fold_pct)?;
    dict.set_item("worst_fold_pct", r.worst_fold_pct)?;
    dict.set_item("total_oos_pnl", r.total_oos_pnl)?;
    dict.set_item("max_consec_neg", r.max_consec_neg)?;

    // Fold details as list of dicts
    let fold_list: Vec<Py<PyDict>> = r.folds.iter().map(|f| {
        let fd = PyDict::new(py);
        fd.set_item("fold_idx", f.fold_idx).unwrap();
        fd.set_item("kc_length", f.kc_length).unwrap();
        fd.set_item("kc_mult", f.kc_mult).unwrap();
        fd.set_item("kc_atr_period", f.kc_atr_period).unwrap();
        fd.set_item("max_dca", f.max_dca).unwrap();
        fd.set_item("train_score", f.train_score).unwrap();
        fd.set_item("train_weeks", f.train_weeks).unwrap();
        fd.set_item("test_pnl", f.test_pnl).unwrap();
        fd.set_item("test_pnl_pct", f.test_pnl_pct).unwrap();
        fd.set_item("test_trades", f.test_trades).unwrap();
        fd.set_item("test_wins", f.test_wins).unwrap();
        fd.set_item("test_max_dd", f.test_max_dd).unwrap();
        fd.set_item("test_tp", f.test_tp).unwrap();
        fd.set_item("test_signal_close", f.test_signal_close).unwrap();
        fd.set_item("test_dca", f.test_dca).unwrap();
        fd.unbind()
    }).collect();
    dict.set_item("folds", fold_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// DCA Walk-Forward Optimizer (KC sabit, sadece DCA grid)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_dca_wf_optimization_py<'py>(
    py: Python<'py>,
    closes_5m: PyReadonlyArray1<'py, f64>,
    highs_5m: PyReadonlyArray1<'py, f64>,
    lows_5m: PyReadonlyArray1<'py, f64>,
    buy_vol_5m: PyReadonlyArray1<'py, f64>,
    sell_vol_5m: PyReadonlyArray1<'py, f64>,
    oi_5m: PyReadonlyArray1<'py, f64>,
    timestamps_5m: PyReadonlyArray1<'py, u64>,
    closes_3m: PyReadonlyArray1<'py, f64>,
    highs_3m: PyReadonlyArray1<'py, f64>,
    lows_3m: PyReadonlyArray1<'py, f64>,
    timestamps_3m: PyReadonlyArray1<'py, u64>,
) -> PyResult<Py<PyDict>> {
    let ts5 = timestamps_5m.as_slice().expect("contiguous");
    let ts3 = timestamps_3m.as_slice().expect("contiguous");

    let r = sweep_kc_dca_optimizer::run_dca_wf_optimization(
        as_slice(&closes_5m), as_slice(&highs_5m), as_slice(&lows_5m),
        as_slice(&buy_vol_5m), as_slice(&sell_vol_5m), as_slice(&oi_5m), ts5,
        as_slice(&closes_3m), as_slice(&highs_3m), as_slice(&lows_3m), ts3,
    );

    let dict = PyDict::new(py);
    dict.set_item("total_folds", r.total_folds)?;
    dict.set_item("positive_folds", r.positive_folds)?;
    dict.set_item("negative_folds", r.negative_folds)?;
    dict.set_item("avg_oos_pnl_pct", r.avg_oos_pnl_pct)?;
    dict.set_item("median_oos_pnl_pct", r.median_oos_pnl_pct)?;
    dict.set_item("best_fold_pct", r.best_fold_pct)?;
    dict.set_item("worst_fold_pct", r.worst_fold_pct)?;
    dict.set_item("total_oos_pnl", r.total_oos_pnl)?;
    dict.set_item("max_consec_neg", r.max_consec_neg)?;
    dict.set_item("dca_choice_counts", PyArray1::from_vec(py,
        r.dca_choice_counts.iter().map(|&x| x as f64).collect()))?;

    let fold_list: Vec<Py<PyDict>> = r.folds.iter().map(|f| {
        let fd = PyDict::new(py);
        fd.set_item("fold_idx", f.fold_idx).unwrap();
        fd.set_item("best_dca", f.best_dca).unwrap();
        fd.set_item("train_score", f.train_score).unwrap();
        fd.set_item("all_dca_train_scores", PyArray1::from_vec(py, f.all_dca_train_scores.clone())).unwrap();
        fd.set_item("test_pnl", f.test_pnl).unwrap();
        fd.set_item("test_pnl_pct", f.test_pnl_pct).unwrap();
        fd.set_item("test_trades", f.test_trades).unwrap();
        fd.set_item("test_wins", f.test_wins).unwrap();
        fd.set_item("test_max_dd", f.test_max_dd).unwrap();
        fd.set_item("test_tp", f.test_tp).unwrap();
        fd.set_item("test_signal_close", f.test_signal_close).unwrap();
        fd.set_item("test_dca", f.test_dca).unwrap();
        fd.unbind()
    }).collect();
    dict.set_item("folds", fold_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Graduated DCA/TP Walk-Forward Optimizer
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_grad_wf_optimization_py<'py>(
    py: Python<'py>,
    closes_5m: PyReadonlyArray1<'py, f64>,
    highs_5m: PyReadonlyArray1<'py, f64>,
    lows_5m: PyReadonlyArray1<'py, f64>,
    buy_vol_5m: PyReadonlyArray1<'py, f64>,
    sell_vol_5m: PyReadonlyArray1<'py, f64>,
    oi_5m: PyReadonlyArray1<'py, f64>,
    timestamps_5m: PyReadonlyArray1<'py, u64>,
    closes_3m: PyReadonlyArray1<'py, f64>,
    highs_3m: PyReadonlyArray1<'py, f64>,
    lows_3m: PyReadonlyArray1<'py, f64>,
    timestamps_3m: PyReadonlyArray1<'py, u64>,
) -> PyResult<Py<PyDict>> {
    let ts5 = timestamps_5m.as_slice().expect("contiguous");
    let ts3 = timestamps_3m.as_slice().expect("contiguous");

    let r = sweep_dca_tp_optimizer::run_grad_wf_optimization(
        as_slice(&closes_5m), as_slice(&highs_5m), as_slice(&lows_5m),
        as_slice(&buy_vol_5m), as_slice(&sell_vol_5m), as_slice(&oi_5m), ts5,
        as_slice(&closes_3m), as_slice(&highs_3m), as_slice(&lows_3m), ts3,
    );

    let dict = PyDict::new(py);
    dict.set_item("total_folds", r.total_folds)?;
    dict.set_item("positive_folds", r.positive_folds)?;
    dict.set_item("negative_folds", r.negative_folds)?;
    dict.set_item("avg_oos_pnl_pct", r.avg_oos_pnl_pct)?;
    dict.set_item("median_oos_pnl_pct", r.median_oos_pnl_pct)?;
    dict.set_item("best_fold_pct", r.best_fold_pct)?;
    dict.set_item("worst_fold_pct", r.worst_fold_pct)?;
    dict.set_item("total_oos_pnl", r.total_oos_pnl)?;
    dict.set_item("max_consec_neg", r.max_consec_neg)?;
    dict.set_item("grid_size", r.grid_size)?;

    let fold_list: Vec<Py<PyDict>> = r.folds.iter().map(|f| {
        let fd = PyDict::new(py);
        fd.set_item("fold_idx", f.fold_idx).unwrap();
        fd.set_item("best_max_dca", f.best_max_dca).unwrap();
        fd.set_item("best_dca_scale", f.best_dca_scale).unwrap();
        fd.set_item("best_tp_levels", f.best_tp_levels).unwrap();
        fd.set_item("best_tp_first_pct", f.best_tp_first_pct).unwrap();
        fd.set_item("train_score", f.train_score).unwrap();
        fd.set_item("train_weeks", f.train_weeks).unwrap();
        fd.set_item("test_pnl", f.test_pnl).unwrap();
        fd.set_item("test_pnl_pct", f.test_pnl_pct).unwrap();
        fd.set_item("test_trades", f.test_trades).unwrap();
        fd.set_item("test_wins", f.test_wins).unwrap();
        fd.set_item("test_max_dd", f.test_max_dd).unwrap();
        fd.set_item("test_tp", f.test_tp).unwrap();
        fd.set_item("test_signal_close", f.test_signal_close).unwrap();
        fd.set_item("test_dca", f.test_dca).unwrap();
        fd.unbind()
    }).collect();
    dict.set_item("folds", fold_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// PyTradingEngine — Stateful engine for dry-run & live
// ═══════════════════════════════════════════════════════════════════

fn trade_event_to_dict(py: Python<'_>, e: &engine::TradeEvent) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", e.id)?;
    d.set_item("symbol", &e.symbol)?;
    d.set_item("side", &e.side)?;
    d.set_item("entry_price", e.entry_price)?;
    d.set_item("entry_time", e.entry_time)?;
    d.set_item("exit_price", e.exit_price)?;
    d.set_item("exit_time", e.exit_time)?;
    d.set_item("exit_reason", &e.exit_reason)?;
    d.set_item("qty_usdt", e.qty_usdt)?;
    d.set_item("leverage", e.leverage)?;
    d.set_item("pnl_usdt", e.pnl_usdt)?;
    d.set_item("pnl_pct", e.pnl_pct)?;
    d.set_item("fee_usdt", e.fee_usdt)?;
    d.set_item("tf_label", &e.tf_label)?;
    Ok(d.unbind())
}

fn position_to_dict(py: Python<'_>, p: &engine::PositionState) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("symbol", &p.symbol)?;
    d.set_item("side", if p.side == 1 { "LONG" } else if p.side == -1 { "SHORT" } else { "FLAT" })?;
    d.set_item("condition", p.side as f64)?;
    d.set_item("entry_time", p.entry_time)?;
    d.set_item("initial_entry_price", p.initial_entry_price)?;
    d.set_item("average_entry_price", p.avg_entry_price)?;
    d.set_item("entry_atr", p.entry_atr)?;
    d.set_item("margin_per_step", p.margin_per_step)?;
    d.set_item("total_position_notional", p.total_notional)?;
    d.set_item("total_fills", p.total_fills)?;
    d.set_item("dca_fills_count", p.dca_fills)?;
    d.set_item("dca_wave_sold", p.dca_wave_sold)?;
    d.set_item("hard_stop_price", p.hard_stop_price)?;
    d.set_item("pending_dca_price", p.pending_dca_price)?;
    d.set_item("pending_tp_price", p.pending_tp_price)?;
    Ok(d.unbind())
}

fn parse_config(d: &Bound<'_, PyDict>) -> PyResult<engine::TradingConfig> {
    let get_f = |key: &str, default: f64| -> f64 {
        d.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(default)
    };
    let get_i = |key: &str, default: i32| -> i32 {
        d.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<i32>().ok())
            .unwrap_or(default)
    };
    let get_b = |key: &str, default: bool| -> bool {
        d.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(default)
    };

    // Parse dyncomp tiers from list of (max_balance, comp_pct) tuples
    let mut dyncomp_tiers = Vec::new();
    if let Ok(Some(tiers_obj)) = d.get_item("dyncomp_tiers") {
        if let Ok(tiers_list) = tiers_obj.extract::<Vec<(f64, f64)>>() {
            for (max_bal, pct) in tiers_list {
                dyncomp_tiers.push(engine::DynCompTier {
                    max_balance: max_bal,
                    comp_pct: pct,
                });
            }
        }
    }

    Ok(engine::TradingConfig {
        initial_balance: get_f("initial_balance", 10000.0),
        leverage: get_f("leverage", 25.0),
        margin_per_trade: get_f("margin_per_trade", 300.0),
        maker_fee: get_f("maker_fee", 0.0002),
        taker_fee: get_f("taker_fee", 0.0005),
        max_dca_steps: get_i("max_dca_steps", 4),
        tp_close_pct: get_f("tp_close_pct", 0.50),
        dyncomp_enabled: get_b("dyncomp_enabled", false),
        dyncomp_tiers,
        pct_stop_enabled: get_b("pct_stop_enabled", false),
        pct_stop_loss: get_f("pct_stop_loss", 2.5),
        dyn_sl_enabled: get_b("dyn_sl_enabled", false),
        dyn_sl_atr_mult: get_f("dyn_sl_atr_mult", 2.5),
        dyn_sl_tighten: get_f("dyn_sl_tighten", 0.95),
        hard_stop_enabled: get_b("hard_stop_enabled", false),
        hard_stop_atr_mult: get_f("hard_stop_atr_mult", 5.0),
    })
}

#[pyclass]
struct PyTradingEngine {
    inner: engine::TradingEngine,
}

#[pymethods]
impl PyTradingEngine {
    #[new]
    fn new(config_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let config = parse_config(config_dict)?;
        Ok(Self {
            inner: engine::TradingEngine::new(config),
        })
    }

    /// PMax crossover signal → kill switch + new entry.
    /// Returns list of trade event dicts (reversal close if any).
    #[pyo3(signature = (symbol, side, price, atr, timestamp, tf_label, size_mult=1.0))]
    fn process_signal(
        &mut self,
        py: Python<'_>,
        symbol: &str,
        side: i8,
        price: f64,
        atr: f64,
        timestamp: i64,
        tf_label: &str,
        size_mult: f64,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let events = self.inner.process_signal(symbol, side, price, atr, timestamp, tf_label, size_mult);
        events.iter().map(|e| trade_event_to_dict(py, e)).collect()
    }

    /// Process one candle — KC DCA/TP + stop checks.
    /// Returns list of trade event dicts.
    #[pyo3(signature = (symbol, tf_label, high, low, close, timestamp, kc_upper, kc_lower, dyn_sl_atr=0.0))]
    fn process_candle(
        &mut self,
        py: Python<'_>,
        symbol: &str,
        tf_label: &str,
        high: f64,
        low: f64,
        close: f64,
        timestamp: i64,
        kc_upper: f64,
        kc_lower: f64,
        dyn_sl_atr: f64,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let events = self.inner.process_candle(
            symbol, tf_label, high, low, close, timestamp,
            kc_upper, kc_lower, dyn_sl_atr,
        );
        events.iter().map(|e| trade_event_to_dict(py, e)).collect()
    }

    /// Check if a position exists for symbol:tf_label.
    fn has_position(&self, symbol: &str, tf_label: &str) -> bool {
        self.inner.has_position(symbol, tf_label)
    }

    /// Get position state as dict (or None if no position).
    fn get_position(&self, py: Python<'_>, symbol: &str, tf_label: &str) -> PyResult<Option<Py<PyDict>>> {
        match self.inner.get_position(symbol, tf_label) {
            Some(p) => Ok(Some(position_to_dict(py, p)?)),
            None => Ok(None),
        }
    }

    /// Get all positions as dict of key → position dict.
    fn get_all_positions(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        for (key, pos) in self.inner.get_all_positions() {
            if pos.side != 0 {
                d.set_item(key, position_to_dict(py, pos)?)?;
            }
        }
        Ok(d.unbind())
    }

    /// Get wallet state as dict.
    fn get_wallet(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let w = self.inner.get_wallet();
        let d = PyDict::new(py);
        d.set_item("initial_balance", w.initial_balance)?;
        d.set_item("balance", w.balance)?;
        d.set_item("peak_balance", w.peak_balance)?;
        d.set_item("total_trades", w.total_trades)?;
        d.set_item("winning_trades", w.winning_trades)?;
        d.set_item("losing_trades", w.losing_trades)?;
        d.set_item("total_pnl", w.total_pnl)?;
        d.set_item("total_fees", w.total_fees)?;
        d.set_item("maker_fees", w.maker_fees)?;
        d.set_item("taker_fees", w.taker_fees)?;
        Ok(d.unbind())
    }

    /// Get all completed trades as list of dicts.
    fn get_trades(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner.get_trades().iter().map(|e| trade_event_to_dict(py, e)).collect()
    }

    /// Get engine stats as dict (same shape as Python Simulator.get_stats).
    fn get_stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let s = self.inner.get_stats();
        let d = PyDict::new(py);
        d.set_item("initial_balance", s.initial_balance)?;
        d.set_item("current_balance", s.current_balance)?;
        d.set_item("peak_balance", s.peak_balance)?;
        d.set_item("total_pnl", s.total_pnl)?;
        d.set_item("total_pnl_pct", s.total_pnl_pct)?;
        d.set_item("total_trades", s.total_trades)?;
        d.set_item("winning_trades", s.winning_trades)?;
        d.set_item("losing_trades", s.losing_trades)?;
        d.set_item("win_rate", s.win_rate)?;
        d.set_item("total_fees", s.total_fees)?;
        d.set_item("maker_fees", s.maker_fees)?;
        d.set_item("taker_fees", s.taker_fees)?;
        d.set_item("leverage", s.leverage)?;
        d.set_item("dynamic_comp_pct", s.dynamic_comp_pct)?;
        d.set_item("current_step_margin", s.current_step_margin)?;
        Ok(d.unbind())
    }
}

// ═══════════════════════════════════════════════════════════════════
// KC Lagged Backtest (kc[i-1]) — look-ahead fix
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_backtest_kc_lagged<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    pmax_line: PyReadonlyArray1<'py, f64>,
    mavg_arr: PyReadonlyArray1<'py, f64>,
    kc_upper: PyReadonlyArray1<'py, f64>,
    kc_lower: PyReadonlyArray1<'py, f64>,
    max_dca_steps: i32,
    tp_close_percent: f64,
) -> PyResult<Py<PyDict>> {
    let result = backtest::run_backtest_kc_lagged(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&pmax_line), as_slice(&mavg_arr),
        as_slice(&kc_upper), as_slice(&kc_lower),
        max_dca_steps, tp_close_percent,
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    dict.set_item("hard_stop_count", result.hard_stop_count)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KC Lagged + Graduated DCA
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_backtest_kc_lagged_graduated<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    pmax_line: PyReadonlyArray1<'py, f64>,
    mavg_arr: PyReadonlyArray1<'py, f64>,
    kc_upper: PyReadonlyArray1<'py, f64>,
    kc_lower: PyReadonlyArray1<'py, f64>,
    max_dca_steps: i32,
    tp_close_percent: f64,
    dca_m1: f64,
    dca_m2: f64,
    dca_m3: f64,
    dca_m4: f64,
) -> PyResult<Py<PyDict>> {
    let result = backtest::run_backtest_kc_lagged_graduated(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&pmax_line), as_slice(&mavg_arr),
        as_slice(&kc_upper), as_slice(&kc_lower),
        max_dca_steps, tp_close_percent,
        dca_m1, dca_m2, dca_m3, dca_m4,
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    dict.set_item("hard_stop_count", result.hard_stop_count)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KC Lagged + Graduated DCA + Graduated TP
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_backtest_kc_lagged_graduated_tp<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    pmax_line: PyReadonlyArray1<'py, f64>,
    mavg_arr: PyReadonlyArray1<'py, f64>,
    kc_upper: PyReadonlyArray1<'py, f64>,
    kc_lower: PyReadonlyArray1<'py, f64>,
    max_dca_steps: i32,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
    tp1: f64, tp2: f64, tp3: f64, tp4: f64,
) -> PyResult<Py<PyDict>> {
    let result = backtest::run_backtest_kc_lagged_graduated_tp(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&pmax_line), as_slice(&mavg_arr),
        as_slice(&kc_upper), as_slice(&kc_lower),
        max_dca_steps,
        dca_m1, dca_m2, dca_m3, dca_m4,
        tp1, tp2, tp3, tp4,
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    dict.set_item("hard_stop_count", result.hard_stop_count)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KAMA slope-based backtest
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_kama_backtest<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    kama_period: usize,
    kama_fast: usize,
    kama_slow: usize,
    slope_lookback: usize,
    slope_threshold: f64,
) -> PyResult<Py<PyDict>> {
    let result = kama::run_kama_backtest(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        kama_period, kama_fast, kama_slow,
        slope_lookback, slope_threshold,
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KAMA + KC lagged backtest
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_kama_kc_backtest<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32, tp_close_pct: f64,
) -> PyResult<Py<PyDict>> {
    let result = kama::run_kama_kc_backtest(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        kama_period, kama_fast, kama_slow,
        slope_lookback, slope_threshold,
        kc_length, kc_multiplier, kc_atr_period,
        max_dca_steps, tp_close_pct,
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// CVD + OI Order Flow backtest
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_cvd_oi_backtest<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
    cvd_period: usize, imb_weight: f64, cvd_threshold: f64,
    oi_period: usize, oi_threshold: f64,
) -> PyResult<Py<PyDict>> {
    let result = cvd_oi::run_cvd_oi_backtest(
        as_slice(&closes), as_slice(&buy_vol), as_slice(&sell_vol),
        as_slice(&oi),
        cvd_period, imb_weight, cvd_threshold, oi_period, oi_threshold,
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// CVD Order Flow backtest
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_cvd_backtest<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    cvd_period: usize,
    imb_weight: f64,
    signal_threshold: f64,
) -> PyResult<Py<PyDict>> {
    let result = cvd::run_cvd_backtest(
        as_slice(&closes), as_slice(&buy_vol), as_slice(&sell_vol),
        cvd_period, imb_weight, signal_threshold,
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KAMA + KC + Graduated DCA + Graduated TP
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_kama_kc_grad_dca_tp<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
    tp1: f64, tp2: f64, tp3: f64, tp4: f64,
) -> PyResult<Py<PyDict>> {
    let result = kama::run_kama_kc_grad_dca_tp(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
        kc_length, kc_multiplier, kc_atr_period, max_dca_steps,
        dca_m1, dca_m2, dca_m3, dca_m4, tp1, tp2, tp3, tp4,
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// KAMA + KC + Graduated DCA
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_kama_kc_grad_dca<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32, tp_close_pct: f64,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
) -> PyResult<Py<PyDict>> {
    let result = kama::run_kama_kc_grad_dca(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
        kc_length, kc_multiplier, kc_atr_period,
        max_dca_steps, tp_close_pct,
        dca_m1, dca_m2, dca_m3, dca_m4,
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// COMBINED: KAMA + PMax + CVD+OI + KC Lagged + Graduated DCA/TP
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_combined_backtest<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
    pmax_line: PyReadonlyArray1<'py, f64>, mavg_arr: PyReadonlyArray1<'py, f64>,
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    cvd_period: usize, imb_weight: f64, cvd_threshold: f64,
    oi_period: usize, oi_threshold: f64,
    kc_length: usize, kc_multiplier: f64, kc_atr_period: usize,
    max_dca_steps: i32,
    dca_m1: f64, dca_m2: f64, dca_m3: f64, dca_m4: f64,
    tp1: f64, tp2: f64, tp3: f64, tp4: f64,
) -> PyResult<Py<PyDict>> {
    let result = kama::run_combined_backtest(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
        as_slice(&pmax_line), as_slice(&mavg_arr),
        kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
        cvd_period, imb_weight, cvd_threshold, oi_period, oi_threshold,
        kc_length, kc_multiplier, kc_atr_period, max_dca_steps,
        dca_m1, dca_m2, dca_m3, dca_m4, tp1, tp2, tp3, tp4,
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", result.net_pct)?;
    dict.set_item("balance", result.balance)?;
    dict.set_item("total_trades", result.total_trades)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("total_pnl", result.total_pnl)?;
    dict.set_item("total_fees", result.total_fees)?;
    dict.set_item("tp_count", result.tp_count)?;
    dict.set_item("rev_count", result.rev_count)?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Pattern Miner (Quantile-based conditional discovery)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn mine_patterns<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
    train_end: usize,
) -> PyResult<Py<PyDict>> {
    let result = pattern_miner::run_pattern_mining(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
        train_end,
    );

    let dict = PyDict::new(py);
    dict.set_item("base_wr_long", result.base_wr_long)?;
    dict.set_item("base_wr_short", result.base_wr_short)?;
    dict.set_item("total_valid_long", result.total_valid_long)?;
    dict.set_item("total_valid_short", result.total_valid_short)?;
    dict.set_item("total_combos_tested", result.total_combos_tested)?;
    dict.set_item("pre_fdr_count", result.pre_fdr_count)?;
    dict.set_item("post_fdr_count", result.post_fdr_count)?;

    let patterns_list = pyo3::types::PyList::empty(py);
    for p in &result.patterns {
        let pdict = PyDict::new(py);

        // Condition string
        let conds: Vec<String> = p.features.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
            format!("{} = {}", pattern_miner::FEATURE_NAMES[fi], pattern_miner::QUANTILE_LABELS[qi as usize])
        }).collect();
        let cond_str = conds.join(" AND ");
        pdict.set_item("conditions", &cond_str)?;
        pdict.set_item("direction", &p.direction)?;
        pdict.set_item("sample", p.sample)?;
        pdict.set_item("wins", p.wins)?;
        pdict.set_item("wr", p.wr)?;
        pdict.set_item("avg_return", p.avg_return)?;
        pdict.set_item("sharpe", p.sharpe)?;
        pdict.set_item("monthly_consistency", p.monthly_consistency)?;
        pdict.set_item("p_value", p.p_value)?;
        pdict.set_item("score", p.score)?;

        let feat_list = pyo3::types::PyList::new(py, &p.features)?;
        pdict.set_item("feature_indices", feat_list)?;
        let quant_list = pyo3::types::PyList::new(py, p.quantiles.iter().map(|&q| q as i32).collect::<Vec<_>>())?;
        pdict.set_item("quantile_indices", quant_list)?;

        patterns_list.append(pdict)?;
    }
    dict.set_item("patterns", patterns_list)?;

    // Feature names
    let names = pyo3::types::PyList::new(py, &pattern_miner::FEATURE_NAMES)?;
    dict.set_item("feature_names", names)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Candle Sweep Strategy + KC
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_candle_kc_strategy_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_candle_strategy_kc::run_kc_strategy(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );
    let dict = PyDict::new(py);
    dict.set_item("net_pct", r.net_pct)?;
    dict.set_item("balance", r.balance)?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("max_drawdown", r.max_drawdown)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("long_trades", r.long_trades)?;
    dict.set_item("short_trades", r.short_trades)?;
    dict.set_item("long_wins", r.long_wins)?;
    dict.set_item("short_wins", r.short_wins)?;
    dict.set_item("tp_count", r.tp_count)?;
    dict.set_item("signal_close_count", r.signal_close_count)?;
    dict.set_item("dca_count", r.dca_count)?;
    dict.set_item("avg_hold_hours", r.avg_hold_hours)?;
    dict.set_item("max_consecutive_loss", r.max_consecutive_loss)?;
    dict.set_item("weekly_returns", PyArray1::from_vec(py, r.weekly_returns))?;
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Candle Sweep Strategy
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_candle_strategy_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_candle_strategy::run_candle_strategy(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", r.net_pct)?;
    dict.set_item("balance", r.balance)?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("max_drawdown", r.max_drawdown)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("long_trades", r.long_trades)?;
    dict.set_item("short_trades", r.short_trades)?;
    dict.set_item("long_wins", r.long_wins)?;
    dict.set_item("short_wins", r.short_wins)?;
    dict.set_item("avg_hold_hours", r.avg_hold_hours)?;
    dict.set_item("max_consecutive_loss", r.max_consecutive_loss)?;
    dict.set_item("weekly_returns", PyArray1::from_vec(py, r.weekly_returns))?;

    let tlist = pyo3::types::PyList::empty(py);
    for t in &r.trade_log {
        let td = PyDict::new(py);
        td.set_item("candle", t.candle_idx)?;
        td.set_item("dir", &t.direction)?;
        td.set_item("entry", t.entry_price)?;
        td.set_item("exit", t.exit_price)?;
        td.set_item("reason", &t.exit_reason)?;
        td.set_item("pnl", t.pnl)?;
        td.set_item("bal", t.balance_after)?;
        td.set_item("hours", t.hours_held)?;
        tlist.append(td)?;
    }
    dict.set_item("trades", tlist)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Sweep Miner V2 — 28 feature kapsamli analiz
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (closes, highs, lows, buy_vol, sell_vol, feat_bar_offset, bars_per_candle))]
fn run_sweep_miner_v2_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>,
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> PyResult<Py<PyDict>> {
    let r = sweep_analysis_v2::run_sweep_miner_v2(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol),
        feat_bar_offset, bars_per_candle,
    );

    let dict = PyDict::new(py);
    dict.set_item("high_base_cont_rate", r.high_base_cont_rate)?;
    dict.set_item("low_base_cont_rate", r.low_base_cont_rate)?;
    dict.set_item("high_total", r.high_total)?;
    dict.set_item("low_total", r.low_total)?;

    // Quantile rows
    let qlist = pyo3::types::PyList::empty(py);
    for qr in &r.quantile_rows {
        let d = PyDict::new(py);
        d.set_item("sweep_type", &qr.sweep_type)?;
        d.set_item("feature", &qr.feature_name)?;
        d.set_item("feature_idx", qr.feature_idx)?;
        d.set_item("quantile", qr.quantile)?;
        d.set_item("n", qr.n)?;
        d.set_item("cont_rate", qr.cont_rate)?;
        qlist.append(d)?;
    }
    dict.set_item("quantile_rows", qlist)?;

    // Patterns
    fn pats_to_list<'a>(py: Python<'a>, patterns: &[sweep_analysis_v2::PatternV2]) -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty(py);
        for p in patterns {
            let d = PyDict::new(py);
            let conds: Vec<String> = p.feature_indices.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
                format!("{} = Q{}", features_v2::FEATURE_NAMES_V2[fi], qi + 1)
            }).collect();
            d.set_item("conditions", conds.join(" AND "))?;
            d.set_item("sweep_type", &p.sweep_type)?;
            d.set_item("target", &p.target)?;
            d.set_item("n", p.n)?;
            d.set_item("target_count", p.target_count)?;
            d.set_item("target_rate", p.target_rate)?;
            d.set_item("p_value", p.p_value)?;
            d.set_item("score", p.score)?;
            d.set_item("wf_positive", p.wf_positive)?;
            d.set_item("wf_total", p.wf_total)?;
            d.set_item("wf_consistency", p.wf_consistency)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    dict.set_item("high_cont_patterns", pats_to_list(py, &r.high_cont_patterns)?)?;
    dict.set_item("high_rev_patterns", pats_to_list(py, &r.high_rev_patterns)?)?;
    dict.set_item("low_cont_patterns", pats_to_list(py, &r.low_cont_patterns)?)?;
    dict.set_item("low_rev_patterns", pats_to_list(py, &r.low_rev_patterns)?)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Cross-Asset Candle Sweep Miner (ETH candles + BTC features)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (candle_closes, candle_highs, candle_lows, feat_closes, feat_highs, feat_lows, feat_buy_vol, feat_sell_vol, feat_oi, feat_bar_offset, bars_per_candle))]
fn run_candle_miner_cross_py<'py>(
    py: Python<'py>,
    candle_closes: PyReadonlyArray1<'py, f64>, candle_highs: PyReadonlyArray1<'py, f64>, candle_lows: PyReadonlyArray1<'py, f64>,
    feat_closes: PyReadonlyArray1<'py, f64>, feat_highs: PyReadonlyArray1<'py, f64>, feat_lows: PyReadonlyArray1<'py, f64>,
    feat_buy_vol: PyReadonlyArray1<'py, f64>, feat_sell_vol: PyReadonlyArray1<'py, f64>, feat_oi: PyReadonlyArray1<'py, f64>,
    feat_bar_offset: usize,
    bars_per_candle: usize,
) -> PyResult<Py<PyDict>> {
    let r = sweep_candle_miner::run_candle_mining_cross(
        as_slice(&candle_closes), as_slice(&candle_highs), as_slice(&candle_lows),
        as_slice(&feat_closes), as_slice(&feat_highs), as_slice(&feat_lows),
        as_slice(&feat_buy_vol), as_slice(&feat_sell_vol), as_slice(&feat_oi),
        feat_bar_offset, bars_per_candle,
    );

    let dict = PyDict::new(py);
    dict.set_item("high_base_cont_rate", r.high_base_cont_rate)?;
    dict.set_item("low_base_cont_rate", r.low_base_cont_rate)?;
    dict.set_item("high_total", r.high_total)?;
    dict.set_item("low_total", r.low_total)?;

    let qlist = pyo3::types::PyList::empty(py);
    for qr in &r.quantile_rows {
        let d = PyDict::new(py);
        d.set_item("sweep_type", &qr.sweep_type)?;
        d.set_item("feature", &qr.feature_name)?;
        d.set_item("quantile", qr.quantile)?;
        d.set_item("n", qr.n)?;
        d.set_item("cont_rate", qr.cont_rate)?;
        qlist.append(d)?;
    }
    dict.set_item("quantile_rows", qlist)?;

    fn patterns_to_list_cross<'a>(py: Python<'a>, patterns: &[sweep_candle_miner::CandlePattern]) -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty(py);
        for p in patterns {
            let d = PyDict::new(py);
            let conds: Vec<String> = p.feature_indices.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
                format!("{} = Q{}", sweep_candle_analysis::FEATURE_NAMES[fi], qi + 1)
            }).collect();
            d.set_item("conditions", conds.join(" AND "))?;
            d.set_item("sweep_type", &p.sweep_type)?;
            d.set_item("target", &p.target)?;
            d.set_item("n", p.n)?;
            d.set_item("target_count", p.target_count)?;
            d.set_item("target_rate", p.target_rate)?;
            d.set_item("p_value", p.p_value)?;
            d.set_item("score", p.score)?;
            d.set_item("wf_positive", p.wf_positive)?;
            d.set_item("wf_total", p.wf_total)?;
            d.set_item("wf_consistency", p.wf_consistency)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    dict.set_item("high_cont_patterns", patterns_to_list_cross(py, &r.high_cont_patterns)?)?;
    dict.set_item("high_rev_patterns", patterns_to_list_cross(py, &r.high_rev_patterns)?)?;
    dict.set_item("low_cont_patterns", patterns_to_list_cross(py, &r.low_cont_patterns)?)?;
    dict.set_item("low_rev_patterns", patterns_to_list_cross(py, &r.low_rev_patterns)?)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Candle Sweep Miner
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (closes, highs, lows, buy_vol, sell_vol, oi, feat_bar_offset=None, bars_per_candle=None))]
fn run_candle_miner_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
    feat_bar_offset: Option<usize>,
    bars_per_candle: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let bp = bars_per_candle.unwrap_or(12); // default = 1H
    let offset = feat_bar_offset.unwrap_or(bp - 1); // default = son bar (mum kapanisi)
    let r = sweep_candle_miner::run_candle_mining_full(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
        offset, bp,
    );

    let dict = PyDict::new(py);
    dict.set_item("high_base_cont_rate", r.high_base_cont_rate)?;
    dict.set_item("low_base_cont_rate", r.low_base_cont_rate)?;
    dict.set_item("high_total", r.high_total)?;
    dict.set_item("low_total", r.low_total)?;

    // Quantile rows
    let qlist = pyo3::types::PyList::empty(py);
    for qr in &r.quantile_rows {
        let d = PyDict::new(py);
        d.set_item("sweep_type", &qr.sweep_type)?;
        d.set_item("feature", &qr.feature_name)?;
        d.set_item("quantile", qr.quantile)?;
        d.set_item("n", qr.n)?;
        d.set_item("cont_rate", qr.cont_rate)?;
        qlist.append(d)?;
    }
    dict.set_item("quantile_rows", qlist)?;

    // Patterns
    fn patterns_to_list<'a>(py: Python<'a>, patterns: &[sweep_candle_miner::CandlePattern]) -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty(py);
        for p in patterns {
            let d = PyDict::new(py);
            let conds: Vec<String> = p.feature_indices.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
                format!("{} = Q{}", sweep_candle_analysis::FEATURE_NAMES[fi], qi + 1)
            }).collect();
            d.set_item("conditions", conds.join(" AND "))?;
            d.set_item("sweep_type", &p.sweep_type)?;
            d.set_item("target", &p.target)?;
            d.set_item("n", p.n)?;
            d.set_item("target_count", p.target_count)?;
            d.set_item("target_rate", p.target_rate)?;
            d.set_item("p_value", p.p_value)?;
            d.set_item("score", p.score)?;
            d.set_item("wf_positive", p.wf_positive)?;
            d.set_item("wf_total", p.wf_total)?;
            d.set_item("wf_consistency", p.wf_consistency)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    dict.set_item("high_cont_patterns", patterns_to_list(py, &r.high_cont_patterns)?)?;
    dict.set_item("high_rev_patterns", patterns_to_list(py, &r.high_rev_patterns)?)?;
    dict.set_item("low_cont_patterns", patterns_to_list(py, &r.low_cont_patterns)?)?;
    dict.set_item("low_rev_patterns", patterns_to_list(py, &r.low_rev_patterns)?)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Candle Sweep Analysis
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (closes, highs, lows, buy_vol, sell_vol, oi, feat_bar_offset=None, bars_per_candle=None))]
fn run_candle_analysis_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
    feat_bar_offset: Option<usize>,
    bars_per_candle: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let bp = bars_per_candle.unwrap_or(12); // default = 1H
    let offset = feat_bar_offset.unwrap_or(bp - 1); // default = son bar (mum kapanisi)
    let r = sweep_candle_analysis::run_candle_analysis_full(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
        offset, bp,
    );

    let dict = PyDict::new(py);
    dict.set_item("total_1h_candles", r.total_1h_candles)?;
    dict.set_item("high_cont", r.high_cont_count)?;
    dict.set_item("high_rev", r.high_rev_count)?;
    dict.set_item("high_ambig", r.high_ambig_count)?;
    dict.set_item("low_cont", r.low_cont_count)?;
    dict.set_item("low_rev", r.low_rev_count)?;
    dict.set_item("low_ambig", r.low_ambig_count)?;
    dict.set_item("inside_bar", r.inside_bar_count)?;

    // Aftermath
    fn aftermath_to_list<'a>(py: Python<'a>, afts: &[sweep_candle_analysis::Aftermath]) -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty(py);
        for a in afts {
            let d = PyDict::new(py);
            d.set_item("horizon", a.horizon_bars)?;
            d.set_item("avg_return", a.avg_return)?;
            d.set_item("median_return", a.median_return)?;
            d.set_item("win_rate", a.win_rate)?;
            d.set_item("sample", a.sample)?;
            d.set_item("max_favorable", a.max_favorable)?;
            d.set_item("max_adverse", a.max_adverse)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    dict.set_item("high_cont_aftermath", aftermath_to_list(py, &r.high_cont_aftermath)?)?;
    dict.set_item("high_rev_aftermath", aftermath_to_list(py, &r.high_rev_aftermath)?)?;
    dict.set_item("low_cont_aftermath", aftermath_to_list(py, &r.low_cont_aftermath)?)?;
    dict.set_item("low_rev_aftermath", aftermath_to_list(py, &r.low_rev_aftermath)?)?;

    // Feature comparisons
    let flist = pyo3::types::PyList::empty(py);
    for fc in &r.feature_comparisons {
        let fd = PyDict::new(py);
        fd.set_item("sweep_type", &fc.sweep_type)?;
        fd.set_item("feature", &fc.feature_name)?;
        fd.set_item("cont_median", fc.cont_median)?;
        fd.set_item("rev_median", fc.rev_median)?;
        fd.set_item("ambig_median", fc.ambig_median)?;
        fd.set_item("p_value", fc.p_value)?;
        fd.set_item("significant", fc.significant)?;
        flist.append(fd)?;
    }
    dict.set_item("feature_comparisons", flist)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// 15M Candle Sweep Analysis
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_candle_analysis_15m_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_candle_analysis_15m::run_candle_analysis_15m(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("total_15m_candles", r.total_15m_candles)?;
    dict.set_item("high_cont", r.high_cont_count)?;
    dict.set_item("high_rev", r.high_rev_count)?;
    dict.set_item("high_ambig", r.high_ambig_count)?;
    dict.set_item("low_cont", r.low_cont_count)?;
    dict.set_item("low_rev", r.low_rev_count)?;
    dict.set_item("low_ambig", r.low_ambig_count)?;
    dict.set_item("inside_bar", r.inside_bar_count)?;

    // Aftermath
    fn aftermath_to_list_15m<'a>(py: Python<'a>, afts: &[sweep_candle_analysis_15m::Aftermath15m]) -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty(py);
        for a in afts {
            let d = PyDict::new(py);
            d.set_item("horizon", a.horizon_bars)?;
            d.set_item("avg_return", a.avg_return)?;
            d.set_item("median_return", a.median_return)?;
            d.set_item("win_rate", a.win_rate)?;
            d.set_item("sample", a.sample)?;
            d.set_item("max_favorable", a.max_favorable)?;
            d.set_item("max_adverse", a.max_adverse)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    dict.set_item("high_cont_aftermath", aftermath_to_list_15m(py, &r.high_cont_aftermath)?)?;
    dict.set_item("high_rev_aftermath", aftermath_to_list_15m(py, &r.high_rev_aftermath)?)?;
    dict.set_item("low_cont_aftermath", aftermath_to_list_15m(py, &r.low_cont_aftermath)?)?;
    dict.set_item("low_rev_aftermath", aftermath_to_list_15m(py, &r.low_rev_aftermath)?)?;

    // Feature comparisons
    let flist = pyo3::types::PyList::empty(py);
    for fc in &r.feature_comparisons {
        let fd = PyDict::new(py);
        fd.set_item("sweep_type", &fc.sweep_type)?;
        fd.set_item("feature", &fc.feature_name)?;
        fd.set_item("cont_median", fc.cont_median)?;
        fd.set_item("rev_median", fc.rev_median)?;
        fd.set_item("ambig_median", fc.ambig_median)?;
        fd.set_item("p_value", fc.p_value)?;
        fd.set_item("significant", fc.significant)?;
        flist.append(fd)?;
    }
    dict.set_item("feature_comparisons", flist)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// 15M Candle Sweep Miner
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_candle_miner_15m_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_candle_miner_15m::run_candle_mining_15m(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("high_base_cont_rate", r.high_base_cont_rate)?;
    dict.set_item("low_base_cont_rate", r.low_base_cont_rate)?;
    dict.set_item("high_total", r.high_total)?;
    dict.set_item("low_total", r.low_total)?;

    // Quantile rows
    let qlist = pyo3::types::PyList::empty(py);
    for qr in &r.quantile_rows {
        let d = PyDict::new(py);
        d.set_item("sweep_type", &qr.sweep_type)?;
        d.set_item("feature", &qr.feature_name)?;
        d.set_item("quantile", qr.quantile)?;
        d.set_item("n", qr.n)?;
        d.set_item("cont_rate", qr.cont_rate)?;
        qlist.append(d)?;
    }
    dict.set_item("quantile_rows", qlist)?;

    // Patterns
    fn patterns_to_list_15m<'a>(py: Python<'a>, patterns: &[sweep_candle_miner_15m::CandlePattern15m]) -> PyResult<Py<pyo3::types::PyList>> {
        let list = pyo3::types::PyList::empty(py);
        for p in patterns {
            let d = PyDict::new(py);
            let conds: Vec<String> = p.feature_indices.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
                format!("{} = Q{}", sweep_candle_analysis_15m::FEATURE_NAMES_15M[fi], qi + 1)
            }).collect();
            d.set_item("conditions", conds.join(" AND "))?;
            d.set_item("sweep_type", &p.sweep_type)?;
            d.set_item("target", &p.target)?;
            d.set_item("n", p.n)?;
            d.set_item("target_count", p.target_count)?;
            d.set_item("target_rate", p.target_rate)?;
            d.set_item("p_value", p.p_value)?;
            d.set_item("score", p.score)?;
            d.set_item("wf_positive", p.wf_positive)?;
            d.set_item("wf_total", p.wf_total)?;
            d.set_item("wf_consistency", p.wf_consistency)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    dict.set_item("high_cont_patterns", patterns_to_list_15m(py, &r.high_cont_patterns)?)?;
    dict.set_item("high_rev_patterns", patterns_to_list_15m(py, &r.high_rev_patterns)?)?;
    dict.set_item("low_cont_patterns", patterns_to_list_15m(py, &r.low_cont_patterns)?)?;
    dict.set_item("low_rev_patterns", patterns_to_list_15m(py, &r.low_rev_patterns)?)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// 1H Sweep Strategy v2
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_1h_sweep_v2_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_strategy_1h_v2::run_1h_v2_strategy(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", r.net_pct)?;
    dict.set_item("balance", r.balance)?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("max_drawdown", r.max_drawdown)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("long_trades", r.long_trades)?;
    dict.set_item("short_trades", r.short_trades)?;
    dict.set_item("long_wins", r.long_wins)?;
    dict.set_item("short_wins", r.short_wins)?;
    dict.set_item("avg_trade_bars", r.avg_trade_bars)?;
    dict.set_item("max_consecutive_loss", r.max_consecutive_loss)?;

    let weekly = PyArray1::from_vec(py, r.weekly_returns);
    dict.set_item("weekly_returns", weekly)?;

    let tlist = pyo3::types::PyList::empty(py);
    for t in &r.trade_log {
        let td = PyDict::new(py);
        td.set_item("bar", t.bar)?;
        td.set_item("direction", &t.direction)?;
        td.set_item("entry_price", t.entry_price)?;
        td.set_item("exit_price", t.exit_price)?;
        td.set_item("exit_reason", &t.exit_reason)?;
        td.set_item("pnl", t.pnl)?;
        td.set_item("balance_after", t.balance_after)?;
        td.set_item("bars_held", t.bars_held)?;
        tlist.append(td)?;
    }
    dict.set_item("trade_log", tlist)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// 1H Sweep Strategy
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_1h_sweep_strategy_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_strategy_1h::run_1h_strategy(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", r.net_pct)?;
    dict.set_item("balance", r.balance)?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("max_drawdown", r.max_drawdown)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("long_trades", r.long_trades)?;
    dict.set_item("short_trades", r.short_trades)?;
    dict.set_item("long_wins", r.long_wins)?;
    dict.set_item("short_wins", r.short_wins)?;
    dict.set_item("tp_count", r.tp_count)?;
    dict.set_item("sl_count", r.sl_count)?;
    dict.set_item("timeout_count", r.timeout_count)?;
    dict.set_item("avg_trade_bars", r.avg_trade_bars)?;
    dict.set_item("max_consecutive_loss", r.max_consecutive_loss)?;

    let weekly = PyArray1::from_vec(py, r.weekly_returns);
    dict.set_item("weekly_returns", weekly)?;

    // Trade log
    let tlist = pyo3::types::PyList::empty(py);
    for t in &r.trade_log {
        let td = PyDict::new(py);
        td.set_item("bar", t.bar)?;
        td.set_item("direction", if t.direction > 0.0 { "LONG" } else { "SHORT" })?;
        td.set_item("entry_price", t.entry_price)?;
        td.set_item("exit_price", t.exit_price)?;
        td.set_item("exit_type", &t.exit_type)?;
        td.set_item("pnl", t.pnl)?;
        td.set_item("balance_after", t.balance_after)?;
        tlist.append(td)?;
    }
    dict.set_item("trade_log", tlist)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Sweep Strategy
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_sweep_strategy_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_strategy::run_sweep_strategy(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("net_pct", r.net_pct)?;
    dict.set_item("balance", r.balance)?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("win_rate", r.win_rate)?;
    dict.set_item("max_drawdown", r.max_drawdown)?;
    dict.set_item("total_pnl", r.total_pnl)?;
    dict.set_item("total_fees", r.total_fees)?;
    dict.set_item("long_trades", r.long_trades)?;
    dict.set_item("short_trades", r.short_trades)?;
    dict.set_item("long_wins", r.long_wins)?;
    dict.set_item("short_wins", r.short_wins)?;
    dict.set_item("avg_trade_bars", r.avg_trade_bars)?;
    dict.set_item("max_consecutive_loss", r.max_consecutive_loss)?;

    let weekly = PyArray1::from_vec(py, r.weekly_returns);
    dict.set_item("weekly_returns", weekly)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Sweep Miner
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_sweep_mining_py<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = sweep_miner::run_sweep_mining(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let dict = PyDict::new(py);
    dict.set_item("total_sweep_events", r.total_sweep_events)?;
    dict.set_item("total_grid_tests", r.total_grid_tests)?;

    let results_list = pyo3::types::PyList::empty(py);
    for sr in &r.sweep_results {
        let sdict = PyDict::new(py);
        sdict.set_item("sweep_type", sr.sweep_type)?;
        sdict.set_item("sweep_type_name", &sr.sweep_type_name)?;
        sdict.set_item("total_events", sr.total_events)?;
        sdict.set_item("base_continuation_rate", sr.base_continuation_rate)?;
        sdict.set_item("base_reversal_rate", sr.base_reversal_rate)?;

        // Best grid
        let gdict = PyDict::new(py);
        gdict.set_item("mult", sr.best_grid.mult)?;
        gdict.set_item("timeout_bars", sr.best_grid.timeout_bars)?;
        gdict.set_item("n_total", sr.best_grid.n_total)?;
        gdict.set_item("n_continuation", sr.best_grid.n_continuation)?;
        gdict.set_item("n_reversal", sr.best_grid.n_reversal)?;
        gdict.set_item("n_timeout", sr.best_grid.n_timeout)?;
        gdict.set_item("continuation_rate", sr.best_grid.continuation_rate)?;
        gdict.set_item("reversal_rate", sr.best_grid.reversal_rate)?;
        gdict.set_item("timeout_rate", sr.best_grid.timeout_rate)?;
        gdict.set_item("separation", sr.best_grid.separation)?;
        sdict.set_item("best_grid", gdict)?;

        // Feature comparisons
        let flist = pyo3::types::PyList::empty(py);
        for fc in &sr.feature_comparisons {
            let fd = PyDict::new(py);
            fd.set_item("feature", &fc.feature_name)?;
            fd.set_item("cont_median", fc.cont_median)?;
            fd.set_item("rev_median", fc.rev_median)?;
            fd.set_item("p_value", fc.p_value)?;
            fd.set_item("significant", fc.significant)?;
            flist.append(fd)?;
        }
        sdict.set_item("feature_comparisons", flist)?;

        // Quantile analysis
        let qlist = pyo3::types::PyList::empty(py);
        for qr in &sr.quantile_analysis {
            let qd = PyDict::new(py);
            qd.set_item("feature_idx", qr.feature_idx)?;
            qd.set_item("feature_name", sweep_miner::FEATURE_NAMES[qr.feature_idx])?;
            qd.set_item("quantile", qr.quantile)?;
            qd.set_item("n", qr.n)?;
            qd.set_item("continuation_rate", qr.continuation_rate)?;
            qlist.append(qd)?;
        }
        sdict.set_item("quantile_analysis", qlist)?;

        // Top patterns
        let plist = pyo3::types::PyList::empty(py);
        for p in &sr.top_patterns {
            let pd = PyDict::new(py);
            let conds: Vec<String> = p.feature_indices.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
                format!("{} = Q{}", sweep_miner::FEATURE_NAMES[fi], qi + 1)
            }).collect();
            pd.set_item("conditions", conds.join(" AND "))?;
            pd.set_item("n", p.n)?;
            pd.set_item("continuation_rate", p.continuation_rate)?;
            pd.set_item("p_value", p.p_value)?;
            pd.set_item("score", p.score)?;
            pd.set_item("wf_positive", p.wf_positive_windows)?;
            pd.set_item("wf_total", p.wf_total_windows)?;
            pd.set_item("wf_consistency", p.wf_consistency)?;
            plist.append(pd)?;
        }
        sdict.set_item("top_patterns", plist)?;

        results_list.append(sdict)?;
    }
    dict.set_item("sweep_results", results_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Pattern Miner V2 (Comprehensive 7-task analysis)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn run_full_analysis<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
    train_end: usize,
) -> PyResult<Py<PyDict>> {
    let r = pattern_miner_v2::run_comprehensive_analysis(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
        train_end,
    );

    let dict = PyDict::new(py);
    dict.set_item("base_wr_long", r.base_wr_long)?;
    dict.set_item("base_wr_short", r.base_wr_short)?;
    dict.set_item("total_combos", r.total_combos)?;
    dict.set_item("regime_up_bars", r.regime_distribution[0])?;
    dict.set_item("regime_down_bars", r.regime_distribution[1])?;
    dict.set_item("regime_side_bars", r.regime_distribution[2])?;

    // Task 1: Symmetry
    dict.set_item("long_pattern_count", r.long_patterns.len())?;
    dict.set_item("short_pattern_count", r.short_patterns.len())?;

    let long_list = pyo3::types::PyList::empty(py);
    for p in r.long_patterns.iter().take(20) {
        let d = PyDict::new(py);
        let conds: Vec<String> = p.features.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
            format!("{} = {}", pattern_miner::FEATURE_NAMES[fi], pattern_miner::QUANTILE_LABELS[qi as usize])
        }).collect();
        d.set_item("conditions", conds.join(" AND "))?;
        d.set_item("wr", p.wr)?; d.set_item("sample", p.sample)?;
        d.set_item("avg_return", p.avg_return)?; d.set_item("sharpe", p.sharpe)?;
        d.set_item("monthly_consistency", p.monthly_consistency)?;
        long_list.append(d)?;
    }
    dict.set_item("long_patterns", long_list)?;

    let short_list = pyo3::types::PyList::empty(py);
    for p in r.short_patterns.iter().take(20) {
        let d = PyDict::new(py);
        let conds: Vec<String> = p.features.iter().zip(p.quantiles.iter()).map(|(&fi, &qi)| {
            format!("{} = {}", pattern_miner::FEATURE_NAMES[fi], pattern_miner::QUANTILE_LABELS[qi as usize])
        }).collect();
        d.set_item("conditions", conds.join(" AND "))?;
        d.set_item("wr", p.wr)?; d.set_item("sample", p.sample)?;
        d.set_item("avg_return", p.avg_return)?; d.set_item("sharpe", p.sharpe)?;
        d.set_item("monthly_consistency", p.monthly_consistency)?;
        short_list.append(d)?;
    }
    dict.set_item("short_patterns", short_list)?;

    // Task 2-5: Enriched patterns
    let enriched_list = pyo3::types::PyList::empty(py);
    for ep in &r.enriched {
        let d = PyDict::new(py);
        let conds: Vec<String> = ep.base.features.iter().zip(ep.base.quantiles.iter()).map(|(&fi, &qi)| {
            format!("{} = {}", pattern_miner::FEATURE_NAMES[fi], pattern_miner::QUANTILE_LABELS[qi as usize])
        }).collect();
        d.set_item("conditions", conds.join(" AND "))?;
        d.set_item("direction", &ep.base.direction)?;
        d.set_item("wr", ep.base.wr)?; d.set_item("sample", ep.base.sample)?;
        d.set_item("avg_return", ep.base.avg_return)?;
        d.set_item("sharpe", ep.base.sharpe)?;
        d.set_item("monthly_consistency", ep.base.monthly_consistency)?;
        // Regime
        d.set_item("wr_uptrend", ep.wr_uptrend)?; d.set_item("n_uptrend", ep.n_uptrend)?;
        d.set_item("wr_downtrend", ep.wr_downtrend)?; d.set_item("n_downtrend", ep.n_downtrend)?;
        d.set_item("wr_sideways", ep.wr_sideways)?; d.set_item("n_sideways", ep.n_sideways)?;
        d.set_item("regime_count", ep.regime_count)?;
        // Volatility
        d.set_item("wr_vol_low", ep.wr_vol_low)?; d.set_item("n_vol_low", ep.n_vol_low)?;
        d.set_item("wr_vol_mid", ep.wr_vol_mid)?; d.set_item("n_vol_mid", ep.n_vol_mid)?;
        d.set_item("wr_vol_high", ep.wr_vol_high)?; d.set_item("n_vol_high", ep.n_vol_high)?;
        // Generalized
        d.set_item("gen_wr", ep.gen_wr)?; d.set_item("gen_n", ep.gen_n)?;
        d.set_item("gen_wr_drop", ep.gen_wr_drop)?;
        // Strength
        d.set_item("strength_top30_wr", ep.strength_top30_wr)?;
        d.set_item("strength_top30_n", ep.strength_top30_n)?;
        enriched_list.append(d)?;
    }
    dict.set_item("enriched_patterns", enriched_list)?;

    // Task 6: Pullback
    let pb_list = pyo3::types::PyList::empty(py);
    for pb in &r.pullback_patterns {
        let d = PyDict::new(py);
        d.set_item("name", &pb.name)?;
        d.set_item("direction", &pb.direction)?;
        d.set_item("sample", pb.sample)?;
        d.set_item("wins", pb.wins)?;
        d.set_item("wr", pb.wr)?;
        let feats = pyo3::types::PyList::new(py, &pb.conditions_desc)?;
        d.set_item("features_used", feats)?;
        pb_list.append(d)?;
    }
    dict.set_item("pullback_patterns", pb_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Pattern Scanner
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
fn scan_patterns<'py>(
    py: Python<'py>,
    closes: PyReadonlyArray1<'py, f64>, highs: PyReadonlyArray1<'py, f64>, lows: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>, sell_vol: PyReadonlyArray1<'py, f64>, oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<pyo3::types::PyList>> {
    let results = pattern_scanner::scan_all_patterns(
        as_slice(&closes), as_slice(&highs), as_slice(&lows),
        as_slice(&buy_vol), as_slice(&sell_vol), as_slice(&oi),
    );

    let list = pyo3::types::PyList::empty(py);
    for pr in &results {
        let pdict = PyDict::new(py);
        pdict.set_item("name", &pr.name)?;
        pdict.set_item("direction", &pr.direction)?;
        pdict.set_item("occurrences", pr.occurrences)?;

        let hlist = pyo3::types::PyList::empty(py);
        for h in &pr.horizons {
            let hdict = PyDict::new(py);
            hdict.set_item("horizon", &h.horizon_name)?;
            hdict.set_item("horizon_bars", h.horizon_bars)?;
            hdict.set_item("count", h.count)?;
            hdict.set_item("win_rate", h.win_rate)?;
            hdict.set_item("avg_return", h.avg_return)?;
            hdict.set_item("median_return", h.median_return)?;
            hdict.set_item("std_return", h.std_return)?;
            hdict.set_item("sharpe", h.sharpe)?;
            hdict.set_item("max_favorable", h.max_favorable)?;
            hdict.set_item("max_adverse", h.max_adverse)?;
            hdict.set_item("monthly_consistency", h.monthly_consistency)?;
            hlist.append(hdict)?;
        }
        pdict.set_item("horizons", hlist)?;
        list.append(pdict)?;
    }
    Ok(list.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Pure PMax Backtest + Optimizer
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_pmax_wf_fold<'py>(
    py: Python<'py>,
    tr_closes: PyReadonlyArray1<'py, f64>, tr_highs: PyReadonlyArray1<'py, f64>, tr_lows: PyReadonlyArray1<'py, f64>,
    te_closes: PyReadonlyArray1<'py, f64>, te_highs: PyReadonlyArray1<'py, f64>, te_lows: PyReadonlyArray1<'py, f64>,
    n_trials: usize, seed: u64,
    warm_start_flat: PyReadonlyArray1<'py, f64>, n_warm: usize,
) -> PyResult<Py<PyDict>> {
    let tr_c = as_slice(&tr_closes).to_vec();
    let tr_h = as_slice(&tr_highs).to_vec();
    let tr_l = as_slice(&tr_lows).to_vec();
    let te_c = as_slice(&te_closes).to_vec();
    let te_h = as_slice(&te_highs).to_vec();
    let te_l = as_slice(&te_lows).to_vec();

    let ws_flat = as_slice(&warm_start_flat);
    let n_params = 12;
    let mut warm_start: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_warm {
        let start = i * n_params;
        if start + n_params <= ws_flat.len() {
            warm_start.push(ws_flat[start..start + n_params].to_vec());
        }
    }

    let results = pmax_pure::optimize_pmax_fold(
        &tr_c, &tr_h, &tr_l, &te_c, &te_h, &te_l,
        n_trials, seed, &warm_start,
    );

    let dict = PyDict::new(py);
    if let Some(best) = results.first() {
        if best.score > -999.0 {
            dict.set_item("status", "OK")?;
            dict.set_item("score", best.score)?;
            dict.set_item("net_pct", best.net_pct)?;
            dict.set_item("max_drawdown", best.max_drawdown)?;
            dict.set_item("win_rate", best.win_rate)?;
            dict.set_item("total_trades", best.total_trades)?;
            dict.set_item("total_fees", best.total_fees)?;

            let params_list = pyo3::types::PyList::new(py, &best.params)?;
            dict.set_item("params", params_list)?;

            let names = vec!["pmax_atr_period","pmax_atr_mult","pmax_ma_length","pmax_lookback",
                "pmax_flip_window","pmax_mult_base","pmax_mult_scale","pmax_ma_base",
                "pmax_ma_scale","pmax_atr_base","pmax_atr_scale","pmax_update_interval"];
            let names_list = pyo3::types::PyList::new(py, &names)?;
            dict.set_item("param_names", names_list)?;

            let mut top5_flat: Vec<f64> = Vec::new();
            for r in results.iter().take(5) {
                if r.score > -999.0 { top5_flat.extend(&r.params); }
            }
            let top5_arr = PyArray1::from_vec(py, top5_flat);
            dict.set_item("top5_params", top5_arr)?;
            dict.set_item("n_top5", results.iter().take(5).filter(|r| r.score > -999.0).count())?;
        } else {
            dict.set_item("status", "FAIL")?;
        }
    } else {
        dict.set_item("status", "FAIL")?;
    }
    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Rust-native TPE Optimizer (Rayon parallel)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_wf_fold<'py>(
    py: Python<'py>,
    // Train data
    tr_closes: PyReadonlyArray1<'py, f64>, tr_highs: PyReadonlyArray1<'py, f64>, tr_lows: PyReadonlyArray1<'py, f64>,
    tr_buy_vol: PyReadonlyArray1<'py, f64>, tr_sell_vol: PyReadonlyArray1<'py, f64>, tr_oi: PyReadonlyArray1<'py, f64>,
    // Test data
    te_closes: PyReadonlyArray1<'py, f64>, te_highs: PyReadonlyArray1<'py, f64>, te_lows: PyReadonlyArray1<'py, f64>,
    te_buy_vol: PyReadonlyArray1<'py, f64>, te_sell_vol: PyReadonlyArray1<'py, f64>, te_oi: PyReadonlyArray1<'py, f64>,
    // KAMA fixed
    kama_period: usize, kama_fast: usize, kama_slow: usize,
    slope_lookback: usize, slope_threshold: f64,
    // Optimizer config
    n_trials: usize,
    seed: u64,
    // Warm-start (flat array: n_warm * 29 params)
    warm_start_flat: PyReadonlyArray1<'py, f64>,
    n_warm: usize,
) -> PyResult<Py<PyDict>> {
    let opt = optimizer::CombinedOptimizer::new(n_trials);

    // Parse warm-start
    let ws_flat = as_slice(&warm_start_flat);
    let n_params = 29;
    let mut warm_start: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_warm {
        let start = i * n_params;
        if start + n_params <= ws_flat.len() {
            warm_start.push(ws_flat[start..start + n_params].to_vec());
        }
    }

    // Copy data before releasing GIL
    let tr_c = as_slice(&tr_closes).to_vec();
    let tr_h = as_slice(&tr_highs).to_vec();
    let tr_l = as_slice(&tr_lows).to_vec();
    let tr_bv = as_slice(&tr_buy_vol).to_vec();
    let tr_sv = as_slice(&tr_sell_vol).to_vec();
    let tr_o = as_slice(&tr_oi).to_vec();
    let te_c = as_slice(&te_closes).to_vec();
    let te_h = as_slice(&te_highs).to_vec();
    let te_l = as_slice(&te_lows).to_vec();
    let te_bv = as_slice(&te_buy_vol).to_vec();
    let te_sv = as_slice(&te_sell_vol).to_vec();
    let te_o = as_slice(&te_oi).to_vec();

    // Run optimization (Rayon handles parallelism internally)
    let results = opt.optimize_fold(
        &tr_c, &tr_h, &tr_l, &tr_bv, &tr_sv, &tr_o,
        &te_c, &te_h, &te_l, &te_bv, &te_sv, &te_o,
        kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold,
        &warm_start, seed,
    );

    let dict = PyDict::new(py);
    if let Some(best) = results.first() {
        if best.score > -999.0 {
            dict.set_item("status", "OK")?;
            dict.set_item("score", best.score)?;
            dict.set_item("net_pct", best.net_pct)?;
            dict.set_item("max_drawdown", best.max_drawdown)?;
            dict.set_item("win_rate", best.win_rate)?;
            dict.set_item("total_trades", best.total_trades)?;
            dict.set_item("tp_count", best.tp_count)?;
            dict.set_item("rev_count", best.rev_count)?;
            dict.set_item("total_fees", best.total_fees)?;

            // Params as list
            let params_list = pyo3::types::PyList::new(py, &best.params)?;
            dict.set_item("params", params_list)?;

            // Param names
            let specs = optimizer::build_param_specs();
            let names = pyo3::types::PyList::new(py,
                specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>())?;
            dict.set_item("param_names", names)?;

            // Top-5 params for warm-start (flat)
            let mut top5_flat: Vec<f64> = Vec::new();
            for r in results.iter().take(5) {
                if r.score > -999.0 {
                    top5_flat.extend(&r.params);
                }
            }
            let top5_arr = PyArray1::from_vec(py, top5_flat);
            dict.set_item("top5_params", top5_arr)?;
            dict.set_item("n_top5", results.iter().take(5).filter(|r| r.score > -999.0).count())?;
        } else {
            dict.set_item("status", "FAIL")?;
        }
    } else {
        dict.set_item("status", "FAIL")?;
    }

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine — Continuous Bias System (Steps 1–2)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bias_engine_step1_2<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, u64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let ts = timestamps.as_slice().expect("timestamps contiguous");
    let o = as_slice(&open);
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);
    let bv = as_slice(&buy_vol);
    let sv = as_slice(&sell_vol);
    let oi_s = as_slice(&oi);

    let r = bias_engine::analyze(ts, o, h, l, c, bv, sv, oi_s);

    let dict = PyDict::new(py);
    dict.set_item("n_bars", r.n_bars)?;
    dict.set_item("n_valid_bars", r.n_valid_bars)?;
    dict.set_item("n_quantized_bars", r.n_quantized_bars)?;
    dict.set_item("warmup_bars", r.warmup_bars)?;

    // HTF bar counts
    let htf = PyDict::new(py);
    for (name, count) in &r.htf_counts {
        htf.set_item(name.as_str(), *count)?;
    }
    dict.set_item("htf_bar_counts", htf)?;

    // State summary
    dict.set_item("total_states_with_data", r.state_counts.len())?;
    dict.set_item("depth1_active", r.depth1_active)?;
    dict.set_item("depth2_active", r.depth2_active)?;
    dict.set_item("depth3_active", r.depth3_active)?;

    // Feature arrays (7 arrays for verification)
    let feat = PyDict::new(py);
    for i in 0..bias_engine::features::N_FEATURES {
        let name = bias_engine::features::FEATURE_NAMES[i];
        feat.set_item(name, PyArray1::from_vec(py, r.features.get(i).to_vec()))?;
    }
    dict.set_item("features", feat)?;

    // Quintile arrays (7 arrays, as i32 for numpy compat)
    let quint = PyDict::new(py);
    for i in 0..bias_engine::features::N_FEATURES {
        let name = bias_engine::features::FEATURE_NAMES[i];
        let arr: Vec<i32> = r.quintiles[i].iter().map(|&x| x as i32).collect();
        quint.set_item(name, PyArray1::from_vec(py, arr))?;
    }
    dict.set_item("quintiles", quint)?;

    // Quintile distribution % (should be ~20% each if correct)
    let dist = PyDict::new(py);
    for f in 0..bias_engine::features::N_FEATURES {
        let name = bias_engine::features::FEATURE_NAMES[f];
        let q = &r.quintiles[f];
        let total_valid = q.iter().filter(|&&x| x > 0).count() as f64;
        if total_valid > 0.0 {
            let pcts: Vec<f64> = (1..=5u8)
                .map(|qv| q.iter().filter(|&&x| x == qv).count() as f64 / total_valid * 100.0)
                .collect();
            dist.set_item(name, PyArray1::from_vec(py, pcts))?;
        }
    }
    dict.set_item("quintile_distribution_pct", dist)?;

    // Top 30 states by frequency
    let mut top: Vec<_> = r.state_counts.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1));
    let top_list: Vec<Py<PyDict>> = top
        .iter()
        .take(30)
        .map(|(&key, &count)| {
            let d = PyDict::new(py);
            d.set_item("state", bias_engine::state::state_to_string(key))
                .unwrap();
            d.set_item("count", count).unwrap();
            d.set_item("depth", bias_engine::state::state_depth(key))
                .unwrap();
            d.unbind()
        })
        .collect();
    dict.set_item("top_states", top_list)?;

    // State count distribution per depth (histogram: how many states have N>100, N>500 etc.)
    let thresholds = [30u32, 50, 100, 200, 500, 1000];
    for depth in 1..=3u32 {
        let depth_counts: Vec<usize> = thresholds
            .iter()
            .map(|&t| {
                r.state_counts
                    .iter()
                    .filter(|(&k, &v)| bias_engine::state::state_depth(k) == depth && v >= t)
                    .count()
            })
            .collect();
        let key = format!("depth{}_count_above", depth);
        let labels: Vec<String> = thresholds.iter().map(|t| format!("N>={}", t)).collect();
        let d = PyDict::new(py);
        for (label, cnt) in labels.iter().zip(depth_counts.iter()) {
            d.set_item(label.as_str(), *cnt)?;
        }
        dict.set_item(key, d)?;
    }

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine — Full Pipeline (Steps 1–4)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bias_engine_full<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, u64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let ts = timestamps.as_slice().expect("timestamps contiguous");
    let o = as_slice(&open);
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);
    let bv = as_slice(&buy_vol);
    let sv = as_slice(&sell_vol);
    let oi_s = as_slice(&oi);

    let r = bias_engine::analyze_full(ts, o, h, l, c, bv, sv, oi_s);

    let dict = PyDict::new(py);

    // ── Step 1-2 summary ──
    dict.set_item("n_bars", r.n_bars)?;
    dict.set_item("n_valid_bars", r.n_valid_bars)?;
    dict.set_item("n_quantized_bars", r.n_quantized_bars)?;
    dict.set_item("warmup_bars", r.warmup_bars)?;

    let htf = PyDict::new(py);
    for (name, count) in &r.htf_counts {
        htf.set_item(name.as_str(), *count)?;
    }
    dict.set_item("htf_bar_counts", htf)?;

    dict.set_item("total_states_with_data", r.state_counts.len())?;
    dict.set_item("depth1_active", r.depth1_active)?;
    dict.set_item("depth2_active", r.depth2_active)?;
    dict.set_item("depth3_active", r.depth3_active)?;

    // ── Step 3 summary ──
    dict.set_item("baseline_bull_rate", r.baseline_bull_rate)?;
    dict.set_item("n_significant", r.n_significant)?;
    dict.set_item("sig_depth1", r.sig_depth1)?;
    dict.set_item("sig_depth2", r.sig_depth2)?;
    dict.set_item("sig_depth3", r.sig_depth3)?;

    // ── Step 4 summary ──
    dict.set_item("n_validated", r.n_validated)?;
    dict.set_item("val_depth1", r.val_depth1)?;
    dict.set_item("val_depth2", r.val_depth2)?;
    dict.set_item("val_depth3", r.val_depth3)?;

    // ── Significance funnel (how many pass each test) ──
    let funnel = PyDict::new(py);
    let total_states = r.state_counts.len();
    let n_100 = r.state_stats.values().filter(|s| s.n_total >= 100).count();
    let n_edge = r.state_stats.values().filter(|s| s.n_total >= 100 && s.bias.abs() >= 0.03).count();
    funnel.set_item("total_states_with_data", total_states)?;
    funnel.set_item("n_gte_100", n_100)?;
    funnel.set_item("n_gte_100_edge_003", n_edge)?;
    funnel.set_item("n_significant", r.n_significant)?;
    funnel.set_item("n_validated", r.n_validated)?;
    dict.set_item("significance_funnel", funnel)?;

    // ── Validated states detail ──
    let mut validated_states: Vec<_> = r
        .robustness
        .iter()
        .filter(|(_, rob)| rob.validated)
        .map(|(&key, rob)| {
            let stats = &r.state_stats[&key];
            (key, stats, rob)
        })
        .collect();
    // Sort by absolute bias descending
    validated_states.sort_by(|a, b| {
        b.1.bias
            .abs()
            .partial_cmp(&a.1.bias.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let val_list: Vec<Py<PyDict>> = validated_states
        .iter()
        .map(|&(key, stats, rob)| {
            let d = PyDict::new(py);
            d.set_item("state", bias_engine::state::state_to_string(key)).unwrap();
            d.set_item("depth", bias_engine::state::state_depth(key)).unwrap();
            d.set_item("n_total", stats.n_total).unwrap();
            d.set_item("n_bull", stats.n_bull).unwrap();
            d.set_item("raw_prob", (stats.raw_prob * 10000.0).round() / 10000.0).unwrap();
            d.set_item("smoothed_prob", (stats.smoothed_prob * 10000.0).round() / 10000.0).unwrap();
            d.set_item("bias", (stats.bias * 10000.0).round() / 10000.0).unwrap();
            d.set_item("ci_95_half", (stats.ci_95_half * 10000.0).round() / 10000.0).unwrap();
            d.set_item("perm_p_value", (rob.perm_p_value * 10000.0).round() / 10000.0).unwrap();
            d.set_item("noise_stability", (rob.noise_stability * 10000.0).round() / 10000.0).unwrap();
            d.set_item("temporal_consistent", rob.temporal_consistent).unwrap();
            let seg: Vec<f64> = rob.segment_edges.iter().map(|&e| (e * 10000.0).round() / 10000.0).collect();
            d.set_item("segment_edges", seg).unwrap();
            d.unbind()
        })
        .collect();
    dict.set_item("validated_states", val_list)?;

    // ── Top 20 significant but NOT validated (for debugging) ──
    let mut sig_not_val: Vec<_> = r
        .state_stats
        .iter()
        .filter(|(_, s)| s.significant)
        .filter(|(k, _)| {
            r.robustness
                .get(k)
                .map_or(true, |rob| !rob.validated)
        })
        .collect();
    sig_not_val.sort_by(|a, b| {
        b.1.bias
            .abs()
            .partial_cmp(&a.1.bias.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let reject_list: Vec<Py<PyDict>> = sig_not_val
        .iter()
        .take(20)
        .map(|(&key, stats)| {
            let d = PyDict::new(py);
            d.set_item("state", bias_engine::state::state_to_string(key)).unwrap();
            d.set_item("depth", bias_engine::state::state_depth(key)).unwrap();
            d.set_item("n_total", stats.n_total).unwrap();
            d.set_item("bias", (stats.bias * 10000.0).round() / 10000.0).unwrap();
            // Include rejection reasons
            if let Some(rob) = r.robustness.get(&key) {
                d.set_item("perm_p_value", (rob.perm_p_value * 10000.0).round() / 10000.0).unwrap();
                d.set_item("noise_stability", (rob.noise_stability * 10000.0).round() / 10000.0).unwrap();
                d.set_item("temporal_consistent", rob.temporal_consistent).unwrap();
                d.set_item("fdr_pass", rob.fdr_pass).unwrap();
                // Reason string
                let mut reasons = Vec::new();
                if !rob.fdr_pass { reasons.push("fdr_reject"); }
                if rob.noise_stability < 0.80 { reasons.push("noise_unstable"); }
                if !rob.temporal_consistent { reasons.push("temporal_inconsistent"); }
                d.set_item("rejection_reasons", reasons).unwrap();
            }
            d.unbind()
        })
        .collect();
    dict.set_item("rejected_states_top20", reject_list)?;

    // ── Robustness test pass rates ──
    let rob_summary = PyDict::new(py);
    let n_rob = r.robustness.len() as f64;
    if n_rob > 0.0 {
        let perm_pass = r.robustness.values().filter(|r| r.fdr_pass).count();
        let noise_pass = r.robustness.values().filter(|r| r.noise_stability >= 0.80).count();
        let temp_pass = r.robustness.values().filter(|r| r.temporal_consistent).count();
        rob_summary.set_item("perm_fdr_pass", perm_pass)?;
        rob_summary.set_item("noise_pass", noise_pass)?;
        rob_summary.set_item("temporal_pass", temp_pass)?;
        rob_summary.set_item("all_pass", r.n_validated)?;
        rob_summary.set_item("total_tested", r.robustness.len())?;
    }
    dict.set_item("robustness_summary", rob_summary)?;

    // ── Validated states by direction ──
    let n_bull_states = validated_states.iter().filter(|(_, s, _)| s.bias > 0.0).count();
    let n_bear_states = validated_states.iter().filter(|(_, s, _)| s.bias < 0.0).count();
    dict.set_item("n_bull_validated", n_bull_states)?;
    dict.set_item("n_bear_validated", n_bear_states)?;

    // ── Bias distribution of validated states ──
    if !validated_states.is_empty() {
        let biases: Vec<f64> = validated_states.iter().map(|(_, s, _)| s.bias).collect();
        let avg_abs_bias = biases.iter().map(|b| b.abs()).sum::<f64>() / biases.len() as f64;
        let max_bull_bias = biases.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_bear_bias = biases.iter().cloned().fold(f64::INFINITY, f64::min);
        dict.set_item("avg_abs_bias", (avg_abs_bias * 10000.0).round() / 10000.0)?;
        dict.set_item("max_bull_bias", (max_bull_bias * 10000.0).round() / 10000.0)?;
        dict.set_item("max_bear_bias", (max_bear_bias * 10000.0).round() / 10000.0)?;
    }

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine — Full Bias Series (Steps 1–9)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bias_engine_compute_bias<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, u64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let ts = timestamps.as_slice().expect("timestamps contiguous");
    let o = as_slice(&open);
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);
    let bv = as_slice(&buy_vol);
    let sv = as_slice(&sell_vol);
    let oi_s = as_slice(&oi);

    let r = bias_engine::compute_bias_series(ts, o, h, l, c, bv, sv, oi_s);

    let dict = PyDict::new(py);

    // ── Summary ──
    dict.set_item("n_bars", r.n_bars)?;
    dict.set_item("n_validated", r.n_validated)?;
    dict.set_item("baseline_bull_rate", (r.baseline_bull_rate * 10000.0).round() / 10000.0)?;

    // ── Coverage ──
    let cov = PyDict::new(py);
    cov.set_item("coverage_pct", (r.coverage_pct * 100.0).round() / 100.0)?;
    cov.set_item("fallback_pct", (r.fallback_pct * 100.0).round() / 100.0)?;
    cov.set_item("depth3_pct", (r.depth3_pct * 100.0).round() / 100.0)?;
    cov.set_item("depth2_pct", (r.depth2_pct * 100.0).round() / 100.0)?;
    cov.set_item("depth1_pct", (r.depth1_pct * 100.0).round() / 100.0)?;
    dict.set_item("coverage", cov)?;

    // ── Accuracy ──
    let acc = PyDict::new(py);
    acc.set_item("direction_accuracy", (r.direction_accuracy * 10000.0).round() / 10000.0)?;
    acc.set_item("strong_signal_accuracy", (r.strong_signal_accuracy * 10000.0).round() / 10000.0)?;
    acc.set_item("n_strong_bars", r.n_strong_bars)?;
    dict.set_item("accuracy", acc)?;

    // ── Regime ──
    let reg = PyDict::new(py);
    reg.set_item("trending_pct", (r.pct_trending * 100.0).round() / 100.0)?;
    reg.set_item("mean_reverting_pct", (r.pct_mean_reverting * 100.0).round() / 100.0)?;
    reg.set_item("high_vol_pct", (r.pct_high_vol * 100.0).round() / 100.0)?;
    dict.set_item("regime", reg)?;

    // ── Calibration ──
    let cal = PyDict::new(py);
    cal.set_item("brier_uncalibrated", (r.brier_uncalibrated * 10000.0).round() / 10000.0)?;
    cal.set_item("brier_calibrated", (r.brier_calibrated * 10000.0).round() / 10000.0)?;
    cal.set_item("calibration_improved", r.brier_calibrated < r.brier_uncalibrated)?;
    dict.set_item("calibration", cal)?;

    // ── Per-bar arrays (for downstream analysis) ──
    let n = r.n_bars;
    let mut final_biases = Vec::with_capacity(n);
    let mut confidences = Vec::with_capacity(n);
    let mut state_biases = Vec::with_capacity(n);
    let mut sweep_biases = Vec::with_capacity(n);
    let mut matched_depths = Vec::with_capacity(n);
    let mut directions = Vec::with_capacity(n);
    let mut regimes_arr = Vec::with_capacity(n);

    for bo in &r.bar_outputs {
        final_biases.push(bo.final_bias);
        confidences.push(bo.confidence);
        state_biases.push(bo.state_bias);
        sweep_biases.push(bo.sweep_bias);
        matched_depths.push(bo.matched_depth as i32);
        directions.push(match bo.direction {
            bias_engine::final_bias::BiasDirection::Bullish => 1i32,
            bias_engine::final_bias::BiasDirection::Neutral => 0i32,
            bias_engine::final_bias::BiasDirection::Bearish => -1i32,
        });
        regimes_arr.push(match bo.regime {
            bias_engine::regime::Regime::Trending => 0i32,
            bias_engine::regime::Regime::MeanReverting => 1i32,
            bias_engine::regime::Regime::HighVolatility => 2i32,
        });
    }

    dict.set_item("final_bias", PyArray1::from_vec(py, final_biases))?;
    dict.set_item("confidence", PyArray1::from_vec(py, confidences))?;
    dict.set_item("state_bias", PyArray1::from_vec(py, state_biases))?;
    dict.set_item("sweep_bias", PyArray1::from_vec(py, sweep_biases))?;
    dict.set_item("matched_depth", PyArray1::from_vec(py, matched_depths))?;
    dict.set_item("direction", PyArray1::from_vec(py, directions))?;
    dict.set_item("regime_class", PyArray1::from_vec(py, regimes_arr))?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine — Walk-Forward Evaluation (Step 12)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bias_engine_walkforward<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, u64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let ts = timestamps.as_slice().expect("timestamps contiguous");
    let o = as_slice(&open);
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);
    let bv = as_slice(&buy_vol);
    let sv = as_slice(&sell_vol);
    let oi_s = as_slice(&oi);

    let wf = bias_engine::walkforward::run_walk_forward(ts, o, h, l, c, bv, sv, oi_s);

    let dict = PyDict::new(py);

    // ── Summary ──
    dict.set_item("n_folds", wf.folds.len())?;
    dict.set_item("overall_accuracy", (wf.overall_accuracy * 10000.0).round() / 10000.0)?;
    dict.set_item("overall_strong_accuracy", (wf.overall_strong_accuracy * 10000.0).round() / 10000.0)?;
    dict.set_item("accuracy_std", (wf.accuracy_std * 10000.0).round() / 10000.0)?;
    dict.set_item("total_test_bars", wf.total_test_bars)?;
    dict.set_item("n_wf_validated_states", wf.wf_validated_states.len())?;

    // ── Per-fold details ──
    let fold_list: Vec<Py<PyDict>> = wf.folds.iter().map(|fold| {
        let d = PyDict::new(py);
        d.set_item("fold_idx", fold.fold_idx).unwrap();
        d.set_item("train_range", format!("{}-{}", fold.train_start, fold.train_end)).unwrap();
        d.set_item("val_range", format!("{}-{}", fold.val_start, fold.val_end)).unwrap();
        d.set_item("test_range", format!("{}-{}", fold.test_start, fold.test_end)).unwrap();
        d.set_item("n_validated", fold.n_validated).unwrap();
        d.set_item("test_accuracy", (fold.test_accuracy * 10000.0).round() / 10000.0).unwrap();
        d.set_item("test_strong_accuracy", (fold.test_strong_accuracy * 10000.0).round() / 10000.0).unwrap();
        d.set_item("test_n_bars", fold.test_n_bars).unwrap();
        d.set_item("test_n_strong", fold.test_n_strong).unwrap();
        d.set_item("test_mean_abs_bias", (fold.test_mean_abs_bias * 10000.0).round() / 10000.0).unwrap();
        d.set_item("brier_uncalibrated", (fold.brier_uncalibrated * 10000.0).round() / 10000.0).unwrap();
        d.set_item("brier_calibrated", (fold.brier_calibrated * 10000.0).round() / 10000.0).unwrap();
        d.unbind()
    }).collect();
    dict.set_item("folds", fold_list)?;

    // ── WF-validated states ──
    let wf_states: Vec<String> = wf.wf_validated_states
        .iter()
        .map(|&key| bias_engine::state::state_to_string(key))
        .collect();
    dict.set_item("wf_validated_states", wf_states)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine — TPE Optimizer (38 params, 2-phase nested)
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (timestamps, open, high, low, close, buy_vol, sell_vol, oi, n_outer_trials, seed, btc_close=None, btc_buy_vol=None, btc_sell_vol=None))]
fn bias_engine_optimize<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, u64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
    n_outer_trials: usize,
    seed: u64,
    btc_close: Option<PyReadonlyArray1<'py, f64>>,
    btc_buy_vol: Option<PyReadonlyArray1<'py, f64>>,
    btc_sell_vol: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let ts = timestamps.as_slice().expect("ts");
    let h = as_slice(&high);
    let l = as_slice(&low);
    let c = as_slice(&close);
    let bv = as_slice(&buy_vol);
    let sv = as_slice(&sell_vol);
    let oi_s = as_slice(&oi);

    // BTC slices (optional)
    let btc_c_vec: Option<Vec<f64>> = btc_close.as_ref().map(|a| as_slice(a).to_vec());
    let btc_bv_vec: Option<Vec<f64>> = btc_buy_vol.as_ref().map(|a| as_slice(a).to_vec());
    let btc_sv_vec: Option<Vec<f64>> = btc_sell_vol.as_ref().map(|a| as_slice(a).to_vec());

    let result = bias_engine::optimizer::run_optimization(
        ts, h, l, c, bv, sv, oi_s,
        btc_c_vec.as_deref(), btc_bv_vec.as_deref(), btc_sv_vec.as_deref(),
        n_outer_trials, seed,
    );

    let dict = PyDict::new(py);
    dict.set_item("best_score", (result.best_score * 10000.0).round() / 10000.0)?;
    dict.set_item("n_validated_states", result.n_validated_states)?;
    dict.set_item("trials_evaluated", result.trials_evaluated)?;

    // Best Group A params
    let pa = PyDict::new(py);
    let a = &result.best_params.a;
    pa.set_item("cvd_micro_window", a.cvd_micro_window)?;
    pa.set_item("cvd_macro_window", a.cvd_macro_window)?;
    pa.set_item("vol_micro_window", a.vol_micro_window)?;
    pa.set_item("vol_macro_window", a.vol_macro_window)?;
    pa.set_item("imbalance_ema_span", a.imbalance_ema_span)?;
    pa.set_item("atr_pct_window", a.atr_pct_window)?;
    pa.set_item("oi_change_window", a.oi_change_window)?;
    pa.set_item("quant_window", a.quant_window)?;
    pa.set_item("quantile_count", a.quantile_count)?;
    pa.set_item("k_horizon", a.k_horizon)?;
    pa.set_item("min_sample_size", a.min_sample_size)?;
    pa.set_item("min_edge", a.min_edge)?;
    pa.set_item("prior_strength", a.prior_strength)?;
    pa.set_item("fdr_alpha", a.fdr_alpha)?;
    pa.set_item("temporal_min_segments", a.temporal_min_segments)?;
    pa.set_item("temporal_max_reversals", a.temporal_max_reversals)?;
    pa.set_item("min_noise_stability", a.min_noise_stability)?;
    pa.set_item("ensemble_min_n", a.ensemble_min_n)?;
    dict.set_item("group_a", pa)?;

    // Best Group B params
    let pb = PyDict::new(py);
    let b = &result.best_params.b;
    pb.set_item("mr_ema_span1", b.mr_ema_span1)?;
    pb.set_item("mr_ema_span2", b.mr_ema_span2)?;
    pb.set_item("rsi_period", b.rsi_period)?;
    pb.set_item("rsi_threshold", b.rsi_threshold)?;
    pb.set_item("w_bias", b.w_bias)?;
    pb.set_item("w_mr1", b.w_mr1)?;
    pb.set_item("w_mr2", b.w_mr2)?;
    pb.set_item("w_rsi", b.w_rsi)?;
    pb.set_item("w_agree", b.w_agree)?;
    pb.set_item("w_cvd", b.w_cvd)?;
    pb.set_item("bias_override_threshold", b.bias_override_threshold)?;
    pb.set_item("bias_override_mult", b.bias_override_mult)?;
    pb.set_item("sweep_scale", b.sweep_scale)?;
    pb.set_item("sweep_aligned_weight", b.sweep_aligned_weight)?;
    pb.set_item("sweep_conflict_mult", b.sweep_conflict_mult)?;
    pb.set_item("regime_dir_lookback", b.regime_dir_lookback)?;
    pb.set_item("trending_threshold", b.trending_threshold)?;
    pb.set_item("high_vol_threshold", b.high_vol_threshold)?;
    pb.set_item("regime_shift_lookback", b.regime_shift_lookback)?;
    pb.set_item("regime_shift_penalty", b.regime_shift_penalty)?;
    // BTC params
    pb.set_item("btc_mom_window", b.btc_mom_window)?;
    pb.set_item("w_btc_mom", b.w_btc_mom)?;
    pb.set_item("btc_lead_window", b.btc_lead_window)?;
    pb.set_item("w_btc_lead", b.w_btc_lead)?;
    pb.set_item("w_btc_cvd", b.w_btc_cvd)?;
    dict.set_item("group_b", pb)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine — Scoring-based Walk-Forward with BTC
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (close, high, low, buy_vol, sell_vol, oi, group_a_vals, group_b_vals, btc_close=None, btc_buy_vol=None, btc_sell_vol=None))]
fn bias_engine_scoring_wf<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    buy_vol: PyReadonlyArray1<'py, f64>,
    sell_vol: PyReadonlyArray1<'py, f64>,
    oi: PyReadonlyArray1<'py, f64>,
    // Group A params (18 values)
    group_a_vals: PyReadonlyArray1<'py, f64>,
    // Group B params (25 values)
    group_b_vals: PyReadonlyArray1<'py, f64>,
    // BTC data (optional)
    btc_close: Option<PyReadonlyArray1<'py, f64>>,
    btc_buy_vol: Option<PyReadonlyArray1<'py, f64>>,
    btc_sell_vol: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let c = as_slice(&close);
    let h = as_slice(&high);
    let l = as_slice(&low);
    let bv = as_slice(&buy_vol);
    let sv = as_slice(&sell_vol);
    let oi_s = as_slice(&oi);

    let ga = as_slice(&group_a_vals);
    let gb = as_slice(&group_b_vals);
    let params_a = bias_engine::params::vec_to_group_a(ga);
    let params_b = bias_engine::params::vec_to_group_b(gb);

    let btc_c_vec: Option<Vec<f64>> = btc_close.as_ref().map(|a| as_slice(a).to_vec());
    let btc_bv_vec: Option<Vec<f64>> = btc_buy_vol.as_ref().map(|a| as_slice(a).to_vec());
    let btc_sv_vec: Option<Vec<f64>> = btc_sell_vol.as_ref().map(|a| as_slice(a).to_vec());

    let r = bias_engine::walkforward::run_scoring_walkforward(
        c, bv, sv, oi_s, h, l,
        btc_c_vec.as_deref(), btc_bv_vec.as_deref(), btc_sv_vec.as_deref(),
        &params_a, &params_b,
    );

    let dict = PyDict::new(py);
    dict.set_item("n_chunks", r.n_chunks)?;
    dict.set_item("overall_accuracy", (r.overall_accuracy * 10000.0).round() / 10000.0)?;
    dict.set_item("accuracy_std", (r.accuracy_std * 10000.0).round() / 10000.0)?;
    dict.set_item("total_bars", r.total_bars)?;
    dict.set_item("total_correct", r.total_correct)?;
    dict.set_item("chunk_accuracies", r.chunk_accuracies.iter()
        .map(|&a| (a * 10000.0).round() / 10000.0).collect::<Vec<f64>>())?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Bias Engine + KC Strategy
// ═══════════════════════════════════════════════════════════════════

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_bias_kc_strategy<'py>(
    py: Python<'py>,
    // 5m data
    ts_5m: PyReadonlyArray1<'py, u64>,
    open_5m: PyReadonlyArray1<'py, f64>,
    high_5m: PyReadonlyArray1<'py, f64>,
    low_5m: PyReadonlyArray1<'py, f64>,
    close_5m: PyReadonlyArray1<'py, f64>,
    buy_vol_5m: PyReadonlyArray1<'py, f64>,
    sell_vol_5m: PyReadonlyArray1<'py, f64>,
    oi_5m: PyReadonlyArray1<'py, f64>,
    // 3m data
    ts_3m: PyReadonlyArray1<'py, u64>,
    high_3m: PyReadonlyArray1<'py, f64>,
    low_3m: PyReadonlyArray1<'py, f64>,
    close_3m: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let r = bias_kc_strategy::run_bias_kc_strategy(
        ts_5m.as_slice().expect("ts_5m"),
        as_slice(&open_5m),
        as_slice(&high_5m),
        as_slice(&low_5m),
        as_slice(&close_5m),
        as_slice(&buy_vol_5m),
        as_slice(&sell_vol_5m),
        as_slice(&oi_5m),
        ts_3m.as_slice().expect("ts_3m"),
        as_slice(&high_3m),
        as_slice(&low_3m),
        as_slice(&close_3m),
    );

    let dict = PyDict::new(py);
    dict.set_item("total_pnl", (r.total_pnl * 100.0).round() / 100.0)?;
    dict.set_item("total_trades", r.total_trades)?;
    dict.set_item("total_wins", r.total_wins)?;
    dict.set_item("win_rate", (r.win_rate * 10000.0).round() / 10000.0)?;
    dict.set_item("total_fees", (r.total_fees * 100.0).round() / 100.0)?;
    dict.set_item("max_dd_pct", (r.max_dd_pct * 100.0).round() / 100.0)?;
    dict.set_item("avg_weekly_pnl", (r.avg_weekly_pnl * 100.0).round() / 100.0)?;
    dict.set_item("profitable_weeks", r.profitable_weeks)?;
    dict.set_item("total_weeks", r.total_weeks)?;

    // Weekly breakdown
    let week_list: Vec<Py<PyDict>> = r.weeks.iter().map(|w| {
        let d = PyDict::new(py);
        d.set_item("week", w.week_idx).unwrap();
        d.set_item("pnl", (w.pnl * 100.0).round() / 100.0).unwrap();
        d.set_item("trades", w.n_trades).unwrap();
        d.set_item("wins", w.n_wins).unwrap();
        d.set_item("tp", w.n_tp).unwrap();
        d.set_item("reversal", w.n_reversal).unwrap();
        d.set_item("max_dd", (w.max_dd_pct * 100.0).round() / 100.0).unwrap();
        d.set_item("end_balance", (w.end_balance * 100.0).round() / 100.0).unwrap();
        d.unbind()
    }).collect();
    dict.set_item("weeks", week_list)?;

    // Trade details (last 50 for inspection)
    let trade_list: Vec<Py<PyDict>> = r.trades.iter().rev().take(50).map(|t| {
        let d = PyDict::new(py);
        d.set_item("dir", if t.direction > 0.0 { "LONG" } else { "SHORT" }).unwrap();
        d.set_item("entry", (t.avg_entry * 100.0).round() / 100.0).unwrap();
        d.set_item("exit", (t.exit_price * 100.0).round() / 100.0).unwrap();
        d.set_item("pnl", (t.pnl * 100.0).round() / 100.0).unwrap();
        d.set_item("reason", t.exit_reason).unwrap();
        d.set_item("dca", t.dca_count).unwrap();
        d.set_item("bias", (t.bias_at_entry * 10000.0).round() / 10000.0).unwrap();
        d.set_item("conf", (t.confidence_at_entry * 10000.0).round() / 10000.0).unwrap();
        d.unbind()
    }).collect();
    dict.set_item("recent_trades", trade_list)?;

    Ok(dict.unbind())
}

// ═══════════════════════════════════════════════════════════════════
// Module Registration
// ═══════════════════════════════════════════════════════════════════

#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Existing functions (backward compat)
    m.add_function(wrap_pyfunction!(precompute_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(compute_adaptive_pmax, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest_dynamic, m)?)?;
    // KC lagged (look-ahead fix)
    m.add_function(wrap_pyfunction!(run_backtest_kc_lagged, m)?)?;
    // KC lagged + graduated DCA
    m.add_function(wrap_pyfunction!(run_backtest_kc_lagged_graduated, m)?)?;
    // KC lagged + graduated DCA + graduated TP
    m.add_function(wrap_pyfunction!(run_backtest_kc_lagged_graduated_tp, m)?)?;
    // KAMA backtest
    m.add_function(wrap_pyfunction!(run_kama_backtest, m)?)?;
    // KAMA + KC lagged backtest
    m.add_function(wrap_pyfunction!(run_kama_kc_backtest, m)?)?;
    // KAMA + KC + graduated DCA
    m.add_function(wrap_pyfunction!(run_kama_kc_grad_dca, m)?)?;
    // KAMA + KC + graduated DCA + graduated TP
    m.add_function(wrap_pyfunction!(run_kama_kc_grad_dca_tp, m)?)?;
    // CVD Order Flow backtest
    m.add_function(wrap_pyfunction!(run_cvd_backtest, m)?)?;
    // CVD + OI backtest
    m.add_function(wrap_pyfunction!(run_cvd_oi_backtest, m)?)?;
    // Combined: KAMA + PMax + CVD+OI + KC + GrDCA + GrTP
    m.add_function(wrap_pyfunction!(run_combined_backtest, m)?)?;
    // Rust-native TPE optimizer
    m.add_function(wrap_pyfunction!(run_wf_fold, m)?)?;
    // Pure PMax optimizer
    m.add_function(wrap_pyfunction!(run_pmax_wf_fold, m)?)?;
    // Pattern scanner
    m.add_function(wrap_pyfunction!(scan_patterns, m)?)?;
    // Pattern miner
    m.add_function(wrap_pyfunction!(mine_patterns, m)?)?;
    // Pattern miner v2 (comprehensive)
    m.add_function(wrap_pyfunction!(run_full_analysis, m)?)?;
    // Sweep miner
    m.add_function(wrap_pyfunction!(run_sweep_mining_py, m)?)?;
    // Sweep strategy
    m.add_function(wrap_pyfunction!(run_sweep_strategy_py, m)?)?;
    // 1H Sweep strategy
    m.add_function(wrap_pyfunction!(run_1h_sweep_strategy_py, m)?)?;
    // 1H Sweep strategy v2
    m.add_function(wrap_pyfunction!(run_1h_sweep_v2_py, m)?)?;
    // Candle sweep analysis
    m.add_function(wrap_pyfunction!(run_candle_analysis_py, m)?)?;
    // Candle sweep miner
    m.add_function(wrap_pyfunction!(run_candle_miner_py, m)?)?;
    // Cross-asset candle sweep miner
    m.add_function(wrap_pyfunction!(run_candle_miner_cross_py, m)?)?;
    // Sweep Miner V2 (28 features)
    m.add_function(wrap_pyfunction!(run_sweep_miner_v2_py, m)?)?;
    // 15M Candle sweep analysis
    m.add_function(wrap_pyfunction!(run_candle_analysis_15m_py, m)?)?;
    // 15M Candle sweep miner
    m.add_function(wrap_pyfunction!(run_candle_miner_15m_py, m)?)?;
    // Candle sweep strategy
    m.add_function(wrap_pyfunction!(run_candle_strategy_py, m)?)?;
    // Candle sweep strategy + KC
    m.add_function(wrap_pyfunction!(run_candle_kc_strategy_py, m)?)?;
    // Multi-TF: 1H sweep + 3m KC
    m.add_function(wrap_pyfunction!(run_multi_tf_strategy_py, m)?)?;
    // Multi-TF with custom KC params
    m.add_function(wrap_pyfunction!(run_multi_tf_params_py, m)?)?;
    // KC WF Optimizer (35-fold)
    m.add_function(wrap_pyfunction!(run_kc_wf_optimization_py, m)?)?;
    // DCA WF Optimizer (KC sabit)
    m.add_function(wrap_pyfunction!(run_dca_wf_optimization_py, m)?)?;
    // Graduated DCA/TP WF Optimizer
    m.add_function(wrap_pyfunction!(run_grad_wf_optimization_py, m)?)?;
    // Bias Engine (continuous bias system)
    m.add_function(wrap_pyfunction!(bias_engine_step1_2, m)?)?;
    // Bias Engine full pipeline (Steps 1-4)
    m.add_function(wrap_pyfunction!(bias_engine_full, m)?)?;
    // Bias Engine compute bias series (Steps 1-9)
    m.add_function(wrap_pyfunction!(bias_engine_compute_bias, m)?)?;
    // Bias Engine walk-forward evaluation (Step 12)
    m.add_function(wrap_pyfunction!(bias_engine_walkforward, m)?)?;
    // Bias Engine TPE optimizer (43 params, BTC support)
    m.add_function(wrap_pyfunction!(bias_engine_optimize, m)?)?;
    // Bias Engine scoring-based walk-forward (BTC support)
    m.add_function(wrap_pyfunction!(bias_engine_scoring_wf, m)?)?;
    // Bias + KC strategy
    m.add_function(wrap_pyfunction!(run_bias_kc_strategy, m)?)?;
    // New: stateful trading engine
    m.add_class::<PyTradingEngine>()?;
    Ok(())
}

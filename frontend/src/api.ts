import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8055",
  timeout: 30000,
});

// --- Types ---

export interface DataStatus {
  symbol: string;
  available: boolean;
  size_mb: number;
  tick_count: number;
  date_from: string | null;
  date_to: string | null;
}

export interface BacktestParams {
  symbol: string;
  w_delta: number;
  w_cvd: number;
  w_logp: number;
  w_obi_w: number;
  w_obi_d: number;
  w_sweep: number;
  w_burst: number;
  w_oi: number;
  trs_confirm_ticks: number;
  trs_bullish_zone: number;
  trs_bearish_zone: number;
  trs_agreement: number;
  stop_loss_pct: number;
  trailing_activation_pct: number;
  trailing_distance_pct: number;
  exit_score_hard: number;
  exit_score_soft: number;
  min_prints_per_sec: number;
  entry_cooldown_sec: number;
  rolling_window_sec: number;
  time_flat_sec: number;
  margin_per_trade: number;
  leverage: number;
  maker_fee: number;
  taker_fee: number;
}

export interface BacktestMetrics {
  net_pnl: number;
  net_pnl_pct: number;
  profit_factor: number;
  win_rate: number;
  max_drawdown: number;
  total_trades: number;
  equity_final: number;
  avg_hold_sec: number;
  exit_types: Record<string, number>;
}

export interface Trade {
  id: number;
  pnl: number;
  pnl_pct: number;
  type: string;
  hold_sec: number;
}

export interface EquityPoint {
  ts: number;
  equity: number;
}

export interface BacktestResult {
  symbol: string;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: Trade[];
  params: Record<string, number>;
  error?: string;
}

export interface OptimizationResult {
  rank: number;
  score: number;
  params: Record<string, number>;
  in_sample: Record<string, number>;
  out_of_sample: Record<string, number>;
}

export interface ProgressState {
  running: boolean;
  progress: number;
  symbol: string | null;
  month?: string;
}

// --- Default params ---

export const DEFAULT_PARAMS: BacktestParams = {
  symbol: "BTCUSDT",
  w_delta: 0.22,
  w_cvd: 0.18,
  w_logp: 0.12,
  w_obi_w: 0.14,
  w_obi_d: 0.12,
  w_sweep: 0.08,
  w_burst: 0.08,
  w_oi: 0.06,
  trs_confirm_ticks: 90,
  trs_bullish_zone: 0.65,
  trs_bearish_zone: 0.35,
  trs_agreement: 0.4,
  stop_loss_pct: 1.5,
  trailing_activation_pct: 0.8,
  trailing_distance_pct: 0.5,
  exit_score_hard: 0.85,
  exit_score_soft: 0.7,
  min_prints_per_sec: 1.0,
  entry_cooldown_sec: 300,
  rolling_window_sec: 3600,
  time_flat_sec: 14400,
  margin_per_trade: 100,
  leverage: 50,
  maker_fee: 0.0002,
  taker_fee: 0.0005,
};

// --- API calls ---

export const getSymbols = () => api.get<{ symbols: string[] }>("/api/symbols");

export const getDataStatus = () =>
  api.get<{ data: DataStatus[] }>("/api/data/status");

export const startDownload = (symbol: string) =>
  api.post(`/api/data/download/${symbol}`);

export const getDownloadProgress = () =>
  api.get<ProgressState>("/api/data/download/progress");

export const runBacktest = (params: BacktestParams) =>
  api.post("/api/backtest/run", params);

export const getBacktestProgress = () =>
  api.get<ProgressState>("/api/backtest/progress");

export const getBacktestResults = () =>
  api.get<BacktestResult>("/api/backtest/results");

export const startOptimization = (symbol: string, n_trials: number) =>
  api.post("/api/optimize/start", { symbol, n_trials });

export const getOptimizeProgress = () =>
  api.get("/api/optimize/progress");

export const getOptimizeResults = (symbol: string) =>
  api.get<{ symbol: string; results: OptimizationResult[] }>(
    `/api/optimize/results/${symbol}`
  );

export const getHealth = () => api.get("/api/health");

// --- Pipeline API ---

export const startPipeline = (pairs: { symbol: string; strategy: string; timeframes: string[] }[]) =>
  api.post("/api/pipeline/start", { pairs });

export const stopPipeline = () => api.post("/api/pipeline/stop");

export const getPipelineStatus = () => api.get("/api/pipeline/status");

export const addToPipeline = (symbol: string, strategy: string, timeframes: string[]) =>
  api.post("/api/pipeline/add", { symbol, strategy, timeframes });

export const removeFromPipeline = (jobId: string) =>
  api.delete(`/api/pipeline/remove/${jobId}`);

export const getPipelineHistory = () => api.get("/api/pipeline/history");

export const getWalkForwardState = () => api.get("/api/pipeline/walkforward-state");

// --- Pipeline V2 API ---

export const v2Init = (symbol: string, timeframe: string = "3m", n_trials: number = 1000) =>
  api.post("/api/v2/init", { symbol, timeframe, n_trials });

export const v2GetState = () => api.get("/api/v2/state");

export const v2StepStart = (step: string) =>
  api.post("/api/v2/step/start", { step });

export const v2StepSelect = (step: string, selected_index: number) =>
  api.post("/api/v2/step/select", { step, selected_index });

export const v2UpdateSettings = (settings: { timeframe?: string; n_trials?: number }) =>
  api.post("/api/v2/settings", settings);

export const v2Stop = () => api.post("/api/v2/stop");

export const v2Reset = () => api.post("/api/v2/reset");

// --- Settings API ---

export const getSettings = () => api.get("/api/settings");

export const updateSettings = (settings: Record<string, any>) =>
  api.post("/api/settings", settings);

// --- Results API ---

export const getAllResults = () => api.get("/api/results");

export default api;

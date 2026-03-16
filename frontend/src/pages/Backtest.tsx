import { useState, useEffect, useRef } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { Play, RotateCcw, Loader2 } from "lucide-react";
import type { BacktestParams, BacktestResult } from "../api";
import {
  DEFAULT_PARAMS,
  runBacktest,
  getBacktestProgress,
  getBacktestResults,
} from "../api";
import ProgressBar from "../components/ProgressBar";

const PARAM_GROUPS = [
  {
    title: "DFS Agirliklari",
    params: [
      { key: "w_delta", label: "Delta", min: 0.02, max: 0.4, step: 0.02 },
      { key: "w_cvd", label: "CVD", min: 0.02, max: 0.35, step: 0.02 },
      { key: "w_logp", label: "LogP", min: 0.02, max: 0.25, step: 0.02 },
      { key: "w_obi_w", label: "OBI_W", min: 0.02, max: 0.25, step: 0.02 },
      { key: "w_obi_d", label: "OBI_D", min: 0.02, max: 0.25, step: 0.02 },
      { key: "w_sweep", label: "Sweep", min: 0.02, max: 0.15, step: 0.02 },
      { key: "w_burst", label: "Burst", min: 0.02, max: 0.15, step: 0.02 },
      { key: "w_oi", label: "OI", min: 0.02, max: 0.15, step: 0.02 },
    ],
  },
  {
    title: "TRS Parametreleri",
    params: [
      { key: "trs_confirm_ticks", label: "Confirm Ticks", min: 30, max: 180, step: 10 },
      { key: "trs_bullish_zone", label: "Bullish Zone", min: 0.55, max: 0.85, step: 0.05 },
      { key: "trs_bearish_zone", label: "Bearish Zone", min: 0.15, max: 0.45, step: 0.05 },
      { key: "trs_agreement", label: "Agreement", min: 0.25, max: 0.6, step: 0.05 },
    ],
  },
  {
    title: "Risk / Cikis",
    params: [
      { key: "stop_loss_pct", label: "Stop Loss %", min: 0.5, max: 3.0, step: 0.25 },
      { key: "trailing_activation_pct", label: "Trail Act %", min: 0.3, max: 2.0, step: 0.1 },
      { key: "trailing_distance_pct", label: "Trail Dist %", min: 0.2, max: 1.5, step: 0.1 },
      { key: "exit_score_hard", label: "Exit Hard", min: 0.7, max: 0.95, step: 0.05 },
      { key: "exit_score_soft", label: "Exit Soft", min: 0.5, max: 0.85, step: 0.05 },
    ],
  },
  {
    title: "Filtreler",
    params: [
      { key: "min_prints_per_sec", label: "Min Prints/s", min: 0.5, max: 3.0, step: 0.5 },
      { key: "entry_cooldown_sec", label: "Cooldown (s)", min: 60, max: 600, step: 60 },
      { key: "rolling_window_sec", label: "Rolling Win (s)", min: 1800, max: 7200, step: 600 },
      { key: "time_flat_sec", label: "Time Flat (s)", min: 3600, max: 28800, step: 3600 },
      { key: "margin_per_trade", label: "Margin (USDT)", min: 10, max: 100, step: 10 },
    ],
  },
];

export default function Backtest() {
  const [params, setParams] = useState<BacktestParams>({ ...DEFAULT_PARAMS });
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({
    "DFS Agirliklari": true,
    "TRS Parametreleri": true,
    "Risk / Cikis": true,
    Filtreler: false,
  });
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const updateParam = (key: string, value: number) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const handleRun = async () => {
    try {
      setRunning(true);
      setProgress(0);
      setResult(null);
      await runBacktest(params);

      pollRef.current = setInterval(async () => {
        try {
          const prog = await getBacktestProgress();
          setProgress(prog.data.progress);
          if (!prog.data.running) {
            clearInterval(pollRef.current);
            const res = await getBacktestResults();
            setResult(res.data);
            setRunning(false);
          }
        } catch {
          clearInterval(pollRef.current);
          setRunning(false);
        }
      }, 1000);
    } catch {
      setRunning(false);
    }
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const toggleGroup = (title: string) => {
    setOpenGroups((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  const exitTypeData = result?.metrics?.exit_types
    ? Object.entries(result.metrics.exit_types).map(([name, count]) => ({
        name,
        count,
      }))
    : [];

  return (
    <div className="flex gap-6 h-full">
      {/* Left Panel - Parameters */}
      <div className="w-80 shrink-0 overflow-y-auto space-y-3 pr-2">
        {/* Symbol Select */}
        <div className="bg-surface rounded-lg border border-border p-4">
          <label className="text-xs text-slate-500 uppercase tracking-wider block mb-2">
            Pair
          </label>
          <select
            value={params.symbol}
            onChange={(e) => setParams(prev => ({ ...prev, symbol: e.target.value }))}
            className="w-full bg-surface-dark border border-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
          >
            {["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {/* Param Groups */}
        {PARAM_GROUPS.map((group) => (
          <div
            key={group.title}
            className="bg-surface rounded-lg border border-border overflow-hidden"
          >
            <button
              onClick={() => toggleGroup(group.title)}
              className="w-full flex items-center justify-between p-3 text-sm font-semibold text-white hover:bg-surface-hover transition-colors"
            >
              {group.title}
              <span className="text-slate-500 text-xs">
                {openGroups[group.title] ? "▲" : "▼"}
              </span>
            </button>
            {openGroups[group.title] && (
              <div className="px-3 pb-3 space-y-3">
                {group.params.map((p) => (
                  <ParamSlider
                    key={p.key}
                    label={p.label}
                    value={(params as any)[p.key]}
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    onChange={(v) => updateParam(p.key, v)}
                  />
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleRun}
            disabled={running}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {running ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {running ? "Calisiyor..." : "Backtest Calistir"}
          </button>
          <button
            onClick={() => setParams({ ...DEFAULT_PARAMS })}
            className="px-3 py-2.5 rounded-lg bg-surface-hover border border-border text-slate-400 hover:text-white transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Progress */}
        {running && <ProgressBar progress={progress} label="Backtest" />}
      </div>

      {/* Right Panel - Results */}
      <div className="flex-1 overflow-y-auto space-y-5">
        {!result && !running && (
          <div className="bg-surface rounded-lg border border-border p-12 text-center">
            <Play className="w-10 h-10 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">
              Parametreleri ayarlayip "Backtest Calistir" butonuna basin
            </p>
          </div>
        )}

        {result && result.metrics && (
          <>
            {/* Metric Cards */}
            <div className="grid grid-cols-3 gap-3">
              <ResultCard
                label="Net P&L"
                value={`${result.metrics.net_pnl >= 0 ? "+" : ""}${result.metrics.net_pnl.toFixed(2)} USDT`}
                sub={`${result.metrics.net_pnl_pct >= 0 ? "+" : ""}${result.metrics.net_pnl_pct.toFixed(2)}%`}
                positive={result.metrics.net_pnl >= 0}
              />
              <ResultCard
                label="Profit Factor"
                value={result.metrics.profit_factor.toFixed(3)}
                positive={result.metrics.profit_factor > 1}
              />
              <ResultCard
                label="Win Rate"
                value={`${result.metrics.win_rate.toFixed(1)}%`}
                positive={result.metrics.win_rate > 50}
              />
              <ResultCard
                label="Max Drawdown"
                value={`${result.metrics.max_drawdown.toFixed(2)}%`}
                positive={false}
              />
              <ResultCard
                label="Toplam Islem"
                value={String(result.metrics.total_trades)}
              />
              <ResultCard
                label="Ort. Hold"
                value={`${(result.metrics.avg_hold_sec / 60).toFixed(1)} dk`}
              />
            </div>

            {/* Exit Types */}
            {exitTypeData.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {exitTypeData.map(({ name, count }) => (
                  <span
                    key={name}
                    className="text-xs px-2.5 py-1 rounded-full bg-surface border border-border text-slate-300"
                  >
                    {name}: {count}
                  </span>
                ))}
              </div>
            )}

            {/* Equity Curve */}
            {result.equity_curve.length > 0 && (
              <div className="bg-surface rounded-lg border border-border p-4">
                <h3 className="text-sm font-semibold text-white mb-3">
                  Equity Curve
                </h3>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={result.equity_curve}>
                      <defs>
                        <linearGradient
                          id="eqGradBt"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor="#3b82f6"
                            stopOpacity={0.3}
                          />
                          <stop
                            offset="100%"
                            stopColor="#3b82f6"
                            stopOpacity={0.02}
                          />
                        </linearGradient>
                      </defs>
                      <XAxis
                        dataKey="ts"
                        tickFormatter={(v) =>
                          new Date(v * 1000).toLocaleDateString("tr-TR", {
                            day: "2-digit",
                            month: "short",
                          })
                        }
                        stroke="#2d3148"
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickLine={false}
                      />
                      <YAxis
                        stroke="#2d3148"
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickLine={false}
                        tickFormatter={(v) => `$${v}`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#1a1d29",
                          border: "1px solid #2d3148",
                          borderRadius: "8px",
                          fontSize: 12,
                        }}
                        labelFormatter={(v) =>
                          new Date(Number(v) * 1000).toLocaleString("tr-TR")
                        }
                        formatter={(v) => [
                          `$${Number(v).toFixed(2)}`,
                          "Equity",
                        ]}
                      />
                      <Area
                        type="monotone"
                        dataKey="equity"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        fill="url(#eqGradBt)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* PnL Distribution */}
            {result.trades.length > 0 && (
              <div className="bg-surface rounded-lg border border-border p-4">
                <h3 className="text-sm font-semibold text-white mb-3">
                  P&L Dagilimi
                </h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.trades.slice(0, 100)}>
                      <XAxis dataKey="id" hide />
                      <YAxis
                        stroke="#2d3148"
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickLine={false}
                        tickFormatter={(v) => `$${v}`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#1a1d29",
                          border: "1px solid #2d3148",
                          borderRadius: "8px",
                          fontSize: 12,
                        }}
                        formatter={(v) => [`$${Number(v).toFixed(2)}`, "P&L"]}
                      />
                      <Bar dataKey="pnl">
                        {result.trades.slice(0, 100).map((t, i) => (
                          <Cell
                            key={i}
                            fill={t.pnl >= 0 ? "#22c55e" : "#ef4444"}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Trade Table */}
            {result.trades.length > 0 && (
              <div className="bg-surface rounded-lg border border-border overflow-hidden">
                <div className="p-4 border-b border-border">
                  <h3 className="text-sm font-semibold text-white">
                    Islem Listesi ({result.trades.length})
                  </h3>
                </div>
                <div className="overflow-x-auto max-h-80 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-surface-dark">
                      <tr className="text-xs text-slate-500 uppercase">
                        <th className="px-4 py-2 text-left">#</th>
                        <th className="px-4 py-2 text-right">P&L</th>
                        <th className="px-4 py-2 text-right">P&L %</th>
                        <th className="px-4 py-2 text-right">Hold</th>
                        <th className="px-4 py-2 text-left">Cikis</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.trades.map((t) => (
                        <tr
                          key={t.id}
                          className="border-t border-border/50 hover:bg-surface-hover"
                        >
                          <td className="px-4 py-2 text-slate-400">{t.id}</td>
                          <td
                            className={`px-4 py-2 text-right font-mono ${
                              t.pnl >= 0 ? "text-profit" : "text-loss"
                            }`}
                          >
                            {t.pnl >= 0 ? "+" : ""}
                            {t.pnl.toFixed(2)}
                          </td>
                          <td
                            className={`px-4 py-2 text-right font-mono ${
                              t.pnl_pct >= 0 ? "text-profit" : "text-loss"
                            }`}
                          >
                            {t.pnl_pct >= 0 ? "+" : ""}
                            {t.pnl_pct.toFixed(2)}%
                          </td>
                          <td className="px-4 py-2 text-right text-slate-400">
                            {t.hold_sec >= 3600
                              ? `${(t.hold_sec / 3600).toFixed(1)}s`
                              : `${(t.hold_sec / 60).toFixed(1)}dk`}
                          </td>
                          <td className="px-4 py-2">
                            <span className="text-xs px-1.5 py-0.5 rounded bg-surface-dark text-slate-300">
                              {t.type}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}

        {result?.error && (
          <div className="bg-loss/10 border border-loss/30 rounded-lg p-4 text-loss text-sm">
            Hata: {result.error}
          </div>
        )}
      </div>
    </div>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          step={step}
          min={min}
          max={max}
          className="w-16 bg-transparent text-right text-white font-mono text-xs focus:outline-none border-b border-border focus:border-blue-500"
        />
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 rounded-full appearance-none bg-surface-dark cursor-pointer accent-blue-500"
      />
    </div>
  );
}

function ResultCard({
  label,
  value,
  sub,
  positive,
}: {
  label: string;
  value: string;
  sub?: string;
  positive?: boolean;
}) {
  return (
    <div className="bg-surface rounded-lg border border-border p-4">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
        {label}
      </p>
      <p
        className={`text-xl font-bold ${
          positive === undefined
            ? "text-white"
            : positive
              ? "text-profit"
              : "text-loss"
        }`}
      >
        {value}
      </p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

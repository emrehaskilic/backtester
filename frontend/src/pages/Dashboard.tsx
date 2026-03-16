import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  FlaskConical,
  Database,
} from "lucide-react";
import type { DataStatus, BacktestResult } from "../api";
import { getDataStatus, getBacktestResults } from "../api";

export default function Dashboard() {
  const navigate = useNavigate();
  const [dataStatuses, setDataStatuses] = useState<DataStatus[]>([]);
  const [lastResult, setLastResult] = useState<BacktestResult | null>(null);

  useEffect(() => {
    getDataStatus()
      .then((r) => setDataStatuses(r.data.data))
      .catch(() => {});
    getBacktestResults()
      .then((r) => {
        if (r.data && !r.data.error) setLastResult(r.data);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Pair Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"] as const).map(
          (symbol) => {
            const data = dataStatuses.find((d) => d.symbol === symbol);
            return (
              <div
                key={symbol}
                className="bg-surface rounded-lg p-4 border border-border hover:border-blue-500/50 cursor-pointer transition-colors"
                onClick={() => navigate("/backtest")}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="font-bold text-white">{symbol}</span>
                  {data?.available ? (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-profit/20 text-profit">
                      HAZIR
                    </span>
                  ) : (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-loss/20 text-loss">
                      VERi YOK
                    </span>
                  )}
                </div>
                {data?.available ? (
                  <div className="space-y-1 text-xs text-slate-400">
                    <p>{(data.tick_count / 1_000_000).toFixed(1)}M tick</p>
                    <p>{data.size_mb.toFixed(0)} MB</p>
                    <p>
                      {data.date_from} &rarr; {data.date_to}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    Veri indirilmedi
                  </p>
                )}
              </div>
            );
          }
        )}
      </div>

      {/* Last Backtest Result */}
      {lastResult && lastResult.metrics && (
        <div className="bg-surface rounded-lg border border-border p-5">
          <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-blue-400" />
            Son Backtest — {lastResult.symbol}
          </h3>

          {/* Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
            <MetricBox
              label="Net P&L"
              value={`${lastResult.metrics.net_pnl >= 0 ? "+" : ""}${lastResult.metrics.net_pnl.toFixed(2)}`}
              positive={lastResult.metrics.net_pnl >= 0}
            />
            <MetricBox
              label="Profit Factor"
              value={lastResult.metrics.profit_factor.toFixed(3)}
              positive={lastResult.metrics.profit_factor > 1}
            />
            <MetricBox
              label="Win Rate"
              value={`${lastResult.metrics.win_rate.toFixed(1)}%`}
              positive={lastResult.metrics.win_rate > 50}
            />
            <MetricBox
              label="Max Drawdown"
              value={`${lastResult.metrics.max_drawdown.toFixed(2)}%`}
              positive={false}
            />
            <MetricBox
              label="Toplam Islem"
              value={String(lastResult.metrics.total_trades)}
            />
            <MetricBox
              label="Ort. Hold"
              value={`${(lastResult.metrics.avg_hold_sec / 60).toFixed(1)}m`}
            />
          </div>

          {/* Equity Curve */}
          {lastResult.equity_curve.length > 0 && (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={lastResult.equity_curve}>
                  <defs>
                    <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
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
                    domain={["dataMin - 50", "dataMax + 50"]}
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
                    formatter={(v) => [`$${Number(v).toFixed(2)}`, "Equity"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fill="url(#eqGrad)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Quick Actions */}
      {!lastResult && (
        <div className="bg-surface rounded-lg border border-border p-8 text-center">
          <FlaskConical className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <h3 className="text-white font-semibold mb-2">Henuz backtest yok</h3>
          <p className="text-sm text-slate-500 mb-4">
            Baslamak icin once veri indirin, sonra backtest calistirin.
          </p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => navigate("/data")}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface-hover border border-border text-sm text-slate-300 hover:text-white transition-colors"
            >
              <Database className="w-4 h-4" />
              Veri Indir
            </button>
            <button
              onClick={() => navigate("/backtest")}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-sm text-white hover:bg-blue-500 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              Backtest
            </button>
          </div>
        </div>
      )}

      {/* Data Status Summary */}
      {dataStatuses.length > 0 && (
        <div className="bg-surface rounded-lg border border-border p-5">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-400" />
            Veri Durumu
          </h3>
          <div className="space-y-2">
            {dataStatuses.map((d) => (
              <div
                key={d.symbol}
                className="flex items-center justify-between text-sm"
              >
                <span className="text-slate-300 font-mono">{d.symbol}</span>
                {d.available ? (
                  <span className="text-profit text-xs">
                    {(d.tick_count / 1_000_000).toFixed(1)}M tick |{" "}
                    {d.size_mb.toFixed(0)} MB
                  </span>
                ) : (
                  <span className="text-loss text-xs">Veri yok</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricBox({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="bg-surface-dark rounded-lg p-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
        {label}
      </p>
      <p
        className={`text-lg font-bold ${
          positive === undefined
            ? "text-white"
            : positive
              ? "text-profit"
              : "text-loss"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

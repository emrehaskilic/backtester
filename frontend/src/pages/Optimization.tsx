import { useState, useEffect, useRef } from "react";
import { Play, Loader2, Trophy } from "lucide-react";
import type { OptimizationResult } from "../api";
import {
  startOptimization,
  getOptimizeProgress,
  getOptimizeResults,
} from "../api";
import ProgressBar from "../components/ProgressBar";

export default function Optimization() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [nTrials, setNTrials] = useState(100);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<OptimizationResult[]>([]);
  const [selectedRank, setSelectedRank] = useState<number | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  useEffect(() => {
    loadResults(symbol);
  }, [symbol]);

  // On mount: check if optimization is already running on backend
  const resumePollingIfActive = async () => {
    try {
      const prog = await getOptimizeProgress();
      if (prog.data.running) {
        setRunning(true);
        setSymbol(prog.data.symbol || "BTCUSDT");
        setProgress(prog.data.progress);
        setNTrials(prog.data.total || 50);
        pollRef.current = setInterval(async () => {
          try {
            const p = await getOptimizeProgress();
            setProgress(p.data.progress);
            if (!p.data.running) {
              clearInterval(pollRef.current);
              await loadResults(prog.data.symbol || "BTCUSDT");
              setRunning(false);
            }
          } catch {
            clearInterval(pollRef.current);
            setRunning(false);
          }
        }, 2000);
      }
    } catch {
      // backend not reachable
    }
  };

  useEffect(() => {
    resumePollingIfActive();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const loadResults = async (sym: string) => {
    try {
      const res = await getOptimizeResults(sym);
      if (res.data.results?.length) {
        setResults(res.data.results);
        setSelectedRank(1);
      } else {
        setResults([]);
        setSelectedRank(null);
      }
    } catch {
      setResults([]);
    }
  };

  const handleStart = async () => {
    try {
      setRunning(true);
      setProgress(0);
      await startOptimization(symbol, nTrials);

      pollRef.current = setInterval(async () => {
        try {
          const prog = await getOptimizeProgress();
          setProgress(prog.data.progress);
          if (!prog.data.running) {
            clearInterval(pollRef.current);
            await loadResults(symbol);
            setRunning(false);
          }
        } catch {
          clearInterval(pollRef.current);
          setRunning(false);
        }
      }, 2000);
    } catch {
      setRunning(false);
    }
  };

  const selected = results.find((r) => r.rank === selectedRank);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-end gap-4">
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider block mb-1">
            Pair
          </label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="bg-surface border border-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
          >
            {["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider block mb-1">
            Trial Sayisi
          </label>
          <input
            type="number"
            value={nTrials}
            onChange={(e) => setNTrials(parseInt(e.target.value) || 100)}
            min={10}
            max={2000}
            step={50}
            className="w-24 bg-surface border border-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
          />
        </div>
        <button
          onClick={handleStart}
          disabled={running}
          className="flex items-center gap-2 px-5 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {running ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {running ? "Calisiyor..." : "Optimizasyon Baslat"}
        </button>
      </div>

      {running && <ProgressBar progress={progress} label="Optimizasyon" />}

      {/* Results Table */}
      {results.length > 0 && (
        <div className="bg-surface rounded-lg border border-border overflow-hidden">
          <div className="p-4 border-b border-border flex items-center gap-2">
            <Trophy className="w-4 h-4 text-yellow-400" />
            <h3 className="text-sm font-semibold text-white">
              En Iyi Sonuclar — {symbol}
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-surface-dark">
                <tr className="text-xs text-slate-500 uppercase">
                  <th className="px-4 py-2 text-left">#</th>
                  <th className="px-4 py-2 text-right">Score</th>
                  <th className="px-4 py-2 text-right">PF (IS)</th>
                  <th className="px-4 py-2 text-right">WR (IS)</th>
                  <th className="px-4 py-2 text-right">DD (IS)</th>
                  <th className="px-4 py-2 text-right">PnL (IS)</th>
                  <th className="px-4 py-2 text-right">PF (OOS)</th>
                  <th className="px-4 py-2 text-right">WR (OOS)</th>
                  <th className="px-4 py-2 text-right">PnL (OOS)</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr
                    key={r.rank}
                    onClick={() => setSelectedRank(r.rank)}
                    className={`border-t border-border/50 cursor-pointer transition-colors ${
                      selectedRank === r.rank
                        ? "bg-blue-500/10"
                        : "hover:bg-surface-hover"
                    }`}
                  >
                    <td className="px-4 py-2 font-bold text-yellow-400">
                      #{r.rank}
                    </td>
                    <td className="px-4 py-2 text-right text-white font-mono">
                      {r.score.toFixed(1)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-profit">
                      {r.in_sample.profit_factor?.toFixed(2)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">
                      {r.in_sample.win_rate?.toFixed(1)}%
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-loss">
                      {r.in_sample.max_drawdown?.toFixed(1)}%
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-profit">
                      {r.in_sample.net_pnl >= 0 ? "+" : ""}
                      {r.in_sample.net_pnl?.toFixed(0)}
                    </td>
                    <td
                      className={`px-4 py-2 text-right font-mono ${
                        (r.out_of_sample as any).profit_factor > 1
                          ? "text-profit"
                          : "text-loss"
                      }`}
                    >
                      {(r.out_of_sample as any).profit_factor?.toFixed(2)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">
                      {(r.out_of_sample as any).win_rate?.toFixed(1)}%
                    </td>
                    <td
                      className={`px-4 py-2 text-right font-mono ${
                        (r.out_of_sample as any).net_pnl >= 0
                          ? "text-profit"
                          : "text-loss"
                      }`}
                    >
                      {(r.out_of_sample as any).net_pnl >= 0 ? "+" : ""}
                      {(r.out_of_sample as any).net_pnl?.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Selected Detail */}
      {selected && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* IS vs OOS Comparison */}
          <div className="bg-surface rounded-lg border border-border p-4">
            <h3 className="text-sm font-semibold text-white mb-4">
              In-Sample vs Out-of-Sample — #{selected.rank}
            </h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500 uppercase border-b border-border">
                  <th className="pb-2 text-left">Metrik</th>
                  <th className="pb-2 text-right">In-Sample</th>
                  <th className="pb-2 text-right">Out-Sample</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                {[
                  { key: "net_pnl", label: "Net P&L", fmt: (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}` },
                  { key: "profit_factor", label: "Profit Factor", fmt: (v: number) => v.toFixed(3) },
                  { key: "win_rate", label: "Win Rate", fmt: (v: number) => `${v.toFixed(1)}%` },
                  { key: "max_drawdown", label: "Max DD", fmt: (v: number) => `${v.toFixed(2)}%` },
                  { key: "total_trades", label: "Trades", fmt: (v: number) => String(Math.round(v)) },
                ].map(({ key, label, fmt }) => (
                  <tr key={key} className="border-t border-border/30">
                    <td className="py-2 text-slate-400">{label}</td>
                    <td className="py-2 text-right font-mono">
                      {fmt(selected.in_sample[key] || 0)}
                    </td>
                    <td className="py-2 text-right font-mono">
                      {fmt((selected.out_of_sample as any)[key] || 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Parameters */}
          <div className="bg-surface rounded-lg border border-border p-4">
            <h3 className="text-sm font-semibold text-white mb-4">
              Parametreler — #{selected.rank}
            </h3>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
              {Object.entries(selected.params).map(([key, val]) => (
                <div key={key} className="flex justify-between py-1 border-b border-border/20">
                  <span className="text-slate-500">{key}</span>
                  <span className="text-white font-mono">
                    {typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(4)) : val}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {results.length === 0 && !running && (
        <div className="bg-surface rounded-lg border border-border p-12 text-center">
          <Trophy className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500 text-sm">
            Henuz optimizasyon sonucu yok. Pair secip "Optimizasyon Baslat"
            butonuna basin.
          </p>
        </div>
      )}
    </div>
  );
}

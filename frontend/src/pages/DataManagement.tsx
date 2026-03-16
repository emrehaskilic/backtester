import { useEffect, useState, useRef } from "react";
import {
  Database,
  Download,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
} from "lucide-react";
import type { DataStatus } from "../api";
import {
  getDataStatus,
  startDownload,
  getDownloadProgress,
} from "../api";
import ProgressBar from "../components/ProgressBar";

export default function DataManagement() {
  const [statuses, setStatuses] = useState<DataStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [downloadSymbol, setDownloadSymbol] = useState("");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadMonth, setDownloadMonth] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const fetchStatuses = async () => {
    try {
      setLoading(true);
      const res = await getDataStatus();
      setStatuses(res.data.data);
    } catch {
      // API bagli degil
    } finally {
      setLoading(false);
    }
  };

  // On mount: check if a download is already running on backend
  const resumePollingIfActive = async () => {
    try {
      const prog = await getDownloadProgress();
      if (prog.data.running) {
        setDownloading(true);
        setDownloadSymbol(prog.data.symbol || "");
        setDownloadProgress(prog.data.progress);
        setDownloadMonth(prog.data.month || "");
        pollRef.current = setInterval(async () => {
          try {
            const p = await getDownloadProgress();
            setDownloadProgress(p.data.progress);
            setDownloadMonth(p.data.month || "");
            if (!p.data.running) {
              clearInterval(pollRef.current);
              setDownloading(false);
              fetchStatuses();
            }
          } catch {
            clearInterval(pollRef.current);
            setDownloading(false);
          }
        }, 2000);
      }
    } catch {
      // backend not reachable
    }
  };

  useEffect(() => {
    fetchStatuses();
    resumePollingIfActive();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleDownload = async (symbol: string) => {
    try {
      setDownloading(true);
      setDownloadSymbol(symbol);
      setDownloadProgress(0);
      setDownloadMonth("");
      await startDownload(symbol);

      pollRef.current = setInterval(async () => {
        try {
          const prog = await getDownloadProgress();
          setDownloadProgress(prog.data.progress);
          setDownloadMonth(prog.data.month || "");
          if (!prog.data.running) {
            clearInterval(pollRef.current);
            setDownloading(false);
            fetchStatuses();
          }
        } catch {
          clearInterval(pollRef.current);
          setDownloading(false);
        }
      }, 2000);
    } catch {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Header Actions */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          Binance aggTrades verisini indirin ve yonetin.
        </p>
        <button
          onClick={fetchStatuses}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-border text-xs text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className="w-3 h-3" />
          Yenile
        </button>
      </div>

      {/* Download Progress */}
      {downloading && (
        <div className="bg-surface rounded-lg border border-blue-500/30 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
              <span className="text-sm text-white font-medium">
                {downloadSymbol} indiriliyor...
              </span>
            </div>
            <span className="text-sm text-blue-400 font-mono font-bold">
              %{Math.max(0, downloadProgress)}
            </span>
          </div>
          <ProgressBar progress={downloadProgress} />
          {downloadMonth && (
            <p className="text-xs text-slate-400 mt-2">
              {downloadMonth}
            </p>
          )}
        </div>
      )}

      {/* Symbol Cards */}
      <div className="space-y-3">
        {statuses.map((d) => (
          <div
            key={d.symbol}
            className="bg-surface rounded-lg border border-border p-5 flex items-center justify-between"
          >
            <div className="flex items-center gap-4">
              {/* Status Icon */}
              {d.available ? (
                <CheckCircle2 className="w-8 h-8 text-profit shrink-0" />
              ) : (
                <XCircle className="w-8 h-8 text-loss shrink-0" />
              )}

              <div>
                <h3 className="text-white font-bold text-lg">{d.symbol}</h3>
                {d.available ? (
                  <div className="space-y-0.5 text-xs text-slate-400 mt-1">
                    <p>
                      <span className="text-slate-500">Boyut:</span>{" "}
                      {d.size_mb >= 1024
                        ? `${(d.size_mb / 1024).toFixed(1)} GB`
                        : `${d.size_mb.toFixed(0)} MB`}
                    </p>
                    <p>
                      <span className="text-slate-500">Tick:</span>{" "}
                      {(d.tick_count / 1_000_000).toFixed(1)}M
                    </p>
                    <p>
                      <span className="text-slate-500">Tarih:</span>{" "}
                      {d.date_from} &rarr; {d.date_to}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 mt-1">
                    Veri henuz indirilmedi
                  </p>
                )}
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={() => handleDownload(d.symbol)}
              disabled={downloading}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                d.available
                  ? "bg-surface-hover border border-border text-slate-300 hover:text-white"
                  : "bg-blue-600 text-white hover:bg-blue-500"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <Download className="w-4 h-4" />
              {d.available ? "Guncelle" : "Indir"}
            </button>
          </div>
        ))}
      </div>

      {loading && statuses.length === 0 && (
        <div className="bg-surface rounded-lg border border-border p-8 text-center">
          <Loader2 className="w-8 h-8 text-slate-600 mx-auto mb-3 animate-spin" />
          <p className="text-sm text-slate-500">Veri durumu yukleniyor...</p>
        </div>
      )}

      {!loading && statuses.length === 0 && (
        <div className="bg-surface rounded-lg border border-border p-8 text-center">
          <Database className="w-8 h-8 text-slate-600 mx-auto mb-3" />
          <p className="text-sm text-slate-500">
            API'ye baglanilamiyor. Backend'in calistigindan emin olun.
          </p>
          <code className="text-xs text-blue-400 mt-2 block">
            uvicorn api:app --reload --port 8000
          </code>
        </div>
      )}

      {/* Info Box */}
      <div className="bg-surface-dark rounded-lg border border-border p-4 text-xs text-slate-500">
        <p className="mb-1 font-medium text-slate-400">Bilgi</p>
        <ul className="space-y-1 list-disc list-inside">
          <li>Veriler Binance data.binance.vision'dan indirilir</li>
          <li>Son 6 aylik aggTrades verisi (aylik ZIP dosyalari)</li>
          <li>Indirilen veriler birlestirilip Parquet formatinda kaydedilir</li>
          <li>Her pair icin 10-30 GB disk alani gerekebilir</li>
          <li>
            Kayit yeri:{" "}
            <code className="text-blue-400">./data/&lt;symbol&gt;_aggtrades.parquet</code>
          </li>
        </ul>
      </div>
    </div>
  );
}

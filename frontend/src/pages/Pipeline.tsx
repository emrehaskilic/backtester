import { useState, useEffect, useRef } from 'react'
import { Play, Square, Plus, Trash2, Loader2, CheckCircle2, XCircle, Clock, Zap } from 'lucide-react'
import { usePipelineStore } from '../stores/pipelineStore'
import { useSettingsStore } from '../stores/settingsStore'
import { startPipeline, stopPipeline, getPipelineStatus } from '../api'
import ProgressBar from '../components/ProgressBar'

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT']
const STRATEGIES = [
  { value: 'pmax_kc', label: 'PMax + Keltner Channel' },
  { value: 'swinginess', label: 'Swinginess (Tick Replay)' },
]
const TIMEFRAMES = ['1m', '3m', '5m', '15m']
const ROUND_NAMES = ['', 'PMAX KEŞİF', 'FINE-TUNE', 'DD OPTİMİZE', 'KC OPTİMİZE', 'KC FINE-TUNE']

export default function Pipeline() {
  const store = usePipelineStore()
  const { settings, fetch: fetchSettings } = useSettingsStore()

  // Form state
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['BTCUSDT'])
  const [strategy, setStrategy] = useState('pmax_kc')
  const [selectedTFs, setSelectedTFs] = useState<string[]>(['3m'])
  const logEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    store.connect()
    fetchSettings()
    // Initial status fetch
    getPipelineStatus().then(res => {
      store.setQueue(res.data.queue || [])
    }).catch(() => {})

    return () => store.disconnect()
  }, [])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [store.logs])

  const handleStart = async () => {
    const pairs = selectedSymbols.map(sym => ({
      symbol: sym,
      strategy,
      timeframes: strategy === 'swinginess' ? ['tick'] : selectedTFs,
    }))
    try {
      await startPipeline(pairs)
    } catch (e: any) {
      store.addLog({ message: `Başlatma hatası: ${e.message}`, level: 'error' })
    }
  }

  const handleStop = async () => {
    try {
      await stopPipeline()
    } catch (e: any) {
      store.addLog({ message: `Durdurma hatası: ${e.message}`, level: 'error' })
    }
  }

  const toggleSymbol = (sym: string) => {
    setSelectedSymbols(prev =>
      prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]
    )
  }

  const toggleTF = (tf: string) => {
    setSelectedTFs(prev =>
      prev.includes(tf) ? prev.filter(t => t !== tf) : [...prev, tf]
    )
  }

  const statusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-4 h-4 text-green-400" />
      case 'running': return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />
      case 'queued': return <Clock className="w-4 h-4 text-yellow-400" />
      default: return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const statusLabel = (status: string) => {
    const map: Record<string, string> = {
      queued: 'Bekliyor', running: 'Çalışıyor', completed: 'Tamamlandı',
      failed: 'Başarısız', downloading: 'İndiriliyor',
    }
    return map[status] || status
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* SOL PANEL — Başlatma */}
      <div className="space-y-4">
        {/* Pair Seçimi */}
        <div className="bg-surface rounded-xl p-4 border border-border">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Sembol Seçimi</h3>
          <div className="grid grid-cols-3 gap-2">
            {SYMBOLS.map(sym => (
              <button
                key={sym}
                onClick={() => toggleSymbol(sym)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedSymbols.includes(sym)
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                    : 'bg-surface-hover text-gray-400 border border-border hover:border-gray-500'
                }`}
              >
                {sym.replace('USDT', '')}
              </button>
            ))}
          </div>
        </div>

        {/* Strateji Seçimi */}
        <div className="bg-surface rounded-xl p-4 border border-border">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Strateji</h3>
          <div className="space-y-2">
            {STRATEGIES.map(s => (
              <button
                key={s.value}
                onClick={() => setStrategy(s.value)}
                className={`w-full px-4 py-3 rounded-lg text-left transition-all ${
                  strategy === s.value
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                    : 'bg-surface-hover text-gray-400 border border-border hover:border-gray-500'
                }`}
              >
                <div className="font-medium">{s.label}</div>
                <div className="text-xs mt-1 opacity-60">
                  {s.value === 'pmax_kc' ? '5 round otomatik pipeline (R1→R5)' : 'Tek geçişli Optuna optimizasyonu'}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Timeframe Seçimi (sadece PMax+KC) */}
        {strategy === 'pmax_kc' && (
          <div className="bg-surface rounded-xl p-4 border border-border">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Timeframe</h3>
            <div className="flex gap-2">
              {TIMEFRAMES.map(tf => (
                <button
                  key={tf}
                  onClick={() => toggleTF(tf)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    selectedTFs.includes(tf)
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                      : 'bg-surface-hover text-gray-400 border border-border hover:border-gray-500'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Başlat / Durdur */}
        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={store.running || selectedSymbols.length === 0}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-xl font-bold text-lg transition-all"
          >
            {store.running ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            {store.running ? 'ÇALIŞIYOR...' : 'BAŞLAT'}
          </button>
          {store.running && (
            <button
              onClick={handleStop}
              className="px-6 py-4 bg-red-600 hover:bg-red-700 text-white rounded-xl font-bold transition-all"
            >
              <Square className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Kuyruk */}
        <div className="bg-surface rounded-xl p-4 border border-border">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Kuyruk</h3>
          {store.queue.length === 0 ? (
            <p className="text-gray-500 text-sm">Kuyruk boş — yukarıdan pair seçip başlatın</p>
          ) : (
            <div className="space-y-2">
              {store.queue.map(job => (
                <div
                  key={job.id}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg border ${
                    job.status === 'running' ? 'border-blue-500/50 bg-blue-500/10' : 'border-border bg-surface-hover'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {statusIcon(job.status)}
                    <span className="font-medium text-sm">{job.symbol}</span>
                    <span className="text-xs text-gray-500">{job.strategy}</span>
                    {job.current_timeframe && (
                      <span className="text-xs text-blue-400">{job.current_timeframe}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {job.status === 'running' && (
                      <span className="text-xs text-blue-400">
                        R{job.current_round}/{job.total_rounds}
                      </span>
                    )}
                    <span className="text-xs text-gray-500">{statusLabel(job.status)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* SAĞ PANEL — Canlı İzleme */}
      <div className="space-y-4">
        {/* Round Timeline */}
        {store.running && (
          <div className="bg-surface rounded-xl p-4 border border-border">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              <Zap className="w-4 h-4 inline mr-1 text-yellow-400" />
              Aktif Optimizasyon
            </h3>

            {/* Round indicators */}
            <div className="flex gap-1 mb-4">
              {[1, 2, 3, 4, 5].map(r => (
                <div
                  key={r}
                  className={`flex-1 h-2 rounded-full ${
                    r < store.currentRound ? 'bg-green-500'
                    : r === store.currentRound ? 'bg-blue-500 animate-pulse'
                    : 'bg-gray-700'
                  }`}
                  title={ROUND_NAMES[r]}
                />
              ))}
            </div>

            <div className="text-center mb-3">
              <div className="text-lg font-bold text-white">
                R{store.currentRound}: {ROUND_NAMES[store.currentRound] || store.currentRoundName}
              </div>
              <div className="text-sm text-gray-400">
                Trial {store.completedTrials} / {store.totalTrials}
              </div>
            </div>

            <ProgressBar
              progress={store.totalTrials > 0 ? Math.round(store.completedTrials / store.totalTrials * 100) : 0}
            />

            {/* Best metrics */}
            <div className="grid grid-cols-3 gap-3 mt-4">
              <div className="text-center">
                <div className="text-xs text-gray-500">En İyi OOS</div>
                <div className={`text-lg font-bold ${store.bestOosNet > 0 ? 'text-green-400' : 'text-gray-400'}`}>
                  {store.bestOosNet > 0 ? `+${store.bestOosNet.toFixed(1)}%` : '—'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500">En Düşük DD</div>
                <div className="text-lg font-bold text-yellow-400">
                  {store.bestDD > 0 ? `${store.bestDD.toFixed(1)}%` : '—'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500">En İyi Ratio</div>
                <div className="text-lg font-bold text-blue-400">
                  {store.bestRatio > 0 ? store.bestRatio.toFixed(1) : '—'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Top 10 Trials (son round) */}
        {store.top10.length > 0 && (
          <div className="bg-surface rounded-xl p-4 border border-border">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Son Round Top 10
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-border">
                    <th className="text-left py-1 px-2">#</th>
                    <th className="text-right py-1 px-2">OOS%</th>
                    <th className="text-right py-1 px-2">DD%</th>
                    <th className="text-right py-1 px-2">Ratio</th>
                    <th className="text-right py-1 px-2">WR%</th>
                  </tr>
                </thead>
                <tbody>
                  {store.top10.map((t, i) => (
                    <tr key={t.tid} className={`border-b border-border/30 ${i === 0 ? 'bg-green-500/10' : ''}`}>
                      <td className="py-1 px-2 text-gray-400">{i + 1}</td>
                      <td className={`py-1 px-2 text-right font-medium ${t.oos_net > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {t.oos_net > 0 ? '+' : ''}{t.oos_net.toFixed(1)}
                      </td>
                      <td className="py-1 px-2 text-right text-yellow-400">{t.dd.toFixed(1)}</td>
                      <td className="py-1 px-2 text-right text-blue-400 font-medium">{t.ratio.toFixed(1)}</td>
                      <td className="py-1 px-2 text-right text-gray-300">{t.wr.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Log Viewer */}
        <div className="bg-surface rounded-xl p-4 border border-border">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-medium text-gray-400">Log</h3>
            <button
              onClick={() => store.clearLogs()}
              className="text-xs text-gray-500 hover:text-gray-300"
            >
              Temizle
            </button>
          </div>
          <div className="h-64 overflow-y-auto font-mono text-xs space-y-0.5 bg-surface-dark rounded-lg p-2">
            {store.logs.length === 0 ? (
              <p className="text-gray-600">Pipeline başlatıldığında loglar burada görünecek...</p>
            ) : (
              store.logs.map((log, i) => (
                <div
                  key={i}
                  className={`${
                    log.level === 'error' ? 'text-red-400'
                    : log.level === 'success' ? 'text-green-400'
                    : log.level === 'warn' ? 'text-yellow-400'
                    : 'text-gray-400'
                  }`}
                >
                  <span className="text-gray-600">[{log.timestamp}]</span> {log.message}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  )
}

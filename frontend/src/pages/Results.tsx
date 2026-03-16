import { useState, useEffect } from 'react'
import { Trophy, ChevronDown, ChevronUp } from 'lucide-react'
import { getAllResults } from '../api'

interface ResultEntry {
  id: string
  symbol: string
  strategy: string
  timeframe: string
  date: string
  source: string
  metrics?: Record<string, any>
  final_result?: any
}

export default function Results() {
  const [results, setResults] = useState<ResultEntry[]>([])
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    setLoading(true)
    try {
      const res = await getAllResults()
      setResults(res.data.results || [])
    } catch (e) {
      console.error('Results load error:', e)
    } finally {
      setLoading(false)
    }
  }

  const getLastRoundMetrics = (entry: ResultEntry) => {
    if (!entry.metrics) return null
    const keys = Object.keys(entry.metrics).sort()
    return keys.length > 0 ? entry.metrics[keys[keys.length - 1]] : null
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Trophy className="w-5 h-5 text-yellow-400" />
          Optimizasyon Sonuçları
        </h2>
        <button
          onClick={loadResults}
          className="px-3 py-1.5 text-sm bg-surface-hover border border-border rounded-lg hover:border-gray-500 text-gray-400"
        >
          Yenile
        </button>
      </div>

      {loading ? (
        <div className="text-center py-12 text-gray-500">Yükleniyor...</div>
      ) : results.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          Henüz tamamlanmış optimizasyon yok. Pipeline sayfasından başlatın.
        </div>
      ) : (
        <div className="space-y-2">
          {results.map(entry => {
            const metrics = getLastRoundMetrics(entry)
            const isExpanded = expandedId === entry.id

            return (
              <div key={entry.id} className="bg-surface rounded-xl border border-border overflow-hidden">
                <button
                  onClick={() => setExpandedId(isExpanded ? null : entry.id)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-surface-hover transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <span className="font-bold text-white">{entry.symbol}</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400">
                      {entry.strategy === 'pmax_kc' ? 'PMax+KC' : 'Swinginess'}
                    </span>
                    <span className="text-xs text-gray-500">{entry.timeframe}</span>
                    {entry.date && (
                      <span className="text-xs text-gray-600">
                        {new Date(entry.date).toLocaleDateString('tr-TR')}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-4">
                    {metrics && (
                      <>
                        <span className={`text-sm font-medium ${(metrics.oos_net ?? 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {(metrics.oos_net ?? 0) > 0 ? '+' : ''}{(metrics.oos_net ?? 0).toFixed(1)}%
                        </span>
                        <span className="text-sm text-yellow-400">DD {(metrics.dd ?? 0).toFixed(1)}%</span>
                        <span className="text-sm text-blue-400">R={(metrics.ratio ?? 0).toFixed(1)}</span>
                      </>
                    )}
                    {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
                  </div>
                </button>

                {isExpanded && entry.final_result && (
                  <div className="px-4 pb-4 border-t border-border pt-3">
                    {/* PMax Params */}
                    {entry.final_result.pmax_params && (
                      <div className="mb-3">
                        <h4 className="text-xs font-medium text-gray-500 mb-1">PMax Parametreleri</h4>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          {Object.entries(entry.final_result.pmax_params).map(([k, v]) => (
                            <div key={k} className="flex justify-between bg-surface-hover rounded px-2 py-1">
                              <span className="text-gray-500">{k}</span>
                              <span className="text-white font-medium">{String(v)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* KC Params */}
                    {entry.final_result.kc_params && (
                      <div className="mb-3">
                        <h4 className="text-xs font-medium text-gray-500 mb-1">KC Parametreleri</h4>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          {Object.entries(entry.final_result.kc_params).map(([k, v]) => (
                            <div key={k} className="flex justify-between bg-surface-hover rounded px-2 py-1">
                              <span className="text-gray-500">{k}</span>
                              <span className="text-white font-medium">{String(v)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Round by round metrics */}
                    {entry.final_result.selected_per_round && (
                      <div>
                        <h4 className="text-xs font-medium text-gray-500 mb-1">Round Sonuçları</h4>
                        <pre className="text-xs text-gray-400 bg-surface-dark rounded p-2 overflow-x-auto">
                          {JSON.stringify(entry.metrics || entry.final_result.selected_metrics, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

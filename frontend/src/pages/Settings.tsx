import { useEffect, useState } from 'react'
import { Settings2, Save } from 'lucide-react'
import { useSettingsStore } from '../stores/settingsStore'

export default function Settings() {
  const { settings, loading, fetch, update } = useSettingsStore()
  const [local, setLocal] = useState(settings)
  const [saved, setSaved] = useState(false)

  useEffect(() => { fetch() }, [])
  useEffect(() => { setLocal(settings) }, [settings])

  const handleSave = async () => {
    await update(local)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const setField = (key: string, value: any) => {
    setLocal(prev => ({ ...prev, [key]: value }))
  }

  const setTrialCount = (round: string, value: number) => {
    setLocal(prev => ({
      ...prev,
      trial_counts: { ...prev.trial_counts, [round]: value },
    }))
  }

  return (
    <div className="max-w-2xl space-y-6">
      <h2 className="text-lg font-bold text-white flex items-center gap-2">
        <Settings2 className="w-5 h-5 text-gray-400" />
        Ayarlar
      </h2>

      {/* Genel */}
      <div className="bg-surface rounded-xl p-5 border border-border space-y-4">
        <h3 className="text-sm font-medium text-gray-400">Genel</h3>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Kaldıraç (Leverage)</label>
            <input
              type="number"
              min={1} max={125}
              value={local.leverage}
              onChange={e => setField('leverage', parseInt(e.target.value) || 25)}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-white text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Lookback Gün</label>
            <input
              type="number"
              min={30} max={365}
              value={local.days}
              onChange={e => setField('days', parseInt(e.target.value) || 180)}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-white text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Paralel İş (n_jobs)</label>
            <input
              type="number"
              min={1} max={16}
              value={local.n_jobs}
              onChange={e => setField('n_jobs', parseInt(e.target.value) || 4)}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-white text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Oto-Seçim Metodu</label>
            <select
              value={local.auto_select_method}
              onChange={e => setField('auto_select_method', e.target.value)}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-white text-sm"
            >
              <option value="ratio">Ratio (Net/DD)</option>
              <option value="score">Score (Optuna)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Trial Counts */}
      <div className="bg-surface rounded-xl p-5 border border-border space-y-4">
        <h3 className="text-sm font-medium text-gray-400">Round Bazlı Trial Sayıları</h3>
        <div className="grid grid-cols-5 gap-3">
          {['r1', 'r2', 'r3', 'r4', 'r5'].map((r, i) => (
            <div key={r}>
              <label className="text-xs text-gray-500 block mb-1">
                R{i + 1}
              </label>
              <input
                type="number"
                min={50} max={1000} step={50}
                value={local.trial_counts?.[r] ?? 200}
                onChange={e => setTrialCount(r, parseInt(e.target.value) || 200)}
                className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-white text-sm text-center"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Compounding */}
      <div className="bg-surface rounded-xl p-5 border border-border space-y-4">
        <h3 className="text-sm font-medium text-gray-400">Dinamik Compounding</h3>
        <div className="grid grid-cols-2 gap-3 text-sm">
          {Object.entries(local.compounding || {}).map(([tier, pct]) => (
            <div key={tier} className="flex items-center justify-between bg-surface-hover rounded-lg px-3 py-2">
              <span className="text-gray-400">{tier}</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={1} max={50}
                  value={pct as number}
                  onChange={e => setLocal(prev => ({
                    ...prev,
                    compounding: {
                      ...prev.compounding,
                      [tier]: parseInt(e.target.value) || 10,
                    },
                  }))}
                  className="w-16 px-2 py-1 bg-surface-dark border border-border rounded text-white text-sm text-center"
                />
                <span className="text-gray-500">%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Kaydet */}
      <button
        onClick={handleSave}
        disabled={loading}
        className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
          saved
            ? 'bg-green-600 text-white'
            : 'bg-blue-600 hover:bg-blue-700 text-white'
        }`}
      >
        <Save className="w-4 h-4" />
        {saved ? 'Kaydedildi!' : 'Kaydet'}
      </button>
    </div>
  )
}

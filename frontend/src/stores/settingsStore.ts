import { create } from 'zustand'
import api from '../api'

interface Settings {
  leverage: number
  days: number
  trial_counts: Record<string, number>
  auto_select_method: string
  compounding: Record<string, number>
  n_jobs: number
}

interface SettingsState {
  settings: Settings
  loading: boolean
  fetch: () => Promise<void>
  update: (updates: Partial<Settings>) => Promise<void>
}

const DEFAULT_SETTINGS: Settings = {
  leverage: 25,
  days: 180,
  trial_counts: { r1: 200, r2: 300, r3: 400, r4: 300, r5: 200 },
  auto_select_method: 'ratio',
  compounding: { '<50K': 10, '50-100K': 10, '100-200K': 5, '200K+': 2 },
  n_jobs: 4,
}

export const useSettingsStore = create<SettingsState>((set) => ({
  settings: DEFAULT_SETTINGS,
  loading: false,

  fetch: async () => {
    set({ loading: true })
    try {
      const res = await api.get('/api/settings')
      set({ settings: { ...DEFAULT_SETTINGS, ...res.data } })
    } catch (e) {
      console.error('Settings fetch error:', e)
    } finally {
      set({ loading: false })
    }
  },

  update: async (updates) => {
    try {
      const res = await api.post('/api/settings', updates)
      set({ settings: { ...DEFAULT_SETTINGS, ...res.data } })
    } catch (e) {
      console.error('Settings update error:', e)
    }
  },
}))

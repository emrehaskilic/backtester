import { create } from 'zustand'

export interface PipelineJob {
  id: string
  symbol: string
  strategy: string
  timeframes: string[]
  status: 'queued' | 'downloading' | 'running' | 'completed' | 'failed'
  current_round: number
  current_timeframe: string | null
  total_rounds: number
  selected_params_per_round: Record<string, any>
  selected_metrics_per_round: Record<string, any>
  final_result: any
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface TrialData {
  tid: number
  score: number
  is_net: number
  oos_net: number
  wr: number
  dd: number
  ratio: number
  trades: number
  params?: Record<string, any>
}

export interface LogEntry {
  message: string
  level: string
  timestamp?: string
}

interface PipelineState {
  // Connection
  ws: WebSocket | null
  connected: boolean

  // Pipeline status
  running: boolean
  queue: PipelineJob[]
  currentJob: PipelineJob | null

  // Current round info
  currentRound: number
  currentRoundName: string
  totalTrials: number
  completedTrials: number
  bestScore: number
  bestOosNet: number
  bestDD: number
  bestRatio: number

  // Top trials for current round
  top10: TrialData[]

  // Logs
  logs: LogEntry[]

  // Actions
  connect: () => void
  disconnect: () => void
  setQueue: (queue: PipelineJob[]) => void
  addLog: (log: LogEntry) => void
  clearLogs: () => void
  reset: () => void
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  ws: null,
  connected: false,
  running: false,
  queue: [],
  currentJob: null,
  currentRound: 0,
  currentRoundName: '',
  totalTrials: 0,
  completedTrials: 0,
  bestScore: 0,
  bestOosNet: 0,
  bestDD: 0,
  bestRatio: 0,
  top10: [],
  logs: [],

  connect: () => {
    const ws = new WebSocket('ws://localhost:8055/ws/pipeline')

    ws.onopen = () => {
      set({ ws, connected: true })
    }

    ws.onclose = () => {
      set({ ws: null, connected: false })
      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (!get().connected) get().connect()
      }, 3000)
    }

    ws.onerror = () => {
      set({ connected: false })
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        const { type, data } = msg

        switch (type) {
          case 'pipeline_status':
            set({
              running: data.running ?? data.pipeline_running ?? false,
              queue: data.queue ?? [],
              currentJob: data.current ?? null,
            })
            break

          case 'round_started':
            set({
              currentRound: data.round,
              currentRoundName: data.name,
              totalTrials: data.n_trials,
              completedTrials: 0,
              bestScore: 0,
              bestOosNet: 0,
              bestDD: 0,
              bestRatio: 0,
              top10: [],
            })
            break

          case 'trial_completed':
            set(s => {
              const newCompleted = s.completedTrials + 1
              return {
                completedTrials: newCompleted,
                bestScore: Math.max(s.bestScore, data.score ?? 0),
                bestOosNet: data.oos_net > s.bestOosNet ? data.oos_net : s.bestOosNet,
                bestDD: data.dd < s.bestDD || s.bestDD === 0 ? data.dd : s.bestDD,
                bestRatio: Math.max(s.bestRatio, data.ratio ?? 0),
              }
            })
            break

          case 'round_completed':
            set({
              top10: data.top10 ?? [],
              currentRound: data.round,
            })
            get().addLog({
              message: `Round ${data.round} (${data.name}) tamamlandı — Seçilen: OOS=${data.selected?.oos_net}% DD=${data.selected?.dd}%`,
              level: 'success',
            })
            break

          case 'pair_completed':
            get().addLog({
              message: `${data.symbol} tamamlandı!`,
              level: 'success',
            })
            break

          case 'log':
            get().addLog(data)
            break

          case 'error':
            get().addLog({ message: data.message, level: 'error' })
            break
        }
      } catch (e) {
        console.error('WS parse error:', e)
      }
    }

    set({ ws })
  },

  disconnect: () => {
    const { ws } = get()
    if (ws) ws.close()
    set({ ws: null, connected: false })
  },

  setQueue: (queue) => set({ queue }),

  addLog: (log) => set(s => ({
    logs: [...s.logs.slice(-299), { ...log, timestamp: new Date().toLocaleTimeString() }]
  })),

  clearLogs: () => set({ logs: [] }),

  reset: () => set({
    running: false,
    currentRound: 0,
    currentRoundName: '',
    totalTrials: 0,
    completedTrials: 0,
    bestScore: 0,
    bestOosNet: 0,
    bestDD: 0,
    bestRatio: 0,
    top10: [],
  }),
}))

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

export interface InvEvent {
  timestamp: string
  type: 'logon' | 'usb' | 'file' | 'email' | 'http'
  description: string
  suspicious: boolean
}

export interface ScorePoint {
  date: string
  score: number
}

export interface EmployeeMeta {
  name: string
  department: string
  role: string
  latest_score: number
  alert_count: number
  status: string
}

export interface CurrentAlert {
  id: string
  user: string
  date: string
  score: number
  status: string
}

interface InvestigationState {
  selectedUser: string | null
  activity: InvEvent[] | null
  scoreHistory: ScorePoint[] | null
  employeeMeta: EmployeeMeta | null
  currentAlert: CurrentAlert | null
  status: 'idle' | 'loading' | 'succeeded' | 'failed'
  actionInProgress: 'learn' | 'mute' | 'block' | null
  notification: string | null
}

const initialState: InvestigationState = {
  selectedUser: null,
  activity: null,
  scoreHistory: null,
  employeeMeta: null,
  currentAlert: null,
  status: 'idle',
  actionInProgress: null,
  notification: null,
}

// ── helpers ──────────────────────────────────────────────────────────────────

interface EmpRow {
  user: string
  department: string
  latest_score: number
  alert_count: number
  status: string
}

interface AlertRow {
  id: string
  user: string
  date: string
  score: number
  status: string
}

// ── thunks ────────────────────────────────────────────────────────────────────

export const fetchInvestigation = createAsyncThunk(
  'investigation/fetchInvestigation',
  async (userId: string) => {
    const [activityData, historyData, employeesData, alertsData] = await Promise.all([
      fetch(`/api/employee/${userId}/activity`).then(r => r.ok ? r.json() : { events: [] }),
      fetch(`/api/employee/${userId}/score-history?days=30`).then(r => r.ok ? r.json() : []),
      fetch('/api/employees').then(r => r.ok ? r.json() : []),
      fetch('/api/alerts').then(r => r.ok ? r.json() : []),
    ])

    const emp = (employeesData as EmpRow[]).find(e => e.user === userId)
    const employeeMeta: EmployeeMeta | null = emp
      ? {
          name: emp.user,
          department: emp.department,
          role: 'N/A',
          latest_score: emp.latest_score,
          alert_count: emp.alert_count,
          status: emp.status,
        }
      : null

    const userAlerts = (alertsData as AlertRow[]).filter(a => a.user === userId)
    const rawAlert = userAlerts.find(a => a.status === 'open') ?? userAlerts[0] ?? null
    const currentAlert: CurrentAlert | null = rawAlert
      ? { id: rawAlert.id, user: rawAlert.user, date: rawAlert.date, score: rawAlert.score, status: rawAlert.status }
      : null

    return {
      userId,
      activity: ((activityData as { events?: InvEvent[] }).events ?? []) as InvEvent[],
      scoreHistory: historyData as ScorePoint[],
      employeeMeta,
      currentAlert,
    }
  },
)

export const submitLearn = createAsyncThunk(
  'investigation/submitLearn',
  async (alertId: string) => {
    const res = await fetch(`/api/alert/${alertId}/learn`, { method: 'POST' })
    if (!res.ok) throw new Error('Failed to submit')
    const data = await res.json()
    return { status: String(data.alert.status) }
  },
)

export const submitMute = createAsyncThunk(
  'investigation/submitMute',
  async (alertId: string) => {
    const res = await fetch(`/api/alert/${alertId}/mute`, { method: 'POST' })
    if (!res.ok) throw new Error('Failed to submit')
    const data = await res.json()
    return { status: String(data.alert.status) }
  },
)

export const submitBlock = createAsyncThunk(
  'investigation/submitBlock',
  async (alertId: string) => {
    const res = await fetch(`/api/alert/${alertId}/block`, { method: 'POST' })
    if (!res.ok) throw new Error('Failed to submit')
    const data = await res.json()
    return { status: String(data.alert.status), notification: String(data.notification) }
  },
)

// ── slice ─────────────────────────────────────────────────────────────────────

const investigationSlice = createSlice({
  name: 'investigation',
  initialState,
  reducers: {
    clearNotification(state) {
      state.notification = null
    },
    resetInvestigation(state) {
      state.selectedUser    = null
      state.activity        = null
      state.scoreHistory    = null
      state.employeeMeta    = null
      state.currentAlert    = null
      state.status          = 'idle'
      state.actionInProgress = null
      state.notification    = null
    },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchInvestigation.pending,   state => { state.status = 'loading' })
      .addCase(fetchInvestigation.fulfilled, (state, action) => {
        state.status        = 'succeeded'
        state.selectedUser  = action.payload.userId
        state.activity      = action.payload.activity
        state.scoreHistory  = action.payload.scoreHistory
        state.employeeMeta  = action.payload.employeeMeta
        state.currentAlert  = action.payload.currentAlert
      })
      .addCase(fetchInvestigation.rejected,  state => { state.status = 'failed' })

      .addCase(submitLearn.pending,    state => { state.actionInProgress = 'learn' })
      .addCase(submitLearn.fulfilled,  (state, action) => {
        state.actionInProgress = null
        if (state.currentAlert) state.currentAlert.status = action.payload.status
      })
      .addCase(submitLearn.rejected,   state => { state.actionInProgress = null })

      .addCase(submitMute.pending,     state => { state.actionInProgress = 'mute' })
      .addCase(submitMute.fulfilled,   (state, action) => {
        state.actionInProgress = null
        if (state.currentAlert) state.currentAlert.status = action.payload.status
      })
      .addCase(submitMute.rejected,    state => { state.actionInProgress = null })

      .addCase(submitBlock.pending,    state => { state.actionInProgress = 'block' })
      .addCase(submitBlock.fulfilled,  (state, action) => {
        state.actionInProgress = null
        state.notification     = action.payload.notification
        if (state.currentAlert) state.currentAlert.status = action.payload.status
      })
      .addCase(submitBlock.rejected,   state => { state.actionInProgress = null })
  },
})

export const { clearNotification, resetInvestigation } = investigationSlice.actions
export default investigationSlice.reducer

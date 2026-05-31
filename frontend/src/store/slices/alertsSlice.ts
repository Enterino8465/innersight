import { createSlice, createAsyncThunk, type PayloadAction } from '@reduxjs/toolkit'

export interface Alert {
  id: string
  user: string
  date: string
  score: number
  status: 'open' | 'muted' | 'learned' | 'blocked'
  created_at: string
  top_features: string[]
  department?: string
}

interface AlertsState {
  items: Alert[]
  status: 'idle' | 'loading' | 'succeeded' | 'failed'
  error?: string
}

const initialState: AlertsState = {
  items: [],
  status: 'idle',
}

export const fetchAlerts = createAsyncThunk('alerts/fetchAlerts', async () => {
  const res = await fetch('/api/alerts?status=open')
  if (!res.ok) throw new Error('Failed to fetch alerts')
  return (await res.json()) as Alert[]
})

const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    clearAlerts(state) {
      state.items = []
      state.status = 'idle'
      state.error = undefined
    },
    updateAlertStatus(
      state,
      action: PayloadAction<{ id: string; status: Alert['status'] }>,
    ) {
      const alert = state.items.find(a => a.id === action.payload.id)
      if (alert) alert.status = action.payload.status
    },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchAlerts.pending, state => {
        state.status = 'loading'
        state.error = undefined
      })
      .addCase(fetchAlerts.fulfilled, (state, action) => {
        state.status = 'succeeded'
        state.items = action.payload
      })
      .addCase(fetchAlerts.rejected, (state, action) => {
        state.status = 'failed'
        state.error = action.error.message
      })
  },
})

export const { clearAlerts, updateAlertStatus } = alertsSlice.actions
export default alertsSlice.reducer

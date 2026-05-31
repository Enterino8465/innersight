import { createSlice, createAsyncThunk, type PayloadAction } from '@reduxjs/toolkit'

export interface TrainingConfig {
  epochs: number
  batch_size: number
  lr: number
  layer_sizes: number[]
  pos_weight: number
  patience: number
}

export interface HistoryEntry {
  epoch: number
  train_loss: number
  val_loss: number
  val_f1: number
  val_recall: number
  val_precision: number
}

interface TrainingState {
  status: 'idle' | 'running' | 'done' | 'failed'
  epoch: number
  totalEpochs: number
  history: HistoryEntry[]
  confusionMatrix: number[][] | null
  classImbalanceWarning: { ratio: number } | null
}

const initialState: TrainingState = {
  status: 'idle',
  epoch: 0,
  totalEpochs: 10,
  history: [],
  confusionMatrix: null,
  classImbalanceWarning: null,
}

// ── module-level streaming refs ───────────────────────────────────────────────

let _es: EventSource | null = null
let _cancelled = false

// Closes the SSE connection without cancelling server-side training.
// Called on component unmount so navigating away leaves no lingering connections.
export const closeStream = () => {
  _cancelled = true
  if (_es) { _es.close(); _es = null }
}

// ── thunks ────────────────────────────────────────────────────────────────────

export const startTraining = createAsyncThunk(
  'training/startTraining',
  async (config: TrainingConfig, { dispatch }) => {
    _cancelled = false
    if (_es) { _es.close(); _es = null }

    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    if (!res.ok) throw new Error('Failed to start training')

    let batchLossSum = 0
    let batchCount   = 0

    await new Promise<void>((resolve, reject) => {
      _es = new EventSource('/api/events')

      _es.onmessage = (e: MessageEvent) => {
        const data = JSON.parse(e.data as string) as Record<string, unknown>

        if (data.class_imbalance) {
          dispatch(classImbalanceReceived(data.class_imbalance as { ratio: number }))
          return
        }

        if (data.status === 'done') {
          dispatch(trainingComplete({ confusionMatrix: data.confusion_matrix as number[][] }))
          _es?.close(); _es = null
          resolve()
          return
        }

        if (data.status === 'error') {
          _es?.close(); _es = null
          reject(new Error((data.message as string) ?? 'Training error'))
          return
        }

        if (data.batch !== undefined) {
          batchLossSum += data.loss as number
          batchCount++
          dispatch(batchReceived({ epoch: data.epoch as number }))
          return
        }

        if (data.val_loss !== undefined) {
          const tl = batchCount > 0 ? Math.round((batchLossSum / batchCount) * 1e6) / 1e6 : 0
          batchLossSum = 0
          batchCount   = 0
          dispatch(epochReceived({
            epoch:         data.epoch         as number,
            train_loss:    tl,
            val_loss:      data.val_loss      as number,
            val_f1:        data.val_f1        as number,
            val_recall:    data.val_recall    as number,
            val_precision: data.val_precision as number,
          }))
        }
      }

      _es.onerror = () => {
        _es?.close(); _es = null
        if (_cancelled) { resolve() } else { reject(new Error('Connection lost')) }
      }
    })
  },
)

export const cancelTraining = createAsyncThunk(
  'training/cancelTraining',
  async (_: undefined, { dispatch }) => {
    _cancelled = true
    if (_es) { _es.close(); _es = null }
    await fetch('/api/train/cancel', { method: 'POST' }).catch(() => {})
    dispatch(reset())
  },
)

// ── slice ─────────────────────────────────────────────────────────────────────

const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    batchReceived(state, action: PayloadAction<{ epoch: number }>) {
      state.epoch = action.payload.epoch
    },
    epochReceived(state, action: PayloadAction<HistoryEntry>) {
      state.epoch = action.payload.epoch
      state.history.push(action.payload)
    },
    trainingComplete(state, action: PayloadAction<{ confusionMatrix: number[][] }>) {
      state.status          = 'done'
      state.confusionMatrix = action.payload.confusionMatrix
    },
    classImbalanceReceived(state, action: PayloadAction<{ ratio: number }>) {
      state.classImbalanceWarning = action.payload
    },
    reset(state) {
      state.status               = 'idle'
      state.epoch                = 0
      state.history              = []
      state.confusionMatrix      = null
      state.classImbalanceWarning = null
    },
  },
  extraReducers: builder => {
    builder
      .addCase(startTraining.pending, (state, action) => {
        state.status      = 'running'
        state.totalEpochs = action.meta.arg.epochs
        state.epoch       = 0
        state.history     = []
        state.confusionMatrix = null
      })
      .addCase(startTraining.fulfilled, () => { /* trainingComplete handles state */ })
      .addCase(startTraining.rejected,  state => {
        if (state.status === 'running') state.status = 'failed'
      })
      .addCase(cancelTraining.fulfilled, () => { /* reset() dispatched from thunk */ })
  },
})

export const {
  batchReceived,
  epochReceived,
  trainingComplete,
  classImbalanceReceived,
  reset,
} = trainingSlice.actions

export default trainingSlice.reducer

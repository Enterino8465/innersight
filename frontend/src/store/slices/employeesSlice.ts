import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

export interface Employee {
  user: string
  department: string
  latest_score: number
  alert_count: number
  status: 'open' | 'muted' | 'learned' | 'blocked'
}

interface EmployeesState {
  items: Employee[]
  status: 'idle' | 'loading' | 'succeeded' | 'failed'
  error?: string
}

const initialState: EmployeesState = {
  items: [],
  status: 'idle',
}

export const fetchEmployees = createAsyncThunk('employees/fetchEmployees', async () => {
  const res = await fetch('/api/employees')
  if (!res.ok) throw new Error('Failed to fetch employees')
  return (await res.json()) as Employee[]
})

const employeesSlice = createSlice({
  name: 'employees',
  initialState,
  reducers: {},
  extraReducers: builder => {
    builder
      .addCase(fetchEmployees.pending, state => {
        state.status = 'loading'
        state.error = undefined
      })
      .addCase(fetchEmployees.fulfilled, (state, action) => {
        state.status = 'succeeded'
        state.items = action.payload
      })
      .addCase(fetchEmployees.rejected, (state, action) => {
        state.status = 'failed'
        state.error = action.error.message
      })
  },
})

export default employeesSlice.reducer

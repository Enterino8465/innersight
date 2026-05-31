import { configureStore } from '@reduxjs/toolkit'
import alertsReducer from './slices/alertsSlice'
import employeesReducer from './slices/employeesSlice'
import investigationReducer from './slices/investigationSlice'
import trainingReducer from './slices/trainingSlice'
import uiReducer from './slices/uiSlice'

const store = configureStore({
  reducer: {
    alerts: alertsReducer,
    employees: employeesReducer,
    investigation: investigationReducer,
    training: trainingReducer,
    ui: uiReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export default store

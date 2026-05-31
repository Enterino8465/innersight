import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Provider } from 'react-redux'
import store from './store/store'
import Navbar from './components/Navbar'
import ErrorBoundary from './components/ErrorBoundary'
import AlertsPage from './pages/Alerts'
import EmployeesPage from './pages/Employees'
import InvestigationPage from './pages/Investigation'
import TrainingPage from './pages/Training'
import NotFound from './pages/NotFound'

const App: React.FC = () => (
  <ErrorBoundary>
  <Provider store={store}>
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Navigate to="/alerts" replace />} />
        <Route path="/alerts" element={<AlertsPage />} />
        <Route path="/employees" element={<EmployeesPage />} />
        <Route path="/employee/:userId" element={<InvestigationPage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  </Provider>
  </ErrorBoundary>
)

export default App

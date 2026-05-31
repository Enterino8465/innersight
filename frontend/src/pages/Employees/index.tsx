import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import { fetchEmployees } from '../../store/slices/employeesSlice'
import Spinner from '../../components/Spinner'
import { useEmployeesFilter, type RiskBand } from './hooks'
import {
  SPage,
  SHeader,
  SHeaderTitle,
  SFilterBar,
  SSearchInput,
  SSelect,
  SDeptSection,
  SDeptLabel,
  SGrid,
  SCell,
  SCellLabel,
  SEmpty,
  SEmptyText,
  SEmptyHint,
  SClearBtn,
  SError,
} from './styled'

const EmployeesPage: React.FC = () => {
  const dispatch   = useAppDispatch()
  const navigate   = useNavigate()
  const { items, status, error } = useAppSelector(state => state.employees)
  const {
    search, setSearch,
    dept, setDept,
    riskBand, setRiskBand,
    departments,
    filtered,
    grouped,
    clearFilters,
    isFiltered,
  } = useEmployeesFilter(items)

  useEffect(() => {
    dispatch(fetchEmployees())
  }, [dispatch])

  return (
    <SPage>
      <SHeader>
        <SHeaderTitle>All Employees ({filtered.length})</SHeaderTitle>
      </SHeader>

      <SFilterBar>
        <SSearchInput
          placeholder="Search by ID…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <SSelect value={dept} onChange={e => setDept(e.target.value)}>
          {departments.map(d => (
            <option key={d} value={d}>
              {d === 'all' ? 'All Departments' : d}
            </option>
          ))}
        </SSelect>
        <SSelect
          value={riskBand}
          onChange={e => setRiskBand(e.target.value as RiskBand)}
        >
          <option value="all">All Risk Bands</option>
          <option value="low">Low (&lt;0.4)</option>
          <option value="medium">Medium (0.4 – 0.7)</option>
          <option value="high">High (&gt;0.7)</option>
        </SSelect>
      </SFilterBar>

      {status === 'loading' && <Spinner />}

      {status === 'failed' && (
        <SError>Failed to load employees{error ? `: ${error}` : '.'}</SError>
      )}

      {status === 'succeeded' && items.length === 0 && (
        <SEmpty>
          <SEmptyText>No employee data available</SEmptyText>
          <SEmptyHint>The backend hasn't returned any employee records yet.</SEmptyHint>
        </SEmpty>
      )}

      {status === 'succeeded' && items.length > 0 && grouped.length === 0 && (
        <SEmpty>
          <SEmptyText>No employees match your filters</SEmptyText>
          <SEmptyHint>Try broadening your search or adjusting the department and risk filters.</SEmptyHint>
          {isFiltered && <SClearBtn onClick={clearFilters}>Clear all filters</SClearBtn>}
        </SEmpty>
      )}

      {status === 'succeeded' &&
        grouped.map(([deptName, employees]) => (
          <SDeptSection key={deptName}>
            <SDeptLabel>
              {deptName} ({employees.length})
            </SDeptLabel>
            <SGrid>
              {employees.map(e => (
                <SCell
                  key={e.user}
                  $score={e.latest_score}
                  $hasAlert={e.status === 'open'}
                  data-tooltip={`${e.user}\n${e.department}\nScore: ${e.latest_score.toFixed(2)}\nAlerts: ${e.alert_count}`}
                  onClick={() => navigate(`/employee/${e.user}`)}
                >
                  <SCellLabel>{e.user}</SCellLabel>
                </SCell>
              ))}
            </SGrid>
          </SDeptSection>
        ))}
    </SPage>
  )
}

export default EmployeesPage

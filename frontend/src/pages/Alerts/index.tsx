import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import { fetchAlerts } from '../../store/slices/alertsSlice'
import Spinner from '../../components/Spinner'
import { useAlertsSort } from './hooks'
import {
  SPage,
  SHeader,
  SHeaderTitle,
  SEmpty,
  SEmptyIcon,
  SEmptyText,
  SError,
  STable,
  STh,
  SSortBtn,
  STr,
  SScoreBarWrap,
  SScoreBarFill,
  SStatusBadge,
  SChevron,
} from './styled'

const AlertsPage: React.FC = () => {
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const { items, status, error } = useAppSelector(state => state.alerts)
  const { sorted, sortKey, sortDir, toggle } = useAlertsSort(items)

  useEffect(() => {
    dispatch(fetchAlerts())
  }, [dispatch])

  const sortLabel = (key: 'score' | 'date') => {
    if (sortKey !== key) return '⇅'
    return sortDir === 'asc' ? '↑' : '↓'
  }

  return (
    <SPage>
      <SHeader>
        <SHeaderTitle>Open Alerts ({items.length})</SHeaderTitle>
      </SHeader>

      {status === 'loading' && <Spinner />}

      {status === 'failed' && (
        <SError>Failed to load alerts{error ? `: ${error}` : '.'}</SError>
      )}

      {status === 'succeeded' && sorted.length === 0 && (
        <SEmpty>
          <SEmptyIcon>✓</SEmptyIcon>
          <SEmptyText>No alerts found</SEmptyText>
          <SEmptyText style={{ fontSize: 13 }}>All clear — no open incidents at this time.</SEmptyText>
        </SEmpty>
      )}

      {status === 'succeeded' && sorted.length > 0 && (
        <STable>
          <thead>
            <tr>
              <STh>Employee</STh>
              <STh>Department</STh>
              <STh>
                <SSortBtn $active={sortKey === 'date'} onClick={() => toggle('date')}>
                  Detected {sortLabel('date')}
                </SSortBtn>
              </STh>
              <STh>
                <SSortBtn $active={sortKey === 'score'} onClick={() => toggle('score')}>
                  Risk Score {sortLabel('score')}
                </SSortBtn>
              </STh>
              <STh>Status</STh>
              <STh />
            </tr>
          </thead>
          <tbody>
            {sorted.map(alert => (
              <STr key={alert.id} onClick={() => navigate(`/employee/${alert.user}`)}>
                <td>{alert.user}</td>
                <td>{alert.department ?? '—'}</td>
                <td>{alert.date}</td>
                <td>
                  <SScoreBarWrap>
                    <SScoreBarFill $score={alert.score} />
                  </SScoreBarWrap>
                </td>
                <td>
                  <SStatusBadge $status={alert.status}>{alert.status}</SStatusBadge>
                </td>
                <td>
                  <SChevron>›</SChevron>
                </td>
              </STr>
            ))}
          </tbody>
        </STable>
      )}
    </SPage>
  )
}

export default AlertsPage

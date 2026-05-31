import { useState, useMemo } from 'react'
import type { Alert } from '../../../store/slices/alertsSlice'

type SortKey = 'score' | 'date'
type SortDir = 'asc' | 'desc'

export const useAlertsSort = (items: Alert[]) => {
  const [sortKey, setSortKey] = useState<SortKey>('score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const toggle = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sorted = useMemo(
    () =>
      [...items].sort((a, b) => {
        const delta =
          sortKey === 'score'
            ? a.score - b.score
            : a.date.localeCompare(b.date)
        return sortDir === 'asc' ? delta : -delta
      }),
    [items, sortKey, sortDir],
  )

  return { sorted, sortKey, sortDir, toggle }
}

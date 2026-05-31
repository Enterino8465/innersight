import { useState, useMemo } from 'react'
import type { Employee } from '../../../store/slices/employeesSlice'

export type RiskBand = 'all' | 'low' | 'medium' | 'high'

const matchesBand = (score: number, band: RiskBand) => {
  if (band === 'low')    return score < 0.4
  if (band === 'medium') return score >= 0.4 && score < 0.7
  if (band === 'high')   return score >= 0.7
  return true
}

export const useEmployeesFilter = (items: Employee[]) => {
  const [search,   setSearch]   = useState('')
  const [dept,     setDept]     = useState('all')
  const [riskBand, setRiskBand] = useState<RiskBand>('all')

  const departments = useMemo(() => {
    const set = new Set(items.map(e => e.department))
    return ['all', ...Array.from(set).sort()]
  }, [items])

  const filtered = useMemo(
    () =>
      items.filter(
        e =>
          (search === '' || e.user.toLowerCase().includes(search.toLowerCase())) &&
          (dept === 'all' || e.department === dept) &&
          matchesBand(e.latest_score, riskBand),
      ),
    [items, search, dept, riskBand],
  )

  const grouped = useMemo(() => {
    const map = new Map<string, Employee[]>()
    for (const e of filtered) {
      if (!map.has(e.department)) map.set(e.department, [])
      map.get(e.department)!.push(e)
    }
    return Array.from(map.entries()).sort(([a], [b]) => a.localeCompare(b))
  }, [filtered])

  const clearFilters = () => { setSearch(''); setDept('all'); setRiskBand('all') }
  const isFiltered = search !== '' || dept !== 'all' || riskBand !== 'all'

  return { search, setSearch, dept, setDept, riskBand, setRiskBand, departments, filtered, grouped, clearFilters, isFiltered }
}

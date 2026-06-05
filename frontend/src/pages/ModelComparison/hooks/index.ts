import { useEffect, useState } from 'react'

export interface LadderRow {
  model: string
  auprc: number | null
  auprc_std: number | null
  p_at_10: number | null
  p_at_20: number | null
  f1: number | null
  median_days: number | null
}

export interface ScenarioMetrics {
  auprc: number
  [key: string]: number
}

export interface ComparePayload {
  ladder: LadderRow[]
  per_scenario: Record<string, Record<string, ScenarioMetrics>>
  results_dir: string
}

type Status = 'idle' | 'loading' | 'succeeded' | 'failed'

export const useModelComparison = () => {
  const [data, setData] = useState<ComparePayload | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | undefined>()

  useEffect(() => {
    let active = true
    setStatus('loading')
    fetch('/api/models/compare')
      .then(res => {
        if (!res.ok) throw new Error(`Request failed (${res.status})`)
        return res.json()
      })
      .then((payload: ComparePayload) => {
        if (active) {
          setData(payload)
          setStatus('succeeded')
        }
      })
      .catch(e => {
        if (active) {
          setStatus('failed')
          setError(e instanceof Error ? e.message : 'Unknown error')
        }
      })
    return () => { active = false }
  }, [])

  return { data, status, error }
}

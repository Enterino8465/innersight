import { useCallback, useEffect, useState } from 'react'

export interface SimilarUser {
  user_id: string
  similarity: number
  score: number | null
  department: string
  version: string
}

type Status = 'idle' | 'loading' | 'succeeded' | 'failed'

// Within-version searches use the default demo version; cross-version omits it.
const DEFAULT_VERSION = 'r4.2'

export const useSuspectDiscovery = () => {
  const [users, setUsers] = useState<string[]>([])
  const [userId, setUserId] = useState('')
  const [crossVersion, setCrossVersion] = useState(false)
  const [results, setResults] = useState<SimilarUser[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | undefined>()

  // Populate the user dropdown from the employees endpoint (best-effort).
  useEffect(() => {
    let active = true
    fetch('/api/employees')
      .then(res => (res.ok ? res.json() : []))
      .then((emps: Array<{ user: string }>) => {
        if (active) setUsers(emps.map(e => e.user).sort())
      })
      .catch(() => { /* dropdown is optional — typing still works */ })
    return () => { active = false }
  }, [])

  const search = useCallback(async () => {
    if (!userId) return
    setStatus('loading')
    setError(undefined)
    const version = crossVersion ? 'null' : DEFAULT_VERSION
    try {
      const res = await fetch(
        `/api/users/${encodeURIComponent(userId)}/similar?k=10&version=${version}`,
      )
      if (res.status === 503) {
        setResults([])
        setStatus('failed')
        setError('Vector search (Qdrant) is unavailable. Start Qdrant and sync embeddings, then try again.')
        return
      }
      if (res.status === 404) {
        setResults([])   // valid response: the user simply has no stored embedding / neighbours
        setStatus('succeeded')
        return
      }
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const data = (await res.json()) as { similar_users?: SimilarUser[] }
      setResults(data.similar_users ?? [])
      setStatus('succeeded')
    } catch (e) {
      setResults([])
      setStatus('failed')
      setError(e instanceof Error ? e.message : 'Unknown error')
    }
  }, [userId, crossVersion])

  return { users, userId, setUserId, crossVersion, setCrossVersion, results, status, error, search }
}

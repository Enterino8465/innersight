import { useState, useEffect } from 'react'
import { fetchData } from '../api'

export const useDataPanel = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetchData()
      .then(setData)
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [])

  return { data, loading, error }
}

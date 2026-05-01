import { useState, useEffect } from 'react'
import { fetchConfig } from '../api'

export const useNetworkDiagram = () => {
  const [layerSizes, setLayerSizes] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetchConfig()
      .then(data => setLayerSizes(data.layer_sizes))
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [])

  return { layerSizes, loading, error }
}

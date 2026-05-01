import { useState, useEffect } from 'react'
import { fetchPrediction } from '../api'

export const usePredictionPanel = (trainingDone) => {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!trainingDone) return
    setLoading(true)
    setError(false)
    fetchPrediction(0)
      .then(setPrediction)
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [trainingDone])

  return { prediction, loading, error }
}

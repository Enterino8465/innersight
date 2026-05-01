import { API_BASE_URL } from '../../../../config'

export const fetchPrediction = (index = 0) =>
  fetch(`${API_BASE_URL}/api/predict?index=${index}`).then(res => {
    if (!res.ok) throw new Error('Prediction failed')
    return res.json()
  })

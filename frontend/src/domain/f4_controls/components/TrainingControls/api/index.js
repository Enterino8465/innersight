import { API_BASE_URL } from '../../../../config'

export const postTrain = (config) =>
  fetch(`${API_BASE_URL}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  }).then(res => {
    if (!res.ok) throw new Error('Failed to start training')
    return res.json()
  })

export const createEventSource = () =>
  new EventSource(`${API_BASE_URL}/api/events`)

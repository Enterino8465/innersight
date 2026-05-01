import { API_BASE_URL } from '../../../../config'

export const fetchData = () =>
  fetch(`${API_BASE_URL}/api/data`).then(res => {
    if (!res.ok) throw new Error('Failed to load data')
    return res.json()
  })

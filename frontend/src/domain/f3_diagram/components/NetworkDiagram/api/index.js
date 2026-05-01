import { API_BASE_URL } from '../../../../config'

export const fetchConfig = () =>
  fetch(`${API_BASE_URL}/api/config`).then(res => {
    if (!res.ok) throw new Error('Failed to load config')
    return res.json()
  })

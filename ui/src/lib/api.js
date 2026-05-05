const BASE = '/api'

async function get(path) {
  const res = await fetch(BASE + path)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  countries:   ()       => get('/countries'),
  risk:        (iso2)   => get(`/risk/${iso2}`),
  riskHistory: (iso2, days = 30) => get(`/risk/${iso2}/history?days=${days}`),
  trade:       (iso2)   => get(`/trade/${iso2}`),
  signals:     (iso2)   => get(`/signals/${iso2}`),
}

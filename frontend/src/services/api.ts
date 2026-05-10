const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  health: () => get<{ status: string; trained: boolean }>('/health'),

  // Data
  trainModels: () => post<{ message: string }>('/data/train'),
  trainingStatus: () => get<import('../types').TrainingStatus>('/data/status'),
  datasetInfo: () => get<Record<string, unknown>>('/data/info'),
  uploadDataset: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`${BASE}/data/upload`, { method: 'POST', body: form }).then(r => r.json())
  },

  // Analytics
  overview: () => get<import('../types').Overview>('/analytics/overview'),
  survival: () => get<import('../types').SurvivalData>('/analytics/survival'),
  riskDistribution: () => get<import('../types').RiskBucket[]>('/analytics/risk-distribution'),
  roc: (modelName?: string) =>
    get<import('../types').RocPoint[]>(`/analytics/roc${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ''}`),

  // Models
  modelComparison: () => get<import('../types').ModelMetrics[]>('/models/comparison'),
  featureImportance: (modelName?: string) =>
    get<import('../types').FeatureImportance[]>(
      `/models/feature-importance${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ''}`
    ),
  modelList: () => get<{ name: string; auc_roc: number; accuracy: number; is_best: boolean }[]>('/models/list'),

  // Predictions
  students: (params: {
    threshold?: number
    page?: number
    pageSize?: number
    riskFilter?: string
  }) => {
    const q = new URLSearchParams()
    if (params.threshold !== undefined) q.set('threshold', String(params.threshold))
    if (params.page) q.set('page', String(params.page))
    if (params.pageSize) q.set('page_size', String(params.pageSize))
    if (params.riskFilter && params.riskFilter !== 'all') q.set('risk_filter', params.riskFilter)
    return get<import('../types').StudentsResponse>(`/predictions/students?${q}`)
  },

  updateThreshold: (threshold: number) =>
    post<import('../types').ThresholdStats>('/predictions/threshold', { threshold }),

  currentThreshold: () => get<{ threshold: number; optimal_threshold: number }>('/predictions/threshold'),
}

import { useEffect, useState } from 'react'
import RiskDistributionChart from '../components/charts/RiskDistributionChart'
import { api } from '../services/api'
import type { RiskBucket, RocPoint } from '../types'

export default function Analytics() {
  const [distribution, setDistribution] = useState<RiskBucket[]>([])
  const [rocData, setRocData] = useState<RocPoint[]>([])
  const [threshold, setThreshold] = useState(0.5)
  const [modelList, setModelList] = useState<{ name: string; auc_roc: number; is_best: boolean }[]>([])
  const [selectedModel, setSelectedModel] = useState<string | undefined>()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
    Promise.all([
      api.riskDistribution(),
      api.modelList(),
      api.currentThreshold(),
    ])
      .then(async ([dist, ml, thInfo]) => {
        setDistribution(dist)
        setModelList(ml)
        setThreshold(thInfo.threshold)
        const best = ml.find(m => m.is_best)
        setSelectedModel(best?.name)
        if (best?.name) {
          const roc = await api.roc(best.name)
          setRocData(roc)
        }
      })
      .catch((e: any) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const loadRoc = async (name: string) => {
    setSelectedModel(name)
    try {
      const roc = await api.roc(name)
      setRocData(roc)
    } catch {}
  }

  const optimalPoint = rocData.length > 0
    ? rocData.reduce(
        (best, p) => p.youden_index > best.youden_index ? p : best,
        rocData[0]
      )
    : null

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-red-500 bg-red-50 border border-red-200 rounded-lg px-6 py-4">{error}</p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto p-6 space-y-6">
      {/* Risk distribution */}
      <RiskDistributionChart data={distribution} threshold={threshold} />

      {/* Youden index analysis */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-slate-800">Análisis del Índice de Youden</h3>
            <p className="text-slate-400 text-xs mt-0.5">
              Umbral óptimo: τ* = argmax[Sensibilidad(τ) + Especificidad(τ) – 1]
              {optimalPoint && (
                <span className="ml-2 text-blue-600 font-medium">
                  τ* = {optimalPoint.threshold.toFixed(2)}
                </span>
              )}
            </p>
          </div>
          {modelList.length > 0 && (
            <select
              value={selectedModel}
              onChange={(e) => loadRoc(e.target.value)}
              className="border border-slate-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {modelList.map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          )}
        </div>

        {rocData.length === 0 ? (
          <p className="text-slate-400 text-sm text-center py-8">Sin datos de curva ROC disponibles.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100">
                  <th className="text-left px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">τ</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Sensibilidad</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Especificidad</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Índice Youden</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {rocData
                  .filter(p => p.threshold >= 0.1 && p.threshold <= 0.9)
                  .filter((_, i) => i % Math.max(1, Math.floor(rocData.length / 15)) === 0)
                  .slice(0, 15)
                  .map((p) => {
                    const isOptimal = optimalPoint && p.threshold === optimalPoint.threshold
                    return (
                      <tr
                        key={p.threshold}
                        className={isOptimal ? 'bg-blue-50' : 'hover:bg-slate-50'}
                      >
                        <td className="px-5 py-2.5 font-mono text-slate-700">
                          {p.threshold.toFixed(3)}
                          {isOptimal && (
                            <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">τ*</span>
                          )}
                        </td>
                        <td className="px-5 py-2.5 text-right font-mono">{(p.sensitivity * 100).toFixed(1)}%</td>
                        <td className="px-5 py-2.5 text-right font-mono">{((1 - p.one_minus_specificity) * 100).toFixed(1)}%</td>
                        <td className="px-5 py-2.5 text-right font-mono font-bold text-blue-600">{p.youden_index.toFixed(4)}</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

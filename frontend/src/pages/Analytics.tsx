import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'
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

  useEffect(() => {
    Promise.all([
      api.riskDistribution(),
      api.modelList(),
      api.currentThreshold(),
    ]).then(([dist, ml, thInfo]) => {
      setDistribution(dist)
      setModelList(ml)
      setThreshold(thInfo.threshold)
      const best = ml.find(m => m.is_best)
      setSelectedModel(best?.name)
      return api.roc(best?.name)
    }).then(setRocData).finally(() => setLoading(false))
  }, [])

  const loadRoc = async (name: string) => {
    setSelectedModel(name)
    const roc = await api.roc(name)
    setRocData(roc)
  }

  const optimalPoint = rocData.reduce(
    (best, p) => p.youden_index > best.youden_index ? p : best,
    { threshold: 0.5, sensitivity: 0, one_minus_specificity: 0, youden_index: -1 }
  )

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto p-6 space-y-6">
      {/* Risk distribution */}
      <RiskDistributionChart data={distribution} threshold={threshold} />

      {/* ROC Curve */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-slate-800">Curva ROC</h3>
            <p className="text-slate-400 text-xs">
              Sensibilidad vs (1 – Especificidad). Punto óptimo por índice de Youden: τ* = {optimalPoint.threshold.toFixed(2)}
            </p>
          </div>
          <select
            value={selectedModel}
            onChange={(e) => loadRoc(e.target.value)}
            className="border border-slate-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {modelList.map(m => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart
            data={rocData.filter((_, i) => i % 5 === 0)}
            margin={{ top: 5, right: 20, left: 0, bottom: 30 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="one_minus_specificity"
              domain={[0, 1]}
              type="number"
              tickFormatter={(v) => v.toFixed(1)}
              label={{ value: '1 – Especificidad (FPR)', position: 'insideBottom', offset: -15, fontSize: 12 }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v) => v.toFixed(1)}
              label={{ value: 'Sensibilidad (TPR)', angle: -90, position: 'insideLeft', offset: 15, fontSize: 12 }}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(v: number, name: string) => [v.toFixed(3), name]}
              labelFormatter={(l) => `FPR: ${Number(l).toFixed(3)}`}
            />
            <ReferenceLine
              x={optimalPoint.one_minus_specificity}
              y={optimalPoint.sensitivity}
              stroke="#ef4444"
              strokeDasharray="4 4"
            />
            {/* Diagonal line (random classifier) */}
            <Line
              data={[{ one_minus_specificity: 0, sensitivity: 0 }, { one_minus_specificity: 1, sensitivity: 1 }]}
              dataKey="sensitivity"
              name="Aleatorio"
              stroke="#cbd5e1"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="sensitivity"
              name="ROC"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Youden index analysis */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800">Análisis del Índice de Youden</h3>
          <p className="text-slate-400 text-xs mt-0.5">
            Umbral óptimo: τ* = argmax[Sensibilidad(τ) + Especificidad(τ) – 1]
          </p>
        </div>
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
                .filter((_, i) => i % 20 === 0)
                .slice(0, 15)
                .map((p) => (
                  <tr
                    key={p.threshold}
                    className={p.threshold === optimalPoint.threshold ? 'bg-blue-50' : 'hover:bg-slate-50'}
                  >
                    <td className="px-5 py-2.5 font-mono text-slate-700">
                      {p.threshold.toFixed(3)}
                      {p.threshold === optimalPoint.threshold && (
                        <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">τ*</span>
                      )}
                    </td>
                    <td className="px-5 py-2.5 text-right font-mono">{(p.sensitivity * 100).toFixed(1)}%</td>
                    <td className="px-5 py-2.5 text-right font-mono">{((1 - p.one_minus_specificity) * 100).toFixed(1)}%</td>
                    <td className="px-5 py-2.5 text-right font-mono font-bold text-blue-600">{p.youden_index.toFixed(4)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

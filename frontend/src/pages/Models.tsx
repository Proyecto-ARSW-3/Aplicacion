import { useEffect, useState } from 'react'
import ModelComparisonChart from '../components/charts/ModelComparisonChart'
import VariableImportanceChart from '../components/charts/VariableImportanceChart'
import { api } from '../services/api'
import type { ModelMetrics, FeatureImportance } from '../types'
import clsx from 'clsx'

type Metric = keyof Pick<ModelMetrics, 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'auc_roc'>

const METRIC_OPTIONS: { value: Metric; label: string }[] = [
  { value: 'accuracy', label: 'Exactitud' },
  { value: 'precision', label: 'Precisión' },
  { value: 'recall', label: 'Sensibilidad' },
  { value: 'f1_score', label: 'F1-Score' },
  { value: 'auc_roc', label: 'AUC-ROC' },
]

export default function Models() {
  const [models, setModels] = useState<ModelMetrics[]>([])
  const [importance, setImportance] = useState<FeatureImportance[]>([])
  const [selectedModel, setSelectedModel] = useState<string | undefined>()
  const [activeMetric, setActiveMetric] = useState<Metric>('auc_roc')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.modelComparison(), api.featureImportance()])
      .then(([m, fi]) => {
        setModels(m)
        setImportance(fi)
        const best = [...m].sort((a, b) => b.auc_roc - a.auc_roc)[0]
        setSelectedModel(best?.name)
      })
      .finally(() => setLoading(false))
  }, [])

  const handleModelSelect = async (name: string) => {
    setSelectedModel(name)
    const fi = await api.featureImportance(name)
    setImportance(fi)
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto p-6 space-y-6">
      {/* Metric selector */}
      <div className="flex flex-wrap gap-2">
        {METRIC_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setActiveMetric(opt.value)}
            className={clsx(
              'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              activeMetric === opt.value
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Bar chart */}
      <ModelComparisonChart data={models} activeMetric={activeMetric} />

      {/* Metrics table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800">Tabla de Métricas Completa</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100">
                <th className="text-left px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Modelo</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Exactitud</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Precisión</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Sensibilidad</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">F1-Score</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">AUC-ROC</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Tiempo (ms)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map((m) => {
                const isBest = models.every(x => x.name === m.name || m.auc_roc >= x.auc_roc)
                return (
                  <tr
                    key={m.name}
                    onClick={() => handleModelSelect(m.name)}
                    className={clsx(
                      'cursor-pointer transition-colors',
                      selectedModel === m.name ? 'bg-blue-50' : 'hover:bg-slate-50'
                    )}
                  >
                    <td className="px-5 py-3 font-medium text-slate-800 flex items-center gap-2">
                      {m.name}
                      {isBest && (
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-normal">
                          Mejor
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{m.accuracy}%</td>
                    <td className="px-4 py-3 text-right font-mono">{m.precision}%</td>
                    <td className="px-4 py-3 text-right font-mono">{m.recall}%</td>
                    <td className="px-4 py-3 text-right font-mono">{m.f1_score}%</td>
                    <td className="px-4 py-3 text-right font-mono font-bold text-blue-600">{m.auc_roc}%</td>
                    <td className="px-4 py-3 text-right font-mono text-slate-400">{m.training_time_ms}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature importance */}
      <VariableImportanceChart data={importance} />
    </div>
  )
}

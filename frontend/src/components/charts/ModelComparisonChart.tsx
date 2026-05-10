import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import type { ModelMetrics } from '../../types'

interface Props {
  data: ModelMetrics[]
  activeMetric: keyof Pick<ModelMetrics, 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'auc_roc'>
}

const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Exactitud',
  precision: 'Precisión',
  recall: 'Sensibilidad',
  f1_score: 'F1-Score',
  auc_roc: 'AUC-ROC',
}

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-semibold text-slate-700 mb-2">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.fill }}>
          {p.name}: {p.value}%
        </p>
      ))}
    </div>
  )
}

export default function ModelComparisonChart({ data, activeMetric }: Props) {
  const metricLabel = METRIC_LABELS[activeMetric] ?? activeMetric

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <h3 className="font-semibold text-slate-800 mb-1">
        Comparativa de Modelos — {metricLabel}
      </h3>
      <p className="text-slate-400 text-xs mb-4">
        Desempeño (%) en el conjunto de prueba
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 45 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11 }}
            angle={-25}
            textAnchor="end"
            interval={0}
          />
          <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey={activeMetric} name={metricLabel} radius={[4, 4, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

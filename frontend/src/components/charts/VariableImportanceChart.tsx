import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import type { FeatureImportance } from '../../types'

interface Props {
  data: FeatureImportance[]
}

const GRADIENT = ['#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload as FeatureImportance
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-semibold text-slate-700">{d.feature}</p>
      <p className="text-blue-600">Importancia: {(d.importance * 100).toFixed(2)}%</p>
    </div>
  )
}

export default function VariableImportanceChart({ data }: Props) {
  const top = data.slice(0, 10)

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <h3 className="font-semibold text-slate-800 mb-1">Importancia de Variables</h3>
      <p className="text-slate-400 text-xs mb-4">
        Top 10 factores predictores de deserción (modelo principal)
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={top}
          layout="vertical"
          margin={{ top: 5, right: 20, left: 140, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
          <XAxis type="number" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
          <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} width={140} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="importance" name="Importancia" radius={[0, 4, 4, 0]}>
            {top.map((_, i) => (
              <Cell key={i} fill={GRADIENT[Math.min(i, GRADIENT.length - 1)]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

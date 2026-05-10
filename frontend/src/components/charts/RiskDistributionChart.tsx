import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts'
import type { RiskBucket } from '../../types'

interface Props {
  data: RiskBucket[]
  threshold: number
}

export default function RiskDistributionChart({ data, threshold }: Props) {
  const bucketToMid = (range: string) => {
    const [low] = range.split('–').map(Number)
    return low + 0.05
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <h3 className="font-semibold text-slate-800 mb-1">Distribución de Riesgo</h3>
      <p className="text-slate-400 text-xs mb-4">
        Número de estudiantes por rango de probabilidad de deserción
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="range" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" interval={0} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(v: number) => [v, 'Estudiantes']}
            labelFormatter={(l) => `Rango: ${l}`}
          />
          <Bar dataKey="count" name="Estudiantes" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => {
              const mid = bucketToMid(d.range)
              const color = mid >= threshold ? '#ef4444' : mid >= 0.4 ? '#f59e0b' : '#22c55e'
              return <Cell key={i} fill={color} />
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-3 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-500 inline-block" />Bajo</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-400 inline-block" />Medio</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-red-500 inline-block" />Alto (≥ τ={threshold.toFixed(2)})</span>
      </div>
    </div>
  )
}

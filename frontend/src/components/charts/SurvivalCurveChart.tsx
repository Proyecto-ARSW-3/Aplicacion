import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import type { SurvivalPoint } from '../../types'

interface Props {
  data: SurvivalPoint[]
  medianSurvival: number
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-semibold text-slate-700">Semestre {label}</p>
      <p className="text-blue-600">
        P(permanencia) = {(payload[0].value * 100).toFixed(1)}%
      </p>
    </div>
  )
}

export default function SurvivalCurveChart({ data, medianSurvival }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <h3 className="font-semibold text-slate-800 mb-1">
        Curva de Supervivencia de Kaplan-Meier
      </h3>
      <p className="text-slate-400 text-xs mb-4">
        Probabilidad de permanencia de la cohorte por semestre
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="semester"
            label={{ value: 'Semestre', position: 'insideBottom', offset: -2, fontSize: 12 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine
            y={0.5}
            stroke="#ef4444"
            strokeDasharray="4 4"
            label={{ value: 'Mediana', fill: '#ef4444', fontSize: 11 }}
          />
          {medianSurvival > 0 && (
            <ReferenceLine
              x={medianSurvival}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{ value: `t½=${medianSurvival}`, fill: '#f59e0b', fontSize: 11 }}
            />
          )}
          <Line
            type="stepAfter"
            dataKey="survival_probability"
            name="Cohorte Total"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={{ r: 4, fill: '#3b82f6' }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

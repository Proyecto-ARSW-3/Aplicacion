import { useEffect, useState, useCallback } from 'react'
import { Activity } from 'lucide-react'
import SurvivalCurveChart from '../components/charts/SurvivalCurveChart'
import { api } from '../services/api'
import type { SurvivalData } from '../types'
import clsx from 'clsx'

export default function SurvivalAnalysis() {
  const [data, setData] = useState<SurvivalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [trained, setTrained] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const health = await api.health()
      if (!health.trained) {
        setTrained(false)
        return
      }
      setTrained(true)
      const survival = await api.survival()
      setData(survival)
    } catch (e: any) {
      setError(e.message ?? 'Error al cargar datos de supervivencia')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-slate-500 text-sm">Cargando análisis de supervivencia…</p>
        </div>
      </div>
    )
  }

  if (!trained) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-sm">
          <Activity size={40} className="text-slate-300 mx-auto mb-3" />
          <p className="text-slate-600 font-semibold">Modelos no entrenados</p>
          <p className="text-slate-400 text-sm mt-1 mb-4">
            Entrena los modelos desde el Dashboard para ver el análisis de supervivencia.
          </p>
          <button
            onClick={load}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
          >
            Reintentar
          </button>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-sm">
          <p className="text-red-600 font-semibold mb-2">Error al cargar datos</p>
          <p className="text-red-500 text-sm bg-red-50 border border-red-200 rounded-lg px-4 py-2 mb-4">{error}</p>
          <button
            onClick={load}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
          >
            Reintentar
          </button>
        </div>
      </div>
    )
  }

  if (!data || data.kaplan_meier_points.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-slate-400">Sin datos de supervivencia disponibles.</p>
      </div>
    )
  }

  const lastPoint = data.kaplan_meier_points[data.kaplan_meier_points.length - 1]

  return (
    <div className="flex-1 overflow-auto p-6 space-y-6">
      {/* Survival curve */}
      <SurvivalCurveChart
        data={data.kaplan_meier_points}
        medianSurvival={data.median_survival}
      />

      {/* Summary stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
          <p className="text-slate-400 text-xs font-medium uppercase tracking-wide">Supervivencia Mediana</p>
          <p className="text-3xl font-bold text-slate-800 mt-1">
            {data.median_survival > 0 && isFinite(data.median_survival)
              ? `${data.median_survival} sem.`
              : 'N/A'}
          </p>
          <p className="text-slate-400 text-xs mt-1">Semestre donde P = 50%</p>
        </div>

        {[
          { label: 'P(Sem 1)', semester: 1, color: 'text-blue-600' },
          { label: 'P(Sem 4)', semester: 4, color: 'text-amber-600' },
          { label: `P(Sem ${lastPoint.semester})`, semester: lastPoint.semester, color: 'text-red-600' },
        ].map(({ label, semester, color }) => {
          const point = data.kaplan_meier_points.find(p => p.semester === semester)
            ?? (semester === lastPoint.semester ? lastPoint : null)
          return (
            <div key={label} className="bg-white rounded-xl border border-slate-200 p-4 text-center">
              <p className="text-slate-400 text-xs font-medium uppercase tracking-wide">{label}</p>
              <p className={clsx('text-3xl font-bold mt-1', color)}>
                {point ? `${(point.survival_probability * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
          )
        })}
      </div>

      {/* At-risk table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800">Tabla de Riesgo por Semestre</h3>
          <p className="text-slate-400 text-xs mt-0.5">
            Número de estudiantes en riesgo, eventos de deserción y censurados por semestre
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100">
                <th className="text-left px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Semestre</th>
                <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">En Riesgo</th>
                <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Deserciones</th>
                <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Censurados</th>
                <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">P(Permanencia)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.at_risk_by_semester.map((row) => {
                const sp = data.kaplan_meier_points.find(p => p.semester === row.semester)
                const prob = sp?.survival_probability ?? 0
                return (
                  <tr key={row.semester} className="hover:bg-slate-50">
                    <td className="px-5 py-3 font-medium text-slate-700">Semestre {row.semester}</td>
                    <td className="px-5 py-3 text-right font-mono">{row.at_risk.toLocaleString()}</td>
                    <td className="px-5 py-3 text-right font-mono text-red-600">{row.events}</td>
                    <td className="px-5 py-3 text-right font-mono text-slate-400">{row.censored}</td>
                    <td className="px-5 py-3 text-right">
                      <span className={clsx(
                        'font-bold font-mono',
                        prob >= 0.7 ? 'text-green-600' : prob >= 0.5 ? 'text-amber-600' : 'text-red-600'
                      )}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cox model */}
      {data.cox_summary.length > 0 && !('error' in (data.cox_summary[0] as object)) && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-100">
            <h3 className="font-semibold text-slate-800">Modelo de Cox — Hazard Ratios</h3>
            <p className="text-slate-400 text-xs mt-0.5">
              HR &gt; 1 indica mayor riesgo de deserción; HR &lt; 1 indica efecto protector
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100">
                  <th className="text-left px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Covariable</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">HR</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">IC 95% Inf</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">IC 95% Sup</th>
                  <th className="text-right px-5 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">p-valor</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.cox_summary.map((row) => (
                  <tr key={row.covariate} className="hover:bg-slate-50">
                    <td className="px-5 py-3 font-medium text-slate-700">{row.covariate}</td>
                    <td className={clsx(
                      'px-5 py-3 text-right font-mono font-bold',
                      row.exp_coef > 1 ? 'text-red-600' : 'text-green-600'
                    )}>
                      {row.exp_coef.toFixed(3)}
                    </td>
                    <td className="px-5 py-3 text-right font-mono text-slate-500">{row.hr_lower.toFixed(3)}</td>
                    <td className="px-5 py-3 text-right font-mono text-slate-500">{row.hr_upper.toFixed(3)}</td>
                    <td className={clsx(
                      'px-5 py-3 text-right font-mono',
                      row.p_value < 0.05 ? 'font-bold text-slate-800' : 'text-slate-400'
                    )}>
                      {row.p_value < 0.001 ? '<0.001' : row.p_value.toFixed(3)}
                      {row.p_value < 0.05 && ' *'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

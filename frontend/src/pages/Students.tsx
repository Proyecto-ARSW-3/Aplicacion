import { useEffect, useState, useCallback } from 'react'
import { Search, ChevronLeft, ChevronRight, Filter } from 'lucide-react'
import Badge from '../components/ui/Badge'
import { api } from '../services/api'
import type { RiskStudent, StudentsResponse } from '../types'
import clsx from 'clsx'

const RISK_FILTERS = [
  { value: 'all', label: 'Todos' },
  { value: 'alto', label: 'Alto Riesgo' },
  { value: 'medio', label: 'Riesgo Medio' },
  { value: 'bajo', label: 'Bajo Riesgo' },
]

export default function Students() {
  const [data, setData] = useState<StudentsResponse | null>(null)
  const [page, setPage] = useState(1)
  const [riskFilter, setRiskFilter] = useState('all')
  const [threshold, setThreshold] = useState<number | undefined>(undefined)
  const [thresholdInput, setThresholdInput] = useState('')
  const [loading, setLoading] = useState(true)

  const pageSize = 20

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.students({
        page,
        pageSize,
        riskFilter,
        threshold,
      })
      setData(res)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [page, riskFilter, threshold])

  useEffect(() => { load() }, [load])

  const totalPages = data ? Math.ceil(data.total / pageSize) : 1

  const applyThreshold = () => {
    const v = parseFloat(thresholdInput)
    if (!isNaN(v) && v >= 0 && v <= 1) {
      setThreshold(v)
      setPage(1)
    }
  }

  return (
    <div className="flex-1 overflow-auto p-6 space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex gap-1.5">
          {RISK_FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => { setRiskFilter(f.value); setPage(1) }}
              className={clsx(
                'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                riskFilter === f.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 ml-auto">
          <Filter size={16} className="text-slate-400" />
          <input
            type="number"
            min={0} max={1} step={0.01}
            value={thresholdInput}
            onChange={(e) => setThresholdInput(e.target.value)}
            placeholder="τ (0–1)"
            className="w-28 border border-slate-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={applyThreshold}
            className="px-3 py-1.5 bg-slate-800 text-white rounded-lg text-sm hover:bg-slate-700"
          >
            Aplicar
          </button>
          {threshold !== undefined && (
            <button
              onClick={() => { setThreshold(undefined); setThresholdInput('') }}
              className="text-xs text-slate-400 hover:text-slate-600"
            >
              Limpiar
            </button>
          )}
        </div>
      </div>

      {/* Summary */}
      {data && (
        <p className="text-slate-500 text-sm">
          {data.total.toLocaleString()} estudiantes encontrados
          {threshold !== undefined && ` (τ = ${threshold.toFixed(2)})`}
        </p>
      )}

      {/* Table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                <th className="text-left px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">ID</th>
                <th className="text-left px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Edad</th>
                <th className="text-left px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Género</th>
                <th className="text-center px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Becario</th>
                <th className="text-center px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Deudor</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Nota Sem 1</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Nota Sem 2</th>
                <th className="text-right px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">P(Deserción)</th>
                <th className="text-center px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Riesgo</th>
                <th className="text-left px-4 py-3 text-slate-500 font-medium text-xs uppercase tracking-wide">Real</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={10} className="text-center py-10 text-slate-400">
                    <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto" />
                  </td>
                </tr>
              ) : data?.students.length === 0 ? (
                <tr>
                  <td colSpan={10} className="text-center py-10 text-slate-400">Sin resultados</td>
                </tr>
              ) : data?.students.map((s: RiskStudent) => (
                <tr key={s.id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3 text-slate-400 font-mono text-xs">{s.id}</td>
                  <td className="px-4 py-3 text-slate-700">{s.age_at_enrollment}</td>
                  <td className="px-4 py-3 text-slate-700">{s.gender}</td>
                  <td className="px-4 py-3 text-center">
                    <span className={s.scholarship_holder ? 'text-green-600' : 'text-slate-300'}>
                      {s.scholarship_holder ? '✓' : '–'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className={s.debtor ? 'text-red-600' : 'text-slate-300'}>
                      {s.debtor ? '✓' : '–'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono">
                    {s.sem1_grade.toFixed(1)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">
                    {s.sem2_grade.toFixed(1)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={clsx(
                      'font-bold font-mono',
                      s.dropout_probability >= 0.7 ? 'text-red-600' :
                      s.dropout_probability >= 0.4 ? 'text-amber-600' : 'text-green-600'
                    )}>
                      {(s.dropout_probability * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <Badge level={s.risk_level} />
                  </td>
                  <td className="px-4 py-3">
                    <span className={clsx(
                      'text-xs font-medium',
                      s.actual_class === 'Desertor' ? 'text-red-600' :
                      s.actual_class === 'Graduado' ? 'text-green-600' : 'text-blue-600'
                    )}>
                      {s.actual_class}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data && totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-slate-100">
            <p className="text-slate-400 text-xs">
              Página {page} de {totalPages}
            </p>
            <div className="flex gap-1">
              <button
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
                className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30"
              >
                <ChevronLeft size={16} />
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
                className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ResponsiveContainer,
} from 'recharts'
import type { RiskBucket } from '../../types'
import clsx from 'clsx'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Props {
  data: RiskBucket[]
  threshold: number
}

type Level = 'bajo' | 'medio' | 'alto' | 'sin_datos'

interface RiskIndicator {
  level: Level
  label: string
  /** Percentage of students above the threshold (0–1) */
  highRiskPct: number
  /** Percentage of students in medium risk zone (0–1) */
  medRiskPct: number
  /** Weighted average dropout probability across all students (0–1) */
  avgRisk: number
  totalStudents: number
  highRiskCount: number
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Parse the lower bound of a bucket range.
 * Handles both ASCII hyphen "0.2-0.3" and en-dash "0.2–0.3".
 */
function bucketLow(range: string): number {
  // Normalise: replace en-dash (U+2013) and em-dash (U+2014) with plain hyphen
  const normalised = range.replace(/[–—]/g, '-')
  const low = parseFloat(normalised.split('-')[0])
  return isNaN(low) ? 0 : low
}

function bucketMid(range: string): number {
  return bucketLow(range) + 0.05
}

/**
 * Classify the institution's overall risk level based on the
 * percentage of students whose predicted probability is ≥ threshold.
 *
 * Thresholds (calibrated for typical HEI dropout rates of 15–45 %):
 *   < 20 %   → Bajo   (manageable — fewer than 1 in 5 flagged)
 *   20–40 %  → Medio  (warning    — significant share at risk)
 *   ≥ 40 %   → Alto   (critical   — more than 4 in 10 flagged)
 */
function computeIndicator(data: RiskBucket[], threshold: number): RiskIndicator {
  const total = data.reduce((s, b) => s + b.count, 0)

  if (total === 0 || data.length === 0) {
    return { level: 'sin_datos', label: 'Sin datos', highRiskPct: 0, medRiskPct: 0, avgRisk: 0, totalStudents: 0, highRiskCount: 0 }
  }

  let highCount = 0
  let medCount = 0
  let weightedSum = 0

  for (const bucket of data) {
    const mid = bucketMid(bucket.range)
    weightedSum += mid * bucket.count
    if (mid >= threshold) highCount += bucket.count
    else if (mid >= 0.3) medCount += bucket.count
  }

  const highPct = highCount / total
  const medPct = medCount / total
  const avgRisk = weightedSum / total

  let level: Level
  if (highPct >= 0.40) level = 'alto'
  else if (highPct >= 0.20) level = 'medio'
  else level = 'bajo'

  const labels: Record<Level, string> = {
    alto: 'Riesgo Alto',
    medio: 'Riesgo Medio',
    bajo: 'Riesgo Bajo',
    sin_datos: 'Sin datos',
  }

  return { level, label: labels[level], highRiskPct: highPct, medRiskPct: medPct, avgRisk, totalStudents: total, highRiskCount: highCount }
}

/** Colour of each bar given its midpoint and the current threshold */
function barColor(mid: number, threshold: number): string {
  if (mid >= threshold) return '#ef4444'   // red  — at risk
  if (mid >= 0.3)       return '#f59e0b'   // amber — borderline
  return '#22c55e'                          // green — safe
}

// ─── Semáforo sub-component ───────────────────────────────────────────────────

interface SemaforoProps {
  indicator: RiskIndicator
}

const LEVEL_META = {
  bajo:      { dot: '#22c55e', glow: 'rgba(34,197,94,0.45)',  text: 'text-green-700',  bg: 'bg-green-50',  border: 'border-green-200'  },
  medio:     { dot: '#f59e0b', glow: 'rgba(245,158,11,0.45)', text: 'text-amber-700',  bg: 'bg-amber-50',  border: 'border-amber-200'  },
  alto:      { dot: '#ef4444', glow: 'rgba(239,68,68,0.45)',  text: 'text-red-700',    bg: 'bg-red-50',    border: 'border-red-200'    },
  sin_datos: { dot: '#94a3b8', glow: 'transparent',           text: 'text-slate-500',  bg: 'bg-slate-50',  border: 'border-slate-200'  },
}

function Semaforo({ indicator }: SemaforoProps) {
  const meta = LEVEL_META[indicator.level]
  const levels: Level[] = ['alto', 'medio', 'bajo']  // top-to-bottom (traffic light order)

  return (
    <div className={clsx(
      'flex items-center gap-3 px-3 py-2 rounded-xl border transition-all duration-500',
      meta.bg, meta.border,
    )}>
      {/* Three traffic-light circles */}
      <div className="flex flex-col items-center gap-[5px]">
        {levels.map((lvl) => {
          const isActive = indicator.level === lvl || indicator.level === 'sin_datos' && lvl === 'bajo'
          const dotMeta = LEVEL_META[lvl]
          return (
            <div
              key={lvl}
              style={{
                width: isActive ? 14 : 10,
                height: isActive ? 14 : 10,
                borderRadius: '50%',
                backgroundColor: isActive ? dotMeta.dot : `${dotMeta.dot}55`,
                boxShadow: isActive ? `0 0 8px 2px ${dotMeta.glow}` : 'none',
                transition: 'all 0.5s ease',
              }}
            />
          )
        })}
      </div>

      {/* Label + percentage */}
      <div className="min-w-0">
        <p className={clsx('text-xs font-bold leading-tight transition-colors duration-500', meta.text)}>
          {indicator.label}
        </p>
        {indicator.totalStudents > 0 && (
          <p className="text-slate-400 text-[10px] leading-tight mt-0.5">
            {(indicator.highRiskPct * 100).toFixed(1)}% sobre τ
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Custom tooltip ───────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label, threshold }: any) {
  if (!active || !payload?.length) return null
  const mid = bucketMid(label)
  const count: number = payload[0].value
  const level = mid >= threshold ? 'Alto riesgo' : mid >= 0.3 ? 'Riesgo medio' : 'Bajo riesgo'
  const textCls = mid >= threshold ? 'text-red-600' : mid >= 0.3 ? 'text-amber-600' : 'text-green-600'
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-semibold text-slate-700 mb-1">Rango: {label}</p>
      <p className="text-slate-600">{count.toLocaleString()} estudiantes</p>
      <p className={clsx('text-xs font-medium mt-0.5', textCls)}>{level}</p>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function RiskDistributionChart({ data, threshold }: Props) {
  const indicator = useMemo(() => computeIndicator(data, threshold), [data, threshold])

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      {/* Header row */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-semibold text-slate-800">Distribución de Riesgo</h3>
          <p className="text-slate-400 text-xs mt-0.5">
            Estudiantes por rango de probabilidad de deserción
          </p>
        </div>
        <Semaforo indicator={indicator} />
      </div>

      {/* Stats row */}
      {indicator.totalStudents > 0 && (
        <div className="grid grid-cols-3 gap-2 mb-4">
          <div className="bg-green-50 rounded-lg px-3 py-2 text-center">
            <p className="text-green-700 font-bold text-sm">
              {((1 - indicator.highRiskPct - indicator.medRiskPct) * 100).toFixed(1)}%
            </p>
            <p className="text-green-600 text-[10px] font-medium uppercase tracking-wide mt-0.5">Bajo riesgo</p>
          </div>
          <div className="bg-amber-50 rounded-lg px-3 py-2 text-center">
            <p className="text-amber-700 font-bold text-sm">
              {(indicator.medRiskPct * 100).toFixed(1)}%
            </p>
            <p className="text-amber-600 text-[10px] font-medium uppercase tracking-wide mt-0.5">Riesgo medio</p>
          </div>
          <div className="bg-red-50 rounded-lg px-3 py-2 text-center">
            <p className="text-red-700 font-bold text-sm">
              {(indicator.highRiskPct * 100).toFixed(1)}%
            </p>
            <p className="text-red-600 text-[10px] font-medium uppercase tracking-wide mt-0.5">Alto riesgo (≥τ)</p>
          </div>
        </div>
      )}

      {/* Bar chart */}
      {data.length === 0 ? (
        <div className="h-52 flex items-center justify-center text-slate-300 text-sm">
          Sin datos de distribución
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="range"
              tick={{ fontSize: 9 }}
              angle={-30}
              textAnchor="end"
              interval={0}
              height={40}
            />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip content={<CustomTooltip threshold={threshold} />} />
            <Bar dataKey="count" name="Estudiantes" radius={[3, 3, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={barColor(bucketMid(d.range), threshold)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      {/* Legend */}
      <div className="flex gap-4 mt-2 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-green-500 inline-block flex-shrink-0" />
          Bajo (&lt;0.30)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-amber-400 inline-block flex-shrink-0" />
          Medio (0.30–τ)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-500 inline-block flex-shrink-0" />
          Alto (≥ τ={threshold.toFixed(2)})
        </span>
      </div>
    </div>
  )
}

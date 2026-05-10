import { useState } from 'react'

interface Props {
  value: number
  optimal: number
  onChange: (val: number) => void
  stats?: {
    at_risk_count: number
    sensitivity: number
    specificity: number
    precision: number
  }
}

export default function ThresholdSlider({ value, optimal, onChange, stats }: Props) {
  const [local, setLocal] = useState(value)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value)
    setLocal(val)
  }

  const handleRelease = () => onChange(local)

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-slate-800">Umbral Adaptativo (τ)</h3>
        <span className="text-blue-600 font-bold text-lg">{local.toFixed(2)}</span>
      </div>

      <input
        type="range"
        min={0.1}
        max={0.9}
        step={0.01}
        value={local}
        onChange={handleChange}
        onMouseUp={handleRelease}
        onTouchEnd={handleRelease}
        className="w-full h-2 bg-slate-200 rounded-full appearance-none cursor-pointer accent-blue-600"
      />

      <div className="flex justify-between text-xs text-slate-400 mt-1">
        <span>0.10 (máx. sensibilidad)</span>
        <span>0.90 (máx. precisión)</span>
      </div>

      <div className="mt-3 text-xs text-slate-500">
        Umbral óptimo (Youden):{' '}
        <button
          onClick={() => { setLocal(optimal); onChange(optimal) }}
          className="text-blue-600 font-semibold hover:underline"
        >
          τ* = {optimal.toFixed(2)}
        </button>
      </div>

      {stats && (
        <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-slate-100">
          <div className="text-center">
            <p className="text-slate-400 text-xs">En riesgo</p>
            <p className="text-slate-800 font-bold">{stats.at_risk_count.toLocaleString()}</p>
          </div>
          <div className="text-center">
            <p className="text-slate-400 text-xs">Sensibilidad</p>
            <p className="text-slate-800 font-bold">{(stats.sensitivity * 100).toFixed(1)}%</p>
          </div>
          <div className="text-center">
            <p className="text-slate-400 text-xs">Especificidad</p>
            <p className="text-slate-800 font-bold">{(stats.specificity * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}
    </div>
  )
}

import { useEffect, useState, useCallback } from 'react'
import { Users, AlertTriangle, GraduationCap, BookOpen, Trophy } from 'lucide-react'
import MetricCard from '../components/ui/MetricCard'
import ThresholdSlider from '../components/ui/ThresholdSlider'
import RiskDistributionChart from '../components/charts/RiskDistributionChart'
import SurvivalCurveChart from '../components/charts/SurvivalCurveChart'
import { api } from '../services/api'
import type { Overview, SurvivalData, RiskBucket, ThresholdStats } from '../types'

export default function Dashboard() {
  const [overview, setOverview] = useState<Overview | null>(null)
  const [survival, setSurvival] = useState<SurvivalData | null>(null)
  const [distribution, setDistribution] = useState<RiskBucket[]>([])
  const [threshold, setThreshold] = useState(0.5)
  const [optimalThreshold, setOptimalThreshold] = useState(0.5)
  const [thresholdStats, setThresholdStats] = useState<ThresholdStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [trained, setTrained] = useState(false)

  const loadAll = useCallback(async () => {
    try {
      const health = await api.health()
      if (!health.trained) {
        setTrained(false)
        setLoading(false)
        return
      }
      setTrained(true)

      const [ov, sv, dist, threshInfo] = await Promise.all([
        api.overview(),
        api.survival(),
        api.riskDistribution(),
        api.currentThreshold(),
      ])
      setOverview(ov)
      setSurvival(sv)
      setDistribution(dist)
      setThreshold(threshInfo.threshold)
      setOptimalThreshold(threshInfo.optimal_threshold)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  const handleThresholdChange = async (val: number) => {
    setThreshold(val)
    try {
      const stats = await api.updateThreshold(val)
      setThresholdStats(stats)
    } catch {}
  }

  const handleTrain = async () => {
    setLoading(true)
    try {
      await api.trainModels()
      // Poll status
      const poll = setInterval(async () => {
        const status = await api.trainingStatus()
        if (status.status === 'complete' || status.status === 'error') {
          clearInterval(poll)
          loadAll()
        }
      }, 2000)
    } catch (e: any) {
      setError(e.message)
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-500">Cargando datos y modelos...</p>
        </div>
      </div>
    )
  }

  if (!trained) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md">
          <Trophy size={48} className="text-blue-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-slate-800 mb-2">Modelos no entrenados</h2>
          <p className="text-slate-500 text-sm mb-6">
            El dataset no fue encontrado o los modelos aún no se han entrenado.
            Asegúrate de colocar el dataset en <code className="bg-slate-100 px-1 rounded">analytics-service/data/dataset.csv</code> y haz clic en Entrenar.
          </p>
          {error && <p className="text-red-500 text-sm mb-4">{error}</p>}
          <button
            onClick={handleTrain}
            className="px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            Entrenar Modelos
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto p-6 space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Estudiantes"
          value={overview?.total_students.toLocaleString() ?? '—'}
          icon={<Users size={20} />}
        />
        <MetricCard
          title="En Riesgo de Deserción"
          value={overview?.at_risk_count.toLocaleString() ?? '—'}
          subtitle={`${((overview?.at_risk_rate ?? 0) * 100).toFixed(1)}% del total`}
          icon={<AlertTriangle size={20} />}
          variant="danger"
        />
        <MetricCard
          title="Desertores Confirmados"
          value={overview?.dropout_count.toLocaleString() ?? '—'}
          subtitle={`${((overview?.dropout_rate ?? 0) * 100).toFixed(1)}% de abandono`}
          icon={<BookOpen size={20} />}
          variant="warning"
        />
        <MetricCard
          title="Graduados"
          value={overview?.graduate_count.toLocaleString() ?? '—'}
          icon={<GraduationCap size={20} />}
          variant="success"
        />
      </div>

      {/* Best model banner */}
      {overview && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl px-5 py-3 flex items-center justify-between">
          <div>
            <p className="text-blue-700 font-semibold text-sm">
              Mejor modelo: <span className="font-bold">{overview.best_model}</span>
            </p>
            <p className="text-blue-500 text-xs">
              Exactitud: {(overview.best_model_accuracy * 100).toFixed(1)}% | Umbral actual: τ = {threshold.toFixed(2)}
            </p>
          </div>
          <span className="text-2xl font-black text-blue-600">
            {(overview.best_model_accuracy * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Charts row */}
      <div className="grid lg:grid-cols-2 gap-6">
        {survival && (
          <SurvivalCurveChart
            data={survival.kaplan_meier_points}
            medianSurvival={survival.median_survival}
          />
        )}
        <RiskDistributionChart data={distribution} threshold={threshold} />
      </div>

      {/* Threshold control */}
      <ThresholdSlider
        value={threshold}
        optimal={optimalThreshold}
        onChange={handleThresholdChange}
        stats={thresholdStats ?? undefined}
      />
    </div>
  )
}

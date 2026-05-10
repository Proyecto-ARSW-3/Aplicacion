import { useEffect, useState, useCallback } from 'react'
import { Users, AlertTriangle, GraduationCap, BookOpen, Trophy } from 'lucide-react'
import MetricCard from '../components/ui/MetricCard'
import ThresholdSlider from '../components/ui/ThresholdSlider'
import DatasetUpload from '../components/ui/DatasetUpload'
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
  const [trained, setTrained] = useState(false)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [training, setTraining] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  const loadAll = useCallback(async () => {
    setLoading(true)
    setTrainError(null)

    try {
      const health = await api.health()
      if (!health.trained) {
        setTrained(false)
        setLoading(false)
        return
      }
      setTrained(true)

      // Fetch each data source independently so one failure does not block others
      const [ovResult, svResult, distResult, thResult] = await Promise.allSettled([
        api.overview(),
        api.survival(),
        api.riskDistribution(),
        api.currentThreshold(),
      ])

      if (ovResult.status === 'fulfilled') setOverview(ovResult.value)
      if (svResult.status === 'fulfilled') setSurvival(svResult.value)
      if (distResult.status === 'fulfilled') setDistribution(distResult.value)
      if (thResult.status === 'fulfilled') {
        setThreshold(thResult.value.threshold)
        setOptimalThreshold(thResult.value.optimal_threshold)
      }
    } catch (e: any) {
      setTrainError(e.message)
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
    setTraining(true)
    setTrainError(null)
    try {
      await api.trainModels()
      const poll = setInterval(async () => {
        try {
          const status = await api.trainingStatus()
          if (status.status === 'complete') {
            clearInterval(poll)
            setTraining(false)
            loadAll()
          } else if (status.status === 'error') {
            clearInterval(poll)
            setTraining(false)
            setTrainError(status.message)
          }
        } catch {
          clearInterval(poll)
          setTraining(false)
        }
      }, 2000)
    } catch (e: any) {
      setTrainError(e.message)
      setTraining(false)
    }
  }

  const handleUploadSuccess = () => {
    setUploadSuccess(true)
  }

  /* ── Loading ─────────────────────────────────────────────────────────────── */
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-500 text-sm">Cargando datos y modelos…</p>
        </div>
      </div>
    )
  }

  /* ── Not trained ─────────────────────────────────────────────────────────── */
  if (!trained) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-full max-w-md space-y-4 px-4">
          <div className="text-center">
            <Trophy size={48} className="text-blue-400 mx-auto mb-3" />
            <h2 className="text-xl font-bold text-slate-800 mb-1">Modelos no entrenados</h2>
            <p className="text-slate-500 text-sm">
              Sube un dataset y entrena los modelos para activar el dashboard.
            </p>
          </div>

          {/* Upload */}
          <DatasetUpload onUploadSuccess={handleUploadSuccess} />

          {uploadSuccess && (
            <p className="text-center text-blue-600 text-xs font-medium">
              Dataset listo. Haz clic en Entrenar para procesar.
            </p>
          )}

          {/* Train button */}
          <button
            onClick={handleTrain}
            disabled={training}
            className="w-full flex items-center justify-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {training && (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            )}
            {training ? 'Entrenando modelos…' : 'Entrenar Modelos'}
          </button>

          {trainError && (
            <p className="text-red-500 text-sm text-center bg-red-50 border border-red-200 rounded-lg px-4 py-2">
              {trainError}
            </p>
          )}

          <p className="text-slate-400 text-xs text-center">
            También puedes colocar el dataset en{' '}
            <code className="bg-slate-100 px-1 rounded">analytics-service/data/dataset.csv</code>{' '}
            y reiniciar el backend.
          </p>
        </div>
      </div>
    )
  }

  /* ── Main dashboard ──────────────────────────────────────────────────────── */
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
          subtitle={overview ? `${(overview.at_risk_rate * 100).toFixed(1)}% del total` : undefined}
          icon={<AlertTriangle size={20} />}
          variant="danger"
        />
        <MetricCard
          title="Desertores Confirmados"
          value={overview?.dropout_count.toLocaleString() ?? '—'}
          subtitle={overview ? `${(overview.dropout_rate * 100).toFixed(1)}% de abandono` : undefined}
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
        {survival && survival.kaplan_meier_points.length > 0 ? (
          <SurvivalCurveChart
            data={survival.kaplan_meier_points}
            medianSurvival={survival.median_survival}
          />
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-center justify-center min-h-[280px]">
            <p className="text-slate-400 text-sm">Sin datos de supervivencia</p>
          </div>
        )}

        {distribution.length > 0 ? (
          <RiskDistributionChart data={distribution} threshold={threshold} />
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-center justify-center min-h-[280px]">
            <p className="text-slate-400 text-sm">Sin datos de distribución de riesgo</p>
          </div>
        )}
      </div>

      {/* Threshold control */}
      <ThresholdSlider
        value={threshold}
        optimal={optimalThreshold}
        onChange={handleThresholdChange}
        stats={thresholdStats ?? undefined}
      />

      {/* Upload section at the bottom of the trained dashboard */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <p className="text-slate-700 font-semibold text-sm mb-1">Actualizar Dataset</p>
        <p className="text-slate-400 text-xs mb-3">
          Sube un nuevo CSV y vuelve a entrenar los modelos con datos actualizados.
        </p>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex-1 min-w-[220px]">
            <DatasetUpload onUploadSuccess={() => {}} />
          </div>
          <button
            onClick={handleTrain}
            disabled={training}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-white rounded-lg text-sm hover:bg-slate-700 transition-colors disabled:opacity-60"
          >
            {training && (
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            )}
            {training ? 'Entrenando…' : 'Reentrenar'}
          </button>
        </div>
      </div>
    </div>
  )
}

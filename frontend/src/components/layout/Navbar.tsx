import { useLocation } from 'react-router-dom'
import { Bell } from 'lucide-react'

const TITLES: Record<string, string> = {
  '/': 'Dashboard General',
  '/students': 'Estudiantes en Riesgo',
  '/models': 'Comparativa de Modelos ML',
  '/survival': 'Análisis de Supervivencia',
  '/analytics': 'Analíticas Avanzadas',
}

export default function Navbar() {
  const { pathname } = useLocation()
  const title = TITLES[pathname] ?? 'AREP'

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 flex-shrink-0">
      <h1 className="text-slate-800 font-semibold text-base">{title}</h1>
      <div className="flex items-center gap-3">
        <button className="text-slate-400 hover:text-slate-600 relative">
          <Bell size={20} />
        </button>
        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-semibold text-sm">
          A
        </div>
      </div>
    </header>
  )
}

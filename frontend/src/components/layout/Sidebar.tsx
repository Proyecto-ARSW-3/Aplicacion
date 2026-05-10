import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Users, BarChart3, Activity, Brain } from 'lucide-react'
import clsx from 'clsx'

const links = [
  { to: '/', label: 'Dashboard', Icon: LayoutDashboard },
  { to: '/students', label: 'Estudiantes', Icon: Users },
  { to: '/models', label: 'Modelos ML', Icon: Brain },
  { to: '/survival', label: 'Supervivencia', Icon: Activity },
  { to: '/analytics', label: 'Analíticas', Icon: BarChart3 },
]

export default function Sidebar() {
  return (
    <aside className="w-64 bg-slate-900 text-white flex flex-col min-h-screen">
      <div className="px-6 py-5 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold text-sm">
            A
          </div>
          <div>
            <p className="font-semibold text-sm leading-tight">AREP</p>
            <p className="text-slate-400 text-xs">Predicción de Deserción</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {links.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              )
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-6 py-4 border-t border-slate-700">
        <p className="text-slate-500 text-xs">ECI — Ing. Sistemas</p>
        <p className="text-slate-600 text-xs">Bogotá, Colombia</p>
      </div>
    </aside>
  )
}

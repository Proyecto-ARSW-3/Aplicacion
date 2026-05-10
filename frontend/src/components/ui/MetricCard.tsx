import { ReactNode } from 'react'
import clsx from 'clsx'

interface Props {
  title: string
  value: string | number
  subtitle?: string
  icon?: ReactNode
  variant?: 'default' | 'danger' | 'success' | 'warning'
  className?: string
}

const variants = {
  default: 'border-slate-200',
  danger: 'border-red-200 bg-red-50',
  success: 'border-green-200 bg-green-50',
  warning: 'border-amber-200 bg-amber-50',
}

export default function MetricCard({ title, value, subtitle, icon, variant = 'default', className }: Props) {
  return (
    <div className={clsx('bg-white rounded-xl border p-5 flex items-start gap-4', variants[variant], className)}>
      {icon && (
        <div className={clsx('p-2.5 rounded-lg', {
          'bg-blue-100 text-blue-600': variant === 'default',
          'bg-red-100 text-red-600': variant === 'danger',
          'bg-green-100 text-green-600': variant === 'success',
          'bg-amber-100 text-amber-600': variant === 'warning',
        })}>
          {icon}
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p className="text-slate-500 text-xs font-medium uppercase tracking-wide truncate">{title}</p>
        <p className="text-2xl font-bold text-slate-900 mt-0.5">{value}</p>
        {subtitle && <p className="text-slate-400 text-xs mt-0.5">{subtitle}</p>}
      </div>
    </div>
  )
}

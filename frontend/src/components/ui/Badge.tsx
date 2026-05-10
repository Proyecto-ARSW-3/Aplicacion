import clsx from 'clsx'

type Level = 'alto' | 'medio' | 'bajo'

const styles: Record<Level, string> = {
  alto: 'bg-red-100 text-red-700 ring-red-200',
  medio: 'bg-amber-100 text-amber-700 ring-amber-200',
  bajo: 'bg-green-100 text-green-700 ring-green-200',
}

const labels: Record<Level, string> = {
  alto: 'Alto Riesgo',
  medio: 'Riesgo Medio',
  bajo: 'Bajo Riesgo',
}

export default function Badge({ level }: { level: Level }) {
  return (
    <span className={clsx('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ring-1', styles[level])}>
      {labels[level]}
    </span>
  )
}

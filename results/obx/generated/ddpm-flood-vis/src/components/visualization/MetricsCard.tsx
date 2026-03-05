import type { ScenarioMetrics } from '#/lib/data/types'
import { cn } from '#/lib/utils'

interface MetricsCardProps {
  metrics: ScenarioMetrics
}

export function MetricsCard({ metrics }: MetricsCardProps) {
  const rows = [
    { label: 'R²', value: metrics.r2.toFixed(3), good: metrics.r2 > 0.5 },
    { label: 'RMSE', value: `${metrics.rmse.toFixed(3)} m` },
    { label: 'Bias', value: `${metrics.bias > 0 ? '+' : ''}${metrics.bias.toFixed(3)} m` },
    { label: 'MAE', value: `${metrics.mae.toFixed(3)} m` },
    { label: 'Uncertainty', value: `${metrics.mean_uncertainty.toFixed(3)} m` },
  ]

  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Metrics</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {rows.map(({ label, value, good }) => (
          <div key={label} className="contents">
            <span className="text-muted-foreground">{label}</span>
            <span className={cn(
              'text-right font-mono',
              good !== undefined && (good ? 'text-green-500' : 'text-red-400'),
            )}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

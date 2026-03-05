import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { AllMetricsEntry } from '#/lib/data/types'
import { CHART_AXIS_STYLE, CHART_GRID_STYLE, CHART_TOOLTIP_STYLE } from '#/lib/constants'
import { cn } from '#/lib/utils'

interface MetricsBarChartProps {
  metrics: AllMetricsEntry[]
  featuredThetas: number[]
}

type MetricKey = 'r2' | 'rmse' | 'bias' | 'mae'

const METRIC_OPTIONS: { key: MetricKey; label: string }[] = [
  { key: 'r2', label: 'R\u00B2' },
  { key: 'rmse', label: 'RMSE' },
  { key: 'bias', label: 'Bias' },
  { key: 'mae', label: 'MAE' },
]

export function MetricsBarChart({ metrics, featuredThetas }: MetricsBarChartProps) {
  const [activeMetric, setActiveMetric] = useState<MetricKey>('r2')

  const activeLabel = METRIC_OPTIONS.find((o) => o.key === activeMetric)!.label

  const data = useMemo(() =>
    metrics
      .slice()
      .sort((a, b) => a.theta - b.theta)
      .map((m) => ({
        ...m,
        label: `${m.storm} ${m.theta.toFixed(2)}`,
        isFeatured: featuredThetas.some((t) => Math.abs(t - m.theta) < 0.01),
      })),
    [metrics, featuredThetas],
  )

  return (
    <div className="space-y-3">
      <div className="flex gap-1.5 flex-wrap">
        {METRIC_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            onClick={() => setActiveMetric(opt.key)}
            className={cn(
              'px-3 py-1 rounded-md text-xs font-medium transition-colors border',
              activeMetric === opt.key
                ? 'bg-primary/15 text-foreground border-primary/50'
                : 'bg-muted text-muted-foreground border-border hover:border-muted-foreground/30',
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <div className="w-full h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis
              dataKey="label" tick={{ ...CHART_AXIS_STYLE, fontSize: 10 }}
              angle={-35} textAnchor="end" height={60}
            />
            <YAxis
              tick={CHART_AXIS_STYLE}
              label={{ value: activeLabel, angle: -90, position: 'insideLeft', offset: 5, ...CHART_AXIS_STYLE }}
            />
            <Tooltip
              contentStyle={CHART_TOOLTIP_STYLE}
              formatter={(value: number) => [value.toFixed(4), activeLabel]}
            />
            <Bar dataKey={activeMetric} radius={[4, 4, 0, 0]}>
              {data.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={barColor(entry[activeMetric], activeMetric)}
                  stroke={entry.isFeatured ? '#a1a1aa' : 'transparent'}
                  strokeWidth={entry.isFeatured ? 2 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function barColor(value: number, metric: MetricKey): string {
  if (metric === 'r2') return value > 0.5 ? '#22c55e' : value > 0 ? '#eab308' : '#ef4444'
  if (metric === 'rmse' || metric === 'mae') return value < 0.2 ? '#22c55e' : value < 0.5 ? '#eab308' : '#ef4444'
  return Math.abs(value) < 0.1 ? '#22c55e' : Math.abs(value) < 0.3 ? '#eab308' : '#ef4444'
}

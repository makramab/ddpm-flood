import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip,
} from 'recharts'
import type { ThetaDistribution, ScenarioSummary } from '#/lib/data/types'
import { CHART_AXIS_STYLE, CHART_GRID_STYLE, CHART_TOOLTIP_STYLE } from '#/lib/constants'
import { binData } from '#/lib/data/histogram'

interface ThetaHistogramProps {
  distribution: ThetaDistribution
  scenarios: ScenarioSummary[]
}

const BIN_COUNT = 20

const SCENARIO_COLORS: Record<string, string> = {
  dorian_low: '#22c55e',
  dorian_high: '#ef4444',
  arthur: '#f59e0b',
}

export function ThetaHistogram({ distribution, scenarios }: ThetaHistogramProps) {
  const bins = useMemo(() => binData(distribution.training, BIN_COUNT), [distribution.training])

  return (
    <div className="w-full h-[350px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={bins} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid {...CHART_GRID_STYLE} />
          <XAxis
            dataKey="label" tick={CHART_AXIS_STYLE} interval={3}
            label={{ value: 'Peak Surge \u03B8 (m)', position: 'insideBottom', offset: -10, ...CHART_AXIS_STYLE }}
          />
          <YAxis
            tick={CHART_AXIS_STYLE}
            label={{ value: 'Count', angle: -90, position: 'insideLeft', offset: 5, ...CHART_AXIS_STYLE }}
          />
          <Tooltip
            contentStyle={CHART_TOOLTIP_STYLE}
            labelFormatter={(label) => `\u03B8 \u2248 ${label}m`}
            formatter={(value: number) => [`${value}`, 'Count']}
          />
          <Bar dataKey="count" fill="#64748b" radius={[2, 2, 0, 0]} />
          {scenarios.map((s) => (
            <ReferenceLine
              key={s.id} x={findClosestBinLabel(bins, s.theta)}
              stroke={SCENARIO_COLORS[s.id] ?? '#ffffff'}
              strokeWidth={2} strokeDasharray="4 2"
              label={{ value: `${s.name.split(' ')[1]} ${s.theta.toFixed(2)}`, fill: SCENARIO_COLORS[s.id] ?? '#ffffff', fontSize: 11, position: 'top' }}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function findClosestBinLabel(bins: { label: string; min: number; max: number }[], value: number): string {
  let closest = bins[0]
  let minDist = Infinity
  for (const bin of bins) {
    const mid = (bin.min + bin.max) / 2
    const dist = Math.abs(mid - value)
    if (dist < minDist) {
      minDist = dist
      closest = bin
    }
  }
  return closest.label
}

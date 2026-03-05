import { useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { NodeData } from '#/lib/data/types'
import { CHART_AXIS_STYLE, CHART_GRID_STYLE, CHART_TOOLTIP_STYLE } from '#/lib/constants'

interface ScatterPlotProps {
  nodeData: NodeData
  r2: number
}

// 34,854 / 17 ≈ 2,050 points — empirically chosen for Recharts SVG render performance
const SAMPLE_STEP = 17

export function ScatterPlot({ nodeData, r2 }: ScatterPlotProps) {
  const { points, max } = useMemo(() => {
    const pts: { truth: number; prediction: number }[] = []
    let maxVal = 0
    for (let i = 0; i < nodeData.truth.length; i += SAMPLE_STEP) {
      const t = nodeData.truth[i]
      const p = nodeData.prediction[i]
      if (t == null || p == null) continue
      pts.push({ truth: t, prediction: p })
      maxVal = Math.max(maxVal, t, p)
    }
    return { points: pts, max: Math.ceil(maxVal * 10) / 10 }
  }, [nodeData.truth, nodeData.prediction])

  return (
    <div className="w-full h-[350px]">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid {...CHART_GRID_STYLE} />
          <XAxis
            type="number" dataKey="truth" name="Ground Truth"
            domain={[0, max]} label={{ value: 'Ground Truth (m)', position: 'insideBottom', offset: -10, ...CHART_AXIS_STYLE }}
            tick={CHART_AXIS_STYLE}
          />
          <YAxis
            type="number" dataKey="prediction" name="Prediction"
            domain={[0, max]} label={{ value: 'Prediction (m)', angle: -90, position: 'insideLeft', offset: 5, ...CHART_AXIS_STYLE }}
            tick={CHART_AXIS_STYLE}
          />
          <Tooltip
            contentStyle={CHART_TOOLTIP_STYLE}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(value: number) => `${value.toFixed(3)} m`}
          />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: max, y: max }]}
            stroke="#6b7280" strokeDasharray="6 4" strokeWidth={1.5}
          />
          <Scatter data={points} fill="#22d3ee" fillOpacity={0.35} r={2.5} />
          {/* R² annotation */}
          <text x={70} y={30} fill="#e5e7eb" fontSize={14} fontWeight={600}>
            R² = {r2.toFixed(4)}
          </text>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

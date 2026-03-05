import type { LayerType } from '#/lib/data/types'
import { getColorScale, getGradientStops } from '#/lib/data/colors'

interface ColorLegendProps {
  layer: LayerType
}

export function ColorLegend({ layer }: ColorLegendProps) {
  const scale = getColorScale(layer)
  const stops = getGradientStops(layer)
  const gradientCSS = stops
    .map((c, i) => `rgba(${c[0]},${c[1]},${c[2]},${c[3] / 255}) ${(i / (stops.length - 1)) * 100}%`)
    .join(', ')

  return (
    <div className="bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 text-white text-xs">
      <p className="mb-1 font-medium">{scale.label}</p>
      <div
        className="h-3 w-40 rounded-sm"
        style={{ background: `linear-gradient(to right, ${gradientCSS})` }}
      />
      <div className="flex justify-between mt-0.5">
        <span>{scale.min} {scale.unit}</span>
        <span>{scale.max} {scale.unit}</span>
      </div>
    </div>
  )
}

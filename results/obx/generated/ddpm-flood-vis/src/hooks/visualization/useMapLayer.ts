import { useMemo } from 'react'
import { ScatterplotLayer } from '@deck.gl/layers'
import type { PickingInfo } from '@deck.gl/core'
import type { NodeCoords, ScenarioData, LayerType } from '#/lib/data/types'
import { surgeColor } from '#/lib/data/colors'
import { COLOR_RANGES } from '#/lib/constants'

export function useMapLayer(
  coords: NodeCoords | null,
  scenario: ScenarioData | null,
  activeLayer: LayerType,
) {
  const layers = useMemo(() => {
    if (!coords || !scenario) return []

    const values = scenario.node_data[activeLayer]
    const positions = Array.from({ length: coords.lats.length }, (_, i) => i)

    return [
      new ScatterplotLayer({
        id: 'surge-nodes',
        data: positions,
        getPosition: (i: number) => [coords.lons[i], coords.lats[i]] as [number, number],
        getFillColor: (i: number) => surgeColor(values[i], activeLayer),
        getRadius: 40,
        radiusMinPixels: 1.5,
        radiusMaxPixels: 8,
        pickable: true,
        updateTriggers: {
          getFillColor: [scenario.id, activeLayer],
        },
      }),
    ]
  }, [coords, scenario, activeLayer])

  const getTooltip = useMemo(() => {
    return (info: PickingInfo) => {
      if (!info.index || info.index < 0 || !coords || !scenario) return null
      const i = info.index
      const val = scenario.node_data[activeLayer][i]
      const range = COLOR_RANGES[activeLayer]
      return `Node ${coords.indices[i]}\n${coords.lats[i].toFixed(4)}, ${coords.lons[i].toFixed(4)}\n${range.label}: ${val !== null ? `${val.toFixed(3)} ${range.unit}` : 'N/A'}`
    }
  }, [coords, scenario, activeLayer])

  return { layers, getTooltip }
}

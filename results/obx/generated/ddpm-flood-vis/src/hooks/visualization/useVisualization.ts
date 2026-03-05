import { useState } from 'react'
import type { LayerType } from '#/lib/data/types'
import { DEFAULT_LAYER, DEFAULT_MAP_STYLE, type MapStyleKey, MAP_STYLES } from '#/lib/constants'
import { useScenarioData } from '#/hooks/visualization/useScenarioData'
import { useMapLayer } from '#/hooks/visualization/useMapLayer'

export function useVisualization(initialScenarioId: string, initialLayer?: LayerType) {
  const [activeScenarioId, setActiveScenarioId] = useState(initialScenarioId)
  const [activeLayer, setActiveLayer] = useState<LayerType>(initialLayer ?? DEFAULT_LAYER)
  const [mapStyleKey, setMapStyleKey] = useState<MapStyleKey>(DEFAULT_MAP_STYLE)

  const { coords, scenario, scenarios, isLoading, error } = useScenarioData(activeScenarioId)
  const { layers, getTooltip } = useMapLayer(coords, scenario, activeLayer)

  return {
    activeScenarioId,
    setActiveScenarioId,
    activeLayer,
    setActiveLayer,
    mapStyleKey,
    setMapStyleKey,
    mapStyle: MAP_STYLES[mapStyleKey],
    coords,
    scenario,
    scenarios,
    isLoading,
    error,
    layers,
    getTooltip,
  }
}

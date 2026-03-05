import type { ScenarioSummary, ScenarioMetrics, LayerType } from '#/lib/data/types'
import type { MapStyleKey } from '#/lib/constants'
import { Card, CardContent } from '#/components/ui/card'
import { Separator } from '#/components/ui/separator'
import { ScenarioPicker } from '#/components/visualization/ScenarioPicker'
import { LayerToggle } from '#/components/visualization/LayerToggle'
import { MetricsCard } from '#/components/visualization/MetricsCard'
import { MapStyleSwitcher } from '#/components/visualization/MapStyleSwitcher'

interface ControlPanelProps {
  scenarios: ScenarioSummary[]
  activeScenarioId: string
  onScenarioChange: (id: string) => void
  activeLayer: LayerType
  onLayerChange: (layer: LayerType) => void
  metrics: ScenarioMetrics | null
  mapStyleKey: MapStyleKey
  onMapStyleChange: (style: MapStyleKey) => void
}

export function ControlPanel({
  scenarios,
  activeScenarioId,
  onScenarioChange,
  activeLayer,
  onLayerChange,
  metrics,
  mapStyleKey,
  onMapStyleChange,
}: ControlPanelProps) {
  return (
    <Card className="w-full md:w-80 bg-background/90 backdrop-blur-sm shadow-lg">
      <CardContent className="p-3 space-y-3">
        <ScenarioPicker scenarios={scenarios} activeId={activeScenarioId} onSelect={onScenarioChange} />
        <Separator />
        <LayerToggle activeLayer={activeLayer} onLayerChange={onLayerChange} />
        <Separator />
        <MapStyleSwitcher activeStyle={mapStyleKey} onStyleChange={onMapStyleChange} />
        {metrics && (
          <>
            <Separator />
            <MetricsCard metrics={metrics} />
          </>
        )}
      </CardContent>
    </Card>
  )
}

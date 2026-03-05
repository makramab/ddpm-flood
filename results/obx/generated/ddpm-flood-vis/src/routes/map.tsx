import { createFileRoute } from '@tanstack/react-router'
import { lazy, Suspense } from 'react'
import type { LayerType } from '#/lib/data/types'
import { DEFAULT_SCENARIO, DEFAULT_LAYER } from '#/lib/constants'
import { useVisualization } from '#/hooks/visualization/useVisualization'
import { ControlPanel } from '#/components/visualization/ControlPanel'
import { ColorLegend } from '#/components/visualization/ColorLegend'
import { LoadingOverlay } from '#/components/visualization/LoadingOverlay'
import { ErrorBanner } from '#/components/visualization/ErrorBanner'

const SurgeMap = lazy(() =>
  import('#/components/visualization/SurgeMap').then((m) => ({ default: m.SurgeMap }))
)

export const Route = createFileRoute('/map')({
  validateSearch: (search: Record<string, unknown>) => ({
    scenario: (search.scenario as string) || DEFAULT_SCENARIO,
    layer: (search.layer as LayerType) || DEFAULT_LAYER,
  }),
  component: MapPage,
})

function MapPage() {
  const { scenario: initialScenario, layer: initialLayer } = Route.useSearch()

  const viz = useVisualization(initialScenario, initialLayer)

  return (
    <div className="relative" style={{ height: 'calc(100vh - 56px)' }}>
      {viz.error && <ErrorBanner message={viz.error.message} />}
      {viz.isLoading && !viz.error && <LoadingOverlay />}

      {typeof window !== 'undefined' && (
        <Suspense fallback={<LoadingOverlay />}>
          <SurgeMap
            layers={viz.layers}
            getTooltip={viz.getTooltip}
            mapStyle={viz.mapStyle}
          />
        </Suspense>
      )}

      {/* Control panel: floating top-right on desktop, bottom on mobile */}
      <div className="absolute bottom-4 left-4 right-4 md:bottom-auto md:top-4 md:right-4 md:left-auto z-20">
        <ControlPanel
          scenarios={viz.scenarios}
          activeScenarioId={viz.activeScenarioId}
          onScenarioChange={viz.setActiveScenarioId}
          activeLayer={viz.activeLayer}
          onLayerChange={viz.setActiveLayer}
          metrics={viz.scenario?.metrics ?? null}
          mapStyleKey={viz.mapStyleKey}
          onMapStyleChange={viz.setMapStyleKey}
        />
      </div>

      {/* Color legend: bottom-left on desktop */}
      <div className="hidden md:block absolute bottom-4 left-4 z-20">
        <ColorLegend layer={viz.activeLayer} />
      </div>
    </div>
  )
}

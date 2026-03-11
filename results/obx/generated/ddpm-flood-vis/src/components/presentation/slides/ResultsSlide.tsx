import { lazy, Suspense } from 'react'
import { Link } from '@tanstack/react-router'
import { cn } from '#/lib/utils'
import { useVisualization } from '#/hooks/visualization/useVisualization'
import { LayerToggle } from '#/components/visualization/LayerToggle'
import { ColorLegend } from '#/components/visualization/ColorLegend'
import { MAP_STYLES } from '#/lib/constants'

const SurgeMap = lazy(() =>
  import('#/components/visualization/SurgeMap').then((m) => ({ default: m.SurgeMap }))
)

const SCENARIOS = [
  {
    id: 'dorian_low',
    title: 'Dorian (low \u03B8)',
    theta: '0.701m',
    r2: '0.95',
    rmse: '0.060m',
    verdict: 'Success',
    variant: 'success' as const,
    detail: 'In-distribution: parametric storm (93% of training data), \u03B8 well within training range',
  },
  {
    id: 'dorian_high',
    title: 'Dorian (high \u03B8)',
    theta: '3.357m',
    r2: '-1.26',
    rmse: '0.956m',
    verdict: 'Failure',
    variant: 'warning' as const,
    detail: 'Extrapolation: \u03B8 above training max (2.88m), parametric storm',
  },
  {
    id: 'arthur',
    title: 'Arthur',
    theta: '2.049m',
    r2: '-2.62',
    rmse: '0.600m',
    verdict: 'Failure',
    variant: 'danger' as const,
    detail: 'Cross-storm: \u03B8 within range, real meteorological forcing',
  },
]

const variantStyles = {
  success: { border: 'border-green-500/30', badge: 'bg-green-500/10 text-green-400', accent: 'text-green-400' },
  warning: { border: 'border-amber-500/30', badge: 'bg-amber-500/10 text-amber-400', accent: 'text-amber-400' },
  danger: { border: 'border-red-500/30', badge: 'bg-red-500/10 text-red-400', accent: 'text-red-400' },
}

export function ResultsSlide() {
  const viz = useVisualization('dorian_low', 'prediction')

  return (
    <div className="max-w-5xl w-full space-y-5">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Results</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">Three Scenarios, One Story</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-5">
        {/* Map panel with controls above */}
        <div className="space-y-3">
          {/* Controls row: scenario pills left, layer toggle right */}
          <div className="flex items-end justify-between gap-4">
            <div className="flex gap-1.5">
              {SCENARIOS.map((s) => {
                const style = variantStyles[s.variant]
                const isActive = viz.activeScenarioId === s.id
                return (
                  <button
                    key={s.id}
                    onClick={() => viz.setActiveScenarioId(s.id)}
                    className={cn(
                      'text-sm px-3 py-1.5 rounded-md font-medium transition-all',
                      isActive
                        ? `${style.badge} ring-1 ring-current`
                        : 'bg-muted/50 text-muted-foreground hover:bg-muted/80',
                    )}
                  >
                    {s.title}
                  </button>
                )
              })}
            </div>
            <LayerToggle activeLayer={viz.activeLayer} onLayerChange={viz.setActiveLayer} />
          </div>

          {/* Map */}
          <div className="relative rounded-xl overflow-hidden border border-border h-[360px] md:h-[460px]">
            {typeof window !== 'undefined' && (
              <Suspense fallback={
                <div className="absolute inset-0 flex items-center justify-center bg-muted/50">
                  <p className="text-sm text-muted-foreground">Loading map...</p>
                </div>
              }>
                <SurgeMap
                  layers={viz.layers}
                  getTooltip={viz.getTooltip}
                  mapStyle={MAP_STYLES.dark}
                />
              </Suspense>
            )}
            <div className="absolute bottom-3 left-3 z-10">
              <ColorLegend layer={viz.activeLayer} />
            </div>
          </div>
        </div>

        {/* Scenario cards */}
        <div className="space-y-3 lg:pt-10">
          {SCENARIOS.map((s) => {
            const style = variantStyles[s.variant]
            const isActive = viz.activeScenarioId === s.id
            return (
              <button
                key={s.title}
                onClick={() => viz.setActiveScenarioId(s.id)}
                className={cn(
                  'w-full text-left rounded-xl border p-4 space-y-2 transition-all',
                  style.border,
                  isActive ? 'ring-1 ring-primary/50' : 'opacity-60 hover:opacity-80',
                )}
              >
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-foreground">{s.title}</h3>
                  <span className={cn('text-xs px-2 py-0.5 rounded-full font-medium', style.badge)}>
                    {s.verdict}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">&theta;</p>
                    <p className="text-sm font-mono font-medium text-foreground">{s.theta}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">R&sup2;</p>
                    <p className={cn('text-sm font-mono font-bold', style.accent)}>{s.r2}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">RMSE</p>
                    <p className="text-sm font-mono font-medium text-foreground">{s.rmse}</p>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground leading-relaxed">{s.detail}</p>
              </button>
            )
          })}
        </div>
      </div>

      <div className="flex justify-end">
        <Link to="/0x-primer" hash="results" className="text-sm text-primary hover:text-primary/80 transition-colors shrink-0">
          Primer: Full results &rarr;
        </Link>
      </div>
    </div>
  )
}

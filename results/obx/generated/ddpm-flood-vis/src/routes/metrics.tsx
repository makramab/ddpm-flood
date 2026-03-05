import { createFileRoute } from '@tanstack/react-router'
import { useState, useMemo } from 'react'
import { DEFAULT_SCENARIO } from '#/lib/constants'
import { useMetricsData } from '#/hooks/visualization/useMetricsData'
import { ScenarioPicker } from '#/components/visualization/ScenarioPicker'
import { ScatterPlot } from '#/components/visualization/ScatterPlot'
import { ThetaHistogram } from '#/components/visualization/ThetaHistogram'
import { MetricsBarChart } from '#/components/visualization/MetricsBarChart'
import { LoadingOverlay } from '#/components/visualization/LoadingOverlay'
import { ErrorBanner } from '#/components/visualization/ErrorBanner'

export const Route = createFileRoute('/metrics')({
  validateSearch: (search: Record<string, unknown>) => ({
    scenario: (search.scenario as string) || DEFAULT_SCENARIO,
  }),
  component: MetricsPage,
})

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <h2 className="text-sm font-semibold text-muted-foreground mb-3">{title}</h2>
      {children}
    </div>
  )
}

function MetricsPage() {
  const { scenario: initialScenario } = Route.useSearch()
  const [activeScenarioId, setActiveScenarioId] = useState(initialScenario)

  const { allMetrics, thetaDistribution, scenario, scenarios, isLoading, error } =
    useMetricsData(activeScenarioId)

  const featuredThetas = useMemo(() => scenarios.map((s) => s.theta), [scenarios])

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-4 md:px-6 py-6 space-y-6">
        <h1 className="text-xl font-bold text-foreground">Metrics Dashboard</h1>

        {error && <ErrorBanner message={error.message} />}
        {isLoading && !error && <LoadingOverlay />}

        {scenarios.length > 0 && (
          <ScenarioPicker
            scenarios={scenarios}
            activeId={activeScenarioId}
            onSelect={setActiveScenarioId}
          />
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {scenario && (
            <ChartCard title={`Prediction vs Ground Truth — ${scenario.name} (${scenario.label})`}>
              <ScatterPlot nodeData={scenario.node_data} r2={scenario.metrics.r2} />
            </ChartCard>
          )}

          {thetaDistribution && scenarios.length > 0 && (
            <ChartCard title="Training &#x3B8; Distribution">
              <ThetaHistogram distribution={thetaDistribution} scenarios={scenarios} />
            </ChartCard>
          )}
        </div>

        {allMetrics.length > 0 && (
          <ChartCard title="Cross-Scenario Metrics Comparison (10 validation scenarios)">
            <MetricsBarChart metrics={allMetrics} featuredThetas={featuredThetas} />
          </ChartCard>
        )}
      </div>
    </div>
  )
}

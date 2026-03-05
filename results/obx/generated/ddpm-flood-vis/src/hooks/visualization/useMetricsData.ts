import { useState, useEffect } from 'react'
import type { AllMetricsEntry, ThetaDistribution } from '#/lib/data/types'
import { loadAllMetrics, loadThetaDistribution } from '#/lib/data/loader'
import { useScenarioData } from '#/hooks/visualization/useScenarioData'

export function useMetricsData(scenarioId: string) {
  const { scenario, scenarios, isLoading, error } = useScenarioData(scenarioId)
  const [allMetrics, setAllMetrics] = useState<AllMetricsEntry[]>([])
  const [thetaDistribution, setThetaDistribution] = useState<ThetaDistribution | null>(null)
  const [metricsError, setMetricsError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([loadAllMetrics(), loadThetaDistribution()])
      .then(([m, t]) => {
        if (cancelled) return
        setAllMetrics(m)
        setThetaDistribution(t)
      })
      .catch((e) => {
        if (!cancelled) setMetricsError(e instanceof Error ? e : new Error(String(e)))
      })
    return () => { cancelled = true }
  }, [])

  return {
    allMetrics,
    thetaDistribution,
    scenario,
    scenarios,
    isLoading,
    error: error ?? metricsError,
  }
}

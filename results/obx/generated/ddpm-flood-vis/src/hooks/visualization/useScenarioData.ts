import { useState, useEffect } from 'react'
import type { NodeCoords, ScenarioData, ScenarioSummary } from '#/lib/data/types'
import { loadNodeCoords, loadScenarioIndex, loadScenarioData } from '#/lib/data/loader'

export function useScenarioData(scenarioId: string) {
  const [coords, setCoords] = useState<NodeCoords | null>(null)
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([])
  const [scenario, setScenario] = useState<ScenarioData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // Load coords + scenario index on mount
  useEffect(() => {
    let cancelled = false
    Promise.all([loadNodeCoords(), loadScenarioIndex()])
      .then(([c, s]) => {
        if (cancelled) return
        setCoords(c)
        setScenarios(s)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e : new Error(String(e)))
      })
    return () => { cancelled = true }
  }, [])

  // Load scenario data on id change
  useEffect(() => {
    if (!scenarioId) return
    let cancelled = false
    setIsLoading(true)
    loadScenarioData(scenarioId)
      .then((data) => {
        if (cancelled) return
        setScenario(data)
        setIsLoading(false)
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e : new Error(String(e)))
          setIsLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [scenarioId])

  return { coords, scenario, scenarios, isLoading, error }
}

import type { AllMetricsEntry, NodeCoords, ScenarioData, ScenarioSummary, ThetaDistribution } from '#/lib/data/types'

const cache = new Map<string, unknown>()

async function fetchJson<T>(path: string): Promise<T> {
  const cached = cache.get(path)
  if (cached) return cached as T

  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`)

  const data = await res.json() as T
  cache.set(path, data)
  return data
}

export function loadNodeCoords(): Promise<NodeCoords> {
  return fetchJson<NodeCoords>('/viz_data/node_coords.json')
}

export function loadScenarioIndex(): Promise<ScenarioSummary[]> {
  return fetchJson<ScenarioSummary[]>('/viz_data/scenarios.json')
}

export function loadScenarioData(id: string): Promise<ScenarioData> {
  return fetchJson<ScenarioData>(`/viz_data/scenario_${id}.json`)
}

export function loadThetaDistribution(): Promise<ThetaDistribution> {
  return fetchJson<ThetaDistribution>('/viz_data/theta_distribution.json')
}

export function loadAllMetrics(): Promise<AllMetricsEntry[]> {
  return fetchJson<AllMetricsEntry[]>('/viz_data/all_metrics.json')
}

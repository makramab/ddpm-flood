export type LayerType = 'truth' | 'prediction' | 'error' | 'uncertainty'

export interface ScenarioMetrics {
  r2: number
  rmse: number
  bias: number
  mae: number
  mean_uncertainty: number
}

export interface ScenarioSummary {
  id: string
  name: string
  label: string
  theta: number
  metrics: ScenarioMetrics
}

export interface NodeData {
  prediction: (number | null)[]
  truth: (number | null)[]
  error: (number | null)[]
  uncertainty: (number | null)[]
}

export interface ScenarioData extends ScenarioSummary {
  node_data: NodeData
  patch_data: NodeData
}

export interface NodeCoords {
  lats: number[]
  lons: number[]
  indices: number[]
}

export interface ThetaDistribution {
  training: number[]
  validation: Record<string, number[]>
  stats: { min: number; max: number; mean: number; std: number }
}

export interface AllMetricsEntry {
  storm: string
  theta: number
  r2: number
  rmse: number
  bias: number
  mae: number
  mean_std: number
}

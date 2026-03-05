import type { LayerType } from '#/lib/data/types'

export const SIDEBAR_WIDTH_EXPANDED = '14rem'
export const SIDEBAR_WIDTH_COLLAPSED = '4rem'

export const MAP_CONFIG = {
  initialViewState: {
    longitude: -75.9,
    latitude: 35.25,
    zoom: 9,
    pitch: 0,
    bearing: 0,
  },
} as const

export const MAP_STYLES = {
  dark: 'mapbox://styles/mapbox/dark-v11',
  light: 'mapbox://styles/mapbox/light-v11',
  satellite: 'mapbox://styles/mapbox/satellite-streets-v12',
} as const

export type MapStyleKey = keyof typeof MAP_STYLES

export const COLOR_RANGES: Record<LayerType, { min: number; max: number; label: string; unit: string }> = {
  truth: { min: 0, max: 3.6, label: 'Ground Truth Surge', unit: 'm' },
  prediction: { min: 0, max: 3.6, label: 'Predicted Surge', unit: 'm' },
  error: { min: -2.0, max: 2.0, label: 'Prediction Error', unit: 'm' },
  uncertainty: { min: 0, max: 0.35, label: 'Uncertainty (Std Dev)', unit: 'm' },
}

export const DEFAULT_SCENARIO = 'dorian_low'
export const DEFAULT_LAYER: LayerType = 'truth'
export const DEFAULT_MAP_STYLE: MapStyleKey = 'dark'

// Shared Recharts styling — neutral grays that work in both light and dark themes
export const CHART_AXIS_STYLE = { fill: '#71717a', fontSize: 12 } as const
export const CHART_TOOLTIP_STYLE = { backgroundColor: 'var(--popover)', color: 'var(--popover-foreground)', border: '1px solid var(--border)', borderRadius: 8 } as const
export const CHART_GRID_STYLE = { strokeDasharray: '3 3', stroke: 'var(--border)' } as const

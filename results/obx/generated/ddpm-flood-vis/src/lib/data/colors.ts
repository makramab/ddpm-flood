import type { LayerType } from '#/lib/data/types'
import { COLOR_RANGES } from '#/lib/constants'

type RGBA = [number, number, number, number]

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function lerpColor(c1: RGBA, c2: RGBA, t: number): RGBA {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
    Math.round(lerp(c1[3], c2[3], t)),
  ]
}

function multiStopGradient(stops: RGBA[], t: number): RGBA {
  const n = stops.length - 1
  const idx = t * n
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, n)
  return lerpColor(stops[lo], stops[hi], idx - lo)
}

// Yellow -> Orange -> Red -> Dark red (sequential surge)
const SURGE_STOPS: RGBA[] = [
  [255, 255, 178, 220], // pale yellow
  [254, 204, 92, 230],  // yellow-orange
  [253, 141, 60, 240],  // orange
  [240, 59, 32, 250],   // red
  [189, 0, 38, 255],    // dark red
]

// Blue (under) -> White (zero) -> Red (over) (diverging error)
const ERROR_STOPS: RGBA[] = [
  [33, 102, 172, 255],  // blue
  [146, 197, 222, 240], // light blue
  [247, 247, 247, 200], // white
  [239, 138, 98, 240],  // light red
  [178, 24, 43, 255],   // red
]

// Light purple -> Dark purple (sequential uncertainty)
const UNCERTAINTY_STOPS: RGBA[] = [
  [242, 240, 247, 200], // very light purple
  [203, 201, 226, 220], // light purple
  [158, 154, 200, 235], // medium purple
  [117, 107, 177, 245], // purple
  [84, 39, 143, 255],   // dark purple
]

export function surgeColor(value: number | null, layer: LayerType): RGBA {
  if (value === null) return [0, 0, 0, 0]

  const range = COLOR_RANGES[layer]
  const t = clamp((value - range.min) / (range.max - range.min), 0, 1)

  switch (layer) {
    case 'truth':
    case 'prediction':
      return multiStopGradient(SURGE_STOPS, t)
    case 'error':
      return multiStopGradient(ERROR_STOPS, t)
    case 'uncertainty':
      return multiStopGradient(UNCERTAINTY_STOPS, t)
  }
}

export function getColorScale(layer: LayerType) {
  return COLOR_RANGES[layer]
}

export function getGradientStops(layer: LayerType): RGBA[] {
  switch (layer) {
    case 'truth':
    case 'prediction':
      return SURGE_STOPS
    case 'error':
      return ERROR_STOPS
    case 'uncertainty':
      return UNCERTAINTY_STOPS
  }
}

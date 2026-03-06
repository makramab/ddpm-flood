export const CELLS = 20
export const GRID_COLS = 5
export const GRID_ROWS = 4

// Real surge data from Patch 1030 (Hurricane Arthur, theta=2.049m)
// 20 ADCIRC mesh nodes near Hatteras Island, Outer Banks
// All 20 values are unique — mix of exposed ocean-side (high) and sheltered sound-side (low)
// Source: scenario_arthur.json, node_data.truth mapped via patches.json assignments[1030]
export const RAW_SURGE = [
  0.4256, 0.6939, 0.2345, 0.8062, 0.7928,
  0.8197, 0.3977, 0.7470, 0.7169, 0.2556,
  0.2416, 0.6605, 0.9153, 0.7461, 0.9646,
  0.7129, 0.5610, 0.6813, 0.6498, 0.2613,
]

// Normalize to [0, 1] for visualization
const MIN_SURGE = Math.min(...RAW_SURGE)
const MAX_SURGE = Math.max(...RAW_SURGE)
export const CLEAN_PROFILE = RAW_SURGE.map(v => (v - MIN_SURGE) / (MAX_SURGE - MIN_SURGE))

// Linear beta schedule matching our pipeline
const BETA_START = 0.0001
const BETA_END = 0.02

export function getAlphaBarT(t: number): number {
  if (t === 0) return 1
  let alphaBar = 1
  for (let i = 1; i <= t; i++) {
    const beta = BETA_START + (BETA_END - BETA_START) * (i / 1000)
    alphaBar *= (1 - beta)
  }
  return alphaBar
}

// Seeded random for deterministic noise across renders
export function seededRandom(seed: number): number {
  const x = Math.sin(seed * 9301 + 49297) * 49297
  return x - Math.floor(x)
}

// Generate noised profile at timestep t: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
export function noisedProfile(t: number): number[] {
  const aBar = getAlphaBarT(t)
  const signalScale = Math.sqrt(aBar)
  const noiseScale = Math.sqrt(1 - aBar)
  return CLEAN_PROFILE.map((val, i) => {
    const noise = (seededRandom(t * 100 + i) - 0.5) * 2
    const noised = signalScale * val + noiseScale * noise
    return Math.max(0, Math.min(1, noised))
  })
}

// Cell color based on surge value (0=low, 1=high) with desaturation for noise
// Blue (low surge) → Cyan → Green → Yellow → Red (high surge)
export function cellColor(value: number, noiseFraction: number): string {
  const hue = 240 - value * 210
  const sat = 80 - noiseFraction * 55
  const light = 50 + noiseFraction * 10
  return `hsl(${hue}, ${sat}%, ${light}%)`
}

// Conditioning values for Patch 1030 (Hurricane Arthur)
export const CONDITIONING = {
  theta: 2.049,
  lat: 35.7402,
  lon: -75.6237,
  depth: 3.61,
} as const

export const COND_LABELS = ['\u03B8', 'lat', 'lon', 'd'] as const

// Raw noise values at timestep t (the seeded noise used in forward process)
export function rawNoise(t: number): number[] {
  return CLEAN_PROFILE.map((_, i) => (seededRandom(t * 100 + i) - 0.5) * 2)
}

// One reverse diffusion step: compute x_{t-1} from x_t and predicted noise
// This is the deterministic part of the posterior mean (ignoring added stochasticity)
export function reverseStep(xt: number[], noise: number[], t: number): number[] {
  const beta = BETA_START + (BETA_END - BETA_START) * (t / 1000)
  const alpha = 1 - beta
  const aBar = getAlphaBarT(t)
  const coeff = beta / Math.sqrt(1 - aBar)
  return xt.map((x, i) => {
    const mean = (x - coeff * noise[i]) / Math.sqrt(alpha)
    return Math.max(0, Math.min(1, mean))
  })
}

// Diverging color for noise values in [-1, 1]: blue <- 0 -> red
export function noiseColor(value: number): string {
  const v = Math.max(-1, Math.min(1, value))
  if (v < 0) {
    const i = -v
    return `hsl(220, ${60 + i * 30}%, ${55 - i * 15}%)`
  }
  return `hsl(10, ${60 + v * 30}%, ${55 - v * 15}%)`
}

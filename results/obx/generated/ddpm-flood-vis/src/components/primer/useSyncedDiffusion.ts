import { useState, useEffect, useRef } from 'react'
import { noisedProfile } from './diffusion-utils'

// Filmstrip keyframes in reverse order (denoising direction)
const KEYFRAMES = [1000, 800, 600, 400, 200, 0]
const SEGMENT_DURATION = 1500 // ms per segment transition
const HOLD_DURATION = 1500 // ms to hold at t=0
const RESET_PAUSE = 600 // ms before restarting

// Precompute profiles at every 10 steps for smooth interpolation
const STEP = 10
const PROFILE_CACHE: Map<number, number[]> = new Map()
for (let t = 0; t <= 1000; t += STEP) {
  PROFILE_CACHE.set(t, noisedProfile(t))
}

export function getProfileAtT(t: number): number[] {
  const tLow = Math.floor(t / STEP) * STEP
  const tHigh = Math.min(tLow + STEP, 1000)
  const frac = (t - tLow) / STEP

  const low = PROFILE_CACHE.get(tLow)!
  const high = PROFILE_CACHE.get(tHigh)!

  if (frac === 0) return low
  return low.map((v, i) => v + (high[i] - v) * frac)
}

export interface SyncedDiffusionState {
  /** Current timestep value (1000 → 0) */
  currentT: number
  /** Current bar profile at currentT */
  profile: number[]
  /** Index into the filmstrip TIMESTEPS array [0,200,400,600,800,1000] that is highlighted.
   *  Maps to the panel index (0=t0, 1=t200, ..., 5=t1000). -1 = none highlighted. */
  highlightIndex: number
}

export function useSyncedDiffusion(isInView: boolean, shouldReduceMotion: boolean | null): SyncedDiffusionState {
  const [state, setState] = useState<SyncedDiffusionState>({
    currentT: 0,
    profile: PROFILE_CACHE.get(0)!,
    highlightIndex: 0,
  })
  const rafRef = useRef<number>(0)

  useEffect(() => {
    if (!isInView || shouldReduceMotion) {
      // Static: show clean state, highlight t=0 panel
      setState({
        currentT: 0,
        profile: PROFILE_CACHE.get(0)!,
        highlightIndex: 0,
      })
      return
    }

    let startTime: number | null = null
    const numSegments = KEYFRAMES.length - 1 // 5 segments
    const denoiseTotal = numSegments * SEGMENT_DURATION
    const totalCycle = denoiseTotal + HOLD_DURATION + RESET_PAUSE

    const animate = (now: number) => {
      if (startTime === null) startTime = now
      const elapsed = (now - startTime) % totalCycle

      if (elapsed < denoiseTotal) {
        // Which segment are we in?
        const segmentIdx = Math.min(Math.floor(elapsed / SEGMENT_DURATION), numSegments - 1)
        const segmentProgress = (elapsed - segmentIdx * SEGMENT_DURATION) / SEGMENT_DURATION

        const fromT = KEYFRAMES[segmentIdx]
        const toT = KEYFRAMES[segmentIdx + 1]
        const currentT = Math.round(fromT + (toT - fromT) * segmentProgress)

        // Highlight = filmstrip panel index for the target keyframe
        // KEYFRAMES are [1000,800,600,400,200,0], filmstrip panels are [0,200,400,600,800,1000]
        // So filmstrip index for a timestep t is: t / 200
        // Highlight the target panel (where we're heading)
        const highlightT = toT
        const highlightIndex = highlightT / 200 // 0→0, 200→1, 400→2, 600→3, 800→4, 1000→5

        setState({
          currentT,
          profile: getProfileAtT(currentT),
          highlightIndex,
        })
      } else if (elapsed < denoiseTotal + HOLD_DURATION) {
        // Hold at clean
        setState({
          currentT: 0,
          profile: PROFILE_CACHE.get(0)!,
          highlightIndex: 0,
        })
      } else {
        // Reset pause: show t=1000
        setState({
          currentT: 1000,
          profile: PROFILE_CACHE.get(1000)!,
          highlightIndex: 5,
        })
      }

      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [isInView, shouldReduceMotion])

  return state
}

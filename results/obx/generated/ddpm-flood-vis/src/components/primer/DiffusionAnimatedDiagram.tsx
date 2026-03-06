import { LazyMotion, domAnimation } from 'motion/react'
import { CELLS, GRID_COLS, GRID_ROWS, cellColor } from './diffusion-utils'
import type { SyncedDiffusionState } from './useSyncedDiffusion'

// Larger grid for the animated panel
const CELL_SIZE = 28
const CELL_GAP = 3
const GRID_PAD = 8
const GRID_W = GRID_COLS * (CELL_SIZE + CELL_GAP) - CELL_GAP
const GRID_H = GRID_ROWS * (CELL_SIZE + CELL_GAP) - CELL_GAP
const PAD = 16
const GRID_X = PAD + GRID_PAD
const GRID_Y = 10 + GRID_PAD
const PANEL_W = GRID_W + GRID_PAD * 2
const PANEL_H = GRID_H + GRID_PAD * 2
const COUNTER_Y = 10 + PANEL_H + 28
const LABEL_Y = COUNTER_Y + 18
const SVG_W = PANEL_W + PAD * 2
const SVG_H = LABEL_Y + 8

interface Props {
  state: SyncedDiffusionState
}

export function DiffusionAnimatedDiagram({ state }: Props) {
  const { currentT, profile } = state
  const noiseFraction = currentT / 1000

  return (
    <div className="my-6 flex justify-center">
      <LazyMotion features={domAnimation}>
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          role="img"
          aria-label="Animated grid showing the reverse denoising process: cells transition from random noise colors at t=1000 to a clear surge pattern at t=0, synchronized with the filmstrip above."
          className="h-auto"
          style={{ maxWidth: 280 }}
        >
          {/* Panel background */}
          <rect
            x={PAD}
            y={10}
            width={PANEL_W}
            height={PANEL_H}
            rx={6}
            fill="var(--muted)"
            stroke="var(--border)"
            strokeWidth={1}
          />

          {/* Grid cells */}
          {profile.map((val, ci) => {
            const col = ci % GRID_COLS
            const row = Math.floor(ci / GRID_COLS)
            const cx = GRID_X + col * (CELL_SIZE + CELL_GAP)
            const cy = GRID_Y + row * (CELL_SIZE + CELL_GAP)
            return (
              <rect
                key={ci}
                x={cx}
                y={cy}
                width={CELL_SIZE}
                height={CELL_SIZE}
                rx={3}
                fill={cellColor(val, noiseFraction)}
              />
            )
          })}

          {/* Step counter */}
          <text
            x={SVG_W / 2}
            y={COUNTER_Y}
            textAnchor="middle"
            fill="var(--foreground)"
            fontSize={16}
            fontWeight={700}
            fontFamily="monospace"
          >
            t = {currentT}
          </text>

          {/* Label */}
          <text
            x={SVG_W / 2}
            y={LABEL_Y}
            textAnchor="middle"
            fill="var(--muted-foreground)"
            fontSize={10}
          >
            {currentT === 0 ? 'Clean surge data' : currentT === 1000 ? 'Pure noise' : 'Denoising...'}
          </text>
        </svg>
      </LazyMotion>
    </div>
  )
}

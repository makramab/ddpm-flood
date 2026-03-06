import { LazyMotion, domAnimation } from 'motion/react'
import { CELLS, GRID_COLS, GRID_ROWS, noisedProfile, cellColor } from './diffusion-utils'
import type { SyncedDiffusionState } from './useSyncedDiffusion'

const PANELS = 6
const TIMESTEPS = [0, 200, 400, 600, 800, 1000]

// Precompute all panel profiles
const PANEL_PROFILES = TIMESTEPS.map(t => noisedProfile(t))

// Grid cell dimensions within each panel
const CELL_SIZE = 14
const CELL_GAP = 2
const PANEL_PAD = 4
const PANEL_W = GRID_COLS * (CELL_SIZE + CELL_GAP) - CELL_GAP + PANEL_PAD * 2
const PANEL_H = GRID_ROWS * (CELL_SIZE + CELL_GAP) - CELL_GAP + PANEL_PAD * 2
const PANEL_GAP = 12
const PANEL_Y = 50
const ARROW_Y_TOP = 28
const ARROW_Y_BOT = PANEL_Y + PANEL_H + 32
const LABEL_Y = PANEL_Y + PANEL_H + 14
const TOTAL_W = PANELS * PANEL_W + (PANELS - 1) * PANEL_GAP + 40
const SVG_H = PANEL_Y + PANEL_H + 58

interface Props {
  state: SyncedDiffusionState
}

export function DiffusionProcessDiagram({ state }: Props) {
  const { highlightIndex } = state
  const panelX = (i: number) => 20 + i * (PANEL_W + PANEL_GAP)

  return (
    <div className="my-6">
      <LazyMotion features={domAnimation}>
        <svg
          viewBox={`0 0 ${TOTAL_W} ${SVG_H}`}
          role="img"
          aria-label="Filmstrip diagram showing the diffusion process: 6 panels display a grid of surge values at increasing noise levels from t=0 (clean data) to t=1000 (pure noise). Cells are colored by surge magnitude. An arrow above shows forward diffusion and an arrow below shows reverse denoising."
          className="w-full h-auto"
        >
          <defs>
            <marker id="arrow-right" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" fill="var(--muted-foreground)" />
            </marker>
            <marker id="arrow-left" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" fill="var(--primary)" />
            </marker>
          </defs>

          {/* Top arrow */}
          <line
            x1={panelX(0) + PANEL_W / 2}
            y1={ARROW_Y_TOP}
            x2={panelX(PANELS - 1) + PANEL_W / 2}
            y2={ARROW_Y_TOP}
            stroke="var(--muted-foreground)"
            strokeWidth={1.2}
            markerEnd="url(#arrow-right)"
          />
          <text
            x={TOTAL_W / 2}
            y={ARROW_Y_TOP - 8}
            textAnchor="middle"
            fill="var(--muted-foreground)"
            fontSize={9}
            fontWeight={500}
          >
            Fixed forward diffusion process
          </text>

          {/* Panels */}
          {PANEL_PROFILES.map((profile, pi) => {
            const px = panelX(pi)
            const isHighlighted = highlightIndex === pi
            const noiseFraction = pi / (PANELS - 1)
            const opacity = isHighlighted ? 1 : 0.4
            return (
              <g
                key={pi}
                opacity={opacity}
                style={{ transition: 'opacity 0.3s ease' }}
              >
                <rect
                  x={px}
                  y={PANEL_Y}
                  width={PANEL_W}
                  height={PANEL_H}
                  rx={4}
                  fill="var(--muted)"
                  stroke={isHighlighted ? 'var(--primary)' : 'var(--border)'}
                  strokeWidth={isHighlighted ? 2 : 1}
                  style={{ transition: 'stroke 0.3s ease, stroke-width 0.3s ease' }}
                />
                {profile.map((val, ci) => {
                  const col = ci % GRID_COLS
                  const row = Math.floor(ci / GRID_COLS)
                  const cx = px + PANEL_PAD + col * (CELL_SIZE + CELL_GAP)
                  const cy = PANEL_Y + PANEL_PAD + row * (CELL_SIZE + CELL_GAP)
                  return (
                    <rect
                      key={ci}
                      x={cx}
                      y={cy}
                      width={CELL_SIZE}
                      height={CELL_SIZE}
                      rx={2}
                      fill={cellColor(val, noiseFraction)}
                    />
                  )
                })}
                <text
                  x={px + PANEL_W / 2}
                  y={LABEL_Y}
                  textAnchor="middle"
                  fill="var(--foreground)"
                  fontSize={8}
                  fontFamily="monospace"
                  fontWeight={isHighlighted ? 700 : 400}
                >
                  t = {TIMESTEPS[pi]}
                </text>
              </g>
            )
          })}

          {/* Side labels */}
          <text x={panelX(0)} y={PANEL_Y - 4} fill="var(--primary)" fontSize={9} fontWeight={600}>
            Surge data
          </text>
          <text
            x={panelX(PANELS - 1) + PANEL_W}
            y={PANEL_Y - 4}
            textAnchor="end"
            fill="var(--muted-foreground)"
            fontSize={9}
            fontWeight={600}
          >
            Pure noise
          </text>

          {/* Bottom arrow */}
          <line
            x1={panelX(PANELS - 1) + PANEL_W / 2}
            y1={ARROW_Y_BOT}
            x2={panelX(0) + PANEL_W / 2}
            y2={ARROW_Y_BOT}
            stroke="var(--primary)"
            strokeWidth={1.2}
            markerEnd="url(#arrow-left)"
          />
          <text
            x={TOTAL_W / 2}
            y={ARROW_Y_BOT + 14}
            textAnchor="middle"
            fill="var(--primary)"
            fontSize={9}
            fontWeight={500}
          >
            Generative reverse denoising process
          </text>
        </svg>
      </LazyMotion>
    </div>
  )
}

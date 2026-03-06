import { useRef } from 'react'
import { LazyMotion, domAnimation, m, useInView, useReducedMotion } from 'motion/react'
import {
  CELLS, GRID_COLS, GRID_ROWS, CLEAN_PROFILE,
  CONDITIONING, COND_LABELS,
  noisedProfile, rawNoise, reverseStep,
  cellColor, noiseColor,
} from './diffusion-utils'

// Same grid format as DDPMSection diagrams — each cell = one node's surge value
// t=600 matches the filmstrip panel in Section 3 exactly (same values, same colors)
const T = 600
const CELL = 18
const GAP = 2
const PAD = 6
const GW = GRID_COLS * (CELL + GAP) - GAP
const GH = GRID_ROWS * (CELL + GAP) - GAP
const PW = GW + PAD * 2
const PH = GH + PAD * 2
const ARROW_W = 52

const MX = 12
const P1 = MX
const P2 = P1 + PW + ARROW_W
const P3 = P2 + PW + ARROW_W
const SVG_W = P3 + PW + MX

const GRID_Y = 28
const LABEL_Y = GRID_Y + PH + 16
const SUB_Y = LABEL_Y + 13
const LOOP_Y = SUB_Y + 18
const NOTE_Y = LOOP_Y + 38
const COND_Y = NOTE_Y + 22
const SVG_H = COND_Y + 22

// Precompute data at t=600 — use same noiseFraction as Section 3 filmstrip
// Filmstrip: t=600 is panel index 3 of [0,200,400,600,800,1000], noiseFraction = 3/5
const NF = 3 / 5
// x_{t-1} uses a slightly lower noise fraction so the color difference is perceptible
// The actual values from reverseStep differ, but at NF=0.6 the desaturation hides the difference
const NF_PREV = NF - 0.04
const noised = noisedProfile(T)
const noise = rawNoise(T)
const oneLessNoisy = reverseStep(noised, noise, T)
const condVals = [CONDITIONING.theta, CONDITIONING.lat, CONDITIONING.lon, CONDITIONING.depth]
const condUnits = ['m', '\u00B0', '\u00B0', 'm']

// Animation timing
const STAGGER = 0.25       // delay between each stage (left to right)
const FADE_DUR = 0.3       // entrance fade duration
const ANNO_DELAY = STAGGER * 5 + 0.1 // after all 3 grids + 2 arrows
const PULSE_DURATION = 3.5 // seconds per full pulse cycle
const PULSE_STAGGER = 0.6  // delay between grids in the pulse

function SurgeGrid({ x, y, values, colorFn, highlight }: {
  x: number; y: number
  values: number[]
  colorFn: (val: number) => string
  highlight?: boolean
}) {
  return (
    <g>
      <rect x={x} y={y} width={PW} height={PH} rx={5}
        fill="var(--muted)"
        stroke={highlight ? 'var(--primary)' : 'var(--border)'}
        strokeWidth={highlight ? 1.5 : 1} />
      {values.map((val, i) => {
        const col = i % GRID_COLS
        const row = Math.floor(i / GRID_COLS)
        return (
          <rect key={i}
            x={x + PAD + col * (CELL + GAP)}
            y={y + PAD + row * (CELL + GAP)}
            width={CELL} height={CELL} rx={2}
            fill={colorFn(val)} />
        )
      })}
    </g>
  )
}

function HArrow({ x1, x2, y, label, sublabel }: {
  x1: number; x2: number; y: number; label: string; sublabel?: string
}) {
  const cy = y + PH / 2
  const mx = (x1 + x2) / 2
  return (
    <g>
      <line x1={x1 + 4} y1={cy} x2={x2 - 6} y2={cy}
        stroke="var(--border)" strokeWidth={1.2} />
      <polygon points={`${x2 - 7},${cy - 3} ${x2 - 1},${cy} ${x2 - 7},${cy + 3}`}
        fill="var(--border)" />
      <text x={mx} y={cy - 8} textAnchor="middle"
        fontSize={9} fontWeight={600} fill="var(--foreground)">{label}</text>
      {sublabel && (
        <text x={mx} y={cy + 12} textAnchor="middle"
          fontSize={6.5} fill="var(--muted-foreground)">{sublabel}</text>
      )}
    </g>
  )
}

export function UNetOutputDiagram() {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: false, amount: 0.3 })
  const reducedMotion = useReducedMotion()

  const showPulse = isInView && !reducedMotion

  // Elements appear left-to-right: grid1, arrow1, grid2, arrow2, grid3
  const entranceProps = (index: number) => ({
    initial: reducedMotion ? undefined : { opacity: 0 as number, y: 6 as number },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: { duration: FADE_DUR, delay: index * STAGGER, ease: 'easeOut' as const },
  })

  const fadeProps = (delay: number) => ({
    initial: reducedMotion ? undefined : { opacity: 0 as number },
    whileInView: { opacity: 1 },
    viewport: { once: true },
    transition: { duration: FADE_DUR, delay, ease: 'easeOut' as const },
  })

  return (
    <div ref={ref} className="my-6">
      <LazyMotion features={domAnimation}>
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} role="img"
          aria-label="Three 5 by 4 grids showing one reverse diffusion step at t=600: noised input, predicted noise, and the result after removing a tiny amount of noise. A loop arrow shows this repeats 1,000 times to reach clean data."
          className="w-full h-auto">

          {/* Pulse keyframes for grids + loop arrow flow */}
          <defs>
            <style>{`
              @keyframes output-pulse {
                0%, 10% { opacity: 0; }
                25% { opacity: 0.08; }
                40%, 100% { opacity: 0; }
              }
              @keyframes loop-flow {
                0% { stroke-dashoffset: 0; }
                100% { stroke-dashoffset: -16; }
              }
            `}</style>
            {/* Arrowhead marker — auto-orients along the path tangent */}
            <marker id="loop-arrow" viewBox="0 0 8 8" refX="1" refY="4"
              markerWidth="6" markerHeight="6" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--primary)" />
            </marker>
          </defs>

          {/* Title */}
          <text x={SVG_W / 2} y={16} textAnchor="middle"
            fontSize={10} fontWeight={600} fill="var(--foreground)">
            One step of reverse diffusion (t = {T})
          </text>
          <text x={SVG_W / 2} y={26} textAnchor="middle"
            fontSize={7} fill="var(--muted-foreground)">
            Same Patch 1030 / Hurricane Arthur data as the diffusion diagrams above
          </text>

          {/* Grid 1: noised input x_t — entrance index 0 */}
          <m.g {...entranceProps(0)}>
            <SurgeGrid x={P1} y={GRID_Y} values={noised}
              colorFn={v => cellColor(v, NF)} />
            {/* Pulse overlay */}
            <rect x={P1} y={GRID_Y} width={PW} height={PH} rx={5}
              fill="var(--primary)" pointerEvents="none"
              style={{
                opacity: 0,
                animation: showPulse
                  ? `output-pulse ${PULSE_DURATION}s ease-in-out 0s infinite`
                  : 'none',
              }} />
          </m.g>

          {/* Arrow 1 -> 2 — entrance index 1 */}
          <m.g {...entranceProps(1)}>
            <HArrow x1={P1 + PW} x2={P2} y={GRID_Y}
              label="U-Net" sublabel="predicts noise" />
          </m.g>

          {/* Grid 2: predicted noise epsilon — entrance index 2 */}
          <m.g {...entranceProps(2)}>
            <SurgeGrid x={P2} y={GRID_Y} values={noise}
              colorFn={v => noiseColor(v)} />
            <rect x={P2} y={GRID_Y} width={PW} height={PH} rx={5}
              fill="var(--primary)" pointerEvents="none"
              style={{
                opacity: 0,
                animation: showPulse
                  ? `output-pulse ${PULSE_DURATION}s ease-in-out ${PULSE_STAGGER}s infinite`
                  : 'none',
              }} />
          </m.g>

          {/* Arrow 2 -> 3 — entrance index 3 */}
          <m.g {...entranceProps(3)}>
            <HArrow x1={P2 + PW} x2={P3} y={GRID_Y}
              label="formula" sublabel="denoise" />
          </m.g>

          {/* Grid 3: x_{t-1} — one step less noisy — entrance index 4 */}
          <m.g {...entranceProps(4)}>
            <SurgeGrid x={P3} y={GRID_Y} values={oneLessNoisy}
              colorFn={v => cellColor(v, NF_PREV)} />
            <rect x={P3} y={GRID_Y} width={PW} height={PH} rx={5}
              fill="var(--primary)" pointerEvents="none"
              style={{
                opacity: 0,
                animation: showPulse
                  ? `output-pulse ${PULSE_DURATION}s ease-in-out ${PULSE_STAGGER * 2}s infinite`
                  : 'none',
              }} />
          </m.g>

          {/* Labels under each grid — fade in with their grid */}
          <m.g {...fadeProps(0 * STAGGER)}>
            <text x={P1 + PW / 2} y={LABEL_Y} textAnchor="middle"
              fontSize={11} fontWeight={700} fill="var(--foreground)" fontFamily="monospace">
              x<tspan fontSize={7} dy={3}>t</tspan>
            </text>
            <text x={P1 + PW / 2} y={SUB_Y} textAnchor="middle"
              fontSize={8} fill="var(--muted-foreground)">noised input</text>
          </m.g>

          <m.g {...fadeProps(2 * STAGGER)}>
            <text x={P2 + PW / 2} y={LABEL_Y} textAnchor="middle"
              fontSize={11} fontWeight={700} fill="var(--foreground)" fontFamily="monospace">
              {'\u03B5'}
            </text>
            <text x={P2 + PW / 2} y={SUB_Y} textAnchor="middle"
              fontSize={8} fill="var(--muted-foreground)">predicted noise</text>
          </m.g>

          <m.g {...fadeProps(4 * STAGGER)}>
            <text x={P3 + PW / 2} y={LABEL_Y} textAnchor="middle"
              fontSize={11} fontWeight={700} fill="var(--foreground)" fontFamily="monospace">
              x<tspan fontSize={7} dy={3}>t-1</tspan>
            </text>
            <text x={P3 + PW / 2} y={SUB_Y} textAnchor="middle"
              fontSize={8} fill="var(--muted-foreground)">slightly less noisy</text>
          </m.g>

          {/* Loop arrow + annotations — appear after all grids */}
          <m.g {...fadeProps(ANNO_DELAY)}>
            {/* Loop arrow with animated dash flow + auto-oriented arrowhead */}
            <path
              d={`M ${P3 + PW / 2} ${LOOP_Y - 4}
                  C ${P3 + PW / 2} ${LOOP_Y + 16},
                    ${P1 + PW / 2} ${LOOP_Y + 16},
                    ${P1 + PW / 2} ${LOOP_Y - 4}`}
              fill="none" stroke="var(--primary)" strokeWidth={1.2}
              strokeDasharray="5 3" opacity={0.6}
              markerEnd="url(#loop-arrow)"
              style={{
                animation: showPulse
                  ? 'loop-flow 1.2s linear infinite'
                  : 'none',
              }} />
            <text x={SVG_W / 2} y={LOOP_Y + 22} textAnchor="middle"
              fontSize={8} fontWeight={600} fill="var(--primary)">
              Repeat 1,000 times (our config): x{'\u2091\u208B\u2081'} becomes the new x{'\u2091'}
            </text>
          </m.g>

          {/* After 1,000 steps note */}
          <m.g {...fadeProps(ANNO_DELAY + 0.15)}>
            <text x={SVG_W / 2} y={NOTE_Y + 6} textAnchor="middle"
              fontSize={8} fill="var(--muted-foreground)">
              After all 1,000 steps: pure noise {'\u2192'} clean surge data (the t=0 grid from Section 3)
            </text>
          </m.g>

          {/* Conditioning row */}
          <m.g {...fadeProps(ANNO_DELAY + 0.3)}>
            <text x={MX} y={COND_Y} fontSize={8} fontWeight={600} fill="var(--foreground)">
              Conditioning (fixed, never noised):
            </text>
            {condVals.map((val, i) => {
              const spacing = (SVG_W - MX * 2 - 160) / 4
              const bx = MX + 160 + i * spacing
              const bw = spacing - 6
              return (
                <g key={i}>
                  <rect x={bx} y={COND_Y - 10} width={bw} height={16} rx={3}
                    fill="var(--primary)" opacity={0.1} />
                  <text x={bx + bw / 2} y={COND_Y + 1} textAnchor="middle"
                    fontSize={7.5} fill="var(--foreground)" fontFamily="monospace">
                    {COND_LABELS[i]} = {Math.abs(val).toFixed(val === CONDITIONING.theta ? 3 : 2)}{condUnits[i]}
                  </text>
                </g>
              )
            })}
          </m.g>
        </svg>
      </LazyMotion>
    </div>
  )
}

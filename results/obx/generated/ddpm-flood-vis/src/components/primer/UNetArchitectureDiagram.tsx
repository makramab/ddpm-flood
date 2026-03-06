import { LazyMotion, domAnimation } from 'motion/react'

// Annotated vertical pipeline — plain English first, tensor shapes as secondary detail
// Tensor shapes verified from DDPM_REMD/denoising_diffusion_pytorch.py

const BOX_W = 300
const BOX_X = 40
const BOX_RX = 8
const CX = BOX_X + BOX_W / 2
const SNAP_X = BOX_X + BOX_W + 16 // right-side annotation column

interface Stage {
  y: number
  h: number
  title: string
  detail: string
  shapes?: string
  accent?: boolean
}

const STAGES: Stage[] = [
  { y: 20, h: 40, title: 'Input: 24 noised values', detail: '4 conditioning + 20 surge values', shapes: '(batch, 1, 24)' },
  { y: 92, h: 62, title: 'Compress (Encoder)', detail: '24 \u2192 12 \u2192 6 \u2192 3 positions', shapes: '32ch \u2192 64ch \u2192 64ch' },
  { y: 186, h: 62, title: 'Bottleneck: most compressed', detail: '3 positions \u00D7 128 features', shapes: 'Attention: every position sees every other', accent: true },
  { y: 280, h: 62, title: 'Reconstruct (Decoder)', detail: '3 \u2192 6 \u2192 12 \u2192 24 positions', shapes: '64ch \u2192 64ch \u2192 32ch' },
  { y: 374, h: 40, title: 'Output: 24 values (predicted noise \u03B5)', detail: 'Columns 0\u20133 masked \u2014 only surge noise matters' },
]

const ARROW_GAP = 32 // gap between boxes where arrows go
const SVG_H = 450

export function UNetArchitectureDiagram() {
  return (
    <div className="my-6">
      <LazyMotion features={domAnimation}>
        <svg viewBox={`0 0 520 ${SVG_H}`} role="img"
          aria-label="U-Net architecture as an annotated vertical pipeline: input flows down through compression (encoder), bottleneck with attention, reconstruction (decoder), to output. Skip connections are explained as encoder snapshots that help the decoder recover fine detail."
          className="w-full h-auto" style={{ maxWidth: 580 }}>

          {/* Boxes */}
          {STAGES.map((s, i) => (
            <g key={i}>
              <rect x={BOX_X} y={s.y} width={BOX_W} height={s.h} rx={BOX_RX}
                fill={s.accent ? 'var(--primary)' : 'var(--muted)'}
                opacity={s.accent ? 0.12 : 1}
                stroke={s.accent ? 'var(--primary)' : 'var(--border)'}
                strokeWidth={s.accent ? 1.5 : 1} />
              {s.accent && (
                <rect x={BOX_X} y={s.y} width={BOX_W} height={s.h} rx={BOX_RX}
                  fill="none" stroke="var(--primary)" strokeWidth={1.5} />
              )}
              <text x={CX} y={s.y + 18} textAnchor="middle"
                fontSize={10} fontWeight={700} fill="var(--foreground)">
                {s.title}
              </text>
              <text x={CX} y={s.y + 32} textAnchor="middle"
                fontSize={8} fill="var(--muted-foreground)">
                {s.detail}
              </text>
              {s.shapes && (
                <text x={CX} y={s.y + 46} textAnchor="middle"
                  fontSize={7} fill="var(--muted-foreground)" fontFamily="monospace" opacity={0.7}>
                  {s.shapes}
                </text>
              )}
            </g>
          ))}

          {/* Down arrows between boxes */}
          {[0, 1, 2, 3].map(i => {
            const fromBot = STAGES[i].y + STAGES[i].h
            const toTop = STAGES[i + 1].y
            const my = (fromBot + toTop) / 2
            return (
              <g key={`a${i}`}>
                <line x1={CX} y1={fromBot + 2} x2={CX} y2={toTop - 2}
                  stroke="var(--border)" strokeWidth={1.2} />
                <polygon
                  points={`${CX - 4},${toTop - 7} ${CX + 4},${toTop - 7} ${CX},${toTop - 1}`}
                  fill="var(--border)" />
              </g>
            )
          })}

          {/* Right-side annotation: skip connections */}
          {/* Bracket from Encoder to Decoder */}
          <path
            d={`M ${SNAP_X} ${STAGES[1].y + 10}
                C ${SNAP_X + 40} ${STAGES[1].y + 10},
                  ${SNAP_X + 40} ${STAGES[3].y + STAGES[3].h - 10},
                  ${SNAP_X} ${STAGES[3].y + STAGES[3].h - 10}`}
            fill="none" stroke="var(--primary)" strokeWidth={1.2}
            strokeDasharray="5 3" opacity={0.6} />
          {/* Arrowhead at decoder end */}
          <polygon
            points={`${SNAP_X + 3},${STAGES[3].y + STAGES[3].h - 15}
                     ${SNAP_X - 2},${STAGES[3].y + STAGES[3].h - 10}
                     ${SNAP_X + 3},${STAGES[3].y + STAGES[3].h - 5}`}
            fill="var(--primary)" opacity={0.6} />

          {/* Skip connection annotation text */}
          <g transform={`translate(${SNAP_X + 44}, ${(STAGES[1].y + STAGES[3].y + STAGES[3].h) / 2 - 20})`}>
            <text x={0} y={0} fontSize={8.5} fontWeight={600} fill="var(--primary)">
              Skip connections
            </text>
            <text x={0} y={14} fontSize={7.5} fill="var(--muted-foreground)">
              Encoder saves a snapshot
            </text>
            <text x={0} y={25} fontSize={7.5} fill="var(--muted-foreground)">
              at each level. Decoder uses
            </text>
            <text x={0} y={36} fontSize={7.5} fill="var(--muted-foreground)">
              these to recover fine detail
            </text>
            <text x={0} y={47} fontSize={7.5} fill="var(--muted-foreground)">
              that compression lost.
            </text>
          </g>

          {/* Timestep annotation (left side, pointing at all 3 middle boxes) */}
          <g>
            <rect x={0} y={STAGES[1].y + 8} width={30} height={20} rx={10}
              fill="var(--primary)" opacity={0.15} />
            <text x={15} y={STAGES[1].y + 22} textAnchor="middle"
              fontSize={10} fontWeight={700} fill="var(--primary)">t</text>
            {/* Arrows from t badge to each of the 3 middle boxes */}
            {[1, 2, 3].map(i => {
              const ty = STAGES[i].y + STAGES[i].h / 2
              return (
                <line key={`t${i}`}
                  x1={30} y1={STAGES[1].y + 18}
                  x2={BOX_X - 3} y2={ty}
                  stroke="var(--primary)" strokeWidth={0.8}
                  strokeDasharray="3 2" opacity={0.4} />
              )
            })}
            <text x={2} y={STAGES[1].y + 42} fontSize={6.5} fill="var(--muted-foreground)">
              Timestep
            </text>
            <text x={2} y={STAGES[1].y + 51} fontSize={6.5} fill="var(--muted-foreground)">
              tells model
            </text>
            <text x={2} y={STAGES[1].y + 60} fontSize={6.5} fill="var(--muted-foreground)">
              how noisy
            </text>
            <text x={2} y={STAGES[1].y + 69} fontSize={6.5} fill="var(--muted-foreground)">
              the input is
            </text>
          </g>
        </svg>
      </LazyMotion>
    </div>
  )
}

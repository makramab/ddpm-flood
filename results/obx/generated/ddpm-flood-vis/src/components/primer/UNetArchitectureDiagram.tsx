import { useRef, useState } from 'react'
import { LazyMotion, domAnimation, m, useInView, useReducedMotion, AnimatePresence } from 'motion/react'

// Annotated vertical pipeline — plain English first, tensor shapes as secondary detail
// Tensor shapes verified from DDPM_REMD/denoising_diffusion_pytorch.py
// Animation: staggered entrance (once) + looping data-flow pulse while in view
// Interactive: click any box to show a detail panel below the diagram

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

// Detail descriptions — all facts sourced from ModelSection, DDPMSection, and PRIMER_DIAGRAMS.md
const STAGE_DESCRIPTIONS: { title: string; paragraphs: string[] }[] = [
  {
    title: 'Input',
    paragraphs: [
      'The model receives a single row of 24 numbers as a 1D sequence with shape (batch, 1, 24). The first 4 values are conditioning columns \u2014 theta (\u03B8, peak surge at the reference gauge), latitude, longitude, and bathymetric depth \u2014 which are never corrupted with noise during training and stay fixed during generation.',
      'The remaining 20 values are surge heights at ADCIRC mesh nodes within a spatial patch. All 24 columns are normalized to the range [\u22121, 1] using min-max normalization before entering the network.',
      'During training, a random noise level t is chosen (from 1 to 1,000) and Gaussian noise of that intensity is added to the surge columns. The conditioning columns remain untouched \u2014 this is the inpainting trick that lets the model generate surge patterns conditioned on storm intensity and location.',
    ],
  },
  {
    title: 'Encoder',
    paragraphs: [
      'Three downsampling stages compress the spatial dimension from 24 positions down to 3. Each stage uses a 1D convolution with stride 2 (Conv1d, kernel=3, stride=2, padding=1), which halves the number of positions while increasing the number of feature channels.',
      'The channel path goes from 32 to 64 to 64 channels (base dimension 32 with multipliers 1, 2, 2). Compression forces the network to learn which patterns matter most \u2014 it cannot keep everything, so it must prioritize the most important spatial features.',
      'At each level, the encoder saves a copy of its output. These copies are the skip connections \u2014 they will be sent directly to the corresponding decoder level later, bypassing the bottleneck.',
    ],
  },
  {
    title: 'Bottleneck',
    paragraphs: [
      'The most compressed representation: just 3 spatial positions with 128 feature channels (multiplier 4). This is where the network has distilled the input into its most abstract form.',
      'A linear attention mechanism operates here, allowing every position to exchange information with every other position. This helps the model reason about relationships across the entire input \u2014 for example, how surge at one node relates to surge at distant nodes within the patch.',
      'The timestep t is also injected here (and at the encoder and decoder stages), telling the model how noisy the current input is. This is critical because the model needs to behave differently at high noise levels (where it must guess the overall structure) versus low noise levels (where it refines fine details).',
    ],
  },
  {
    title: 'Decoder',
    paragraphs: [
      'Three upsampling stages reverse the encoder\u2019s compression, expanding from 3 positions back to 24. Each stage uses a transposed 1D convolution (ConvTranspose1d, kernel=4, stride=2, padding=1) to double the spatial dimension, while channels decrease from 64 to 64 to 32.',
      'At each level, the decoder receives the skip connection from the corresponding encoder level \u2014 a copy of the encoder\u2019s output at that same resolution. This lets the decoder recover fine-grained spatial details that compression lost, rather than having to reconstruct everything from the bottleneck alone.',
    ],
  },
  {
    title: 'Output',
    paragraphs: [
      'The output is a 24-value sequence with the same shape as the input: (batch, 1, 24). Crucially, the network predicts noise (\u03B5), not clean data. It estimates what Gaussian noise was added to the input at timestep t.',
      'A separate, fixed mathematical formula (not learned) then uses this noise prediction to compute a slightly less noisy version of the input. One step barely changes anything visible \u2014 you need all 1,000 steps to go from pure noise to clean data.',
      'Columns 0\u20133 correspond to the conditioning values and are masked \u2014 the loss function (L2 / MSE) is only computed on columns 4\u201323, so the model only learns to predict noise on the surge values.',
    ],
  },
]

const SVG_H = 450

// Entrance timing (plays once)
const STAGGER = 0.2
const FADE_DUR = 0.35
const SKIP_DELAY = STAGGER * 5 + 0.1
const T_DELAY = SKIP_DELAY + 0.4

// Pulse timing (loops while in view)
const PULSE_DURATION = 4
const PULSE_STAGGER = 0.5

export function UNetArchitectureDiagram() {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: false, amount: 0.2 })
  const reducedMotion = useReducedMotion()
  const [selected, setSelected] = useState<number | null>(null)
  const [hasInteracted, setHasInteracted] = useState(false)

  // Center the t badge vertically between Encoder, Bottleneck, and Decoder midpoints
  const midY1 = STAGES[1].y + STAGES[1].h / 2
  const midY3 = STAGES[3].y + STAGES[3].h / 2
  const centerY = (midY1 + midY3) / 2

  const showPulse = isInView && !reducedMotion

  function handleBoxClick(i: number) {
    if (!hasInteracted) setHasInteracted(true)
    setSelected(prev => prev === i ? null : i)
  }

  return (
    <div className="my-6" ref={ref}>
      <LazyMotion features={domAnimation}>
        <svg viewBox={`0 0 520 ${SVG_H}`} role="img"
          aria-label="U-Net architecture as an annotated vertical pipeline: input flows down through compression (encoder), bottleneck with attention, reconstruction (decoder), to output. Click any box for a detailed explanation."
          className="w-full h-auto" style={{ maxWidth: 580 }}>

          {/* Pulse keyframes + hover style */}
          <defs>
            <style>{`
              @keyframes unet-pulse {
                0%, 12% { opacity: 0; }
                25% { opacity: 0.09; }
                38%, 100% { opacity: 0; }
              }
              .unet-box-hit { cursor: pointer; }
              .unet-box-hit:hover rect.unet-hover-target { opacity: 0.06; }
            `}</style>
          </defs>

          {/* Boxes — staggered entrance + looping pulse overlay + click targets */}
          {STAGES.map((s, i) => (
            <m.g key={i}
              initial={reducedMotion ? undefined : { opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: FADE_DUR, delay: i * STAGGER, ease: 'easeOut' as const }}
            >
              {/* Base box */}
              <rect x={BOX_X} y={s.y} width={BOX_W} height={s.h} rx={BOX_RX}
                fill={s.accent ? 'var(--primary)' : 'var(--muted)'}
                opacity={s.accent ? 0.12 : 1}
                stroke={selected === i ? 'var(--primary)' : (s.accent ? 'var(--primary)' : 'var(--border)')}
                strokeWidth={selected === i ? 2 : (s.accent ? 1.5 : 1)} />
              {s.accent && selected !== i && (
                <rect x={BOX_X} y={s.y} width={BOX_W} height={s.h} rx={BOX_RX}
                  fill="none" stroke="var(--primary)" strokeWidth={1.5} />
              )}

              {/* Pulse overlay */}
              <rect x={BOX_X} y={s.y} width={BOX_W} height={s.h} rx={BOX_RX}
                fill="var(--primary)" pointerEvents="none"
                style={{
                  opacity: 0,
                  animation: showPulse
                    ? `unet-pulse ${PULSE_DURATION}s ease-in-out ${i * PULSE_STAGGER}s infinite`
                    : 'none',
                }} />

              <text x={CX} y={s.y + 18} textAnchor="middle"
                fontSize={10} fontWeight={700} fill="var(--foreground)"
                pointerEvents="none">
                {s.title}
              </text>
              <text x={CX} y={s.y + 32} textAnchor="middle"
                fontSize={8} fill="var(--muted-foreground)"
                pointerEvents="none">
                {s.detail}
              </text>
              {s.shapes && (
                <text x={CX} y={s.y + 46} textAnchor="middle"
                  fontSize={7} fill="var(--muted-foreground)" fontFamily="monospace" opacity={0.7}
                  pointerEvents="none">
                  {s.shapes}
                </text>
              )}

              {/* Clickable hit area + hover highlight */}
              <g className="unet-box-hit" onClick={() => handleBoxClick(i)}
                role="button" tabIndex={0} aria-label={`Learn more about ${STAGE_DESCRIPTIONS[i].title}`}
                onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleBoxClick(i) } }}>
                <rect className="unet-hover-target"
                  x={BOX_X} y={s.y} width={BOX_W} height={s.h} rx={BOX_RX}
                  fill="var(--primary)" opacity={0} />
              </g>
            </m.g>
          ))}

          {/* Down arrows */}
          {[0, 1, 2, 3].map(i => {
            const fromBot = STAGES[i].y + STAGES[i].h
            const toTop = STAGES[i + 1].y
            return (
              <m.g key={`a${i}`}
                initial={reducedMotion ? undefined : { opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: FADE_DUR, delay: (i + 1) * STAGGER, ease: 'easeOut' as const }}
              >
                <line x1={CX} y1={fromBot + 2} x2={CX} y2={toTop - 2}
                  stroke="var(--border)" strokeWidth={1.2} />
                <polygon
                  points={`${CX - 4},${toTop - 7} ${CX + 4},${toTop - 7} ${CX},${toTop - 1}`}
                  fill="var(--border)" />
              </m.g>
            )
          })}

          {/* Right-side annotation: skip connections */}
          <m.g
            initial={reducedMotion ? undefined : { opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: FADE_DUR, delay: SKIP_DELAY, ease: 'easeOut' as const }}
          >
            <path
              d={`M ${SNAP_X} ${STAGES[1].y + 10}
                  C ${SNAP_X + 40} ${STAGES[1].y + 10},
                    ${SNAP_X + 40} ${STAGES[3].y + STAGES[3].h - 10},
                    ${SNAP_X} ${STAGES[3].y + STAGES[3].h - 10}`}
              fill="none" stroke="var(--primary)" strokeWidth={1.2}
              strokeDasharray="5 3" opacity={0.6} />
            <polygon
              points={`${SNAP_X + 3},${STAGES[3].y + STAGES[3].h - 15}
                       ${SNAP_X - 2},${STAGES[3].y + STAGES[3].h - 10}
                       ${SNAP_X + 3},${STAGES[3].y + STAGES[3].h - 5}`}
              fill="var(--primary)" opacity={0.6} />
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
          </m.g>

          {/* Timestep annotation */}
          <m.g
            initial={reducedMotion ? undefined : { opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: FADE_DUR, delay: T_DELAY, ease: 'easeOut' as const }}
          >
            <rect x={0} y={centerY - 10} width={30} height={20} rx={10}
              fill="var(--primary)" opacity={0.15} />
            <text x={15} y={centerY + 4} textAnchor="middle"
              fontSize={10} fontWeight={700} fill="var(--primary)">t</text>
            {[1, 2, 3].map(i => {
              const ty = STAGES[i].y + STAGES[i].h / 2
              return (
                <line key={`t${i}`}
                  x1={30} y1={centerY}
                  x2={BOX_X - 3} y2={ty}
                  stroke="var(--primary)" strokeWidth={0.8}
                  strokeDasharray="3 2" opacity={0.4} />
              )
            })}
            <text x={2} y={centerY + 24} fontSize={6.5} fill="var(--muted-foreground)">
              Timestep
            </text>
            <text x={2} y={centerY + 33} fontSize={6.5} fill="var(--muted-foreground)">
              tells model
            </text>
            <text x={2} y={centerY + 42} fontSize={6.5} fill="var(--muted-foreground)">
              how noisy
            </text>
            <text x={2} y={centerY + 51} fontSize={6.5} fill="var(--muted-foreground)">
              the input is
            </text>
          </m.g>
        </svg>

        {/* Hint text — fades away after first interaction */}
        <AnimatePresence>
          {!hasInteracted && (
            <m.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.5 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="text-center text-xs text-muted-foreground -mt-2 mb-2 select-none"
            >
              Click any box to learn more
            </m.p>
          )}
        </AnimatePresence>

        {/* Detail panel — appears below the diagram when a box is selected */}
        <AnimatePresence mode="wait">
          {selected !== null && (
            <m.div
              key={selected}
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2 }}
              className="rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 mt-1"
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-primary">
                  {STAGE_DESCRIPTIONS[selected].title}
                </h4>
                <button
                  onClick={() => setSelected(null)}
                  className="text-muted-foreground hover:text-foreground text-xs px-1.5 py-0.5 rounded hover:bg-muted transition-colors"
                  aria-label="Close detail panel"
                >
                  &times;
                </button>
              </div>
              <div className="space-y-2">
                {STAGE_DESCRIPTIONS[selected].paragraphs.map((p, i) => (
                  <p key={i} className="text-xs leading-relaxed text-muted-foreground">
                    {p}
                  </p>
                ))}
              </div>
            </m.div>
          )}
        </AnimatePresence>
      </LazyMotion>
    </div>
  )
}

import { useRef } from 'react'
import { useInView, useReducedMotion } from 'motion/react'
import { Section, Subsection, Prose, B, Mono, InfoCard, StepList, StatGrid } from './shared'
import { RAW_SURGE, GRID_COLS, GRID_ROWS, CLEAN_PROFILE, cellColor } from './diffusion-utils'
import { DiffusionProcessDiagram } from './DiffusionProcessDiagram'
import { DiffusionAnimatedDiagram } from './DiffusionAnimatedDiagram'
import { useSyncedDiffusion } from './useSyncedDiffusion'

export function DDPMSection() {
  const diagramRef = useRef<HTMLDivElement>(null)
  const isInView = useInView(diagramRef, { once: false, amount: 0.2 })
  const shouldReduceMotion = useReducedMotion()
  const diffusionState = useSyncedDiffusion(isInView, shouldReduceMotion)

  return (
    <Section id="ddpms" title="3. Background: Denoising Diffusion Probabilistic Models">
      <Subsection title="The intuition">
        <Prose>
          <p>
            Imagine you have a clear photograph. If you gradually add random static noise to it — a little more at each
            step — eventually the image becomes pure noise, indistinguishable from random pixels. A DDPM learns to{' '}
            <B>reverse this process</B>: given a noisy image, it predicts what the slightly-less-noisy version looks
            like. By chaining these small denoising steps together (typically 1,000 steps), it can start from pure
            random noise and gradually produce a realistic image.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="The training process (forward diffusion)">
        <Prose>
          <p>
            During training, the model is shown real data (in this project, real surge patterns from ADCIRC
            simulations). For each training example:
          </p>
        </Prose>
        <StepList
          steps={[
            'A random noise level t is chosen (from 1 to 1,000, where 1 = almost no noise and 1,000 = almost pure noise)',
            'Gaussian noise of that intensity is added to the data',
            'The model is asked to predict what noise was added',
            'The model\'s prediction is compared to the actual noise, and the error (mean squared error) is used to update the model\'s weights',
          ]}
        />
        <Prose>
          <p>
            Over hundreds of thousands of training steps, the model learns to recognize and remove noise at every
            intensity level.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="The generation process (reverse diffusion)">
        <StepList
          steps={[
            'Start with pure random noise (a vector of random numbers)',
            'Ask the model: "if this were data at noise level 1,000, what would it look like at noise level 999?"',
            'Repeat, stepping down through all 1,000 noise levels',
            'The result is a new, realistic data sample',
          ]}
        />
        <div ref={diagramRef}>
          <DiffusionProcessDiagram state={diffusionState} />
          <DiffusionAnimatedDiagram state={diffusionState} />
        </div>
        <InfoCard title="Reading the diagram">
          <p>
            Each grid above represents <B>one training row</B> — the surge values at 20 ADCIRC mesh
            nodes within a single spatial patch. Cell color encodes surge height:{' '}
            <span className="text-blue-400">blue</span> = low surge (sheltered, sound-side nodes),{' '}
            <span className="text-red-400">red</span> = high surge (exposed, ocean-side nodes). The values shown are
            real data from Patch 1030 during Hurricane Arthur (theta = 2.049 m, surge range 0.235 m &ndash; 0.965 m).
          </p>
          <p className="mt-2">
            In reality, the model processes these 20 values as a <B>flat 1D vector</B>, not a 2D grid &mdash; the grid
            layout is a visual convenience to make the noise-to-signal transition easier to see, similar to how classic
            diffusion papers show a photograph dissolving into static. The nodes within a patch are geographically
            nearby but irregularly spaced across the ADCIRC mesh, not arranged on a regular grid.
          </p>
          <details className="mt-4 group">
            <summary className="cursor-pointer text-xs font-medium text-primary hover:text-primary/80 select-none list-none flex items-center gap-1.5">
              <svg
                className="w-3.5 h-3.5 transition-transform group-open:rotate-90"
                viewBox="0 0 16 16"
                fill="currentColor"
              >
                <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              View actual data row used in this animation
            </summary>
            <div className="mt-3 space-y-3">
              <div>
                <p className="text-xs font-medium text-foreground mb-2">Conditioning columns (fixed)</p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {[
                    { label: '\u03B8 (theta)', value: '2.049 m' },
                    { label: 'Latitude', value: '35.7402\u00B0' },
                    { label: 'Longitude', value: '\u221275.6237\u00B0' },
                    { label: 'Depth', value: '3.61 m' },
                  ].map((item, i) => (
                    <div key={i} className="bg-primary/10 rounded px-2 py-1.5">
                      <p className="text-[10px] text-muted-foreground">{item.label}</p>
                      <p className="text-xs font-mono font-medium text-foreground">{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-xs font-medium text-foreground mb-2">Surge columns (generated by diffusion) — 5 &times; 4 grid matches the animation above</p>
                <div className="overflow-x-auto rounded border border-border">
                  <table className="w-full text-xs font-mono">
                    <tbody>
                      {Array.from({ length: GRID_ROWS }, (_, row) => (
                        <tr key={row} className={row > 0 ? 'border-t border-border' : ''}>
                          {Array.from({ length: GRID_COLS }, (_, col) => {
                            const idx = row * GRID_COLS + col
                            const val = RAW_SURGE[idx]
                            const norm = CLEAN_PROFILE[idx]
                            const color = cellColor(norm, 0)
                            return (
                              <td key={col} className="px-2 py-1.5 text-center">
                                <span className="inline-flex items-center gap-1.5">
                                  <span
                                    className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                                    style={{ backgroundColor: color }}
                                  />
                                  <span className="text-muted-foreground">{val.toFixed(3)} m</span>
                                </span>
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Source: Hurricane Arthur (Patch 1030), ADCIRC ground truth &mdash; scenario_arthur.json
              </p>
            </div>
          </details>
        </InfoCard>
      </Subsection>

      <Subsection title="The architecture: 1D U-Net">
        <Prose>
          <p>
            The neural network inside the DDPM is a <B>U-Net</B> — an architecture originally designed for image
            segmentation. It has an <B>encoder</B> path (which compresses the input into a compact representation) and a{' '}
            <B>decoder</B> path (which expands it back), connected by <B>skip connections</B> (shortcuts that allow
            fine-grained details to bypass the compression).
          </p>
          <p>
            In this project, the data is not a 2D image but a <B>1D sequence</B> of numbers (24 values: 4 conditioning
            values + 20 surge values). So the model uses <B>1D convolutions</B> instead of 2D, but the architecture is
            otherwise standard.
          </p>
        </Prose>
        <StatGrid
          items={[
            { label: 'Base dimension', value: '32 channels' },
            { label: 'Channel multipliers', value: '(1, 2, 2, 4)' },
            { label: 'Group normalization', value: '8 groups' },
            { label: 'Attention', value: 'Linear (middle block)' },
            { label: 'Resolution scales', value: '32 → 64 → 64 → 128' },
          ]}
        />
      </Subsection>

      <Subsection title="Conditioning via inpainting">
        <InfoCard title="The key trick" accent>
          <p>
            Some columns of the data are designated as "conditioning columns" and are{' '}
            <B>never corrupted with noise</B> during training. During generation, these columns are set to the desired
            values and held fixed at every denoising step. The model learns to generate the remaining columns in a way
            that is consistent with the fixed conditioning columns.
          </p>
          <p className="mt-2">
            This is identical to how image inpainting works — you fix part of the image (the known region) and ask the
            model to fill in the rest (the unknown region). Here, the "known region" is the storm severity and spatial
            location, and the "unknown region" is the surge pattern.
          </p>
        </InfoCard>
      </Subsection>

      <Subsection title="The noise schedule">
        <Prose>
          <p>
            The noise schedule defines how much noise is added at each of the 1,000 timesteps. This project uses a{' '}
            <B>linear schedule</B> from <Mono>beta_1 = 0.0001</Mono> to <Mono>beta_1000 = 0.02</Mono>. At early timesteps (t near
            0), very little noise is added. At late timesteps (t near 1,000), the data is almost completely overwhelmed
            by noise. The linear schedule is the simplest and most common choice.
          </p>
        </Prose>
      </Subsection>
    </Section>
  )
}

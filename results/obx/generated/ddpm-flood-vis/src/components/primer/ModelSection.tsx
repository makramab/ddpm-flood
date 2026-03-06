import { Section, Subsection, Prose, B, Mono, DataTable, InfoCard, CodeBlock, StatGrid, StepList } from './shared'
import { UNetArchitectureDiagram } from './UNetArchitectureDiagram'
import { UNetOutputDiagram } from './UNetOutputDiagram'

export function ModelSection() {
  return (
    <Section id="model" title="7. The Model: Architecture and Training Details">
      <Subsection title="Architecture summary">
        <DataTable
          headers={['Parameter', 'Value', 'Explanation']}
          rows={[
            ['Model type', '1D U-Net', 'Processes 1D sequences through encoder-decoder with skip connections'],
            ['Base dimension', '32', 'Number of feature channels at the first layer'],
            ['Channel multipliers', '(1, 2, 2, 4)', 'Channels at each depth: 32 -> 64 -> 64 -> 128'],
            ['Group normalization', '8 groups', 'Divides channels into 8 groups, normalizes within each'],
            ['Attention', 'Linear, middle block', 'Models relationships between positions in the 24-value input'],
            ['Diffusion timesteps', '1,000', 'Number of noise levels in forward/reverse process'],
            ['Noise schedule', 'Linear, 0.0001 to 0.02', 'How much noise at each step'],
            ['Loss function', 'L2 (MSE)', 'Squared difference between predicted and actual noise'],
            ['Input shape', '(batch, 1, 24)', 'Single-channel sequence of 24 values'],
          ]}
        />
        <UNetArchitectureDiagram />
      </Subsection>

      <Subsection title="What one forward pass produces">
        <Prose>
          <p>
            The U-Net <B>only predicts noise</B> — it never directly outputs clean data. Given a noisy input
            x<sub>t</sub> and a timestep t, it estimates the noise &#949; that was added. A separate,{' '}
            <B>fixed math formula</B> (not learned) then uses that prediction to remove a tiny amount of noise,
            producing x<sub>t-1</sub> — a slightly less noisy version.
          </p>
          <p>
            This single step barely changes anything visible. To go from pure noise to clean data, you repeat
            this 1,000 times — each time the U-Net identifies the noise, and the formula removes a little more.
            Think of the U-Net as an expert eye identifying dirt, and the formula as the hand wiping it away.
          </p>
          <p>
            The loss is only computed on <B>columns 4-23</B> (the surge values) — columns 0-3 (conditioning)
            are masked out and held fixed throughout.
          </p>
        </Prose>
        <details className="my-3 group">
          <summary className="cursor-pointer text-xs font-medium text-primary hover:text-primary/80 select-none list-none flex items-center gap-1.5">
            <svg
              className="w-3.5 h-3.5 transition-transform group-open:rotate-90"
              viewBox="0 0 16 16"
              fill="currentColor"
            >
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            View the exact reverse-step formula
          </summary>
          <div className="mt-3 space-y-3 text-sm text-muted-foreground">
            <p>Each reverse step computes x<sub>t-1</sub> from x<sub>t</sub> in two stages:</p>
            <div className="space-y-2">
              <p className="text-xs font-medium text-foreground">1. Estimate clean data from the noise prediction:</p>
              <code className="block bg-muted rounded px-3 py-2 text-xs font-mono">
                x&#x302;&#x2080; = &#x221A;(1/&#x101;&#x209C;) &middot; x&#x209C; &minus; &#x221A;(1/&#x101;&#x209C; &minus; 1) &middot; &#949;
              </code>
              <p className="text-xs font-medium text-foreground">2. Compute the posterior mean (blend of estimated clean data and current noisy data):</p>
              <code className="block bg-muted rounded px-3 py-2 text-xs font-mono">
                &#956;&#x209C; = (&#946;&#x209C; &middot; &#x221A;&#x101;&#x209C;&#x208B;&#x2081;) / (1 &minus; &#x101;&#x209C;) &middot; x&#x302;&#x2080; + ((1 &minus; &#x101;&#x209C;&#x208B;&#x2081;) &middot; &#x221A;&#945;&#x209C;) / (1 &minus; &#x101;&#x209C;) &middot; x&#x209C;
              </code>
              <p className="text-xs font-medium text-foreground">3. Add a small amount of fresh noise (except at the final step):</p>
              <code className="block bg-muted rounded px-3 py-2 text-xs font-mono">
                x&#x209C;&#x208B;&#x2081; = &#956;&#x209C; + &#963;&#x209C; &middot; z, &nbsp; where z ~ N(0, I) and &#963;&#x209C; = &#x221A;(&#946;&#x209C; &middot; (1 &minus; &#x101;&#x209C;&#x208B;&#x2081;) / (1 &minus; &#x101;&#x209C;))
              </code>
            </div>
            <p className="text-xs">
              Where &#x101;&#x209C; = &#x220F;&#945;&#x2081;...&#945;&#x209C; (cumulative product of alphas),
              &#946;&#x209C; is the noise schedule at step t,
              and &#945;&#x209C; = 1 &minus; &#946;&#x209C;.
              None of these formulas are learned — they are derived from Bayes&apos; theorem and fixed before training.
            </p>
          </div>
        </details>
        <UNetOutputDiagram />
      </Subsection>

      <Subsection title="Conditioning columns">
        <DataTable
          headers={['Column', 'Content', 'Fixed during generation?']}
          rows={[
            ['0', 'theta (peak surge in meters)', 'Yes — set to desired value'],
            ['1', 'Patch centroid latitude', 'Yes — set from patch geometry'],
            ['2', 'Patch centroid longitude', 'Yes — set from patch geometry'],
            ['3', 'Patch mean bathymetric depth', 'Yes — set from patch geometry'],
            ['4-23', 'Surge at 20 nodes within the patch', 'No — generated by reverse diffusion'],
          ]}
        />
        <Prose>
          <p>
            <Mono>UNMASK_NUMBER = 4</Mono> means columns 0-3 are never corrupted with noise during training and are
            held fixed during generation. The model generates only columns 4-23.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Data normalization">
        <Prose>
          <p>
            All 24 columns are normalized to the range [-1, 1] using <B>min-max normalization</B>:
          </p>
        </Prose>
        <CodeBlock>{'x_normalized = 2 * (x - min) / (max - min) - 1'}</CodeBlock>
        <Prose>
          <p>
            The min and max values are computed per column from the training data and stored in the model checkpoint for
            use during generation. This ensures all features are on the same scale, which is important for the diffusion
            process (the noise schedule assumes data is roughly in [-1, 1]).
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Theta noise injection">
        <InfoCard title="Addressing the discrete theta problem" accent>
          <p>
            Because theta is a deterministic value per scenario (one fixed number per storm), the ~1,974 patches from a
            single scenario all share the exact same theta. The training data has only ~183 distinct theta values — far
            from the smooth, continuous distribution that Wang et al.'s temperature naturally provides.
          </p>
          <p className="mt-2">
            To address this, <B>Gaussian noise</B> is added to theta during training:
          </p>
        </InfoCard>
        <CodeBlock>{'theta_noisy = theta + epsilon,  where epsilon ~ N(0, sigma^2),  sigma = 0.15 meters'}</CodeBlock>
        <Prose>
          <p>
            This is applied independently to each training row on every epoch, so even rows from the same scenario get
            slightly different theta values. The effect is to "smear" the discrete theta distribution into a continuous
            one, creating smooth coverage across the 0.4-4.0m range.
          </p>
          <p>
            <B>Why sigma = 0.15m?</B> This is consistent with real-world surge forecast uncertainty. NOAA's operational
            storm surge forecasts typically have 1-sigma errors of 0.3-0.5m. Using 0.15m is conservative — well within
            measurement and forecast uncertainty — while being large enough to create meaningful overlap between
            adjacent scenarios. The noise is added <B>before</B> min-max normalization, so it operates in physical units
            (meters).
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Training configuration">
        <StatGrid
          items={[
            { label: 'Platform', value: 'Google Colab, A100 80GB' },
            { label: 'Batch size', value: '2,048' },
            { label: 'Learning rate', value: '4x10^-5' },
            { label: 'Training steps', value: '50,000' },
            { label: 'Total rows seen', value: '~102 million' },
            { label: 'Training time', value: '~45 minutes' },
            { label: 'Final loss', value: '~0.006 (MSE)' },
            { label: 'Optimizer', value: 'Adam' },
            { label: 'EMA decay', value: '0.995' },
          ]}
        />
        <Prose>
          <p>
            The learning rate is linearly scaled from Wang et al.'s 1x10^-5 to account for the 4x larger batch size.
            Many training rows are seen multiple times (the dataset has 361,242 rows, but 102 million rows are
            processed across all steps).
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Exponential Moving Average (EMA)">
        <Prose>
          <p>
            During training, a separate copy of the model weights is maintained as an{' '}
            <B>exponential moving average</B> of the training weights:
          </p>
        </Prose>
        <CodeBlock>{'EMA_weights = 0.995 * EMA_weights + 0.005 * training_weights'}</CodeBlock>
        <Prose>
          <p>
            This produces a smoother, more stable model that is less sensitive to recent noisy gradient updates. The EMA
            model is used for all generation and evaluation — it consistently outperforms the raw training model.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Generation process">
        <Prose>
          <p>
            To generate a new surge pattern for a given storm intensity (theta), the model runs the reverse
            diffusion loop — the same loop shown in the diagram above:
          </p>
        </Prose>
        <StepList
          steps={[
            'Start with pure random noise (24 values)',
            'Set columns 0-3 to the conditioning values: [theta, lat, lon, depth]',
            'For each timestep from t=1000 down to t=1: the U-Net predicts the noise, the formula removes a small amount, and columns 0-3 are clamped back to their fixed values',
            'After 1,000 steps, columns 4-23 contain the generated surge values',
            'Denormalize the output back to physical units (meters)',
          ]}
        />
        <InfoCard title="Why 10 samples per patch?">
          <p>
            Each generation run starts from different random noise, so it produces a slightly different result.
            By generating <B>10 samples per patch</B> and averaging them, we get a more stable prediction and can
            compute a standard deviation as an <B>uncertainty estimate</B>. Areas where the 10 samples agree have
            low uncertainty; areas where they disagree indicate the model is less confident.
          </p>
        </InfoCard>
        <Prose>
          <p>
            Generation for all 1,974 patches with 10 samples each (19,740 reverse loops of 1,000 steps) takes
            approximately 2-5 minutes on an A100 GPU.
          </p>
        </Prose>
      </Subsection>
    </Section>
  )
}

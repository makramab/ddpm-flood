import { Section, Subsection, Prose, B, Mono, NumberedCard, InfoCard, DataTable, StatGrid } from './shared'

export function LimitationsSection() {
  return (
    <Section id="limitations" title="9. The Limitations: Why the Model Falls Short">
      <div className="space-y-3">
        <NumberedCard number={1} title="Theta does not uniquely determine the flood pattern">
          <p>
            The scalar theta (peak surge at one reference gauge) is a <B>downstream point measurement</B> — it tells
            you how bad the storm was at one location, but does not encode the storm's track, size, speed, or wind field
            structure. Two storms that produce the same peak surge at the reference gauge can produce completely
            different flooding patterns elsewhere.
          </p>
          <p className="mt-2">
            <B>Evidence</B>: Hurricane Arthur (CERA, theta = 2.05m) and the closest Danso training scenario (theta =
            2.04m) have a spatial correlation of only <B>0.31</B> — barely better than random.
          </p>
          <p className="mt-2">
            This contrasts fundamentally with Wang et al.'s temperature, which is a thermodynamic state variable that
            genuinely determines the equilibrium distribution through the laws of physics.
          </p>
        </NumberedCard>

        <NumberedCard number={2} title="Synthetic-to-real generalization gap">
          <p>
            All 183 training scenarios come from <B>Danso's parametric Holland model</B> — synthetic, idealized wind
            fields. The model learns spatial patterns specific to how the Holland model distributes wind and surge.
          </p>
          <p className="mt-2">
            At theta approx. 2.0m: Danso storms produce broad, intense flooding (63% of nodes above 1m surge). Arthur (real)
            produces focused, moderate flooding (18% of nodes above 1m). The model predicts the Danso pattern because
            that is all it learned.
          </p>
        </NumberedCard>

        <NumberedCard number={3} title="Skewed theta distribution in training data">
          <DataTable
            compact
            headers={['Theta range', 'Scenarios', 'Percentage']}
            rows={[
              ['0.45 - 1.00m', '135', '82%'],
              ['1.00 - 2.00m', '20', '12%'],
              ['2.00 - 2.88m', '9', '6%'],
            ]}
          />
          <p className="mt-2">
            The model is overwhelmingly trained on low-surge events. It has very few examples of moderate-to-high surge,
            which are precisely the events that matter most for risk assessment.
          </p>
        </NumberedCard>

        <NumberedCard number={4} title="Confident extrapolation failures">
          <p>
            Adding spatial conditioning made the model <B>more confident everywhere</B>, including in regimes where it
            should be uncertain. Before spatial conditioning, the model failed gracefully (R2 approx. -0.08, near-zero
            predictions). After spatial conditioning, it fails <B>confidently</B> — it predicts high surge in the wrong
            pattern (R2 approx. -1.5), which is actively worse than predicting nothing.
          </p>
        </NumberedCard>
      </div>

      <InfoCard title="What the model IS good for" accent>
        <p>
          Despite these limitations, the model <B>works excellently as a within-ensemble surrogate</B>: for Danso
          synthetic storms at in-distribution theta values (below ~1.0m), it achieves R2 above 0.94 with RMSE of only
          6 cm. It could serve as a fast alternative to ADCIRC for exploring scenarios within the Danso parameter space
          — generating predictions in seconds instead of hours.
        </p>
      </InfoCard>
    </Section>
  )
}

export function WhatsNextSection() {
  return (
    <Section id="whats-next" title="10. What's Next: Time-Series Conditioning">
      <Subsection title="The core idea">
        <Prose>
          <p>
            Instead of using <B>peak surge</B> (one theta value per storm, from <Mono>maxele.63.nc</Mono>), use the{' '}
            <B>full surge time series</B> (one theta(t) value per timestep per storm, from <Mono>fort.63.nc</Mono>).
          </p>
          <p>
            At each timestep during a hurricane simulation, the reference gauge reads a different surge value — the
            water rises, peaks, and falls over hours. Each timestep also produces a complete spatial snapshot of surge
            across all nodes. A single storm contributes not 1 training example, but <B>200-500 training examples</B>
            (one per timestep), each with a different theta(t) and a corresponding spatial pattern.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="How this fixes the core problems">
        <div className="space-y-3">
          <InfoCard title="Fixes discrete theta">
            <p>
              With time-series data, theta varies continuously from 0 to the storm's peak and back. No gaps in the
              distribution — every value between 0 and the storm's maximum is naturally covered. No noise injection
              needed.
            </p>
          </InfoCard>
          <InfoCard title="Fixes synthetic-to-real gap">
            <p>
              The time-series data comes from <B>CERA historical storms</B> (real meteorological forcing), not Danso
              parametric storms. Both training and validation use real storms, eliminating the mismatch.
            </p>
          </InfoCard>
          <InfoCard title="Fixes skewed distribution">
            <p>
              Every storm passes through every theta level between 0 and its peak. At theta = 0.5m, all 13 storms
              contribute. At theta = 1.0m, 5 storms contribute. Naturally dense at low theta and thinner but still
              present at high theta.
            </p>
          </InfoCard>
          <InfoCard title="Fixes insufficient training data">
            <p>
              13 storms x ~300 timesteps x 1,974 patches = approximately <B>7 million training rows</B> — a 20x
              increase over the current 361,000 rows.
            </p>
          </InfoCard>
        </div>
      </Subsection>

      <Subsection title="Making the Wang et al. analogy exact">
        <Prose>
          <p>
            Wang et al.'s temperature fluctuates naturally at every molecular dynamics frame. With time-series data,
            theta(t) fluctuates naturally at every ADCIRC timestep. The structural parallel becomes exact:
          </p>
        </Prose>
        <DataTable
          headers={['Property', 'Wang et al.', 'Current (maxele)', 'Time-series (fort.63)']}
          rows={[
            ['Theta per simulation', '~thousands', '1 (peak only)', '~200-500 (per timestep)'],
            ['Theta distribution', 'Naturally continuous', 'Discrete, needs noise', 'Naturally continuous'],
            ['Training rows', '~500,000', '~361,000', '~7,000,000'],
            ['Analogy strength', '(reference)', 'Approximate', 'Structurally exact'],
          ]}
        />
      </Subsection>

      <Subsection title="What is needed">
        <Prose>
          <p>
            The time-series data (<Mono>fort.63.nc</Mono> files) for the 13 CERA HSOFS storms is available on{' '}
            <B>TACC</B> (Texas Advanced Computing Center) but has not been downloaded yet. Each file is several
            gigabytes (the full time series at 1.8 million nodes), totaling approximately 30-100 GB for all 13 storms.
          </p>
          <p>
            The preprocessing pipeline would need a new mode to iterate over timesteps within each{' '}
            <Mono>fort.63.nc</Mono> file, extracting the spatial surge snapshot and reference gauge theta(t) at each
            step. The DDPM training code itself requires <B>no changes</B> — it still receives a <Mono>.npy</Mono>{' '}
            matrix of 24-column rows. Only the data changes.
          </p>
        </Prose>
        <StatGrid
          items={[
            { label: 'Training storms', value: '13 CERA historical' },
            { label: 'Effective rows', value: '~7M (20x increase)' },
            { label: 'Theta coverage', value: '0 - 2.19m continuous' },
          ]}
        />
      </Subsection>
    </Section>
  )
}

import { Section, Subsection, Prose, B, InfoCard } from './shared'

export function ProblemSection() {
  return (
    <Section id="the-problem" title="1. The Problem: Why Predicting Coastal Flooding Is Hard">
      <Subsection title="What is storm surge?">
        <Prose>
          <p>
            When a hurricane approaches the coast, its powerful winds and low atmospheric pressure push ocean water
            toward land. This abnormal rise in sea level is called <B>storm surge</B>. It is distinct from normal ocean
            waves and tides — storm surge is a broad, sustained rise of water that can extend miles inland and persist
            for hours.
          </p>
          <p>
            Storm surge is the deadliest aspect of hurricanes, responsible for roughly 50% of all hurricane-related
            fatalities in the United States. During Hurricane Katrina (2005), surge reached over 8 meters (28 feet) in
            parts of Mississippi. During Hurricane Sandy (2012), surge of about 4 meters (13 feet) flooded lower
            Manhattan, causing over $19 billion in damage to New York City alone.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="How is storm surge currently predicted?">
        <Prose>
          <p>
            The gold standard is a physics-based computer model called <B>ADCIRC</B> (Advanced Circulation Model).
            ADCIRC solves the shallow water equations — mathematical equations that describe how water flows under the
            influence of wind, pressure, tides, and the shape of the ocean floor (bathymetry).
          </p>
          <p>
            ADCIRC divides the ocean and coastline into a mesh of triangular elements. Each triangle vertex is called a{' '}
            <B>node</B>. The mesh used in this project is called <B>HSOFS</B> (Hurricane Surge On-Demand Forecast
            System), covering the entire US Atlantic and Gulf coasts with approximately <B>1.8 million nodes</B>. The
            mesh is denser near coastlines where detail matters, and sparser in the open ocean.
          </p>
          <p>
            For a single hurricane scenario, ADCIRC computes how water levels change at every node over time, typically
            producing output every few minutes over a 24-72 hour simulation window. Running one scenario takes{' '}
            <B>hours to days</B> on a high-performance computing cluster with hundreds of CPU cores.
          </p>
        </Prose>
      </Subsection>

      <InfoCard title="Why this is a problem" accent>
        <p>
          For real-time emergency management, forecasters need surge predictions quickly — ideally within minutes of a
          new hurricane forecast. For long-term risk assessment and urban planning, analysts need to evaluate thousands
          of possible storm scenarios to understand the probability distribution of flooding outcomes. Running ADCIRC
          thousands of times is computationally prohibitive.
        </p>
        <p className="mt-2">
          This creates a need for <B>surrogate models</B> — fast approximations that can produce reasonable flood
          predictions in seconds instead of hours, trained on a library of pre-computed ADCIRC simulations. This project
          investigates whether a particular type of generative AI model can serve as such a surrogate.
        </p>
      </InfoCard>
    </Section>
  )
}

export function IdeaSection() {
  return (
    <Section id="the-idea" title="2. The Idea: Using Generative AI as a Fast Surrogate">
      <InfoCard accent>
        <p className="text-foreground font-medium">
          Can a generative model learn the relationship between a simple scalar input (how severe is this storm?) and
          the resulting spatial pattern of flooding across a coastal region — without understanding any of the
          underlying physics?
        </p>
        <p className="mt-2">
          More formally: given <B>theta (the peak water level at a reference tide gauge, in meters)</B>, can a model generate a
          realistic spatial map of surge values at thousands of coastal mesh nodes?
        </p>
      </InfoCard>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Traditional Regression">
          <p>
            Outputs a <B>single best-guess value</B> at each location. One answer, no uncertainty information.
          </p>
        </InfoCard>
        <InfoCard title="Generative Model" accent>
          <p>
            Produces <B>samples from a probability distribution</B>. Run it 100 times with the same input and it
            produces 100 different plausible flood maps — each slightly different, capturing inherent uncertainty.
          </p>
          <p className="mt-2">
            This tells us not just "what is the most likely flood pattern?" but also "how confident should we be?" and
            "what are the plausible worst-case scenarios?"
          </p>
        </InfoCard>
      </div>

      <Prose>
        <p>
          This project uses a <B>Denoising Diffusion Probabilistic Model (DDPM)</B> — a type of generative AI that has
          shown remarkable success in image generation (e.g., DALL-E, Stable Diffusion). The specific reason for
          choosing DDPMs is not that they are the best possible architecture for flood prediction, but that a research
          group (Wang et al., 2022) demonstrated that DDPMs can learn conditional distributions from physics simulations
          in a closely analogous setting (molecular dynamics). This project adapts their exact framework to a new
          domain.
        </p>
      </Prose>
    </Section>
  )
}

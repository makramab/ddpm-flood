import { Card, CardContent } from '#/components/ui/card'

export function BackgroundSection() {
  return (
    <section className="space-y-6">
      <h2 className="text-2xl font-bold text-foreground">Background & Motivation</h2>

      <div className="space-y-4 text-muted-foreground leading-relaxed">
        <p>
          Physics-based storm surge simulators like{' '}
          <span className="text-foreground font-medium">ADCIRC</span> are the gold standard for
          predicting coastal flooding from hurricanes. However, each simulation takes hours to days
          on high-performance computing clusters, making it impractical to explore thousands of
          storm scenarios needed for probabilistic risk assessment and real-time forecasting.
        </p>

        <p>
          This project investigates whether{' '}
          <span className="text-foreground font-medium">Denoising Diffusion Probabilistic Models (DDPMs)</span>{' '}
          can serve as fast surrogates for ADCIRC — generating realistic spatial flood patterns
          conditioned on a simple scalar input, rather than running a full physics simulation.
        </p>
      </div>

      <Card>
        <CardContent className="p-4 md:p-6 space-y-4">
          <h3 className="text-lg font-semibold text-foreground">The Wang et al. Framework</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            This approach adapts the DDPM framework from{' '}
            <span className="text-foreground">Wang et al. (2022)</span>, originally developed for
            molecular dynamics. Their model trains a 1D U-Net DDPM on molecular trajectories,
            conditioning on instantaneous kinetic temperature via an inpainting mechanism — the
            temperature column is never noised during diffusion, while coordinate columns are
            generated through reverse diffusion.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
            <MappingCard
              from="Kinetic temperature"
              to="Peak surge at reference gauge (θ)"
            />
            <MappingCard
              from="Molecular coordinates"
              to="Surge values at 20 mesh nodes"
            />
            <MappingCard
              from="MD trajectory frames"
              to="ADCIRC hurricane outputs"
            />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The model learns{' '}
            <span className="font-mono text-foreground text-xs">P(surge pattern | θ, spatial location)</span>{' '}
            — given a peak surge severity and a location on the coast, it generates a plausible
            spatial distribution of flooding at nearby mesh nodes.
          </p>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        Reference: Wang, D., et al. (2022). "Generative coarse-graining of molecular conformations."
        <span className="italic"> ICML 2022</span>.
      </p>
    </section>
  )
}

function MappingCard({ from, to }: { from: string; to: string }) {
  return (
    <div className="bg-muted/50 rounded-lg p-3 space-y-1">
      <p className="text-muted-foreground line-through decoration-muted-foreground/40">{from}</p>
      <p className="text-foreground font-medium">{to}</p>
    </div>
  )
}

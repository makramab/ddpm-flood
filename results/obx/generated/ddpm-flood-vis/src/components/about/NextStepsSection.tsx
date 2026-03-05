import { Card, CardContent } from '#/components/ui/card'

export function NextStepsSection() {
  return (
    <section className="space-y-6">
      <h2 className="text-2xl font-bold text-foreground">Limitations & Next Steps</h2>

      <div className="space-y-4 text-muted-foreground leading-relaxed">
        <p>
          The current model works well as a{' '}
          <span className="text-foreground font-medium">within-ensemble surrogate</span> — it
          accurately predicts spatial surge patterns for Danso synthetic storms when θ is within the
          training range (R² = 0.95). However, three key limitations were identified:
        </p>
      </div>

      <div className="space-y-3">
        <LimitationCard
          number={1}
          title="θ is insufficient as sole storm descriptor"
          description="Two storms with the same peak surge (θ ≈ 2.05m) produce fundamentally different spatial patterns — spatial correlation of only 0.31. A single scalar cannot uniquely determine the flood map."
        />
        <LimitationCard
          number={2}
          title="Synthetic-to-real generalization gap"
          description="All training data comes from Danso's parametric Holland model storms. The model learns Danso-specific spatial patterns and fails on real CERA historical storms, even when θ is within training range."
        />
        <LimitationCard
          number={3}
          title="Skewed θ distribution"
          description="82% of training scenarios have θ < 1.0m. Only 6% have θ > 2.0m. The model is poorly calibrated for the high-surge events that matter most for risk assessment."
        />
      </div>

      <Card className="border-primary/30">
        <CardContent className="p-4 md:p-6 space-y-4">
          <h3 className="text-lg font-semibold text-foreground">Next Step: Time-Series Conditioning</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Instead of using peak surge (one θ per storm), the next iteration will use the{' '}
            <span className="text-foreground font-medium">full surge time series</span> from ADCIRC{' '}
            <span className="font-mono text-xs">fort.63.nc</span> files. At each timestep, the
            reference gauge reads θ(t) and every other node has a corresponding surge value —
            producing one complete spatial snapshot per timestep.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            This makes the Wang et al. analogy{' '}
            <span className="text-foreground font-medium">structurally exact</span>: just as their
            instantaneous kinetic temperature fluctuates at every MD frame, θ(t) rises and falls
            throughout each storm, producing a naturally continuous conditioning distribution.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
            <StatCard label="Training storms" value="13 CERA historical" />
            <StatCard label="Effective rows" value="~7M (20x increase)" />
            <StatCard label="θ coverage" value="0 – 2.19m continuous" />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            By training on real historical storms rather than synthetic ones, this approach eliminates the
            cross-storm generalization gap. Each storm contributes ~200-500 timesteps with distinct
            spatial snapshots at different phases, providing rich diversity even with 13 storms.
          </p>
        </CardContent>
      </Card>
    </section>
  )
}

function LimitationCard({ number, title, description }: { number: number; title: string; description: string }) {
  return (
    <Card>
      <CardContent className="p-4 md:p-6 flex gap-4">
        <span className="text-2xl font-bold text-muted-foreground/30 shrink-0">{number}</span>
        <div className="space-y-1">
          <h4 className="font-semibold text-foreground text-sm">{title}</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
        </div>
      </CardContent>
    </Card>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-muted/50 rounded-lg p-3 space-y-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-foreground font-medium">{value}</p>
    </div>
  )
}

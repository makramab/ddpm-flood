import { createFileRoute } from '@tanstack/react-router'
import { Hero } from '#/components/landing/Hero'
import { ScenarioPreviewCard } from '#/components/landing/ScenarioPreviewCard'
import type { ScenarioSummary } from '#/lib/data/types'

const SCENARIOS: ScenarioSummary[] = [
  {
    id: 'dorian_low',
    name: 'Hurricane Dorian',
    label: 'In-distribution (success)',
    theta: 0.701,
    metrics: { r2: 0.9534, rmse: 0.0597, bias: -0.0313, mae: 0.0458, mean_uncertainty: 0.0592 },
  },
  {
    id: 'dorian_high',
    name: 'Hurricane Dorian',
    label: 'Extrapolation (failure)',
    theta: 3.357,
    metrics: { r2: -1.2636, rmse: 0.9562, bias: 0.1659, mae: 0.6892, mean_uncertainty: 0.1251 },
  },
  {
    id: 'arthur',
    name: 'Hurricane Arthur',
    label: 'Cross-storm (failure)',
    theta: 2.049,
    metrics: { r2: -2.6159, rmse: 0.6002, bias: 0.3294, mae: 0.4754, mean_uncertainty: 0.302 },
  },
]

export const Route = createFileRoute('/')({ component: LandingPage })

function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <Hero />
      <section className="px-4 md:px-6 pb-16 max-w-5xl mx-auto">
        <h2 className="text-xl font-semibold text-foreground mb-6 text-center">Three Scenarios, One Story</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
          {SCENARIOS.map((s) => (
            <ScenarioPreviewCard key={s.id} scenario={s} />
          ))}
        </div>
      </section>
    </div>
  )
}

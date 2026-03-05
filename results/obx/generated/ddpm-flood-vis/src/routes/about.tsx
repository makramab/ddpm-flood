import { createFileRoute } from '@tanstack/react-router'
import { BackgroundSection } from '#/components/about/BackgroundSection'
import { DataSection } from '#/components/about/DataSection'
import { NextStepsSection } from '#/components/about/NextStepsSection'

export const Route = createFileRoute('/about')({ component: AboutPage })

function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-3xl mx-auto px-4 md:px-6 py-10 md:py-16 space-y-12">
        <header>
          <h1 className="text-3xl md:text-4xl font-black text-foreground mb-2">About This Project</h1>
          <p className="text-muted-foreground">
            Adapting diffusion models from molecular dynamics to coastal flood prediction.
          </p>
        </header>
        <BackgroundSection />
        <DataSection />
        <NextStepsSection />
      </div>
    </div>
  )
}

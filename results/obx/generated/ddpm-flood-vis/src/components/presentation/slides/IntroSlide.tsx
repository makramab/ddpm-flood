import { Link } from '@tanstack/react-router'
import { Waves } from 'lucide-react'

export function IntroSlide() {
  return (
    <div className="max-w-3xl w-full text-center space-y-8">
      <div className="flex justify-center">
        <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
          <Waves size={40} className="text-primary" />
        </div>
      </div>

      <div className="space-y-5">
        <h1 className="text-4xl md:text-6xl font-black text-foreground leading-tight">
          DDPM for Hurricane
          <br />
          Storm Surge Prediction
        </h1>
        <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
          Adapting Denoising Diffusion Probabilistic Models from molecular dynamics
          to spatial storm surge prediction using ADCIRC simulations
        </p>
      </div>

      <div className="pt-4 space-y-2">
        <p className="text-lg text-foreground font-medium">Afra Izzati Kamili</p>
      </div>

      <div className="pt-6">
        <Link
          to="/0x-primer"
          className="text-sm text-primary hover:text-primary/80 transition-colors"
        >
          Full technical reference &rarr;
        </Link>
      </div>
    </div>
  )
}

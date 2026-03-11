import { Link } from '@tanstack/react-router'

export function ResearchQuestionSlide() {
  return (
    <div className="max-w-4xl w-full space-y-10">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Research Question</p>
        <h2 className="text-3xl md:text-5xl font-bold text-foreground leading-snug">
          Can a generative model learn spatial surge patterns from simulation data?
        </h2>
      </div>

      <div className="bg-muted/50 rounded-xl p-6 md:p-8 space-y-5">
        <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
          Given a scalar storm severity index <span className="font-mono text-foreground">&theta;</span>{' '}
          (peak surge at a reference tide gauge) and a spatial location (lat, lon, depth),
          can a DDPM generate realistic surge values at nearby mesh nodes?
        </p>
        <div className="flex justify-center pt-2">
          <code className="text-2xl md:text-3xl font-mono text-primary font-bold">
            P(surge pattern | &theta;, location)
          </code>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-muted/30 rounded-lg p-5 space-y-2">
          <p className="text-sm text-muted-foreground">Based on</p>
          <p className="text-lg text-foreground font-medium">Wang et al. 2022</p>
          <p className="text-base text-muted-foreground">
            DDPM for molecular dynamics &mdash; learns P(coordinates | temperature)
          </p>
        </div>
        <div className="bg-muted/30 rounded-lg p-5 space-y-2">
          <p className="text-sm text-muted-foreground">This adaptation</p>
          <p className="text-lg text-foreground font-medium">Temperature &rarr; &theta; + location, coordinates &rarr; surge pattern</p>
          <p className="text-base text-muted-foreground">
            Inpainting mechanism: conditioning columns never noised
          </p>
        </div>
      </div>

      <div className="flex justify-end">
        <Link to="/0x-primer" hash="the-idea" className="text-sm text-primary hover:text-primary/80 transition-colors">
          Primer: The Idea &amp; Wang et al. &rarr;
        </Link>
      </div>
    </div>
  )
}

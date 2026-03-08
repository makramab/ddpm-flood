import { Link } from '@tanstack/react-router'
import { UNetArchitectureDiagram } from '#/components/primer/UNetArchitectureDiagram'

export function ModelSlide() {
  return (
    <div className="max-w-5xl w-full space-y-6">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Methodology</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">The Model: 1D U-Net</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-8 items-start">
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Input', value: '24 values (1D sequence)' },
              { label: 'Conditioning', value: '4 cols: \u03B8, lat, lon, depth' },
              { label: 'Diffusion steps', value: '1,000' },
              { label: 'Output', value: 'Predicted noise \u03B5' },
            ].map((item) => (
              <div key={item.label} className="bg-muted/50 rounded-lg p-4 space-y-1">
                <p className="text-sm text-muted-foreground">{item.label}</p>
                <p className="text-base text-foreground font-medium">{item.value}</p>
              </div>
            ))}
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-5 space-y-2">
            <p className="text-lg font-semibold text-foreground">Key trick: inpainting</p>
            <p className="text-base text-muted-foreground leading-relaxed">
              Columns 0&ndash;3 (conditioning) are <span className="text-foreground font-medium">never noised</span> during
              training. At generation time, they are set to the desired values and held fixed. The model learns to
              generate surge (columns 4&ndash;23) consistent with the conditioning.
            </p>
          </div>

          <div className="flex justify-end">
            <Link to="/0x-primer" hash="model" className="text-sm text-primary hover:text-primary/80 transition-colors">
              Primer: Architecture &amp; training &rarr;
            </Link>
          </div>
        </div>

        <div className="w-full min-w-0 [&>div]:my-0 -mt-4">
          <UNetArchitectureDiagram />
        </div>
      </div>
    </div>
  )
}

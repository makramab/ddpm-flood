export function BroaderLandscapeSlide() {
  return (
    <div className="max-w-4xl w-full space-y-6">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Broader Landscape</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">
          Earth-System Foundation Models
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Aurora card */}
        <div className="bg-muted/30 rounded-xl p-5 space-y-3 md:col-span-2">
          <div className="flex items-baseline justify-between gap-3">
            <p className="text-lg font-semibold text-foreground">Microsoft Aurora</p>
            <span className="text-xs text-muted-foreground shrink-0">Nature, 2025</span>
          </div>
          <p className="text-base text-muted-foreground leading-relaxed">
            A 1.3B-parameter foundation model trained on over one million hours of geophysical data.
            Predicts atmospheric and oceanic state variables &mdash; wind fields, pressure,
            cyclone tracks, ocean waves &mdash; but <span className="text-foreground font-medium">not</span> downstream
            hazard outcomes like storm surge.
          </p>
        </div>

        {/* Pipeline positioning */}
        <div className="bg-primary/5 border border-primary/20 rounded-xl p-5 space-y-3 md:col-span-2">
          <p className="text-sm font-semibold text-foreground">Complementary, not competing</p>
          <div className="flex flex-col sm:flex-row items-center gap-2 text-sm font-mono">
            <span className="bg-muted/50 px-3 py-1.5 rounded text-muted-foreground">Aurora</span>
            <span className="text-muted-foreground">&rarr;</span>
            <span className="text-muted-foreground">wind, pressure, tracks</span>
            <span className="text-muted-foreground">&rarr;</span>
            <span className="bg-primary/10 px-3 py-1.5 rounded text-primary font-medium">DDPM</span>
            <span className="text-muted-foreground">&rarr;</span>
            <span className="text-muted-foreground">surge maps</span>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Aurora operates at the <span className="text-foreground">weather forecasting layer</span>.
            Our DDPM operates at the <span className="text-foreground">hazard generation layer</span>.
            A hybrid pipeline could replace the scalar &theta; with spatially-resolved
            wind and pressure fields &mdash; directly addressing our core limitation.
          </p>
        </div>
      </div>

      {/* Related work */}
      <div className="space-y-3">
        <p className="text-sm font-medium text-muted-foreground uppercase tracking-widest">Related work</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            {
              authors: 'Bodnar et al.',
              year: '2025',
              venue: 'Nature',
              title: 'Aurora: A foundation model of the atmosphere',
              note: '1.3B-parameter Earth-system foundation model for atmospheric and oceanic forecasting',
              url: 'https://www.nature.com/articles/s41586-025-09005-y',
            },
            {
              authors: 'Weng & Gori',
              year: '2026',
              venue: 'JGR Atmospheres',
              title: 'Climatological benchmarking of AI-generated tropical cyclones',
              note: 'Compares Aurora vs Pangu-Weather on cyclone intensity, tracks, and size across NA and WPac basins',
              url: 'https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JD044753',
            },
          ].map((ref) => (
            <a
              key={ref.title}
              href={ref.url}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-muted/20 rounded-lg p-4 space-y-1.5 block hover:bg-muted/40 transition-colors"
            >
              <p className="text-sm font-medium text-foreground leading-snug">{ref.title}</p>
              <p className="text-xs text-muted-foreground">
                {ref.authors} ({ref.year}) &middot; {ref.venue}
              </p>
              <p className="text-xs text-muted-foreground/80 leading-relaxed">{ref.note}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground leading-relaxed">
          <span className="text-foreground font-medium">Key insight:</span>{' '}
          Few studies currently adapt foundation models for coastal flood map generation.
          Most work focuses on weather forecasting and cyclone dynamics.
          The hazard-generation layer &mdash; where our DDPM operates &mdash; remains an open research gap.
        </p>
      </div>
    </div>
  )
}

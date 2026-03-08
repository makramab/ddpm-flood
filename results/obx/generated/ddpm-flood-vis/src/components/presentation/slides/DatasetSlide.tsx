import { Link } from '@tanstack/react-router'

// Real data from Patch 1030, Hurricane Arthur — same values used in primer diffusion diagrams
const SAMPLE_ROW = {
  conditioning: [
    { label: '\u03B8', value: '2.049 m' },
    { label: 'lat', value: '35.740\u00B0' },
    { label: 'lon', value: '\u221275.624\u00B0' },
    { label: 'depth', value: '3.61 m' },
  ],
  surge: [
    0.620, 0.603, 0.571, 0.649, 0.965,
    0.235, 0.538, 0.569, 0.590, 0.605,
    0.616, 0.617, 0.616, 0.615, 0.614,
    0.620, 0.626, 0.607, 0.614, 0.616,
  ],
}

export function DatasetSlide() {
  return (
    <div className="max-w-5xl w-full space-y-6">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Dataset</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">Training Data: ADCIRC Simulations</h2>
        <p className="text-base text-muted-foreground leading-relaxed">
          <span className="text-foreground font-medium">ADCIRC</span> (ADvanced CIRCulation) is a physics-based
          hydrodynamic model that simulates storm surge propagation across coastal ocean meshes.
          It is considered the gold standard for hurricane surge prediction but requires hours of compute per storm.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { value: '183', label: 'Scenarios' },
          { value: '1,974', label: 'Spatial patches' },
          { value: '361K', label: 'Training rows' },
          { value: '24', label: 'Columns per row' },
        ].map((stat) => (
          <div key={stat.label} className="bg-muted/50 rounded-xl p-5 text-center space-y-1">
            <p className="text-3xl md:text-4xl font-bold text-foreground">{stat.value}</p>
            <p className="text-sm text-muted-foreground">{stat.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground">Data sources</h3>
          <div className="space-y-3">
            <div className="bg-muted/30 rounded-lg p-4 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-base font-medium text-foreground">Danso et al. 2025</p>
                <span className="text-xs font-mono text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">170 scenarios</span>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                20 real Atlantic hurricanes simulated with ADCIRC. Wind fields generated using the
                parametric Holland model, then run through full hydrodynamic simulation. Each storm
                has 9 variants (3 sea levels &times; 3 forcing types).
              </p>
              <p className="text-xs text-muted-foreground">
                Source: <a href="https://zenodo.org/records/17601768" target="_blank" rel="noopener noreferrer" className="text-primary hover:text-primary/80 underline underline-offset-2">Zenodo 17601768</a> &middot; HSOFS mesh (1.8M nodes) &middot; CC-BY-4.0
              </p>
            </div>
            <div className="bg-muted/30 rounded-lg p-4 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-base font-medium text-foreground">CERA Historical Hindcasts</p>
                <span className="text-xs font-mono text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">13 scenarios</span>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                31 real historical storms with actual meteorological forcing (not parametric) run through
                ADCIRC. 13 of 31 use the HSOFS mesh. Includes high-surge events: Sandy (2012), Irene (2011), Arthur (2014).
              </p>
              <p className="text-xs text-muted-foreground">
                Source: <a href="https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3932" target="_blank" rel="noopener noreferrer" className="text-primary hover:text-primary/80 underline underline-offset-2">DesignSafe PRJ-3932</a> &middot; Downloaded from TACC via SSH
              </p>
            </div>
            <div className="bg-muted/20 border border-border rounded-lg p-3">
              <p className="text-sm text-muted-foreground leading-relaxed">
                <span className="text-foreground font-medium">Why 13 of 31 CERA storms?</span>{' '}
                CERA uses 4 different ADCIRC meshes. Only the 13 HSOFS-mesh storms share the same
                node grid as Danso&rsquo;s data, making them directly compatible for combined training.
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground">Sample training row</h3>
          <p className="text-sm text-muted-foreground">
            Real data from <span className="text-foreground font-medium">Patch 1030, Hurricane Arthur</span> &mdash; the same row used in the primer diffusion diagrams.
          </p>

          {/* Conditioning columns */}
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-primary uppercase tracking-wide">Conditioning (cols 0&ndash;3) &mdash; never noised</p>
            <div className="grid grid-cols-4 gap-2">
              {SAMPLE_ROW.conditioning.map((item) => (
                <div key={item.label} className="bg-primary/10 rounded-lg px-3 py-2 text-center">
                  <p className="text-xs text-muted-foreground">{item.label}</p>
                  <p className="text-sm font-mono font-medium text-foreground">{item.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Surge columns */}
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-foreground uppercase tracking-wide">Target surge (cols 4&ndash;23) &mdash; generated by DDPM</p>
            <div className="grid grid-cols-5 gap-1">
              {SAMPLE_ROW.surge.map((val, i) => (
                <div key={i} className="bg-muted/40 rounded px-1.5 py-1 text-center">
                  <p className="text-xs font-mono text-muted-foreground">{val.toFixed(3)}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              Surge range: 0.235&ndash;0.965 m across 20 nodes in this patch
            </p>
          </div>

          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
            <p className="text-sm text-amber-400 font-medium">&theta; distribution is skewed</p>
            <p className="text-sm text-muted-foreground mt-1">
              82% of scenarios have &theta; &lt; 1.0m. Only 6% have &theta; &gt; 2.0m.
            </p>
          </div>
        </div>
      </div>

      {/* Stitching pipeline */}
      <div className="bg-muted/30 rounded-xl p-5 space-y-3">
        <p className="text-base font-semibold text-foreground">From patches to map: the stitching pipeline</p>
        <div className="flex items-center gap-2 flex-wrap text-sm">
          {[
            { label: '34,854 nodes', sub: 'ADCIRC mesh (OBX region)' },
            { label: '1,974 patches', sub: 'KMeans clusters of ~20 nearby nodes' },
            { label: '361K rows', sub: '183 scenarios \u00D7 1,974 patches' },
            { label: 'DDPM generates', sub: '20 surge values per patch' },
            { label: 'Stitch back', sub: 'Each value \u2192 its original node lat/lon' },
            { label: 'Map display', sub: '34,854 colored dots' },
          ].map((step, i) => (
            <div key={step.label} className="flex items-center gap-2">
              {i > 0 && <span className="text-muted-foreground/40 text-lg shrink-0">&rarr;</span>}
              <div className="bg-background/60 rounded-lg px-3 py-2 border border-border/50">
                <p className="text-sm font-medium text-foreground whitespace-nowrap">{step.label}</p>
                <p className="text-xs text-muted-foreground">{step.sub}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Each column position (0&ndash;19) in a patch consistently maps to the same physical ADCIRC node.
          After generation, surge values are assigned back to their original node coordinates &mdash;
          so every dot on the map is one specific surge value from one specific column in one specific patch.
        </p>
      </div>

      <div className="flex justify-end">
        <Link to="/0x-primer" hash="data" className="text-sm text-primary hover:text-primary/80 transition-colors">
          Primer: Data &amp; preprocessing pipeline &rarr;
        </Link>
      </div>
    </div>
  )
}

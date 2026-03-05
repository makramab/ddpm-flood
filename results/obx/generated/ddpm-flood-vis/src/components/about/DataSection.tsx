import { Card, CardContent } from '#/components/ui/card'

export function DataSection() {
  return (
    <section className="space-y-6">
      <h2 className="text-2xl font-bold text-foreground">Data & Scope</h2>

      <div className="space-y-4 text-muted-foreground leading-relaxed">
        <Card className="border-primary/30">
          <CardContent className="p-4 md:p-6 space-y-2">
            <h3 className="text-lg font-semibold text-foreground">Original Objective</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              The original goal was to train a DDPM for predicting flood maps in{' '}
              <span className="text-foreground font-medium">New York City</span> — a high-impact
              target for coastal flood risk. However, available high-resolution ADCIRC simulation
              datasets have insufficient spatial coverage in the NYC metro area. The NACCS database
              has only 3 save points near NYC with accessible per-storm data, and the DeepSurge
              dataset has only 3-7 nodes in the region.
            </p>
          </CardContent>
        </Card>

        <p>
          Instead, the{' '}
          <span className="text-foreground font-medium">Outer Banks, North Carolina</span> was chosen as a
          proof-of-method region, where the Danso et al. (2025) ADCIRC dataset provides excellent
          mesh coverage (~39,500 valid nodes in the OBX bounding box).
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4 md:p-6 space-y-3">
            <h3 className="font-semibold text-foreground">Training Data</h3>
            <a
              href="https://zenodo.org/records/17601768"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-primary hover:underline"
            >
              Danso et al. 2025 (Zenodo)
            </a>
            <ul className="text-sm text-muted-foreground space-y-1.5">
              <li>20 real hurricanes x 9 variants = <span className="text-foreground font-medium">180 scenarios</span></li>
              <li>ADCIRC HSOFS mesh (1.8M nodes)</li>
              <li>3 sea-level rise levels x 3 forcing types</li>
              <li>Holland parametric model (synthetic forcing)</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 md:p-6 space-y-3">
            <h3 className="font-semibold text-foreground">Validation Data</h3>
            <p className="text-xs font-mono text-muted-foreground">CERA Historical Storms</p>
            <ul className="text-sm text-muted-foreground space-y-1.5">
              <li><span className="text-foreground font-medium">13 real historical storms</span> (2004-2022)</li>
              <li>Same HSOFS mesh as training data</li>
              <li>Full physics forcing (not parametric)</li>
              <li>Tests cross-storm generalization</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardContent className="p-4 md:p-6 space-y-3">
          <h3 className="font-semibold text-foreground">Model Architecture</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Each training row has 24 columns:{' '}
            <span className="font-mono text-xs text-foreground">[θ, lat, lon, depth, surge₁, ..., surge₂₀]</span>.
            The first 4 columns (scalar conditioning + spatial location) are never noised during
            diffusion — the model generates only the 20 surge values through reverse diffusion. Nodes
            are clustered into <span className="text-foreground font-medium">1,974 patches</span> of
            20 nodes each, covering the Outer Banks region.
          </p>
        </CardContent>
      </Card>
    </section>
  )
}

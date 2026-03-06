import { Card, CardContent } from '#/components/ui/card'
import { Section, Subsection, Prose, B, Mono, DataTable, InfoCard, StepList } from './shared'

export function DataSection() {
  return (
    <Section id="data" title="6. The Data: What Goes Into the Model">
      <Subsection title="Why the Outer Banks instead of New York City?">
        <InfoCard title="Original objective" accent>
          <p>
            The original goal was to train a DDPM for predicting flood maps in <B>New York City</B> — a high-impact
            target for coastal flood risk. However, available ADCIRC datasets have insufficient spatial coverage in NYC:
          </p>
          <ul className="mt-2 space-y-1 list-disc list-inside">
            <li>The <B>NACCS</B> database has only 3 usable save points near NYC</li>
            <li>The <B>DeepSurge</B> dataset has only 3-7 nodes in the NYC area</li>
            <li>The 20 training hurricanes primarily impacted the Gulf Coast and Carolinas — at NYC, they all produce similar low surge (0.7-1.2m)</li>
          </ul>
        </InfoCard>
        <Prose>
          <p>
            The <B>Outer Banks of North Carolina</B> was chosen as a proof-of-method region because several hurricanes
            made direct or near-direct landfall there (Dorian, Isabel, Arthur, Floyd, Florence), the theta distribution
            spans 0.4m to 4.0m (a wide range), and the ADCIRC mesh has ~39,500 valid nodes — providing dense spatial
            coverage.
          </p>
        </Prose>
      </Subsection>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4 md:p-6 space-y-3">
            <h3 className="font-semibold text-foreground">Training Data: Danso et al. 2025</h3>
            <a
              href="https://zenodo.org/records/17601768"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-primary hover:underline"
            >
              Zenodo Dataset
            </a>
            <ul className="text-sm text-muted-foreground space-y-1.5">
              <li>20 real hurricanes x 9 variants = <B>180 scenarios</B></li>
              <li>ADCIRC HSOFS mesh (1.8M nodes)</li>
              <li>3 sea-level rise levels x 3 forcing types</li>
              <li><B>Holland parametric model</B> (synthetic wind forcing)</li>
            </ul>
            <p className="text-xs text-muted-foreground leading-relaxed mt-2">
              Each scenario stored as <Mono>maxele.63.nc</Mono> — a NetCDF file containing the peak surge at every node
              over the entire simulation. One value per node, not a time series.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 md:p-6 space-y-3">
            <h3 className="font-semibold text-foreground">Validation Data: CERA Historical</h3>
            <p className="text-xs font-mono text-muted-foreground">Coastal Emergency Risks Assessment</p>
            <ul className="text-sm text-muted-foreground space-y-1.5">
              <li><B>13 real historical storms</B> (2004-2022)</li>
              <li>Same HSOFS mesh as training data</li>
              <li><B>Full physics forcing</B> (not parametric)</li>
              <li>Tests cross-storm generalization</li>
            </ul>
            <p className="text-xs text-muted-foreground leading-relaxed mt-2">
              CERA runs real-time ADCIRC simulations during active hurricanes using actual meteorological data.
              Fundamentally different from Danso's parametric storms.
            </p>
          </CardContent>
        </Card>
      </div>

      <InfoCard title="Holland parametric model — why it matters">
        <p>
          The Danso dataset uses the <B>Holland parametric wind model</B> to generate hurricane wind fields. This is a
          mathematical formula that approximates winds from a few parameters (central pressure, radius of maximum winds,
          etc.). It produces <B>idealized, symmetric</B> wind patterns that differ from real hurricane wind fields, which
          are asymmetric and contain complex features like rain bands and eyewall mesovortices. This distinction becomes
          important when discussing generalization failures.
        </p>
      </InfoCard>

      <Subsection title="CERA storm catalog">
        <DataTable
          compact
          headers={['Storm', 'Year', 'Peak theta (m)', 'Category']}
          rows={[
            ['Frances', '2004', '0.70', 'Low'],
            ['Omar', '2008', '0.75', 'Low-Med'],
            ['Earl', '2010', '1.18', 'Medium'],
            ['Irene', '2011', '2.19', 'High'],
            ['Debby', '2012', '0.55', 'Low'],
            ['Sandy', '2012', '1.62', 'High'],
            ['Andrea', '2013', '0.80', 'Low-Med'],
            ['Arthur', '2014', '2.05', 'High'],
            ['Maria', '2017', '0.68', 'Low'],
            ['Alberto', '2018', '0.69', 'Low'],
            ['Michael', '2018', '1.23', 'Medium'],
            ['Fiona', '2022', '0.56', 'Low'],
            ['Nicole', '2022', '0.86', 'Medium'],
          ]}
        />
        <Prose>
          <p>
            <B>Mesh compatibility</B>: CERA archives contain storms run on different meshes. Only storms with exactly
            1,813,443 nodes (the HSOFS mesh) are compatible with the Danso training data. 13 of 31 available CERA files
            meet this criterion.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Training/validation split">
        <Prose>
          <p>For the Outer Banks experiment:</p>
          <ul className="space-y-1 list-disc list-inside">
            <li><B>Training</B>: 171 Danso scenarios (all 180 minus 9 Dorian variants) + 12 CERA storms (all 13 minus Arthur) = <B>183 scenarios</B></li>
            <li><B>Validation</B>: 9 Dorian variants (testing in-distribution prediction) + 1 Arthur scenario (testing cross-storm generalization)</li>
          </ul>
          <p>
            After spatial patching (183 scenarios x 1,974 patches), the training set contains <B>361,242 rows</B>, each
            a 24-dimensional vector.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="Preprocessing pipeline">
        <StepList
          steps={[
            'Scenario discovery: Find all maxele.63.nc files in the Danso and CERA directories. Handle two different directory layouts (Type A and Type B storms).',
            'Mesh extraction: Read node coordinates (latitude, longitude) and bathymetric depth for the entire 1.8M-node HSOFS mesh. Identify nodes within the Outer Banks bounding box (lat 34.50-36.00, lon -76.50 to -75.30).',
            'Valid node filtering: Keep nodes with valid surge data in at least 90% of all scenarios. ADCIRC uses special fill values (very large numbers like 1x10^10 or -99999) for nodes that were never wet. Result: 39,494 of 58,645 regional nodes retained.',
            'Theta computation: For each scenario, theta = maximum surge across all nodes in the Cape Hatteras reference area.',
            'KMeans clustering: Group 39,494 valid nodes into 1,974 patches of 20 nodes each, based on geographic proximity. Deterministic and identical for all scenarios.',
            'Matrix construction: For each (scenario, patch) pair, construct a 24-dimensional row: [theta, centroid_lat, centroid_lon, mean_depth, surge_1, ..., surge_20]. Stack all 361,242 rows into a single NumPy array saved as train_data.npy.',
          ]}
        />
      </Subsection>

      <Subsection title="ADCIRC file format">
        <DataTable
          headers={['Variable', 'Description', 'Shape']}
          rows={[
            ['x', 'Longitude of each node', '(1,813,443,)'],
            ['y', 'Latitude of each node', '(1,813,443,)'],
            ['depth', 'Bathymetric depth (positive = below water, negative = above sea level)', '(1,813,443,)'],
            ['zeta_max', 'Maximum water surface elevation (peak surge)', '(1,813,443,)'],
            ['time_of_zeta_max', 'Time (in seconds) when peak surge occurred', '(1,813,443,)'],
          ]}
        />
        <Prose>
          <p>
            <B>Technical note</B>: ADCIRC NetCDF files must be read with the <Mono>netCDF4</Mono> Python library, not
            xarray. The files contain a dimension called <Mono>neta</Mono> that causes xarray's dimension-inference to
            fail.
          </p>
        </Prose>
      </Subsection>
    </Section>
  )
}

import { Section, Subsection, Prose, B, Mono, DataTable, InfoCard, CodeBlock } from './shared'

export function AdaptationSection() {
  return (
    <Section id="adaptation" title="5. The Adaptation: From Molecules to Floods">
      <Subsection title="The mapping">
        <DataTable
          headers={['Concept', 'Wang et al. (Molecular Dynamics)', 'This Project (Flood Prediction)']}
          rows={[
            ['Conditioning variable (never noised)', 'Kinetic temperature (1 scalar)', 'theta + latitude + longitude + depth (4 values)'],
            ['Target variables (generated)', '18 interatomic distances', '20 surge values at nearby mesh nodes'],
            ['One training row', 'One MD simulation frame', 'One (scenario, patch) pair'],
            ['Full training set', '~500,000 MD frames', '~361,000 (scenario, patch) pairs'],
            ['UNMASK_NUMBER', '1', '4'],
          ]}
        />
      </Subsection>

      <Subsection title="What is theta?">
        <InfoCard accent>
          <p>
            <B>Theta</B> is the <B>peak water surface elevation</B> at a reference tide gauge location, measured in
            meters above mean sea level. For the Outer Banks region, the reference area is near{' '}
            <B>Cape Hatteras</B> (latitude 35.0-35.4 degrees, longitude -76.1 to -75.5 degrees), where storm surge funnels into
            Pamlico Sound through the barrier islands.
          </p>
          <p className="mt-2">
            Theta is computed for each storm scenario by taking the maximum surge value across all ADCIRC mesh nodes
            within the reference area. It serves as a <B>scalar severity index</B> — a single number that summarizes
            "how bad is this storm for this region."
          </p>
          <p className="mt-2">
            At generation time, a user specifies a desired theta value (e.g., "what does the flood map look like if the
            reference gauge reads 2.0 meters?"), and the model generates a plausible spatial surge pattern consistent
            with that severity.
          </p>
        </InfoCard>
      </Subsection>

      <Subsection title="What is a spatial patch?">
        <Prose>
          <p>
            The Outer Banks region of the ADCIRC mesh contains approximately 39,500 nodes with valid surge data.
            Processing all nodes simultaneously would create an impractically large input for the 1D U-Net (which
            expects short sequences). Instead, the nodes are grouped into <B>patches</B> of 20 nearby nodes using{' '}
            <B>KMeans clustering</B> (a standard algorithm that groups data points by geographic proximity).
          </p>
          <p>
            This produces <B>1,974 patches</B>, each covering a small geographic area. The DDPM processes one patch at
            a time — it generates 20 surge values for the 20 nodes in that patch, conditioned on theta and the patch's
            location.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="The spatial conditioning columns">
        <Prose>
          <p>
            The first iteration of the model used only theta as conditioning (1 column). This produced predictions that
            captured the correct average surge level but were <B>spatially flat</B> — every patch received the same
            statistical output regardless of its location, because the model had no way to know <em>where</em> it was
            generating.
          </p>
          <p>Three spatial conditioning columns were added:</p>
        </Prose>
        <DataTable
          headers={['Column', 'Content', 'Source', 'Why it matters']}
          rows={[
            ['0', 'theta (peak surge)', 'Computed per scenario', 'Storm severity'],
            ['1', 'Patch centroid latitude', 'Mean of 20 node latitudes', 'North-south position (southern patches tend to get more surge)'],
            ['2', 'Patch centroid longitude', 'Mean of 20 node longitudes', 'East-west position (ocean-facing vs sound-side)'],
            ['3', 'Patch mean bathymetric depth', 'Mean of 20 node depths', 'Shallow areas amplify surge (shoaling); negative = land above sea level'],
          ]}
        />
        <Prose>
          <p>
            With these 4 conditioning columns (<Mono>UNMASK_NUMBER = 4</Mono>), each patch receives a{' '}
            <B>unique conditioning vector</B>. The model now learns:
          </p>
          <p className="text-foreground font-medium text-center">
            P(surge_1, ..., surge_20 | theta, latitude, longitude, depth)
          </p>
          <p>
            This means it can produce location-specific predictions — coastal patches get high surge, deep-water patches
            get low surge, and the pattern changes with theta.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="The training row format">
        <CodeBlock>{`[theta, lat, lon, depth, surge_1, surge_2, ..., surge_20]
 |__________________|  |___________________________|
 Conditioning (fixed)    Target (generated by DDPM)
 Columns 0-3             Columns 4-23`}</CodeBlock>
        <Prose>
          <p>
            During training, columns 0-3 are never corrupted with noise. The model learns to denoise columns 4-23
            given the fixed values in columns 0-3.
          </p>
        </Prose>
      </Subsection>
    </Section>
  )
}

import { Section, Subsection, Prose, B, DataTable, InfoCard } from './shared'

export function WangSection() {
  return (
    <Section id="wang-et-al" title="4. The Reference Paper: Wang et al. 2022">
      <Subsection title="What they did">
        <Prose>
          <p>
            Wang, D., et al. (2022) published "Generative coarse-graining of molecular conformations" at ICML 2022.
            They trained a 1D U-Net DDPM on <B>molecular dynamics (MD) trajectories</B> — sequences of snapshots
            showing how atoms in a molecule move over time.
          </p>
          <p>
            In molecular dynamics, a simulation computes the position of every atom at every timestep (typically
            millions of timesteps). At each timestep, the molecule has a measurable property called{' '}
            <B>instantaneous kinetic temperature</B> — a scalar that reflects how fast the atoms are vibrating.
            Temperature naturally fluctuates from frame to frame, creating a smooth, continuous distribution of
            conditioning values.
          </p>
          <p>
            Their model learns: <B>P(molecular coordinates | temperature)</B>. Given a temperature, generate a
            plausible arrangement of atoms. This is useful for exploring molecular configurations at different energy
            states without running expensive simulations.
          </p>
        </Prose>
      </Subsection>

      <Subsection title="The structural parallel">
        <DataTable
          headers={['Concept', 'Wang et al.', 'This Project']}
          rows={[
            ['System', 'Molecule', 'Coastal region'],
            ['Simulation', 'Molecular dynamics', 'ADCIRC hurricane simulation'],
            ['Data per snapshot', 'Positions of atoms', 'Surge values at mesh nodes'],
            ['Conditioning variable', 'Instantaneous kinetic temperature', 'Peak surge at reference gauge (theta)'],
            ['Distribution learned', 'P(atom positions | temperature)', 'P(surge pattern | theta, location)'],
            ['Conditioning mechanism', 'Inpainting (column 0 never noised)', 'Inpainting (columns 0-3 never noised)'],
          ]}
        />
        <Prose>
          <p>
            Critically, their DDPM code is used <B>unmodified</B> — the same 1D U-Net, the same GaussianDiffusion
            class, the same inpainting mechanism. Only the data changes.
          </p>
        </Prose>
      </Subsection>

      <InfoCard title="An important difference" accent>
        <p>
          Wang et al.'s temperature is a <B>thermodynamic state variable</B> — it genuinely determines the equilibrium
          distribution of molecular configurations through the laws of statistical mechanics. There is a rigorous
          physical guarantee that knowing the temperature tells you everything about the expected distribution of
          molecular states.
        </p>
        <p className="mt-2">
          In this project, theta (peak surge at one gauge) is a <B>downstream point measurement</B>, not a state
          variable. It does not uniquely determine the spatial flood pattern — two storms with the same peak surge at
          the reference gauge can produce very different flooding elsewhere, depending on the storm's track, size,
          speed, and wind field structure. This fundamental difference is the source of several limitations discussed
          later.
        </p>
      </InfoCard>
    </Section>
  )
}

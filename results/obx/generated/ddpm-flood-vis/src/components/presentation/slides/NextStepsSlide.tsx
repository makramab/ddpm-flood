import { Link } from '@tanstack/react-router'

export function NextStepsSlide() {
  return (
    <div className="max-w-4xl w-full space-y-8">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Next Steps</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">
          Time-Series Conditioning
        </h2>
      </div>

      <div className="bg-primary/5 border border-primary/20 rounded-xl p-6 md:p-8 space-y-3">
        <p className="text-lg font-semibold text-foreground">The core idea</p>
        <p className="text-base text-muted-foreground leading-relaxed">
          Replace peak surge (<span className="font-mono text-foreground">maxele.63.nc</span> &mdash; one &theta; per storm)
          with the full surge time series (<span className="font-mono text-foreground">fort.63.nc</span> &mdash;
          one &theta;(t) per timestep). Each storm contributes 200&ndash;500 training examples instead of 1.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {[
          {
            problem: 'Discrete \u03B8',
            fix: '\u03B8(t) varies continuously from 0 to peak and back. No noise injection needed.',
          },
          {
            problem: 'Synthetic-to-real gap',
            fix: 'Train on 13 CERA historical storms. Both training and validation use real data.',
          },
          {
            problem: 'Skewed distribution',
            fix: 'Every storm passes through every \u03B8 level between 0 and its peak.',
          },
          {
            problem: 'Insufficient data',
            fix: '13 storms \u00D7 ~300 timesteps \u00D7 1,974 patches = ~7M rows (20\u00D7 increase).',
          },
        ].map((item) => (
          <div key={item.problem} className="bg-muted/30 rounded-lg p-5 space-y-2">
            <p className="text-sm text-red-400 font-medium line-through">{item.problem}</p>
            <p className="text-base text-muted-foreground leading-relaxed">{item.fix}</p>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-muted/50">
              <th className="px-4 py-3 text-left font-medium text-foreground">Property</th>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground">Current (maxele)</th>
              <th className="px-4 py-3 text-left font-medium text-primary">Time-series (fort.63)</th>
            </tr>
          </thead>
          <tbody className="text-muted-foreground">
            <tr className="border-t border-border">
              <td className="px-4 py-3">&theta; per simulation</td>
              <td className="px-4 py-3">1 (peak only)</td>
              <td className="px-4 py-3 text-foreground font-medium">~200&ndash;500</td>
            </tr>
            <tr className="border-t border-border">
              <td className="px-4 py-3">&theta; distribution</td>
              <td className="px-4 py-3">Discrete, needs noise</td>
              <td className="px-4 py-3 text-foreground font-medium">Naturally continuous</td>
            </tr>
            <tr className="border-t border-border">
              <td className="px-4 py-3">Training rows</td>
              <td className="px-4 py-3">~361K</td>
              <td className="px-4 py-3 text-foreground font-medium">~7M</td>
            </tr>
            <tr className="border-t border-border">
              <td className="px-4 py-3">Wang et al. analogy</td>
              <td className="px-4 py-3">Approximate</td>
              <td className="px-4 py-3 text-primary font-medium">Structurally exact</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="flex justify-end">
        <Link to="/0x-primer" hash="whats-next" className="text-sm text-primary hover:text-primary/80 transition-colors">
          Primer: Time-series approach in detail &rarr;
        </Link>
      </div>
    </div>
  )
}

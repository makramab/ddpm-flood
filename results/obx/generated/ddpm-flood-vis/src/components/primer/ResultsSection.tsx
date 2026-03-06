import { Card, CardContent } from '#/components/ui/card'
import { Section, Subsection, Prose, B, DataTable, StatGrid } from './shared'

export function ResultsSection() {
  return (
    <Section id="results" title="8. The Results: What Worked and What Didn't">
      <Subsection title="Evaluation metrics">
        <DataTable
          headers={['Metric', 'Formula', 'Interpretation']}
          rows={[
            ['R2', '1 - sum(pred-truth)^2 / sum(truth-mean)^2', '1.0 = perfect. 0.0 = predicting the mean everywhere. Negative = worse than the mean.'],
            ['RMSE', 'sqrt(mean((pred-truth)^2))', 'Average prediction error in meters. Lower is better.'],
            ['MAE', 'mean(|pred-truth|)', 'Average absolute error in meters. Less sensitive to outliers than RMSE.'],
            ['Bias', 'mean(pred-truth)', 'Systematic over- (+) or under-prediction (-). Zero = unbiased.'],
          ]}
        />
      </Subsection>

      <Subsection title="Three scenarios tell one story">
        <Prose>
          <p>
            The results are best understood through three carefully chosen validation scenarios that demonstrate the
            model's capabilities <B>and</B> limitations:
          </p>
        </Prose>

        <div className="space-y-4">
          <ScenarioCard
            title="Dorian, low theta (0.701m)"
            subtitle="In-distribution success"
            variant="success"
            metrics={[
              { label: 'R2', value: '0.95' },
              { label: 'RMSE', value: '0.060 m' },
              { label: 'Bias', value: '-0.031 m' },
              { label: 'Uncertainty', value: '0.059 m' },
            ]}
          >
            <p>
              This is an <B>in-distribution</B> test: Dorian is from the Danso training ensemble (though the specific
              variants were held out), and theta = 0.70m is well within the training range (82% of training data is
              below 1.0m). The model accurately reproduces the spatial surge pattern with errors of only 6 cm.
            </p>
          </ScenarioCard>

          <ScenarioCard
            title="Dorian, high theta (3.357m)"
            subtitle="Extrapolation failure"
            variant="warning"
            metrics={[
              { label: 'R2', value: '-1.26' },
              { label: 'RMSE', value: '0.956 m' },
              { label: 'Bias', value: '+0.166 m' },
            ]}
          >
            <p>
              This tests <B>extrapolation</B>: theta = 3.36m is above the training maximum of 2.88m. Even though this
              is the same storm family (Danso synthetic), the model has never seen surge this severe. It confidently
              predicts the wrong spatial pattern, producing high surge everywhere in the characteristic Danso pattern.
            </p>
          </ScenarioCard>

          <ScenarioCard
            title="Arthur, theta = 2.049m"
            subtitle="Cross-storm generalization failure"
            variant="danger"
            metrics={[
              { label: 'R2', value: '-2.62' },
              { label: 'RMSE', value: '0.600 m' },
              { label: 'Bias', value: '+0.329 m' },
            ]}
          >
            <p>
              The most scientifically important result. Theta = 2.05m <B>is within the training range</B>, but Arthur is
              a <B>CERA historical storm</B> — a real hurricane simulated with real meteorological data. Despite having
              the "right" theta, the model fails because:
            </p>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>The training data is 100% Danso parametric storms</li>
              <li>At theta approx. 2.0m, Danso storms produce 63% of nodes above 1m surge — a broad, intense pattern</li>
              <li>Arthur produces only 18% of nodes above 1m — a more focused, realistic pattern</li>
              <li>The spatial correlation between Arthur and the closest Danso scenario is only <B>0.31</B></li>
            </ul>
          </ScenarioCard>
        </div>
      </Subsection>

      <Subsection title="Complete validation results">
        <DataTable
          compact
          headers={['Scenario', 'theta (m)', 'R2', 'RMSE (m)', 'Category']}
          rows={[
            ['Dorian', '0.701', '0.950', '0.062', 'In-distribution (success)'],
            ['Dorian', '0.702', '0.947', '0.063', 'In-distribution (success)'],
            ['Dorian', '0.766', '0.943', '0.066', 'In-distribution (success)'],
            ['Dorian', '2.781', '-1.263', '0.821', 'Near training max (failure)'],
            ['Dorian', '2.899', '-1.390', '0.853', 'Extrapolation (failure)'],
            ['Dorian', '2.984', '-1.465', '0.876', 'Extrapolation (failure)'],
            ['Dorian', '3.097', '-1.372', '0.882', 'Extrapolation (failure)'],
            ['Dorian', '3.237', '-1.280', '0.929', 'Extrapolation (failure)'],
            ['Dorian', '3.357', '-1.530', '0.960', 'Extrapolation (failure)'],
            ['Arthur', '2.049', '-2.568', '0.602', 'Cross-storm (failure)'],
          ]}
        />
        <Prose>
          <p>
            The sharp R2 drop between theta = 0.77m and theta = 2.78m reflects both the training data gap (82% of data
            is below 1.0m, only 6% above 2.0m) and the fundamental theta-insufficiency problem.
          </p>
        </Prose>
      </Subsection>
    </Section>
  )
}

function ScenarioCard({
  title,
  subtitle,
  variant,
  metrics,
  children,
}: {
  title: string
  subtitle: string
  variant: 'success' | 'warning' | 'danger'
  metrics: { label: string; value: string }[]
  children: React.ReactNode
}) {
  const borderColor =
    variant === 'success' ? 'border-green-500/30' : variant === 'warning' ? 'border-amber-500/30' : 'border-red-500/30'
  const badgeColor =
    variant === 'success'
      ? 'bg-green-500/10 text-green-400'
      : variant === 'warning'
        ? 'bg-amber-500/10 text-amber-400'
        : 'bg-red-500/10 text-red-400'

  return (
    <Card className={borderColor}>
      <CardContent className="p-4 md:p-6 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h4 className="font-semibold text-foreground">{title}</h4>
          <span className={`text-xs px-2 py-0.5 rounded-full ${badgeColor}`}>{subtitle}</span>
        </div>
        <StatGrid items={metrics} />
        <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
      </CardContent>
    </Card>
  )
}

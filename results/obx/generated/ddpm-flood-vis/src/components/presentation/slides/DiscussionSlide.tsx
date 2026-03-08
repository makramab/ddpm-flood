import { Link } from '@tanstack/react-router'

export function DiscussionSlide() {
  return (
    <div className="max-w-4xl w-full space-y-8">
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Discussion &amp; Limitations</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">Why the Model Falls Short</h2>
      </div>

      <div className="space-y-4">
        {[
          {
            num: 1,
            title: '\u03B8 does not uniquely determine the surge pattern',
            text: 'Two storms with \u03B8 \u2248 2.05m produce spatial correlation of only 0.31. Unlike Wang et al.\u2019s temperature (a thermodynamic state variable), \u03B8 is a downstream point measurement that doesn\u2019t encode storm track, size, or speed.',
          },
          {
            num: 2,
            title: 'Synthetic-to-real generalization gap',
            text: 'Training is 100% Danso parametric storms. At \u03B8 \u2248 2.0m: Danso produces 63% of nodes > 1m surge (broad pattern), Arthur produces only 18% (focused pattern). The model only knows the Danso pattern.',
          },
          {
            num: 3,
            title: 'Skewed \u03B8 distribution',
            text: '82% of training data has \u03B8 < 1.0m, only 6% above 2.0m. The model is poorly trained for high-surge events \u2014 precisely the ones that matter most.',
          },
          {
            num: 4,
            title: 'Confident extrapolation failures',
            text: 'Spatial conditioning made the model more confident everywhere. V1 failed gracefully (R\u00B2 \u2248 -0.08). V2 fails confidently (R\u00B2 \u2248 -1.5) \u2014 actively worse than predicting nothing.',
          },
        ].map((item) => (
          <div key={item.num} className="flex gap-5 bg-muted/30 rounded-lg p-5">
            <span className="text-3xl font-bold text-muted-foreground/20 shrink-0 leading-none pt-0.5">
              {item.num}
            </span>
            <div className="space-y-1">
              <p className="text-base font-semibold text-foreground">{item.title}</p>
              <p className="text-base text-muted-foreground leading-relaxed">{item.text}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <Link to="/0x-primer" hash="limitations" className="text-sm text-primary hover:text-primary/80 transition-colors">
          Primer: Full limitation analysis &rarr;
        </Link>
      </div>
    </div>
  )
}

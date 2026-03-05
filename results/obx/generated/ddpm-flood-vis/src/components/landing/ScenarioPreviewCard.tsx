import { Link } from '@tanstack/react-router'
import type { ScenarioSummary } from '#/lib/data/types'
import { Badge } from '#/components/ui/badge'
import { Card, CardContent } from '#/components/ui/card'

interface ScenarioPreviewCardProps {
  scenario: ScenarioSummary
}

export function ScenarioPreviewCard({ scenario }: ScenarioPreviewCardProps) {
  const isSuccess = scenario.metrics.r2 > 0.5

  return (
    <Link to="/map" search={{ scenario: scenario.id }}>
      <Card className="hover:border-primary/50 transition-all duration-300 hover:shadow-lg h-full">
        <CardContent className="p-4 md:p-6 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-lg font-semibold">{scenario.name}</h3>
            <Badge variant={isSuccess ? 'default' : 'destructive'}>
              R²={scenario.metrics.r2.toFixed(2)}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground">{scenario.label}</p>
          <div className="text-sm">
            <span className="text-muted-foreground">Peak surge (θ): </span>
            <span className="font-mono font-medium">{scenario.theta.toFixed(2)} m</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
            <div>RMSE: <span className="font-mono text-foreground">{scenario.metrics.rmse.toFixed(3)}</span></div>
            <div>Bias: <span className="font-mono text-foreground">{scenario.metrics.bias > 0 ? '+' : ''}{scenario.metrics.bias.toFixed(3)}</span></div>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}

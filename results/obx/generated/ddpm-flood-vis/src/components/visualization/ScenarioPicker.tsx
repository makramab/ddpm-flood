import type { ScenarioSummary } from '#/lib/data/types'
import { Badge } from '#/components/ui/badge'
import { cn } from '#/lib/utils'

interface ScenarioPickerProps {
  scenarios: ScenarioSummary[]
  activeId: string
  onSelect: (id: string) => void
}

export function ScenarioPicker({ scenarios, activeId, onSelect }: ScenarioPickerProps) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Scenario</p>
      <div className="space-y-1">
        {scenarios.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={cn(
              'w-full text-left px-3 py-2 rounded-md border transition-colors text-sm',
              s.id === activeId
                ? 'border-primary bg-primary/10'
                : 'border-border hover:border-muted-foreground/30 hover:bg-muted/50',
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium truncate">{s.name}</span>
              <Badge variant={s.metrics.r2 > 0.5 ? 'default' : 'destructive'} className="text-[10px] shrink-0">
                R²={s.metrics.r2.toFixed(2)}
              </Badge>
            </div>
            <div className="text-xs text-muted-foreground mt-0.5">
              {s.label} · θ={s.theta.toFixed(2)}m
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

import type { LayerType } from '#/lib/data/types'
import { ToggleGroup, ToggleGroupItem } from '#/components/ui/toggle-group'

interface LayerToggleProps {
  activeLayer: LayerType
  onLayerChange: (layer: LayerType) => void
}

const LAYERS: { value: LayerType; label: string }[] = [
  { value: 'truth', label: 'Truth' },
  { value: 'prediction', label: 'Pred' },
  { value: 'error', label: 'Error' },
  { value: 'uncertainty', label: 'Uncert' },
]

export function LayerToggle({ activeLayer, onLayerChange }: LayerToggleProps) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Layer</p>
      <ToggleGroup
        type="single"
        value={activeLayer}
        onValueChange={(v) => { if (v) onLayerChange(v as LayerType) }}
        className="justify-start"
      >
        {LAYERS.map(({ value, label }) => (
          <ToggleGroupItem key={value} value={value} size="sm" className="text-xs px-2.5">
            {label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  )
}

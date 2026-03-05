import { Moon, Sun, Globe } from 'lucide-react'
import type { MapStyleKey } from '#/lib/constants'
import { cn } from '#/lib/utils'

interface MapStyleSwitcherProps {
  activeStyle: MapStyleKey
  onStyleChange: (style: MapStyleKey) => void
}

const STYLES: { key: MapStyleKey; icon: typeof Moon; label: string }[] = [
  { key: 'dark', icon: Moon, label: 'Dark' },
  { key: 'light', icon: Sun, label: 'Light' },
  { key: 'satellite', icon: Globe, label: 'Satellite' },
]

export function MapStyleSwitcher({ activeStyle, onStyleChange }: MapStyleSwitcherProps) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Basemap</p>
      <div className="flex gap-1">
        {STYLES.map(({ key, icon: Icon, label }) => (
          <button
            key={key}
            onClick={() => onStyleChange(key)}
            title={label}
            className={cn(
              'p-1.5 rounded-md transition-colors',
              key === activeStyle
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted',
            )}
          >
            <Icon size={16} />
          </button>
        ))}
      </div>
    </div>
  )
}

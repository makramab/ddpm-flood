import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '#/lib/utils'

interface Props {
  current: number
  total: number
  onPrev: () => void
  onNext: () => void
  onGo: (index: number) => void
}

export function SlideNavigation({ current, total, onPrev, onNext, onGo }: Props) {
  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-sm border-t border-border">
      <button
        onClick={onPrev}
        disabled={current === 0}
        className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-30 disabled:pointer-events-none"
        aria-label="Previous slide"
      >
        <ChevronLeft size={20} />
      </button>

      <div className="flex items-center gap-1.5">
        {Array.from({ length: total }, (_, i) => (
          <button
            key={i}
            onClick={() => onGo(i)}
            className={cn(
              'w-2 h-2 rounded-full transition-all',
              i === current
                ? 'bg-primary w-6'
                : 'bg-muted-foreground/30 hover:bg-muted-foreground/50',
            )}
            aria-label={`Go to slide ${i + 1}`}
          />
        ))}
        <span className="ml-3 text-xs text-muted-foreground font-mono tabular-nums">
          {current + 1} / {total}
        </span>
      </div>

      <button
        onClick={onNext}
        disabled={current === total - 1}
        className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-30 disabled:pointer-events-none"
        aria-label="Next slide"
      >
        <ChevronRight size={20} />
      </button>
    </div>
  )
}

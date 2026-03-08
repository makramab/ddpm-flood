import { useRef } from 'react'
import { useInView, useReducedMotion } from 'motion/react'
import { Link } from '@tanstack/react-router'
import { DiffusionProcessDiagram } from '#/components/primer/DiffusionProcessDiagram'
import { DiffusionAnimatedDiagram } from '#/components/primer/DiffusionAnimatedDiagram'
import { useSyncedDiffusion } from '#/components/primer/useSyncedDiffusion'

export function DiffusionSlide() {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: false, amount: 0.2 })
  const shouldReduceMotion = useReducedMotion()
  const state = useSyncedDiffusion(isInView, shouldReduceMotion)

  return (
    <div className="max-w-5xl w-full space-y-6" ref={ref}>
      <div className="space-y-3">
        <p className="text-sm font-medium text-primary uppercase tracking-widest">Methodology</p>
        <h2 className="text-3xl md:text-4xl font-bold text-foreground">How DDPMs Work</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-muted/50 rounded-lg p-4 space-y-1">
          <p className="text-lg font-medium text-foreground">Forward</p>
          <p className="text-base text-muted-foreground">Gradually add noise to real data over 1,000 steps</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-4 space-y-1">
          <p className="text-lg font-medium text-foreground">Train</p>
          <p className="text-base text-muted-foreground">Model learns to predict the noise at each level</p>
        </div>
        <div className="bg-primary/10 rounded-lg p-4 space-y-1">
          <p className="text-lg font-medium text-primary">Generate</p>
          <p className="text-base text-muted-foreground">Start from noise, denoise 1,000 steps to produce new data</p>
        </div>
      </div>

      <div className="flex flex-col md:flex-row items-center gap-4">
        <div className="flex-1 w-full">
          <DiffusionProcessDiagram state={state} />
        </div>
        <div className="w-full md:w-auto shrink-0">
          <DiffusionAnimatedDiagram state={state} />
        </div>
      </div>

      <div className="flex justify-end">
        <Link to="/0x-primer" hash="ddpms" className="text-sm text-primary hover:text-primary/80 transition-colors">
          Primer: DDPMs explained &rarr;
        </Link>
      </div>
    </div>
  )
}

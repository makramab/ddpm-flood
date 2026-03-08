import { LazyMotion, domAnimation, AnimatePresence, m } from 'motion/react'
import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  slideKey: number
  direction: number
}

const variants = {
  enter: (dir: number) => ({
    x: dir > 0 ? 60 : -60,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
  },
  exit: (dir: number) => ({
    x: dir > 0 ? -60 : 60,
    opacity: 0,
  }),
}

export function SlideShell({ children, slideKey, direction }: Props) {
  return (
    <div className="relative h-[calc(100dvh-3.5rem)] w-full overflow-hidden overflow-y-auto">
      <LazyMotion features={domAnimation}>
        <AnimatePresence mode="wait" custom={direction}>
          <m.div
            key={slideKey}
            custom={direction}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="min-h-full flex items-center justify-center p-6 pb-16 md:p-12 md:pb-20 lg:p-16 lg:pb-24"
          >
            {children}
          </m.div>
        </AnimatePresence>
      </LazyMotion>
    </div>
  )
}

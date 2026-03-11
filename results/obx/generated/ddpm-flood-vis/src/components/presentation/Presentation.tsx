import { useRef, useState } from 'react'
import { useSlideNavigation } from '#/hooks/presentation/useSlideNavigation'
import { SlideShell } from '#/components/presentation/SlideShell'
import { SlideNavigation } from '#/components/presentation/SlideNavigation'
import { IntroSlide } from '#/components/presentation/slides/IntroSlide'
import { ResearchQuestionSlide } from '#/components/presentation/slides/ResearchQuestionSlide'
import { DiffusionSlide } from '#/components/presentation/slides/DiffusionSlide'
import { DatasetSlide } from '#/components/presentation/slides/DatasetSlide'
import { ResultsSlide } from '#/components/presentation/slides/ResultsSlide'
import { DiscussionSlide } from '#/components/presentation/slides/DiscussionSlide'
import { NextStepsSlide } from '#/components/presentation/slides/NextStepsSlide'
import { BroaderLandscapeSlide } from '#/components/presentation/slides/BroaderLandscapeSlide'

const SLIDES = [
  IntroSlide,
  ResearchQuestionSlide,
  DiffusionSlide,
  DatasetSlide,
  ResultsSlide,
  DiscussionSlide,
  NextStepsSlide,
  BroaderLandscapeSlide,
]

export function Presentation() {
  const { current, total, go, next, prev } = useSlideNavigation(SLIDES.length)
  const [direction, setDirection] = useState(0)
  const prevSlide = useRef(current)

  // Track direction for animation
  if (prevSlide.current !== current) {
    setDirection(current > prevSlide.current ? 1 : -1)
    prevSlide.current = current
  }

  const SlideComponent = SLIDES[current]

  return (
    <div className="relative bg-background select-none">
      <SlideShell slideKey={current} direction={direction}>
        <SlideComponent />
      </SlideShell>
      <SlideNavigation
        current={current}
        total={total}
        onPrev={prev}
        onNext={next}
        onGo={go}
      />
    </div>
  )
}

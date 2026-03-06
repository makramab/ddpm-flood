import { createFileRoute } from '@tanstack/react-router'
import { useEffect } from 'react'
import { cn } from '#/lib/utils'
import { useSectionObserver } from '#/hooks/useSectionObserver'
import { ProblemSection, IdeaSection } from '#/components/primer/IntroSections'
import { DDPMSection } from '#/components/primer/DDPMSection'
import { WangSection } from '#/components/primer/WangSection'
import { AdaptationSection } from '#/components/primer/AdaptationSection'
import { DataSection } from '#/components/primer/DataSection'
import { ModelSection } from '#/components/primer/ModelSection'
import { ResultsSection } from '#/components/primer/ResultsSection'
import { LimitationsSection, WhatsNextSection } from '#/components/primer/OutlookSection'
import { GlossarySection } from '#/components/primer/GlossarySection'

const SECTIONS = [
  { id: 'the-problem', label: 'The Problem' },
  { id: 'the-idea', label: 'The Idea' },
  { id: 'ddpms', label: 'DDPMs Explained' },
  { id: 'wang-et-al', label: 'Wang et al.' },
  { id: 'adaptation', label: 'The Adaptation' },
  { id: 'data', label: 'The Data' },
  { id: 'model', label: 'The Model' },
  { id: 'results', label: 'The Results' },
  { id: 'limitations', label: 'Limitations' },
  { id: 'whats-next', label: "What's Next" },
  { id: 'glossary', label: 'Glossary' },
] as const

const SECTION_IDS = SECTIONS.map((s) => s.id)

export const Route = createFileRoute('/0x-primer')({ component: TechnicalPrimerPage })

function TechnicalPrimerPage() {
  const activeId = useSectionObserver(SECTION_IDS)

  useEffect(() => {
    document.documentElement.style.scrollBehavior = 'smooth'
    return () => {
      document.documentElement.style.scrollBehavior = ''
    }
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-5xl mx-auto px-4 md:px-6 py-10 md:py-16">
        <div className="lg:grid lg:grid-cols-[200px_1fr] lg:gap-12">
          <DesktopTOC activeId={activeId} />
          <div className="space-y-16">
            <header className="space-y-3">
              <h1 className="text-3xl md:text-4xl font-black text-foreground">Technical Reference</h1>
              <p className="text-muted-foreground text-sm leading-relaxed max-w-2xl">
                A self-contained guide to understanding every aspect of this project. Written so that someone with no
                prior background can read this document, fully understand the work, and confidently explain it to others
                — including experts.
              </p>
            </header>
            <MobileTOC activeId={activeId} />
            <ProblemSection />
            <IdeaSection />
            <DDPMSection />
            <WangSection />
            <AdaptationSection />
            <DataSection />
            <ModelSection />
            <ResultsSection />
            <LimitationsSection />
            <WhatsNextSection />
            <GlossarySection />
          </div>
        </div>
      </div>
    </div>
  )
}

function DesktopTOC({ activeId }: { activeId: string }) {
  return (
    <nav className="hidden lg:block" aria-label="Table of contents">
      <div className="sticky top-20 space-y-0.5">
        <p className="text-xs font-medium text-muted-foreground mb-3 uppercase tracking-wider">Contents</p>
        {SECTIONS.map((s, i) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className={cn(
              'block text-xs py-1.5 px-3 rounded-md transition-colors',
              activeId === s.id
                ? 'text-primary bg-primary/10 font-medium'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
            )}
          >
            {i + 1}. {s.label}
          </a>
        ))}
      </div>
    </nav>
  )
}

function MobileTOC({ activeId }: { activeId: string }) {
  return (
    <nav
      className="lg:hidden sticky top-14 z-10 -mx-4 px-4 py-2 bg-background/95 backdrop-blur-sm border-b border-border"
      aria-label="Table of contents"
    >
      <div className="flex gap-2 overflow-x-auto pb-1 no-scrollbar">
        {SECTIONS.map((s, i) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className={cn(
              'shrink-0 text-xs py-1 px-2.5 rounded-full transition-colors whitespace-nowrap',
              activeId === s.id
                ? 'text-primary bg-primary/10 font-medium'
                : 'text-muted-foreground hover:text-foreground bg-muted/30',
            )}
          >
            {i + 1}. {s.label}
          </a>
        ))}
      </div>
    </nav>
  )
}

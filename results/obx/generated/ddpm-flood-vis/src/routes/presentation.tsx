import { createFileRoute } from '@tanstack/react-router'
import { Presentation } from '#/components/presentation/Presentation'

export const Route = createFileRoute('/presentation')({ component: PresentationPage })

function PresentationPage() {
  return <Presentation />
}

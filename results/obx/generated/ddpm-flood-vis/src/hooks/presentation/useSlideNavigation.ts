import { useState, useEffect, useCallback } from 'react'

export function useSlideNavigation(totalSlides: number) {
  const [current, setCurrent] = useState(() => {
    if (typeof window === 'undefined') return 0
    const hash = window.location.hash.replace('#', '')
    const n = parseInt(hash, 10)
    return n >= 0 && n < totalSlides ? n : 0
  })

  const go = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, totalSlides - 1))
      setCurrent(clamped)
      window.history.replaceState(null, '', `#${clamped}`)
    },
    [totalSlides],
  )

  const next = useCallback(() => go(current + 1), [go, current])
  const prev = useCallback(() => go(current - 1), [go, current])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Don't intercept if user is in an interactive element
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
        e.preventDefault()
        next()
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault()
        prev()
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [next, prev])

  return { current, total: totalSlides, go, next, prev }
}

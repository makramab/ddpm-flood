import { useRef, useState, useEffect, useCallback } from 'react'
import { LazyMotion, domAnimation, m, useInView, useReducedMotion } from 'motion/react'

type Phase = 'input' | 'encode' | 'bottleneck' | 'skip' | 'decode' | 'output' | 'hold' | 'reset'

interface NodeDef {
  id: string
  x: number
  y: number
  w: number
  h: number
  label: string
  sublabel?: string
  level: number
}

const NODES: NodeDef[] = [
  { id: 'e0', x: 75, y: 30, w: 90, h: 32, label: '32ch', level: 0 },
  { id: 'e1', x: 90, y: 100, w: 80, h: 32, label: '64ch', level: 1 },
  { id: 'e2', x: 90, y: 170, w: 80, h: 32, label: '64ch', level: 2 },
  { id: 'bn', x: 270, y: 240, w: 120, h: 36, label: '128ch', sublabel: 'Attention', level: 3 },
  { id: 'd2', x: 490, y: 170, w: 80, h: 32, label: '64ch', level: 2 },
  { id: 'd1', x: 490, y: 100, w: 80, h: 32, label: '64ch', level: 1 },
  { id: 'd0', x: 495, y: 30, w: 90, h: 32, label: '32ch', level: 0 },
]

// Connections: encoder downward, decoder upward
const EDGES: [string, string][] = [
  ['e0', 'e1'],
  ['e1', 'e2'],
  ['e2', 'bn'],
  ['bn', 'd2'],
  ['d2', 'd1'],
  ['d1', 'd0'],
]

// Skip connections (encoder → decoder at same level)
const SKIPS: [string, string][] = [
  ['e0', 'd0'],
  ['e1', 'd1'],
  ['e2', 'd2'],
]

// Pulse path: node ids in order
const PULSE_PATH = ['e0', 'e1', 'e2', 'bn', 'd2', 'd1', 'd0']

function getCenter(n: NodeDef): [number, number] {
  return [n.x + n.w / 2, n.y + n.h / 2]
}

function nodeById(id: string): NodeDef {
  return NODES.find(n => n.id === id)!
}

const PHASE_DURATIONS: Record<Phase, number> = {
  input: 400,
  encode: 1200,
  bottleneck: 500,
  skip: 400,
  decode: 1200,
  output: 500,
  hold: 300,
  reset: 300,
}

export function UNetFlowDiagram() {
  const containerRef = useRef<HTMLDivElement>(null)
  const isInView = useInView(containerRef, { once: false, amount: 0.3 })
  const shouldReduceMotion = useReducedMotion()
  const [phase, setPhase] = useState<Phase>('reset')
  const [activeNodes, setActiveNodes] = useState<Set<string>>(new Set())
  const [pulsePos, setPulsePos] = useState<[number, number]>([0, 0])
  const [showPulse, setShowPulse] = useState(false)
  const [skipProgress, setSkipProgress] = useState(0)

  const startCycle = useCallback(() => {
    setActiveNodes(new Set())
    setShowPulse(false)
    setSkipProgress(0)
    setPhase('input')
  }, [])

  useEffect(() => {
    if (!isInView || shouldReduceMotion) return

    const timers: ReturnType<typeof setTimeout>[] = []

    if (phase === 'input') {
      const c = getCenter(nodeById('e0'))
      setPulsePos(c)
      setShowPulse(true)
      setActiveNodes(new Set(['e0']))
      timers.push(setTimeout(() => setPhase('encode'), PHASE_DURATIONS.input))
    } else if (phase === 'encode') {
      // Animate pulse through e0 → e1 → e2 → bn
      const encodeNodes = ['e1', 'e2', 'bn']
      const stepTime = PHASE_DURATIONS.encode / encodeNodes.length
      encodeNodes.forEach((id, i) => {
        timers.push(setTimeout(() => {
          const c = getCenter(nodeById(id))
          setPulsePos(c)
          setActiveNodes(prev => new Set([...prev, id]))
        }, stepTime * i))
      })
      timers.push(setTimeout(() => setPhase('bottleneck'), PHASE_DURATIONS.encode))
    } else if (phase === 'bottleneck') {
      timers.push(setTimeout(() => setPhase('skip'), PHASE_DURATIONS.bottleneck))
    } else if (phase === 'skip') {
      // Animate skip connections
      const start = performance.now()
      let rafId: number
      const animate = (now: number) => {
        const t = Math.min((now - start) / PHASE_DURATIONS.skip, 1)
        setSkipProgress(t)
        if (t < 1) {
          rafId = requestAnimationFrame(animate)
        } else {
          setPhase('decode')
        }
      }
      rafId = requestAnimationFrame(animate)
      return () => cancelAnimationFrame(rafId)
    } else if (phase === 'decode') {
      const decodeNodes = ['d2', 'd1', 'd0']
      const stepTime = PHASE_DURATIONS.decode / decodeNodes.length
      decodeNodes.forEach((id, i) => {
        timers.push(setTimeout(() => {
          const c = getCenter(nodeById(id))
          setPulsePos(c)
          setActiveNodes(prev => new Set([...prev, id]))
        }, stepTime * i))
      })
      timers.push(setTimeout(() => setPhase('output'), PHASE_DURATIONS.decode))
    } else if (phase === 'output') {
      timers.push(setTimeout(() => setPhase('hold'), PHASE_DURATIONS.output))
    } else if (phase === 'hold') {
      timers.push(setTimeout(() => setPhase('reset'), PHASE_DURATIONS.hold))
    } else if (phase === 'reset') {
      setActiveNodes(new Set())
      setShowPulse(false)
      setSkipProgress(0)
      timers.push(setTimeout(() => startCycle(), PHASE_DURATIONS.reset))
    }

    return () => timers.forEach(clearTimeout)
  }, [phase, isInView, shouldReduceMotion, startCycle])

  useEffect(() => {
    if (isInView && !shouldReduceMotion) {
      startCycle()
    }
  }, [isInView, shouldReduceMotion, startCycle])

  // Reduced motion: show all nodes active, no pulse
  const displayActive = shouldReduceMotion
    ? new Set(NODES.map(n => n.id))
    : activeNodes
  const displaySkipProgress = shouldReduceMotion ? 1 : skipProgress

  return (
    <div ref={containerRef} className="my-6">
      <LazyMotion features={domAnimation}>
        <svg
          viewBox="0 0 660 300"
          role="img"
          aria-label="Animated U-Net architecture diagram showing encoder layers descending to a bottleneck with attention, then decoder layers ascending, with skip connections between matching levels and an animated pulse traveling through the network."
          className="w-full h-auto"
        >
          {/* Glow filter for bottleneck */}
          <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Input/Output labels */}
          <text x={20} y={50} fill="var(--foreground)" fontSize={10} fontWeight={600}>
            Input(24)
          </text>
          <text x={600} y={50} fill="var(--foreground)" fontSize={10} fontWeight={600}>
            Output(20)
          </text>

          {/* Edges */}
          {EDGES.map(([fromId, toId]) => {
            const from = nodeById(fromId)
            const to = nodeById(toId)
            const [fx, fy] = getCenter(from)
            const [tx, ty] = getCenter(to)
            return (
              <line
                key={`${fromId}-${toId}`}
                x1={fx}
                y1={fy}
                x2={tx}
                y2={ty}
                stroke="var(--border)"
                strokeWidth={1.5}
              />
            )
          })}

          {/* Skip connections */}
          {SKIPS.map(([fromId, toId]) => {
            const from = nodeById(fromId)
            const to = nodeById(toId)
            const fromRight = from.x + from.w
            const toLeft = to.x
            const cy = from.y + from.h / 2
            const totalLen = toLeft - fromRight
            const visibleLen = totalLen * displaySkipProgress
            return (
              <g key={`skip-${fromId}-${toId}`}>
                <line
                  x1={fromRight}
                  y1={cy}
                  x2={fromRight + visibleLen}
                  y2={cy}
                  stroke="var(--primary)"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  opacity={displaySkipProgress > 0 ? 0.7 : 0}
                />
                {displaySkipProgress > 0.5 && (
                  <text
                    x={fromRight + totalLen / 2}
                    y={cy - 6}
                    textAnchor="middle"
                    fill="var(--muted-foreground)"
                    fontSize={7}
                    opacity={Math.min((displaySkipProgress - 0.5) * 2, 1)}
                  >
                    skip
                  </text>
                )}
              </g>
            )
          })}

          {/* Nodes */}
          {NODES.map(node => {
            const isActive = displayActive.has(node.id)
            const isBn = node.id === 'bn'
            const isGlowing = isBn && (phase === 'bottleneck' || shouldReduceMotion)
            return (
              <g key={node.id} filter={isGlowing ? 'url(#glow)' : undefined}>
                <rect
                  x={node.x}
                  y={node.y}
                  width={node.w}
                  height={node.h}
                  rx={5}
                  fill={isActive ? 'var(--primary)' : 'var(--muted)'}
                  stroke={isActive ? 'var(--primary)' : 'var(--border)'}
                  strokeWidth={1.5}
                  style={{ transition: 'fill 0.3s ease, stroke 0.3s ease' }}
                />
                <text
                  x={node.x + node.w / 2}
                  y={node.y + (node.sublabel ? node.h / 2 - 2 : node.h / 2 + 3)}
                  textAnchor="middle"
                  fill={isActive ? 'var(--primary-foreground)' : 'var(--foreground)'}
                  fontSize={9}
                  fontWeight={600}
                  style={{ transition: 'fill 0.3s ease' }}
                >
                  {node.label}
                </text>
                {node.sublabel && (
                  <text
                    x={node.x + node.w / 2}
                    y={node.y + node.h / 2 + 10}
                    textAnchor="middle"
                    fill={isActive ? 'var(--primary-foreground)' : 'var(--muted-foreground)'}
                    fontSize={7}
                    style={{ transition: 'fill 0.3s ease' }}
                  >
                    {node.sublabel}
                  </text>
                )}
              </g>
            )
          })}

          {/* Animated pulse */}
          {showPulse && !shouldReduceMotion && (
            <m.circle
              r={6}
              fill="var(--primary)"
              opacity={0.8}
              animate={{ cx: pulsePos[0], cy: pulsePos[1] }}
              transition={{ type: 'tween', duration: 0.3, ease: 'easeInOut' }}
            />
          )}

          {/* Encoder / Decoder labels */}
          <text x={120} y={288} textAnchor="middle" fill="var(--muted-foreground)" fontSize={9}>
            Encoder
          </text>
          <text x={330} y={288} textAnchor="middle" fill="var(--muted-foreground)" fontSize={9}>
            Bottleneck
          </text>
          <text x={530} y={288} textAnchor="middle" fill="var(--muted-foreground)" fontSize={9}>
            Decoder
          </text>
        </svg>
      </LazyMotion>
    </div>
  )
}

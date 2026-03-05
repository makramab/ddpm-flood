import { Link } from '@tanstack/react-router'
import { Home, Map, BarChart3, Waves, PanelLeftClose, PanelLeft, Sun, Moon } from 'lucide-react'
import { cn } from '#/lib/utils'
import { useSidebar } from '#/hooks/useSidebar'
import { useTheme } from '#/hooks/useTheme'
import { SIDEBAR_WIDTH_EXPANDED, SIDEBAR_WIDTH_COLLAPSED } from '#/lib/constants'

const NAV_LINKS = [
  { to: '/' as const, label: 'Home', icon: Home },
  { to: '/map' as const, label: 'Map', icon: Map },
  { to: '/metrics' as const, label: 'Metrics', icon: BarChart3 },
]

export default function Header() {
  const { expanded, setExpanded } = useSidebar()
  const { dark, toggle } = useTheme()

  const linkBase = (exp: boolean) => cn(
    'flex items-center gap-3 rounded-md transition-colors text-sm',
    exp ? 'px-3 py-2' : 'justify-center px-0 py-2',
  )

  return (
    <>
      {/* Top header bar */}
      <header
        className="fixed top-0 right-0 z-30 h-14 flex items-center justify-between px-4 border-b border-border bg-background transition-all"
        style={{ left: expanded ? SIDEBAR_WIDTH_EXPANDED : SIDEBAR_WIDTH_COLLAPSED }}
      >
        <Link to="/" className="text-sm font-semibold text-foreground hover:text-foreground/80 transition-colors">
          DDPM Flood Prediction
        </Link>
        <button
          onClick={toggle}
          className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {dark ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </header>

      {/* Sidebar — always visible */}
      <aside
        className="fixed top-0 left-0 h-full z-40 flex flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground transition-all duration-200"
        style={{ width: expanded ? SIDEBAR_WIDTH_EXPANDED : SIDEBAR_WIDTH_COLLAPSED }}
      >
        {/* Logo + collapse toggle */}
        <div className="group h-14 flex items-center justify-between px-3 border-b border-sidebar-border shrink-0">
          {expanded ? (
            <>
              <Link to="/" className="flex items-center gap-2 overflow-hidden">
                <Waves size={22} className="shrink-0 text-foreground" />
                <span className="text-sm font-bold whitespace-nowrap">DDPM Flood</span>
              </Link>
              <button
                onClick={() => setExpanded(false)}
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-sidebar-accent transition-colors"
                aria-label="Collapse sidebar"
              >
                <PanelLeftClose size={18} />
              </button>
            </>
          ) : (
            <div className="w-full flex items-center justify-center">
              <Link to="/" className="group-hover:hidden">
                <Waves size={22} className="text-foreground" />
              </Link>
              <button
                onClick={() => setExpanded(true)}
                className="hidden group-hover:block p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-sidebar-accent transition-colors"
                aria-label="Expand sidebar"
              >
                <PanelLeft size={18} />
              </button>
            </div>
          )}
        </div>

        {/* Nav links */}
        <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
          {NAV_LINKS.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={cn(linkBase(expanded), 'text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent')}
              activeProps={{
                className: cn(linkBase(expanded), 'bg-sidebar-accent text-sidebar-accent-foreground font-medium'),
              }}
            >
              <Icon size={18} className="shrink-0" />
              {expanded && <span>{label}</span>}
            </Link>
          ))}
        </nav>
      </aside>
    </>
  )
}

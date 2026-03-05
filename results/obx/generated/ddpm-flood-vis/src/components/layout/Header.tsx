import { Link } from '@tanstack/react-router'
import { useState } from 'react'
import { Home, Map, BarChart3, Menu, X, Waves } from 'lucide-react'

const NAV_LINKS = [
  { to: '/' as const, label: 'Home', icon: Home },
  { to: '/map' as const, label: 'Map', icon: Map },
  { to: '/metrics' as const, label: 'Metrics', icon: BarChart3 },
]

export default function Header() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <header className="p-3 md:p-4 flex items-center bg-gray-800 text-white shadow-lg">
        <button
          onClick={() => setIsOpen(true)}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          aria-label="Open menu"
        >
          <Menu size={24} />
        </button>
        <Link to="/" className="ml-3 flex items-center gap-2">
          <Waves size={28} className="text-cyan-400" />
          <span className="text-xl font-semibold">DDPM Flood</span>
        </Link>
        <nav className="hidden md:flex ml-auto gap-1">
          {NAV_LINKS.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:text-white hover:bg-gray-700 transition-colors"
              activeProps={{ className: 'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-white bg-gray-700 transition-colors' }}
            >
              <Icon size={16} />
              {label}
            </Link>
          ))}
        </nav>
      </header>

      <aside
        className={`fixed top-0 left-0 h-full w-72 bg-gray-900 text-white shadow-2xl z-50 transform transition-transform duration-300 ease-in-out flex flex-col ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <Waves size={24} className="text-cyan-400" />
            <span className="text-lg font-bold">DDPM Flood</span>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            aria-label="Close menu"
          >
            <X size={24} />
          </button>
        </div>
        <nav className="flex-1 p-4 overflow-y-auto">
          {NAV_LINKS.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              onClick={() => setIsOpen(false)}
              className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-800 transition-colors mb-1"
              activeProps={{
                className: 'flex items-center gap-3 p-3 rounded-lg bg-cyan-600 hover:bg-cyan-700 transition-colors mb-1',
              }}
            >
              <Icon size={20} />
              <span className="font-medium">{label}</span>
            </Link>
          ))}
        </nav>
      </aside>

      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  )
}

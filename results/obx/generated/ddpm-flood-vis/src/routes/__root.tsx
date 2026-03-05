import { HeadContent, Scripts, createRootRoute } from '@tanstack/react-router'
import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import { TanStackDevtools } from '@tanstack/react-devtools'

import Header from '#/components/layout/Header'
import { SidebarProvider } from '#/components/layout/SidebarContext'
import { useSidebar } from '#/hooks/useSidebar'
import { SIDEBAR_WIDTH_EXPANDED, SIDEBAR_WIDTH_COLLAPSED } from '#/lib/constants'

import appCss from '#/styles.css?url'
import mapboxCss from 'mapbox-gl/dist/mapbox-gl.css?url'

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'DDPM Flood Visualization' },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'stylesheet', href: mapboxCss },
    ],
  }),
  shellComponent: RootDocument,
})

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <HeadContent />
      </head>
      <body>
        <SidebarProvider>
          <Header />
          <PageContent>{children}</PageContent>
        </SidebarProvider>
        <TanStackDevtools
          config={{ position: 'bottom-right' }}
          plugins={[
            { name: 'Tanstack Router', render: <TanStackRouterDevtoolsPanel /> },
          ]}
        />
        <Scripts />
      </body>
    </html>
  )
}

function PageContent({ children }: { children: React.ReactNode }) {
  const { expanded } = useSidebar()
  return (
    <div
      className="pt-14 transition-all duration-200"
      style={{ marginLeft: expanded ? SIDEBAR_WIDTH_EXPANDED : SIDEBAR_WIDTH_COLLAPSED }}
    >
      {children}
    </div>
  )
}

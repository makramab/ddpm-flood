import { createContext, useState, type ReactNode } from 'react'

export interface SidebarState {
  expanded: boolean
  setExpanded: (v: boolean | ((prev: boolean) => boolean)) => void
}

export const SidebarContext = createContext<SidebarState | null>(null)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <SidebarContext.Provider value={{ expanded, setExpanded }}>
      {children}
    </SidebarContext.Provider>
  )
}

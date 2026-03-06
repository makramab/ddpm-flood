import { Card, CardContent } from '#/components/ui/card'

export function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 space-y-6">
      <h2 className="text-2xl font-bold text-foreground">{title}</h2>
      {children}
    </section>
  )
}

export function Subsection({ id, title, children }: { id?: string; title: string; children: React.ReactNode }) {
  return (
    <div id={id} className={id ? 'scroll-mt-20 space-y-4' : 'space-y-4'}>
      <h3 className="text-lg font-semibold text-foreground">{title}</h3>
      {children}
    </div>
  )
}

export function Prose({ children }: { children: React.ReactNode }) {
  return <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">{children}</div>
}

export function B({ children }: { children: React.ReactNode }) {
  return <span className="text-foreground font-medium">{children}</span>
}

export function Mono({ children }: { children: React.ReactNode }) {
  return <span className="font-mono text-xs text-foreground bg-muted/50 px-1.5 py-0.5 rounded">{children}</span>
}

export function InfoCard({ title, children, accent = false }: { title?: string; children: React.ReactNode; accent?: boolean }) {
  return (
    <Card className={accent ? 'border-primary/30' : ''}>
      <CardContent className="p-4 md:p-6 space-y-2">
        {title && <h4 className="font-semibold text-foreground text-sm">{title}</h4>}
        <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
      </CardContent>
    </Card>
  )
}

export function DataTable({ headers, rows, compact = false }: { headers: string[]; rows: (string | React.ReactNode)[][]; compact?: boolean }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-muted/50">
            {headers.map((h, i) => (
              <th key={i} className={`${compact ? 'px-2 py-1.5' : 'px-3 py-2'} text-left font-medium text-foreground text-xs`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-border">
              {row.map((cell, j) => (
                <td key={j} className={`${compact ? 'px-2 py-1.5' : 'px-3 py-2'} text-muted-foreground`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="bg-muted/50 rounded-lg p-4 overflow-x-auto text-xs font-mono text-foreground leading-relaxed whitespace-pre">
      {children}
    </pre>
  )
}

export function StatGrid({ items }: { items: { label: string; value: string }[] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
      {items.map((item, i) => (
        <div key={i} className="bg-muted/50 rounded-lg p-3 space-y-1">
          <p className="text-xs text-muted-foreground">{item.label}</p>
          <p className="text-foreground font-medium text-sm">{item.value}</p>
        </div>
      ))}
    </div>
  )
}

export function NumberedCard({ number, title, children }: { number: number; title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardContent className="p-4 md:p-6 flex gap-4">
        <span className="text-2xl font-bold text-muted-foreground/30 shrink-0">{number}</span>
        <div className="space-y-1">
          <h4 className="font-semibold text-foreground text-sm">{title}</h4>
          <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
        </div>
      </CardContent>
    </Card>
  )
}

export function StepList({ steps }: { steps: string[] }) {
  return (
    <ol className="space-y-2 text-sm text-muted-foreground">
      {steps.map((step, i) => (
        <li key={i} className="flex gap-3">
          <span className="shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">{i + 1}</span>
          <span className="leading-relaxed pt-0.5">{step}</span>
        </li>
      ))}
    </ol>
  )
}

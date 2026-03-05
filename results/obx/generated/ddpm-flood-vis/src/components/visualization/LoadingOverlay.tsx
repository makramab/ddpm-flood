export function LoadingOverlay() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm z-10">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 border-2 border-foreground border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-muted-foreground">Loading scenario data...</p>
      </div>
    </div>
  )
}

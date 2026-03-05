interface ErrorBannerProps {
  message: string
  onRetry?: () => void
}

export function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
      <div className="text-center space-y-3 p-6">
        <p className="text-destructive font-medium">Failed to load data</p>
        <p className="text-sm text-muted-foreground max-w-xs">{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  )
}

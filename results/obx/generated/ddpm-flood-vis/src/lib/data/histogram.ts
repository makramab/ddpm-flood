export interface HistogramBin {
  label: string
  count: number
  min: number
  max: number
}

export function binData(values: number[], binCount: number): HistogramBin[] {
  if (values.length === 0) return []

  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }

  if (min === max) {
    return [{ label: min.toFixed(2), count: values.length, min, max }]
  }

  const binWidth = (max - min) / binCount

  const bins: HistogramBin[] = Array.from({ length: binCount }, (_, i) => ({
    label: (min + binWidth * (i + 0.5)).toFixed(2),
    count: 0,
    min: min + binWidth * i,
    max: min + binWidth * (i + 1),
  }))

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1)
    bins[idx].count++
  }

  return bins
}

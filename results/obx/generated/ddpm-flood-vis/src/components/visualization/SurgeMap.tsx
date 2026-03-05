import { Map } from 'react-map-gl/mapbox'
import DeckGL from '@deck.gl/react'
import type { Layer } from '@deck.gl/core'
import type { PickingInfo } from '@deck.gl/core'
import { MAP_CONFIG } from '#/lib/constants'

interface SurgeMapProps {
  layers: Layer[]
  getTooltip: (info: PickingInfo) => string | null
  mapStyle: string
}

export function SurgeMap({ layers, getTooltip, mapStyle }: SurgeMapProps) {
  return (
    <DeckGL
      initialViewState={MAP_CONFIG.initialViewState}
      controller={{ dragRotate: true }}
      layers={layers}
      getTooltip={getTooltip}
      style={{ width: '100%', height: '100%' }}
    >
      <Map
        mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}
        mapStyle={mapStyle}
      />
    </DeckGL>
  )
}

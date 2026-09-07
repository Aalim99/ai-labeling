export interface Box {
  id: string
  x: number
  y: number
  width: number
  height: number
  className: string
  confidence?: number
}

export interface LabeledImage {
  id: string
  name: string
  url: string
  width: number
  height: number
  boxes: Box[]
  detected: boolean
  edited: boolean
}

export type LabelDisplay = 'confidence' | 'label' | 'none'

/** Tiling mode: auto turns it on for images large enough to need it. */
export type TileMode = 'auto' | 'on' | 'off'

/** Canvas interaction mode. */
export type Tool = 'select' | 'draw' | 'pan'

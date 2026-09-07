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
  base64: string
  width: number
  height: number
  boxes: Box[]
  detected: boolean
  edited: boolean
}

export type LabelDisplay = 'confidence' | 'label' | 'none'

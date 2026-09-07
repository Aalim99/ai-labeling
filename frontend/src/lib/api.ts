import type { Box, LabeledImage } from '../types'

// Defaults to the Vite dev proxy; override with VITE_API_BASE for a backend on
// another host or port (e.g. VITE_API_BASE=http://localhost:8010/api).
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export interface HealthStatus {
  ready: boolean
  loading: boolean
  model: string
  engine: string
  /** False for a custom trained checkpoint, whose classes are fixed. */
  prompted: boolean
  classes: string[]
  device: string
  error: string | null
}

export interface DetectOptions {
  imgsz: number
  tiled: boolean
  tileSize: number
  tileOverlap: number
  /** Run each tile at this input size; 0 means the tile's own pixel size. */
  tileImgsz: number
  /** Add a whole-image pass alongside the tiles, so big parts are seen too. */
  multiscale: boolean
}

interface RawPrediction {
  x: number
  y: number
  width: number
  height: number
  confidence: number
  class_name: string
  class_id: number
  detection_id: string
}

/** Turns fetch's opaque "Failed to fetch" into something actionable. */
function networkError(): Error {
  return new Error(
    `Cannot reach the backend at ${API_BASE}. Start it with "uvicorn app.main:app --port 8000" ` +
      `in the backend folder, and check the port matches the proxy in vite.config.ts.`,
  )
}

async function readError(res: Response, fallback: string): Promise<Error> {
  const body = await res.json().catch(() => null)
  return new Error(body?.detail || `${fallback} (HTTP ${res.status})`)
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw await readError(res, 'Health check failed')
  return res.json()
}

export async function detectObjects(
  file: File,
  prompts: string[],
  confidence: number,
  iou: number,
  options: DetectOptions,
): Promise<Box[]> {
  const form = new FormData()
  form.append('image', file)
  form.append('prompts', prompts.join(','))
  form.append('confidence', String(confidence))
  form.append('iou', String(iou))
  form.append('imgsz', String(options.imgsz))
  form.append('tiled', String(options.tiled))
  form.append('tile_size', String(options.tileSize))
  form.append('tile_overlap', String(options.tileOverlap))
  form.append('tile_imgsz', String(options.tileImgsz))
  form.append('multiscale', String(options.multiscale))

  let res: Response
  try {
    res = await fetch(`${API_BASE}/detect`, { method: 'POST', body: form })
  } catch {
    throw networkError()
  }
  if (!res.ok) throw await readError(res, 'Detection failed')

  const data: { predictions: RawPrediction[] } = await res.json()
  return data.predictions.map((p) => ({
    id: p.detection_id,
    x: p.x,
    y: p.y,
    width: p.width,
    height: p.height,
    className: p.class_name,
    confidence: p.confidence,
  }))
}

/** An image plus its data URL, resolved lazily at export time. */
export interface ExportImage extends LabeledImage {
  base64: string
}

export async function exportYolo(
  images: ExportImage[],
  classes: string[],
  datasetName: string,
): Promise<Blob> {
  const payload = {
    dataset_name: datasetName,
    classes,
    images: images.map((img) => ({
      filename: img.name,
      image_base64: img.base64,
      width: img.width,
      height: img.height,
      boxes: img.boxes.map((b) => ({
        x: b.x,
        y: b.y,
        width: b.width,
        height: b.height,
        class_name: b.className,
      })),
    })),
  }

  let res: Response
  try {
    res = await fetch(`${API_BASE}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw networkError()
  }
  if (!res.ok) throw await readError(res, 'Export failed')
  return res.blob()
}

export interface TrainStatus {
  state: 'idle' | 'preparing' | 'training' | 'done' | 'error' | 'cancelled'
  epoch: number
  total_epochs: number
  images: number
  classes: string[]
  metrics: Record<string, number>
  error: string | null
  weights: string | null
  message: string
}

export interface ModelList {
  prompted: string[]
  base: string[]
  trained: { path: string; name: string; modified: number }[]
  active: string
}

interface TrainImage {
  filename: string
  image_base64: string
  width: number
  height: number
  boxes: { x: number; y: number; width: number; height: number; class_name: string }[]
}

export async function fetchModels(): Promise<ModelList> {
  const res = await fetch(`${API_BASE}/models`)
  if (!res.ok) throw await readError(res, 'Could not list models')
  return res.json()
}

export async function selectModel(model: string): Promise<HealthStatus> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}/model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    })
  } catch {
    throw networkError()
  }
  if (!res.ok) throw await readError(res, 'Could not switch model')
  return res.json()
}

export async function fetchTrainStatus(): Promise<TrainStatus> {
  const res = await fetch(`${API_BASE}/train/status`)
  if (!res.ok) throw await readError(res, 'Could not read training status')
  return res.json()
}

export async function startTraining(
  images: TrainImage[],
  classes: string[],
  epochs: number,
  baseModel: string,
): Promise<TrainStatus> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images, classes, epochs, base_model: baseModel }),
    })
  } catch {
    throw networkError()
  }
  if (!res.ok) throw await readError(res, 'Could not start training')
  return res.json()
}

export async function cancelTraining(): Promise<TrainStatus> {
  const res = await fetch(`${API_BASE}/train/cancel`, { method: 'POST' })
  if (!res.ok) throw await readError(res, 'Could not cancel training')
  return res.json()
}

import type { Box, LabeledImage } from '../types'

export interface DetectResponse {
  image: { width: number; height: number }
  predictions: {
    x: number
    y: number
    width: number
    height: number
    confidence: number
    class_name: string
    class_id: number
    detection_id: string
  }[]
}

export async function detectObjects(
  file: File,
  prompts: string[],
  confidence: number,
  iou: number,
): Promise<Box[]> {
  const form = new FormData()
  form.append('image', file)
  form.append('prompts', prompts.join(','))
  form.append('confidence', String(confidence))
  form.append('iou', String(iou))

  const res = await fetch('/api/detect', { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Detection failed (${res.status})`)
  }
  const data: DetectResponse = await res.json()
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

export async function exportYolo(
  images: LabeledImage[],
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

  const res = await fetch('/api/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Export failed (${res.status})`)
  }
  return res.blob()
}

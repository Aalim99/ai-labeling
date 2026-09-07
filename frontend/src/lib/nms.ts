import type { Box } from '../types'

function iou(a: Box, b: Box): number {
  const ax1 = a.x - a.width / 2
  const ay1 = a.y - a.height / 2
  const ax2 = a.x + a.width / 2
  const ay2 = a.y + a.height / 2
  const bx1 = b.x - b.width / 2
  const by1 = b.y - b.height / 2
  const bx2 = b.x + b.width / 2
  const by2 = b.y + b.height / 2

  const interX = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
  const interY = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1))
  const interArea = interX * interY
  if (interArea <= 0) return 0

  const areaA = a.width * a.height
  const areaB = b.width * b.height
  return interArea / (areaA + areaB - interArea)
}

/**
 * Filters raw detections by confidence, then applies per-class greedy NMS at
 * the given overlap threshold. Lets the confidence/overlap sliders re-render
 * instantly from one raw detection pass instead of re-calling the backend.
 */
export function filterAndNms(
  boxes: Box[],
  confidenceThreshold: number,
  overlapThreshold: number,
): Box[] {
  const byClass = new Map<string, Box[]>()
  for (const box of boxes) {
    if ((box.confidence ?? 1) < confidenceThreshold) continue
    const arr = byClass.get(box.className) ?? []
    arr.push(box)
    byClass.set(box.className, arr)
  }

  const kept: Box[] = []
  for (const group of byClass.values()) {
    const active = [...group].sort((a, b) => (b.confidence ?? 1) - (a.confidence ?? 1))
    while (active.length) {
      const current = active.shift()!
      kept.push(current)
      for (let i = active.length - 1; i >= 0; i--) {
        if (iou(current, active[i]) > overlapThreshold) {
          active.splice(i, 1)
        }
      }
    }
  }
  return kept
}

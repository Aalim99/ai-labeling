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
 * Filters raw detections by confidence, then suppresses overlapping duplicates.
 * Lets the confidence/overlap sliders re-render instantly from one raw
 * detection pass instead of re-calling the backend.
 *
 * mergeClasses collapses duplicates that landed on the same part under
 * different labels — a chip resistor and a chip capacitor look identical, so
 * open-vocabulary models routinely emit both for one component.
 */
export function filterAndNms(
  boxes: Box[],
  confidenceThreshold: number,
  overlapThreshold: number,
  mergeClasses = false,
): Box[] {
  const candidates = boxes
    .filter((box) => (box.confidence ?? 1) >= confidenceThreshold)
    .sort((a, b) => (b.confidence ?? 1) - (a.confidence ?? 1))

  const kept: Box[] = []
  for (const box of candidates) {
    const duplicate = kept.some(
      (k) => (mergeClasses || k.className === box.className) && iou(k, box) > overlapThreshold,
    )
    if (!duplicate) kept.push(box)
  }
  return kept
}

const PALETTE = [
  '#22d3ee', // cyan
  '#f472b6', // pink
  '#a3e635', // lime
  '#c084fc', // purple
  '#fb923c', // orange
  '#facc15', // yellow
  '#4ade80', // green
  '#60a5fa', // blue
  '#f87171', // red
  '#2dd4bf', // teal
]

function hashString(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i++) {
    hash = (hash << 5) - hash + value.charCodeAt(i)
    hash |= 0
  }
  return Math.abs(hash)
}

export function colorForClass(className: string): string {
  if (!className) return '#94a3b8'
  return PALETTE[hashString(className) % PALETTE.length]
}

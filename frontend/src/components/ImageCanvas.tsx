import { useCallback, useEffect, useRef, useState } from 'react'
import { colorForClass } from '../lib/colors'
import type { Box, LabelDisplay, LabeledImage } from '../types'

interface Props {
  image: LabeledImage | null
  labelDisplay: LabelDisplay
  opacity: number
  activeClass: string
  promptClasses: string[]
  hiddenClasses: Set<string>
  detecting: boolean
  progressLabel: string | null
  onBoxesChange: (boxes: Box[]) => void
}

interface Draft {
  x1: number
  y1: number
  x2: number
  y2: number
}

const MIN_ZOOM = 0.1
const MAX_ZOOM = 20

function alphaHex(opacity: number): string {
  return Math.round((opacity / 100) * 255)
    .toString(16)
    .padStart(2, '0')
}

export default function ImageCanvas({
  image,
  labelDisplay,
  opacity,
  activeClass,
  promptClasses,
  hiddenClasses,
  detecting,
  progressLabel,
  onBoxesChange,
}: Props) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const [viewport, setViewport] = useState({ w: 800, h: 600 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [spaceHeld, setSpaceHeld] = useState(false)

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setViewport({ w: width, h: height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Scale that fits the whole image in view; zoom multiplies it.
  const fitScale = image ? Math.min(viewport.w / image.width, viewport.h / image.height) : 1
  const scale = fitScale * zoom
  const displayW = image ? image.width * scale : 0
  const displayH = image ? image.height * scale : 0

  const fitToView = useCallback(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])

  useEffect(() => {
    fitToView()
    setSelectedBoxId(null)
  }, [image?.id, fitToView])

  // Centered when it fits, otherwise offset by the pan amount.
  const originX = displayW < viewport.w ? (viewport.w - displayW) / 2 + pan.x : pan.x
  const originY = displayH < viewport.h ? (viewport.h - displayH) / 2 + pan.y : pan.y

  /** Keeps the image from being dragged out of view; 0 means centered. */
  const clampPan = useCallback(
    (p: { x: number; y: number }, w: number, h: number) => ({
      x: Math.min(0, Math.max(p.x, -Math.max(0, w - viewport.w))),
      y: Math.min(0, Math.max(p.y, -Math.max(0, h - viewport.h))),
    }),
    [viewport.w, viewport.h],
  )

  useEffect(() => {
    function down(e: KeyboardEvent) {
      if (e.code === 'Space') {
        const target = e.target as HTMLElement
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
        e.preventDefault()
        setSpaceHeld(true)
      }
    }
    function up(e: KeyboardEvent) {
      if (e.code === 'Space') setSpaceHeld(false)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return

      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBoxId && image) {
        e.preventDefault()
        onBoxesChange(image.boxes.filter((b) => b.id !== selectedBoxId))
        setSelectedBoxId(null)
      }
      if (e.key === 'Escape') setSelectedBoxId(null)
      if (e.key === '0') fitToView()
      if (e.key === '+' || e.key === '=') setZoom((z) => Math.min(MAX_ZOOM, z * 1.25))
      if (e.key === '-') setZoom((z) => Math.max(MIN_ZOOM, z / 1.25))
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedBoxId, image, onBoxesChange, fitToView])

  function toImagePoint(clientX: number, clientY: number) {
    const rect = wrapperRef.current!.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(image!.width, (clientX - rect.left) / scale)),
      y: Math.max(0, Math.min(image!.height, (clientY - rect.top) / scale)),
    }
  }

  // Zoom toward the cursor so the point under it stays put. Registered natively
  // because React's wheel listener is passive and cannot preventDefault.
  useEffect(() => {
    const el = viewportRef.current
    if (!el || !image) return

    function onWheel(e: WheelEvent) {
      e.preventDefault()
      const rect = el!.getBoundingClientRect()
      const cursorX = e.clientX - rect.left
      const cursorY = e.clientY - rect.top

      const nextZoom = Math.min(
        MAX_ZOOM,
        Math.max(MIN_ZOOM, zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)),
      )
      const nextScale = fitScale * nextZoom
      const nextW = image!.width * nextScale
      const nextH = image!.height * nextScale

      // Image coordinate currently under the cursor.
      const imgX = (cursorX - originX) / scale
      const imgY = (cursorY - originY) / scale
      const nextOriginX = cursorX - imgX * nextScale
      const nextOriginY = cursorY - imgY * nextScale

      setZoom(nextZoom)
      setPan(
        clampPan(
          {
            x: nextW < viewport.w ? nextOriginX - (viewport.w - nextW) / 2 : nextOriginX,
            y: nextH < viewport.h ? nextOriginY - (viewport.h - nextH) / 2 : nextOriginY,
          },
          nextW,
          nextH,
        ),
      )
    }

    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [image, zoom, fitScale, scale, originX, originY, viewport.w, viewport.h, clampPan])

  function startPan(e: React.MouseEvent) {
    e.preventDefault()
    const start = { x: e.clientX, y: e.clientY }
    const origin = { ...pan }
    const onMove = (ev: MouseEvent) => {
      setPan(
        clampPan(
          { x: origin.x + (ev.clientX - start.x), y: origin.y + (ev.clientY - start.y) },
          displayW,
          displayH,
        ),
      )
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  function startDraw(e: React.MouseEvent) {
    if (!image) return
    // Middle mouse or space-drag pans instead of drawing.
    if (e.button === 1 || spaceHeld) {
      startPan(e)
      return
    }
    if (e.button !== 0) return
    const target = e.target as HTMLElement
    if (target !== wrapperRef.current && target.tagName !== 'IMG') return
    beginDraw(e)
  }

  function beginDraw(e: React.MouseEvent) {
    if (!image) return
    e.preventDefault()
    setSelectedBoxId(null)

    const start = toImagePoint(e.clientX, e.clientY)
    let current = start
    setDraft({ x1: start.x, y1: start.y, x2: start.x, y2: start.y })

    const onMove = (ev: MouseEvent) => {
      current = toImagePoint(ev.clientX, ev.clientY)
      setDraft({ x1: start.x, y1: start.y, x2: current.x, y2: current.y })
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      setDraft(null)
      const w = Math.abs(current.x - start.x)
      const h = Math.abs(current.y - start.y)
      if (w * scale < 5 || h * scale < 5) return
      const newBox: Box = {
        id: crypto.randomUUID(),
        x: (start.x + current.x) / 2,
        y: (start.y + current.y) / 2,
        width: w,
        height: h,
        className: activeClass || promptClasses[0] || 'object',
      }
      onBoxesChange([...image.boxes, newBox])
      setSelectedBoxId(newBox.id)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  function startMove(e: React.MouseEvent, box: Box) {
    if (!image) return
    if (e.button === 1 || spaceHeld) {
      startPan(e)
      return
    }
    if (e.button !== 0) return
    e.stopPropagation()
    // On a dense board nearly every pixel sits inside some box, so shift-drag
    // forces a new box instead of moving the one underneath.
    if (e.shiftKey) {
      beginDraw(e)
      return
    }
    e.preventDefault()
    setSelectedBoxId(box.id)

    const start = toImagePoint(e.clientX, e.clientY)
    const onMove = (ev: MouseEvent) => {
      const cur = toImagePoint(ev.clientX, ev.clientY)
      const dx = cur.x - start.x
      const dy = cur.y - start.y
      onBoxesChange(
        image.boxes.map((b) => (b.id === box.id ? { ...b, x: box.x + dx, y: box.y + dy } : b)),
      )
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  /** Corner handles resize from the opposite corner, which stays anchored. */
  function startResize(e: React.MouseEvent, box: Box, corner: 'nw' | 'ne' | 'sw' | 'se') {
    if (!image) return
    e.stopPropagation()
    e.preventDefault()
    setSelectedBoxId(box.id)

    const left = box.x - box.width / 2
    const top = box.y - box.height / 2
    const right = box.x + box.width / 2
    const bottom = box.y + box.height / 2
    const anchorX = corner === 'nw' || corner === 'sw' ? right : left
    const anchorY = corner === 'nw' || corner === 'ne' ? bottom : top

    const onMove = (ev: MouseEvent) => {
      const cur = toImagePoint(ev.clientX, ev.clientY)
      const width = Math.max(2, Math.abs(cur.x - anchorX))
      const height = Math.max(2, Math.abs(cur.y - anchorY))
      onBoxesChange(
        image.boxes.map((b) =>
          b.id === box.id
            ? {
                ...b,
                width,
                height,
                x: (anchorX + cur.x) / 2,
                y: (anchorY + cur.y) / 2,
              }
            : b,
        ),
      )
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const visibleBoxes = image ? image.boxes.filter((b) => !hiddenClasses.has(b.className)) : []
  const handles: Array<'nw' | 'ne' | 'sw' | 'se'> = ['nw', 'ne', 'sw', 'se']

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden bg-gray-100">
      <div
        ref={viewportRef}
        className="relative flex-1 overflow-hidden"
        onMouseDown={startDraw}
        onContextMenu={(e) => e.preventDefault()}
        style={{ cursor: spaceHeld ? 'grab' : image ? 'crosshair' : 'default' }}
      >
        {!image ? (
          <div className="flex h-full items-center justify-center text-center text-sm text-gray-400">
            <div>
              <p className="mb-1 font-medium text-gray-500">No image selected</p>
              <p>Upload images on the left, then enter classes and run detection.</p>
            </div>
          </div>
        ) : (
          <div
            ref={wrapperRef}
            className="absolute select-none"
            style={{ left: originX, top: originY, width: displayW, height: displayH }}
          >
            <img
              src={image.url}
              alt={image.name}
              draggable={false}
              className="absolute inset-0 h-full w-full"
              style={{ imageRendering: zoom > 2 ? 'pixelated' : 'auto' }}
            />

            {visibleBoxes.map((box) => {
              const color = colorForClass(box.className)
              const selected = box.id === selectedBoxId
              const label =
                labelDisplay === 'none'
                  ? null
                  : labelDisplay === 'label'
                    ? box.className
                    : box.confidence != null
                      ? `${box.className} ${Math.round(box.confidence * 100)}%`
                      : box.className

              return (
                <div
                  key={box.id}
                  className="absolute"
                  style={{
                    left: (box.x - box.width / 2) * scale,
                    top: (box.y - box.height / 2) * scale,
                    width: box.width * scale,
                    height: box.height * scale,
                    border: `${selected ? 2 : 1}px solid ${selected ? '#111827' : color}`,
                    backgroundColor: `${color}${alphaHex(opacity)}`,
                    cursor: spaceHeld ? 'grab' : 'move',
                  }}
                  onMouseDown={(e) => startMove(e, box)}
                >
                  {label && (
                    <span
                      className="absolute -top-4 left-0 whitespace-nowrap rounded-sm px-1 text-[10px] font-semibold leading-4 text-white"
                      style={{ backgroundColor: color }}
                    >
                      {label}
                    </span>
                  )}

                  {selected && (
                    <>
                      {handles.map((corner) => (
                        <div
                          key={corner}
                          className="absolute h-2.5 w-2.5 rounded-sm border border-white bg-gray-900"
                          style={{
                            left: corner.includes('w') ? -5 : undefined,
                            right: corner.includes('e') ? -5 : undefined,
                            top: corner.startsWith('n') ? -5 : undefined,
                            bottom: corner.startsWith('s') ? -5 : undefined,
                            cursor: corner === 'nw' || corner === 'se' ? 'nwse-resize' : 'nesw-resize',
                          }}
                          onMouseDown={(e) => startResize(e, box, corner)}
                        />
                      ))}
                      <div
                        className="absolute -top-8 left-0 flex items-center gap-2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-[11px] text-white"
                        onMouseDown={(e) => e.stopPropagation()}
                      >
                        <select
                          value={box.className}
                          onChange={(e) =>
                            onBoxesChange(
                              image.boxes.map((b) =>
                                b.id === box.id ? { ...b, className: e.target.value } : b,
                              ),
                            )
                          }
                          className="bg-transparent text-white outline-none"
                        >
                          {[...new Set([box.className, ...promptClasses])].map((c) => (
                            <option key={c} value={c} className="text-black">
                              {c}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => {
                            onBoxesChange(image.boxes.filter((b) => b.id !== box.id))
                            setSelectedBoxId(null)
                          }}
                          className="text-red-400 hover:text-red-300"
                        >
                          delete
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )
            })}

            {draft && (
              <div
                className="absolute border-2 border-dashed border-gray-900/70 bg-gray-900/10"
                style={{
                  left: Math.min(draft.x1, draft.x2) * scale,
                  top: Math.min(draft.y1, draft.y2) * scale,
                  width: Math.abs(draft.x2 - draft.x1) * scale,
                  height: Math.abs(draft.y2 - draft.y1) * scale,
                }}
              />
            )}
          </div>
        )}
      </div>

      {image && (
        <div className="flex items-center justify-between border-t border-gray-200 bg-white px-3 py-1.5 text-[11px] text-gray-500">
          <span className="font-mono">
            {detecting
              ? (progressLabel ?? 'detecting…')
              : `${image.boxes.length} objects${image.edited ? ' · hand-edited' : ''}`}
          </span>
          <span className="hidden md:inline text-gray-400">
            scroll = zoom · space-drag = pan · drag = new box · shift-drag = new box over an existing
            one · 0 = fit
          </span>
          <span className="flex items-center gap-1">
            <button
              onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.25))}
              className="rounded border border-gray-300 px-1.5 hover:bg-gray-50"
            >
              −
            </button>
            <span className="w-12 text-center font-mono">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.25))}
              className="rounded border border-gray-300 px-1.5 hover:bg-gray-50"
            >
              +
            </button>
            <button onClick={fitToView} className="ml-1 rounded border border-gray-300 px-1.5 hover:bg-gray-50">
              fit
            </button>
          </span>
        </div>
      )}
    </div>
  )
}

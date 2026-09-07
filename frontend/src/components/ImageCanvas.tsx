import { useEffect, useRef, useState } from 'react'
import { colorForClass } from '../lib/colors'
import type { Box, LabeledImage, LabelDisplay } from '../types'

interface Props {
  image: LabeledImage | null
  labelDisplay: LabelDisplay
  opacity: number
  activeClass: string
  promptClasses: string[]
  detecting: boolean
  onBoxesChange: (boxes: Box[]) => void
}

interface Draft {
  x1: number
  y1: number
  x2: number
  y2: number
}

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
  detecting,
  onBoxesChange,
}: Props) {
  const outerRef = useRef<HTMLDivElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const [containerSize, setContainerSize] = useState({ w: 800, h: 600 })
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)

  useEffect(() => {
    const el = outerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setContainerSize({ w: width, h: height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    setSelectedBoxId(null)
  }, [image?.id])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return
      if (!selectedBoxId || !image) return
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return
      e.preventDefault()
      onBoxesChange(image.boxes.filter((b) => b.id !== selectedBoxId))
      setSelectedBoxId(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedBoxId, image, onBoxesChange])

  const scale = image
    ? Math.min(containerSize.w / image.width, containerSize.h / image.height)
    : 1
  const displayW = image ? image.width * scale : 0
  const displayH = image ? image.height * scale : 0

  function toImagePoint(clientX: number, clientY: number) {
    const rect = wrapperRef.current!.getBoundingClientRect()
    const x = (clientX - rect.left) / scale
    const y = (clientY - rect.top) / scale
    return {
      x: Math.max(0, Math.min(image!.width, x)),
      y: Math.max(0, Math.min(image!.height, y)),
    }
  }

  function startDraw(e: React.MouseEvent) {
    if (!image) return
    const target = e.target as HTMLElement
    if (target !== wrapperRef.current && target.tagName !== 'IMG') return
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
      if (w * scale < 6 || h * scale < 6) return
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
    e.stopPropagation()
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

  function startResize(e: React.MouseEvent, box: Box) {
    if (!image) return
    e.stopPropagation()
    e.preventDefault()
    setSelectedBoxId(box.id)

    const start = toImagePoint(e.clientX, e.clientY)
    const left = box.x - box.width / 2
    const top = box.y - box.height / 2

    const onMove = (ev: MouseEvent) => {
      const cur = toImagePoint(ev.clientX, ev.clientY)
      const width = Math.max(4, box.width + (cur.x - start.x))
      const height = Math.max(4, box.height + (cur.y - start.y))
      onBoxesChange(
        image.boxes.map((b) =>
          b.id === box.id
            ? { ...b, width, height, x: left + width / 2, y: top + height / 2 }
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

  function relabel(box: Box, className: string) {
    if (!image) return
    onBoxesChange(image.boxes.map((b) => (b.id === box.id ? { ...b, className } : b)))
  }

  function remove(box: Box) {
    if (!image) return
    onBoxesChange(image.boxes.filter((b) => b.id !== box.id))
    setSelectedBoxId(null)
  }

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden bg-gray-50">
      <div ref={outerRef} className="flex flex-1 items-center justify-center overflow-hidden p-6">
        {!image ? (
          <div className="text-center text-sm text-gray-400">
            <p className="mb-1 font-medium text-gray-500">No image selected</p>
            <p>Upload images on the left, then enter classes and run detection.</p>
          </div>
        ) : (
          <div
            ref={wrapperRef}
            className="relative select-none"
            style={{ width: displayW, height: displayH, cursor: 'crosshair' }}
            onMouseDown={startDraw}
          >
            <img
              src={image.url}
              alt={image.name}
              draggable={false}
              className="absolute inset-0 h-full w-full"
            />

            {image.boxes.map((box) => {
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
                    cursor: 'move',
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
                      <div
                        className="absolute -bottom-1 -right-1 h-3 w-3 rounded-sm border border-white bg-gray-900"
                        style={{ cursor: 'nwse-resize' }}
                        onMouseDown={(e) => startResize(e, box)}
                      />
                      <div
                        className="absolute -top-8 left-0 flex items-center gap-2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-[11px] text-white"
                        onMouseDown={(e) => e.stopPropagation()}
                      >
                        <select
                          value={box.className}
                          onChange={(e) => relabel(box, e.target.value)}
                          className="bg-transparent text-white outline-none"
                        >
                          {[...new Set([box.className, ...promptClasses])].map((c) => (
                            <option key={c} value={c} className="text-black">
                              {c}
                            </option>
                          ))}
                        </select>
                        <button onClick={() => remove(box)} className="text-red-400 hover:text-red-300">
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
        <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 rounded-md bg-gray-900/80 px-3 py-1.5 font-mono text-xs text-white">
          {detecting
            ? 'detecting…'
            : `${image.boxes.length} objects${image.edited ? ' · hand-edited' : ' detected'}`}
        </div>
      )}
    </div>
  )
}

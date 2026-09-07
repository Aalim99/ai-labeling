import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ControlsPanel from './components/ControlsPanel'
import ImageCanvas from './components/ImageCanvas'
import Sidebar from './components/Sidebar'
import {
  detectObjects,
  exportYolo,
  fetchHealth,
  type ExportImage,
  type HealthStatus,
} from './lib/api'
import { filterAndNms } from './lib/nms'
import {
  clearImages,
  loadImages,
  loadSettings,
  saveImages,
  saveSettings,
  type StoredImage,
} from './lib/storage'
import type { Box, LabelDisplay, LabeledImage, TileMode } from './types'

// Detections are fetched once at a permissive threshold; the sliders then
// filter that raw set client-side so they stay instant.
const RAW_CONFIDENCE = 0.02
const RAW_IOU = 0.7

interface ImageState extends LabeledImage {
  file: File
  rawBoxes: Box[]
}

interface Settings {
  promptText: string
  confidence: number
  overlap: number
  opacity: number
  labelDisplay: LabelDisplay
  tileMode: TileMode
  tileSize: number
  mergeClasses: boolean
  highRecall: boolean
  selectedId: string | null
}

const DEFAULT_SETTINGS: Settings = {
  promptText: 'integrated circuit chip, capacitor, resistor, connector',
  confidence: 10,
  overlap: 50,
  // Light fill by default: dense boards need the image visible under the boxes.
  opacity: 18,
  labelDisplay: 'confidence',
  tileMode: 'auto',
  tileSize: 640,
  // Visually identical parts (chip resistor vs chip capacitor) otherwise get
  // one box per label on the same component.
  mergeClasses: true,
  highRecall: false,
  selectedId: null,
}

function measure(file: File): Promise<{ url: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => resolve({ url, width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => reject(new Error(`Could not read ${file.name}`))
    img.src = url
  })
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error(`Could not encode ${file.name}`))
    reader.readAsDataURL(file)
  })
}

// Measured: on an 834px image with 25px objects, every input size finds the
// objects, but only at 1280 do they score above a usable confidence (19/20
// above 25%, versus 0/20 at 640, 864, 1024, 1408, 1536 and 1920). So a
// non-tiled pass always runs at 1280, upsampling small images to get there.
const WHOLE_IMAGE_IMGSZ = 1280

/** Tiling only pays off once the image is bigger than the model input. */
function shouldTile(image: { width: number; height: number }, mode: TileMode): boolean {
  if (mode === 'on') return true
  if (mode === 'off') return false
  return Math.max(image.width, image.height) > WHOLE_IMAGE_IMGSZ
}

export default function App() {
  // Read once on mount; later writes must not re-trigger the restore effect.
  const [stored] = useState(() => loadSettings(DEFAULT_SETTINGS))

  const [images, setImages] = useState<ImageState[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [promptText, setPromptText] = useState(stored.promptText)
  const [activeClass, setActiveClass] = useState('')
  const [confidence, setConfidence] = useState(stored.confidence)
  const [overlap, setOverlap] = useState(stored.overlap)
  const [opacity, setOpacity] = useState(stored.opacity)
  const [labelDisplay, setLabelDisplay] = useState<LabelDisplay>(stored.labelDisplay)
  const [tileMode, setTileMode] = useState<TileMode>(stored.tileMode)
  const [tileSize, setTileSize] = useState(stored.tileSize)
  const [mergeClasses, setMergeClasses] = useState(stored.mergeClasses)
  const [highRecall, setHighRecall] = useState(stored.highRecall)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [progressLabel, setProgressLabel] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [restored, setRestored] = useState(false)

  const history = useRef<{ imageId: string; boxes: Box[]; edited: boolean }[]>([])

  const promptClasses = useMemo(
    () =>
      promptText
        .split(',')
        .map((c) => c.trim())
        .filter(Boolean),
    [promptText],
  )

  const selected = images.find((img) => img.id === selectedId) ?? null
  const willTile = !!selected && shouldTile(selected, tileMode)

  // Backend status drives the header pill and disables detection until ready.
  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const status = await fetchHealth()
        if (!cancelled) setHealth(status)
      } catch {
        if (!cancelled) setHealth(null)
      }
    }
    poll()
    const id = setInterval(poll, 4000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  // Restore the previous session's images and annotations.
  useEffect(() => {
    loadImages().then((saved) => {
      if (saved.length) {
        setImages(
          saved.map((img) => ({
            id: img.id,
            name: img.name,
            url: URL.createObjectURL(img.file),
            width: img.width,
            height: img.height,
            boxes: img.boxes,
            rawBoxes: img.rawBoxes,
            detected: img.detected,
            edited: img.edited,
            file: img.file,
          })),
        )
        // Come back to the image you were working on, not the first one.
        const previous = saved.find((img) => img.id === stored.selectedId)
        setSelectedId((previous ?? saved[0]).id)
      }
      setRestored(true)
    })
    // stored is read once on mount, so this runs exactly once.
  }, [stored.selectedId])

  useEffect(() => {
    if (!restored) return
    const id = setTimeout(() => {
      const payload: StoredImage[] = images.map((img) => ({
        id: img.id,
        name: img.name,
        width: img.width,
        height: img.height,
        boxes: img.boxes,
        rawBoxes: img.rawBoxes,
        detected: img.detected,
        edited: img.edited,
        file: img.file,
      }))
      saveImages(payload)
    }, 800)
    return () => clearTimeout(id)
  }, [images, restored])

  useEffect(() => {
    saveSettings({
      promptText,
      confidence,
      overlap,
      opacity,
      labelDisplay,
      tileMode,
      tileSize,
      mergeClasses,
      highRecall,
      selectedId,
    })
  }, [
    promptText,
    confidence,
    overlap,
    opacity,
    labelDisplay,
    tileMode,
    tileSize,
    mergeClasses,
    highRecall,
    selectedId,
  ])

  // Re-apply the threshold sliders to raw detections for every image the user
  // hasn't hand-edited yet.
  useEffect(() => {
    setImages((prev) =>
      prev.map((img) => {
        if (img.edited || !img.detected) return img
        return { ...img, boxes: filterAndNms(img.rawBoxes, confidence / 100, overlap / 100, mergeClasses) }
      }),
    )
  }, [confidence, overlap, mergeClasses])

  const updateBoxes = useCallback(
    (boxes: Box[]) => {
      if (!selectedId) return
      setImages((prev) => {
        const current = prev.find((img) => img.id === selectedId)
        if (current) {
          history.current.push({
            imageId: selectedId,
            boxes: current.boxes,
            edited: current.edited,
          })
          if (history.current.length > 100) history.current.shift()
        }
        return prev.map((img) => (img.id === selectedId ? { ...img, boxes, edited: true } : img))
      })
    },
    [selectedId],
  )

  const undo = useCallback(() => {
    const entry = history.current.pop()
    if (!entry) return
    setImages((prev) =>
      prev.map((img) =>
        img.id === entry.imageId ? { ...img, boxes: entry.boxes, edited: entry.edited } : img,
      ),
    )
    setSelectedId(entry.imageId)
  }, [])

  // Keyboard shortcuts for fast review: image paging, class picking, undo.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        undo()
        return
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return

      if (e.key === 'ArrowRight' || e.key === ']') {
        const index = images.findIndex((img) => img.id === selectedId)
        if (index >= 0 && index < images.length - 1) setSelectedId(images[index + 1].id)
      }
      if (e.key === 'ArrowLeft' || e.key === '[') {
        const index = images.findIndex((img) => img.id === selectedId)
        if (index > 0) setSelectedId(images[index - 1].id)
      }
      if (/^[1-9]$/.test(e.key)) {
        const cls = promptClasses[Number(e.key) - 1]
        if (cls) setActiveClass(cls)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [images, selectedId, promptClasses, undo])

  async function addFiles(files: FileList) {
    setError(null)
    try {
      const loaded: ImageState[] = []
      for (const file of Array.from(files)) {
        const { url, width, height } = await measure(file)
        loaded.push({
          id: crypto.randomUUID(),
          name: file.name,
          url,
          width,
          height,
          boxes: [],
          rawBoxes: [],
          detected: false,
          edited: false,
          file,
        })
      }
      setImages((prev) => [...prev, ...loaded])
      setSelectedId((prev) => prev ?? loaded[0]?.id ?? null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  function removeImage(id: string) {
    setImages((prev) => prev.filter((img) => img.id !== id))
    setSelectedId((prev) => (prev === id ? null : prev))
  }

  async function clearAll() {
    setImages([])
    setSelectedId(null)
    history.current = []
    await clearImages()
  }

  async function detectImage(target: ImageState) {
    const rawBoxes = await detectObjects(target.file, promptClasses, RAW_CONFIDENCE, RAW_IOU, {
      imgsz: WHOLE_IMAGE_IMGSZ,
      tiled: shouldTile(target, tileMode),
      tileSize,
      tileOverlap: 0.25,
      // Running each tile at twice its pixel size upsamples it, which recovers
      // the last few parts at roughly 4x the runtime.
      tileImgsz: highRecall ? tileSize * 2 : 0,
    })

    setImages((prev) =>
      prev.map((img) =>
        img.id === target.id
          ? {
              ...img,
              rawBoxes,
              boxes: filterAndNms(rawBoxes, confidence / 100, overlap / 100),
              detected: true,
              edited: false,
            }
          : img,
      ),
    )
  }

  async function runDetect(targets: ImageState[]) {
    if (!targets.length) return
    setDetecting(true)
    setError(null)

    // Poll tile progress so long tiled runs aren't a silent spinner.
    const poller = setInterval(async () => {
      try {
        const res = await fetch('/api/progress')
        const p = await res.json()
        setProgressLabel(p.active ? `tile ${p.current}/${p.total}` : null)
      } catch {
        setProgressLabel(null)
      }
    }, 700)

    try {
      for (const [index, target] of targets.entries()) {
        if (targets.length > 1) setProgressLabel(`image ${index + 1}/${targets.length}`)
        await detectImage(target)
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      clearInterval(poller)
      setProgressLabel(null)
      setDetecting(false)
    }
  }

  async function handleExport() {
    setExporting(true)
    setError(null)
    try {
      const usedClasses = [
        ...new Set([...promptClasses, ...images.flatMap((i) => i.boxes.map((b) => b.className))]),
      ]
      const payload: ExportImage[] = []
      for (const img of images) {
        payload.push({ ...img, base64: await fileToBase64(img.file) })
      }
      const blob = await exportYolo(payload, usedClasses, 'labeled-dataset')
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'labeled-dataset.zip'
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setExporting(false)
    }
  }

  const statusPill = !health
    ? { text: 'backend offline', color: 'bg-red-100 text-red-700' }
    : health.error
      ? { text: 'model failed', color: 'bg-red-100 text-red-700' }
      : health.ready
        ? { text: `ready · ${health.device}`, color: 'bg-green-100 text-green-700' }
        : { text: 'loading model…', color: 'bg-amber-100 text-amber-700' }

  return (
    <div className="flex h-full flex-col bg-gray-50">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-5 py-2.5">
        <div>
          <h1 className="text-sm font-semibold text-gray-900">AI Auto-Labeling</h1>
          <p className="text-[11px] text-gray-500">
            Prompt-driven object labeling · exports YOLO format
          </p>
        </div>
        <div className="flex items-center gap-3">
          <p className="text-[11px] text-gray-500">
            {images.length} image{images.length === 1 ? '' : 's'} ·{' '}
            {images.reduce((sum, img) => sum + img.boxes.length, 0)} labels
          </p>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${statusPill.color}`}
            title={health?.error ?? health?.model ?? 'Backend not reachable'}
          >
            {statusPill.text}
          </span>
        </div>
      </header>

      {!health && (
        <p className="border-b border-red-200 bg-red-50 px-5 py-1.5 text-[11px] text-red-700">
          Backend not reachable. Run{' '}
          <code className="font-mono">uvicorn app.main:app --port 8000</code> in the backend folder.
        </p>
      )}

      <div className="flex min-h-0 flex-1">
        <Sidebar
          images={images}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onAddFiles={addFiles}
          onRemove={removeImage}
          onClearAll={clearAll}
        />

        <ImageCanvas
          image={selected}
          labelDisplay={labelDisplay}
          opacity={opacity}
          activeClass={activeClass}
          promptClasses={promptClasses}
          hiddenClasses={hiddenClasses}
          detecting={detecting}
          progressLabel={progressLabel}
          onBoxesChange={updateBoxes}
        />

        <ControlsPanel
          promptText={promptText}
          onPromptTextChange={setPromptText}
          promptClasses={promptClasses}
          activeClass={activeClass || promptClasses[0] || ''}
          onActiveClassChange={setActiveClass}
          confidence={confidence}
          onConfidenceChange={setConfidence}
          overlap={overlap}
          onOverlapChange={setOverlap}
          opacity={opacity}
          onOpacityChange={setOpacity}
          labelDisplay={labelDisplay}
          onLabelDisplayChange={setLabelDisplay}
          tileMode={tileMode}
          onTileModeChange={setTileMode}
          tileSize={tileSize}
          onTileSizeChange={setTileSize}
          mergeClasses={mergeClasses}
          onMergeClassesChange={setMergeClasses}
          highRecall={highRecall}
          onHighRecallChange={setHighRecall}
          boxes={selected?.boxes ?? []}
          rawCount={selected?.rawBoxes.length ?? 0}
          hiddenClasses={hiddenClasses}
          onToggleClass={(name) =>
            setHiddenClasses((prev) => {
              const next = new Set(prev)
              if (next.has(name)) next.delete(name)
              else next.add(name)
              return next
            })
          }
          imageSize={selected ? { width: selected.width, height: selected.height } : null}
          willTile={willTile}
          health={health}
          detecting={detecting}
          exporting={exporting}
          hasImages={images.length > 0}
          onDetect={() => runDetect(selected ? [selected] : [])}
          onDetectAll={() => runDetect(images)}
          onExport={handleExport}
          error={error}
        />
      </div>
    </div>
  )
}

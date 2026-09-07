import { useCallback, useEffect, useMemo, useState } from 'react'
import ControlsPanel from './components/ControlsPanel'
import Filmstrip from './components/Filmstrip'
import ImageCanvas from './components/ImageCanvas'
import Toolbar from './components/Toolbar'
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
import type { Box, LabelDisplay, LabeledImage, TileMode, Tool } from './types'

// Detections are fetched once at a permissive threshold; the sliders then
// filter that raw set client-side so they stay instant.
// Floor for the one detection pass. Parts on niche classes score very low, so
// this sits well under any useful threshold to give the slider room.
const RAW_CONFIDENCE = 0.005
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
  promptText:
    'chip | microchip | integrated circuit, ' +
    'smd component | small rectangular chip component, ' +
    'capacitor | electrolytic capacitor, connector | socket',
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
  const [tool, setTool] = useState<Tool>('select')
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [progressLabel, setProgressLabel] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [restored, setRestored] = useState(false)

  type Snapshot = { imageId: string; boxes: Box[]; edited: boolean }
  const [history, setHistory] = useState<Snapshot[]>([])
  const [future, setFuture] = useState<Snapshot[]>([])

  // A class may offer several phrasings ("chip | microchip | ic"). The whole
  // group goes to the model; the first phrasing is what the UI and the export
  // call the class, matching the label the backend reports back.
  const promptGroups = useMemo(
    () =>
      promptText
        .split(',')
        .map((c) => c.trim())
        .filter(Boolean),
    [promptText],
  )

  const promptClasses = useMemo(
    () => promptGroups.map((group) => group.split('|')[0].trim()).filter(Boolean),
    [promptGroups],
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

  const snapshot = useCallback(
    (imageId: string): Snapshot | null => {
      const current = images.find((img) => img.id === imageId)
      return current ? { imageId, boxes: current.boxes, edited: current.edited } : null
    },
    [images],
  )

  const updateBoxes = useCallback(
    (boxes: Box[]) => {
      if (!selectedId) return
      const previous = snapshot(selectedId)
      if (previous) {
        setHistory((prev) => [...prev, previous].slice(-100))
        setFuture([])
      }
      setImages((prev) =>
        prev.map((img) => (img.id === selectedId ? { ...img, boxes, edited: true } : img)),
      )
    },
    [selectedId, snapshot],
  )

  const applySnapshot = useCallback((entry: Snapshot) => {
    setImages((prev) =>
      prev.map((img) =>
        img.id === entry.imageId ? { ...img, boxes: entry.boxes, edited: entry.edited } : img,
      ),
    )
    setSelectedId(entry.imageId)
  }, [])

  const undo = useCallback(() => {
    const entry = history[history.length - 1]
    if (!entry) return
    const current = snapshot(entry.imageId)
    if (current) setFuture((prev) => [...prev, current])
    setHistory((prev) => prev.slice(0, -1))
    applySnapshot(entry)
  }, [history, snapshot, applySnapshot])

  const redo = useCallback(() => {
    const entry = future[future.length - 1]
    if (!entry) return
    const current = snapshot(entry.imageId)
    if (current) setHistory((prev) => [...prev, current])
    setFuture((prev) => prev.slice(0, -1))
    applySnapshot(entry)
  }, [future, snapshot, applySnapshot])

  /** Bulk-fixes a class the model got wrong across the whole image. */
  const relabelClass = useCallback(
    (from: string, to: string) => {
      if (!selected) return
      updateBoxes(selected.boxes.map((b) => (b.className === from ? { ...b, className: to } : b)))
    },
    [selected, updateBoxes],
  )

  const deleteClass = useCallback(
    (name: string) => {
      if (!selected) return
      updateBoxes(selected.boxes.filter((b) => b.className !== name))
    },
    [selected, updateBoxes],
  )

  const deleteSelectedBox = useCallback(() => {
    if (!selected || !selectedBoxId) return
    updateBoxes(selected.boxes.filter((b) => b.id !== selectedBoxId))
    setSelectedBoxId(null)
  }, [selected, selectedBoxId, updateBoxes])

  // Keyboard shortcuts for fast review: image paging, class picking, undo.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) redo()
        else undo()
        return
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        redo()
        return
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        deleteSelectedBox()
      }
      if (e.key === 'Escape') setSelectedBoxId(null)
      if (e.key.toLowerCase() === 'v') setTool('select')
      if (e.key.toLowerCase() === 'd') setTool('draw')
      if (e.key.toLowerCase() === 'h') setTool('pan')

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
        if (!cls) return
        setActiveClass(cls)
        // With a box selected, the number retags it — correcting a wrong label
        // is the common case, and reaching for the dropdown each time is slow.
        if (selectedBoxId && selected) {
          updateBoxes(
            selected.boxes.map((b) => (b.id === selectedBoxId ? { ...b, className: cls } : b)),
          )
        }
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    images,
    selectedId,
    promptClasses,
    undo,
    redo,
    deleteSelectedBox,
    selectedBoxId,
    selected,
    updateBoxes,
  ])

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
    setHistory([])
    setFuture([])
    await clearImages()
  }

  async function detectImage(target: ImageState) {
    const rawBoxes = await detectObjects(target.file, promptGroups, RAW_CONFIDENCE, RAW_IOU, {
      imgsz: WHOLE_IMAGE_IMGSZ,
      tiled: shouldTile(target, tileMode),
      tileSize,
      tileOverlap: 0.25,
      // Running each tile at twice its pixel size upsamples it, which recovers
      // the last few parts at roughly 4x the runtime.
      tileImgsz: highRecall ? tileSize * 2 : 0,
      // Tiles cannot see anything bigger than one tile, so a board with a
      // large chip needs the whole-image pass merged in as well.
      multiscale: true,
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
          {images.length > 0 && (
            <button onClick={clearAll} className="text-[11px] text-gray-400 hover:text-red-600">
              clear all
            </button>
          )}
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

      {health?.error && (
        <p className="border-b border-red-200 bg-red-50 px-5 py-1.5 text-[11px] text-red-700">
          <strong>Model failed to load:</strong> {health.error}
        </p>
      )}

      <div className="flex min-h-0 flex-1">
        <div
          className="flex min-w-0 flex-1 flex-col"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
          }}
        >
          <Toolbar
            tool={tool}
            onToolChange={setTool}
            activeClass={activeClass || promptClasses[0] || ''}
            promptClasses={promptClasses}
            onActiveClassChange={setActiveClass}
            canUndo={history.length > 0}
            canRedo={future.length > 0}
            onUndo={undo}
            onRedo={redo}
            hasSelection={!!selectedBoxId}
            onDeleteSelected={deleteSelectedBox}
            disabled={!selected}
          />

          <ImageCanvas
            image={selected}
            labelDisplay={labelDisplay}
            opacity={opacity}
            activeClass={activeClass}
            promptClasses={promptClasses}
            hiddenClasses={hiddenClasses}
            tool={tool}
            selectedBoxId={selectedBoxId}
            onSelectBox={setSelectedBoxId}
            detecting={detecting}
            progressLabel={progressLabel}
            onBoxesChange={updateBoxes}
          />

          <Filmstrip
            images={images}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onAddFiles={addFiles}
            onRemove={removeImage}
          />
        </div>

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
          onRelabelClass={relabelClass}
          onDeleteClass={deleteClass}
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

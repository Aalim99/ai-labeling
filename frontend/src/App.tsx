import { useCallback, useEffect, useMemo, useState } from 'react'
import ControlsPanel from './components/ControlsPanel'
import ImageCanvas from './components/ImageCanvas'
import Sidebar from './components/Sidebar'
import { detectObjects, exportYolo } from './lib/api'
import { filterAndNms } from './lib/nms'
import type { Box, LabelDisplay, LabeledImage } from './types'

// Detections are fetched once at a permissive threshold; the sliders then
// filter that raw set client-side so they stay instant.
const RAW_CONFIDENCE = 0.01
const RAW_IOU = 0.9

interface ImageState extends LabeledImage {
  file: File
  rawBoxes: Box[]
}

function readFile(file: File): Promise<ImageState> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`))
    reader.onload = () => {
      const base64 = reader.result as string
      const img = new Image()
      img.onerror = () => reject(new Error(`Could not decode ${file.name}`))
      img.onload = () => {
        resolve({
          id: crypto.randomUUID(),
          name: file.name,
          url: base64,
          base64,
          width: img.naturalWidth,
          height: img.naturalHeight,
          boxes: [],
          rawBoxes: [],
          detected: false,
          edited: false,
          file,
        })
      }
      img.src = base64
    }
    reader.readAsDataURL(file)
  })
}

export default function App() {
  const [images, setImages] = useState<ImageState[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [promptText, setPromptText] = useState('capacitor, resistor, ic, connector')
  const [activeClass, setActiveClass] = useState('')
  const [confidence, setConfidence] = useState(50)
  const [overlap, setOverlap] = useState(50)
  const [opacity, setOpacity] = useState(35)
  const [labelDisplay, setLabelDisplay] = useState<LabelDisplay>('confidence')
  const [detecting, setDetecting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const promptClasses = useMemo(
    () =>
      promptText
        .split(',')
        .map((c) => c.trim())
        .filter(Boolean),
    [promptText],
  )

  const selected = images.find((img) => img.id === selectedId) ?? null

  // Re-apply the threshold sliders to raw detections for every image the user
  // hasn't hand-edited yet.
  useEffect(() => {
    setImages((prev) =>
      prev.map((img) => {
        if (img.edited || !img.detected) return img
        return { ...img, boxes: filterAndNms(img.rawBoxes, confidence / 100, overlap / 100) }
      }),
    )
  }, [confidence, overlap])

  async function addFiles(files: FileList) {
    setError(null)
    try {
      const loaded = await Promise.all(Array.from(files).map(readFile))
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

  const updateBoxes = useCallback(
    (boxes: Box[]) => {
      if (!selectedId) return
      setImages((prev) =>
        prev.map((img) => (img.id === selectedId ? { ...img, boxes, edited: true } : img)),
      )
    },
    [selectedId],
  )

  async function detectImage(target: ImageState) {
    const rawBoxes = await detectObjects(target.file, promptClasses, RAW_CONFIDENCE, RAW_IOU)
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
    try {
      for (const target of targets) {
        await detectImage(target)
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
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
      const blob = await exportYolo(images, usedClasses, 'labeled-dataset')
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

  return (
    <div className="flex h-full flex-col bg-gray-50">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-5 py-3">
        <div>
          <h1 className="text-sm font-semibold text-gray-900">AI Auto-Labeling</h1>
          <p className="text-[11px] text-gray-500">
            Prompt-driven object labeling · exports YOLO format
          </p>
        </div>
        <p className="text-[11px] text-gray-500">
          {images.length} image{images.length === 1 ? '' : 's'} ·{' '}
          {images.reduce((sum, img) => sum + img.boxes.length, 0)} labels
        </p>
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar
          images={images}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onAddFiles={addFiles}
          onRemove={removeImage}
        />

        <ImageCanvas
          image={selected}
          labelDisplay={labelDisplay}
          opacity={opacity}
          activeClass={activeClass}
          promptClasses={promptClasses}
          detecting={detecting}
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
          boxes={selected?.boxes ?? []}
          imageSize={selected ? { width: selected.width, height: selected.height } : null}
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

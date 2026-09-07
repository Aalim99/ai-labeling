import { useState } from 'react'
import { colorForClass } from '../lib/colors'
import type { Box, LabelDisplay } from '../types'

interface Props {
  promptText: string
  onPromptTextChange: (value: string) => void
  promptClasses: string[]
  activeClass: string
  onActiveClassChange: (value: string) => void
  confidence: number
  onConfidenceChange: (value: number) => void
  overlap: number
  onOverlapChange: (value: number) => void
  opacity: number
  onOpacityChange: (value: number) => void
  labelDisplay: LabelDisplay
  onLabelDisplayChange: (value: LabelDisplay) => void
  boxes: Box[]
  imageSize: { width: number; height: number } | null
  detecting: boolean
  exporting: boolean
  hasImages: boolean
  onDetect: () => void
  onDetectAll: () => void
  onExport: () => void
  error: string | null
}

function Slider({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <div className="mb-4">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-gray-700">{label}</span>
        <span className="text-gray-500">{value}%</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-purple-600"
      />
      <div className="flex justify-between text-[10px] text-gray-400">
        <span>0%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

export default function ControlsPanel(props: Props) {
  const [copied, setCopied] = useState(false)

  const outputJson = JSON.stringify(
    {
      image: props.imageSize,
      predictions: props.boxes.map((b) => ({
        x: Math.round(b.x),
        y: Math.round(b.y),
        width: Math.round(b.width),
        height: Math.round(b.height),
        confidence: b.confidence != null ? Number(b.confidence.toFixed(3)) : 1,
        class: b.className,
        class_id: props.promptClasses.indexOf(b.className),
        detection_id: b.id,
      })),
    },
    null,
    2,
  )

  async function copyOutput() {
    await navigator.clipboard.writeText(outputJson)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-gray-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-gray-900">What to label</h2>

      <textarea
        value={props.promptText}
        onChange={(e) => props.onPromptTextChange(e.target.value)}
        rows={3}
        placeholder="capacitor, resistor, ic, connector, transistor"
        className="mb-2 w-full resize-none rounded-md border border-gray-300 p-2 text-xs outline-none focus:border-purple-500"
      />

      {props.promptClasses.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1">
          {props.promptClasses.map((c) => (
            <button
              key={c}
              onClick={() => props.onActiveClassChange(c)}
              className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${
                props.activeClass === c ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-700'
              }`}
              title="Class used when you draw a box by hand"
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: colorForClass(c) }}
              />
              {c}
            </button>
          ))}
        </div>
      )}

      <div className="mb-4 flex gap-2">
        <button
          onClick={props.onDetect}
          disabled={props.detecting || !props.hasImages || props.promptClasses.length === 0}
          className="flex-1 rounded-md bg-purple-600 px-3 py-2 text-xs font-semibold text-white hover:bg-purple-700 disabled:opacity-40"
        >
          {props.detecting ? 'Detecting…' : 'Auto-label image'}
        </button>
        <button
          onClick={props.onDetectAll}
          disabled={props.detecting || !props.hasImages || props.promptClasses.length === 0}
          className="rounded-md border border-gray-300 px-3 py-2 text-xs font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-40"
        >
          All
        </button>
      </div>

      {props.error && (
        <p className="mb-3 rounded-md bg-red-50 p-2 text-[11px] text-red-700">{props.error}</p>
      )}

      <h2 className="mb-3 text-sm font-semibold text-gray-900">Model Visualizations</h2>
      <Slider
        label="Confidence Threshold"
        value={props.confidence}
        onChange={props.onConfidenceChange}
      />
      <Slider label="Overlap Threshold" value={props.overlap} onChange={props.onOverlapChange} />
      <Slider label="Opacity Threshold" value={props.opacity} onChange={props.onOpacityChange} />

      <p className="mb-4 text-[10px] leading-4 text-gray-400">
        Thresholds re-filter the last detection instantly. Once you edit boxes by hand, that image
        keeps your edits until you re-run detection.
      </p>

      <label className="mb-1 block text-xs font-medium text-gray-700">Label Display</label>
      <select
        value={props.labelDisplay}
        onChange={(e) => props.onLabelDisplayChange(e.target.value as LabelDisplay)}
        className="mb-4 w-full rounded-md border border-gray-300 p-2 text-xs outline-none focus:border-purple-500"
      >
        <option value="confidence">Draw Confidence</option>
        <option value="label">Draw Labels</option>
        <option value="none">Draw None</option>
      </select>

      <button
        onClick={props.onExport}
        disabled={props.exporting || !props.hasImages}
        className="mb-4 w-full rounded-md bg-gray-900 px-3 py-2 text-xs font-semibold text-white hover:bg-gray-800 disabled:opacity-40"
      >
        {props.exporting ? 'Building dataset…' : 'Export YOLO dataset (.zip)'}
      </button>

      <div className="mb-1 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">Output</h2>
        <button onClick={copyOutput} className="text-[11px] text-purple-600 hover:underline">
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre className="max-h-64 overflow-auto rounded-md bg-gray-50 p-2 font-mono text-[10px] leading-4 text-gray-700">
        {outputJson}
      </pre>
    </aside>
  )
}

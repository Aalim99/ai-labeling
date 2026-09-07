import { useMemo, useState } from 'react'
import type { HealthStatus } from '../lib/api'
import { colorForClass } from '../lib/colors'
import type { Box, LabelDisplay, TileMode } from '../types'

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
  tileMode: TileMode
  onTileModeChange: (value: TileMode) => void
  tileSize: number
  onTileSizeChange: (value: number) => void
  mergeClasses: boolean
  onMergeClassesChange: (value: boolean) => void
  highRecall: boolean
  onHighRecallChange: (value: boolean) => void
  boxes: Box[]
  rawCount: number
  hiddenClasses: Set<string>
  onToggleClass: (className: string) => void
  imageSize: { width: number; height: number } | null
  willTile: boolean
  health: HealthStatus | null
  detecting: boolean
  exporting: boolean
  hasImages: boolean
  onDetect: () => void
  onDetectAll: () => void
  onExport: () => void
  error: string | null
}

// Each class offers several phrasings (separated by |) because open-vocabulary
// models are very sensitive to wording — one phrasing can miss a part entirely
// that another finds confidently. Hits report under the first phrasing.
const PRESETS: Record<string, string> = {
  // Chip resistors and chip capacitors are the same black rectangle in a photo,
  // so on a dense board one honest class beats two guessed ones.
  'SMD board (coarse)':
    'chip | microchip | integrated circuit | black chip on circuit board, ' +
    'smd component | small rectangular chip component | surface mount part, ' +
    'connector | socket | port',
  'PCB components':
    'chip | microchip | integrated circuit | processor chip, ' +
    'capacitor | electrolytic capacitor | cylindrical capacitor, ' +
    'resistor | chip resistor | resistor array, ' +
    'connector | socket | header pins, ' +
    'transistor, inductor | coil, diode',
  'Through-hole board':
    'electrolytic capacitor | cylindrical capacitor can, ' +
    'resistor | axial resistor with colour bands, ' +
    'chip | integrated circuit | dip chip, ' +
    'connector | header pins | terminal block, transformer | coil, relay, fuse',
  Connectors:
    'usb port, hdmi port, ethernet jack | rj45 socket, ribbon connector, ' +
    'pin header, screw terminal',
}

function Slider({
  label,
  value,
  onChange,
  hint,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  hint?: string
}) {
  return (
    <div className="mb-3">
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
      {hint && <p className="text-[10px] text-gray-400">{hint}</p>}
    </div>
  )
}

export default function ControlsPanel(props: Props) {
  const [copied, setCopied] = useState(false)

  const classCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const box of props.boxes) counts.set(box.className, (counts.get(box.className) ?? 0) + 1)
    for (const name of props.promptClasses) if (!counts.has(name)) counts.set(name, 0)
    return [...counts.entries()].sort((a, b) => b[1] - a[1])
  }, [props.boxes, props.promptClasses])

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

  const modelBusy = !props.health || !props.health.ready
  const detectDisabled =
    props.detecting || !props.hasImages || props.promptClasses.length === 0 || modelBusy

  return (
    <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-gray-200 bg-white p-4">
      <h2 className="mb-2 text-sm font-semibold text-gray-900">What to label</h2>

      {props.health && !props.health.prompted ? (
        <p className="mb-3 rounded-md bg-blue-50 p-2 text-[11px] leading-4 text-blue-800">
          <span className="font-mono">{props.health.model}</span> is a trained model with fixed
          classes, so prompts don't apply. It detects: {props.health.classes.join(', ') || '—'}
        </p>
      ) : (
        <>
          <textarea
            value={props.promptText}
            onChange={(e) => props.onPromptTextChange(e.target.value)}
            rows={3}
            placeholder="chip | microchip, capacitor, resistor"
            className="mb-1 w-full resize-none rounded-md border border-gray-300 p-2 text-xs outline-none focus:border-purple-500"
          />

          <p className="mb-2 text-[10px] leading-4 text-gray-400">
            Commas separate classes. Within a class, <code className="font-mono">|</code> adds
            alternative wordings — the model is very sensitive to phrasing, and one wording often
            finds parts another misses entirely. Results are labelled with the first wording.
          </p>

          <div className="mb-2 flex flex-wrap gap-1">
            {Object.entries(PRESETS).map(([name, value]) => (
              <button
                key={name}
                onClick={() => props.onPromptTextChange(value)}
                className="rounded border border-gray-200 px-1.5 py-0.5 text-[10px] text-gray-600 hover:bg-gray-50"
              >
                {name}
              </button>
            ))}
          </div>
        </>
      )}

      {props.promptClasses.length > 0 && (
        <div className="mb-3 space-y-0.5">
          {classCounts.map(([name, count]) => {
            const hidden = props.hiddenClasses.has(name)
            return (
              <div key={name} className="flex items-center gap-1.5 text-[11px]">
                <button
                  onClick={() => props.onActiveClassChange(name)}
                  className={`flex flex-1 items-center gap-1.5 rounded px-1.5 py-0.5 text-left ${
                    props.activeClass === name ? 'bg-gray-900 text-white' : 'hover:bg-gray-100'
                  }`}
                  title="Class assigned to boxes you draw by hand"
                >
                  <span
                    className="inline-block h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: colorForClass(name) }}
                  />
                  <span className={`truncate ${hidden ? 'line-through opacity-50' : ''}`}>{name}</span>
                </button>
                <span className="w-6 text-right tabular-nums text-gray-500">{count}</span>
                <button
                  onClick={() => props.onToggleClass(name)}
                  className="w-4 text-center text-gray-400 hover:text-gray-700"
                  title={hidden ? 'Show class' : 'Hide class'}
                >
                  {hidden ? '○' : '●'}
                </button>
              </div>
            )
          })}
        </div>
      )}

      <div className="mb-3 flex gap-2">
        <button
          onClick={props.onDetect}
          disabled={detectDisabled}
          className="flex-1 rounded-md bg-purple-600 px-3 py-2 text-xs font-semibold text-white hover:bg-purple-700 disabled:opacity-40"
        >
          {props.detecting ? 'Detecting…' : 'Auto-label image'}
        </button>
        <button
          onClick={props.onDetectAll}
          disabled={detectDisabled}
          className="rounded-md border border-gray-300 px-3 py-2 text-xs font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-40"
        >
          All
        </button>
      </div>

      {props.error && (
        <p className="mb-3 rounded-md bg-red-50 p-2 text-[11px] leading-4 text-red-700">
          {props.error}
        </p>
      )}

      <h2 className="mb-2 text-sm font-semibold text-gray-900">Small object mode</h2>
      <select
        value={props.tileMode}
        onChange={(e) => props.onTileModeChange(e.target.value as TileMode)}
        className="mb-1 w-full rounded-md border border-gray-300 p-2 text-xs outline-none focus:border-purple-500"
      >
        <option value="auto">Auto (tile large images)</option>
        <option value="on">Always tile</option>
        <option value="off">Off (whole image)</option>
      </select>
      <p className="mb-2 text-[10px] leading-4 text-gray-400">
        Detection normally shrinks the image to {props.tileSize}px, which erases small parts. Tiling
        scans the board in overlapping crops at full resolution — much better recall, slower.
        {props.willTile ? ' Tiling is on for this image.' : ' Whole-image pass for this image.'}
      </p>

      <label className="mb-1 block text-xs font-medium text-gray-700">
        Tile size: {props.tileSize}px
      </label>
      <input
        type="range"
        min={320}
        max={1280}
        step={64}
        value={props.tileSize}
        onChange={(e) => props.onTileSizeChange(Number(e.target.value))}
        className="mb-1 w-full accent-purple-600"
      />
      <p className="mb-2 text-[10px] text-gray-400">
        Smaller tiles find smaller parts but take longer.
      </p>

      <label className="mb-4 flex cursor-pointer items-start gap-2 text-[11px] text-gray-700">
        <input
          type="checkbox"
          checked={props.highRecall}
          onChange={(e) => props.onHighRecallChange(e.target.checked)}
          className="mt-0.5 accent-purple-600"
        />
        <span>
          High recall (slower)
          <span className="block text-[10px] leading-4 text-gray-400">
            Runs each tile at double size. Recovered the last missing part in testing (19/20 to
            20/20) for about 4x the runtime.
          </span>
        </span>
      </label>

      <h2 className="mb-2 text-sm font-semibold text-gray-900">Model Visualizations</h2>
      <Slider
        label="Confidence Threshold"
        value={props.confidence}
        onChange={props.onConfidenceChange}
        hint={
          props.rawCount > props.boxes.length
            ? `${props.rawCount - props.boxes.length} more detections are below this threshold — lower it to see them.`
            : 'Lower it if parts are missing.'
        }
      />
      <Slider label="Overlap Threshold" value={props.overlap} onChange={props.onOverlapChange} />

      <label className="mb-3 flex cursor-pointer items-start gap-2 text-[11px] text-gray-700">
        <input
          type="checkbox"
          checked={props.mergeClasses}
          onChange={(e) => props.onMergeClassesChange(e.target.checked)}
          className="mt-0.5 accent-purple-600"
        />
        <span>
          One box per part
          <span className="block text-[10px] leading-4 text-gray-400">
            Drops duplicate boxes on the same component under different labels — a chip resistor
            and a chip capacitor look alike, so both often get predicted.
          </span>
        </span>
      </label>

      <Slider label="Opacity Threshold" value={props.opacity} onChange={props.onOpacityChange} />

      <p className="mb-3 text-[10px] leading-4 text-gray-400">
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
      <pre className="max-h-56 overflow-auto rounded-md bg-gray-50 p-2 font-mono text-[10px] leading-4 text-gray-700">
        {outputJson}
      </pre>
    </aside>
  )
}

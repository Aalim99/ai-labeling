import { colorForClass } from '../lib/colors'
import type { Tool } from '../types'

interface Props {
  tool: Tool
  onToolChange: (tool: Tool) => void
  activeClass: string
  promptClasses: string[]
  onActiveClassChange: (value: string) => void
  canUndo: boolean
  canRedo: boolean
  onUndo: () => void
  onRedo: () => void
  hasSelection: boolean
  onDeleteSelected: () => void
  disabled: boolean
}

const TOOLS: { id: Tool; label: string; key: string; hint: string }[] = [
  { id: 'select', label: 'Select', key: 'V', hint: 'Click to select · drag a box to move · drag empty space for a new box' },
  { id: 'draw', label: 'Draw', key: 'D', hint: 'Drag anywhere to add a box, even on top of another' },
  { id: 'pan', label: 'Pan', key: 'H', hint: 'Drag to move around the image' },
]

export default function Toolbar(props: Props) {
  const active = TOOLS.find((t) => t.id === props.tool)

  return (
    <div className="flex items-center gap-3 border-b border-gray-200 bg-white px-3 py-1.5">
      <div className="flex rounded-md border border-gray-300">
        {TOOLS.map((tool, index) => (
          <button
            key={tool.id}
            onClick={() => props.onToolChange(tool.id)}
            disabled={props.disabled}
            title={`${tool.hint}  (${tool.key})`}
            className={`px-2.5 py-1 text-xs font-medium disabled:opacity-40 ${
              props.tool === tool.id ? 'bg-gray-900 text-white' : 'text-gray-700 hover:bg-gray-50'
            } ${index === 0 ? 'rounded-l-md' : ''} ${
              index === TOOLS.length - 1 ? 'rounded-r-md' : 'border-r border-gray-300'
            }`}
          >
            {tool.label} <span className="opacity-50">{tool.key}</span>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-1.5">
        <label className="text-[11px] text-gray-500">New box:</label>
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: colorForClass(props.activeClass) }}
        />
        <select
          value={props.activeClass}
          onChange={(e) => props.onActiveClassChange(e.target.value)}
          disabled={props.disabled || props.promptClasses.length === 0}
          className="rounded-md border border-gray-300 px-1.5 py-1 text-xs outline-none focus:border-purple-500 disabled:opacity-40"
        >
          {props.promptClasses.map((name, index) => (
            <option key={name} value={name}>
              {index < 9 ? `${index + 1}. ` : ''}
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex gap-1">
        <button
          onClick={props.onUndo}
          disabled={!props.canUndo}
          title="Undo (Ctrl+Z)"
          className="rounded-md border border-gray-300 px-2 py-1 text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-30"
        >
          ↶ Undo
        </button>
        <button
          onClick={props.onRedo}
          disabled={!props.canRedo}
          title="Redo (Ctrl+Shift+Z)"
          className="rounded-md border border-gray-300 px-2 py-1 text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-30"
        >
          ↷ Redo
        </button>
        <button
          onClick={props.onDeleteSelected}
          disabled={!props.hasSelection}
          title="Delete selected box (Del)"
          className="rounded-md border border-gray-300 px-2 py-1 text-xs text-red-600 hover:bg-red-50 disabled:opacity-30"
        >
          Delete box
        </button>
      </div>

      <p className="ml-auto hidden truncate text-[11px] text-gray-400 lg:block">{active?.hint}</p>
    </div>
  )
}

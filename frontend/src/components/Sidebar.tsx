import { useRef } from 'react'
import type { LabeledImage } from '../types'

interface Props {
  images: LabeledImage[]
  selectedId: string | null
  onSelect: (id: string) => void
  onAddFiles: (files: FileList) => void
  onRemove: (id: string) => void
  onClearAll: () => void
}

export default function Sidebar({
  images,
  selectedId,
  onSelect,
  onAddFiles,
  onRemove,
  onClearAll,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-gray-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">Images</h2>
        {images.length > 0 && (
          <button onClick={onClearAll} className="text-[10px] text-gray-400 hover:text-red-600">
            clear all
          </button>
        )}
      </div>

      {images.length > 0 && (
        <div className="mb-3 min-h-0 flex-1 space-y-1 overflow-y-auto">
          {images.map((img) => (
            <div
              key={img.id}
              onClick={() => onSelect(img.id)}
              className={`group flex cursor-pointer items-center gap-2 rounded-md border p-1 ${
                img.id === selectedId
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-transparent hover:bg-gray-50'
              }`}
            >
              <img
                src={img.url}
                alt={img.name}
                className="h-9 w-9 shrink-0 rounded object-cover"
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-[11px] text-gray-700">{img.name}</p>
                <p className="text-[10px] text-gray-400">
                  {img.detected || img.edited ? `${img.boxes.length} labels` : 'not labeled'}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove(img.id)
                }}
                className="hidden px-1 text-xs text-gray-400 hover:text-red-600 group-hover:block"
                aria-label={`Remove ${img.name}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div
        className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 p-4 text-center hover:border-purple-400"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          if (e.dataTransfer.files.length) onAddFiles(e.dataTransfer.files)
        }}
      >
        <p className="mb-2 text-[11px] text-gray-500">Drop images here</p>
        <button
          onClick={() => inputRef.current?.click()}
          className="rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
        >
          + Select Files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) onAddFiles(e.target.files)
            e.target.value = ''
          }}
        />
      </div>
    </aside>
  )
}

import { useRef } from 'react'
import type { LabeledImage } from '../types'

interface Props {
  images: LabeledImage[]
  selectedId: string | null
  onSelect: (id: string) => void
  onAddFiles: (files: FileList) => void
  onRemove: (id: string) => void
}

export default function Sidebar({ images, selectedId, onSelect, onAddFiles, onRemove }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-gray-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-gray-900">Images</h2>

      {images.length > 0 && (
        <div className="mb-4 grid grid-cols-2 gap-2 overflow-y-auto">
          {images.map((img) => (
            <div
              key={img.id}
              className={`group relative cursor-pointer overflow-hidden rounded-md border-2 ${
                img.id === selectedId ? 'border-purple-500' : 'border-transparent hover:border-gray-300'
              }`}
              onClick={() => onSelect(img.id)}
            >
              <img src={img.url} alt={img.name} className="aspect-square w-full object-cover" />
              {img.detected && (
                <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px] text-white">
                  {img.boxes.length}
                </span>
              )}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove(img.id)
                }}
                className="absolute right-1 top-1 hidden h-4 w-4 items-center justify-center rounded-full bg-black/60 text-[10px] leading-none text-white group-hover:flex"
                aria-label={`Remove ${img.name}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div
        className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 p-6 text-center hover:border-purple-400"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          if (e.dataTransfer.files.length) onAddFiles(e.dataTransfer.files)
        }}
      >
        <p className="mb-2 text-xs text-gray-500">Drop images here</p>
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

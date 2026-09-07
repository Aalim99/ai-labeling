import { useEffect, useRef } from 'react'
import type { LabeledImage } from '../types'

interface Props {
  images: LabeledImage[]
  selectedId: string | null
  onSelect: (id: string) => void
  onAddFiles: (files: FileList) => void
  onRemove: (id: string) => void
}

export default function Filmstrip({ images, selectedId, onSelect, onAddFiles, onRemove }: Props) {
  const filesRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)
  const stripRef = useRef<HTMLDivElement>(null)

  const index = images.findIndex((img) => img.id === selectedId)

  // webkitdirectory isn't in the JSX types, so set it on the element itself.
  useEffect(() => {
    folderRef.current?.setAttribute('webkitdirectory', '')
  }, [])

  // Keep the current image visible while paging with the keyboard.
  useEffect(() => {
    const strip = stripRef.current
    if (!strip || index < 0) return
    strip.querySelector(`[data-id="${selectedId}"]`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest',
      inline: 'center',
    })
  }, [selectedId, index])

  function step(delta: number) {
    const next = images[index + delta]
    if (next) onSelect(next.id)
  }

  return (
    <div className="flex items-center gap-2 border-t border-gray-200 bg-white px-2 py-2">
      <button
        onClick={() => step(-1)}
        disabled={index <= 0}
        title="Previous image (←)"
        className="h-14 shrink-0 rounded-md border border-gray-300 px-2 text-gray-600 hover:bg-gray-50 disabled:opacity-30"
      >
        ‹
      </button>

      <div ref={stripRef} className="flex min-w-0 flex-1 gap-2 overflow-x-auto">
        {images.map((img, i) => (
          <button
            key={img.id}
            data-id={img.id}
            onClick={() => onSelect(img.id)}
            title={img.name}
            className={`group relative h-14 w-20 shrink-0 overflow-hidden rounded-md border-2 ${
              img.id === selectedId ? 'border-purple-500' : 'border-gray-200 hover:border-gray-400'
            }`}
          >
            <img src={img.url} alt={img.name} className="h-full w-full object-cover" />

            <span className="absolute left-0 top-0 bg-black/60 px-1 text-[9px] leading-4 text-white">
              {i + 1}
            </span>

            {(img.detected || img.edited) && (
              <span className="absolute bottom-0 left-0 right-0 bg-black/60 text-[9px] leading-4 text-white">
                {img.boxes.length} {img.edited ? '✓' : ''}
              </span>
            )}

            <span
              onClick={(e) => {
                e.stopPropagation()
                onRemove(img.id)
              }}
              className="absolute right-0 top-0 hidden bg-black/70 px-1 text-[10px] leading-4 text-white hover:bg-red-600 group-hover:block"
            >
              ×
            </span>
          </button>
        ))}

        <div className="flex shrink-0 flex-col gap-1">
          <button
            onClick={() => filesRef.current?.click()}
            className="h-6 w-20 rounded border border-dashed border-gray-300 text-[10px] text-gray-500 hover:border-purple-400 hover:text-purple-600"
          >
            + Files
          </button>
          <button
            onClick={() => folderRef.current?.click()}
            className="h-7 w-20 rounded border border-dashed border-gray-300 text-[10px] text-gray-500 hover:border-purple-400 hover:text-purple-600"
          >
            + Folder
          </button>
        </div>
      </div>

      <span className="shrink-0 px-1 font-mono text-[11px] text-gray-500">
        {images.length ? `${index + 1} / ${images.length}` : '0'}
      </span>

      <button
        onClick={() => step(1)}
        disabled={index < 0 || index >= images.length - 1}
        title="Next image (→)"
        className="h-14 shrink-0 rounded-md border border-gray-300 px-2 text-gray-600 hover:bg-gray-50 disabled:opacity-30"
      >
        ›
      </button>

      <input
        ref={filesRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) onAddFiles(e.target.files)
          e.target.value = ''
        }}
      />
      <input
        ref={folderRef}
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
  )
}

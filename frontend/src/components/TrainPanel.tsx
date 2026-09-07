import { useEffect, useState } from 'react'
import {
  cancelTraining,
  fetchModels,
  fetchTrainStatus,
  selectModel,
  startTraining,
  type ModelList,
  type TrainStatus,
} from '../lib/api'
import type { LabeledImage } from '../types'

interface Props {
  images: LabeledImage[]
  classes: string[]
  activeModel: string
  /** Resolves each image to a data URL, done lazily to keep memory down. */
  toBase64: (image: LabeledImage) => Promise<string>
  onModelChanged: () => void
}

const RUNNING = new Set(['preparing', 'training'])

export default function TrainPanel(props: Props) {
  const [status, setStatus] = useState<TrainStatus | null>(null)
  const [models, setModels] = useState<ModelList | null>(null)
  const [epochs, setEpochs] = useState(60)
  const [baseModel, setBaseModel] = useState('yolo11s.pt')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const labelled = props.images.filter((img) => img.boxes.length > 0)
  const boxCount = labelled.reduce((sum, img) => sum + img.boxes.length, 0)
  const running = !!status && RUNNING.has(status.state)

  const refreshModels = () => fetchModels().then(setModels).catch(() => {})

  useEffect(() => {
    refreshModels()
  }, [props.activeModel])

  // Poll quickly while a run is in flight, lazily otherwise.
  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const next = await fetchTrainStatus()
        if (!cancelled) setStatus(next)
      } catch {
        /* backend down; the header pill already says so */
      }
    }
    tick()
    const id = setInterval(tick, running ? 2000 : 8000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [running])

  async function handleTrain() {
    setBusy(true)
    setError(null)
    try {
      const payload = []
      for (const img of labelled) {
        payload.push({
          filename: img.name,
          image_base64: await props.toBase64(img),
          width: img.width,
          height: img.height,
          boxes: img.boxes.map((b) => ({
            x: b.x,
            y: b.y,
            width: b.width,
            height: b.height,
            class_name: b.className,
          })),
        })
      }
      const usedClasses = [
        ...new Set(labelled.flatMap((img) => img.boxes.map((b) => b.className))),
      ]
      setStatus(await startTraining(payload, usedClasses, epochs, baseModel))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function applyModel(path: string) {
    setBusy(true)
    setError(null)
    try {
      await selectModel(path)
      props.onModelChanged()
      await refreshModels()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const progress = status?.total_epochs
    ? Math.round((status.epoch / status.total_epochs) * 100)
    : 0
  const mAP = status?.metrics?.mAP50

  return (
    <div className="mb-4 rounded-md border border-gray-200 p-2">
      <h2 className="mb-1 text-sm font-semibold text-gray-900">Train on your labels</h2>
      <p className="mb-2 text-[10px] leading-4 text-gray-400">
        Prompting can only guess what a class means. Training teaches the model your actual
        convention, which is the only thing that fixes look-alike parts being confused.
      </p>

      <p className="mb-2 text-[11px] text-gray-600">
        {labelled.length} labelled image{labelled.length === 1 ? '' : 's'} · {boxCount} boxes ·{' '}
        {props.classes.length} classes
        {labelled.length > 0 && labelled.length < 20 && (
          <span className="block text-[10px] text-amber-600">
            Few images — expect a weak model. 50+ per class is a realistic target.
          </span>
        )}
      </p>

      {running ? (
        <div className="mb-2">
          <div className="mb-1 flex items-center justify-between text-[11px] text-gray-600">
            <span>
              {status?.state === 'preparing'
                ? 'Preparing dataset…'
                : `Epoch ${status?.epoch}/${status?.total_epochs}`}
            </span>
            {mAP != null && <span className="font-mono">mAP50 {(mAP * 100).toFixed(1)}%</span>}
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded bg-gray-200">
            <div className="h-full bg-purple-600 transition-all" style={{ width: `${progress}%` }} />
          </div>
          <button
            onClick={() => cancelTraining().then(setStatus)}
            className="mt-1 text-[10px] text-gray-500 hover:text-red-600"
          >
            stop after this epoch
          </button>
        </div>
      ) : (
        <>
          <div className="mb-2 flex gap-1">
            <label className="flex-1 text-[10px] text-gray-500">
              Epochs
              <input
                type="number"
                min={1}
                max={600}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-1 py-0.5 text-[11px] text-gray-800"
              />
            </label>
            <label className="flex-1 text-[10px] text-gray-500">
              Start from
              <select
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                className="w-full rounded border border-gray-300 px-1 py-0.5 text-[11px] text-gray-800"
              >
                {(models?.base ?? ['yolo11s.pt']).map((m) => (
                  <option key={m} value={m}>
                    {m.replace('.pt', '')}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <button
            onClick={handleTrain}
            disabled={busy || labelled.length === 0}
            className="mb-2 w-full rounded-md bg-emerald-600 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-700 disabled:opacity-40"
          >
            {busy ? 'Starting…' : `Train on ${labelled.length} image${labelled.length === 1 ? '' : 's'}`}
          </button>
        </>
      )}

      {status && !running && status.state !== 'idle' && (
        <p
          className={`mb-2 rounded p-1.5 text-[10px] leading-4 ${
            status.state === 'error' ? 'bg-red-50 text-red-700' : 'bg-gray-50 text-gray-600'
          }`}
        >
          {status.error || status.message}
          {status.weights && status.state !== 'error' && (
            <button
              onClick={() => applyModel(status.weights!)}
              className="mt-1 block font-semibold text-purple-600 hover:underline"
            >
              → Use this model for labeling
            </button>
          )}
        </p>
      )}

      {error && <p className="mb-2 text-[10px] text-red-600">{error}</p>}

      <label className="block text-[10px] text-gray-500">
        Labeling model
        <select
          value={props.activeModel}
          disabled={busy}
          onChange={(e) => applyModel(e.target.value)}
          className="mt-0.5 w-full rounded border border-gray-300 px-1 py-1 text-[11px] text-gray-800"
        >
          <optgroup label="Prompt-driven">
            {(models?.prompted ?? []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </optgroup>
          {models?.trained?.length ? (
            <optgroup label="Trained by you">
              {models.trained.map((m) => (
                <option key={m.path} value={m.path}>
                  {m.name}
                </option>
              ))}
            </optgroup>
          ) : null}
          {models && !models.prompted.includes(props.activeModel) &&
            !models.trained.some((m) => m.path === props.activeModel) && (
              <option value={props.activeModel}>{props.activeModel}</option>
            )}
        </select>
      </label>
    </div>
  )
}

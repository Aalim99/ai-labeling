import type { Box } from '../types'

const DB_NAME = 'ai-labeling'
const STORE = 'session'
const KEY = 'images'

export interface StoredImage {
  id: string
  name: string
  width: number
  height: number
  boxes: Box[]
  rawBoxes: Box[]
  detected: boolean
  edited: boolean
  file: File
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1)
    request.onupgradeneeded = () => request.result.createObjectStore(STORE)
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function withStore<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb()
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(STORE, mode)
    const request = fn(tx.objectStore(STORE))
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
    tx.oncomplete = () => db.close()
  })
}

/** Annotations are real work; losing them to a page refresh is unacceptable. */
export async function saveImages(images: StoredImage[]): Promise<void> {
  try {
    await withStore('readwrite', (store) => store.put(images, KEY))
  } catch {
    // Storage is a convenience, never a hard failure.
  }
}

export async function loadImages(): Promise<StoredImage[]> {
  try {
    return (await withStore('readonly', (store) => store.get(KEY))) ?? []
  } catch {
    return []
  }
}

export async function clearImages(): Promise<void> {
  try {
    await withStore('readwrite', (store) => store.delete(KEY))
  } catch {
    // ignore
  }
}

const SETTINGS_KEY = 'ai-labeling:settings'

export function loadSettings<T>(fallback: T): T {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY)
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback
  } catch {
    return fallback
  }
}

export function saveSettings(settings: unknown): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
  } catch {
    // ignore
  }
}

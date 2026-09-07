# AI Auto-Labeling

Prompt-driven image labeling for computer vision datasets. Type the objects you want
(`capacitor, resistor, integrated circuit chip`), auto-detect them, correct the boxes by hand, and
export a YOLO-format dataset you can train on directly.

The detector is open-vocabulary, so it is not limited to electronics — any class you can name in
the prompt box works. No training data is needed to start labeling.

## How it works

- **Backend** — FastAPI + Ultralytics. Three model types are supported (see
  [Choosing a model](#choosing-a-model)): [YOLOE](https://docs.ultralytics.com/models/yoloe/)
  (default) and YOLO-World take text prompts; any custom trained `.pt` uses its own classes.
- **Frontend** — React + Vite + Tailwind. Three panels: image list, annotation canvas, controls.
- **Export** — images + `labels/*.txt` in YOLO format plus a `data.yaml`, zipped for download.

Detection runs once per image at a permissive threshold; the confidence and overlap sliders then
re-filter that result client-side (confidence cutoff + per-class NMS), so they respond instantly
without re-running the model.

### Small object mode (tiling)

Detection normally shrinks the whole image to ~640px, which erases small parts — a 0402 resistor in
a 4000px board photo becomes a few pixels. Tiling scans the image in overlapping crops at full
resolution instead, and merges the results.

Measured on a 4000×3000 test image with 20 small objects (3% of image height):

| mode | objects found |
| --- | --- |
| whole image @640 | 0 / 20 |
| whole image @1280 | 0 / 20 |
| tiled 640px | **20 / 20** |

`Auto` (the default) turns tiling on for images large enough to need it. It is slower — cost grows
with tile count — so smaller tiles find smaller parts but take longer.

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Weights (~70 MB detector + ~600 MB text encoder) download on first startup, not on the first
request, so the UI shows `loading model…` instead of hanging. Watch the terminal for progress.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to port 8000.

If the backend runs elsewhere, either change the proxy target in `frontend/vite.config.ts` or set
`VITE_API_BASE=http://localhost:8010/api` when starting the frontend.

## Using it

1. Drop images into the left panel.
2. Type the classes you want in **What to label**, comma-separated (or use a preset).
3. **Auto-label image** labels the selected image; **All** runs the whole batch.
4. Correct the results on the canvas:
   - drag a box to move it, drag any corner handle to resize
   - click a box to select it, then relabel from the dropdown or delete it
   - drag on empty image area to draw a new box (uses the highlighted class)
   - shift-drag to draw a box *on top of* an existing one — on a dense board almost every pixel is
     already inside some box, so a plain drag there would move it instead
   - scroll to zoom, space-drag or middle-drag to pan
5. Tune **Confidence** and **Overlap** to filter detections; **Opacity** controls box fill.
6. **Export YOLO dataset (.zip)**.

Once you hand-edit an image, the threshold sliders stop rewriting its boxes so your corrections are
not lost. Re-running detection on that image resets it.

Images and annotations are saved in the browser, so a refresh doesn't lose your work.

### Shortcuts

| key | action |
| --- | --- |
| `←` / `→` | previous / next image |
| `1`–`9` | pick the class for new boxes |
| `Delete` | delete the selected box |
| `Ctrl`/`Cmd` + `Z` | undo |
| scroll | zoom · `+` / `-` / `0` (fit) |
| space + drag | pan |
| shift + drag | draw a box over an existing one |

## Choosing a model

Set `LABELING_MODEL` before starting the backend:

```bash
LABELING_MODEL=yoloe-11s-seg.pt uvicorn app.main:app --port 8000    # smaller/faster
LABELING_MODEL=yolov8s-worldv2.pt uvicorn app.main:app --port 8000  # YOLO-World
LABELING_MODEL=runs/detect/train/weights/best.pt uvicorn app.main:app --port 8000  # your own
```

- **`yoloe-*`** (default) — open-vocabulary, prompt-driven.
- **`*-world*`** — YOLO-World, also prompt-driven. Worth comparing on your images; it needs
  OpenAI's CLIP weights, which download from a different host than the Ultralytics assets.
- **anything else** — treated as a trained model with fixed classes. The prompt box is disabled and
  the UI shows the classes it was trained on.

`LABELING_MAX_DET` (default 1000) caps detections per pass; raise it for very dense boards.

## Getting good accuracy on electronics

Open-vocabulary detection is a way to *start* labeling without any data, not a finished detector
for a niche domain. Small SMD parts are genuinely hard for it. What helps, in order:

1. **Turn on tiling** — by far the biggest factor for small parts (table above).
2. **Use concrete nouns.** `electrolytic capacitor` and `integrated circuit chip` work better than
   `cap` or `ic`. Try several phrasings; open-vocabulary models are sensitive to wording.
3. **Lower the confidence threshold.** Niche classes score lower than everyday objects. The panel
   tells you how many detections are hidden below the current threshold.
4. **Close the loop.** Label a batch (auto-label, then correct), export, train a YOLO model on it,
   then point `LABELING_MODEL` at your `best.pt` to pre-label the next batch far more accurately.
   That trained-model loop — not prompting — is what gets you to production accuracy on a fixed
   class set.

```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=100 imgsz=640
```

## Export format

```
data.yaml
images/train/photo.jpg
images/val/other.jpg
labels/train/photo.txt      # class_id cx cy w h   (normalized 0-1)
labels/val/other.txt
```

An 80/20 train/val split is applied automatically; with a single image everything goes to `train`
and `data.yaml` points `val` at it. Boxes dragged past the image edge are clipped, and
fully-outside boxes are dropped, so coordinates are always valid.

## API

| Endpoint | Method | Body | Returns |
| --- | --- | --- | --- |
| `/api/health` | GET | — | model name/engine, readiness, device, fixed classes |
| `/api/progress` | GET | — | tile progress of the running detection |
| `/api/detect` | POST | multipart: `image`, `prompts`, `confidence`, `iou`, `imgsz`, `tiled`, `tile_size`, `tile_overlap` | image size + predictions (`x`, `y` = box center, pixels) |
| `/api/export` | POST | JSON: `images[]` (base64 + boxes), `classes[]`, `dataset_name` | dataset `.zip` |

## Troubleshooting

**"Cannot reach the backend"** — the backend isn't running, or is on a different port than the
proxy expects. Start it and check the header pill turns green.

**`WinError 10013` on Windows** — the port is blocked, usually by a Hyper-V/WSL reserved range.
Check with `netsh interface ipv4 show excludedportrange protocol=tcp`, then use a free port
(`--port 8010`) and point the frontend at it, or run `net stop winnat && net start winnat` in an
elevated shell.

**numpy fails to build during `pip install`** — your Python is newer than the pinned wheel. Use
Python 3.11/3.12, or upgrade pip first (`python -m pip install --upgrade pip setuptools wheel`).

**Detection is slow** — it runs on CPU unless a CUDA GPU is available (the header pill shows
which). Tiling multiplies the work by the tile count; raise the tile size or turn it off for small
images.

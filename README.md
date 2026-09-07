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

What decides whether a small part is found is **how big it ends up in the model's input tensor**.
Two regimes, both measured on synthetic scenes with 20 known objects:

**A big board scan** (4000×3000, parts ~45px) has to be downscaled to fit the model, which erases
the parts. Tiling scans it in overlapping crops at full resolution instead:

| mode | found | confident (≥25%) | time |
| --- | --- | --- | --- |
| whole image @1280 | 14 / 20 | 0 | 10s |
| whole image @2560 | 18 / 20 | 0 | 14s |
| **tiled 640px** | **20 / 20** | 14 | 28s |
| tiled 640px, high recall | 20 / 20 | **20** | 106s |

**A macro shot** (834×500, parts ~25px) is the opposite case — there is nothing to slice, and
tiling at native size makes things *worse*. Upsampling the whole image is what helps:

| input size | found | confident (≥25%) |
| --- | --- | --- |
| whole image @640 | 19 / 20 | 0 |
| whole image @864 | 20 / 20 | 0 |
| **whole image @1280** | 20 / 20 | **19** |
| whole image @1536 | 20 / 20 | 0 |
| whole image @1920 | 20 / 20 | 0 |
| tiled 640px | 3 / 20 | 0 |

Note what changes in that second table: the objects are found at nearly every size — only at 1280
do they score high enough to survive a normal confidence threshold. **A detection you can't see
because it scored 8% looks exactly like no detection at all**, which is why the confidence slider
defaults low.

So `Auto` runs a whole-image pass at 1280 and only tiles when the image is larger than that.
**High recall** runs each tile at double size: in testing it turned every detection confident, for
about 4× the runtime.

## Setup

### Backend

**Windows** (PowerShell or Command Prompt — `source` is a Unix command and won't work here):

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

**macOS / Linux:**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Weights (~70 MB detector + ~600 MB text encoder) download on first startup, not on the first
request, so the UI shows `loading model…` instead of hanging. Watch the terminal for progress.

Git is **not** required. Ultralytics would otherwise try to install its CLIP fork from GitHub;
`requirements.txt` pulls the same module from PyPI (`openai-clip`) instead.

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

1. Add images with **+ Files** or **+ Folder** in the strip along the bottom, or drop them onto the
   canvas. A whole folder imports at once.
2. Type the classes you want in **What to label**, comma-separated (or use a preset).
3. **Auto-label image** labels the current image; **All** runs the whole batch.
4. Correct the results with the toolbar above the canvas:

   | tool | what a drag does |
   | --- | --- |
   | **Select** (`V`) | drag a box to move it, drag a corner handle to resize, click to select |
   | **Draw** (`D`) | draw a new box anywhere, including on top of an existing one |
   | **Pan** (`H`) | move around the image |

   With a box selected you can relabel it from the dropdown on the box, delete it with
   **Delete box** or the `Delete` key, and step back with **Undo**. New boxes take the class shown
   in **New box** (or press `1`–`9`).
5. Page through a bundle with the filmstrip arrows or `←` / `→`. Each thumbnail shows its label
   count, and a `✓` once you have hand-edited it.
6. Tune **Confidence** and **Overlap** to filter detections; **Opacity** controls box fill.
7. **Export YOLO dataset (.zip)**.

Once you hand-edit an image, the threshold sliders stop rewriting its boxes so your corrections are
not lost. Re-running detection on that image resets it.

Images and annotations are saved in the browser, so a refresh doesn't lose your work.

### Shortcuts

| key | action |
| --- | --- |
| `V` / `D` / `H` | select / draw / pan tool |
| `←` / `→` | previous / next image |
| `1`–`9` | pick the class for new boxes |
| `Delete` | delete the selected box |
| `Esc` | deselect |
| `Ctrl`/`Cmd` + `Z` | undo · add `Shift` to redo |
| scroll | zoom · `+` / `-` / `0` (fit) |
| space + drag | pan, whatever tool is active |
| shift + drag | draw a box, whatever tool is active |

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

## Checking one image from the command line

To see what the model finds without going through the UI:

```bash
cd backend
python scripts/try_image.py board.jpg --classes "capacitor, resistor, integrated circuit chip"
```

(On Windows, activate the venv first with `.venv\Scripts\activate`.)

It prints detection counts grouped by confidence band and writes `board_detected.png` with the
boxes drawn on. Useful for comparing prompts or tile settings quickly:

```bash
python scripts/try_image.py board.jpg --tiled --tile-size 512
python scripts/try_image.py board.jpg --tiled --tile-imgsz 1280   # high recall
```

## Getting good accuracy on electronics

Open-vocabulary detection is a way to *start* labeling without any data, not a finished detector
for a niche domain. Small SMD parts are genuinely hard for it. What helps, in order:

1. **Lower the confidence threshold.** This is the first thing to try, not the last. Niche classes
   score far lower than everyday objects, and the measurements above show parts are often detected
   at 5–20% confidence. The panel tells you how many detections are hidden below the slider.
2. **Match the mode to the image** — `Auto` handles it, but if parts are missing on a big scan turn
   tiling on, and on a macro shot turn it off (table above).
3. **Use concrete nouns.** `electrolytic capacitor` and `integrated circuit chip` work better than
   `cap` or `ic`. Try several phrasings; open-vocabulary models are sensitive to wording.
4. **Keep "one box per part" on** unless you deliberately want overlapping classes. A chip resistor
   and a chip capacitor are the same black rectangle, so both labels get predicted for one part.
5. **Close the loop.** Label a batch (auto-label, then correct), export, train a YOLO model on it,
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
| `/api/detect` | POST | multipart: `image`, `prompts`, `confidence`, `iou`, `imgsz`, `tiled`, `tile_size`, `tile_overlap`, `tile_imgsz` | image size + predictions (`x`, `y` = box center, pixels) |
| `/api/export` | POST | JSON: `images[]` (base64 + boxes), `classes[]`, `dataset_name` | dataset `.zip` |

## Troubleshooting

**"Cannot reach the backend"** — the backend isn't running, or is on a different port than the
proxy expects. Start it and check the header pill turns green.

**`WinError 10013` on Windows** — the port is blocked, usually by a Hyper-V/WSL reserved range.
Check with `netsh interface ipv4 show excludedportrange protocol=tcp`, then use a free port
(`--port 8010`) and point the frontend at it, or run `net stop winnat && net start winnat` in an
elevated shell.

**`ModuleNotFoundError: No module named 'clip'`**, or pip failing on
`git+https://github.com/ultralytics/CLIP.git` with `Cannot find command 'git'` — Ultralytics is
trying to fetch its CLIP fork from GitHub. Install the PyPI build instead, no git needed:
`pip install openai-clip`. It is already in `requirements.txt`, so re-running
`pip install -r requirements.txt` also fixes it.

**`'source' is not recognized`** — that is a Unix command. On Windows activate the venv with
`.venv\Scripts\activate` (see the Windows block above).

**numpy fails to build during `pip install`** — your Python is newer than the pinned wheel. Use
Python 3.11/3.12, or upgrade pip first (`python -m pip install --upgrade pip setuptools wheel`).

**Nothing is detected** — drag the confidence threshold down to 0 first. Parts on a niche class
often score under 10%, and the panel reports how many are hidden. If they appear, raise the
threshold until the false positives start outnumbering the real ones.

**Detection is slow** — it runs on CPU unless a CUDA GPU is available (the header pill shows
which). Tiling multiplies the work by the tile count; raise the tile size or turn it off for small
images.

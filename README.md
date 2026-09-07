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
- **Training** — fine-tunes a YOLO detector on the labels you corrected, then uses it for the next
  batch. See [Training on your own labels](#training-on-your-own-labels).

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
2. Type the classes you want in **What to label**, comma-separated (or use a preset). Within a
   class, `|` adds alternative wordings — `chip | microchip | integrated circuit` tries all three
   and labels every hit `chip`. Worth doing: wording changes results dramatically.
3. **Auto-label image** labels the current image; **All** runs the whole batch.
4. Correct the results with the toolbar above the canvas:

   | tool | what a drag does |
   | --- | --- |
   | **Select** (`V`) | drag a box to move it, drag a corner handle to resize, click to select |
   | **Draw** (`D`) | draw a new box anywhere, including on top of an existing one |
   | **Pan** (`H`) | move around the image |

   With a box selected, press `1`–`9` to retag it, or use the dropdown on the box; delete it with
   **Delete box** or the `Delete` key, and step back with **Undo**. New boxes take the class shown
   in **New box**.

   To fix a class the model got wrong across the whole image, use the `⋯` menu next to it in the
   class list: **→ other class** retags every box of that class at once, **✕ delete all** removes
   them. Both are undoable. This is usually much faster than correcting boxes one at a time.
5. Page through a bundle with the filmstrip arrows or `←` / `→`. Each thumbnail shows its label
   count, and a `✓` once you have hand-edited it.
6. Tune **Confidence** and **Overlap** to filter detections; **Opacity** controls box fill.
7. **Export YOLO dataset (.zip)**, and/or **train on your labels** (below).

**All** deliberately skips images you have hand-edited, so a batch run cannot overwrite
corrections you already made. Re-running detection on a single image does replace its boxes.

Once you hand-edit an image, the threshold sliders stop rewriting its boxes so your corrections are
not lost. Re-running detection on that image resets it.

Images and annotations are saved in the browser, so a refresh doesn't lose your work.

### Shortcuts

| key | action |
| --- | --- |
| `V` / `D` / `H` | select / draw / pan tool |
| `←` / `→` | previous / next image |
| `1`–`9` | retag the selected box · also sets the class for new boxes |
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

You rarely need this variable: the **Labeling model** dropdown switches models at runtime, and a
model you train in-app can be adopted with one click. `LABELING_MODEL` just sets the startup
default.

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

## Training on your own labels

Prompting can only ever guess what a class means. Training on corrected labels is what teaches a
model your actual convention — which rectangle on *your* boards counts as a resistor rather than a
capacitor. That is the only thing that fixes look-alike parts being confused, and it happens
in-app:

1. Label a batch (auto-label, then correct — bulk relabel makes this quick).
2. In **Train on your labels**, set epochs and the base model, then press **Train**.
3. Watch epoch and mAP50 progress. **Stop after this epoch** ends it early but keeps the weights.
4. Press **Use this model for labeling**. The prompt box disappears, replaced by the classes the
   model was trained on.
5. Label the next batch — now pre-labelled far better — correct it, and train again on everything.

Each round should need less correcting. The dataset and weights go under `backend/training/`
(`dataset/` is rewritten each run, `runs/<timestamp>/weights/best.pt` is kept), and you can switch
between any trained model and the prompt-driven ones at any time from the **Labeling model**
dropdown.

Practical notes:

- **Training runs on CPU unless you have a CUDA GPU**, and CPU is slow. Start with `yolo11n`, few
  epochs, to confirm the loop works before committing to a long run.
- **50+ labelled images per class** is a realistic target for a usable model; the panel warns when
  you have far fewer. A model trained on 5 images will be worse than prompting.
- Training only uses images that have at least one box.
- `LABELING_WORKSPACE` moves where datasets and runs are written.

## Getting good accuracy on electronics

Open-vocabulary detection is a way to *start* labeling without any data, not a finished detector
for a niche domain. Small SMD parts are genuinely hard for it. What helps, in order:

1. **Lower the confidence threshold.** This is the first thing to try, not the last. Niche classes
   score far lower than everyday objects, and the measurements above show parts are often detected
   at 5–20% confidence. The panel tells you how many detections are hidden below the slider.
2. **Match the mode to the image** — `Auto` handles it, but if parts are missing on a big scan turn
   tiling on, and on a macro shot turn it off (table above).
3. **Give each class several phrasings**, separated by `|`. Wording matters enormously and the
   failure is silent — on the reference photo, `coach` finds the bus **0 times**, while
   `omnibus | coach | bus | vehicle` finds it at **90%**. Every phrasing is tried and hits are
   reported under the first one:

   ```
   chip | microchip | integrated circuit, capacitor | electrolytic capacitor
   ```
4. **Ask for classes that are actually visible.** A chip resistor and a chip capacitor are the
   *same black rectangle* in a photo — the marking is often the only difference, and frequently
   there isn't one. No detector can separate them from shape alone, so asking for both mostly
   produces confident nonsense. Prefer one honest class (`smd component`) and split later by
   position or by reading the silkscreen. The **SMD board (coarse)** preset does this.
5. **Keep "one box per part" on** unless you deliberately want overlapping classes.
6. **Close the loop.** Label a batch (auto-label, then correct), export, train a YOLO model on it,
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
| `/api/models` | GET | — | prompt-driven models, training bases, and your trained runs |
| `/api/model` | POST | JSON: `model` | switches the active labeling model |
| `/api/train` | POST | JSON: `images[]`, `classes[]`, `epochs`, `base_model`, `imgsz` | starts training in the background |
| `/api/train/status` | GET | — | state, epoch, metrics, resulting weights path |
| `/api/train/cancel` | POST | — | stops after the current epoch, keeping weights |

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
`pip install -r requirements.txt` also fixes it — as long as your `requirements.txt` is up to date;
re-download the repo if `findstr openai-clip requirements.txt` (Windows) / `grep openai-clip
requirements.txt` (macOS/Linux) prints nothing.

**`ModuleNotFoundError: No module named 'pkg_resources'`** (after `clip` imports successfully in
every other respect) — `openai-clip`'s code does `from pkg_resources import packaging`, but
setuptools **removed pkg_resources in version 82.0.0** (confirmed present in 81.x, gone in 82.x),
and a fresh venv on Python 3.12+ doesn't install setuptools at all. Fix:

```
pip install "setuptools<82" --force-reinstall
```

`requirements.txt` pins this already; if you still hit it, your venv's setuptools was upgraded
after that pin was applied (a `pip install --upgrade` elsewhere in the setup, or the venv was
created before this pin existed in a version you downloaded) — the command above forces it back.

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

# AI Auto-Labeling

Prompt-driven image labeling for computer vision datasets. Type the objects you want
(`capacitor, resistor, ic, connector`), run detection, correct the boxes by hand, and export a
YOLO-format dataset you can train on directly.

The detector is open-vocabulary, so it is not limited to electronics — any class you can name in
the prompt box works. No training data is needed to start labeling.

## How it works

- **Backend** — FastAPI + [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/), an
  open-vocabulary detector. Text prompts are embedded once per class list, then used to detect
  those classes in an image.
- **Frontend** — React + Vite + Tailwind. Three panels: image list, annotation canvas, controls.
- **Export** — images + `labels/*.txt` in YOLO format plus a `data.yaml`, zipped for download.

Detection runs once per image at a permissive threshold; the confidence and overlap sliders then
re-filter that result client-side (confidence cutoff + per-class NMS), so they respond instantly
without re-running the model.

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Model weights (~70 MB for the detector, ~600 MB for the text encoder) download automatically on
the first detection request. First request is slow; later ones are fast.

Set a different model with `LABELING_MODEL` (default `yoloe-11l-seg.pt`):

```bash
LABELING_MODEL=yoloe-11s-seg.pt uvicorn app.main:app --port 8000   # smaller/faster
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend on port 8000.

## Using it

1. Drop images into the left panel.
2. Type the classes you want in **What to label**, comma-separated.
3. **Auto-label image** labels the selected image; **All** runs the whole batch.
4. Correct the results on the canvas:
   - drag a box to move it, drag the corner handle to resize
   - click a box to select it, then relabel it from the dropdown or delete it
   - drag on empty image area to draw a new box (uses the highlighted class chip)
   - `Delete` / `Backspace` removes the selected box
5. Tune **Confidence** and **Overlap** to filter detections; **Opacity** controls box fill.
6. **Export YOLO dataset (.zip)**.

Once you hand-edit an image, the threshold sliders stop rewriting its boxes so your corrections
are not lost. Re-running detection on that image resets it.

## Export format

```
data.yaml
images/train/photo.jpg
images/val/other.jpg
labels/train/photo.txt      # class_id cx cy w h   (normalized 0-1)
labels/val/other.txt
```

An 80/20 train/val split is applied automatically; with only one image everything goes to `train`
and `data.yaml` points `val` at it. Train on it with:

```bash
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## API

| Endpoint | Method | Body | Returns |
| --- | --- | --- | --- |
| `/api/health` | GET | — | `{"status":"ok"}` |
| `/api/detect` | POST | multipart: `image`, `prompts` (comma-separated), `confidence`, `iou` | image size + predictions (`x`, `y` = box center, pixels) |
| `/api/export` | POST | JSON: `images[]` (base64 + boxes), `classes[]`, `dataset_name` | dataset `.zip` |

## Notes

- Accuracy on dense boards varies by class wording — concrete nouns like `electrolytic capacitor`
  or `integrated circuit chip` usually beat abbreviations like `ic`.
- Runs on CPU; a CUDA GPU is used automatically when available and is much faster for batches.
- A natural next step is training a YOLO model on an exported dataset and using it to pre-label
  the next batch, which beats the open-vocabulary model on a fixed class set.

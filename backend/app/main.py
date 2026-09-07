import io
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from .detection import detect, load_model, progress, status
from .export import build_yolo_zip
from .schemas import ExportRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger("labeling")

# Pillow refuses very large images by default as a decompression-bomb guard;
# board scans from a DSLR routinely exceed it.
Image.MAX_IMAGE_PIXELS = 300_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model off the request path. The first load downloads ~670MB, and
    # doing that inside a detect request just times the browser out.
    logger.info("Warming up model in the background…")
    threading.Thread(target=_safe_load, daemon=True).start()
    yield


def _safe_load() -> None:
    try:
        load_model()
    except Exception:
        # Already logged and recorded in status(); /api/health reports it.
        pass


app = FastAPI(title="AI Auto-Labeling API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", **status()}


@app.get("/api/progress")
def detection_progress():
    """Tile progress for the running detection, polled by the UI."""
    return progress()


@app.post("/api/detect")
async def detect_endpoint(
    image: UploadFile = File(...),
    prompts: str = Form(...),
    confidence: float = Form(0.05),
    iou: float = Form(0.7),
    imgsz: int = Form(640),
    tiled: bool = Form(False),
    tile_size: int = Form(640),
    tile_overlap: float = Form(0.25),
):
    classes = [c.strip() for c in prompts.split(",") if c.strip()]
    if not classes:
        raise HTTPException(status_code=400, detail="At least one class prompt is required")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file")

    try:
        # Inference is blocking and slow; keep the event loop free so health
        # checks and other requests still respond while it runs.
        predictions = await run_in_threadpool(
            detect,
            img,
            classes,
            confidence,
            iou,
            imgsz,
            tiled,
            tile_size,
            tile_overlap,
        )
    except Exception as exc:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")

    return {
        "image": {"width": img.width, "height": img.height},
        "predictions": predictions,
    }


@app.post("/api/export")
async def export_endpoint(req: ExportRequest):
    if not req.classes:
        raise HTTPException(status_code=400, detail="classes list is required")
    if not req.images:
        raise HTTPException(status_code=400, detail="at least one image is required")

    buf = await run_in_threadpool(build_yolo_zip, req)
    filename = f"{req.dataset_name or 'dataset'}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from .detection import detect
from .export import build_yolo_zip
from .schemas import ExportRequest

app = FastAPI(title="AI Auto-Labeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect_endpoint(
    image: UploadFile = File(...),
    prompts: str = Form(...),
    confidence: float = Form(0.5),
    iou: float = Form(0.5),
):
    classes = [c.strip() for c in prompts.split(",") if c.strip()]
    if not classes:
        raise HTTPException(status_code=400, detail="At least one class prompt is required")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    predictions = detect(img, classes, confidence, iou)
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

    buf = build_yolo_zip(req)
    filename = f"{req.dataset_name or 'dataset'}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

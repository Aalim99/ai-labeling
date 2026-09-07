import logging
import os
import threading
import uuid
from typing import List, Sequence, Tuple, TypedDict

import torch
from PIL import Image
from ultralytics import YOLO, YOLOE, YOLOWorld

logger = logging.getLogger("labeling.detection")

DEFAULT_MODEL = os.environ.get("LABELING_MODEL", "yoloe-11l-seg.pt")

# Ultralytics caps detections per forward pass; dense boards blow past the default 300.
MAX_DET = int(os.environ.get("LABELING_MAX_DET", "1000"))


def _engine_for(name: str) -> str:
    """Open-vocabulary models take text prompts; a custom .pt has fixed classes."""
    lowered = os.path.basename(name).lower()
    if "yoloe" in lowered:
        return "yoloe"
    if "world" in lowered:
        return "world"
    return "custom"


# Mutable so a model trained during the session can be adopted without a
# restart, which is the whole point of the label -> train -> relabel loop.
_model_name = DEFAULT_MODEL
_engine = _engine_for(DEFAULT_MODEL)

_model: YOLO | None = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()
_current_classes: List[str] = []
_state = {"ready": False, "loading": False, "error": None}
_progress = {"active": False, "current": 0, "total": 0}


def progress() -> dict:
    return dict(_progress)


class Prediction(TypedDict):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    class_id: int
    detection_id: str


def is_prompted() -> bool:
    return _engine in ("yoloe", "world")


def model_name() -> str:
    return _model_name


def status() -> dict:
    classes: List[str] = []
    if _model is not None and not is_prompted():
        names = _model.names
        classes = list(names.values()) if isinstance(names, dict) else list(names)
    return {
        "model": _model_name,
        "engine": _engine,
        "prompted": is_prompted(),
        "classes": classes,
        "ready": _state["ready"],
        "loading": _state["loading"],
        "error": _state["error"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _check_text_encoder() -> None:
    """Fails early with a fixable message when the CLIP module is missing.

    Ultralytics' text_model.py does a top-level `import clip` and, when that
    fails, tries to pip-install its fork straight from GitHub — which dies with
    a confusing traceback on any machine without git.
    """
    if not is_prompted():
        return
    try:
        import clip  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            # openai-clip does `from pkg_resources import packaging`, but
            # setuptools removed pkg_resources in 82.0.0, and a fresh venv on
            # Python 3.12+ doesn't install setuptools at all. requirements.txt
            # pins setuptools<82, so this means that pin didn't take effect.
            raise RuntimeError(
                "'clip' is installed but needs pkg_resources, which recent setuptools no longer "
                "ships. Fix with:  pip install \"setuptools<82\" --force-reinstall"
            ) from exc
        raise RuntimeError(
            "The 'clip' package is missing. Prompt-driven models need it to read text prompts. "
            "Install it with:  pip install openai-clip  (no git required), "
            "or re-run: pip install -r requirements.txt"
        ) from exc


def load_model() -> YOLO:
    """Loads weights and warms the text encoder. Safe to call repeatedly."""
    global _model

    with _model_lock:
        if _model is not None:
            return _model
        _state["loading"] = True
        _state["error"] = None
        try:
            _check_text_encoder()
            logger.info("Loading %s (first run downloads weights, this can take a while)", _model_name)
            if _engine == "yoloe":
                model = YOLOE(_model_name)
                # Pull the MobileCLIP text encoder now so the first detect
                # request isn't stuck behind a ~600MB download.
                _embed(model, ["object"])
            elif _engine == "world":
                model = YOLOWorld(_model_name)
            else:
                model = YOLO(_model_name)
            _model = model
            _state["ready"] = True
            logger.info("Model ready: %s engine=%s device=%s", _model_name, _engine, status()["device"])
            return model
        except Exception as exc:  # surfaced through /api/health
            _state["error"] = str(exc)
            logger.exception("Model failed to load")
            raise
        finally:
            _state["loading"] = False


def switch_model(name: str) -> dict:
    """Swaps the active model, e.g. to a checkpoint just trained in-session."""
    global _model, _model_name, _engine, _current_classes

    with _model_lock:
        previous = (_model_name, _engine, _model)
        _model_name = name
        _engine = _engine_for(name)
        _model = None
        _current_classes = []
        _state.update(ready=False, error=None)

    try:
        load_model()
    except Exception:
        # Put the working model back rather than leaving the app with none.
        with _model_lock:
            _model_name, _engine, _model = previous
            _state["ready"] = _model is not None
        raise
    return status()


def _embed(model: YOLOE, prompts: Sequence[str]) -> torch.Tensor:
    """Embeds class names, keeping the CLIP model in memory between calls.

    YOLOE.get_text_pe() defaults to cache_clip_model=False, which reloads the
    600MB TorchScript encoder on every prompt change.
    """
    return model.model.get_text_pe(list(prompts), cache_clip_model=True)


def expand_synonyms(groups: List[str]) -> Tuple[List[str], List[str]]:
    """Splits "chip | microchip | ic" groups into flat prompts plus their labels.

    Open-vocabulary models are very sensitive to wording — one phrasing finds a
    part another misses entirely — so a class can offer several phrasings and
    every hit is reported under the first one.
    """
    flat: List[str] = []
    labels: List[str] = []
    for group in groups:
        variants = [v.strip() for v in group.split("|") if v.strip()]
        if not variants:
            continue
        for variant in variants:
            flat.append(variant)
            labels.append(variants[0])
    return flat, labels


def _set_classes(model: YOLO, prompts: List[str]) -> None:
    """No-op for fixed-class models, which detect whatever they were trained on."""
    global _current_classes
    if not is_prompted() or prompts == _current_classes:
        return
    if _engine == "yoloe":
        model.set_classes(list(prompts), _embed(model, prompts))
    else:
        model.set_classes(list(prompts))
    _current_classes = list(prompts)


def _tiles(width: int, height: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Splits an image into overlapping tiles, snapping the last one to the edge."""
    stride = max(1, int(tile * (1 - overlap)))

    def starts(total: int) -> List[int]:
        if total <= tile:
            return [0]
        values = list(range(0, total - tile + 1, stride))
        if values[-1] != total - tile:
            values.append(total - tile)
        return values

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]


def _overlap_ratio(a: Prediction, b: Prediction) -> float:
    """max(IoU, intersection-over-smaller).

    A component clipped by a tile seam yields a partial box whose IoU with the
    full box is low, but which is almost entirely contained by it — so plain IoU
    leaves duplicates along every seam.
    """
    ax1, ay1 = a["x"] - a["width"] / 2, a["y"] - a["height"] / 2
    ax2, ay2 = a["x"] + a["width"] / 2, a["y"] + a["height"] / 2
    bx1, by1 = b["x"] - b["width"] / 2, b["y"] - b["height"] / 2
    bx2, by2 = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0

    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    smaller = min(area_a, area_b)
    union = area_a + area_b - inter
    return max(inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def _merge(predictions: List[Prediction], threshold: float = 0.55) -> List[Prediction]:
    """Greedy per-class dedupe of boxes seen in more than one tile or scale.

    Keyed on the class name rather than the id: two synonyms for one class have
    different ids but must still collapse onto a single part.
    """
    kept: List[Prediction] = []
    for pred in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        if any(
            other["class_name"] == pred["class_name"] and _overlap_ratio(pred, other) > threshold
            for other in kept
        ):
            continue
        kept.append(pred)
    return kept


def _predict(
    model: YOLOE,
    image: Image.Image,
    prompts: List[str],
    confidence: float,
    iou: float,
    imgsz: int,
    offset: Tuple[int, int] = (0, 0),
    labels: List[str] | None = None,
) -> List[Prediction]:
    results = model.predict(
        image,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        max_det=MAX_DET,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None:
        return []

    # The model's own names are authoritative: for a custom checkpoint they are
    # its trained classes, not whatever was typed in the prompt box.
    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))

    dx, dy = offset
    predictions: List[Prediction] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        cls_id = int(box.cls[0].item())
        # Synonyms report under their canonical label, not the phrasing that hit.
        if labels and 0 <= cls_id < len(labels):
            fallback = labels[cls_id]
            canonical = labels[cls_id]
        else:
            fallback = prompts[cls_id] if 0 <= cls_id < len(prompts) else str(cls_id)
            canonical = None
        predictions.append(
            Prediction(
                x=x1 + width / 2 + dx,
                y=y1 + height / 2 + dy,
                width=width,
                height=height,
                confidence=float(box.conf[0].item()),
                class_name=canonical or str(names.get(cls_id, fallback)),
                class_id=cls_id,
                detection_id=str(uuid.uuid4()),
            )
        )
    return predictions


def detect(
    image: Image.Image,
    prompts: List[str],
    confidence: float,
    iou: float,
    imgsz: int = 640,
    tiled: bool = False,
    tile_size: int = 640,
    tile_overlap: float = 0.25,
    upscale: float = 1.0,
    tile_imgsz: int = 0,
    multiscale: bool = True,
) -> List[Prediction]:
    """Detects the prompted classes.

    With tiled=True the image is sliced into overlapping crops that are each run
    at native resolution, which is what makes small parts on a high-resolution
    board detectable at all — a whole-image pass shrinks them to a few pixels.

    upscale enlarges the image before tiling. A small photo of small parts has
    no resolution to recover by slicing alone; enlarging first gives the model
    objects big enough to recognise.
    """
    model = load_model()
    flat_prompts, labels = expand_synonyms(prompts)

    if upscale and upscale != 1.0:
        work = image.resize(
            (round(image.width * upscale), round(image.height * upscale)),
            Image.LANCZOS,
        )
        scaled = detect(
            work,
            prompts,
            confidence,
            iou,
            imgsz,
            tiled,
            tile_size,
            tile_overlap,
            upscale=1.0,
            tile_imgsz=tile_imgsz,
            multiscale=multiscale,
        )
        for pred in scaled:
            pred["x"] /= upscale
            pred["y"] /= upscale
            pred["width"] /= upscale
            pred["height"] /= upscale
        return scaled

    # One inference at a time: the model object is shared and set_classes mutates it.
    with _infer_lock:
        _set_classes(model, flat_prompts)

        if not tiled:
            return _predict(model, image, flat_prompts, confidence, iou, imgsz, labels=labels)

        # A tiled pass alone cannot see anything bigger than one tile, so a
        # board with a large chip plus tiny parts needs both scales. The
        # whole-image pass is one extra inference against many tiles.
        whole: List[Prediction] = []
        if multiscale:
            whole = _predict(model, image, flat_prompts, confidence, iou, imgsz, labels=labels)

        tiles = _tiles(image.width, image.height, tile_size, tile_overlap)
        logger.info("Tiled detection: %d tiles of %dpx", len(tiles), tile_size)

        # Running a tile at more than its own pixel size upsamples it, which
        # makes small parts bigger in the model's input tensor.
        crop_imgsz = tile_imgsz or tile_size

        _progress.update(active=True, current=0, total=len(tiles))
        predictions: List[Prediction] = []
        try:
            for index, (x1, y1, x2, y2) in enumerate(tiles, start=1):
                crop = image.crop((x1, y1, x2, y2))
                predictions.extend(
                    _predict(
                        model,
                        crop,
                        flat_prompts,
                        confidence,
                        iou,
                        crop_imgsz,
                        offset=(x1, y1),
                        labels=labels,
                    )
                )
                _progress["current"] = index
        finally:
            _progress["active"] = False

        # Whole-image boxes come last so an equally-scored tile box wins ties;
        # _merge sorts by confidence anyway.
        return _merge(predictions + whole)

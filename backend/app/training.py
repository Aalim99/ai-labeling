"""Fine-tunes a YOLO detector on the labels made in the app.

Prompting an open-vocabulary model can only ever guess at what a class means.
Training on corrected labels is what teaches it the actual convention — which
rectangle on this board counts as a resistor rather than a capacitor.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from .export import write_yolo_dataset
from .schemas import ExportRequest

logger = logging.getLogger("labeling.training")

# Kept beside the backend so trained weights survive restarts and are easy to find.
WORKSPACE = Path(os.environ.get("LABELING_WORKSPACE", Path(__file__).resolve().parents[1] / "training"))
DATASET_DIR = WORKSPACE / "dataset"
RUNS_DIR = WORKSPACE / "runs"

# Base checkpoints to fine-tune from. Small is the sane default: with a few
# dozen images a bigger model mostly overfits and trains far slower.
BASE_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]

_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_cancel = threading.Event()

_state = {
    "state": "idle",  # idle | preparing | training | done | error | cancelled
    "epoch": 0,
    "total_epochs": 0,
    "images": 0,
    "classes": [],
    "metrics": {},
    "error": None,
    "weights": None,
    "started_at": None,
    "finished_at": None,
    "message": "",
}


def status() -> dict:
    return dict(_state)


def list_trained_models() -> List[dict]:
    """Every best.pt this app has produced, newest first."""
    found = []
    for weights in RUNS_DIR.glob("*/weights/best.pt"):
        found.append(
            {
                "path": str(weights),
                "name": weights.parent.parent.name,
                "modified": weights.stat().st_mtime,
            }
        )
    return sorted(found, key=lambda m: m["modified"], reverse=True)


def cancel() -> None:
    _cancel.set()
    _state["message"] = "Stopping after the current epoch…"


def _reset(total_epochs: int, images: int, classes: List[str]) -> None:
    _state.update(
        state="preparing",
        epoch=0,
        total_epochs=total_epochs,
        images=images,
        classes=classes,
        metrics={},
        error=None,
        weights=None,
        started_at=time.time(),
        finished_at=None,
        message="Writing dataset…",
    )


def start(
    req: ExportRequest,
    epochs: int = 60,
    base_model: str = "yolo11s.pt",
    imgsz: int = 640,
) -> dict:
    """Kicks training off in the background; poll status() for progress."""
    global _thread

    with _lock:
        if _thread is not None and _thread.is_alive():
            raise RuntimeError("Training is already running")

        labelled = [img for img in req.images if img.boxes]
        if not labelled:
            raise RuntimeError("Nothing to train on: no image has any boxes")

        _cancel.clear()
        _reset(epochs, len(labelled), list(req.classes))
        _thread = threading.Thread(
            target=_run,
            args=(req, epochs, base_model, imgsz),
            daemon=True,
        )
        _thread.start()

    return status()


def _run(req: ExportRequest, epochs: int, base_model: str, imgsz: int) -> None:
    from ultralytics import YOLO

    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        data_yaml = write_yolo_dataset(req, DATASET_DIR)
        logger.info("Training on %d images, %d classes", len(req.images), len(req.classes))

        _state.update(state="training", message=f"Training {base_model} for {epochs} epochs")
        run_name = time.strftime("run-%Y%m%d-%H%M%S")

        model = YOLO(base_model)

        def on_epoch_end(trainer) -> None:
            # Clamp: the counter is cosmetic and must not exceed the total, or
            # the progress bar overshoots 100%.
            _state["epoch"] = min(int(getattr(trainer, "epoch", 0)) + 1, epochs)
            raw = getattr(trainer, "metrics", None) or {}
            # Ultralytics prefixes keys like "metrics/mAP50(B)"; keep the useful few.
            _state["metrics"] = {
                key.split("/")[-1].replace("(B)", ""): round(float(value), 4)
                for key, value in raw.items()
                if isinstance(value, (int, float)) and "metrics/" in key
            }
            if _cancel.is_set():
                # trainer.stop is the flag the training loop checks right after
                # this callback. Setting trainer.epoch instead does nothing —
                # the loop counter is a local, so training ran to completion.
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", on_epoch_end)

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=False,
            plots=False,
        )

        weights = RUNS_DIR / run_name / "weights" / "best.pt"
        if _cancel.is_set():
            _state.update(
                state="cancelled",
                weights=str(weights) if weights.exists() else None,
                message="Cancelled. Weights from completed epochs were kept.",
            )
        elif weights.exists():
            _state.update(
                state="done",
                weights=str(weights),
                message="Training finished. Switch to this model to use it for labeling.",
            )
        else:
            _state.update(state="error", error="Training produced no weights file")

    except Exception as exc:
        logger.exception("Training failed")
        _state.update(state="error", error=str(exc), message="")
    finally:
        _state["finished_at"] = time.time()

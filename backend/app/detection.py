import os
import uuid
from typing import List, TypedDict

from PIL import Image
from ultralytics import YOLOE

_MODEL_NAME = os.environ.get("LABELING_MODEL", "yoloe-11l-seg.pt")
_model: YOLOE | None = None
_current_classes: List[str] = []


class Prediction(TypedDict):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    class_id: int
    detection_id: str


def get_model() -> YOLOE:
    global _model
    if _model is None:
        _model = YOLOE(_MODEL_NAME)
    return _model


def detect(
    image: Image.Image,
    prompts: List[str],
    confidence: float,
    iou: float,
) -> List[Prediction]:
    global _current_classes

    model = get_model()
    # Embedding the text prompts is the slow part, so only redo it when they change.
    if prompts != _current_classes:
        model.set_classes(prompts, model.get_text_pe(prompts))
        _current_classes = list(prompts)

    results = model.predict(image, conf=confidence, iou=iou, verbose=False)
    result = results[0]

    predictions: List[Prediction] = []
    if result.boxes is None:
        return predictions

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        cls_id = int(box.cls[0].item())
        class_name = prompts[cls_id] if 0 <= cls_id < len(prompts) else str(cls_id)
        predictions.append(
            Prediction(
                x=x1 + width / 2,
                y=y1 + height / 2,
                width=width,
                height=height,
                confidence=float(box.conf[0].item()),
                class_name=class_name,
                class_id=cls_id,
                detection_id=str(uuid.uuid4()),
            )
        )
    return predictions

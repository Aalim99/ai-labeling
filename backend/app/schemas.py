from typing import List, Optional

from pydantic import BaseModel


class BoxAnnotation(BaseModel):
    x: float
    y: float
    width: float
    height: float
    class_name: str


class ExportImage(BaseModel):
    filename: str
    image_base64: str
    width: int
    height: int
    boxes: List[BoxAnnotation]


class ExportRequest(BaseModel):
    images: List[ExportImage]
    classes: List[str]
    dataset_name: Optional[str] = "dataset"
    train_split: Optional[float] = 0.8


class SwitchModelRequest(BaseModel):
    model: str


class TrainRequest(BaseModel):
    images: List[ExportImage]
    classes: List[str]
    epochs: int = 60
    base_model: str = "yolo11s.pt"
    imgsz: int = 640

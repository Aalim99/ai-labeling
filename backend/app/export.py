import base64
import io
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .schemas import ExportImage, ExportRequest


def _clip_box(box, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """Clips a box to the image and normalizes it.

    Boxes dragged past the edge would otherwise export coordinates outside
    [0, 1], which the YOLO loader rejects.
    """
    x1 = max(0.0, box.x - box.width / 2)
    y1 = max(0.0, box.y - box.height / 2)
    x2 = min(float(width), box.x + box.width / 2)
    y2 = min(float(height), box.y + box.height / 2)

    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return None

    return (x1 + w / 2) / width, (y1 + h / 2) / height, w / width, h / height


def _unique_names(images: List[ExportImage]) -> List[str]:
    """Keeps same-named uploads from overwriting each other inside the zip."""
    seen: dict[str, int] = {}
    names = []
    for img in images:
        name = img.filename
        if name in seen:
            seen[name] += 1
            stem, _, ext = name.rpartition(".")
            name = f"{stem}_{seen[img.filename]}.{ext}" if ext else f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def _build_files(req: ExportRequest) -> Iterator[Tuple[str, bytes]]:
    """Yields (path, bytes) for every file of the YOLO dataset.

    Shared by the zip download and the on-disk copy that training reads, so both
    get the same clipping, splits and data.yaml.
    """
    class_to_id = {name: idx for idx, name in enumerate(req.classes)}
    filenames = _unique_names(req.images)

    indices = list(range(len(req.images)))
    random.shuffle(indices)
    split_at = round(len(indices) * (req.train_split or 0.8))
    train_indices = set(indices[:split_at]) if len(indices) > 1 else set(indices)

    for i, img in enumerate(req.images):
        split = "train" if i in train_indices else "val"
        filename = filenames[i]

        header, _, encoded = img.image_base64.partition(",")
        yield f"images/{split}/{filename}", base64.b64decode(encoded or header)

        lines = []
        for box in img.boxes:
            cls_id = class_to_id.get(box.class_name)
            if cls_id is None:
                continue
            clipped = _clip_box(box, img.width, img.height)
            if clipped is None:
                continue
            cx, cy, w, h = clipped
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_name = filename.rsplit(".", 1)[0] + ".txt"
        yield f"labels/{split}/{label_name}", "\n".join(lines).encode()

    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(req.classes))
    has_val = len(train_indices) < len(req.images)
    yaml_content = (
        "train: images/train\n"
        f"val: {'images/val' if has_val else 'images/train'}\n"
        f"nc: {len(req.classes)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    yield "data.yaml", yaml_content.encode()


def build_yolo_zip(req: ExportRequest) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, blob in _build_files(req):
            zf.writestr(path, blob)
    buf.seek(0)
    return buf


def write_yolo_dataset(req: ExportRequest, destination: Path) -> Path:
    """Writes the dataset to disk and returns the path of its data.yaml."""
    if destination.exists():
        shutil.rmtree(destination)
    for path, blob in _build_files(req):
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return destination / "data.yaml"

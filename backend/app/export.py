import base64
import io
import random
import zipfile

from .schemas import ExportRequest


def build_yolo_zip(req: ExportRequest) -> io.BytesIO:
    class_to_id = {name: idx for idx, name in enumerate(req.classes)}

    indices = list(range(len(req.images)))
    random.shuffle(indices)
    split_at = round(len(indices) * (req.train_split or 0.8))
    train_indices = set(indices[:split_at]) if len(indices) > 1 else set(indices)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(req.images):
            split = "train" if i in train_indices else "val"

            header, _, encoded = img.image_base64.partition(",")
            raw = base64.b64decode(encoded or header)
            zf.writestr(f"images/{split}/{img.filename}", raw)

            lines = []
            for box in img.boxes:
                cls_id = class_to_id.get(box.class_name)
                if cls_id is None:
                    continue
                cx = box.x / img.width
                cy = box.y / img.height
                w = box.width / img.width
                h = box.height / img.height
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_name = img.filename.rsplit(".", 1)[0] + ".txt"
            zf.writestr(f"labels/{split}/{label_name}", "\n".join(lines))

        names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(req.classes))
        has_val = len(train_indices) < len(req.images)
        yaml_content = (
            "train: images/train\n"
            f"val: {'images/val' if has_val else 'images/train'}\n"
            f"nc: {len(req.classes)}\n"
            "names:\n"
            f"{names_block}\n"
        )
        zf.writestr("data.yaml", yaml_content)

    buf.seek(0)
    return buf

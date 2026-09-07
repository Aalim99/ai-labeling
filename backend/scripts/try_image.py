"""Run detection on one image and report what was found, without the UI.

    python scripts/try_image.py board.jpg
    python scripts/try_image.py board.jpg --classes "capacitor, resistor" --tiled

Writes an annotated preview next to the input so you can eyeball the result.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detection import detect, model_name  # noqa: E402

DEFAULT_CLASSES = "integrated circuit chip, capacitor, resistor, connector"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--classes", default=DEFAULT_CLASSES)
    parser.add_argument("--conf", type=float, default=0.02, help="raw floor, as the app uses")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-imgsz", type=int, default=0, help="e.g. 1280 for 2x upsampled tiles")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    tiled = args.tiled or max(image.size) > args.imgsz

    print(f"{args.image.name}  {image.width}x{image.height}  model={model_name()}")
    print(f"classes: {classes}")
    print(f"mode: {'tiled ' + str(args.tile_size) + 'px' if tiled else 'whole image'} @{args.imgsz}\n")

    started = time.time()
    predictions = detect(
        image,
        classes,
        args.conf,
        0.7,
        args.imgsz,
        tiled,
        args.tile_size,
        0.25,
        1.0,
        args.tile_imgsz,
    )
    elapsed = time.time() - started

    print(f"{len(predictions)} detections in {elapsed:.1f}s")
    for band, low, high in ((">=50%", 0.5, 1.01), ("25-50%", 0.25, 0.5), ("10-25%", 0.10, 0.25),
                            ("2-10%", 0.02, 0.10)):
        in_band = [p for p in predictions if low <= p["confidence"] < high]
        if in_band:
            counts = Counter(p["class_name"] for p in in_band)
            print(f"  {band:<7} {len(in_band):>4}  {dict(counts)}")

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for p in predictions:
        if p["confidence"] < 0.10:
            continue
        x1 = p["x"] - p["width"] / 2
        y1 = p["y"] - p["height"] / 2
        draw.rectangle([x1, y1, x1 + p["width"], y1 + p["height"]], outline=(255, 0, 200), width=2)

    out = args.image.with_name(f"{args.image.stem}_detected.png")
    preview.save(out)
    print(f"\npreview (>=10% confidence): {out}")


if __name__ == "__main__":
    main()

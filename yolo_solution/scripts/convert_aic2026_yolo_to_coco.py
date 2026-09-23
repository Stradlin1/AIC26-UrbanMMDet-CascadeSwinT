#!/usr/bin/env python3

import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


# ============================================================
# Paths
#
# 当前脚本:
#   ldds/yolo_solution/scripts/convert_aic2026_yolo_to_coco.py
#
# 因此:
#   PROJECT_ROOT = ldds/yolo_solution
#   LDDS_ROOT    = ldds
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_PATH.parents[1]
LDDS_ROOT = SCRIPT_PATH.parents[2]

DATA_ROOT = LDDS_ROOT / "datasets" / "AIC2026_Train_2000"

VISIBLE_DIR = DATA_ROOT / "visible"
LABEL_DIR = DATA_ROOT / "labels"

SPLIT_DIR = (
    PROJECT_ROOT
    / "splits"
    / "aic2026_group_stratified_v2"
)

OUTPUT_DIR = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
)


# ============================================================
# Dataset definition
# ============================================================

CLASS_NAMES = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage_can",
    "uav",
    "tricycle",
)

EXPECTED_COUNTS = {
    "train": 1600,
    "val": 400,
}

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def build_image_map():
    """Map image stem -> actual visible image path."""

    if not VISIBLE_DIR.is_dir():
        raise FileNotFoundError(
            f"Visible directory does not exist:\n{VISIBLE_DIR}"
        )

    image_map = {}

    for path in sorted(VISIBLE_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        stem = path.stem

        if stem in image_map:
            raise RuntimeError(
                "Duplicate visible image stem detected:\n"
                f"  {stem}\n"
                f"  {image_map[stem]}\n"
                f"  {path}"
            )

        image_map[stem] = path

    return image_map


def load_split(split):
    split_path = SPLIT_DIR / f"{split}.txt"

    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split file does not exist:\n{split_path}"
        )

    stems = [
        line.strip()
        for line in split_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    expected = EXPECTED_COUNTS[split]

    if len(stems) != expected:
        raise RuntimeError(
            f"{split}: expected {expected} stems, "
            f"got {len(stems)}"
        )

    if len(stems) != len(set(stems)):
        raise RuntimeError(
            f"{split}: duplicate stems exist"
        )

    return stems


def parse_yolo_label(
    label_path,
    image_width,
    image_height,
    image_id,
    annotation_id_start,
):
    """
    Convert one YOLO label file to COCO bbox annotations.

    YOLO:
        class cx cy w h
        normalized to [0, 1]

    COCO:
        bbox = [x_min, y_min, width, height]
        pixel coordinates

    Important:
        This script does NOT rewrite or repair source labels.
        Invalid labels cause an exception.
    """

    annotations = []
    class_counter = Counter()

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing label file:\n{label_path}"
        )

    raw_lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    lines = [
        line.strip()
        for line in raw_lines
        if line.strip()
    ]

    annotation_id = annotation_id_start

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                "Invalid YOLO label row:\n"
                f"{label_path}:{line_number}\n"
                f"{line}"
            )

        try:
            cls_raw, cx, cy, bw, bh = map(float, parts)
        except ValueError as exc:
            raise RuntimeError(
                "Non-numeric YOLO label row:\n"
                f"{label_path}:{line_number}\n"
                f"{line}"
            ) from exc

        values = (cls_raw, cx, cy, bw, bh)

        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(
                "Non-finite YOLO value:\n"
                f"{label_path}:{line_number}\n"
                f"{line}"
            )

        cls_id = int(cls_raw)

        if float(cls_id) != cls_raw:
            raise RuntimeError(
                "Class ID is not an integer:\n"
                f"{label_path}:{line_number}\n"
                f"{line}"
            )

        if not 0 <= cls_id < len(CLASS_NAMES):
            raise RuntimeError(
                "Class ID out of range:\n"
                f"{label_path}:{line_number}\n"
                f"class={cls_id}"
            )

        # Match the tolerance used by the existing Ultralytics
        # dataset verifier. Some official labels slightly exceed
        # the nominal [0, 1] range near image boundaries.
        normalized_tolerance = 0.01

        if not (
            -normalized_tolerance <= cx <= 1.0 + normalized_tolerance
            and -normalized_tolerance <= cy <= 1.0 + normalized_tolerance
            and 0.0 < bw <= 1.0 + normalized_tolerance
            and 0.0 < bh <= 1.0 + normalized_tolerance
        ):
            raise RuntimeError(
                "YOLO bbox exceeds accepted normalized tolerance:\n"
                f"{label_path}:{line_number}\n"
                f"{line}"
            )

        x1 = (cx - bw / 2.0) * image_width
        y1 = (cy - bh / 2.0) * image_height
        x2 = (cx + bw / 2.0) * image_width
        y2 = (cy + bh / 2.0) * image_height

        # YOLO annotations can describe objects truncated by
        # the image boundary. Keep the source TXT unchanged and
        # serialize only the visible intersection into COCO.
        raw_x1, raw_y1 = x1, y1
        raw_x2, raw_y2 = x2, y2

        # The original box must intersect the image.
        if (
            raw_x2 <= 0.0
            or raw_y2 <= 0.0
            or raw_x1 >= image_width
            or raw_y1 >= image_height
        ):
            raise RuntimeError(
                "BBox lies completely outside the image:\n"
                f"{label_path}:{line_number}\n"
                f"image={image_width}x{image_height}\n"
                f"xyxy=({raw_x1}, {raw_y1}, {raw_x2}, {raw_y2})"
            )

        x1 = max(0.0, raw_x1)
        y1 = max(0.0, raw_y1)
        x2 = min(float(image_width), raw_x2)
        y2 = min(float(image_height), raw_y2)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            raise RuntimeError(
                "Degenerate bbox:\n"
                f"{label_path}:{line_number}\n"
                f"{line}"
            )

        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,

                # COCO category IDs use 1..12 here.
                "category_id": cls_id + 1,

                "bbox": [
                    float(x1),
                    float(y1),
                    float(box_width),
                    float(box_height),
                ],

                "area": float(
                    box_width * box_height
                ),

                "iscrowd": 0,
            }
        )

        class_counter[cls_id] += 1
        annotation_id += 1

    return annotations, annotation_id, class_counter


def convert_split(split, image_map):

    stems = load_split(split)

    coco = {
        "info": {
            "description": (
                f"AIC2026 RGB {split} split "
                f"converted from YOLO labels"
            ),
            "version": "1.0",
        },

        "licenses": [],

        "categories": [
            {
                "id": class_id + 1,
                "name": class_name,
                "supercategory": "object",
            }
            for class_id, class_name
            in enumerate(CLASS_NAMES)
        ],

        "images": [],

        "annotations": [],
    }

    annotation_id = 1

    class_counter = Counter()
    empty_label_files = 0

    image_suffix_counter = Counter()
    image_size_counter = Counter()

    for image_id, stem in enumerate(stems, start=1):

        if stem not in image_map:
            raise FileNotFoundError(
                f"{split}: missing visible image for stem:\n"
                f"{stem}"
            )

        image_path = image_map[stem]
        label_path = LABEL_DIR / f"{stem}.txt"

        with Image.open(image_path) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid image dimensions:\n"
                f"{image_path}\n"
                f"{width}x{height}"
            )

        image_suffix_counter[
            image_path.suffix.lower()
        ] += 1

        image_size_counter[
            (width, height)
        ] += 1

        coco["images"].append(
            {
                "id": image_id,

                # MMDetection will combine this with data_prefix.
                "file_name": image_path.name,

                "width": width,
                "height": height,
            }
        )

        annotations, annotation_id, image_classes = (
            parse_yolo_label(
                label_path=label_path,
                image_width=width,
                image_height=height,
                image_id=image_id,
                annotation_id_start=annotation_id,
            )
        )

        if len(annotations) == 0:
            empty_label_files += 1

        coco["annotations"].extend(
            annotations
        )

        class_counter.update(
            image_classes
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"instances_{split}.json"
    )

    output_path.write_text(
        json.dumps(
            coco,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(f"Split               : {split}")
    print(f"Images              : {len(coco['images'])}")
    print(f"Annotations         : {len(coco['annotations'])}")
    print(f"Empty label files   : {empty_label_files}")
    print(f"Image suffix counts : {dict(image_suffix_counter)}")
    print(f"Image size types    : {len(image_size_counter)}")
    print()

    print("Class annotation counts:")

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        print(
            f"  {cls_id:2d} "
            f"{cls_name:12s}: "
            f"{class_counter[cls_id]}"
        )

    print()
    print(f"Output              : {output_path}")
    print("=" * 72)


def main():

    print("=" * 72)
    print("AIC2026 YOLO -> COCO converter")
    print("=" * 72)
    print("LDDS root   :", LDDS_ROOT)
    print("Project root:", PROJECT_ROOT)
    print("Data root   :", DATA_ROOT)
    print("Visible dir :", VISIBLE_DIR)
    print("Label dir   :", LABEL_DIR)
    print("Split dir   :", SPLIT_DIR)
    print("Output dir  :", OUTPUT_DIR)

    for required in (
        DATA_ROOT,
        VISIBLE_DIR,
        LABEL_DIR,
        SPLIT_DIR,
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required path missing:\n{required}"
            )

    image_map = build_image_map()

    print("Visible images found:", len(image_map))

    if len(image_map) != 2000:
        raise RuntimeError(
            f"Expected 2000 visible images, "
            f"found {len(image_map)}"
        )

    train_stems = set(load_split("train"))
    val_stems = set(load_split("val"))

    overlap = train_stems & val_stems

    if overlap:
        raise RuntimeError(
            f"Train/val overlap detected: "
            f"{len(overlap)} stems"
        )

    if len(train_stems | val_stems) != 2000:
        raise RuntimeError(
            "Train/val union is not 2000"
        )

    print("Train/val overlap    : 0")
    print("Train/val union      : 2000")

    convert_split(
        "train",
        image_map,
    )

    convert_split(
        "val",
        image_map,
    )

    print()
    print("Conversion completed successfully.")


if __name__ == "__main__":
    main()

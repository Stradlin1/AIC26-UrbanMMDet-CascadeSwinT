#!/usr/bin/env python3

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch
from PIL import Image


# ============================================================
# Project paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# /ldds/yolo_solution/scripts/xxx.py
YOLO_SOLUTION_ROOT = SCRIPT_PATH.parents[1]
LDDS_ROOT = SCRIPT_PATH.parents[2]

MMDET_ROOT = LDDS_ROOT / "mmdetection"

sys.path.insert(0, str(MMDET_ROOT))

from mmcv.transforms import Compose
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector


# ============================================================
# Experiment
# ============================================================

EXP_NAME = "c02_cascade_swin_t_rgb_1333x800"

RUN_DIR = (
    LDDS_ROOT
    / "runs"
    / EXP_NAME
)

CONFIG_PATH = (
    RUN_DIR
    / "resolved_config.py"
)

CHECKPOINT_PATH = (
    RUN_DIR
    / "best_coco_bbox_mAP_epoch_11.pth"
)


# ============================================================
# Phase-1 test dataset
# ============================================================

TEST_ROOT = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_PHASE_1_1000"
)

VISIBLE_DIR = TEST_ROOT / "visible"


# ============================================================
# Submission output
# ============================================================

SUBMISSION_DIR = (
    RUN_DIR
    / "submission_phase1"
)

LABEL_DIR = (
    SUBMISSION_DIR
    / "labels"
)

ZIP_PATH = (
    SUBMISSION_DIR
    / "c02_cascade_swin_t_rgb_phase1.zip"
)

SUMMARY_PATH = (
    SUBMISSION_DIR
    / "inference_summary.txt"
)


# ============================================================
# Fixed inference protocol
# ============================================================

IMAGE_SCALE = (1333, 800)

SCORE_THR = 0.001
NMS_IOU = 0.5
MAX_DET = 100

DEVICE = "cuda:0"

EXPECTED_IMAGES = 1000

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


CLASSES = (
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


# ============================================================
# Preflight checks
# ============================================================

def collect_images():
    if not VISIBLE_DIR.exists():
        raise FileNotFoundError(
            f"Visible test directory not found:\n{VISIBLE_DIR}"
        )

    images = sorted(
        p
        for p in VISIBLE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
    )

    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} test images, "
            f"but found {len(images)}."
        )

    stems = [p.stem for p in images]

    if len(set(stems)) != len(stems):
        duplicate_stems = [
            stem
            for stem, count in Counter(stems).items()
            if count > 1
        ]

        raise RuntimeError(
            "Duplicate image stems detected:\n"
            + "\n".join(duplicate_stems[:20])
        )

    return images


def check_paths():
    required = (
        MMDET_ROOT,
        CONFIG_PATH,
        CHECKPOINT_PATH,
        TEST_ROOT,
        VISIBLE_DIR,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )


# ============================================================
# Build model
# ============================================================

def build_model():
    cfg = Config.fromfile(
        str(CONFIG_PATH)
    )

    # Never load the partial COCO initialization checkpoint here.
    # We are loading the final trained C02 checkpoint.
    cfg.load_from = None

    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    # Defensive checks.
    if len(cfg.model.roi_head.bbox_head) != 3:
        raise RuntimeError(
            "Expected 3 Cascade bbox heads."
        )

    for index, head in enumerate(
        cfg.model.roi_head.bbox_head,
        start=1,
    ):
        if head.num_classes != len(CLASSES):
            raise RuntimeError(
                f"Cascade stage {index} num_classes "
                f"is {head.num_classes}, expected 12."
            )

    # Freeze official C02 inference settings.
    cfg.model.test_cfg.rcnn.score_thr = SCORE_THR
    cfg.model.test_cfg.rcnn.nms.iou_threshold = NMS_IOU
    cfg.model.test_cfg.rcnn.max_per_img = MAX_DET

    model = init_detector(
        cfg,
        checkpoint=str(CHECKPOINT_PATH),
        device=DEVICE,
        palette="random",
    )

    # We use numeric class IDs for submission, but keep metadata explicit.
    model.dataset_meta["classes"] = CLASSES

    return model


# ============================================================
# Test pipeline
#
# IMPORTANT:
# No LoadAnnotations here because Phase-1 has no GT labels.
# ============================================================

def build_test_pipeline():
    pipeline = Compose(
        [
            dict(
                type="LoadImageFromFile",
            ),
            dict(
                type="Resize",
                scale=IMAGE_SCALE,
                keep_ratio=True,
            ),
            dict(
                type="PackDetInputs",
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
            ),
        ]
    )

    return pipeline


# ============================================================
# Prediction conversion
# ============================================================

def convert_prediction(
    result,
    image_path,
):
    pred = result.pred_instances

    bboxes = (
        pred.bboxes
        .detach()
        .cpu()
    )

    scores = (
        pred.scores
        .detach()
        .cpu()
    )

    labels = (
        pred.labels
        .detach()
        .cpu()
    )

    if len(scores) == 0:
        return []

    # Defensive sorting by confidence.
    order = torch.argsort(
        scores,
        descending=True,
    )

    order = order[:MAX_DET]

    bboxes = bboxes[order]
    scores = scores[order]
    labels = labels[order]

    with Image.open(image_path) as im:
        width, height = im.size

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid image size: {image_path}"
        )

    output = []

    for bbox, score, label in zip(
        bboxes,
        scores,
        labels,
    ):
        class_id = int(label.item())
        confidence = float(score.item())

        if not (0 <= class_id < len(CLASSES)):
            raise RuntimeError(
                f"Invalid class id {class_id} "
                f"in {image_path.name}"
            )

        if not math.isfinite(confidence):
            raise RuntimeError(
                f"Invalid confidence in {image_path.name}"
            )

        if confidence < SCORE_THR:
            continue

        x1, y1, x2, y2 = [
            float(v)
            for v in bbox.tolist()
        ]

        # Predictions are rescaled back to original image coordinates
        # by MMDetection inference.
        x1 = max(0.0, min(x1, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        x2 = max(0.0, min(x2, float(width)))
        y2 = max(0.0, min(y2, float(height)))

        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height

        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        values = (
            cx,
            cy,
            bw,
            bh,
            confidence,
        )

        if not all(
            math.isfinite(v)
            for v in values
        ):
            raise RuntimeError(
                f"NaN/Inf prediction in {image_path.name}"
            )

        # Defensive numerical clipping.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        if bw <= 0.0 or bh <= 0.0:
            continue

        output.append(
            (
                class_id,
                cx,
                cy,
                bw,
                bh,
                confidence,
            )
        )

    # Keep confidence order and enforce competition limit.
    output.sort(
        key=lambda x: x[5],
        reverse=True,
    )

    return output[:MAX_DET]


# ============================================================
# Write TXT
# ============================================================

def write_prediction_txt(
    image_path,
    predictions,
):
    txt_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    with txt_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for (
            class_id,
            cx,
            cy,
            bw,
            bh,
            confidence,
        ) in predictions:
            f.write(
                f"{class_id} "
                f"{cx:.8f} "
                f"{cy:.8f} "
                f"{bw:.8f} "
                f"{bh:.8f} "
                f"{confidence:.8f}\n"
            )


# ============================================================
# Submission validation
# ============================================================

def validate_submission(images):
    expected_stems = {
        p.stem
        for p in images
    }

    txt_files = sorted(
        LABEL_DIR.glob("*.txt")
    )

    actual_stems = {
        p.stem
        for p in txt_files
    }

    if len(txt_files) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"TXT count = {len(txt_files)}, "
            f"expected {EXPECTED_IMAGES}."
        )

    missing = sorted(
        expected_stems
        - actual_stems
    )

    extra = sorted(
        actual_stems
        - expected_stems
    )

    if missing:
        raise RuntimeError(
            f"Missing prediction TXT files: {missing[:20]}"
        )

    if extra:
        raise RuntimeError(
            f"Extra prediction TXT files: {extra[:20]}"
        )

    total_boxes = 0
    empty_files = 0
    max_boxes = 0
    class_counts = Counter()
    all_scores = []

    for txt_path in txt_files:
        lines = [
            line.strip()
            for line in txt_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(lines) == 0:
            empty_files += 1

        if len(lines) > MAX_DET:
            raise RuntimeError(
                f"{txt_path.name} has "
                f"{len(lines)} boxes > {MAX_DET}."
            )

        max_boxes = max(
            max_boxes,
            len(lines),
        )

        total_boxes += len(lines)

        previous_score = float("inf")

        for line_no, line in enumerate(
            lines,
            start=1,
        ):
            parts = line.split()

            if len(parts) != 6:
                raise RuntimeError(
                    f"{txt_path.name}:{line_no} "
                    f"does not contain 6 fields."
                )

            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                score = float(parts[5])
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid numeric field in "
                    f"{txt_path.name}:{line_no}"
                ) from exc

            if not (0 <= class_id < len(CLASSES)):
                raise RuntimeError(
                    f"Invalid class id in "
                    f"{txt_path.name}:{line_no}"
                )

            for name, value in (
                ("cx", cx),
                ("cy", cy),
                ("w", bw),
                ("h", bh),
                ("score", score),
            ):
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"NaN/Inf {name} in "
                        f"{txt_path.name}:{line_no}"
                    )

            if not (
                0.0 <= cx <= 1.0
                and 0.0 <= cy <= 1.0
                and 0.0 < bw <= 1.0
                and 0.0 < bh <= 1.0
                and 0.0 <= score <= 1.0
            ):
                raise RuntimeError(
                    f"Out-of-range prediction in "
                    f"{txt_path.name}:{line_no}"
                )

            if score > previous_score + 1e-12:
                raise RuntimeError(
                    f"Scores not sorted in "
                    f"{txt_path.name}"
                )

            previous_score = score

            class_counts[class_id] += 1
            all_scores.append(score)

    return {
        "txt_count": len(txt_files),
        "total_boxes": total_boxes,
        "empty_files": empty_files,
        "max_boxes": max_boxes,
        "class_counts": class_counts,
        "scores": all_scores,
    }


# ============================================================
# ZIP
# ============================================================

def build_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    txt_files = sorted(
        LABEL_DIR.glob("*.txt")
    )

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for txt_path in txt_files:
            # IMPORTANT:
            # TXT files are placed directly in the ZIP root,
            # not inside a labels/ directory.
            zf.write(
                txt_path,
                arcname=txt_path.name,
            )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:
        names = zf.namelist()

    if len(names) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"ZIP contains {len(names)} files, "
            f"expected {EXPECTED_IMAGES}."
        )

    if any("/" in name for name in names):
        raise RuntimeError(
            "ZIP unexpectedly contains subdirectories."
        )


# ============================================================
# Main
# ============================================================

def main():
    check_paths()

    images = collect_images()

    if SUBMISSION_DIR.exists():
        shutil.rmtree(
            SUBMISSION_DIR
        )

    LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("AIC2026 C02 Phase-1 inference")
    print("=" * 72)

    print("Model:")
    print("  Cascade R-CNN + Swin-T + FPN")
    print()

    print("Config:")
    print(f"  {CONFIG_PATH}")

    print("Checkpoint:")
    print(f"  {CHECKPOINT_PATH}")

    print()
    print("Test visible:")
    print(f"  {VISIBLE_DIR}")

    print()
    print("Images          :", len(images))
    print("Input scale     :", IMAGE_SCALE)
    print("Keep ratio      : True")
    print("Score threshold :", SCORE_THR)
    print("NMS IoU         :", NMS_IOU)
    print("Max detections  :", MAX_DET)
    print("Device          :", DEVICE)

    print("=" * 72)
    print()

    model = build_model()

    test_pipeline = (
        build_test_pipeline()
    )

    start_time = time.time()

    total_raw_predictions = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        result = inference_detector(
            model,
            str(image_path),
            test_pipeline=test_pipeline,
        )

        predictions = convert_prediction(
            result,
            image_path,
        )

        total_raw_predictions += len(
            predictions
        )

        write_prediction_txt(
            image_path,
            predictions,
        )

        if (
            index == 1
            or index % 50 == 0
            or index == len(images)
        ):
            elapsed = (
                time.time()
                - start_time
            )

            speed = (
                index / elapsed
                if elapsed > 0
                else 0.0
            )

            eta = (
                (len(images) - index) / speed
                if speed > 0
                else 0.0
            )

            print(
                f"[{index:4d}/{len(images)}] "
                f"{image_path.name}  "
                f"boxes={len(predictions):3d}  "
                f"speed={speed:.2f} img/s  "
                f"ETA={eta / 60:.1f} min"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("Inference finished.")
    print(
        f"Elapsed: {elapsed / 60:.2f} min"
    )

    print()
    print("Validating submission...")

    stats = validate_submission(
        images
    )

    print("Submission validation: PASS")

    print()
    print("Building ZIP...")

    build_zip()

    print("ZIP validation: PASS")

    scores = stats["scores"]

    if scores:
        score_min = min(scores)
        score_max = max(scores)
        score_mean = (
            sum(scores)
            / len(scores)
        )
    else:
        score_min = 0.0
        score_max = 0.0
        score_mean = 0.0

    summary_lines = [
        "AIC2026 C02 Phase-1 inference summary",
        "=" * 60,
        f"checkpoint: {CHECKPOINT_PATH}",
        f"images: {len(images)}",
        f"txt files: {stats['txt_count']}",
        f"total boxes: {stats['total_boxes']}",
        f"empty txt files: {stats['empty_files']}",
        f"max boxes/image: {stats['max_boxes']}",
        f"score min: {score_min:.8f}",
        f"score mean: {score_mean:.8f}",
        f"score max: {score_max:.8f}",
        "",
        "class counts:",
    ]

    for class_id, class_name in enumerate(
        CLASSES
    ):
        summary_lines.append(
            f"{class_id:2d} "
            f"{class_name:12s}: "
            f"{stats['class_counts'][class_id]}"
        )

    summary_lines.extend(
        [
            "",
            f"labels: {LABEL_DIR}",
            f"zip: {ZIP_PATH}",
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        "TXT files        :",
        stats["txt_count"],
    )

    print(
        "Total boxes      :",
        stats["total_boxes"],
    )

    print(
        "Empty TXT        :",
        stats["empty_files"],
    )

    print(
        "Max boxes/image  :",
        stats["max_boxes"],
    )

    print(
        "Score min/mean/max:",
        f"{score_min:.6f} / "
        f"{score_mean:.6f} / "
        f"{score_max:.6f}",
    )

    print()
    print("ZIP:")
    print(ZIP_PATH)

    print()
    print("Summary:")
    print(SUMMARY_PATH)

    print("=" * 72)


if __name__ == "__main__":
    main()

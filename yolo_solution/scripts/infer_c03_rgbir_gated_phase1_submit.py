#!/usr/bin/env python3

import os

# PyTorch >= 2.6:
# checkpoint is our own trusted local MMEngine checkpoint.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Keep deterministic behavior aligned with validation.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch


# ============================================================
# Project paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# ldds/yolo_solution/scripts/xxx.py -> ldds
LDDS_ROOT = SCRIPT_PATH.parents[2]

MMDET_ROOT = (
    LDDS_ROOT
    / "mmdetection"
)

COMPONENT_DIR = (
    LDDS_ROOT
    / "yolo_solution"
    / "scripts"
)

sys.path.insert(
    0,
    str(MMDET_ROOT),
)

sys.path.insert(
    0,
    str(COMPONENT_DIR),
)


from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS

# Register custom C03 components.
from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
)


# ============================================================
# C03 formal model
# ============================================================

EXP_NAME = (
    "c03_cascade_swin_t_rgbir_gated_1333x800"
)

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
    / "best_coco_bbox_mAP_epoch_9.pth"
)


# ============================================================
# Phase-1 test dataset
# ============================================================

TEST_ROOT = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_PHASE_1_1000"
)

VISIBLE_DIR = (
    TEST_ROOT
    / "visible"
)

IR_DIR = (
    TEST_ROOT
    / "infrared"
)


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
    / "submission_c03_rgbir_gated_phase1.zip"
)

MANIFEST_PATH = (
    SUBMISSION_DIR
    / "submission_manifest.json"
)


# ============================================================
# Competition settings
# ============================================================

EXPECTED_IMAGES = 1000

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
}

NUM_CLASSES = 12

SCORE_THR = 0.001
NMS_IOU = 0.5
MAX_DET = 100

IMAGE_SCALE = (
    1333,
    800,
)


# ============================================================
# Helpers
# ============================================================

def check_paths():
    required = (
        MMDET_ROOT,
        COMPONENT_DIR,
        CONFIG_PATH,
        CHECKPOINT_PATH,
        TEST_ROOT,
        VISIBLE_DIR,
        IR_DIR,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )


def find_visible_images():
    images = sorted(
        [
            path
            for path in VISIBLE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_SUFFIXES
            )
        ],
        key=lambda p: p.name,
    )

    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Unexpected number of Phase-1 visible images:\n"
            f"expected = {EXPECTED_IMAGES}\n"
            f"actual   = {len(images)}"
        )

    # TXT names are based on stems, so stems must be unique.
    stems = [
        path.stem
        for path in images
    ]

    if len(stems) != len(set(stems)):
        raise RuntimeError(
            "Duplicate image stems detected. "
            "TXT filenames would collide."
        )

    missing_ir = []

    for rgb_path in images:
        ir_path = (
            IR_DIR
            / rgb_path.name
        )

        if not ir_path.exists():
            missing_ir.append(
                ir_path
            )

    if missing_ir:
        print()
        print("Missing IR files:")

        for path in missing_ir[:20]:
            print(" ", path)

        raise RuntimeError(
            f"{len(missing_ir)} RGB images "
            "do not have matching IR images."
        )

    return images


def load_checkpoint_state(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


def build_model(device):
    cfg = Config.fromfile(
        str(CONFIG_PATH)
    )

    # Enforce the exact validated C03 test protocol.
    cfg.model.test_cfg.rcnn.score_thr = (
        SCORE_THR
    )

    cfg.model.test_cfg.rcnn.nms.iou_threshold = (
        NMS_IOU
    )

    cfg.model.test_cfg.rcnn.max_per_img = (
        MAX_DET
    )

    model = MODELS.build(
        cfg.model
    )

    state_dict = load_checkpoint_state(
        CHECKPOINT_PATH
    )

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "Checkpoint strict load:",
        result,
    )

    model = (
        model
        .to(device)
        .eval()
    )

    return model


def build_pipeline():
    # Same inference geometry as C03 val400.
    pipeline = Compose(
        [
            dict(
                type="LoadRGBIRFromFile",
            ),

            dict(
                type="Resize",
                scale=IMAGE_SCALE,
                keep_ratio=True,
            ),

            dict(
                type="PackDetInputs",
                meta_keys=(
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
            ),
        ]
    )

    return pipeline


def extract_ori_hw(data_sample):
    ori_shape = data_sample.metainfo.get(
        "ori_shape"
    )

    if ori_shape is None:
        raise RuntimeError(
            "ori_shape missing from data sample."
        )

    height = int(
        ori_shape[0]
    )

    width = int(
        ori_shape[1]
    )

    if (
        height <= 0
        or width <= 0
    ):
        raise RuntimeError(
            f"Invalid ori_shape: {ori_shape}"
        )

    return height, width


def clamp(value, low, high):
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def prediction_to_lines(
    pred_instances,
    width,
    height,
):
    boxes = (
        pred_instances
        .bboxes
        .detach()
        .float()
        .cpu()
    )

    scores = (
        pred_instances
        .scores
        .detach()
        .float()
        .cpu()
    )

    labels = (
        pred_instances
        .labels
        .detach()
        .long()
        .cpu()
    )

    if not (
        len(boxes)
        == len(scores)
        == len(labels)
    ):
        raise RuntimeError(
            "Prediction tensor lengths differ."
        )

    if len(scores) > 0:
        order = torch.argsort(
            scores,
            descending=True,
        )

        order = order[
            :MAX_DET
        ]

        boxes = boxes[
            order
        ]

        scores = scores[
            order
        ]

        labels = labels[
            order
        ]

    lines = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):
        class_id = int(
            label.item()
        )

        confidence = float(
            score.item()
        )

        if not (
            0 <= class_id < NUM_CLASSES
        ):
            raise RuntimeError(
                f"Illegal class id: {class_id}"
            )

        if not math.isfinite(
            confidence
        ):
            raise RuntimeError(
                "NaN/Inf confidence detected."
            )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):
            raise RuntimeError(
                f"Illegal confidence: {confidence}"
            )

        x1 = float(
            box[0].item()
        )

        y1 = float(
            box[1].item()
        )

        x2 = float(
            box[2].item()
        )

        y2 = float(
            box[3].item()
        )

        values = (
            x1,
            y1,
            x2,
            y2,
        )

        if not all(
            math.isfinite(x)
            for x in values
        ):
            raise RuntimeError(
                "NaN/Inf bbox detected."
            )

        # rescale=True returns original-image coordinates.
        # Clip numerical overflow to valid image boundaries.
        x1 = clamp(
            x1,
            0.0,
            float(width),
        )

        x2 = clamp(
            x2,
            0.0,
            float(width),
        )

        y1 = clamp(
            y1,
            0.0,
            float(height),
        )

        y2 = clamp(
            y2,
            0.0,
            float(height),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue

        center_x = (
            (x1 + x2)
            * 0.5
            / width
        )

        center_y = (
            (y1 + y2)
            * 0.5
            / height
        )

        norm_w = (
            (x2 - x1)
            / width
        )

        norm_h = (
            (y2 - y1)
            / height
        )

        center_x = clamp(
            center_x,
            0.0,
            1.0,
        )

        center_y = clamp(
            center_y,
            0.0,
            1.0,
        )

        norm_w = clamp(
            norm_w,
            0.0,
            1.0,
        )

        norm_h = clamp(
            norm_h,
            0.0,
            1.0,
        )

        normalized = (
            center_x,
            center_y,
            norm_w,
            norm_h,
        )

        if not all(
            math.isfinite(x)
            for x in normalized
        ):
            raise RuntimeError(
                "NaN/Inf normalized bbox detected."
            )

        line = (
            f"{class_id:d} "
            f"{center_x:.8f} "
            f"{center_y:.8f} "
            f"{norm_w:.8f} "
            f"{norm_h:.8f} "
            f"{confidence:.8f}"
        )

        lines.append(
            line
        )

    if len(lines) > MAX_DET:
        raise RuntimeError(
            f"Generated {len(lines)} boxes; "
            f"max allowed is {MAX_DET}."
        )

    return lines


def validate_txt_files(
    expected_images,
):
    txt_files = sorted(
        LABEL_DIR.glob(
            "*.txt"
        )
    )

    if len(txt_files) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Wrong number of TXT files:\n"
            f"expected = {EXPECTED_IMAGES}\n"
            f"actual   = {len(txt_files)}"
        )

    expected_stems = {
        path.stem
        for path in expected_images
    }

    actual_stems = {
        path.stem
        for path in txt_files
    }

    if expected_stems != actual_stems:
        missing = sorted(
            expected_stems
            - actual_stems
        )

        extra = sorted(
            actual_stems
            - expected_stems
        )

        raise RuntimeError(
            "TXT/image stem mismatch:\n"
            f"missing={missing[:20]}\n"
            f"extra={extra[:20]}"
        )

    total_boxes = 0
    empty_files = 0
    max_boxes_found = 0

    class_counts = [
        0
        for _ in range(
            NUM_CLASSES
        )
    ]

    for txt_path in txt_files:
        text = txt_path.read_text(
            encoding="utf-8"
        )

        stripped = text.strip()

        if stripped == "":
            empty_files += 1
            continue

        lines = stripped.splitlines()

        if len(lines) > MAX_DET:
            raise RuntimeError(
                f"{txt_path.name} has "
                f"{len(lines)} predictions."
            )

        max_boxes_found = max(
            max_boxes_found,
            len(lines),
        )

        total_boxes += len(
            lines
        )

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            parts = line.split()

            if len(parts) != 6:
                raise RuntimeError(
                    f"{txt_path.name}:"
                    f"{line_number}: "
                    "expected 6 columns."
                )

            class_id = int(
                parts[0]
            )

            values = [
                float(x)
                for x in parts[1:]
            ]

            if not (
                0 <= class_id < NUM_CLASSES
            ):
                raise RuntimeError(
                    f"Illegal class id in "
                    f"{txt_path.name}"
                )

            class_counts[
                class_id
            ] += 1

            for value in values:
                if not math.isfinite(
                    value
                ):
                    raise RuntimeError(
                        f"NaN/Inf in "
                        f"{txt_path.name}"
                    )

            for value in values[
                0:4
            ]:
                if not (
                    0.0
                    <= value
                    <= 1.0
                ):
                    raise RuntimeError(
                        f"Illegal normalized bbox "
                        f"in {txt_path.name}"
                    )

            confidence = values[
                4
            ]

            if not (
                0.0
                <= confidence
                <= 1.0
            ):
                raise RuntimeError(
                    f"Illegal confidence in "
                    f"{txt_path.name}"
                )

    return dict(
        txt_files=len(
            txt_files
        ),

        total_boxes=(
            total_boxes
        ),

        empty_files=(
            empty_files
        ),

        max_boxes_per_image=(
            max_boxes_found
        ),

        class_counts=(
            class_counts
        ),
    )


def create_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:

        for txt_path in sorted(
            LABEL_DIR.glob(
                "*.txt"
            )
        ):
            # TXT files are stored directly at ZIP root.
            zf.write(
                txt_path,
                arcname=txt_path.name,
            )


def validate_zip():
    with zipfile.ZipFile(
        ZIP_PATH,
        mode="r",
    ) as zf:
        members = zf.namelist()

        if len(members) != EXPECTED_IMAGES:
            raise RuntimeError(
                "Wrong number of ZIP members:\n"
                f"expected = {EXPECTED_IMAGES}\n"
                f"actual   = {len(members)}"
            )

        if len(
            members
        ) != len(
            set(members)
        ):
            raise RuntimeError(
                "Duplicate ZIP members found."
            )

        for member in members:
            path = Path(
                member
            )

            if (
                path.parent
                != Path(".")
            ):
                raise RuntimeError(
                    "Submission ZIP contains "
                    "a directory layer:\n"
                    f"{member}"
                )

            if (
                path.suffix.lower()
                != ".txt"
            ):
                raise RuntimeError(
                    f"Non-TXT ZIP member: {member}"
                )

        bad = zf.testzip()

        if bad is not None:
            raise RuntimeError(
                f"Corrupt ZIP member: {bad}"
            )


# ============================================================
# Main
# ============================================================

def main():
    check_paths()

    init_default_scope(
        "mmdet"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("AIC2026 C03 RGB+IR PHASE-1 INFERENCE")
    print("=" * 72)

    print(
        "Experiment :",
        EXP_NAME,
    )

    print(
        "Checkpoint :",
        CHECKPOINT_PATH,
    )

    print(
        "Test root  :",
        TEST_ROOT,
    )

    print(
        "Device     :",
        device,
    )

    print(
        "Scale      :",
        IMAGE_SCALE,
    )

    print(
        "score_thr  :",
        SCORE_THR,
    )

    print(
        "NMS IoU    :",
        NMS_IOU,
    )

    print(
        "max_det    :",
        MAX_DET,
    )

    print(
        "Precision  : FP32"
    )

    print()

    images = find_visible_images()

    print(
        "Visible images:",
        len(images),
    )

    print(
        "IR pairs      :",
        len(images),
    )

    # --------------------------------------------------------
    # Clean submission output
    # --------------------------------------------------------

    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LABEL_DIR.exists():
        shutil.rmtree(
            LABEL_DIR
        )

    LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model / pipeline
    # --------------------------------------------------------

    model = build_model(
        device
    )

    pipeline = build_pipeline()

    raw_gates = (
        model
        .backbone
        .gates
        .detach()
        .float()
        .cpu()
    )

    alpha_gates = torch.tanh(
        raw_gates
    )

    print()
    print(
        "Raw gates  :",
        raw_gates.tolist(),
    )

    print(
        "Alpha gates:",
        alpha_gates.tolist(),
    )

    print()
    print("=" * 72)
    print("STARTING PHASE-1 INFERENCE")
    print("=" * 72)
    print()

    start_time = time.time()

    total_predictions = 0
    empty_predictions = 0

    with torch.inference_mode():

        for index, rgb_path in enumerate(
            images,
            start=1,
        ):
            packed = pipeline(
                dict(
                    img_path=str(
                        rgb_path
                    ),
                )
            )

            data_sample = packed[
                "data_samples"
            ]

            height, width = (
                extract_ori_hw(
                    data_sample
                )
            )

            batch = dict(
                inputs=[
                    packed[
                        "inputs"
                    ]
                ],

                data_samples=[
                    data_sample
                ],
            )

            processed = (
                model
                .data_preprocessor(
                    batch,
                    training=False,
                )
            )

            outputs = model.predict(
                processed[
                    "inputs"
                ],

                processed[
                    "data_samples"
                ],

                rescale=True,
            )

            if len(outputs) != 1:
                raise RuntimeError(
                    "Expected one prediction "
                    "result per image."
                )

            pred_instances = (
                outputs[0]
                .pred_instances
            )

            lines = prediction_to_lines(
                pred_instances,
                width,
                height,
            )

            total_predictions += len(
                lines
            )

            if len(lines) == 0:
                empty_predictions += 1

            txt_path = (
                LABEL_DIR
                / f"{rgb_path.stem}.txt"
            )

            # Empty predictions intentionally create an empty TXT.
            with open(
                txt_path,
                "w",
                encoding="utf-8",
            ) as file:

                if lines:
                    file.write(
                        "\n".join(
                            lines
                        )
                    )

                    file.write(
                        "\n"
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
                    elapsed
                    / index
                )

                remaining = (
                    len(images)
                    - index
                ) * speed

                print(
                    f"[{index:4d}/{len(images)}] "
                    f"{rgb_path.name} "
                    f"pred={len(lines):3d} "
                    f"elapsed={elapsed / 60:.1f} min "
                    f"ETA={remaining / 60:.1f} min"
                )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Validate TXT files
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("VALIDATING SUBMISSION TXT")
    print("=" * 72)

    validation = validate_txt_files(
        images
    )

    print(
        "TXT files           :",
        validation[
            "txt_files"
        ],
    )

    print(
        "Total predictions   :",
        validation[
            "total_boxes"
        ],
    )

    print(
        "Empty TXT files     :",
        validation[
            "empty_files"
        ],
    )

    print(
        "Max boxes / image   :",
        validation[
            "max_boxes_per_image"
        ],
    )

    print(
        "Class counts        :",
        validation[
            "class_counts"
        ],
    )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    print()
    print(
        "Creating ZIP..."
    )

    create_zip()

    validate_zip()

    zip_size_mb = (
        ZIP_PATH.stat().st_size
        / 1024
        / 1024
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = dict(
        experiment=EXP_NAME,

        checkpoint=str(
            CHECKPOINT_PATH
        ),

        test_images=len(
            images
        ),

        modalities=[
            "RGB",
            "Infrared",
        ],

        image_scale=list(
            IMAGE_SCALE
        ),

        keep_ratio=True,

        score_thr=SCORE_THR,

        nms_iou=NMS_IOU,

        max_det=MAX_DET,

        precision="FP32",

        raw_gates=[
            float(x)
            for x in raw_gates.tolist()
        ],

        alpha_gates=[
            float(x)
            for x in alpha_gates.tolist()
        ],

        inference_seconds=(
            elapsed
        ),

        validation=validation,

        zip_path=str(
            ZIP_PATH
        ),

        zip_size_mb=(
            zip_size_mb
        ),
    )

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE-1 SUBMISSION READY")
    print("=" * 72)

    print(
        "Images             :",
        len(images),
    )

    print(
        "TXT files          :",
        validation[
            "txt_files"
        ],
    )

    print(
        "Total predictions  :",
        validation[
            "total_boxes"
        ],
    )

    print(
        "Empty files        :",
        validation[
            "empty_files"
        ],
    )

    print(
        "Max boxes/image    :",
        validation[
            "max_boxes_per_image"
        ],
    )

    print(
        "Inference time     :",
        f"{elapsed / 60:.2f} min",
    )

    print(
        "ZIP size           :",
        f"{zip_size_mb:.2f} MB",
    )

    print()
    print(
        "Submission ZIP:"
    )

    print(
        ZIP_PATH
    )

    print()
    print(
        "Manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()
    print(
        "STATUS = READY TO SUBMIT"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()

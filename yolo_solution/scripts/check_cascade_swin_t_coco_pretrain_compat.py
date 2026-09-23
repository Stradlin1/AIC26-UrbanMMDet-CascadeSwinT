#!/usr/bin/env python3

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS


# ============================================================
# Paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

YOLO_SOLUTION_ROOT = SCRIPT_PATH.parents[1]
LDDS_ROOT = YOLO_SOLUTION_ROOT.parent

MMDET_ROOT = LDDS_ROOT / "mmdetection"

CONFIG_PATH = (
    MMDET_ROOT
    / "configs"
    / "aic2026"
    / "cascade-rcnn_swin-t_fpn_coco_smoke.py"
)

PRETRAINED_DIR = LDDS_ROOT / "pretrained"

CHECKPOINT_NAME = (
    "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
    "20210902_120937-9d6b7cfa.pth"
)

CHECKPOINT_PATH = PRETRAINED_DIR / CHECKPOINT_NAME

CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/"
    "swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/"
    "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
    "20210902_120937-9d6b7cfa.pth"
)


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
# Utilities
# ============================================================

def section(name):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)


def group_name(key):
    if key.startswith("backbone."):
        return "backbone"

    if key.startswith("neck."):
        return "neck/FPN"

    if key.startswith("rpn_head."):
        return "rpn_head"

    if key.startswith("roi_head.bbox_head."):
        return "roi_bbox_head"

    if key.startswith("roi_head.mask_head."):
        return "roi_mask_head"

    return "other"


def tensor_numel(tensor):
    return int(tensor.numel())


def download_checkpoint():
    PRETRAINED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CHECKPOINT_PATH.exists():
        size_mb = (
            CHECKPOINT_PATH.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"Checkpoint already exists: "
            f"{CHECKPOINT_PATH}"
        )
        print(
            f"Checkpoint size        : "
            f"{size_mb:.2f} MB"
        )

        return

    print("Downloading official MMDetection checkpoint:")
    print(CHECKPOINT_URL)
    print()
    print("Destination:")
    print(CHECKPOINT_PATH)

    torch.hub.download_url_to_file(
        CHECKPOINT_URL,
        str(CHECKPOINT_PATH),
        progress=True,
    )


def load_checkpoint_state_dict():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return checkpoint, cleaned


def build_aic_model():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    cfg = Config.fromfile(
        str(CONFIG_PATH)
    )

    init_default_scope(
        cfg.get(
            "default_scope",
            "mmdet",
        )
    )

    # Convert Cascade detector to AIC2026 12 classes.
    for head in cfg.model.roi_head.bbox_head:
        head.num_classes = len(CLASSES)

    model = MODELS.build(cfg.model)

    return model


# ============================================================
# Compatibility analysis
# ============================================================

def analyze(model, checkpoint_state):

    model_state = model.state_dict()

    matched = {}
    shape_mismatch = {}
    checkpoint_only = {}

    for key, checkpoint_tensor in checkpoint_state.items():

        if key not in model_state:
            checkpoint_only[key] = checkpoint_tensor
            continue

        model_tensor = model_state[key]

        if tuple(checkpoint_tensor.shape) == tuple(model_tensor.shape):
            matched[key] = checkpoint_tensor
        else:
            shape_mismatch[key] = (
                tuple(checkpoint_tensor.shape),
                tuple(model_tensor.shape),
            )

    model_missing = {
        key: value
        for key, value in model_state.items()
        if key not in matched
    }

    return (
        model_state,
        matched,
        shape_mismatch,
        checkpoint_only,
        model_missing,
    )


def print_group_report(
    model_state,
    checkpoint_state,
    matched,
    shape_mismatch,
):

    groups = (
        "backbone",
        "neck/FPN",
        "rpn_head",
        "roi_bbox_head",
        "roi_mask_head",
        "other",
    )

    stats = defaultdict(
        lambda: {
            "model_tensors": 0,
            "model_numel": 0,
            "ckpt_tensors": 0,
            "ckpt_numel": 0,
            "matched_tensors": 0,
            "matched_numel": 0,
            "shape_mismatch": 0,
        }
    )

    for key, tensor in model_state.items():
        group = group_name(key)

        stats[group]["model_tensors"] += 1
        stats[group]["model_numel"] += tensor_numel(tensor)

    for key, tensor in checkpoint_state.items():
        group = group_name(key)

        stats[group]["ckpt_tensors"] += 1
        stats[group]["ckpt_numel"] += tensor_numel(tensor)

    for key, tensor in matched.items():
        group = group_name(key)

        stats[group]["matched_tensors"] += 1
        stats[group]["matched_numel"] += tensor_numel(tensor)

    for key in shape_mismatch:
        group = group_name(key)
        stats[group]["shape_mismatch"] += 1

    print(
        f"{'group':16s}"
        f"{'model':>10s}"
        f"{'ckpt':>10s}"
        f"{'matched':>10s}"
        f"{'coverage':>12s}"
        f"{'shape bad':>12s}"
    )

    print("-" * 80)

    for group in groups:
        s = stats[group]

        if s["model_numel"]:
            coverage = (
                100.0
                * s["matched_numel"]
                / s["model_numel"]
            )
        else:
            coverage = 0.0

        print(
            f"{group:16s}"
            f"{s['model_tensors']:10d}"
            f"{s['ckpt_tensors']:10d}"
            f"{s['matched_tensors']:10d}"
            f"{coverage:11.2f}%"
            f"{s['shape_mismatch']:12d}"
        )

    return stats


def print_key_examples(
    title,
    items,
    limit=30,
):
    section(title)

    if not items:
        print("None")
        return

    if isinstance(items, dict):
        keys = sorted(items.keys())
    else:
        keys = sorted(items)

    for key in keys[:limit]:
        value = items[key]

        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
        ):
            print(
                f"{key}\n"
                f"    checkpoint: {value[0]}\n"
                f"    model     : {value[1]}"
            )
        else:
            print(key)

    if len(keys) > limit:
        print(
            f"... and {len(keys) - limit} more"
        )


def verify_partial_load(
    model,
    matched,
):
    section("Actual partial load test")

    result = model.load_state_dict(
        matched,
        strict=False,
    )

    print("Partial load executed successfully.")
    print()
    print(
        "Missing keys after partial load :",
        len(result.missing_keys),
    )

    print(
        "Unexpected keys after load      :",
        len(result.unexpected_keys),
    )

    if result.unexpected_keys:
        print()
        print("Unexpected examples:")

        for key in result.unexpected_keys[:20]:
            print(" ", key)


def main():

    section("Checkpoint")

    print("Model config :", CONFIG_PATH)
    print("Checkpoint   :", CHECKPOINT_PATH)

    download_checkpoint()

    section("Build target AIC2026 Cascade model")

    model = build_aic_model()

    model_state = model.state_dict()

    print("Target model tensors :", len(model_state))
    print(
        "Target model params  :",
        f"{sum(x.numel() for x in model.parameters()):,}",
    )

    section("Load COCO checkpoint")

    _, checkpoint_state = (
        load_checkpoint_state_dict()
    )

    print(
        "Checkpoint tensors:",
        len(checkpoint_state),
    )

    (
        model_state,
        matched,
        shape_mismatch,
        checkpoint_only,
        model_missing,
    ) = analyze(
        model=model,
        checkpoint_state=checkpoint_state,
    )

    section("Compatibility summary")

    stats = print_group_report(
        model_state=model_state,
        checkpoint_state=checkpoint_state,
        matched=matched,
        shape_mismatch=shape_mismatch,
    )

    model_numel = sum(
        tensor_numel(v)
        for v in model_state.values()
    )

    matched_numel = sum(
        tensor_numel(v)
        for v in matched.values()
    )

    print()
    print(
        "Overall matched tensors :",
        f"{len(matched)} / {len(model_state)}",
    )

    print(
        "Overall parameter cover :",
        f"{100.0 * matched_numel / model_numel:.2f}%",
    )

    print(
        "Shape mismatches        :",
        len(shape_mismatch),
    )

    print(
        "Checkpoint-only tensors :",
        len(checkpoint_only),
    )

    print(
        "Target missing tensors  :",
        len(model_missing),
    )

    print_key_examples(
        "Shape mismatches",
        shape_mismatch,
    )

    roi_checkpoint_only = {
        k: v
        for k, v in checkpoint_only.items()
        if k.startswith("roi_head.")
    }

    print_key_examples(
        "Checkpoint ROI keys not used by Cascade target",
        roi_checkpoint_only,
    )

    cascade_missing = {
        k: v
        for k, v in model_missing.items()
        if k.startswith("roi_head.")
    }

    print_key_examples(
        "Cascade ROI keys that remain randomly initialized",
        cascade_missing,
    )

    verify_partial_load(
        model=model,
        matched=matched,
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    section("Decision")

    backbone = stats["backbone"]
    neck = stats["neck/FPN"]
    rpn = stats["rpn_head"]

    backbone_ok = (
        backbone["model_numel"] > 0
        and backbone["matched_numel"]
        == backbone["model_numel"]
    )

    neck_ok = (
        neck["model_numel"] > 0
        and neck["matched_numel"]
        == neck["model_numel"]
    )

    rpn_ok = (
        rpn["model_numel"] > 0
        and rpn["matched_numel"]
        == rpn["model_numel"]
    )

    print("Backbone full compatible :", backbone_ok)
    print("FPN full compatible      :", neck_ok)
    print("RPN full compatible      :", rpn_ok)

    print()

    if backbone_ok and neck_ok and rpn_ok:
        print(
            "PASS: COCO detector checkpoint can safely "
            "initialize Swin-T + FPN + RPN."
        )

        print(
            "Cascade-specific ROI heads should remain "
            "AIC2026/random initialized."
        )
    else:
        print(
            "CHECK REQUIRED: one or more shared detector "
            "components are not fully compatible."
        )

        print(
            "Do not use this checkpoint for formal C02 "
            "until the mismatched keys are reviewed."
        )


if __name__ == "__main__":
    main()

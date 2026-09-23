#!/usr/bin/env python3

import os

# Same deterministic setting as C02.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import json
import sys
from copy import deepcopy
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


from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

# Register C04 custom classes before Runner builds config.
from c04_rgbd_components import (
    LoadRGBDepthFromFile,
    RGBDDetDataPreprocessor,
    DualSwinDepthGatedBackbone,
)


# ============================================================
# C02 reference
# ============================================================

C02_RUN_DIR = (
    LDDS_ROOT
    / "runs"
    / "c02_cascade_swin_t_rgb_1333x800"
)

C02_CONFIG = (
    C02_RUN_DIR
    / "resolved_config.py"
)


# ============================================================
# C04 parity checkpoint
# ============================================================

C04_RUN_DIR = (
    LDDS_ROOT
    / "runs"
    / "c04_cascade_swin_t_rgbd_gated_1333x800"
)

PARITY_CKPT = (
    C04_RUN_DIR
    / "parity"
    / "c04_gate0_from_c02_best_PARITY_ONLY.pth"
)

VAL_WORK_DIR = (
    C04_RUN_DIR
    / "parity_val400"
)

METRICS_JSON = (
    VAL_WORK_DIR
    / "metrics.json"
)


# ============================================================
# Dataset
# ============================================================

VAL_ANN = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
    / "instances_val.json"
)

VISIBLE_DIR = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
    / "visible"
)

DEPTH_DIR = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
    / "depth"
)


# ============================================================
# Helpers
# ============================================================

def check_paths():
    required = (
        MMDET_ROOT,
        C02_CONFIG,
        PARITY_CKPT,
        VAL_ANN,
        VISIBLE_DIR,
        DEPTH_DIR,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )


def verify_parity_checkpoint():
    checkpoint = torch.load(
        PARITY_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    meta = checkpoint.get(
        "meta",
        {},
    )

    purpose = str(
        meta.get(
            "purpose",
            "",
        )
    )

    warning = str(
        meta.get(
            "warning",
            "",
        )
    )

    print("Checkpoint purpose :", purpose)
    print("Checkpoint warning :", warning)

    if "parity" not in purpose.lower():
        raise RuntimeError(
            "Checkpoint is not marked as a parity checkpoint."
        )

    state_dict = checkpoint.get(
        "state_dict",
        {},
    )

    gate_key = "backbone.gates"

    if gate_key not in state_dict:
        raise RuntimeError(
            f"Missing gate tensor: {gate_key}"
        )

    gates = state_dict[
        gate_key
    ].float()

    print(
        "Stored raw gates    :",
        gates.tolist(),
    )

    if not torch.equal(
        gates,
        torch.zeros_like(gates),
    ):
        raise RuntimeError(
            "Parity checkpoint gates are not exactly zero."
        )


# ============================================================
# Build C04 config from actual C02 resolved config
# ============================================================

def build_cfg():
    cfg = Config.fromfile(
        str(C02_CONFIG)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    cfg.work_dir = str(
        VAL_WORK_DIR
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    original_backbone = deepcopy(
        cfg.model.backbone
    )

    original_backbone.init_cfg = None

    cfg.model.backbone = dict(
        type="DualSwinDepthGatedBackbone",

        rgb_backbone=deepcopy(
            original_backbone
        ),

        depth_backbone=deepcopy(
            original_backbone
        ),

        channels=(
            96,
            192,
            384,
            768,
        ),
    )

    # Preserve exact C02 preprocessing parameters.
    old_dp = deepcopy(
        cfg.model.data_preprocessor
    )

    cfg.model.data_preprocessor = dict(
        type="RGBDDetDataPreprocessor",

        mean=list(
            old_dp.mean
        ),

        std=list(
            old_dp.std
        ),

        bgr_to_rgb=bool(
            old_dp.bgr_to_rgb
        ),

        pad_size_divisor=int(
            old_dp.pad_size_divisor
        ),
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = deepcopy(
        cfg.val_dataloader.dataset
    )

    test_dataset.data_root = (
        str(LDDS_ROOT) + "/"
    )

    test_dataset.ann_file = (
        "generated/"
        "aic2026_coco/"
        "instances_val.json"
    )

    test_dataset.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    test_dataset.test_mode = True

    pipeline = deepcopy(
        test_dataset.pipeline
    )

    if len(pipeline) == 0:
        raise RuntimeError(
            "C02 validation pipeline is empty."
        )

    # C02:
    # LoadImageFromFile
    #
    # C04:
    # LoadRGBDepthFromFile
    #
    # Everything after this remains unchanged.
    pipeline[0] = dict(
        type="LoadRGBDepthFromFile",
    )

    test_dataset.pipeline = pipeline

    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,

        sampler=dict(
            type="DefaultSampler",
            shuffle=False,
        ),

        dataset=test_dataset,
    )

    # --------------------------------------------------------
    # COCO evaluator
    # --------------------------------------------------------

    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(
            VAL_ANN
        ),
        metric="bbox",
        format_only=False,
        classwise=True,
        backend_args=None,
    )

    cfg.test_cfg = dict(
        type="TestLoop",
    )

    # --------------------------------------------------------
    # Load parity checkpoint
    # --------------------------------------------------------

    cfg.load_from = str(
        PARITY_CKPT
    )

    cfg.resume = False

    # --------------------------------------------------------
    # Determinism
    # --------------------------------------------------------

    cfg.randomness = dict(
        seed=2026,
        deterministic=True,
    )

    cfg.env_cfg.cudnn_benchmark = False

    return cfg


# ============================================================
# Save metrics
# ============================================================

def save_metrics(metrics):
    VAL_WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable = {}

    for key, value in metrics.items():

        if hasattr(
            value,
            "item",
        ):
            value = value.item()

        serializable[
            str(key)
        ] = value

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():
    check_paths()

    VAL_WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_default_scope(
        "mmdet"
    )

    print("=" * 72)
    print("AIC2026 C04 GATE=0 PARITY VAL400")
    print("=" * 72)

    print("LDDS root       :", LDDS_ROOT)
    print("C02 config      :", C02_CONFIG)
    print("Parity checkpoint:", PARITY_CKPT)
    print("Val annotation  :", VAL_ANN)
    print("Visible         :", VISIBLE_DIR)
    print("Depth           :", DEPTH_DIR)
    print("Work dir        :", VAL_WORK_DIR)

    print()
    verify_parity_checkpoint()

    cfg = build_cfg()

    print()
    print("Architecture:")
    print(
        "RGB Swin-T + Depth Swin-T "
        "+ gated residual fusion "
        "+ FPN + RPN + Cascade"
    )

    print()
    print("Gate state      : exactly zero")
    print("Adapter state   : identity")
    print("Dataset         : val400")
    print("Input scale     : (1333, 800)")
    print("Batch size      : 1")
    print("Max detections  : 100")

    print()
    print("Starting val400 parity evaluation...")
    print()

    runner = Runner.from_cfg(
        cfg
    )

    metrics = runner.test()

    save_metrics(
        metrics
    )

    print()
    print("=" * 72)
    print("PARITY VAL400 RESULTS")
    print("=" * 72)

    for key in sorted(
        metrics.keys()
    ):
        print(
            f"{key:30s}: "
            f"{metrics[key]}"
        )

    print()
    print(
        "Metrics saved:",
        METRICS_JSON,
    )

    # Primary sanity check.
    main_key = (
        "coco/bbox_mAP"
    )

    if main_key not in metrics:
        raise RuntimeError(
            f"Expected metric missing: {main_key}"
        )

    print()
    print(
        "Primary mAP50-95 :",
        metrics[main_key],
    )

    # Strict C02 reference check. These are the frozen C02 val400
    # metrics under the same COCO evaluator protocol.
    expected = {
        "coco/bbox_mAP": 0.396,
        "coco/bbox_mAP_50": 0.746,
        "coco/bbox_mAP_75": 0.345,
        "coco/bbox_mAP_s": 0.233,
        "coco/bbox_mAP_m": 0.338,
        "coco/bbox_mAP_l": 0.547,
    }

    print()
    print("Strict C02 reference check:")

    failures = []

    for key, target in expected.items():
        if key not in metrics:
            failures.append(
                f"missing metric: {key}"
            )
            continue

        actual = float(metrics[key])
        diff = abs(actual - target)

        print(
            f"  {key:22s}: "
            f"actual={actual:.6f} "
            f"target={target:.6f} "
            f"diff={diff:.9f}"
        )

        if diff > 1e-6:
            failures.append(
                f"{key}: actual={actual}, target={target}"
            )

    if failures:
        print()
        print("STATUS = FAIL")
        for item in failures:
            print("  -", item)
        raise RuntimeError(
            "C04 gate=0 val400 does not reproduce the frozen C02 reference."
        )

    print()
    print("PASS: C04 gate=0 reproduces C02 val400 metrics exactly")
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()


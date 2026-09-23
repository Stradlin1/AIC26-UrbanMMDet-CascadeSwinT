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

# Register C03 custom classes before Runner builds config.
from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
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
# C03 parity checkpoint
# ============================================================

C03_RUN_DIR = (
    LDDS_ROOT
    / "runs"
    / "c03_cascade_swin_t_rgbir_gated_1333x800"
)

PARITY_CKPT = (
    C03_RUN_DIR
    / "parity"
    / "c03_gate0_from_c02_best_PARITY_ONLY.pth"
)

VAL_WORK_DIR = (
    C03_RUN_DIR
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

IR_DIR = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
    / "infrared"
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
        IR_DIR,
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
# Build C03 config from actual C02 resolved config
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
        type="DualSwinGatedBackbone",

        rgb_backbone=deepcopy(
            original_backbone
        ),

        ir_backbone=deepcopy(
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
        type="RGBIRDetDataPreprocessor",

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
    # C03:
    # LoadRGBIRFromFile
    #
    # Everything after this remains unchanged.
    pipeline[0] = dict(
        type="LoadRGBIRFromFile",
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
    print("AIC2026 C03 GATE=0 PARITY VAL400")
    print("=" * 72)

    print("LDDS root       :", LDDS_ROOT)
    print("C02 config      :", C02_CONFIG)
    print("Parity checkpoint:", PARITY_CKPT)
    print("Val annotation  :", VAL_ANN)
    print("Visible         :", VISIBLE_DIR)
    print("Infrared        :", IR_DIR)
    print("Work dir        :", VAL_WORK_DIR)

    print()
    verify_parity_checkpoint()

    cfg = build_cfg()

    print()
    print("Architecture:")
    print(
        "RGB Swin-T + IR Swin-T "
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

    print()
    print(
        "Expected C02 reference "
        "is approximately 0.396."
    )

    print()
    print("STATUS = COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()

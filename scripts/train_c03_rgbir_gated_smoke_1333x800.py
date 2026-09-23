#!/usr/bin/env python3

import os

# Must be set before importing torch.
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

# ldds/scripts/xxx.py -> ldds
LDDS_ROOT = SCRIPT_PATH.parents[1]

MMDET_ROOT = LDDS_ROOT / "mmdetection"

C03_COMPONENT_DIR = (
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
    str(C03_COMPONENT_DIR),
)


from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS

# Register custom C03 components.
from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
)


# ============================================================
# Experiment
# ============================================================

EXP_NAME = (
    "c03_cascade_swin_t_rgbir_gated_"
    "smoke_1333x800"
)

WORK_DIR = (
    LDDS_ROOT
    / "runs"
    / EXP_NAME
)


# ============================================================
# C02 reference config
#
# We derive C03 directly from the exact resolved C02 config.
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
# Exact COCO partial pretrain used by C02
# ============================================================

C02_SHARED_COCO_CKPT = (
    LDDS_ROOT
    / "pretrained"
    / (
        "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
        "backbone_fpn_rpn_only.pth"
    )
)

# C03-specific mapped initialization.
C03_COCO_INIT = (
    LDDS_ROOT
    / "pretrained"
    / (
        "c03_rgbir_swin_t_coco_"
        "dual_backbone_fpn_rpn_fusion_init.pth"
    )
)


# ============================================================
# Dataset
# ============================================================

DATA_ROOT = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)

VISIBLE_DIR = DATA_ROOT / "visible"
IR_DIR = DATA_ROOT / "infrared"

COCO_ROOT = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
)

TRAIN_ANN = (
    COCO_ROOT
    / "instances_train.json"
)

VAL_ANN = (
    COCO_ROOT
    / "instances_val.json"
)


# ============================================================
# Smoke settings
# ============================================================

IMAGE_SCALE = (
    1333,
    800,
)

TRAIN_IMAGES = 64
VAL_IMAGES = 32

TRAIN_INDICES = list(
    range(TRAIN_IMAGES)
)

VAL_INDICES = list(
    range(VAL_IMAGES)
)

BATCH_SIZE = 1
NUM_WORKERS = 2

EPOCHS = 1

SEED = 2026


# ============================================================
# Output
# ============================================================

SUMMARY_JSON = (
    WORK_DIR
    / "smoke_summary.json"
)


# ============================================================
# Helpers
# ============================================================

def check_paths():
    required = (
        MMDET_ROOT,
        C03_COMPONENT_DIR,
        C02_CONFIG,
        C02_SHARED_COCO_CKPT,
        DATA_ROOT,
        VISIBLE_DIR,
        IR_DIR,
        TRAIN_ANN,
        VAL_ANN,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )


def load_state_dict(path):
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
        if key.startswith("module."):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


def get_model(model):
    if hasattr(
        model,
        "module",
    ):
        return model.module

    return model


def state_delta_sum(
    initial_state,
    final_state,
    prefix,
):
    total = 0.0
    matched = 0

    for key, initial in (
        initial_state.items()
    ):
        if not key.startswith(
            prefix
        ):
            continue

        if key not in final_state:
            raise RuntimeError(
                "Final checkpoint missing tensor:\n"
                f"{key}"
            )

        final = final_state[
            key
        ]

        # Ignore integer/non-floating buffers.
        if not (
            torch.is_floating_point(
                initial
            )
            or torch.is_complex(
                initial
            )
        ):
            continue

        if (
            tuple(initial.shape)
            != tuple(final.shape)
        ):
            raise RuntimeError(
                "State shape mismatch:\n"
                f"{key}"
            )

        total += float(
            (
                final.float()
                - initial.float()
            )
            .abs()
            .sum()
            .item()
        )

        matched += 1

    if matched == 0:
        raise RuntimeError(
            f"No floating tensors matched prefix: {prefix}"
        )

    return total, matched


def resolve_epoch_checkpoint():
    candidate = (
        WORK_DIR
        / "epoch_1.pth"
    )

    if candidate.exists():
        return candidate

    last_pointer = (
        WORK_DIR
        / "last_checkpoint"
    )

    if last_pointer.exists():
        path_string = (
            last_pointer
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        path = Path(
            path_string
        )

        if not path.is_absolute():
            path = (
                WORK_DIR
                / path
            )

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate epoch-1 checkpoint."
    )


# ============================================================
# Build C03 configuration
# ============================================================

def build_cfg():
    cfg = Config.fromfile(
        str(C02_CONFIG)
    )

    cfg.work_dir = str(
        WORK_DIR
    )

    # --------------------------------------------------------
    # C03 dual backbone
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

    # --------------------------------------------------------
    # C03 RGB+IR preprocessing
    #
    # Numerical preprocessing remains exactly C02.
    # --------------------------------------------------------

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
    # Train pipeline
    # --------------------------------------------------------

    train_pipeline = [
        dict(
            type="LoadRGBIRFromFile",
        ),
        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),
        dict(
            type="Resize",
            scale=IMAGE_SCALE,
            keep_ratio=True,
        ),
        dict(
            type="RandomFlip",
            prob=0.5,
        ),
        dict(
            type="PackDetInputs",
        ),
    ]

    train_dataset = (
        cfg.train_dataloader.dataset
    )

    train_dataset.data_root = (
        str(LDDS_ROOT) + "/"
    )

    train_dataset.ann_file = (
        "generated/"
        "aic2026_coco/"
        "instances_train.json"
    )

    train_dataset.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    train_dataset.pipeline = (
        train_pipeline
    )

    train_dataset.indices = (
        TRAIN_INDICES
    )

    train_dataset.filter_cfg = dict(
        filter_empty_gt=False,
        min_size=32,
    )

    cfg.train_dataloader.batch_size = (
        BATCH_SIZE
    )

    cfg.train_dataloader.num_workers = (
        NUM_WORKERS
    )

    cfg.train_dataloader.persistent_workers = (
        True
    )

    # --------------------------------------------------------
    # Validation pipeline
    # --------------------------------------------------------

    val_pipeline = [
        dict(
            type="LoadRGBIRFromFile",
        ),
        dict(
            type="Resize",
            scale=IMAGE_SCALE,
            keep_ratio=True,
        ),
        dict(
            type="LoadAnnotations",
            with_bbox=True,
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

    val_dataset = (
        cfg.val_dataloader.dataset
    )

    val_dataset.data_root = (
        str(LDDS_ROOT) + "/"
    )

    val_dataset.ann_file = (
        "generated/"
        "aic2026_coco/"
        "instances_val.json"
    )

    val_dataset.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    val_dataset.pipeline = (
        val_pipeline
    )

    val_dataset.indices = (
        VAL_INDICES
    )

    val_dataset.test_mode = True

    cfg.val_dataloader.batch_size = 1

    cfg.val_dataloader.num_workers = (
        NUM_WORKERS
    )

    cfg.val_dataloader.persistent_workers = (
        True
    )

    cfg.val_dataloader.drop_last = False

    cfg.test_dataloader = deepcopy(
        cfg.val_dataloader
    )

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    cfg.val_evaluator = dict(
        type="CocoMetric",
        ann_file=str(
            VAL_ANN
        ),
        metric="bbox",
        format_only=False,
        classwise=True,
        backend_args=None,
    )

    cfg.test_evaluator = deepcopy(
        cfg.val_evaluator
    )

    # --------------------------------------------------------
    # One-epoch smoke only.
    #
    # Optimizer / LR / warmup remain exactly C02.
    # --------------------------------------------------------

    cfg.train_cfg.max_epochs = (
        EPOCHS
    )

    cfg.train_cfg.val_begin = 1
    cfg.train_cfg.val_interval = 1

    # --------------------------------------------------------
    # Logging / checkpoint
    # --------------------------------------------------------

    cfg.default_hooks.logger.interval = (
        10
    )

    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        interval=1,
        by_epoch=True,
        save_last=True,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=2,
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    cfg.randomness = dict(
        seed=SEED,
        deterministic=True,
    )

    cfg.env_cfg.cudnn_benchmark = False

    cfg.resume = False

    return cfg


# ============================================================
# Prepare C03 formal-style initialization
#
# Source:
#   exact backbone/FPN/RPN COCO checkpoint used by C02.
#
# Mapping:
#
#   C02 backbone.*
#       -> C03 backbone.rgb_backbone.*
#       -> C03 backbone.ir_backbone.*
#
#   C02 neck.*
#       -> C03 neck.*
#
#   C02 rpn_head.*
#       -> C03 rpn_head.*
#
# New:
#   adapters = identity
#   gates    = zero
#
# ROI heads remain random 12-class initialization.
# ============================================================

def prepare_c03_coco_init(cfg):
    print()
    print("=" * 72)
    print("Preparing C03 COCO initialization")
    print("=" * 72)

    source_state = load_state_dict(
        C02_SHARED_COCO_CKPT
    )

    target_model = MODELS.build(
        cfg.model
    )

    target_state = (
        target_model.state_dict()
    )

    mapped = {}

    rgb_count = 0
    ir_count = 0
    shared_count = 0

    used_source_keys = set()

    for source_key, source_tensor in (
        source_state.items()
    ):
        # ----------------------------------------------------
        # Swin backbone -> both branches
        # ----------------------------------------------------

        if source_key.startswith(
            "backbone."
        ):
            suffix = source_key[
                len("backbone.") :
            ]

            rgb_key = (
                "backbone."
                "rgb_backbone."
                + suffix
            )

            ir_key = (
                "backbone."
                "ir_backbone."
                + suffix
            )

            for target_key in (
                rgb_key,
                ir_key,
            ):
                if target_key not in target_state:
                    raise RuntimeError(
                        "Missing C03 target tensor:\n"
                        f"{target_key}"
                    )

                if (
                    tuple(
                        source_tensor.shape
                    )
                    != tuple(
                        target_state[
                            target_key
                        ].shape
                    )
                ):
                    raise RuntimeError(
                        "Backbone shape mismatch:\n"
                        f"{source_key}\n"
                        f"{target_key}"
                    )

                mapped[
                    target_key
                ] = (
                    source_tensor
                    .detach()
                    .clone()
                )

            rgb_count += 1
            ir_count += 1

            used_source_keys.add(
                source_key
            )

            continue

        # ----------------------------------------------------
        # FPN / RPN keep same names
        # ----------------------------------------------------

        if (
            source_key.startswith(
                "neck."
            )
            or source_key.startswith(
                "rpn_head."
            )
        ):
            if source_key not in target_state:
                raise RuntimeError(
                    "Missing shared C03 target tensor:\n"
                    f"{source_key}"
                )

            if (
                tuple(
                    source_tensor.shape
                )
                != tuple(
                    target_state[
                        source_key
                    ].shape
                )
            ):
                raise RuntimeError(
                    "Shared shape mismatch:\n"
                    f"{source_key}"
                )

            mapped[
                source_key
            ] = (
                source_tensor
                .detach()
                .clone()
            )

            shared_count += 1

            used_source_keys.add(
                source_key
            )

            continue

        raise RuntimeError(
            "Unexpected tensor in C02 shared "
            "COCO checkpoint:\n"
            f"{source_key}"
        )

    unmapped = sorted(
        set(source_state)
        - used_source_keys
    )

    if unmapped:
        raise RuntimeError(
            "Some C02 COCO tensors were not mapped."
        )

    # --------------------------------------------------------
    # Explicit fusion initialization
    # --------------------------------------------------------

    fusion_keys = []

    for key, tensor in (
        target_state.items()
    ):
        if (
            key == "backbone.gates"
            or key.startswith(
                "backbone.adapters."
            )
        ):
            mapped[key] = (
                tensor
                .detach()
                .clone()
            )

            fusion_keys.append(
                key
            )

    if (
        "backbone.gates"
        not in fusion_keys
    ):
        raise RuntimeError(
            "C03 gate tensor not found."
        )

    gates = mapped[
        "backbone.gates"
    ]

    if not torch.equal(
        gates,
        torch.zeros_like(
            gates
        ),
    ):
        raise RuntimeError(
            "C03 gates are not zero initialized."
        )

    checkpoint = dict(
        meta=dict(
            purpose=(
                "C03 formal-style initialization"
            ),

            source=(
                str(
                    C02_SHARED_COCO_CKPT
                )
            ),

            rgb_backbone=(
                "COCO detector backbone"
            ),

            ir_backbone=(
                "same COCO detector backbone"
            ),

            fpn=(
                "COCO detector FPN"
            ),

            rpn=(
                "COCO detector RPN"
            ),

            roi_head=(
                "random 12-class initialization"
            ),

            adapter_init="identity",

            gate_init=0.0,

            formal_training_source=(
                "public COCO pretrain, "
                "NOT C02 best checkpoint"
            ),
        ),

        state_dict=mapped,
    )

    C03_COCO_INIT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        C03_COCO_INIT,
    )

    print(
        "Source tensors      :",
        len(source_state),
    )

    print(
        "RGB backbone mapped :",
        rgb_count,
    )

    print(
        "IR backbone mapped  :",
        ir_count,
    )

    print(
        "FPN/RPN mapped      :",
        shared_count,
    )

    print(
        "Fusion tensors      :",
        len(fusion_keys),
    )

    print(
        "Raw gates           :",
        gates.tolist(),
    )

    print(
        "C03 init checkpoint :",
        C03_COCO_INIT,
    )

    print("=" * 72)

    cfg.load_from = str(
        C03_COCO_INIT
    )


# ============================================================
# Summary
# ============================================================

def print_summary(cfg):
    print()
    print("=" * 72)
    print("AIC2026 C03 RGB+IR GATED SMOKE")
    print("=" * 72)

    print(
        "Experiment     :",
        EXP_NAME,
    )

    print(
        "Work dir       :",
        WORK_DIR,
    )

    print()
    print(
        "Architecture   : "
        "Dual Swin-T + gated residual fusion "
        "+ FPN + RPN + Cascade"
    )

    print(
        "Modalities     : RGB + IR"
    )

    print(
        "Fusion         : "
        "RGB_i + tanh(g_i) * Adapter(IR_i)"
    )

    print(
        "Adapter init   : identity"
    )

    print(
        "Gate init      : zero"
    )

    print()
    print(
        "Train images   :",
        TRAIN_IMAGES,
    )

    print(
        "Val images     :",
        VAL_IMAGES,
    )

    print(
        "Input scale    :",
        IMAGE_SCALE,
    )

    print(
        "Batch size     :",
        BATCH_SIZE,
    )

    print(
        "Epochs         :",
        EPOCHS,
    )

    print()
    print(
        "RGB with_cp    :",
        cfg.model.backbone[
            "rgb_backbone"
        ]["with_cp"],
    )

    print(
        "IR with_cp     :",
        cfg.model.backbone[
            "ir_backbone"
        ]["with_cp"],
    )

    print(
        "AMP            :",
        cfg.optim_wrapper.type
        == "AmpOptimWrapper",
    )

    print()
    print(
        "Initialization :"
    )

    print(
        "  RGB Swin     : COCO detector"
    )

    print(
        "  IR Swin      : same COCO detector"
    )

    print(
        "  FPN          : COCO detector"
    )

    print(
        "  RPN          : COCO detector"
    )

    print(
        "  Cascade      : random 12-class"
    )

    print(
        "  C02 best     : NOT USED"
    )

    print("=" * 72)


# ============================================================
# Main
# ============================================================

def main():
    check_paths()

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_default_scope(
        "mmdet"
    )

    cfg = build_cfg()

    prepare_c03_coco_init(
        cfg
    )

    print_summary(
        cfg
    )

    resolved_config = (
        WORK_DIR
        / "resolved_config.py"
    )

    cfg.dump(
        str(
            resolved_config
        )
    )

    print()
    print(
        "Resolved config:",
        resolved_config,
    )

    print()
    print(
        "Building Runner..."
    )

    runner = Runner.from_cfg(
        cfg
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print()
    print("=" * 72)
    print("STARTING 1-EPOCH SMOKE TRAINING")
    print("=" * 72)
    print()

    runner.train()

    # --------------------------------------------------------
    # Gate state after real optimizer steps
    # --------------------------------------------------------

    model = get_model(
        runner.model
    )

    raw_gates = (
        model
        .backbone
        .gates
        .detach()
        .cpu()
    )

    alpha_gates = torch.tanh(
        raw_gates
    )

    print()
    print("=" * 72)
    print("POST-TRAIN GATES")
    print("=" * 72)

    print(
        "Raw gates :",
        [
            f"{x:.12e}"
            for x in raw_gates.tolist()
        ],
    )

    print(
        "Alpha     :",
        [
            f"{x:.12e}"
            for x in alpha_gates.tolist()
        ],
    )

    if torch.equal(
        raw_gates,
        torch.zeros_like(
            raw_gates
        ),
    ):
        raise RuntimeError(
            "All gates are still exactly zero "
            "after smoke training."
        )

    print(
        "PASS: optimizer moved gates away from zero"
    )

    # --------------------------------------------------------
    # Explicit validation once more so metrics are returned.
    # Only 32 images, so this is cheap.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL SMOKE VAL32")
    print("=" * 72)
    print()

    metrics = runner.val()

    # --------------------------------------------------------
    # Locate and reload checkpoint
    # --------------------------------------------------------

    epoch_ckpt = (
        resolve_epoch_checkpoint()
    )

    print()
    print(
        "Epoch checkpoint:",
        epoch_ckpt,
    )

    initial_state = load_state_dict(
        C03_COCO_INIT
    )

    final_state = load_state_dict(
        epoch_ckpt
    )

    # --------------------------------------------------------
    # Parameter-change evidence
    # --------------------------------------------------------

    rgb_delta, rgb_tensors = (
        state_delta_sum(
            initial_state,
            final_state,
            "backbone.rgb_backbone.",
        )
    )

    ir_delta, ir_tensors = (
        state_delta_sum(
            initial_state,
            final_state,
            "backbone.ir_backbone.",
        )
    )

    adapter_delta, adapter_tensors = (
        state_delta_sum(
            initial_state,
            final_state,
            "backbone.adapters.",
        )
    )

    gate_delta, gate_tensors = (
        state_delta_sum(
            initial_state,
            final_state,
            "backbone.gates",
        )
    )

    print()
    print("=" * 72)
    print("PARAMETER CHANGE FROM INITIALIZATION")
    print("=" * 72)

    print(
        "RGB backbone delta :",
        f"{rgb_delta:.9f}",
        f"({rgb_tensors} tensors)",
    )

    print(
        "IR backbone delta  :",
        f"{ir_delta:.9f}",
        f"({ir_tensors} tensors)",
    )

    print(
        "Adapters delta     :",
        f"{adapter_delta:.9f}",
        f"({adapter_tensors} tensors)",
    )

    print(
        "Gates delta        :",
        f"{gate_delta:.12f}",
        f"({gate_tensors} tensor)",
    )

    if rgb_delta <= 0:
        raise RuntimeError(
            "RGB backbone did not update."
        )

    if ir_delta <= 0:
        raise RuntimeError(
            "IR backbone did not update."
        )

    if adapter_delta <= 0:
        raise RuntimeError(
            "Adapters did not update."
        )

    if gate_delta <= 0:
        raise RuntimeError(
            "Gates did not update."
        )

    print()
    print(
        "PASS: RGB backbone updated"
    )

    print(
        "PASS: IR backbone updated"
    )

    print(
        "PASS: adapters updated"
    )

    print(
        "PASS: gates updated"
    )

    # --------------------------------------------------------
    # Strict checkpoint reload
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CHECKPOINT RELOAD TEST")
    print("=" * 72)

    reload_model = MODELS.build(
        deepcopy(
            cfg.model
        )
    )

    reload_result = (
        reload_model.load_state_dict(
            final_state,
            strict=True,
        )
    )

    print(
        "Strict load:",
        reload_result,
    )

    reloaded_gates = (
        reload_model
        .backbone
        .gates
        .detach()
        .cpu()
    )

    gate_reload_diff = float(
        (
            reloaded_gates
            - raw_gates
        )
        .abs()
        .max()
        .item()
    )

    print(
        "Gate reload max diff:",
        f"{gate_reload_diff:.12f}",
    )

    if gate_reload_diff != 0.0:
        raise RuntimeError(
            "Checkpoint gate reload parity failed."
        )

    print(
        "PASS: checkpoint strict reload"
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0

    if torch.cuda.is_available():
        peak_allocated_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024
            / 1024
            / 1024
        )

        peak_reserved_gb = (
            torch.cuda
            .max_memory_reserved()
            / 1024
            / 1024
            / 1024
        )

    # --------------------------------------------------------
    # Save smoke summary
    # --------------------------------------------------------

    serializable_metrics = {}

    for key, value in (
        metrics.items()
    ):
        if hasattr(
            value,
            "item",
        ):
            value = value.item()

        serializable_metrics[
            str(key)
        ] = value

    summary = dict(
        experiment=EXP_NAME,

        train_images=TRAIN_IMAGES,

        val_images=VAL_IMAGES,

        epochs=EPOCHS,

        image_scale=list(
            IMAGE_SCALE
        ),

        raw_gates=[
            float(x)
            for x in raw_gates.tolist()
        ],

        alpha_gates=[
            float(x)
            for x in alpha_gates.tolist()
        ],

        rgb_backbone_delta=rgb_delta,

        ir_backbone_delta=ir_delta,

        adapter_delta=adapter_delta,

        gate_delta=gate_delta,

        peak_allocated_gb=(
            peak_allocated_gb
        ),

        peak_reserved_gb=(
            peak_reserved_gb
        ),

        checkpoint=str(
            epoch_ckpt
        ),

        metrics=(
            serializable_metrics
        ),
    )

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("C03 SMOKE SUMMARY")
    print("=" * 72)

    print(
        "Train / val       :",
        f"{TRAIN_IMAGES} / {VAL_IMAGES}",
    )

    print(
        "Resolution        :",
        IMAGE_SCALE,
    )

    print(
        "Epochs            :",
        EPOCHS,
    )

    print(
        "Final raw gates   :",
        raw_gates.tolist(),
    )

    print(
        "Final alpha gates :",
        alpha_gates.tolist(),
    )

    if "coco/bbox_mAP" in metrics:
        print(
            "Val32 mAP50-95    :",
            metrics[
                "coco/bbox_mAP"
            ],
        )

    if torch.cuda.is_available():
        print(
            "Peak allocated    :",
            f"{peak_allocated_gb:.3f} GB",
        )

        print(
            "Peak reserved     :",
            f"{peak_reserved_gb:.3f} GB",
        )

    print(
        "Checkpoint        :",
        epoch_ckpt,
    )

    print(
        "Summary JSON      :",
        SUMMARY_JSON,
    )

    print()
    print(
        "Gate moved        : PASS"
    )

    print(
        "IR updated        : PASS"
    )

    print(
        "Adapter updated   : PASS"
    )

    print(
        "Checkpoint reload : PASS"
    )

    print()
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

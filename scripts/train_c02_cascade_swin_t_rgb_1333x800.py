#!/usr/bin/env python3

import os

# Required for deterministic CUDA/cuBLAS execution.
# Must be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from copy import deepcopy
from pathlib import Path

# ============================================================
# Project paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# /home/xhm/Desktop/ldds/scripts/xxx.py
# -> /home/xhm/Desktop/ldds
LDDS_ROOT = SCRIPT_PATH.parents[1]

MMDET_ROOT = LDDS_ROOT / "mmdetection"

# Prefer the local editable MMDetection source tree.
sys.path.insert(0, str(MMDET_ROOT))

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS


# ============================================================
# Experiment
# ============================================================

EXP_NAME = "c02_cascade_swin_t_rgb_1333x800"

WORK_DIR = (
    LDDS_ROOT
    / "runs"
    / EXP_NAME
)

BASE_CONFIG = (
    MMDET_ROOT
    / "configs"
    / "aic2026"
    / "cascade-rcnn_swin-t_fpn_coco_smoke.py"
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

COCO_ROOT = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
)

TRAIN_ANN = COCO_ROOT / "instances_train.json"
VAL_ANN = COCO_ROOT / "instances_val.json"

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

METAINFO = dict(
    classes=CLASSES,
)


# ============================================================
# COCO detector pretrain
# ============================================================

PRETRAINED_DIR = LDDS_ROOT / "pretrained"

FULL_COCO_CKPT_NAME = (
    "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
    "20210902_120937-9d6b7cfa.pth"
)

FULL_COCO_CKPT = (
    PRETRAINED_DIR
    / FULL_COCO_CKPT_NAME
)

FULL_COCO_CKPT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/"
    "swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/"
    "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
    "20210902_120937-9d6b7cfa.pth"
)

SHARED_COCO_CKPT = (
    PRETRAINED_DIR
    / "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
      "backbone_fpn_rpn_only.pth"
)

SHARED_PREFIXES = (
    "backbone.",
    "neck.",
    "rpn_head.",
)


# ============================================================
# Training settings
# ============================================================

IMAGE_SCALE = (1333, 800)

BATCH_SIZE = 1
NUM_WORKERS = 2

EPOCHS = 12

LR = 2.5e-5
WEIGHT_DECAY = 0.05

SEED = 2026

SCORE_THR = 0.001
MAX_DET = 100


# ============================================================
# Path checks
# ============================================================

def check_paths():
    required = (
        MMDET_ROOT,
        BASE_CONFIG,
        DATA_ROOT,
        VISIBLE_DIR,
        TRAIN_ANN,
        VAL_ANN,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )


# ============================================================
# Official COCO checkpoint
# ============================================================

def ensure_full_coco_checkpoint():
    PRETRAINED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FULL_COCO_CKPT.exists():
        size_mb = (
            FULL_COCO_CKPT.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"COCO checkpoint found : "
            f"{FULL_COCO_CKPT}"
        )

        print(
            f"Checkpoint size       : "
            f"{size_mb:.2f} MB"
        )

        return

    print()
    print("Official COCO detector checkpoint not found.")
    print("Downloading:")
    print(FULL_COCO_CKPT_URL)
    print()

    torch.hub.download_url_to_file(
        FULL_COCO_CKPT_URL,
        str(FULL_COCO_CKPT),
        progress=True,
    )

    print(
        "Checkpoint downloaded:",
        FULL_COCO_CKPT,
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
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# ============================================================
# Config
# ============================================================

def build_cfg():
    cfg = Config.fromfile(
        str(BASE_CONFIG)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    cfg.work_dir = str(WORK_DIR)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    cfg.model.backbone.with_cp = True

    # We use detector-level COCO weights instead.
    # Avoid separately loading ImageNet init_cfg.
    cfg.model.backbone.init_cfg = None

    # Three Cascade stages -> all 12 AIC classes.
    for bbox_head in cfg.model.roi_head.bbox_head:
        bbox_head.num_classes = len(CLASSES)

    # Competition maximum.
    cfg.model.test_cfg.rcnn.score_thr = SCORE_THR
    cfg.model.test_cfg.rcnn.max_per_img = MAX_DET

    # --------------------------------------------------------
    # Pipelines
    # --------------------------------------------------------

    train_pipeline = [
        dict(
            type="LoadImageFromFile",
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

    val_pipeline = [
        dict(
            type="LoadImageFromFile",
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

    # --------------------------------------------------------
    # Train dataset
    # --------------------------------------------------------

    train_dataset = cfg.train_dataloader.dataset

    train_dataset.data_root = (
        str(LDDS_ROOT) + "/"
    )

    train_dataset.ann_file = (
        "generated/aic2026_coco/"
        "instances_train.json"
    )

    train_dataset.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    train_dataset.metainfo = METAINFO

    # Keep the legitimate pure-background image.
    train_dataset.filter_cfg = dict(
        filter_empty_gt=False,
        min_size=32,
    )

    train_dataset.pipeline = train_pipeline

    # Defensive cleanup in case a smoke config ever contains it.
    train_dataset.pop(
        "indices",
        None,
    )

    cfg.train_dataloader.batch_size = (
        BATCH_SIZE
    )

    cfg.train_dataloader.num_workers = (
        NUM_WORKERS
    )

    cfg.train_dataloader.persistent_workers = True

    cfg.train_dataloader.sampler = dict(
        type="DefaultSampler",
        shuffle=True,
    )

    cfg.train_dataloader.batch_sampler = dict(
        type="AspectRatioBatchSampler",
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = cfg.val_dataloader.dataset

    val_dataset.data_root = (
        str(LDDS_ROOT) + "/"
    )

    val_dataset.ann_file = (
        "generated/aic2026_coco/"
        "instances_val.json"
    )

    val_dataset.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    val_dataset.metainfo = METAINFO
    val_dataset.test_mode = True
    val_dataset.pipeline = val_pipeline

    val_dataset.pop(
        "indices",
        None,
    )

    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = (
        NUM_WORKERS
    )
    cfg.val_dataloader.persistent_workers = True
    cfg.val_dataloader.drop_last = False

    cfg.val_dataloader.sampler = dict(
        type="DefaultSampler",
        shuffle=False,
    )

    # Test config mirrors validation for later use.
    cfg.test_dataloader = deepcopy(
        cfg.val_dataloader
    )

    # --------------------------------------------------------
    # COCO evaluator
    # --------------------------------------------------------

    cfg.val_evaluator = dict(
        type="CocoMetric",
        ann_file=str(VAL_ANN),
        metric="bbox",
        format_only=False,
        classwise=True,
        backend_args=None,
    )

    cfg.test_evaluator = deepcopy(
        cfg.val_evaluator
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    cfg.optim_wrapper = dict(
        type="AmpOptimWrapper",
        loss_scale="dynamic",
        optimizer=dict(
            type="AdamW",
            lr=LR,
            betas=(0.9, 0.999),
            weight_decay=WEIGHT_DECAY,
        ),
        paramwise_cfg=dict(
            custom_keys={
                "absolute_pos_embed": dict(
                    decay_mult=0.0,
                ),
                "relative_position_bias_table": dict(
                    decay_mult=0.0,
                ),
                "norm": dict(
                    decay_mult=0.0,
                ),
            },
        ),
    )

    # --------------------------------------------------------
    # LR schedule
    #
    # 500-iteration linear warmup,
    # then official 1x-style milestones 8 / 11.
    # --------------------------------------------------------

    cfg.param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=500,
        ),
        dict(
            type="MultiStepLR",
            by_epoch=True,
            begin=0,
            end=EPOCHS,
            milestones=[8, 11],
            gamma=0.1,
        ),
    ]

    # --------------------------------------------------------
    # Train / val loops
    # --------------------------------------------------------

    cfg.train_cfg = dict(
        type="EpochBasedTrainLoop",
        max_epochs=EPOCHS,
        val_begin=1,
        val_interval=1,
    )

    cfg.val_cfg = dict(
        type="ValLoop",
    )

    cfg.test_cfg = dict(
        type="TestLoop",
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    cfg.randomness = dict(
        seed=SEED,
        deterministic=True,
    )

    cfg.env_cfg.cudnn_benchmark = False

    # --------------------------------------------------------
    # Logging / checkpoint
    # --------------------------------------------------------

    cfg.default_hooks.logger.interval = 50

    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        interval=1,
        by_epoch=True,
        save_last=True,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=3,
    )

    # Do not auto-scale LR.
    cfg.auto_scale_lr = dict(
        enable=False,
        base_batch_size=16,
    )

    cfg.resume = False

    return cfg


# ============================================================
# Build compatible detector-level checkpoint
# ============================================================

def prepare_shared_coco_checkpoint(cfg):
    print()
    print("=" * 72)
    print("Preparing COCO detector partial pretrain")
    print("=" * 72)

    ensure_full_coco_checkpoint()

    init_default_scope(
        cfg.get(
            "default_scope",
            "mmdet",
        )
    )

    # Build target architecture only for compatibility checking.
    target_model = MODELS.build(
        cfg.model
    )

    target_state = (
        target_model.state_dict()
    )

    coco_state = load_state_dict(
        FULL_COCO_CKPT
    )

    expected_shared_keys = {
        key
        for key in target_state
        if key.startswith(
            SHARED_PREFIXES
        )
    }

    filtered_state = {}
    shape_mismatch = []

    for key, source_tensor in coco_state.items():

        if not key.startswith(
            SHARED_PREFIXES
        ):
            continue

        if key not in target_state:
            continue

        target_tensor = target_state[key]

        if (
            tuple(source_tensor.shape)
            != tuple(target_tensor.shape)
        ):
            shape_mismatch.append(
                (
                    key,
                    tuple(source_tensor.shape),
                    tuple(target_tensor.shape),
                )
            )
            continue

        filtered_state[key] = (
            source_tensor
        )

    missing_shared = sorted(
        expected_shared_keys
        - set(filtered_state)
    )

    if shape_mismatch:
        print()
        print("Shape mismatches:")

        for item in shape_mismatch[:20]:
            print(item)

        raise RuntimeError(
            "Shared COCO checkpoint contains "
            "shape mismatches."
        )

    if missing_shared:
        print()
        print("Missing shared keys:")

        for key in missing_shared[:30]:
            print(key)

        raise RuntimeError(
            "COCO checkpoint does not fully cover "
            "backbone + FPN + RPN."
        )

    matched_numel = sum(
        tensor.numel()
        for tensor in filtered_state.values()
    )

    total_shared_numel = sum(
        target_state[key].numel()
        for key in expected_shared_keys
    )

    coverage = (
        100.0
        * matched_numel
        / total_shared_numel
    )

    checkpoint = dict(
        meta=dict(
            source=str(
                FULL_COCO_CKPT
            ),
            components=(
                "backbone",
                "neck/FPN",
                "rpn_head",
            ),
            excluded=(
                "roi_head",
                "mask_head",
                "COCO classification heads",
            ),
        ),
        state_dict=filtered_state,
    )

    torch.save(
        checkpoint,
        SHARED_COCO_CKPT,
    )

    print(
        "Shared tensors       :",
        len(filtered_state),
    )

    print(
        "Shared parameter cov.:",
        f"{coverage:.2f}%",
    )

    print(
        "Partial checkpoint   :",
        SHARED_COCO_CKPT,
    )

    print(
        "Cascade ROI heads    : "
        "random initialized"
    )

    print("=" * 72)

    # Important: let Runner load only this filtered checkpoint.
    cfg.load_from = str(
        SHARED_COCO_CKPT
    )


# ============================================================
# Summary
# ============================================================

def print_summary(cfg):
    print()
    print("=" * 72)
    print("AIC2026 C02 FORMAL TRAINING")
    print("=" * 72)

    print("Experiment        :", EXP_NAME)
    print("LDDS root         :", LDDS_ROOT)
    print("Work dir          :", WORK_DIR)

    print()
    print("Architecture      : Cascade R-CNN + Swin-T + FPN")
    print("Modality          : RGB only")
    print("Classes           :", len(CLASSES))

    print()
    print("Train images      : 1600")
    print("Val images        : 400")
    print("Input scale       :", IMAGE_SCALE)
    print("Keep ratio        : True")

    print()
    print("Batch size        :", BATCH_SIZE)
    print("AMP               : True")
    print("Swin with_cp      :", cfg.model.backbone.with_cp)
    print("Epochs            :", EPOCHS)

    print()
    print("Optimizer         : AdamW")
    print("LR                :", LR)
    print("Weight decay      :", WEIGHT_DECAY)
    print("Warmup iterations : 500")
    print("LR milestones     : [8, 11]")

    print()
    print("Seed              :", SEED)
    print("Deterministic     : True")
    print("Score threshold   :", SCORE_THR)
    print("Max det / image   :", MAX_DET)

    print()
    print("Pretrain:")
    print("  Swin-T backbone : COCO detector")
    print("  FPN             : COCO detector")
    print("  RPN             : COCO detector")
    print("  Cascade stage 1 : random / 12 classes")
    print("  Cascade stage 2 : random / 12 classes")
    print("  Cascade stage 3 : random / 12 classes")

    print()
    print("Cascade heads:")

    for index, head in enumerate(
        cfg.model.roi_head.bbox_head,
        start=1,
    ):
        print(
            f"  stage {index}: "
            f"num_classes={head.num_classes}"
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

    cfg = build_cfg()

    prepare_shared_coco_checkpoint(
        cfg
    )

    print_summary(cfg)

    # Save the exact resolved configuration used by C02.
    resolved_config = (
        WORK_DIR
        / "resolved_config.py"
    )

    cfg.dump(
        str(resolved_config)
    )

    print()
    print(
        "Resolved config saved:",
        resolved_config,
    )

    print()
    print("Starting formal training...")
    print()

    runner = Runner.from_cfg(
        cfg
    )

    runner.train()


if __name__ == "__main__":
    main()

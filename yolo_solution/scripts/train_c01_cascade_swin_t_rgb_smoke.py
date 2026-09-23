#!/usr/bin/env python3

import os

# Required by PyTorch deterministic CUDA execution.
# Must be set before importing PyTorch / MMEngine.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


# ============================================================
# Project paths
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

DATA_ROOT = LDDS_ROOT

TRAIN_ANN = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
    / "instances_train.json"
)

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

WORK_DIR = (
    LDDS_ROOT
    / "runs"
    / "c01_cascade_swin_t_rgb_smoke"
)


# ============================================================
# Dataset definition
# ============================================================

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
# Smoke settings
# ============================================================

TRAIN_SMOKE_IMAGES = 64
VAL_SMOKE_IMAGES = 32

IMAGE_SCALE = (960, 960)

BATCH_SIZE = 1
NUM_WORKERS = 2

SEED = 2026


def check_paths():
    required = (
        MMDET_ROOT,
        CONFIG_PATH,
        TRAIN_ANN,
        VAL_ANN,
        VISIBLE_DIR,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )


def build_cfg():
    cfg = Config.fromfile(str(CONFIG_PATH))

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    cfg.work_dir = str(WORK_DIR)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    # Swin-T gradient checkpointing helps the 8 GB GPU.
    cfg.model.backbone.with_cp = True

    # Cascade has three bbox heads.
    for bbox_head in cfg.model.roi_head.bbox_head:
        bbox_head.num_classes = len(CLASSES)

    # Competition / validation convention.
    cfg.model.test_cfg.rcnn.score_thr = 0.001
    cfg.model.test_cfg.rcnn.max_per_img = 100

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

    train_dataset.data_root = str(DATA_ROOT) + "/"

    train_dataset.ann_file = (
        "generated/aic2026_coco/"
        "instances_train.json"
    )

    train_dataset.data_prefix = dict(
        img="datasets/AIC2026_Train_2000/visible/",
    )

    train_dataset.metainfo = METAINFO

    # Preserve the one legitimate background image.
    train_dataset.filter_cfg = dict(
        filter_empty_gt=False,
        min_size=32,
    )

    train_dataset.pipeline = train_pipeline

    # Only first 64 images for smoke.
    train_dataset.indices = TRAIN_SMOKE_IMAGES

    cfg.train_dataloader.batch_size = BATCH_SIZE
    cfg.train_dataloader.num_workers = NUM_WORKERS
    cfg.train_dataloader.persistent_workers = True

    # --------------------------------------------------------
    # Val dataset
    # --------------------------------------------------------

    val_dataset = cfg.val_dataloader.dataset

    val_dataset.data_root = str(DATA_ROOT) + "/"

    val_dataset.ann_file = (
        "generated/aic2026_coco/"
        "instances_val.json"
    )

    val_dataset.data_prefix = dict(
        img="datasets/AIC2026_Train_2000/visible/",
    )

    val_dataset.metainfo = METAINFO
    val_dataset.pipeline = val_pipeline
    val_dataset.indices = VAL_SMOKE_IMAGES

    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = NUM_WORKERS
    cfg.val_dataloader.persistent_workers = True

    # Test is not used during smoke, but keep it consistent.
    cfg.test_dataloader = cfg.val_dataloader

    # --------------------------------------------------------
    # COCO evaluator
    # --------------------------------------------------------

    cfg.val_evaluator.ann_file = str(VAL_ANN)
    cfg.val_evaluator.metric = "bbox"
    cfg.val_evaluator.classwise = True

    cfg.test_evaluator = cfg.val_evaluator

    # --------------------------------------------------------
    # Optimizer
    #
    # Swin official configs use AdamW.
    # Smoke uses conservative LR because the goal is stability,
    # not final accuracy.
    # --------------------------------------------------------

    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.optim_wrapper.loss_scale = "dynamic"

    cfg.optim_wrapper.optimizer.lr = 2.5e-5

    # No LR schedule is needed for a 1-epoch smoke.
    cfg.param_scheduler = []

    # --------------------------------------------------------
    # Train / val loop
    # --------------------------------------------------------

    cfg.train_cfg = dict(
        type="EpochBasedTrainLoop",
        max_epochs=1,
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

    # --------------------------------------------------------
    # Logging / checkpoint
    # --------------------------------------------------------

    cfg.default_hooks.logger.interval = 10

    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2

    cfg.resume = False
    cfg.load_from = None

    return cfg


def print_summary(cfg):
    print("=" * 72)
    print("AIC2026 Cascade R-CNN + Swin-T RGB smoke")
    print("=" * 72)

    print("LDDS root        :", LDDS_ROOT)
    print("MMDetection      :", MMDET_ROOT)
    print("Config           :", CONFIG_PATH)
    print("Train annotation :", TRAIN_ANN)
    print("Val annotation   :", VAL_ANN)
    print("Visible images   :", VISIBLE_DIR)
    print("Work dir         :", cfg.work_dir)

    print()
    print("Classes          :", len(CLASSES))
    print("Train smoke imgs :", TRAIN_SMOKE_IMAGES)
    print("Val smoke imgs   :", VAL_SMOKE_IMAGES)
    print("Image scale      :", IMAGE_SCALE)
    print("Batch size       :", BATCH_SIZE)
    print("AMP              :", cfg.optim_wrapper.type)
    print("Max det/image    :", cfg.model.test_cfg.rcnn.max_per_img)
    print("Score threshold  :", cfg.model.test_cfg.rcnn.score_thr)
    print("Swin with_cp     :", cfg.model.backbone.with_cp)

    print()
    print("Cascade heads:")

    for i, head in enumerate(
        cfg.model.roi_head.bbox_head,
        start=1,
    ):
        print(
            f"  stage {i}: "
            f"num_classes={head.num_classes}"
        )

    print("=" * 72)


def main():
    check_paths()

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = build_cfg()

    print_summary(cfg)

    runner = Runner.from_cfg(cfg)

    runner.train()


if __name__ == "__main__":
    main()

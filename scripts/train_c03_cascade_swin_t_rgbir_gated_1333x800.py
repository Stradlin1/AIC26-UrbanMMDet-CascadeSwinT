#!/usr/bin/env python3

import os

# Required for deterministic CUDA/cuBLAS execution.
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

MMDET_ROOT = (
    LDDS_ROOT
    / "mmdetection"
)

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
from mmengine.hooks import Hook
from mmengine.registry import (
    HOOKS,
    init_default_scope,
)
from mmengine.runner import Runner

from mmdet.registry import MODELS

# Register C03 components.
from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
)


# ============================================================
# Experiment
# ============================================================

EXP_NAME = (
    "c03_cascade_swin_t_"
    "rgbir_gated_1333x800"
)

WORK_DIR = (
    LDDS_ROOT
    / "runs"
    / EXP_NAME
)

GATE_HISTORY = (
    WORK_DIR
    / "gate_history.jsonl"
)

SETUP_JSON = (
    WORK_DIR
    / "experiment_setup.json"
)


# ============================================================
# C02 reference
#
# C03 is derived from the exact resolved C02 configuration.
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
# Public COCO initialization
#
# This is the exact backbone/FPN/RPN partial checkpoint
# prepared and used by C02.
# ============================================================

C02_SHARED_COCO_CKPT = (
    LDDS_ROOT
    / "pretrained"
    / (
        "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_"
        "backbone_fpn_rpn_only.pth"
    )
)

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

VISIBLE_DIR = (
    DATA_ROOT
    / "visible"
)

IR_DIR = (
    DATA_ROOT
    / "infrared"
)

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
# Formal experiment settings
# ============================================================

IMAGE_SCALE = (
    1333,
    800,
)

BATCH_SIZE = 1
NUM_WORKERS = 2

EPOCHS = 12

LR = 2.5e-5
WEIGHT_DECAY = 0.05

SEED = 2026

SCORE_THR = 0.001
MAX_DET = 100


# ============================================================
# Gate monitoring hook
# ============================================================

@HOOKS.register_module()
class C03GateMonitorHook(Hook):
    """
    Record raw gates and tanh(alpha) after every train epoch.
    """

    def _get_model(self, runner):
        model = runner.model

        if hasattr(
            model,
            "module",
        ):
            model = model.module

        return model

    def before_train(
        self,
        runner,
    ):
        model = self._get_model(
            runner
        )

        raw = (
            model
            .backbone
            .gates
            .detach()
            .cpu()
        )

        if not torch.equal(
            raw,
            torch.zeros_like(
                raw
            ),
        ):
            raise RuntimeError(
                "Formal C03 must start with "
                "exactly zero gates."
            )

        runner.logger.info(
            "C03 formal initial raw gates: "
            f"{raw.tolist()}"
        )

    def after_train_epoch(
        self,
        runner,
    ):
        model = self._get_model(
            runner
        )

        raw = (
            model
            .backbone
            .gates
            .detach()
            .float()
            .cpu()
        )

        alpha = torch.tanh(
            raw
        )

        record = dict(
            epoch=int(
                runner.epoch + 1
            ),

            raw_gates=[
                float(x)
                for x in raw.tolist()
            ],

            alpha_gates=[
                float(x)
                for x in alpha.tolist()
            ],
        )

        if torch.cuda.is_available():
            record[
                "peak_allocated_gb"
            ] = (
                torch.cuda
                .max_memory_allocated()
                / 1024
                / 1024
                / 1024
            )

            record[
                "peak_reserved_gb"
            ] = (
                torch.cuda
                .max_memory_reserved()
                / 1024
                / 1024
                / 1024
            )

        path = (
            Path(
                runner.work_dir
            )
            / "gate_history.jsonl"
        )

        with open(
            path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        runner.logger.info(
            "C03 gates epoch "
            f"{runner.epoch + 1}: "
            f"raw={record['raw_gates']} "
            f"alpha={record['alpha_gates']}"
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


def check_fresh_work_dir():
    """
    Prevent accidental overwrite of an existing formal run.
    """

    if not WORK_DIR.exists():
        return

    existing = []

    for pattern in (
        "epoch_*.pth",
        "best_*.pth",
    ):
        existing.extend(
            WORK_DIR.glob(
                pattern
            )
        )

    if existing:
        print()
        print(
            "Existing formal checkpoints:"
        )

        for path in sorted(
            existing
        ):
            print(
                " ",
                path,
            )

        raise RuntimeError(
            "Formal C03 work directory already contains "
            "checkpoints. Refusing to overwrite it."
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

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


# ============================================================
# Build formal C03 configuration
# ============================================================

def build_cfg():
    cfg = Config.fromfile(
        str(C02_CONFIG)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    cfg.work_dir = str(
        WORK_DIR
    )

    # --------------------------------------------------------
    # Backbone
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
    # RGB + IR preprocessing
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
    # Competition inference settings
    # --------------------------------------------------------

    cfg.model.test_cfg.rcnn.score_thr = (
        SCORE_THR
    )

    cfg.model.test_cfg.rcnn.max_per_img = (
        MAX_DET
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

    train_dataset.filter_cfg = dict(
        filter_empty_gt=False,
        min_size=32,
    )

    # Formal experiment uses all 1600 train images.
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

    cfg.train_dataloader.persistent_workers = (
        True
    )

    cfg.train_dataloader.sampler = dict(
        type="DefaultSampler",
        shuffle=True,
    )

    cfg.train_dataloader.batch_sampler = dict(
        type="AspectRatioBatchSampler",
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

    val_dataset.test_mode = True

    # Formal experiment uses all 400 validation images.
    val_dataset.pop(
        "indices",
        None,
    )

    cfg.val_dataloader.batch_size = 1

    cfg.val_dataloader.num_workers = (
        NUM_WORKERS
    )

    cfg.val_dataloader.persistent_workers = (
        True
    )

    cfg.val_dataloader.drop_last = False

    cfg.val_dataloader.sampler = dict(
        type="DefaultSampler",
        shuffle=False,
    )

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
    # Optimizer
    # --------------------------------------------------------

    cfg.optim_wrapper = dict(
        type="AmpOptimWrapper",

        loss_scale="dynamic",

        optimizer=dict(
            type="AdamW",

            lr=LR,

            betas=(
                0.9,
                0.999,
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),
        ),

        paramwise_cfg=dict(
            custom_keys={
                "absolute_pos_embed":
                    dict(
                        decay_mult=0.0,
                    ),

                "relative_position_bias_table":
                    dict(
                        decay_mult=0.0,
                    ),

                "norm":
                    dict(
                        decay_mult=0.0,
                    ),
            },
        ),
    )

    # --------------------------------------------------------
    # Exact C02 LR schedule
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

            milestones=[
                8,
                11,
            ],

            gamma=0.1,
        ),
    ]

    # --------------------------------------------------------
    # Train / val
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
    # Determinism
    # --------------------------------------------------------

    cfg.randomness = dict(
        seed=SEED,
        deterministic=True,
    )

    cfg.env_cfg.cudnn_benchmark = False

    # --------------------------------------------------------
    # Logging / checkpoint
    # --------------------------------------------------------

    cfg.default_hooks.logger.interval = (
        50
    )

    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",

        interval=1,

        by_epoch=True,

        save_last=True,

        save_best="coco/bbox_mAP",

        rule="greater",

        max_keep_ckpts=3,
    )

    # Record gates every epoch.
    cfg.custom_hooks = [
        dict(
            type="C03GateMonitorHook",
        ),
    ]

    cfg.auto_scale_lr = dict(
        enable=False,
        base_batch_size=16,
    )

    cfg.resume = False

    return cfg


# ============================================================
# Prepare formal C03 initialization
#
# Source:
#   exact public COCO backbone/FPN/RPN checkpoint used by C02.
#
# NO C02 best weights are used.
# ============================================================

def prepare_c03_coco_init(cfg):
    print()
    print("=" * 72)
    print("Preparing C03 formal COCO initialization")
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

    used_source = set()

    rgb_count = 0
    ir_count = 0
    shared_count = 0

    # --------------------------------------------------------
    # Public COCO backbone/FPN/RPN
    # --------------------------------------------------------

    for source_key, source_tensor in (
        source_state.items()
    ):

        # Swin backbone -> RGB and IR.
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
                        "Missing target key:\n"
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

            used_source.add(
                source_key
            )

            continue

        # FPN / RPN retain their C02 names.
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
                    "Missing shared target key:\n"
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

            used_source.add(
                source_key
            )

            continue

        raise RuntimeError(
            "Unexpected key in C02 shared "
            "COCO checkpoint:\n"
            f"{source_key}"
        )

    unmapped_source = sorted(
        set(
            source_state
        )
        - used_source
    )

    if unmapped_source:
        raise RuntimeError(
            "Some public COCO tensors were not mapped."
        )

    # --------------------------------------------------------
    # New C03 fusion parameters
    #
    # DualSwinGatedBackbone constructor guarantees:
    # adapters = identity
    # gates    = zero
    # --------------------------------------------------------

    fusion_keys = []

    for key, tensor in (
        target_state.items()
    ):
        if (
            key
            == "backbone.gates"
            or key.startswith(
                "backbone.adapters."
            )
        ):
            mapped[
                key
            ] = (
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
            "C03 gate tensor was not found."
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
            "Formal C03 gates are not exactly zero."
        )

    checkpoint = dict(
        meta=dict(
            experiment=EXP_NAME,

            purpose=(
                "C03 formal initialization"
            ),

            source_public_checkpoint=str(
                C02_SHARED_COCO_CKPT
            ),

            rgb_swin=(
                "public COCO detector backbone"
            ),

            ir_swin=(
                "same public COCO detector backbone"
            ),

            fpn=(
                "public COCO detector"
            ),

            rpn=(
                "public COCO detector"
            ),

            cascade_heads=(
                "random 12-class initialization"
            ),

            adapter_init=(
                "identity"
            ),

            gate_init=0.0,

            c02_best_used=False,
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
        "Source tensors       :",
        len(source_state),
    )

    print(
        "RGB backbone mapped  :",
        rgb_count,
    )

    print(
        "IR backbone mapped   :",
        ir_count,
    )

    print(
        "FPN/RPN mapped       :",
        shared_count,
    )

    print(
        "Fusion tensors       :",
        len(fusion_keys),
    )

    print(
        "Initial raw gates    :",
        gates.tolist(),
    )

    print(
        "Initialization file  :",
        C03_COCO_INIT,
    )

    print(
        "C02 best used        : False"
    )

    print("=" * 72)

    cfg.load_from = str(
        C03_COCO_INIT
    )


# ============================================================
# Experiment summary
# ============================================================

def print_summary(cfg):
    print()
    print("=" * 72)
    print("AIC2026 C03 FORMAL TRAINING")
    print("=" * 72)

    print(
        "Experiment        :",
        EXP_NAME,
    )

    print(
        "LDDS root         :",
        LDDS_ROOT,
    )

    print(
        "Work dir          :",
        WORK_DIR,
    )

    print()
    print(
        "Architecture      : "
        "Cascade R-CNN + Dual Swin-T "
        "+ gated residual fusion + FPN"
    )

    print(
        "Modalities        : RGB + IR"
    )

    print(
        "Fusion            : "
        "RGB_i + tanh(g_i) * Adapter(IR_i)"
    )

    print(
        "Adapter init      : identity"
    )

    print(
        "Gate init         : zero"
    )

    print()
    print(
        "Train images      : 1600"
    )

    print(
        "Val images        : 400"
    )

    print(
        "Input scale       :",
        IMAGE_SCALE,
    )

    print(
        "Keep ratio        : True"
    )

    print()
    print(
        "Batch size        :",
        BATCH_SIZE,
    )

    print(
        "AMP               : True"
    )

    print(
        "RGB with_cp       :",
        cfg.model.backbone[
            "rgb_backbone"
        ]["with_cp"],
    )

    print(
        "IR with_cp        :",
        cfg.model.backbone[
            "ir_backbone"
        ]["with_cp"],
    )

    print(
        "Epochs            :",
        EPOCHS,
    )

    print()
    print(
        "Optimizer         : AdamW"
    )

    print(
        "LR                :",
        LR,
    )

    print(
        "Weight decay      :",
        WEIGHT_DECAY,
    )

    print(
        "Warmup iterations : 500"
    )

    print(
        "LR milestones     : [8, 11]"
    )

    print()
    print(
        "Seed              :",
        SEED,
    )

    print(
        "Deterministic     : True"
    )

    print(
        "Score threshold   :",
        SCORE_THR,
    )

    print(
        "Max det / image   :",
        MAX_DET,
    )

    print()
    print("Pretrain:")

    print(
        "  RGB Swin-T      : public COCO detector"
    )

    print(
        "  IR Swin-T       : same public COCO detector"
    )

    print(
        "  FPN             : public COCO detector"
    )

    print(
        "  RPN             : public COCO detector"
    )

    print(
        "  Cascade heads   : random / 12 classes"
    )

    print(
        "  C02 best        : NOT USED"
    )

    print()
    print(
        "Gate history      :",
        GATE_HISTORY,
    )

    print("=" * 72)


# ============================================================
# Save setup metadata
# ============================================================

def save_setup():
    setup = dict(
        experiment=EXP_NAME,

        train_images=1600,

        val_images=400,

        image_scale=list(
            IMAGE_SCALE
        ),

        batch_size=BATCH_SIZE,

        epochs=EPOCHS,

        lr=LR,

        weight_decay=(
            WEIGHT_DECAY
        ),

        warmup_iterations=500,

        milestones=[
            8,
            11,
        ],

        seed=SEED,

        deterministic=True,

        score_thr=SCORE_THR,

        max_det=MAX_DET,

        initialization=str(
            C03_COCO_INIT
        ),

        c02_best_used=False,

        gate_history=str(
            GATE_HISTORY
        ),
    )

    with open(
        SETUP_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            setup,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():
    check_paths()

    check_fresh_work_dir()

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ensure a new formal run starts with a clean gate log.
    if GATE_HISTORY.exists():
        GATE_HISTORY.unlink()

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

    save_setup()

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
        "Resolved config saved:",
        resolved_config,
    )

    print(
        "Experiment setup saved:",
        SETUP_JSON,
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
    print("STARTING C03 FORMAL TRAINING")
    print("=" * 72)
    print()

    runner.train()

    # --------------------------------------------------------
    # Final gate state
    # --------------------------------------------------------

    model = runner.model

    if hasattr(
        model,
        "module",
    ):
        model = model.module

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
    print("=" * 72)
    print("C03 FORMAL TRAINING FINISHED")
    print("=" * 72)

    print(
        "Final raw gates   :",
        raw_gates.tolist(),
    )

    print(
        "Final alpha gates :",
        alpha_gates.tolist(),
    )

    if torch.cuda.is_available():
        allocated = (
            torch.cuda
            .max_memory_allocated()
            / 1024
            / 1024
            / 1024
        )

        reserved = (
            torch.cuda
            .max_memory_reserved()
            / 1024
            / 1024
            / 1024
        )

        print(
            "Peak allocated    :",
            f"{allocated:.3f} GB",
        )

        print(
            "Peak reserved     :",
            f"{reserved:.3f} GB",
        )

    print(
        "Gate history      :",
        GATE_HISTORY,
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  validate the saved best checkpoint "
        "on the full val400 set."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()

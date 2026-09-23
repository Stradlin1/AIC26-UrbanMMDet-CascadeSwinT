#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AIC2026 C04 formal RGB+Depth gated Dual-Swin training.

Protocol:
- Cascade R-CNN + Swin-T + FPN
- RGB main branch + Depth auxiliary branch
- four-stage gated residual fusion
- 1600 train / 400 val fixed split
- 1333x800 keep-ratio, batch=1, AMP
- AdamW lr=2.5e-5, wd=0.05
- 500-iter linear warmup
- 12 epochs, milestones [8, 11], gamma=0.1
- seed=2026, deterministic=True
- formal public-COCO-derived C04 initialization only

This script MUST NOT load C02/C03 trained best checkpoints or parity-only weights.
"""

from __future__ import annotations

# PyTorch deterministic CUDA matmul requires this to be set before CUDA/cuBLAS
# is initialized. Set it before importing torch.
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
# <ldds>/yolo_solution/scripts/train_xxx.py -> <ldds>
LDDS_ROOT = SCRIPT_PATH.parents[2]
YOLO_SOLUTION_ROOT = SCRIPT_PATH.parents[1]
SCRIPTS_DIR = YOLO_SOLUTION_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
from c04_rgbd_components import (  # noqa: E402,F401
    DualSwinDepthGatedBackbone,
    LoadRGBDepthFromFile,
    RGBDDetDataPreprocessor,
)

C03_FORMAL_CONFIG = (
    LDDS_ROOT
    / "runs"
    / "c03_cascade_swin_t_rgbir_gated_1333x800"
    / "resolved_config.py"
)

FORMAL_INIT = (
    LDDS_ROOT
    / "pretrained"
    / "c04_rgbd_swin_t_coco_dual_backbone_fpn_rpn_fusion_init.pth"
)

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

WORK_DIR = (
    LDDS_ROOT
    / "runs"
    / "c04_cascade_swin_t_rgbd_gated_1333x800"
)

SEED = 2026
EXPECTED_TRAIN_IMAGES = 1600
EXPECTED_VAL_IMAGES = 400


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def coco_image_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("images", []))


def replace_loader_with_rgbd(pipeline) -> None:
    if not pipeline:
        raise RuntimeError("Empty dataset pipeline")

    first_type = str(pipeline[0].get("type"))
    allowed = {
        "LoadRGBIRFromFile",
        "LoadImageFromFile",
        "LoadRGBDepthFromFile",
    }
    if first_type not in allowed:
        raise RuntimeError(f"Unexpected first transform: {first_type}")

    pipeline[0] = dict(type="LoadRGBDepthFromFile")


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if state and all(str(k).startswith("module.") for k in state.keys()):
        state = {str(k)[7:]: v for k, v in state.items()}

    return state


def find_best_checkpoints(work_dir: Path) -> List[Path]:
    return sorted(
        work_dir.glob("best_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )


# -----------------------------------------------------------------------------
# Gate monitor hook
# -----------------------------------------------------------------------------


@HOOKS.register_module(force=True)
class C04GateMonitorHook(Hook):
    """Record raw/tanh gates and peak allocated CUDA memory after each epoch."""

    priority = "LOW"

    def before_train(self, runner) -> None:
        self.path = Path(runner.work_dir) / "gate_history.jsonl"
        if self.path.exists():
            self.path.unlink()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def after_train_epoch(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        backbone = model.backbone

        raw = backbone.gates.detach().float().cpu().tolist()
        alpha = torch.tanh(backbone.gates.detach().float()).cpu().tolist()
        peak_mb = (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available()
            else 0.0
        )

        record = {
            "epoch": int(runner.epoch + 1),
            "raw_gates": raw,
            "alpha_gates": alpha,
            "peak_allocated_mb": peak_mb,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        runner.logger.info(
            "C04 gates epoch %d | raw=%s | alpha=%s | peak=%.2f MB",
            runner.epoch + 1,
            [round(x, 8) for x in raw],
            [round(x, 8) for x in alpha],
            peak_mb,
        )


# -----------------------------------------------------------------------------
# Formal configuration
# -----------------------------------------------------------------------------


def build_formal_cfg() -> Config:
    require_file(C03_FORMAL_CONFIG, "C03 formal resolved config")
    require_file(FORMAL_INIT, "C04 formal initialization checkpoint")
    require_file(TRAIN_ANN, "train COCO annotation")
    require_file(VAL_ANN, "val COCO annotation")

    cfg = Config.fromfile(str(C03_FORMAL_CONFIG))

    # ----- Model: exact C03 mirror, auxiliary IR -> Depth -----
    old_backbone = cfg.model.backbone
    rgb_backbone = deepcopy(old_backbone.rgb_backbone)
    depth_backbone = deepcopy(old_backbone.ir_backbone)

    # Formal initialization is loaded through cfg.load_from. Prevent each Swin
    # branch from separately trying to fetch/load another pretrained source.
    rgb_backbone.init_cfg = None
    depth_backbone.init_cfg = None

    cfg.model.backbone = dict(
        type="DualSwinDepthGatedBackbone",
        rgb_backbone=rgb_backbone,
        depth_backbone=depth_backbone,
        channels=(96, 192, 384, 768),
    )
    cfg.model.data_preprocessor.type = "RGBDDetDataPreprocessor"

    # ----- Fixed official project split -----
    ldds_root_str = str(LDDS_ROOT) + "/"

    cfg.train_dataloader.dataset.data_root = ldds_root_str
    cfg.train_dataloader.dataset.ann_file = str(TRAIN_ANN.relative_to(LDDS_ROOT))
    cfg.train_dataloader.dataset.data_prefix = dict(
        img="datasets/AIC2026_Train_2000/visible/"
    )
    replace_loader_with_rgbd(cfg.train_dataloader.dataset.pipeline)

    cfg.val_dataloader.dataset.data_root = ldds_root_str
    cfg.val_dataloader.dataset.ann_file = str(VAL_ANN.relative_to(LDDS_ROOT))
    cfg.val_dataloader.dataset.data_prefix = dict(
        img="datasets/AIC2026_Train_2000/visible/"
    )
    replace_loader_with_rgbd(cfg.val_dataloader.dataset.pipeline)

    cfg.test_dataloader = deepcopy(cfg.val_dataloader)

    cfg.val_evaluator.ann_file = str(VAL_ANN)
    cfg.test_evaluator = deepcopy(cfg.val_evaluator)

    # ----- Formal training protocol: mirror C03 exactly -----
    cfg.train_cfg.max_epochs = 12
    cfg.train_cfg.val_begin = 1
    cfg.train_cfg.val_interval = 1

    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.optim_wrapper.optimizer.type = "AdamW"
    cfg.optim_wrapper.optimizer.lr = 2.5e-5
    cfg.optim_wrapper.optimizer.weight_decay = 0.05
    cfg.optim_wrapper.optimizer.betas = (0.9, 0.999)

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
            begin=0,
            end=12,
            by_epoch=True,
            milestones=[8, 11],
            gamma=0.1,
        ),
    ]

    cfg.load_from = str(FORMAL_INIT)
    cfg.resume = False
    cfg.randomness = dict(seed=SEED, deterministic=True)

    cfg.work_dir = str(WORK_DIR)
    cfg.custom_hooks = [dict(type="C04GateMonitorHook")]

    # Keep C03 logging/checkpoint protocol.
    cfg.default_hooks.logger.interval = 50
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.by_epoch = True
    cfg.default_hooks.checkpoint.save_last = True
    cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
    cfg.default_hooks.checkpoint.rule = "greater"
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3

    return cfg


# -----------------------------------------------------------------------------
# Preflight / summary
# -----------------------------------------------------------------------------


def verify_formal_init() -> None:
    checkpoint = torch.load(FORMAL_INIT, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)

    required_prefixes = [
        "backbone.rgb_backbone.",
        "backbone.depth_backbone.",
        "backbone.adapters.",
        "backbone.gates",
        "neck.",
        "rpn_head.",
    ]

    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in state):
            raise RuntimeError(f"Formal init missing expected prefix: {prefix}")

    if any(k.startswith("backbone.ir_backbone.") for k in state):
        raise RuntimeError("Formal init still contains IR backbone keys")

    gate_key = "backbone.gates"
    gates = state[gate_key].detach().float().cpu()
    if not torch.equal(gates, torch.zeros_like(gates)):
        raise RuntimeError(f"Formal init gates are not zero: {gates.tolist()}")

    # Cascade head is intentionally random for the 12-class task.
    roi_count = sum(k.startswith("roi_head.") for k in state)
    if roi_count != 0:
        raise RuntimeError(
            "Formal init unexpectedly contains roi_head parameters; "
            f"count={roi_count}"
        )

    print("Formal-init audit: PASS")
    print("  RGB backbone   : present")
    print("  Depth backbone : present")
    print("  FPN / RPN      : present")
    print("  Gates          : exact zero")
    print("  Cascade head   : random 12-class init (expected)")


def summarize_run() -> None:
    print()
    print("=" * 72)
    print("C04 FORMAL TRAINING COMPLETE")
    print("=" * 72)

    best = find_best_checkpoints(WORK_DIR)
    if best:
        print("Best checkpoint:", best[-1])
    else:
        print("Best checkpoint: NOT FOUND")

    last = WORK_DIR / "last_checkpoint"
    if last.is_file():
        print("last_checkpoint :", last.read_text(encoding="utf-8").strip())

    gate_history = WORK_DIR / "gate_history.jsonl"
    if gate_history.is_file():
        records = []
        with gate_history.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        print("Gate records    :", len(records))
        if records:
            print("Final raw gates :", records[-1]["raw_gates"])
            print("Final alpha     :", records[-1]["alpha_gates"])
            print(
                "Peak allocated  :",
                f"{max(r['peak_allocated_mb'] for r in records):.2f} MB",
            )
    else:
        print("Gate history    : NOT FOUND")

    print("Resolved config :", WORK_DIR / "resolved_config.py")
    print("STATUS = TRAINING FINISHED")
    print("=" * 72)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("AIC2026 C04 RGB+DEPTH GATED FORMAL TRAINING")
    print("=" * 72)
    print("LDDS root    :", LDDS_ROOT)
    print("C03 template :", C03_FORMAL_CONFIG)
    print("Formal init  :", FORMAL_INIT)
    print("Train ann    :", TRAIN_ANN)
    print("Val ann      :", VAL_ANN)
    print("Work dir     :", WORK_DIR)
    print("cuBLAS config:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    print()

    if WORK_DIR.exists() and any(WORK_DIR.iterdir()):
        raise RuntimeError(
            "Formal work_dir already contains files. Refusing to overwrite:\n"
            f"{WORK_DIR}\n"
            "Do not delete a real formal run accidentally."
        )

    train_count = coco_image_count(TRAIN_ANN)
    val_count = coco_image_count(VAL_ANN)

    if train_count != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(
            f"Train split mismatch: expected {EXPECTED_TRAIN_IMAGES}, got {train_count}"
        )
    if val_count != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Val split mismatch: expected {EXPECTED_VAL_IMAGES}, got {val_count}"
        )

    verify_formal_init()
    cfg = build_formal_cfg()

    print()
    print("Formal protocol:")
    print("  train images  :", train_count)
    print("  val images    :", val_count)
    print("  epochs        :", cfg.train_cfg.max_epochs)
    print("  batch         :", cfg.train_dataloader.batch_size)
    print("  input scale   : 1333 x 800 keep_ratio")
    print("  optimizer     :", cfg.optim_wrapper.optimizer.type)
    print("  lr            :", cfg.optim_wrapper.optimizer.lr)
    print("  weight_decay  :", cfg.optim_wrapper.optimizer.weight_decay)
    print("  warmup        : 500 iterations")
    print("  milestones    : [8, 11]")
    print("  AMP           :", cfg.optim_wrapper.type)
    print("  seed          :", cfg.randomness.seed)
    print("  deterministic :", cfg.randomness.deterministic)
    print("  init          : public-COCO-derived C04 formal init")
    print()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(WORK_DIR / "resolved_config.py"))

    runner = Runner.from_cfg(cfg)
    runner.train()

    summarize_run()


if __name__ == "__main__":
    main()

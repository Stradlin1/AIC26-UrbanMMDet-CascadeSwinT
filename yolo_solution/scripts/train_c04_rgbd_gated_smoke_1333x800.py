#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AIC2026 C04 RGB+Depth gated Dual-Swin smoke training.

Engineering objective only:
- deterministic 64 train / 32 val subset
- 1 epoch
- 1333x800, batch=1, AMP, Swin with_cp=True
- formal public-COCO-derived C04 initialization
- verify finite training, gate movement, auxiliary-path updates,
  checkpoint creation, and strict checkpoint reload

This script deliberately does NOT use C02/C03 trained best weights.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS, MODELS


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
# <ldds>/yolo_solution/scripts/train_xxx.py -> <ldds>
LDDS_ROOT = SCRIPT_PATH.parents[2]
YOLO_SOLUTION_ROOT = SCRIPT_PATH.parents[1]
SCRIPTS_DIR = YOLO_SOLUTION_ROOT / "scripts"

# Make custom C04 components importable and register them with MMDetection.
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

SOURCE_TRAIN_ANN = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
    / "instances_train.json"
)

SOURCE_VAL_ANN = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
    / "instances_val.json"
)

SMOKE_ANN_DIR = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
    / "c04_smoke"
)

SMOKE_TRAIN_ANN = SMOKE_ANN_DIR / "instances_train_64.json"
SMOKE_VAL_ANN = SMOKE_ANN_DIR / "instances_val_32.json"

WORK_DIR = (
    LDDS_ROOT
    / "runs"
    / "c04_cascade_swin_t_rgbd_gated_smoke_1333x800"
)

TRAIN_COUNT = 64
VAL_COUNT = 32
SEED = 2026


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    # Normal MMEngine single-GPU checkpoints do not use module., but keep this
    # normalization so strict smoke verification is robust.
    if state and all(str(k).startswith("module.") for k in state.keys()):
        state = {str(k)[7:]: v for k, v in state.items()}

    return state


def save_coco_subset(src: Path, dst: Path, image_count: int) -> Tuple[int, int]:
    """Create a deterministic COCO subset by the first sorted image ids."""
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = list(data.get("images", []))
    annotations = list(data.get("annotations", []))

    if len(images) < image_count:
        raise RuntimeError(
            f"{src} has only {len(images)} images; requested {image_count}."
        )

    selected = sorted(
        images,
        key=lambda x: (int(x.get("id", 0)), str(x.get("file_name", ""))),
    )[:image_count]

    selected_ids = {int(img["id"]) for img in selected}
    selected_annotations = [
        ann for ann in annotations if int(ann["image_id"]) in selected_ids
    ]

    subset = {
        key: value
        for key, value in data.items()
        if key not in {"images", "annotations"}
    }
    subset["images"] = selected
    subset["annotations"] = selected_annotations

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    return len(selected), len(selected_annotations)


def group_max_abs_diff(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    prefix: str,
) -> Tuple[float, int]:
    keys = sorted(k for k in before if k.startswith(prefix) and k in after)
    if not keys:
        raise RuntimeError(f"No comparable parameters for prefix: {prefix}")

    max_diff = 0.0
    changed = 0

    for key in keys:
        a = before[key].detach().cpu()
        b = after[key].detach().cpu()
        if a.shape != b.shape:
            raise RuntimeError(f"Shape mismatch for {key}: {a.shape} vs {b.shape}")
        diff = float((a - b).abs().max().item())
        max_diff = max(max_diff, diff)
        if diff > 0.0:
            changed += 1

    return max_diff, changed


def locate_training_checkpoint(work_dir: Path) -> Path:
    candidates = list(work_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint produced in: {work_dir}")

    best = sorted(p for p in candidates if p.name.startswith("best_"))
    if best:
        return best[-1]

    epoch = sorted(p for p in candidates if p.name.startswith("epoch_"))
    if epoch:
        return epoch[-1]

    return max(candidates, key=lambda p: p.stat().st_mtime)


def all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(x)) for x in values)


# -----------------------------------------------------------------------------
# Gate monitor hook
# -----------------------------------------------------------------------------


@HOOKS.register_module(force=True)
class C04GateMonitorHook(Hook):
    """Write raw/tanh gates and peak allocated CUDA memory after each epoch."""

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
# Configuration
# -----------------------------------------------------------------------------


def replace_loader_with_rgbd(pipeline) -> None:
    if not pipeline:
        raise RuntimeError("Empty dataset pipeline")

    first_type = str(pipeline[0].get("type"))
    if first_type not in {"LoadRGBIRFromFile", "LoadImageFromFile", "LoadRGBDepthFromFile"}:
        raise RuntimeError(f"Unexpected first transform: {first_type}")

    pipeline[0] = dict(type="LoadRGBDepthFromFile")


def build_smoke_cfg() -> Config:
    require_file(C03_FORMAL_CONFIG, "C03 formal resolved config")
    require_file(FORMAL_INIT, "C04 formal initialization checkpoint")
    require_file(SOURCE_TRAIN_ANN, "train COCO annotation")
    require_file(SOURCE_VAL_ANN, "val COCO annotation")

    cfg = Config.fromfile(str(C03_FORMAL_CONFIG))

    # ----- Model: exact C03 mirror, IR -> Depth -----
    old_backbone = cfg.model.backbone
    rgb_backbone = deepcopy(old_backbone.rgb_backbone)
    depth_backbone = deepcopy(old_backbone.ir_backbone)

    rgb_backbone.init_cfg = None
    depth_backbone.init_cfg = None

    cfg.model.backbone = dict(
        type="DualSwinDepthGatedBackbone",
        rgb_backbone=rgb_backbone,
        depth_backbone=depth_backbone,
        channels=(96, 192, 384, 768),
    )

    cfg.model.data_preprocessor.type = "RGBDDetDataPreprocessor"

    # ----- Data -----
    ldds_root_str = str(LDDS_ROOT) + "/"

    cfg.train_dataloader.dataset.data_root = ldds_root_str
    cfg.train_dataloader.dataset.ann_file = str(
        SMOKE_TRAIN_ANN.relative_to(LDDS_ROOT)
    )
    cfg.train_dataloader.dataset.data_prefix = dict(
        img="datasets/AIC2026_Train_2000/visible/"
    )
    replace_loader_with_rgbd(cfg.train_dataloader.dataset.pipeline)

    cfg.val_dataloader.dataset.data_root = ldds_root_str
    cfg.val_dataloader.dataset.ann_file = str(
        SMOKE_VAL_ANN.relative_to(LDDS_ROOT)
    )
    cfg.val_dataloader.dataset.data_prefix = dict(
        img="datasets/AIC2026_Train_2000/visible/"
    )
    replace_loader_with_rgbd(cfg.val_dataloader.dataset.pipeline)

    cfg.test_dataloader = deepcopy(cfg.val_dataloader)

    cfg.val_evaluator.ann_file = str(SMOKE_VAL_ANN)
    cfg.test_evaluator = deepcopy(cfg.val_evaluator)

    # ----- Smoke train protocol -----
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_begin = 1
    cfg.train_cfg.val_interval = 1

    # Keep the C03 formal optimizer/scheduler shape. In one smoke epoch the
    # formal 500-iteration warmup simply remains in its early phase.
    cfg.optim_wrapper.optimizer.lr = 2.5e-5
    cfg.optim_wrapper.optimizer.weight_decay = 0.05

    cfg.load_from = str(FORMAL_INIT)
    cfg.resume = False
    cfg.randomness = dict(seed=SEED, deterministic=True)

    cfg.work_dir = str(WORK_DIR)
    cfg.custom_hooks = [dict(type="C04GateMonitorHook")]

    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.save_last = True
    cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
    cfg.default_hooks.checkpoint.rule = "greater"
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2

    return cfg


# -----------------------------------------------------------------------------
# Post-train validation
# -----------------------------------------------------------------------------


def verify_training(cfg: Config, checkpoint_path: Path) -> None:
    init_checkpoint = torch.load(FORMAL_INIT, map_location="cpu", weights_only=False)
    train_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    init_state = extract_state_dict(init_checkpoint)
    trained_state = extract_state_dict(train_checkpoint)

    groups = {
        "RGB backbone": "backbone.rgb_backbone.",
        "Depth backbone": "backbone.depth_backbone.",
        "Adapters": "backbone.adapters.",
    }

    print()
    print("Parameter movement from formal initialization:")
    movement = {}

    for label, prefix in groups.items():
        max_diff, changed = group_max_abs_diff(init_state, trained_state, prefix)
        movement[label] = (max_diff, changed)
        print(f"  {label:<15s}: max_diff={max_diff:.12g}, changed_tensors={changed}")

    gate_key = "backbone.gates"
    if gate_key not in init_state or gate_key not in trained_state:
        raise RuntimeError("Missing backbone.gates in checkpoint comparison")

    gate_before = init_state[gate_key].detach().float().cpu()
    gate_after = trained_state[gate_key].detach().float().cpu()
    gate_diff = float((gate_after - gate_before).abs().max().item())

    print("  Gates          :", gate_after.tolist())
    print(f"  Gate max diff  : {gate_diff:.12g}")

    if movement["RGB backbone"][0] <= 0.0:
        raise RuntimeError("Smoke failed: RGB backbone did not update")
    if movement["Depth backbone"][0] <= 0.0:
        raise RuntimeError("Smoke failed: Depth backbone did not update")
    if movement["Adapters"][0] <= 0.0:
        raise RuntimeError("Smoke failed: adapters did not update")
    if gate_diff <= 0.0:
        raise RuntimeError("Smoke failed: gates did not move")

    # Strict reload into a newly constructed detector.
    fresh_model = MODELS.build(cfg.model)
    incompatible = fresh_model.load_state_dict(trained_state, strict=True)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict reload reported incompatibilities:\n"
            f"missing={incompatible.missing_keys}\n"
            f"unexpected={incompatible.unexpected_keys}"
        )

    print("PASS: checkpoint strict reload")

    gate_history = WORK_DIR / "gate_history.jsonl"
    if not gate_history.is_file():
        raise FileNotFoundError(f"Gate history missing: {gate_history}")

    records = []
    with gate_history.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 1:
        raise RuntimeError(f"Expected exactly 1 gate-history record, got {len(records)}")

    record = records[0]
    if not all_finite(record["raw_gates"] + record["alpha_gates"]):
        raise RuntimeError("Non-finite gate history values")

    print("PASS: gate history recorded")
    print(f"Peak allocated memory: {record['peak_allocated_mb']:.2f} MB")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("AIC2026 C04 RGB+DEPTH GATED SMOKE TRAINING")
    print("=" * 72)
    print("LDDS root   :", LDDS_ROOT)
    print("C03 template:", C03_FORMAL_CONFIG)
    print("Formal init :", FORMAL_INIT)
    print("Work dir    :", WORK_DIR)
    print()

    if WORK_DIR.exists():
        # Do not silently overwrite a previous smoke result.
        existing = list(WORK_DIR.iterdir())
        if existing:
            raise RuntimeError(
                "Smoke work_dir already contains files. Refusing to overwrite:\n"
                f"{WORK_DIR}\n"
                "Remove it manually only if you intentionally want to rerun smoke."
            )
        shutil.rmtree(WORK_DIR)

    train_images, train_anns = save_coco_subset(
        SOURCE_TRAIN_ANN, SMOKE_TRAIN_ANN, TRAIN_COUNT
    )
    val_images, val_anns = save_coco_subset(
        SOURCE_VAL_ANN, SMOKE_VAL_ANN, VAL_COUNT
    )

    print(f"Smoke train subset: {train_images} images, {train_anns} annotations")
    print(f"Smoke val subset  : {val_images} images, {val_anns} annotations")

    if train_images != TRAIN_COUNT or val_images != VAL_COUNT:
        raise RuntimeError("Smoke subset cardinality mismatch")

    cfg = build_smoke_cfg()

    print()
    print("Smoke protocol:")
    print("  train images :", TRAIN_COUNT)
    print("  val images   :", VAL_COUNT)
    print("  epochs       :", cfg.train_cfg.max_epochs)
    print("  batch        :", cfg.train_dataloader.batch_size)
    print("  scale        : 1333 x 800 keep_ratio")
    print("  optimizer    :", cfg.optim_wrapper.optimizer.type)
    print("  lr           :", cfg.optim_wrapper.optimizer.lr)
    print("  weight_decay :", cfg.optim_wrapper.optimizer.weight_decay)
    print("  AMP          :", cfg.optim_wrapper.type)
    print("  seed         :", cfg.randomness.seed)
    print("  deterministic:", cfg.randomness.deterministic)

    # Save the exact smoke config before execution.
    WORK_DIR.mkdir(parents=True, exist_ok=False)
    cfg.dump(str(WORK_DIR / "resolved_config.py"))

    runner = Runner.from_cfg(cfg)
    runner.train()

    checkpoint_path = locate_training_checkpoint(WORK_DIR)

    print()
    print("=" * 72)
    print("POST-TRAIN SMOKE VERIFICATION")
    print("=" * 72)
    print("Checkpoint:", checkpoint_path)

    verify_training(cfg, checkpoint_path)

    print()
    print("=" * 72)
    print("C04-08 SMOKE SUMMARY")
    print("=" * 72)
    print("64/32 subset                  PASS")
    print("1333x800 training loop        PASS")
    print("AMP                           PASS")
    print("validation                    PASS")
    print("checkpoint produced           PASS")
    print("RGB backbone updated          PASS")
    print("Depth backbone updated        PASS")
    print("Adapters updated              PASS")
    print("Gates moved                   PASS")
    print("checkpoint strict reload      PASS")
    print("gate history                  PASS")
    print()
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

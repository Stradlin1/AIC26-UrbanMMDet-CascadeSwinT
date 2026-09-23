#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch

SCRIPT_PATH = Path(__file__).resolve()
LDDS_ROOT = SCRIPT_PATH.parents[2]

MMDET_ROOT = LDDS_ROOT / "mmdetection"
COMPONENT_DIR = LDDS_ROOT / "yolo_solution" / "scripts"

sys.path.insert(0, str(MMDET_ROOT))
sys.path.insert(0, str(COMPONENT_DIR))

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
)


# ============================================================
# Paths
# ============================================================

RUN_DIR = (
    LDDS_ROOT
    / "runs"
    / "c03_cascade_swin_t_rgbir_gated_1333x800"
)

CONFIG = RUN_DIR / "resolved_config.py"

VAL_ANN = (
    LDDS_ROOT
    / "generated"
    / "aic2026_coco"
    / "instances_val.json"
)

OUTPUT_DIR = RUN_DIR / "independent_val400"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

# gate=0 val400 is numerically identical to C02 best.
BASELINE_JSON = (
    RUN_DIR
    / "parity_val400"
    / "metrics.json"
)


def find_best_checkpoint():
    candidates = sorted(
        RUN_DIR.glob(
            "best_coco_bbox_mAP_epoch_*.pth"
        )
    )

    if len(candidates) == 0:
        candidates = sorted(
            RUN_DIR.glob("best*.pth")
        )

    if len(candidates) != 1:
        print("Best checkpoint candidates:")

        for path in candidates:
            print(" ", path)

        raise RuntimeError(
            f"Expected exactly one best checkpoint, "
            f"found {len(candidates)}"
        )

    return candidates[0]


def patch_dataset(dataset_cfg):
    dataset_cfg.data_root = str(LDDS_ROOT) + "/"

    dataset_cfg.ann_file = (
        "generated/aic2026_coco/"
        "instances_val.json"
    )

    dataset_cfg.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        )
    )

    dataset_cfg.test_mode = True

    dataset_cfg.pop(
        "indices",
        None,
    )


def build_cfg(best_ckpt):
    cfg = Config.fromfile(
        str(CONFIG)
    )

    # Test-only run: remove the training-only gate monitor hook.
    cfg.custom_hooks = []

    cfg.work_dir = str(
        OUTPUT_DIR
    )

    # Fix paths so this script remains project-root portable.
    patch_dataset(
        cfg.test_dataloader.dataset
    )

    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.drop_last = False

    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(VAL_ANN),
        metric="bbox",
        format_only=False,
        classwise=True,
        backend_args=None,
    )

    cfg.test_cfg = dict(
        type="TestLoop",
    )

    cfg.load_from = str(
        best_ckpt
    )

    cfg.resume = False

    cfg.randomness = dict(
        seed=2026,
        deterministic=True,
    )

    cfg.env_cfg.cudnn_benchmark = False

    return cfg


def save_json(path, obj):
    serializable = {}

    for key, value in obj.items():
        if hasattr(value, "item"):
            value = value.item()

        serializable[str(key)] = value

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            serializable,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return serializable


def print_comparison(current):
    if not BASELINE_JSON.exists():
        print()
        print(
            "Baseline JSON not found; "
            "skipping C02 comparison."
        )
        return

    with open(
        BASELINE_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        baseline = json.load(f)

    keys = [
        "coco/bbox_mAP",
        "coco/bbox_mAP_50",
        "coco/bbox_mAP_75",
        "coco/bbox_mAP_s",
        "coco/bbox_mAP_m",
        "coco/bbox_mAP_l",
    ]

    print()
    print("=" * 72)
    print("C02 vs C03")
    print("=" * 72)

    print(
        f"{'Metric':22s}"
        f"{'C02':>10s}"
        f"{'C03':>10s}"
        f"{'Delta':>12s}"
    )

    print("-" * 54)

    for key in keys:
        if (
            key not in baseline
            or key not in current
        ):
            continue

        old = float(
            baseline[key]
        )

        new = float(
            current[key]
        )

        delta = new - old

        name = key.replace(
            "coco/",
            "",
        )

        print(
            f"{name:22s}"
            f"{old:10.3f}"
            f"{new:10.3f}"
            f"{delta:+12.3f}"
        )

    if (
        "coco/bbox_mAP" in baseline
        and "coco/bbox_mAP" in current
    ):
        delta = (
            float(
                current["coco/bbox_mAP"]
            )
            - float(
                baseline["coco/bbox_mAP"]
            )
        )

        print()
        print(
            "Primary Delta mAP50-95:",
            f"{delta:+.6f}",
        )

        if delta >= 0.005:
            conclusion = (
                "CLEAR POSITIVE: IR fusion is clearly useful."
            )

        elif delta >= 0.002:
            conclusion = (
                "WEAK POSITIVE: promising; consider seed 2027."
            )

        elif delta > -0.002:
            conclusion = (
                "INCONCLUSIVE / essentially tied."
            )

        elif delta > -0.005:
            conclusion = (
                "WEAK NEGATIVE."
            )

        else:
            conclusion = (
                "CLEAR NEGATIVE for this IR fusion design."
            )

        print(
            "Decision:",
            conclusion,
        )


def main():
    if not CONFIG.exists():
        raise FileNotFoundError(
            f"Config not found:\n{CONFIG}"
        )

    if not VAL_ANN.exists():
        raise FileNotFoundError(
            f"Val annotations not found:\n{VAL_ANN}"
        )

    best_ckpt = find_best_checkpoint()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_default_scope(
        "mmdet"
    )

    print("=" * 72)
    print("AIC2026 C03 INDEPENDENT BEST VAL400")
    print("=" * 72)

    print(
        "Config     :",
        CONFIG,
    )

    print(
        "Checkpoint :",
        best_ckpt,
    )

    print(
        "Val images : 400"
    )

    print(
        "Scale      : (1333, 800)"
    )

    print(
        "Max det    : 100"
    )

    cfg = build_cfg(
        best_ckpt
    )

    runner = Runner.from_cfg(
        cfg
    )

    metrics = runner.test()

    metrics = save_json(
        METRICS_JSON,
        metrics,
    )

    print()
    print("=" * 72)
    print("C03 BEST VAL400 RESULTS")
    print("=" * 72)

    for key in sorted(
        metrics.keys()
    ):
        print(
            f"{key:30s}: "
            f"{metrics[key]}"
        )

    print_comparison(
        metrics
    )

    print()
    print(
        "Metrics saved:",
        METRICS_JSON,
    )

    print()
    print("STATUS = COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()

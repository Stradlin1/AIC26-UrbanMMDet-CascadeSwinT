#!/usr/bin/env python3

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import gc
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

MMDET_ROOT = LDDS_ROOT / "mmdetection"

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

from mmdet.registry import (
    DATASETS,
    MODELS,
)

from c04_rgbd_components import (
    LoadRGBDepthFromFile,
    RGBDDetDataPreprocessor,
    DualSwinDepthGatedBackbone,
)


# ============================================================
# Files
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


# ============================================================
# Unit-test settings
#
# This is NOT the formal experimental resolution.
# Reduced size is intentionally used for a cheap gradient test.
# ============================================================

TEST_SCALE = (
    640,
    384,
)

SEED = 2026


# ============================================================
# Helpers
# ============================================================

def check_paths():
    required = (
        MMDET_ROOT,
        C02_CONFIG,
        PARITY_CKPT,
        LDDS_ROOT
        / "datasets"
        / "AIC2026_Train_2000"
        / "visible",
        LDDS_ROOT
        / "datasets"
        / "AIC2026_Train_2000"
        / "depth",
        LDDS_ROOT
        / "generated"
        / "aic2026_coco"
        / "instances_train.json",
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )


def load_checkpoint_state(path):
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


def grad_abs_sum(module):
    total = 0.0

    for parameter in module.parameters():

        if parameter.grad is None:
            continue

        total += float(
            parameter
            .grad
            .detach()
            .abs()
            .sum()
            .item()
        )

    return total


def tensor_grad_abs_sum(parameter):
    if parameter.grad is None:
        return 0.0

    return float(
        parameter
        .grad
        .detach()
        .abs()
        .sum()
        .item()
    )


def loss_to_scalar(value):
    if torch.is_tensor(value):
        return value.mean()

    if isinstance(
        value,
        (list, tuple),
    ):
        result = None

        for item in value:
            scalar = loss_to_scalar(
                item
            )

            if result is None:
                result = scalar
            else:
                result = result + scalar

        return result

    raise TypeError(
        f"Unsupported loss value type: {type(value)}"
    )


def collect_total_loss(losses):
    total_loss = None

    printable = {}

    for name, value in losses.items():

        # Detection models may return metrics besides losses.
        if "loss" not in name:
            continue

        scalar = loss_to_scalar(
            value
        )

        printable[name] = float(
            scalar.detach().cpu().item()
        )

        if total_loss is None:
            total_loss = scalar
        else:
            total_loss = (
                total_loss
                + scalar
            )

    if total_loss is None:
        raise RuntimeError(
            "No loss tensors were returned."
        )

    return total_loss, printable


def gpu_peak_mb():
    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )


# ============================================================
# C04 model config
# ============================================================

def build_c04_model_cfg(c02_cfg):
    model_cfg = deepcopy(
        c02_cfg.model
    )

    original_backbone = deepcopy(
        c02_cfg.model.backbone
    )

    original_backbone.init_cfg = None

    model_cfg.backbone = dict(
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

    old_dp = deepcopy(
        c02_cfg.model.data_preprocessor
    )

    model_cfg.data_preprocessor = dict(
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

    return model_cfg


# ============================================================
# Real training dataset
# ============================================================

def build_train_dataset(c02_cfg):
    dataset_cfg = deepcopy(
        c02_cfg.train_dataloader.dataset
    )

    dataset_cfg.data_root = (
        str(LDDS_ROOT) + "/"
    )

    dataset_cfg.ann_file = (
        "generated/"
        "aic2026_coco/"
        "instances_train.json"
    )

    dataset_cfg.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    # Deterministic real-sample gradient test.
    # No random flip here.
    dataset_cfg.pipeline = [
        dict(
            type="LoadRGBDepthFromFile",
        ),
        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),
        dict(
            type="Resize",
            scale=TEST_SCALE,
            keep_ratio=True,
        ),
        dict(
            type="PackDetInputs",
        ),
    ]

    dataset_cfg.pop(
        "indices",
        None,
    )

    return DATASETS.build(
        dataset_cfg
    )


def find_nonempty_sample(dataset):
    max_search = min(
        len(dataset),
        100,
    )

    for index in range(
        max_search
    ):
        sample = dataset[index]

        data_sample = (
            sample["data_samples"]
        )

        gt_instances = getattr(
            data_sample,
            "gt_instances",
            None,
        )

        if (
            gt_instances is not None
            and len(gt_instances) > 0
        ):
            return index, sample

    raise RuntimeError(
        "Could not find a non-empty training sample."
    )


def make_batch(sample):
    return dict(
        inputs=[
            sample[
                "inputs"
            ].clone()
        ],

        data_samples=[
            deepcopy(
                sample[
                    "data_samples"
                ]
            )
        ],
    )


# ============================================================
# One backward pass
# ============================================================

def run_backward(
    model,
    sample,
    device,
):

    model.zero_grad(
        set_to_none=True
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch = make_batch(
        sample
    )

    processed = (
        model.data_preprocessor(
            batch,
            training=True,
        )
    )

    inputs = processed[
        "inputs"
    ]

    data_samples = processed[
        "data_samples"
    ]

    if inputs.shape[1] != 6:
        raise RuntimeError(
            "Expected 6-channel processed input, "
            f"got {tuple(inputs.shape)}"
        )

    losses = model.loss(
        inputs,
        data_samples,
    )

    total_loss, printable = (
        collect_total_loss(
            losses
        )
    )

    if not torch.isfinite(
        total_loss
    ):
        raise RuntimeError(
            f"Non-finite total loss: {total_loss}"
        )

    total_loss.backward()

    stats = dict(
        total_loss=float(
            total_loss
            .detach()
            .cpu()
            .item()
        ),

        printable=printable,

        rgb_grad=grad_abs_sum(
            model.backbone.rgb_backbone
        ),

        depth_grad=grad_abs_sum(
            model.backbone.depth_backbone
        ),

        adapter_grad=grad_abs_sum(
            model.backbone.adapters
        ),

        gate_grad=tensor_grad_abs_sum(
            model.backbone.gates
        ),

        neck_grad=grad_abs_sum(
            model.neck
        ),

        rpn_grad=grad_abs_sum(
            model.rpn_head
        ),

        roi_grad=grad_abs_sum(
            model.roi_head
        ),

        gate_grad_vector=(
            None
            if model.backbone.gates.grad is None
            else
            model.backbone.gates
            .grad
            .detach()
            .cpu()
            .tolist()
        ),

        peak_memory_mb=gpu_peak_mb(),

        processed_shape=tuple(
            inputs.shape
        ),
    )

    return stats


# ============================================================
# Report
# ============================================================

def print_stats(
    title,
    stats,
):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)

    print(
        "Processed input :",
        stats["processed_shape"],
    )

    print(
        "Total loss      :",
        f'{stats["total_loss"]:.9f}',
    )

    print()
    print("Loss components:")

    for name, value in sorted(
        stats["printable"].items()
    ):
        print(
            f"  {name:28s}: "
            f"{value:.9f}"
        )

    print()
    print("Gradient absolute sums:")

    print(
        "  RGB backbone :",
        f'{stats["rgb_grad"]:.9f}',
    )

    print(
        "  Depth backbone:",
        f'{stats["depth_grad"]:.9f}',
    )

    print(
        "  Adapters     :",
        f'{stats["adapter_grad"]:.9f}',
    )

    print(
        "  Gates        :",
        f'{stats["gate_grad"]:.9f}',
    )

    print(
        "  FPN          :",
        f'{stats["neck_grad"]:.9f}',
    )

    print(
        "  RPN          :",
        f'{stats["rpn_grad"]:.9f}',
    )

    print(
        "  Cascade      :",
        f'{stats["roi_grad"]:.9f}',
    )

    print(
        "Gate grad vector:",
        stats["gate_grad_vector"],
    )

    if torch.cuda.is_available():
        print(
            "Peak allocated :",
            f'{stats["peak_memory_mb"]:.2f} MB',
        )


# ============================================================
# Main
# ============================================================

def main():
    check_paths()

    init_default_scope(
        "mmdet"
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

        torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("AIC2026 C04 FULL DETECTOR BACKWARD TEST")
    print("=" * 72)

    print("Device        :", device)
    print("C02 config    :", C02_CONFIG)
    print("Parity ckpt   :", PARITY_CKPT)
    print("Test scale    :", TEST_SCALE)
    print()
    print(
        "NOTE: 640x384 is used only for this structural "
        "gradient test."
    )
    print(
        "Formal C04 remains 1333x800."
    )

    # --------------------------------------------------------
    # Config / model
    # --------------------------------------------------------

    c02_cfg = Config.fromfile(
        str(C02_CONFIG)
    )

    model_cfg = build_c04_model_cfg(
        c02_cfg
    )

    model = MODELS.build(
        model_cfg
    )

    model.init_weights()

    parity_state = load_checkpoint_state(
        PARITY_CKPT
    )

    result = model.load_state_dict(
        parity_state,
        strict=True,
    )

    print()
    print(
        "Checkpoint strict load:",
        result,
    )

    model = model.to(
        device
    )

    model.train()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = build_train_dataset(
        c02_cfg
    )

    sample_index, sample = (
        find_nonempty_sample(
            dataset
        )
    )

    gt_count = len(
        sample[
            "data_samples"
        ].gt_instances
    )

    print()
    print(
        "Train dataset   :",
        len(dataset),
    )

    print(
        "Selected sample :",
        sample_index,
    )

    print(
        "GT boxes        :",
        gt_count,
    )

    print(
        "Packed input    :",
        tuple(
            sample[
                "inputs"
            ].shape
        ),
    )

    # ========================================================
    # PASS 1: gate = 0
    # ========================================================

    with torch.no_grad():
        model.backbone.gates.zero_()

    print()
    print(
        "Raw gates before pass 1:",
        model.backbone.gates
        .detach()
        .cpu()
        .tolist(),
    )

    stats_zero = run_backward(
        model,
        sample,
        device,
    )

    print_stats(
        "PASS 1: gate = 0",
        stats_zero,
    )

    # --------------------------------------------------------
    # Expected gradient structure
    # --------------------------------------------------------

    if stats_zero["rgb_grad"] <= 0:
        raise RuntimeError(
            "RGB backbone has no gradient at gate=0."
        )

    if stats_zero["neck_grad"] <= 0:
        raise RuntimeError(
            "FPN has no gradient."
        )

    if stats_zero["rpn_grad"] <= 0:
        raise RuntimeError(
            "RPN has no gradient."
        )

    if stats_zero["roi_grad"] <= 0:
        raise RuntimeError(
            "Cascade ROI head has no gradient."
        )

    if stats_zero["gate_grad"] <= 0:
        raise RuntimeError(
            "Gate parameters have no gradient."
        )

    if stats_zero["depth_grad"] != 0.0:
        raise RuntimeError(
            "Depth backbone should have zero gradient "
            "when gate=0."
        )

    if stats_zero["adapter_grad"] != 0.0:
        raise RuntimeError(
            "Adapters should have zero gradient "
            "when gate=0."
        )

    print()
    print(
        "PASS: gate=0 detector gradient structure"
    )

    # ========================================================
    # PASS 2: non-zero gate
    # ========================================================

    model.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        model.backbone.gates.fill_(
            0.01
        )

    print()
    print(
        "Raw gates before pass 2:",
        model.backbone.gates
        .detach()
        .cpu()
        .tolist(),
    )

    stats_nonzero = run_backward(
        model,
        sample,
        device,
    )

    print_stats(
        "PASS 2: gate = 0.01",
        stats_nonzero,
    )

    if stats_nonzero["depth_grad"] <= 0:
        raise RuntimeError(
            "Depth backbone has no gradient "
            "with non-zero gates."
        )

    if stats_nonzero[
        "adapter_grad"
    ] <= 0:
        raise RuntimeError(
            "Adapters have no gradient "
            "with non-zero gates."
        )

    if stats_nonzero["gate_grad"] <= 0:
        raise RuntimeError(
            "Gates have no gradient "
            "with non-zero gates."
        )

    print()
    print(
        "PASS: Depth learning path enabled "
        "when gates become non-zero"
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        "Real RGB+Depth sample        PASS"
    )

    print(
        "Detection losses finite   PASS"
    )

    print(
        "RGB gradient              PASS"
    )

    print(
        "FPN/RPN/Cascade gradient  PASS"
    )

    print(
        "Gate gradient @ zero      PASS"
    )

    print(
        "Depth blocked @ gate=0       PASS"
    )

    print(
        "Adapter blocked @ gate=0  PASS"
    )

    print(
        "Depth learns @ gate!=0       PASS"
    )

    print(
        "Adapter learns @ gate!=0  PASS"
    )

    print()
    print("STATUS = PASS")
    print("=" * 72)

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


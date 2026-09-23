#!/usr/bin/env python3

import os

# Same deterministic cuBLAS setting as C02.
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
# Paths
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

# Register C03 custom components.
from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
)


# ============================================================
# C02
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

C02_BEST = (
    C02_RUN_DIR
    / "best_coco_bbox_mAP_epoch_11.pth"
)


# ============================================================
# C03 parity output
# ============================================================

C03_RUN_DIR = (
    LDDS_ROOT
    / "runs"
    / "c03_cascade_swin_t_rgbir_gated_1333x800"
)

PARITY_DIR = (
    C03_RUN_DIR
    / "parity"
)

PARITY_CKPT = (
    PARITY_DIR
    / "c03_gate0_from_c02_best_PARITY_ONLY.pth"
)


# ============================================================
# Helpers
# ============================================================

def check_paths():

    required = (
        MMDET_ROOT,
        C02_CONFIG,
        C02_BEST,
        LDDS_ROOT
        / "datasets"
        / "AIC2026_Train_2000"
        / "visible",
        LDDS_ROOT
        / "datasets"
        / "AIC2026_Train_2000"
        / "infrared",
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
        state_dict = (
            checkpoint["state_dict"]
        )
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def tensor_max_diff(a, b):

    if tuple(a.shape) != tuple(b.shape):
        raise RuntimeError(
            "Tensor shape mismatch:\n"
            f"A: {tuple(a.shape)}\n"
            f"B: {tuple(b.shape)}"
        )

    return float(
        torch.max(
            torch.abs(
                a.float()
                - b.float()
            )
        ).item()
    )


def cpu_clone(tensor):

    return (
        tensor
        .detach()
        .cpu()
        .clone()
    )


def cleanup_model(model):

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Build C03 model config from actual C02 config
# ============================================================

def build_c03_model_cfg(c02_cfg):

    model_cfg = deepcopy(
        c02_cfg.model
    )

    original_backbone = deepcopy(
        c02_cfg.model.backbone
    )

    # No independent init_cfg here.
    # This parity test explicitly maps C02 best.
    original_backbone.init_cfg = None

    model_cfg.backbone = dict(
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

    # Preserve exactly the C02 preprocessing constants.
    c02_dp = c02_cfg.model.data_preprocessor

    model_cfg.data_preprocessor = dict(
        type="RGBIRDetDataPreprocessor",
        mean=list(
            c02_dp.mean
        ),
        std=list(
            c02_dp.std
        ),
        bgr_to_rgb=bool(
            c02_dp.bgr_to_rgb
        ),
        pad_size_divisor=int(
            c02_dp.pad_size_divisor
        ),
    )

    return model_cfg


# ============================================================
# Build paired validation dataset
# ============================================================

def build_c03_val_dataset(c02_cfg):

    dataset_cfg = deepcopy(
        c02_cfg.val_dataloader.dataset
    )

    # Make new code relocatable.
    dataset_cfg.data_root = (
        str(LDDS_ROOT) + "/"
    )

    dataset_cfg.ann_file = (
        "generated/aic2026_coco/"
        "instances_val.json"
    )

    dataset_cfg.data_prefix = dict(
        img=(
            "datasets/"
            "AIC2026_Train_2000/"
            "visible/"
        ),
    )

    pipeline = deepcopy(
        dataset_cfg.pipeline
    )

    if len(pipeline) == 0:
        raise RuntimeError(
            "Validation pipeline is empty."
        )

    # Replace only the image loader.
    # Resize / annotations / PackDetInputs remain C02-identical.
    pipeline[0] = dict(
        type="LoadRGBIRFromFile",
    )

    dataset_cfg.pipeline = pipeline

    dataset = DATASETS.build(
        dataset_cfg
    )

    return dataset


# ============================================================
# Find a non-empty validation sample
# ============================================================

def find_sample(dataset):

    max_search = min(
        len(dataset),
        50,
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

    # Fallback.
    return 0, dataset[0]


# ============================================================
# C02 -> C03 checkpoint surgery
# ============================================================

def map_c02_to_c03(
    c02_state,
    c03_model,
):

    target_state = (
        c03_model.state_dict()
    )

    source_keys_mapped = set()

    rgb_count = 0
    ir_count = 0
    shared_count = 0

    # --------------------------------------------------------
    # Map every C02 tensor
    # --------------------------------------------------------

    for source_key, source_tensor in (
        c02_state.items()
    ):

        # ----------------------------------------------------
        # Backbone duplicated into RGB + IR branches
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

            for (
                target_key,
                branch_name,
            ) in (
                (rgb_key, "RGB"),
                (ir_key, "IR"),
            ):

                if target_key not in target_state:
                    raise RuntimeError(
                        f"{branch_name} target key missing:\n"
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
                        "Shape mismatch:\n"
                        f"source: {source_key} "
                        f"{tuple(source_tensor.shape)}\n"
                        f"target: {target_key} "
                        f"{tuple(target_state[target_key].shape)}"
                    )

                target_state[
                    target_key
                ] = (
                    source_tensor
                    .detach()
                    .clone()
                )

            rgb_count += 1
            ir_count += 1
            source_keys_mapped.add(
                source_key
            )

            continue

        # ----------------------------------------------------
        # FPN / RPN / Cascade / preprocessor / other buffers
        # remain under identical names.
        # ----------------------------------------------------

        if source_key not in target_state:
            raise RuntimeError(
                "C02 non-backbone key is absent "
                "from C03 target:\n"
                f"{source_key}"
            )

        if (
            tuple(source_tensor.shape)
            != tuple(
                target_state[
                    source_key
                ].shape
            )
        ):
            raise RuntimeError(
                "Shared tensor shape mismatch:\n"
                f"source: {source_key} "
                f"{tuple(source_tensor.shape)}\n"
                f"target: {source_key} "
                f"{tuple(target_state[source_key].shape)}"
            )

        target_state[
            source_key
        ] = (
            source_tensor
            .detach()
            .clone()
        )

        shared_count += 1

        source_keys_mapped.add(
            source_key
        )

    # --------------------------------------------------------
    # Every C02 key must have been used
    # --------------------------------------------------------

    missing_source = sorted(
        set(c02_state)
        - source_keys_mapped
    )

    if missing_source:

        print()
        print("Unmapped C02 keys:")

        for key in missing_source[:50]:
            print(" ", key)

        raise RuntimeError(
            "Not every C02 tensor was mapped."
        )

    # --------------------------------------------------------
    # New C03-only tensors
    # --------------------------------------------------------

    new_keys = sorted(
        key
        for key in target_state
        if (
            key.startswith(
                "backbone.adapters."
            )
            or key
            == "backbone.gates"
        )
    )

    if not new_keys:
        raise RuntimeError(
            "No C03-specific fusion tensors found."
        )

    # Strict load of complete C03 state.
    c03_model.load_state_dict(
        target_state,
        strict=True,
    )

    print()
    print("Checkpoint mapping:")
    print(
        "  C02 source tensors :",
        len(c02_state),
    )
    print(
        "  RGB backbone mapped:",
        rgb_count,
    )
    print(
        "  IR backbone mapped :",
        ir_count,
    )
    print(
        "  shared tensors     :",
        shared_count,
    )
    print(
        "  C03 fusion tensors :",
        len(new_keys),
    )

    # --------------------------------------------------------
    # Verify zero gates
    # --------------------------------------------------------

    gates = (
        c03_model
        .backbone
        .gate_values()
    )

    print(
        "  gates              :",
        gates,
    )

    if any(
        value != 0.0
        for value in gates
    ):
        raise RuntimeError(
            "Mapped C03 model does not have zero gates."
        )

    return target_state


# ============================================================
# Prediction comparison
# ============================================================

def compare_predictions(
    c02_pred,
    c03_pred,
):

    boxes02 = c02_pred.bboxes
    boxes03 = c03_pred.bboxes

    scores02 = c02_pred.scores
    scores03 = c03_pred.scores

    labels02 = c02_pred.labels
    labels03 = c03_pred.labels

    if (
        boxes02.shape
        != boxes03.shape
    ):
        raise RuntimeError(
            "Prediction bbox shape mismatch:\n"
            f"C02: {boxes02.shape}\n"
            f"C03: {boxes03.shape}"
        )

    if (
        scores02.shape
        != scores03.shape
    ):
        raise RuntimeError(
            "Prediction score shape mismatch."
        )

    if (
        labels02.shape
        != labels03.shape
    ):
        raise RuntimeError(
            "Prediction label shape mismatch."
        )

    if not torch.equal(
        labels02,
        labels03,
    ):
        raise RuntimeError(
            "Prediction labels differ."
        )

    if boxes02.numel() == 0:
        bbox_diff = 0.0
    else:
        bbox_diff = tensor_max_diff(
            boxes02,
            boxes03,
        )

    if scores02.numel() == 0:
        score_diff = 0.0
    else:
        score_diff = tensor_max_diff(
            scores02,
            scores03,
        )

    return (
        bbox_diff,
        score_diff,
        int(
            scores02.numel()
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    check_paths()

    PARITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_default_scope(
        "mmdet"
    )

    torch.manual_seed(
        2026
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            2026
        )

        torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("AIC2026 C02 -> C03 FULL DETECTOR PARITY TEST")
    print("=" * 72)

    print("Device       :", device)
    print("C02 config   :", C02_CONFIG)
    print("C02 best     :", C02_BEST)
    print("Parity output:", PARITY_CKPT)

    # --------------------------------------------------------
    # Config / dataset
    # --------------------------------------------------------

    c02_cfg = Config.fromfile(
        str(C02_CONFIG)
    )

    dataset = build_c03_val_dataset(
        c02_cfg
    )

    sample_index, sample = (
        find_sample(
            dataset
        )
    )

    print()
    print(
        "Val dataset  :",
        len(dataset),
        "images",
    )

    print(
        "Test sample  :",
        sample_index,
    )

    print(
        "Raw packed input:",
        tuple(
            sample["inputs"].shape
        ),
    )

    if (
        sample["inputs"].ndim != 3
        or sample["inputs"].shape[0] != 6
    ):
        raise RuntimeError(
            "Expected packed C03 input "
            "with shape 6xHxW."
        )

    # --------------------------------------------------------
    # Load C02 checkpoint
    # --------------------------------------------------------

    c02_state = load_state_dict(
        C02_BEST
    )

    print(
        "C02 state tensors:",
        len(c02_state),
    )

    # ========================================================
    # PHASE A: C02 reference
    # ========================================================

    print()
    print("-" * 72)
    print("PHASE A: C02 reference")
    print("-" * 72)

    c02_model = MODELS.build(
        deepcopy(
            c02_cfg.model
        )
    )

    c02_model.init_weights()

    result = c02_model.load_state_dict(
        c02_state,
        strict=True,
    )

    print(
        "C02 strict load:",
        result,
    )

    c02_model = (
        c02_model
        .to(device)
        .eval()
    )

    # Same paired sample, but C02 receives only RGB transport
    # channels 0:3.
    c02_batch = dict(
        inputs=[
            sample["inputs"][
                0:3
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

    with torch.no_grad():

        c02_processed = (
            c02_model
            .data_preprocessor(
                c02_batch,
                training=False,
            )
        )

        c02_input_ref = cpu_clone(
            c02_processed[
                "inputs"
            ]
        )

        c02_backbone = (
            c02_model.backbone(
                c02_processed[
                    "inputs"
                ]
            )
        )

        c02_backbone_ref = tuple(
            cpu_clone(x)
            for x in c02_backbone
        )

        c02_neck = (
            c02_model.neck(
                c02_backbone
            )
        )

        c02_neck_ref = tuple(
            cpu_clone(x)
            for x in c02_neck
        )

        c02_output = (
            c02_model.predict(
                c02_processed[
                    "inputs"
                ],
                c02_processed[
                    "data_samples"
                ],
                rescale=True,
            )
        )

        c02_pred_ref = (
            c02_output[0]
            .pred_instances
            .cpu()
        )

    print(
        "C02 processed input:",
        tuple(
            c02_input_ref.shape
        ),
    )

    print(
        "C02 predictions    :",
        len(c02_pred_ref),
    )

    # Free C02 GPU memory before creating the dual-backbone C03.
    c02_model = c02_model.cpu()

    del c02_model
    del c02_processed
    del c02_backbone
    del c02_neck
    del c02_output

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # PHASE B: C03 mapped parity model
    # ========================================================

    print()
    print("-" * 72)
    print("PHASE B: C03 mapped gate=0 model")
    print("-" * 72)

    c03_model_cfg = (
        build_c03_model_cfg(
            c02_cfg
        )
    )

    c03_model = MODELS.build(
        c03_model_cfg
    )

    c03_model.init_weights()

    mapped_state = map_c02_to_c03(
        c02_state,
        c03_model,
    )

    # --------------------------------------------------------
    # Save parity-only checkpoint
    # --------------------------------------------------------

    checkpoint_to_save = dict(
        meta=dict(
            purpose=(
                "C02 -> C03 gate=0 parity only"
            ),
            warning=(
                "DO NOT USE AS FORMAL C03 "
                "TRAINING INITIALIZATION"
            ),
            source_c02_best=str(
                C02_BEST
            ),
            gate_init=0.0,
            adapter_init="identity",
        ),
        state_dict={
            key: value.detach().cpu()
            for key, value
            in mapped_state.items()
        },
    )

    torch.save(
        checkpoint_to_save,
        PARITY_CKPT,
    )

    print()
    print(
        "Parity checkpoint saved:",
        PARITY_CKPT,
    )

    c03_model = (
        c03_model
        .to(device)
        .eval()
    )

    c03_batch = dict(
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

    with torch.no_grad():

        c03_processed = (
            c03_model
            .data_preprocessor(
                c03_batch,
                training=False,
            )
        )

        # ----------------------------------------------------
        # RGB preprocessing parity
        # ----------------------------------------------------

        c03_rgb_input = (
            c03_processed[
                "inputs"
            ][:, 0:3]
        )

        input_diff = tensor_max_diff(
            c02_input_ref,
            c03_rgb_input.cpu(),
        )

        # ----------------------------------------------------
        # Backbone parity
        # ----------------------------------------------------

        c03_backbone = (
            c03_model.backbone(
                c03_processed[
                    "inputs"
                ]
            )
        )

        backbone_diffs = []

        for ref, current in zip(
            c02_backbone_ref,
            c03_backbone,
        ):

            backbone_diffs.append(
                tensor_max_diff(
                    ref,
                    current.cpu(),
                )
            )

        # ----------------------------------------------------
        # FPN parity
        # ----------------------------------------------------

        c03_neck = (
            c03_model.neck(
                c03_backbone
            )
        )

        neck_diffs = []

        for ref, current in zip(
            c02_neck_ref,
            c03_neck,
        ):

            neck_diffs.append(
                tensor_max_diff(
                    ref,
                    current.cpu(),
                )
            )

        # ----------------------------------------------------
        # Final prediction parity
        # ----------------------------------------------------

        c03_output = (
            c03_model.predict(
                c03_processed[
                    "inputs"
                ],
                c03_processed[
                    "data_samples"
                ],
                rescale=True,
            )
        )

        c03_pred = (
            c03_output[0]
            .pred_instances
            .cpu()
        )

    (
        bbox_diff,
        score_diff,
        prediction_count,
    ) = compare_predictions(
        c02_pred_ref,
        c03_pred,
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 72)
    print("PARITY RESULTS")
    print("=" * 72)

    print(
        "RGB preprocess max diff:",
        f"{input_diff:.12f}",
    )

    print()
    print("Backbone stage diffs:")

    for index, diff in enumerate(
        backbone_diffs
    ):
        print(
            f"  C{index + 2}: "
            f"{diff:.12f}"
        )

    print()
    print("FPN level diffs:")

    for index, diff in enumerate(
        neck_diffs
    ):
        print(
            f"  P{index + 2}: "
            f"{diff:.12f}"
        )

    print()
    print(
        "Prediction count      :",
        prediction_count,
    )

    print(
        "BBox max diff         :",
        f"{bbox_diff:.12f}",
    )

    print(
        "Score max diff        :",
        f"{score_diff:.12f}",
    )

    # ========================================================
    # Assertions
    # ========================================================

    if input_diff != 0.0:
        raise RuntimeError(
            "RGB preprocessing parity FAILED."
        )

    if max(backbone_diffs) != 0.0:
        raise RuntimeError(
            "Backbone gate=0 parity FAILED."
        )

    if max(neck_diffs) != 0.0:
        raise RuntimeError(
            "FPN parity FAILED."
        )

    # Prediction stages can theoretically show extremely
    # tiny floating-point differences, so allow a tight
    # numerical tolerance.
    PRED_TOL = 1e-6

    if bbox_diff > PRED_TOL:
        raise RuntimeError(
            "BBox prediction parity FAILED."
        )

    if score_diff > PRED_TOL:
        raise RuntimeError(
            "Score prediction parity FAILED."
        )

    print()
    print("PASS: checkpoint mapping")
    print("PASS: RGB preprocessing")
    print("PASS: gate=0 backbone")
    print("PASS: FPN")
    print("PASS: labels")
    print("PASS: bounding boxes")
    print("PASS: confidence scores")

    print()
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

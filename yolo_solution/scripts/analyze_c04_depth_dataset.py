#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 C04 RGB+Depth
Depth dataset audit.

Purpose
-------
Before implementing the RGBD model, audit the actual Depth representation:

1. RGB / Depth sample-stem parity
2. file extensions
3. image resolutions
4. dtype / channel layout
5. RGB / Depth spatial-size parity
6. uint16 metric-depth statistics
7. invalid-depth statistics
8. global depth percentiles
9. uint8/JPG depth statistics
10. low-valid-ratio samples

No training data is modified.

Path convention
---------------
This script lives in:

    <workspace>/yolo_solution/scripts/

Therefore:

    CODE_ROOT      = <workspace>/yolo_solution
    WORKSPACE_ROOT = <workspace>

Default dataset:

    <workspace>/datasets/AIC2026_Train_2000

Default output:

    <workspace>/runs/c04_depth_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent

DEFAULT_DATASET_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)

DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "runs"
    / "c04_depth_analysis"
)

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def collect_images(
    directory: Path,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
    )

    for path in files:
        stem = path.stem

        if stem in mapping:
            duplicates.setdefault(
                stem,
                [mapping[stem]],
            ).append(path)
        else:
            mapping[stem] = path

    return mapping, duplicates


def channel_count(image: np.ndarray) -> int:
    if image.ndim == 2:
        return 1

    if image.ndim == 3:
        return int(image.shape[2])

    return -1


def scalar_depth_and_channel_diff(
    image: np.ndarray,
) -> Tuple[np.ndarray, int]:

    if image.ndim == 2:
        return image, 0

    if image.ndim != 3:
        raise ValueError(
            f"Unsupported depth shape: {image.shape}"
        )

    base = image[..., 0]

    max_diff = 0

    base_i64 = base.astype(
        np.int64,
        copy=False,
    )

    for c in range(1, image.shape[2]):
        channel_i64 = image[..., c].astype(
            np.int64,
            copy=False,
        )

        diff = np.abs(
            channel_i64 - base_i64
        )

        if diff.size:
            max_diff = max(
                max_diff,
                int(diff.max()),
            )

    return base, max_diff


def hist_percentile(
    hist: np.ndarray,
    percentile: float,
) -> int | None:

    total = int(hist.sum())

    if total <= 0:
        return None

    cumulative = np.cumsum(
        hist,
        dtype=np.int64,
    )

    target = (
        percentile
        / 100.0
        * (total - 1)
    )

    index = int(
        np.searchsorted(
            cumulative,
            target + 1,
            side="left",
        )
    )

    return index


def histogram_nonzero_range(
    hist: np.ndarray,
) -> Tuple[int | None, int | None]:

    nonzero = np.flatnonzero(
        hist > 0
    )

    if len(nonzero) == 0:
        return None, None

    return (
        int(nonzero[0]),
        int(nonzero[-1]),
    )


def ratio(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def format_ratio(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def percentile_dict(
    hist: np.ndarray,
) -> dict:

    percentiles = [
        0.1,
        1,
        5,
        50,
        95,
        99,
        99.9,
    ]

    return {
        str(p): hist_percentile(
            hist,
            p,
        )
        for p in percentiles
    }


def safe_numpy_percentiles(
    values: List[float],
) -> dict:

    if not values:
        return {}

    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    qs = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    return {
        str(q): float(
            np.percentile(
                arr,
                q,
            )
        )
        for q in qs
    }


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def run_audit(
    dataset_root: Path,
    output_dir: Path,
) -> dict:

    visible_dir = (
        dataset_root
        / "visible"
    )

    depth_dir = (
        dataset_root
        / "depth"
    )

    print("=" * 72)
    print("AIC2026 C04 Depth Dataset Audit")
    print("=" * 72)
    print(f"Code root      : {CODE_ROOT}")
    print(f"Workspace root : {WORKSPACE_ROOT}")
    print(f"Dataset root   : {dataset_root}")
    print(f"Visible dir    : {visible_dir}")
    print(f"Depth dir      : {depth_dir}")
    print(f"Output dir     : {output_dir}")
    print("=" * 72)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visible_map, visible_duplicates = (
        collect_images(
            visible_dir
        )
    )

    depth_map, depth_duplicates = (
        collect_images(
            depth_dir
        )
    )

    visible_stems = set(
        visible_map
    )

    depth_stems = set(
        depth_map
    )

    common_stems = sorted(
        visible_stems
        & depth_stems
    )

    visible_only = sorted(
        visible_stems
        - depth_stems
    )

    depth_only = sorted(
        depth_stems
        - visible_stems
    )

    extension_counts = Counter()
    extension_pair_counts = Counter()

    depth_shape_counts = Counter()
    visible_shape_counts = Counter()

    dtype_counts = Counter()
    dtype_channel_counts = Counter()

    size_mismatch_samples = []

    metric_hist = np.zeros(
        65536,
        dtype=np.int64,
    )

    uint8_hist = np.zeros(
        256,
        dtype=np.int64,
    )

    metric_file_count = 0
    uint8_file_count = 0
    other_file_count = 0

    metric_total_pixels = 0

    # Official useful metric range:
    #
    #     300 mm <= depth <= 19999 mm
    #
    metric_valid_pixels = 0
    metric_invalid_low_pixels = 0
    metric_invalid_high_pixels = 0
    metric_zero_pixels = 0

    per_image_metric_valid_ratio = []

    low_valid_samples = []

    max_depth_channel_difference = 0

    file_rows = []

    unreadable_depth = []
    unreadable_visible = []

    for index, stem in enumerate(
        common_stems,
        start=1,
    ):

        visible_path = (
            visible_map[stem]
        )

        depth_path = (
            depth_map[stem]
        )

        visible = cv2.imread(
            str(visible_path),
            cv2.IMREAD_UNCHANGED,
        )

        depth = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if visible is None:
            unreadable_visible.append(
                str(visible_path)
            )
            continue

        if depth is None:
            unreadable_depth.append(
                str(depth_path)
            )
            continue

        vh, vw = visible.shape[:2]
        dh, dw = depth.shape[:2]

        vc = channel_count(
            visible
        )

        dc = channel_count(
            depth
        )

        visible_shape_counts[
            f"{vh}x{vw}x{vc}"
        ] += 1

        depth_shape_counts[
            f"{dh}x{dw}x{dc}"
        ] += 1

        extension_counts[
            depth_path.suffix.lower()
        ] += 1

        extension_pair_counts[
            (
                visible_path.suffix.lower(),
                depth_path.suffix.lower(),
            )
        ] += 1

        dtype_name = str(
            depth.dtype
        )

        dtype_counts[
            dtype_name
        ] += 1

        dtype_channel_counts[
            f"{dtype_name}_c{dc}"
        ] += 1

        size_match = (
            vh == dh
            and vw == dw
        )

        if not size_match:
            size_mismatch_samples.append(
                {
                    "stem": stem,
                    "visible_shape": [
                        vh,
                        vw,
                        vc,
                    ],
                    "depth_shape": [
                        dh,
                        dw,
                        dc,
                    ],
                }
            )

        depth_scalar, channel_diff = (
            scalar_depth_and_channel_diff(
                depth
            )
        )

        max_depth_channel_difference = max(
            max_depth_channel_difference,
            channel_diff,
        )

        depth_min = int(
            depth_scalar.min()
        )

        depth_max = int(
            depth_scalar.max()
        )

        valid_ratio_value = None

        if depth_scalar.dtype == np.uint16:

            metric_file_count += 1

            flattened = (
                depth_scalar.reshape(-1)
            )

            hist = np.bincount(
                flattened,
                minlength=65536,
            ).astype(
                np.int64,
                copy=False,
            )

            metric_hist += hist

            n_pixels = int(
                flattened.size
            )

            n_zero = int(
                hist[0]
            )

            n_invalid_low = int(
                hist[:300].sum()
            )

            n_valid = int(
                hist[300:20000].sum()
            )

            n_invalid_high = int(
                hist[20000:].sum()
            )

            metric_total_pixels += (
                n_pixels
            )

            metric_zero_pixels += (
                n_zero
            )

            metric_invalid_low_pixels += (
                n_invalid_low
            )

            metric_valid_pixels += (
                n_valid
            )

            metric_invalid_high_pixels += (
                n_invalid_high
            )

            valid_ratio_value = ratio(
                n_valid,
                n_pixels,
            )

            per_image_metric_valid_ratio.append(
                valid_ratio_value
            )

            if valid_ratio_value < 0.05:
                low_valid_samples.append(
                    {
                        "stem": stem,
                        "valid_ratio": (
                            valid_ratio_value
                        ),
                        "path": str(
                            depth_path
                        ),
                    }
                )

        elif depth_scalar.dtype == np.uint8:

            uint8_file_count += 1

            flattened = (
                depth_scalar.reshape(-1)
            )

            hist = np.bincount(
                flattened,
                minlength=256,
            ).astype(
                np.int64,
                copy=False,
            )

            uint8_hist += hist

        else:
            other_file_count += 1

        file_rows.append(
            {
                "stem": stem,
                "visible_file": (
                    visible_path.name
                ),
                "depth_file": (
                    depth_path.name
                ),
                "visible_h": vh,
                "visible_w": vw,
                "depth_h": dh,
                "depth_w": dw,
                "depth_channels": dc,
                "depth_dtype": dtype_name,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "channel_max_diff": (
                    channel_diff
                ),
                "size_match": int(
                    size_match
                ),
                "metric_valid_ratio": (
                    ""
                    if valid_ratio_value is None
                    else f"{valid_ratio_value:.8f}"
                ),
            }
        )

        if (
            index % 100 == 0
            or index == len(common_stems)
        ):
            print(
                f"[{index:4d}/"
                f"{len(common_stems):4d}] "
                f"processed"
            )

    # -----------------------------------------------------------------
    # Metric depth statistics
    # -----------------------------------------------------------------

    metric_global_min, metric_global_max = (
        histogram_nonzero_range(
            metric_hist
        )
    )

    metric_valid_hist = (
        metric_hist.copy()
    )

    metric_valid_hist[:300] = 0
    metric_valid_hist[20000:] = 0

    metric_valid_min, metric_valid_max = (
        histogram_nonzero_range(
            metric_valid_hist
        )
    )

    # -----------------------------------------------------------------
    # uint8 statistics
    # -----------------------------------------------------------------

    uint8_global_min, uint8_global_max = (
        histogram_nonzero_range(
            uint8_hist
        )
    )

    uint8_total_pixels = int(
        uint8_hist.sum()
    )

    uint8_zero_pixels = int(
        uint8_hist[0]
    )

    # -----------------------------------------------------------------
    # Sort problematic samples
    # -----------------------------------------------------------------

    low_valid_samples = sorted(
        low_valid_samples,
        key=lambda x: (
            x["valid_ratio"],
            x["stem"],
        ),
    )

    # -----------------------------------------------------------------
    # Result JSON
    # -----------------------------------------------------------------

    result = {
        "paths": {
            "code_root": str(
                CODE_ROOT
            ),
            "workspace_root": str(
                WORKSPACE_ROOT
            ),
            "dataset_root": str(
                dataset_root
            ),
            "visible_dir": str(
                visible_dir
            ),
            "depth_dir": str(
                depth_dir
            ),
            "output_dir": str(
                output_dir
            ),
        },

        "sample_alignment": {
            "visible_count": len(
                visible_map
            ),
            "depth_count": len(
                depth_map
            ),
            "common_count": len(
                common_stems
            ),
            "visible_only_count": len(
                visible_only
            ),
            "depth_only_count": len(
                depth_only
            ),
            "visible_only": (
                visible_only
            ),
            "depth_only": (
                depth_only
            ),
            "visible_duplicates": {
                k: [
                    str(p)
                    for p in v
                ]
                for k, v
                in visible_duplicates.items()
            },
            "depth_duplicates": {
                k: [
                    str(p)
                    for p in v
                ]
                for k, v
                in depth_duplicates.items()
            },
        },

        "file_formats": {
            "depth_extension_counts": dict(
                sorted(
                    extension_counts.items()
                )
            ),
            "visible_depth_extension_pairs": {
                f"{k[0]} -> {k[1]}": v
                for k, v
                in sorted(
                    extension_pair_counts.items()
                )
            },
            "depth_dtype_counts": dict(
                sorted(
                    dtype_counts.items()
                )
            ),
            "depth_dtype_channel_counts": dict(
                sorted(
                    dtype_channel_counts.items()
                )
            ),
            "depth_shape_counts": dict(
                sorted(
                    depth_shape_counts.items()
                )
            ),
            "visible_shape_counts": dict(
                sorted(
                    visible_shape_counts.items()
                )
            ),
            "max_depth_channel_difference": (
                max_depth_channel_difference
            ),
        },

        "spatial_alignment": {
            "size_mismatch_count": len(
                size_mismatch_samples
            ),
            "size_mismatch_samples": (
                size_mismatch_samples
            ),
        },

        "metric_uint16": {
            "file_count": (
                metric_file_count
            ),
            "total_pixels": (
                metric_total_pixels
            ),
            "global_min": (
                metric_global_min
            ),
            "global_max": (
                metric_global_max
            ),
            "zero_pixels": (
                metric_zero_pixels
            ),
            "zero_ratio": ratio(
                metric_zero_pixels,
                metric_total_pixels,
            ),
            "invalid_low_definition": (
                "depth < 300 mm"
            ),
            "invalid_low_pixels": (
                metric_invalid_low_pixels
            ),
            "invalid_low_ratio": ratio(
                metric_invalid_low_pixels,
                metric_total_pixels,
            ),
            "invalid_high_definition": (
                "depth >= 20000 mm"
            ),
            "invalid_high_pixels": (
                metric_invalid_high_pixels
            ),
            "invalid_high_ratio": ratio(
                metric_invalid_high_pixels,
                metric_total_pixels,
            ),
            "valid_definition": (
                "300 <= depth <= 19999 mm"
            ),
            "valid_pixels": (
                metric_valid_pixels
            ),
            "valid_ratio": ratio(
                metric_valid_pixels,
                metric_total_pixels,
            ),
            "valid_min": (
                metric_valid_min
            ),
            "valid_max": (
                metric_valid_max
            ),
            "valid_percentiles_mm": (
                percentile_dict(
                    metric_valid_hist
                )
            ),
            "per_image_valid_ratio_percentiles": (
                safe_numpy_percentiles(
                    per_image_metric_valid_ratio
                )
            ),
            "low_valid_ratio_threshold": (
                0.05
            ),
            "low_valid_sample_count": len(
                low_valid_samples
            ),
            "low_valid_samples": (
                low_valid_samples
            ),
        },

        "uint8_depth": {
            "file_count": (
                uint8_file_count
            ),
            "total_pixels": (
                uint8_total_pixels
            ),
            "global_min": (
                uint8_global_min
            ),
            "global_max": (
                uint8_global_max
            ),
            "zero_pixels": (
                uint8_zero_pixels
            ),
            "zero_ratio": ratio(
                uint8_zero_pixels,
                uint8_total_pixels,
            ),
            "percentiles": (
                percentile_dict(
                    uint8_hist
                )
            ),
        },

        "other_depth": {
            "file_count": (
                other_file_count
            ),
        },

        "io_errors": {
            "unreadable_visible": (
                unreadable_visible
            ),
            "unreadable_depth": (
                unreadable_depth
            ),
        },
    }

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

    csv_path = (
        output_dir
        / "depth_file_stats.csv"
    )

    if file_rows:
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    file_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                file_rows
            )

    # -----------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------

    json_path = (
        output_dir
        / "depth_stats.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Human-readable report
    # -----------------------------------------------------------------

    lines = []

    def add(text=""):
        lines.append(
            str(text)
        )

    add("=" * 72)
    add("AIC2026 C04 DEPTH DATASET AUDIT")
    add("=" * 72)

    add()
    add("[1] SAMPLE ALIGNMENT")
    add(
        f"Visible files : "
        f"{len(visible_map)}"
    )
    add(
        f"Depth files   : "
        f"{len(depth_map)}"
    )
    add(
        f"Common stems  : "
        f"{len(common_stems)}"
    )
    add(
        f"Visible only  : "
        f"{len(visible_only)}"
    )
    add(
        f"Depth only    : "
        f"{len(depth_only)}"
    )

    add()
    add("[2] DEPTH FILE EXTENSIONS")

    for key, value in sorted(
        extension_counts.items()
    ):
        add(
            f"{key:>8s} : {value}"
        )

    add()
    add("[3] DEPTH DTYPE / CHANNELS")

    for key, value in sorted(
        dtype_channel_counts.items()
    ):
        add(
            f"{key:>16s} : {value}"
        )

    add()
    add("[4] DEPTH SHAPES")

    for key, value in sorted(
        depth_shape_counts.items()
    ):
        add(
            f"{key:>20s} : {value}"
        )

    add()
    add("[5] SPATIAL ALIGNMENT")
    add(
        f"RGB/Depth size mismatch : "
        f"{len(size_mismatch_samples)}"
    )

    add()
    add("[6] UINT16 METRIC DEPTH")
    add(
        f"Files              : "
        f"{metric_file_count}"
    )
    add(
        f"Global min / max   : "
        f"{metric_global_min} / "
        f"{metric_global_max}"
    )
    add(
        f"Valid min / max    : "
        f"{metric_valid_min} / "
        f"{metric_valid_max} mm"
    )
    add(
        f"Zero ratio         : "
        f"{format_ratio(ratio(metric_zero_pixels, metric_total_pixels))}"
    )
    add(
        f"Invalid <300 ratio : "
        f"{format_ratio(ratio(metric_invalid_low_pixels, metric_total_pixels))}"
    )
    add(
        f"Invalid >=20m      : "
        f"{format_ratio(ratio(metric_invalid_high_pixels, metric_total_pixels))}"
    )
    add(
        f"Valid ratio        : "
        f"{format_ratio(ratio(metric_valid_pixels, metric_total_pixels))}"
    )

    add()
    add("Valid depth global percentiles (mm):")

    metric_percentiles = (
        percentile_dict(
            metric_valid_hist
        )
    )

    for key, value in (
        metric_percentiles.items()
    ):
        add(
            f"P{key:<5s} : {value}"
        )

    add()
    add(
        "Per-image metric valid-ratio "
        "percentiles:"
    )

    per_image_percentiles = (
        safe_numpy_percentiles(
            per_image_metric_valid_ratio
        )
    )

    for key, value in (
        per_image_percentiles.items()
    ):
        add(
            f"P{key:<5s} : "
            f"{100.0 * value:.4f}%"
        )

    add()
    add(
        f"Samples with <5% valid metric depth: "
        f"{len(low_valid_samples)}"
    )

    for sample in (
        low_valid_samples[:30]
    ):
        add(
            f"  {sample['stem']} : "
            f"{100.0 * sample['valid_ratio']:.4f}%"
        )

    if len(low_valid_samples) > 30:
        add(
            f"  ... "
            f"{len(low_valid_samples) - 30} "
            f"more in JSON"
        )

    add()
    add("[7] UINT8 DEPTH / VISUALIZATION DEPTH")
    add(
        f"Files            : "
        f"{uint8_file_count}"
    )
    add(
        f"Global min / max : "
        f"{uint8_global_min} / "
        f"{uint8_global_max}"
    )
    add(
        f"Zero ratio       : "
        f"{format_ratio(ratio(uint8_zero_pixels, uint8_total_pixels))}"
    )

    add()
    add("uint8 global percentiles:")

    uint8_percentiles = (
        percentile_dict(
            uint8_hist
        )
    )

    for key, value in (
        uint8_percentiles.items()
    ):
        add(
            f"P{key:<5s} : {value}"
        )

    add()
    add("[8] MULTI-CHANNEL DEPTH CHECK")
    add(
        "Maximum inter-channel absolute "
        f"difference : {max_depth_channel_difference}"
    )

    add()
    add("[9] OTHER / UNSUPPORTED DEPTH")
    add(
        f"Files : {other_file_count}"
    )

    add()
    add("[10] IO ERRORS")
    add(
        f"Unreadable visible : "
        f"{len(unreadable_visible)}"
    )
    add(
        f"Unreadable depth   : "
        f"{len(unreadable_depth)}"
    )

    add()
    add("=" * 72)
    add("C04-01 AUDIT OUTPUT")
    add("=" * 72)
    add(
        f"JSON : {json_path}"
    )
    add(
        f"TXT  : "
        f"{output_dir / 'depth_stats.txt'}"
    )
    add(
        f"CSV  : {csv_path}"
    )
    add()
    add(
        "Do NOT freeze D_near / D_far "
        "until this report is reviewed."
    )
    add("=" * 72)

    report_text = "\n".join(
        lines
    )

    txt_path = (
        output_dir
        / "depth_stats.txt"
    )

    txt_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print(report_text)

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "AIC2026 training dataset root. "
            "Default is derived from script location."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Audit output directory. "
            "Default is derived from script location."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: "
            f"{dataset_root}"
        )

    run_audit(
        dataset_root=dataset_root,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

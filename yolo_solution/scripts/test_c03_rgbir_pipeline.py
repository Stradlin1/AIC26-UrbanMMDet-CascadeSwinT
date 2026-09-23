#!/usr/bin/env python3

import sys
from pathlib import Path

import mmcv
import numpy as np
import torch


# ============================================================
# Project paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# ldds/yolo_solution/scripts/xxx.py -> ldds
LDDS_ROOT = SCRIPT_PATH.parents[2]

MMDET_ROOT = LDDS_ROOT / "mmdetection"
sys.path.insert(0, str(MMDET_ROOT))

from mmengine.registry import init_default_scope
from mmdet.registry import TRANSFORMS
from mmdet.models.data_preprocessors import DetDataPreprocessor


# ============================================================
# Paths
# ============================================================

DATA_ROOT = (
    LDDS_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)

VISIBLE_DIR = DATA_ROOT / "visible"
IR_DIR = DATA_ROOT / "infrared"


# ============================================================
# C02 preprocessing constants
# ============================================================

IMAGE_SCALE = (1333, 800)

MEAN = [
    123.675,
    116.28,
    103.53,
]

STD = [
    58.395,
    57.12,
    57.375,
]

PAD_SIZE_DIVISOR = 32

NUM_TEST_IMAGES = 8


# ============================================================
# Intended C03 preprocessor
# ============================================================

class RGBIRDetDataPreprocessor(DetDataPreprocessor):
    """
    C03 data preprocessor.

    Input:
        6-channel image

        [B, G, R, B_ir, G_ir, R_ir]

    The two 3-channel blocks are processed independently using
    exactly the same DetDataPreprocessor configuration as C02.

    After bgr_to_rgb=True:

        [R, G, B, R_ir, G_ir, B_ir]
    """

    def forward(self, data, training=False):

        if self.batch_augments is not None:
            raise RuntimeError(
                "C03-v1 does not support batch_augments."
            )

        inputs = data["inputs"]

        # ----------------------------------------------------
        # pseudo_collate:
        # list[Tensor(C,H,W)]
        # ----------------------------------------------------

        if isinstance(inputs, (list, tuple)):

            rgb_inputs = []
            ir_inputs = []

            for index, tensor in enumerate(inputs):

                if tensor.ndim != 3:
                    raise ValueError(
                        f"Sample {index}: expected CHW tensor, "
                        f"got shape {tuple(tensor.shape)}"
                    )

                if tensor.shape[0] != 6:
                    raise ValueError(
                        f"Sample {index}: expected 6 channels, "
                        f"got shape {tuple(tensor.shape)}"
                    )

                rgb_inputs.append(
                    tensor[0:3]
                )

                ir_inputs.append(
                    tensor[3:6]
                )

        # ----------------------------------------------------
        # default_collate:
        # Tensor(B,C,H,W)
        # ----------------------------------------------------

        elif torch.is_tensor(inputs):

            if inputs.ndim != 4:
                raise ValueError(
                    "Expected BCHW tensor, "
                    f"got {tuple(inputs.shape)}"
                )

            if inputs.shape[1] != 6:
                raise ValueError(
                    "Expected 6 channels, "
                    f"got {tuple(inputs.shape)}"
                )

            rgb_inputs = inputs[:, 0:3]
            ir_inputs = inputs[:, 3:6]

        else:
            raise TypeError(
                f"Unsupported inputs type: {type(inputs)}"
            )

        # RGB receives the original data_samples so that
        # batch_input_shape / pad_shape are written exactly
        # as standard DetDataPreprocessor does.
        rgb_data = dict(
            inputs=rgb_inputs,
            data_samples=data.get(
                "data_samples",
                None,
            ),
        )

        # IR has identical spatial dimensions, so it does not
        # need to touch detection metadata.
        ir_data = dict(
            inputs=ir_inputs,
            data_samples=None,
        )

        rgb_out = super().forward(
            rgb_data,
            training=training,
        )

        ir_out = super().forward(
            ir_data,
            training=False,
        )

        rgb_tensor = rgb_out["inputs"]
        ir_tensor = ir_out["inputs"]

        if rgb_tensor.shape != ir_tensor.shape:
            raise RuntimeError(
                "RGB / IR processed tensor shapes differ:\n"
                f"RGB: {tuple(rgb_tensor.shape)}\n"
                f"IR : {tuple(ir_tensor.shape)}"
            )

        fused = torch.cat(
            [rgb_tensor, ir_tensor],
            dim=1,
        )

        return dict(
            inputs=fused,
            data_samples=rgb_out["data_samples"],
        )


# ============================================================
# Helpers
# ============================================================

def check_paths():

    required = (
        MMDET_ROOT,
        DATA_ROOT,
        VISIBLE_DIR,
        IR_DIR,
    )

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )


def find_pairs():

    suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
    }

    pairs = []

    for rgb_path in sorted(
        VISIBLE_DIR.iterdir()
    ):

        if not rgb_path.is_file():
            continue

        if rgb_path.suffix.lower() not in suffixes:
            continue

        ir_path = (
            IR_DIR
            / rgb_path.name
        )

        if not ir_path.exists():
            continue

        pairs.append(
            (rgb_path, ir_path)
        )

        if len(pairs) >= NUM_TEST_IMAGES:
            break

    if not pairs:
        raise RuntimeError(
            "No RGB / IR image pairs found."
        )

    return pairs


def read_color(path):

    image = mmcv.imread(
        str(path),
        flag="color",
        channel_order="bgr",
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    if image.ndim != 3:
        raise RuntimeError(
            f"Expected HWC image: {path}"
        )

    if image.shape[2] != 3:
        raise RuntimeError(
            f"Expected 3 channels: {path}"
        )

    return image


def make_results(
    image,
    img_path,
    img_id=0,
):

    h, w = image.shape[:2]

    return dict(
        img=image.copy(),
        img_id=img_id,
        img_path=str(img_path),
        ori_shape=(h, w),
        img_shape=(h, w),
    )


def max_numpy_diff(a, b):

    if a.shape != b.shape:
        raise RuntimeError(
            "Shape mismatch:\n"
            f"{a.shape}\n"
            f"{b.shape}"
        )

    aa = a.astype(
        np.int32,
        copy=False,
    )

    bb = b.astype(
        np.int32,
        copy=False,
    )

    return int(
        np.abs(
            aa - bb
        ).max()
    )


# ============================================================
# Main tests
# ============================================================

def main():

    check_paths()

    init_default_scope(
        "mmdet"
    )

    print("=" * 72)
    print("AIC2026 C03 RGB+IR PIPELINE COMPATIBILITY TEST")
    print("=" * 72)

    print("LDDS root      :", LDDS_ROOT)
    print("Visible dir    :", VISIBLE_DIR)
    print("Infrared dir   :", IR_DIR)

    print()
    print("C02 preprocessing:")
    print("  scale        :", IMAGE_SCALE)
    print("  keep_ratio   : True")
    print("  bgr_to_rgb   : True")
    print("  mean         :", MEAN)
    print("  std          :", STD)
    print("  pad divisor  :", PAD_SIZE_DIVISOR)

    # --------------------------------------------------------
    # Build actual MMDetection transforms
    # --------------------------------------------------------

    resize = TRANSFORMS.build(
        dict(
            type="Resize",
            scale=IMAGE_SCALE,
            keep_ratio=True,
        )
    )

    flip = TRANSFORMS.build(
        dict(
            type="RandomFlip",
            prob=1.0,
            direction="horizontal",
        )
    )

    pack = TRANSFORMS.build(
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
            ),
        )
    )

    # --------------------------------------------------------
    # Build standard C02 preprocessor
    # --------------------------------------------------------

    c02_preprocessor = (
        DetDataPreprocessor(
            mean=MEAN,
            std=STD,
            bgr_to_rgb=True,
            pad_size_divisor=(
                PAD_SIZE_DIVISOR
            ),
        )
    )

    # --------------------------------------------------------
    # Build intended C03 preprocessor
    # --------------------------------------------------------

    c03_preprocessor = (
        RGBIRDetDataPreprocessor(
            mean=MEAN,
            std=STD,
            bgr_to_rgb=True,
            pad_size_divisor=(
                PAD_SIZE_DIVISOR
            ),
        )
    )

    pairs = find_pairs()

    print()
    print(
        "Testing paired images:",
        len(pairs),
    )

    print()

    overall_max_resize_diff = 0
    overall_max_flip_diff = 0
    overall_max_rgb_pre_diff = 0
    overall_max_ir_pre_diff = 0

    for index, (
        rgb_path,
        ir_path,
    ) in enumerate(
        pairs,
        start=1,
    ):

        rgb = read_color(
            rgb_path
        )

        ir = read_color(
            ir_path
        )

        # ----------------------------------------------------
        # Raw alignment
        # ----------------------------------------------------

        if rgb.shape != ir.shape:
            raise RuntimeError(
                f"Spatial mismatch:\n"
                f"RGB: {rgb_path} {rgb.shape}\n"
                f"IR : {ir_path} {ir.shape}"
            )

        six = np.concatenate(
            [rgb, ir],
            axis=2,
        )

        if six.shape[2] != 6:
            raise RuntimeError(
                "6-channel concatenation failed."
            )

        # ----------------------------------------------------
        # Resize:
        #
        # Compare one 6-channel resize against independent
        # RGB / IR 3-channel resize operations.
        # ----------------------------------------------------

        rgb_res = resize(
            make_results(
                rgb,
                rgb_path,
                index,
            )
        )

        ir_res = resize(
            make_results(
                ir,
                ir_path,
                index,
            )
        )

        six_res = resize(
            make_results(
                six,
                rgb_path,
                index,
            )
        )

        rgb_r = rgb_res["img"]
        ir_r = ir_res["img"]
        six_r = six_res["img"]

        if six_r.ndim != 3:
            raise RuntimeError(
                f"Resize lost HWC layout: "
                f"{six_r.shape}"
            )

        if six_r.shape[2] != 6:
            raise RuntimeError(
                "MMDetection Resize does not preserve "
                f"6 channels: {six_r.shape}"
            )

        resize_rgb_diff = (
            max_numpy_diff(
                six_r[:, :, 0:3],
                rgb_r,
            )
        )

        resize_ir_diff = (
            max_numpy_diff(
                six_r[:, :, 3:6],
                ir_r,
            )
        )

        resize_diff = max(
            resize_rgb_diff,
            resize_ir_diff,
        )

        overall_max_resize_diff = max(
            overall_max_resize_diff,
            resize_diff,
        )

        # ----------------------------------------------------
        # RandomFlip:
        #
        # Again compare 6-channel flip against independent
        # RGB / IR flip operations.
        # ----------------------------------------------------

        rgb_flip = flip(
            make_results(
                rgb_r,
                rgb_path,
                index,
            )
        )["img"]

        ir_flip = flip(
            make_results(
                ir_r,
                ir_path,
                index,
            )
        )["img"]

        six_flip = flip(
            make_results(
                six_r,
                rgb_path,
                index,
            )
        )["img"]

        if six_flip.shape[2] != 6:
            raise RuntimeError(
                "RandomFlip does not preserve "
                f"6 channels: {six_flip.shape}"
            )

        flip_rgb_diff = (
            max_numpy_diff(
                six_flip[:, :, 0:3],
                rgb_flip,
            )
        )

        flip_ir_diff = (
            max_numpy_diff(
                six_flip[:, :, 3:6],
                ir_flip,
            )
        )

        flip_diff = max(
            flip_rgb_diff,
            flip_ir_diff,
        )

        overall_max_flip_diff = max(
            overall_max_flip_diff,
            flip_diff,
        )

        # ----------------------------------------------------
        # PackDetInputs
        # ----------------------------------------------------

        packed_rgb = pack(
            rgb_res
        )

        packed_ir = pack(
            ir_res
        )

        packed_six = pack(
            six_res
        )

        rgb_input = (
            packed_rgb["inputs"]
        )

        ir_input = (
            packed_ir["inputs"]
        )

        six_input = (
            packed_six["inputs"]
        )

        if six_input.ndim != 3:
            raise RuntimeError(
                "PackDetInputs output should "
                f"be CHW: {six_input.shape}"
            )

        if six_input.shape[0] != 6:
            raise RuntimeError(
                "PackDetInputs failed to preserve "
                f"6 channels: {six_input.shape}"
            )

        # ----------------------------------------------------
        # C02 standard preprocessing
        # ----------------------------------------------------

        c02_rgb_out = (
            c02_preprocessor(
                dict(
                    inputs=[
                        rgb_input
                    ],
                    data_samples=None,
                ),
                training=False,
            )["inputs"]
        )

        c02_ir_out = (
            c02_preprocessor(
                dict(
                    inputs=[
                        ir_input
                    ],
                    data_samples=None,
                ),
                training=False,
            )["inputs"]
        )

        # ----------------------------------------------------
        # Intended C03 preprocessing
        # ----------------------------------------------------

        c03_out = (
            c03_preprocessor(
                dict(
                    inputs=[
                        six_input
                    ],
                    data_samples=None,
                ),
                training=False,
            )["inputs"]
        )

        if c03_out.ndim != 4:
            raise RuntimeError(
                "C03 preprocessor output "
                f"must be BCHW: {c03_out.shape}"
            )

        if c03_out.shape[1] != 6:
            raise RuntimeError(
                "C03 preprocessor must output "
                f"6 channels: {c03_out.shape}"
            )

        c03_rgb = (
            c03_out[:, 0:3]
        )

        c03_ir = (
            c03_out[:, 3:6]
        )

        rgb_pre_diff = float(
            torch.max(
                torch.abs(
                    c03_rgb
                    - c02_rgb_out
                )
            ).item()
        )

        ir_pre_diff = float(
            torch.max(
                torch.abs(
                    c03_ir
                    - c02_ir_out
                )
            ).item()
        )

        overall_max_rgb_pre_diff = max(
            overall_max_rgb_pre_diff,
            rgb_pre_diff,
        )

        overall_max_ir_pre_diff = max(
            overall_max_ir_pre_diff,
            ir_pre_diff,
        )

        print(
            f"[{index:02d}] "
            f"{rgb_path.name}"
        )

        print(
            "     raw        :",
            rgb.shape,
            "->",
            six.shape,
        )

        print(
            "     resized    :",
            six_r.shape,
        )

        print(
            "     packed     :",
            tuple(
                six_input.shape
            ),
        )

        print(
            "     preprocessed:",
            tuple(
                c03_out.shape
            ),
        )

        print(
            "     resize diff:",
            resize_diff,
        )

        print(
            "     flip diff  :",
            flip_diff,
        )

        print(
            "     RGB parity :",
            f"{rgb_pre_diff:.9f}",
        )

        print(
            "     IR parity  :",
            f"{ir_pre_diff:.9f}",
        )

    # --------------------------------------------------------
    # Final assertions
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        "Resize max diff       :",
        overall_max_resize_diff,
    )

    print(
        "Flip max diff         :",
        overall_max_flip_diff,
    )

    print(
        "RGB preprocessing diff:",
        f"{overall_max_rgb_pre_diff:.9f}",
    )

    print(
        "IR preprocessing diff :",
        f"{overall_max_ir_pre_diff:.9f}",
    )

    if overall_max_resize_diff != 0:
        raise RuntimeError(
            "6ch Resize is not identical to "
            "independent 3ch resizing."
        )

    if overall_max_flip_diff != 0:
        raise RuntimeError(
            "6ch RandomFlip is not identical "
            "to independent 3ch flipping."
        )

    if overall_max_rgb_pre_diff != 0.0:
        raise RuntimeError(
            "C03 RGB preprocessing is not "
            "bit-identical to C02."
        )

    if overall_max_ir_pre_diff != 0.0:
        raise RuntimeError(
            "C03 IR branch preprocessing does "
            "not match standard C02 preprocessing."
        )

    print()
    print("PASS: raw RGB / IR spatial alignment")
    print("PASS: 6-channel Resize")
    print("PASS: 6-channel RandomFlip")
    print("PASS: 6-channel PackDetInputs")
    print("PASS: RGB preprocessing parity with C02")
    print("PASS: IR preprocessing parity")
    print()
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

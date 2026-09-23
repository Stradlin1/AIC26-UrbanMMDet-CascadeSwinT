#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import Compose

SCRIPT_PATH = Path(__file__).resolve()
LDDS_ROOT = SCRIPT_PATH.parents[2]
MMDET_ROOT = LDDS_ROOT / "mmdetection"
SCRIPTS_DIR = SCRIPT_PATH.parent
sys.path.insert(0, str(MMDET_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from mmengine.registry import init_default_scope
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS, TRANSFORMS

import c04_rgbd_components  # noqa: F401,E402
from c04_rgbd_components import DEPTH_FAR_MM, DEPTH_NEAR_MM, depth_to_uint8_v1

DATA_ROOT = LDDS_ROOT / "datasets" / "AIC2026_Train_2000"
VISIBLE_DIR = DATA_ROOT / "visible"
DEPTH_DIR = DATA_ROOT / "depth"
IMAGE_SCALE = (1333, 800)
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
PAD_SIZE_DIVISOR = 32


def assert_np_equal(a, b, name):
    if a.shape != b.shape or not np.array_equal(a, b):
        diff = float("inf") if a.shape != b.shape else int(
            np.abs(a.astype(np.int64) - b.astype(np.int64)).max()
        )
        raise AssertionError(f"{name}: shape/diff failure, max_diff={diff}")


def max_torch_diff(a, b):
    if a.shape != b.shape:
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def find_examples():
    metric = None
    jpg = None
    for rgb_path in sorted(VISIBLE_DIR.iterdir()):
        if not rgb_path.is_file():
            continue
        depth_path = DEPTH_DIR / rgb_path.name
        if not depth_path.exists():
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        if metric is None and depth.dtype == np.uint16 and depth.ndim == 2:
            metric = (rgb_path, depth_path)
        if jpg is None and depth.dtype == np.uint8 and depth.ndim == 3:
            jpg = (rgb_path, depth_path)
        if metric is not None and jpg is not None:
            return metric, jpg
    raise RuntimeError("Could not find both uint16 and uint8 Depth examples")


def reference_depth(depth_raw):
    if depth_raw.dtype == np.uint8:
        if depth_raw.ndim == 2:
            return depth_raw.copy()
        return cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

    if depth_raw.dtype != np.uint16 or depth_raw.ndim != 2:
        raise RuntimeError(f"Unexpected depth format: {depth_raw.dtype} {depth_raw.shape}")

    out = np.zeros(depth_raw.shape, dtype=np.uint8)
    valid = (depth_raw >= DEPTH_NEAR_MM) & (depth_raw < DEPTH_FAR_MM)
    d = depth_raw[valid].astype(np.float64, copy=False)
    if d.size:
        mapped = 1.0 + np.rint(
            (DEPTH_FAR_MM - d) * 254.0 / (DEPTH_FAR_MM - DEPTH_NEAR_MM)
        )
        out[valid] = np.clip(mapped, 1, 255).astype(np.uint8)
    return out


def raw_loader_test(pair):
    rgb_path, depth_path = pair
    loader = TRANSFORMS.build(dict(type="LoadRGBDepthFromFile"))
    out = loader(dict(img_path=str(rgb_path)))
    image = out["img"]
    if image.ndim != 3 or image.shape[2] != 6:
        raise AssertionError(f"Expected HxWx6, got {image.shape}")

    rgb_ref = mmcv.imread(str(rgb_path), flag="color", channel_order="bgr")
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth_ref = reference_depth(depth_raw)

    assert_np_equal(image[:, :, 0:3], rgb_ref, "raw RGB parity")
    assert_np_equal(image[:, :, 3], depth_ref, "Depth-v1 parity")
    assert_np_equal(image[:, :, 3], image[:, :, 4], "Depth replicate 3/4")
    assert_np_equal(image[:, :, 3], image[:, :, 5], "Depth replicate 3/5")
    return out


def geometry_test(rgb_path):
    loader = TRANSFORMS.build(dict(type="LoadRGBDepthFromFile"))
    resize = TRANSFORMS.build(dict(type="Resize", scale=IMAGE_SCALE, keep_ratio=True))

    rgbd = Compose([loader, resize])(dict(img_path=str(rgb_path)))

    rgb = mmcv.imread(str(rgb_path), flag="color", channel_order="bgr")
    rgb_only = TRANSFORMS.build(
        dict(type="Resize", scale=IMAGE_SCALE, keep_ratio=True)
    )(
        dict(img=rgb.copy(), img_shape=rgb.shape[:2], ori_shape=rgb.shape[:2])
    )
    assert_np_equal(rgbd["img"][:, :, 0:3], rgb_only["img"], "Resize RGB parity")

    before = rgbd["img"].copy()
    flipped = TRANSFORMS.build(
        dict(type="RandomFlip", prob=1.0, direction="horizontal")
    )(dict(rgbd, img=before.copy()))
    assert_np_equal(flipped["img"], np.flip(before, axis=1), "6ch flip sync")


def preprocess_test(rgb_path):
    pipeline = Compose([
        TRANSFORMS.build(dict(type="LoadRGBDepthFromFile")),
        TRANSFORMS.build(dict(type="Resize", scale=IMAGE_SCALE, keep_ratio=True)),
        TRANSFORMS.build(dict(
            type="PackDetInputs",
            meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor"),
        )),
    ])
    packed = pipeline(dict(img_path=str(rgb_path)))
    tensor = packed["inputs"]
    if tensor.ndim != 3 or tensor.shape[0] != 6:
        raise AssertionError(f"PackDetInputs expected 6xHxW, got {tuple(tensor.shape)}")

    c02 = DetDataPreprocessor(
        mean=MEAN,
        std=STD,
        bgr_to_rgb=True,
        pad_size_divisor=PAD_SIZE_DIVISOR,
    )
    c04 = MODELS.build(dict(
        type="RGBDDetDataPreprocessor",
        mean=MEAN,
        std=STD,
        bgr_to_rgb=True,
        pad_size_divisor=PAD_SIZE_DIVISOR,
    ))

    c02_out = c02(
        dict(inputs=[tensor[0:3]], data_samples=None), training=False
    )["inputs"]
    c04_out_1 = c04(
        dict(inputs=[tensor], data_samples=None), training=False
    )["inputs"]
    c04_out_2 = c04(
        dict(inputs=[tensor.clone()], data_samples=None), training=False
    )["inputs"]

    rgb_diff = max_torch_diff(c02_out, c04_out_1[:, 0:3])
    repeat_diff = max_torch_diff(c04_out_1, c04_out_2)
    if rgb_diff != 0.0:
        raise AssertionError(f"C02 RGB preprocessing parity failed: {rgb_diff}")
    if repeat_diff != 0.0:
        raise AssertionError(f"C04 preprocessing is non-deterministic: {repeat_diff}")
    if not torch.isfinite(c04_out_1).all():
        raise AssertionError("C04 preprocessing produced NaN/Inf")

    print("Packed tensor        :", tuple(tensor.shape), tensor.dtype)
    print("Preprocessed tensor  :", tuple(c04_out_1.shape), c04_out_1.dtype)
    print("RGB C02 max diff     :", rgb_diff)
    print("Depth processed min  :", float(c04_out_1[:, 3:6].min().item()))
    print("Depth processed max  :", float(c04_out_1[:, 3:6].max().item()))


def mapping_definition_test():
    probe = np.asarray([[0, 299, 300, 1000, 7377, 19999, 20000, 65535]], dtype=np.uint16)
    mapped = depth_to_uint8_v1(probe)
    expected_endpoint = [0, 0, 255, None, None, 1, 0, 0]
    for i, expected in enumerate(expected_endpoint):
        if expected is not None and int(mapped[0, i]) != expected:
            raise AssertionError(
                f"Depth mapping endpoint failed at {int(probe[0,i])}: "
                f"got {int(mapped[0,i])}, expected {expected}"
            )


def main():
    print("=" * 72)
    print("AIC2026 C04 RGBD PIPELINE TEST")
    print("=" * 72)
    print("LDDS root    :", LDDS_ROOT)
    print("Dataset root :", DATA_ROOT)

    for path in (MMDET_ROOT, VISIBLE_DIR, DEPTH_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    init_default_scope("mmdet")
    metric_pair, jpg_pair = find_examples()
    print("Metric sample:", metric_pair[0].name)
    print("JPG sample   :", jpg_pair[0].name)

    mapping_definition_test()
    raw_loader_test(metric_pair)
    raw_loader_test(jpg_pair)
    geometry_test(metric_pair[0])
    preprocess_test(metric_pair[0])

    print()
    print("PASS: Depth representation v1 definition")
    print("PASS: uint16 metric Depth loader")
    print("PASS: uint8/JPG Depth loader")
    print("PASS: raw RGB / Depth spatial alignment")
    print("PASS: 6-channel Resize")
    print("PASS: 6-channel RandomFlip")
    print("PASS: 6-channel PackDetInputs")
    print("PASS: RGB preprocessing parity with C02")
    print("PASS: Depth preprocessing deterministic / finite")
    print("STATUS = PASS")


if __name__ == "__main__":
    main()

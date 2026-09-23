#!/usr/bin/env python3

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
LDDS_ROOT = SCRIPT_PATH.parents[2]

MMDET_ROOT = LDDS_ROOT / "mmdetection"
COMPONENT_DIR = LDDS_ROOT / "yolo_solution" / "scripts"

sys.path.insert(
    0,
    str(MMDET_ROOT),
)

sys.path.insert(
    0,
    str(COMPONENT_DIR),
)

import mmcv

print("=" * 72)
print("C03 COMPONENT IMPORT TEST")
print("=" * 72)

print("mmcv module  :", mmcv)
print(
    "mmcv file    :",
    getattr(mmcv, "__file__", None),
)
print(
    "mmcv version :",
    getattr(mmcv, "__version__", "MISSING"),
)

if not hasattr(mmcv, "__version__"):
    raise RuntimeError(
        "Wrong MMCV module was imported."
    )

from c03_rgbir_components import (
    LoadRGBIRFromFile,
    RGBIRDetDataPreprocessor,
    DualSwinGatedBackbone,
)

print()
print("Import PASS")
print("Loader       :", LoadRGBIRFromFile)
print("Preprocessor :", RGBIRDetDataPreprocessor)
print("Backbone     :", DualSwinGatedBackbone)

print()
print("STATUS = PASS")
print("=" * 72)

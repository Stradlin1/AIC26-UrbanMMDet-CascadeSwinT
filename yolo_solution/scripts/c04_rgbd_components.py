#!/usr/bin/env python3

from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import BaseTransform
from mmengine.model import BaseModule
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS, TRANSFORMS

DEPTH_NEAR_MM = 300
DEPTH_FAR_MM = 20000


def depth_to_uint8_v1(depth: np.ndarray) -> np.ndarray:
    """C04-v1 Depth representation: invalid=0, near=bright, far=dark."""
    if depth is None:
        raise ValueError("Depth image is None")

    if depth.dtype == np.uint16:
        if depth.ndim != 2:
            raise RuntimeError(f"uint16 Depth must be HxW, got {depth.shape}")
        out = np.zeros(depth.shape, dtype=np.uint8)
        valid = (depth >= DEPTH_NEAR_MM) & (depth < DEPTH_FAR_MM)
        if np.any(valid):
            d = depth[valid].astype(np.float32, copy=False)
            mapped = 1.0 + np.rint(
                (DEPTH_FAR_MM - d) * 254.0
                / (DEPTH_FAR_MM - DEPTH_NEAR_MM)
            )
            out[valid] = np.clip(mapped, 1, 255).astype(np.uint8)
        return out

    if depth.dtype == np.uint8:
        if depth.ndim == 2:
            return depth.copy()
        if depth.ndim == 3 and depth.shape[2] == 3:
            return cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        raise RuntimeError(f"uint8 Depth must be HxW or HxWx3, got {depth.shape}")

    raise RuntimeError(f"Unsupported Depth dtype={depth.dtype}, shape={depth.shape}")


@TRANSFORMS.register_module()
class LoadRGBDepthFromFile(BaseTransform):
    """Load RGB + Depth as HxWx6 transport tensor [B,G,R,D,D,D]."""

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def transform(self, results):
        if "img_path" not in results:
            raise KeyError("LoadRGBDepthFromFile requires results['img_path']")

        rgb_path = Path(results["img_path"])
        if rgb_path.parent.name != "visible":
            raise RuntimeError(f"Expected RGB path under visible/: {rgb_path}")

        depth_path = rgb_path.parent.parent / "depth" / rgb_path.name
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB not found: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth not found: {depth_path}")

        rgb = mmcv.imread(str(rgb_path), flag="color", channel_order="bgr")
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError(f"Failed to load RGB: {rgb_path}")
        if depth_raw is None:
            raise RuntimeError(f"Failed to load Depth: {depth_path}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"RGB must be HxWx3, got {rgb.shape}")

        depth_gray = depth_to_uint8_v1(depth_raw)
        if rgb.shape[:2] != depth_gray.shape[:2]:
            raise RuntimeError(
                "RGB/Depth spatial mismatch: "
                f"RGB={rgb.shape}, Depth={depth_raw.shape}, file={rgb_path.name}"
            )

        depth_3ch = np.repeat(depth_gray[..., None], 3, axis=2)
        image = np.concatenate([rgb, depth_3ch], axis=2)
        if self.to_float32:
            image = image.astype(np.float32)

        h, w = image.shape[:2]
        results["img"] = image
        results["img_shape"] = (h, w)
        results["ori_shape"] = (h, w)
        results["depth_path"] = str(depth_path)
        return results


@MODELS.register_module()
class RGBDDetDataPreprocessor(DetDataPreprocessor):
    """Apply C02 preprocessing independently to RGB and replicated Depth."""

    def forward(self, data, training=False):
        if self.batch_augments is not None:
            raise RuntimeError("C04-v1 does not use batch_augments")

        inputs = data["inputs"]
        if isinstance(inputs, (list, tuple)):
            rgb_inputs, depth_inputs = [], []
            for i, x in enumerate(inputs):
                if x.ndim != 3 or x.shape[0] != 6:
                    raise ValueError(f"Sample {i}: expected 6xHxW, got {tuple(x.shape)}")
                rgb_inputs.append(x[0:3])
                depth_inputs.append(x[3:6])
        elif torch.is_tensor(inputs):
            if inputs.ndim != 4 or inputs.shape[1] != 6:
                raise ValueError(f"Expected Bx6xHxW, got {tuple(inputs.shape)}")
            rgb_inputs = inputs[:, 0:3]
            depth_inputs = inputs[:, 3:6]
        else:
            raise TypeError(f"Unsupported inputs type: {type(inputs)}")

        rgb_out = super().forward(
            dict(inputs=rgb_inputs, data_samples=data.get("data_samples", None)),
            training=training,
        )
        depth_out = super().forward(
            dict(inputs=depth_inputs, data_samples=None),
            training=False,
        )

        rgb_tensor = rgb_out["inputs"]
        depth_tensor = depth_out["inputs"]
        if rgb_tensor.shape != depth_tensor.shape:
            raise RuntimeError(
                f"Processed RGB/Depth mismatch: {tuple(rgb_tensor.shape)} vs "
                f"{tuple(depth_tensor.shape)}"
            )

        return dict(
            inputs=torch.cat([rgb_tensor, depth_tensor], dim=1),
            data_samples=rgb_out["data_samples"],
        )


@MODELS.register_module()
class DualSwinDepthGatedBackbone(BaseModule):
    """C03-mirrored RGB+Depth Dual-Swin with zero-init gated residual fusion."""

    def __init__(
        self,
        rgb_backbone,
        depth_backbone,
        channels=(96, 192, 384, 768),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.channels = tuple(channels)
        if len(self.channels) != 4:
            raise ValueError("C04 expects four Swin stages")

        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.depth_backbone = MODELS.build(depth_backbone)
        self.adapters = nn.ModuleList([
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
            for c in self.channels
        ])
        self.gates = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.reset_fusion_parameters()

    def reset_fusion_parameters(self):
        with torch.no_grad():
            for adapter, channel in zip(self.adapters, self.channels):
                adapter.weight.zero_()
                idx = torch.arange(channel, device=adapter.weight.device)
                adapter.weight[idx, idx, 0, 0] = 1.0
                if adapter.bias is not None:
                    adapter.bias.zero_()
            self.gates.zero_()

    def init_weights(self):
        super().init_weights()
        self.reset_fusion_parameters()

    def gate_values_tensor(self):
        return torch.tanh(self.gates)

    def gate_values(self):
        return self.gate_values_tensor().detach().cpu().tolist()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected Bx6xHxW input, got {tuple(x.shape)}")

        rgb_features = self.rgb_backbone(x[:, 0:3])
        depth_features = self.depth_backbone(x[:, 3:6])
        if len(rgb_features) != 4 or len(depth_features) != 4:
            raise RuntimeError("Both Swin backbones must return four stages")

        alphas = self.gate_values_tensor()
        fused = []
        for i, (rgb, depth, adapter, channel) in enumerate(
            zip(rgb_features, depth_features, self.adapters, self.channels)
        ):
            if rgb.shape != depth.shape:
                raise RuntimeError(
                    f"Stage {i} RGB/Depth shape mismatch: {tuple(rgb.shape)} vs "
                    f"{tuple(depth.shape)}"
                )
            if rgb.shape[1] != channel:
                raise RuntimeError(
                    f"Stage {i}: expected {channel} channels, got {rgb.shape[1]}"
                )
            alpha = alphas[i].to(dtype=rgb.dtype, device=rgb.device)
            fused.append(rgb + alpha * adapter(depth))

        return tuple(fused)

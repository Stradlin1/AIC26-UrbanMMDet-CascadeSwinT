#!/usr/bin/env python3

from pathlib import Path

import mmcv
import torch
import torch.nn as nn

from mmcv.transforms import BaseTransform
from mmengine.model import BaseModule

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS, TRANSFORMS


# ============================================================
# RGB + IR loader
# ============================================================

@TRANSFORMS.register_module()
class LoadRGBIRFromFile(BaseTransform):
    """
    Load paired RGB + IR images.

    COCO annotations continue to point to:

        datasets/AIC2026_Train_2000/visible/<name>

    The matching IR image is resolved as:

        datasets/AIC2026_Train_2000/infrared/<name>

    Output image layout:

        H x W x 6

        [
            B_rgb,
            G_rgb,
            R_rgb,
            B_ir,
            G_ir,
            R_ir,
        ]

    This 6-channel representation is only a transport format.
    The model later splits RGB and IR into independent backbones.
    """

    def __init__(
        self,
        to_float32=False,
    ):
        self.to_float32 = to_float32

    def transform(self, results):

        if "img_path" not in results:
            raise KeyError(
                "LoadRGBIRFromFile requires results['img_path']"
            )

        rgb_path = Path(
            results["img_path"]
        )

        if rgb_path.parent.name != "visible":
            raise RuntimeError(
                "Expected RGB path inside a 'visible' directory:\n"
                f"{rgb_path}"
            )

        ir_path = (
            rgb_path.parent.parent
            / "infrared"
            / rgb_path.name
        )

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB image not found:\n{rgb_path}"
            )

        if not ir_path.exists():
            raise FileNotFoundError(
                f"IR image not found:\n{ir_path}"
            )

        rgb = mmcv.imread(
            str(rgb_path),
            flag="color",
            channel_order="bgr",
        )

        ir = mmcv.imread(
            str(ir_path),
            flag="color",
            channel_order="bgr",
        )

        if rgb is None:
            raise RuntimeError(
                f"Failed to load RGB image:\n{rgb_path}"
            )

        if ir is None:
            raise RuntimeError(
                f"Failed to load IR image:\n{ir_path}"
            )

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(
                "RGB must be HWC 3-channel, got "
                f"{rgb.shape}"
            )

        if ir.ndim != 3 or ir.shape[2] != 3:
            raise RuntimeError(
                "IR must be HWC 3-channel, got "
                f"{ir.shape}"
            )

        if rgb.shape != ir.shape:
            raise RuntimeError(
                "RGB / IR spatial mismatch:\n"
                f"RGB: {rgb_path} {rgb.shape}\n"
                f"IR : {ir_path} {ir.shape}"
            )

        image = torch.from_numpy(
            __import__("numpy").concatenate(
                [rgb, ir],
                axis=2,
            )
        ).numpy()

        if self.to_float32:
            image = image.astype(
                "float32"
            )

        h, w = image.shape[:2]

        results["img"] = image

        results["img_shape"] = (
            h,
            w,
        )

        results["ori_shape"] = (
            h,
            w,
        )

        results["ir_path"] = str(
            ir_path
        )

        return results


# ============================================================
# RGB + IR data preprocessor
# ============================================================

@MODELS.register_module()
class RGBIRDetDataPreprocessor(
    DetDataPreprocessor
):
    """
    Preprocess a 6-channel RGB+IR transport tensor.

    Input before preprocessing:

        [B,G,R, B_ir,G_ir,R_ir]

    Each 3-channel block is independently processed using the
    exact same DetDataPreprocessor configuration as C02.

    With bgr_to_rgb=True, output becomes:

        [R,G,B, R_ir,G_ir,B_ir]

    RGB preprocessing is therefore identical to C02.
    """

    def forward(
        self,
        data,
        training=False,
    ):

        if self.batch_augments is not None:
            raise RuntimeError(
                "C03-v1 does not use batch_augments."
            )

        inputs = data["inputs"]

        # ----------------------------------------------------
        # pseudo_collate case
        # list[Tensor(C,H,W)]
        # ----------------------------------------------------

        if isinstance(
            inputs,
            (list, tuple),
        ):

            rgb_inputs = []
            ir_inputs = []

            for index, tensor in enumerate(
                inputs
            ):

                if tensor.ndim != 3:
                    raise ValueError(
                        f"Sample {index}: expected CHW, "
                        f"got {tuple(tensor.shape)}"
                    )

                if tensor.shape[0] != 6:
                    raise ValueError(
                        f"Sample {index}: expected 6 channels, "
                        f"got {tuple(tensor.shape)}"
                    )

                rgb_inputs.append(
                    tensor[0:3]
                )

                ir_inputs.append(
                    tensor[3:6]
                )

        # ----------------------------------------------------
        # default_collate case
        # Tensor(B,C,H,W)
        # ----------------------------------------------------

        elif torch.is_tensor(
            inputs
        ):

            if inputs.ndim != 4:
                raise ValueError(
                    "Expected BCHW tensor, got "
                    f"{tuple(inputs.shape)}"
                )

            if inputs.shape[1] != 6:
                raise ValueError(
                    "Expected 6 channels, got "
                    f"{tuple(inputs.shape)}"
                )

            rgb_inputs = (
                inputs[:, 0:3]
            )

            ir_inputs = (
                inputs[:, 3:6]
            )

        else:
            raise TypeError(
                "Unsupported input type: "
                f"{type(inputs)}"
            )

        # ----------------------------------------------------
        # RGB:
        # preserve DataSample metadata exactly as standard C02
        # ----------------------------------------------------

        rgb_data = dict(
            inputs=rgb_inputs,
            data_samples=data.get(
                "data_samples",
                None,
            ),
        )

        # ----------------------------------------------------
        # IR:
        # same numerical preprocessing, no metadata mutation
        # ----------------------------------------------------

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

        rgb_tensor = (
            rgb_out["inputs"]
        )

        ir_tensor = (
            ir_out["inputs"]
        )

        if (
            rgb_tensor.shape
            != ir_tensor.shape
        ):
            raise RuntimeError(
                "Processed RGB / IR shape mismatch:\n"
                f"RGB: {tuple(rgb_tensor.shape)}\n"
                f"IR : {tuple(ir_tensor.shape)}"
            )

        inputs_out = torch.cat(
            [
                rgb_tensor,
                ir_tensor,
            ],
            dim=1,
        )

        return dict(
            inputs=inputs_out,
            data_samples=(
                rgb_out["data_samples"]
            ),
        )


# ============================================================
# Dual Swin + gated residual fusion
# ============================================================

@MODELS.register_module()
class DualSwinGatedBackbone(
    BaseModule
):
    """
    C03 RGB + IR backbone.

    RGB:
        independent Swin-T

    IR:
        independent Swin-T

    Stage-wise fusion:

        F_i =
            RGB_i
            +
            tanh(g_i) * Adapter_i(IR_i)

    Gate initialization:

        g_i = 0
        tanh(g_i) = 0

    Therefore at initialization:

        F_i == RGB_i

    which preserves the RGB path exactly.
    """

    def __init__(
        self,
        rgb_backbone,
        ir_backbone,
        channels=(
            96,
            192,
            384,
            768,
        ),
        init_cfg=None,
    ):
        super().__init__(
            init_cfg=init_cfg
        )

        self.channels = tuple(
            channels
        )

        if len(self.channels) != 4:
            raise ValueError(
                "C03 expects exactly four Swin stages."
            )

        self.rgb_backbone = (
            MODELS.build(
                rgb_backbone
            )
        )

        self.ir_backbone = (
            MODELS.build(
                ir_backbone
            )
        )

        self.adapters = (
            nn.ModuleList()
        )

        for channel in self.channels:

            adapter = nn.Conv2d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

            self.adapters.append(
                adapter
            )

        # Raw trainable gate parameters.
        #
        # Actual gate:
        #     alpha = tanh(gate)
        #
        # Initial:
        #     gate = 0
        #     alpha = 0
        self.gates = nn.Parameter(
            torch.zeros(
                len(self.channels),
                dtype=torch.float32,
            )
        )

        self.reset_fusion_parameters()

    # --------------------------------------------------------
    # Fusion initialization
    # --------------------------------------------------------

    def reset_fusion_parameters(
        self,
    ):

        with torch.no_grad():

            for (
                adapter,
                channel,
            ) in zip(
                self.adapters,
                self.channels,
            ):

                adapter.weight.zero_()

                diagonal = torch.arange(
                    channel,
                    device=(
                        adapter.weight.device
                    ),
                )

                adapter.weight[
                    diagonal,
                    diagonal,
                    0,
                    0,
                ] = 1.0

                if adapter.bias is not None:
                    adapter.bias.zero_()

            self.gates.zero_()

    def init_weights(self):

        # Let MMEngine/MMDetection initialize the two
        # Swin modules normally first.
        super().init_weights()

        # Then restore the deterministic fusion initialization.
        #
        # External checkpoints are loaded afterwards by Runner,
        # so RGB / IR backbone weights can safely overwrite
        # their initialized values without affecting adapters
        # or zero gates.
        self.reset_fusion_parameters()

    # --------------------------------------------------------
    # Gate helpers
    # --------------------------------------------------------

    def gate_values_tensor(
        self,
    ):
        return torch.tanh(
            self.gates
        )

    def gate_values(
        self,
    ):
        return (
            self
            .gate_values_tensor()
            .detach()
            .cpu()
            .tolist()
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x,
    ):

        if x.ndim != 4:
            raise ValueError(
                "DualSwinGatedBackbone expects BCHW input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != 6:
            raise ValueError(
                "DualSwinGatedBackbone expects 6 channels, "
                f"got {tuple(x.shape)}"
            )

        rgb = x[:, 0:3]
        ir = x[:, 3:6]

        rgb_features = (
            self.rgb_backbone(
                rgb
            )
        )

        ir_features = (
            self.ir_backbone(
                ir
            )
        )

        if len(rgb_features) != 4:
            raise RuntimeError(
                "RGB Swin must return four feature stages."
            )

        if len(ir_features) != 4:
            raise RuntimeError(
                "IR Swin must return four feature stages."
            )

        alphas = (
            self.gate_values_tensor()
        )

        fused = []

        for index, (
            rgb_feature,
            ir_feature,
            adapter,
            expected_channel,
        ) in enumerate(
            zip(
                rgb_features,
                ir_features,
                self.adapters,
                self.channels,
            )
        ):

            if (
                rgb_feature.shape
                != ir_feature.shape
            ):
                raise RuntimeError(
                    f"Stage {index}: RGB/IR feature "
                    "shape mismatch:\n"
                    f"RGB {tuple(rgb_feature.shape)}\n"
                    f"IR  {tuple(ir_feature.shape)}"
                )

            if (
                rgb_feature.shape[1]
                != expected_channel
            ):
                raise RuntimeError(
                    f"Stage {index}: expected "
                    f"{expected_channel} channels, got "
                    f"{rgb_feature.shape[1]}"
                )

            alpha = alphas[index].to(
                dtype=rgb_feature.dtype,
                device=rgb_feature.device,
            )

            auxiliary = adapter(
                ir_feature
            )

            output = (
                rgb_feature
                +
                alpha * auxiliary
            )

            fused.append(
                output
            )

        return tuple(
            fused
        )

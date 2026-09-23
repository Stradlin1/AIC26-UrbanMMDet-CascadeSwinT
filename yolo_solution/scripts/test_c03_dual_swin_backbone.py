#!/usr/bin/env python3

import sys
from copy import deepcopy
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope


# ============================================================
# Paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# ldds/yolo_solution/scripts/xxx.py -> ldds
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

from mmdet.registry import MODELS

from c03_rgbir_components import (
    DualSwinGatedBackbone,
)


C02_CONFIG = (
    LDDS_ROOT
    / "runs"
    / "c02_cascade_swin_t_rgb_1333x800"
    / "resolved_config.py"
)

TEST_SIZE = 128


# ============================================================
# Helpers
# ============================================================

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


def check_adapter_identity(adapter):
    weight = (
        adapter.weight
        .detach()
        .cpu()
    )

    channel = weight.shape[0]

    expected = torch.zeros_like(
        weight
    )

    index = torch.arange(
        channel
    )

    expected[
        index,
        index,
        0,
        0,
    ] = 1.0

    weight_diff = float(
        torch.max(
            torch.abs(
                weight - expected
            )
        ).item()
    )

    bias_diff = 0.0

    if adapter.bias is not None:
        bias_diff = float(
            torch.max(
                torch.abs(
                    adapter.bias
                    .detach()
                    .cpu()
                )
            ).item()
        )

    return weight_diff, bias_diff


# ============================================================
# Main
# ============================================================

def main():
    if not C02_CONFIG.exists():
        raise FileNotFoundError(
            f"C02 config not found:\n{C02_CONFIG}"
        )

    init_default_scope(
        "mmdet"
    )

    cfg = Config.fromfile(
        str(C02_CONFIG)
    )

    base_backbone = deepcopy(
        cfg.model.backbone
    )

    # Unit test does not load any pretrained checkpoint.
    base_backbone.init_cfg = None

    dual_cfg = dict(
        type="DualSwinGatedBackbone",
        rgb_backbone=deepcopy(
            base_backbone
        ),
        ir_backbone=deepcopy(
            base_backbone
        ),
        channels=(
            96,
            192,
            384,
            768,
        ),
    )

    backbone = MODELS.build(
        dual_cfg
    )

    backbone.init_weights()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backbone = backbone.to(
        device
    )

    print("=" * 72)
    print("AIC2026 C03 DUAL-SWIN BACKBONE UNIT TEST")
    print("=" * 72)

    print("C02 config :", C02_CONFIG)
    print("Device     :", device)

    # --------------------------------------------------------
    # 1. Gate zero initialization
    # --------------------------------------------------------

    gate_values = backbone.gate_values()

    print()
    print("Initial gates:")
    print(gate_values)

    if any(
        value != 0.0
        for value in gate_values
    ):
        raise RuntimeError(
            "Gate initialization is not zero."
        )

    print("Gate zero-init       PASS")

    # --------------------------------------------------------
    # 2. Adapter identity initialization
    # --------------------------------------------------------

    print()
    print("Adapter checks:")

    for index, adapter in enumerate(
        backbone.adapters
    ):
        weight_diff, bias_diff = (
            check_adapter_identity(
                adapter
            )
        )

        print(
            f"  stage {index}: "
            f"weight_diff={weight_diff:.9f}, "
            f"bias_diff={bias_diff:.9f}"
        )

        if weight_diff != 0.0:
            raise RuntimeError(
                f"Adapter {index} weight "
                "is not identity initialized."
            )

        if bias_diff != 0.0:
            raise RuntimeError(
                f"Adapter {index} bias "
                "is not zero initialized."
            )

    print("Adapter identity-init PASS")

    # --------------------------------------------------------
    # 3. Forward + gate=0 RGB parity
    # --------------------------------------------------------

    backbone.eval()

    x = torch.randn(
        1,
        6,
        TEST_SIZE,
        TEST_SIZE,
        device=device,
    )

    with torch.no_grad():
        rgb_reference = (
            backbone.rgb_backbone(
                x[:, 0:3]
            )
        )

        fused = backbone(
            x
        )

    expected_channels = (
        96,
        192,
        384,
        768,
    )

    print()
    print("Feature shapes:")

    max_parity_diff = 0.0

    for index, (
        rgb_feature,
        fused_feature,
        expected_channel,
    ) in enumerate(
        zip(
            rgb_reference,
            fused,
            expected_channels,
        )
    ):
        if (
            rgb_feature.shape
            != fused_feature.shape
        ):
            raise RuntimeError(
                f"Stage {index} shape mismatch."
            )

        if (
            fused_feature.shape[1]
            != expected_channel
        ):
            raise RuntimeError(
                f"Stage {index}: expected "
                f"{expected_channel} channels, got "
                f"{fused_feature.shape[1]}"
            )

        diff = float(
            torch.max(
                torch.abs(
                    rgb_feature
                    - fused_feature
                )
            ).item()
        )

        max_parity_diff = max(
            max_parity_diff,
            diff,
        )

        print(
            f"  stage {index}: "
            f"{tuple(fused_feature.shape)}, "
            f"parity_diff={diff:.9f}"
        )

    if max_parity_diff != 0.0:
        raise RuntimeError(
            "gate=0 fused features are not "
            "bit-identical to RGB features."
        )

    print()
    print("Gate=0 RGB parity    PASS")

    # --------------------------------------------------------
    # 4. First backward at gate=0
    #
    # Expected:
    # RGB grad      > 0
    # Gate grad     > 0
    # IR grad       = 0
    # Adapter grad  = 0
    # --------------------------------------------------------

    backbone.train()

    backbone.zero_grad(
        set_to_none=True
    )

    x1 = torch.randn(
        1,
        6,
        TEST_SIZE,
        TEST_SIZE,
        device=device,
        requires_grad=True,
    )

    outputs = backbone(
        x1
    )

    loss = sum(
        feature.square().mean()
        for feature in outputs
    )

    loss.backward()

    rgb_grad = grad_abs_sum(
        backbone.rgb_backbone
    )

    ir_grad = grad_abs_sum(
        backbone.ir_backbone
    )

    adapter_grad = grad_abs_sum(
        backbone.adapters
    )

    gate_grad = 0.0

    if backbone.gates.grad is not None:
        gate_grad = float(
            backbone.gates
            .grad
            .detach()
            .abs()
            .sum()
            .item()
        )

    print()
    print("Backward @ gate=0:")
    print(
        "  RGB grad     :",
        f"{rgb_grad:.9f}",
    )
    print(
        "  IR grad      :",
        f"{ir_grad:.9f}",
    )
    print(
        "  Adapter grad :",
        f"{adapter_grad:.9f}",
    )
    print(
        "  Gate grad    :",
        f"{gate_grad:.9f}",
    )

    if rgb_grad <= 0:
        raise RuntimeError(
            "RGB branch has no gradient."
        )

    if gate_grad <= 0:
        raise RuntimeError(
            "Gate parameters have no gradient."
        )

    if ir_grad != 0.0:
        raise RuntimeError(
            "IR gradient should be zero "
            "when alpha=0."
        )

    if adapter_grad != 0.0:
        raise RuntimeError(
            "Adapter gradient should be zero "
            "when alpha=0."
        )

    print("Backward gate=0      PASS")

    # --------------------------------------------------------
    # 5. Set gates non-zero
    # --------------------------------------------------------

    backbone.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():
        backbone.gates.fill_(
            0.01
        )

    x2 = torch.randn(
        1,
        6,
        TEST_SIZE,
        TEST_SIZE,
        device=device,
        requires_grad=True,
    )

    outputs2 = backbone(
        x2
    )

    loss2 = sum(
        feature.square().mean()
        for feature in outputs2
    )

    loss2.backward()

    ir_grad2 = grad_abs_sum(
        backbone.ir_backbone
    )

    adapter_grad2 = grad_abs_sum(
        backbone.adapters
    )

    print()
    print("Backward @ gate!=0:")
    print(
        "  IR grad      :",
        f"{ir_grad2:.9f}",
    )
    print(
        "  Adapter grad :",
        f"{adapter_grad2:.9f}",
    )

    if ir_grad2 <= 0:
        raise RuntimeError(
            "IR branch still has no gradient "
            "after enabling gates."
        )

    if adapter_grad2 <= 0:
        raise RuntimeError(
            "Adapters still have no gradient "
            "after enabling gates."
        )

    print("IR learning path     PASS")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print("Gate zero-init       PASS")
    print("Adapter identity     PASS")
    print("4-stage feature      PASS")
    print("Gate=0 RGB parity    PASS")
    print("Gate gradient        PASS")
    print("IR blocked @ gate=0  PASS")
    print("IR learns @ gate!=0  PASS")

    print()
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

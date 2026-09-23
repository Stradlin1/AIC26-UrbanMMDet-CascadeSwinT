#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
LDDS_ROOT = SCRIPT_PATH.parents[2]

SRC = (
    LDDS_ROOT
    / "pretrained"
    / "c03_rgbir_swin_t_coco_dual_backbone_fpn_rpn_fusion_init.pth"
)

DST = (
    LDDS_ROOT
    / "pretrained"
    / "c04_rgbd_swin_t_coco_dual_backbone_fpn_rpn_fusion_init.pth"
)


def extract_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError(
        f"Unsupported checkpoint type: {type(checkpoint)}"
    )


def max_abs(tensor):
    return float(
        tensor.detach().abs().max().item()
    )


def check_rgb_ir_exact(state_dict):

    rgb_prefix = "backbone.rgb_backbone."
    ir_prefix = "backbone.ir_backbone."

    rgb = {
        k[len(rgb_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(rgb_prefix)
    }

    ir = {
        k[len(ir_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(ir_prefix)
    }

    if not rgb:
        raise RuntimeError(
            "No RGB backbone parameters found."
        )

    if not ir:
        raise RuntimeError(
            "No IR backbone parameters found."
        )

    if set(rgb) != set(ir):
        missing_ir = sorted(
            set(rgb) - set(ir)
        )

        missing_rgb = sorted(
            set(ir) - set(rgb)
        )

        raise RuntimeError(
            "RGB/IR backbone key sets differ.\n"
            f"Missing IR : {missing_ir[:10]}\n"
            f"Missing RGB: {missing_rgb[:10]}"
        )

    max_diff = 0.0

    for suffix in sorted(rgb):
        a = rgb[suffix]
        b = ir[suffix]

        if a.shape != b.shape:
            raise RuntimeError(
                f"Shape mismatch at {suffix}: "
                f"{tuple(a.shape)} vs {tuple(b.shape)}"
            )

        if not torch.equal(a, b):
            diff = float(
                (a - b).abs().max().item()
            )

            max_diff = max(
                max_diff,
                diff,
            )

    if max_diff != 0.0:
        raise RuntimeError(
            "RGB and IR branches are not exact copies. "
            f"max_diff={max_diff}"
        )

    return len(rgb)


def check_gates(state_dict):

    gate_key = "backbone.gates"

    if gate_key not in state_dict:
        raise RuntimeError(
            f"Missing {gate_key}"
        )

    gates = state_dict[
        gate_key
    ].detach().cpu()

    print(
        "Source raw gates :",
        gates.tolist(),
    )

    if not torch.equal(
        gates,
        torch.zeros_like(gates),
    ):
        raise RuntimeError(
            "Source initialization gates are not zero."
        )


def check_adapters(state_dict):

    channels = (
        96,
        192,
        384,
        768,
    )

    for index, channel in enumerate(
        channels
    ):

        weight_key = (
            f"backbone.adapters."
            f"{index}.weight"
        )

        bias_key = (
            f"backbone.adapters."
            f"{index}.bias"
        )

        if weight_key not in state_dict:
            raise RuntimeError(
                f"Missing {weight_key}"
            )

        weight = (
            state_dict[weight_key]
            .detach()
            .cpu()
        )

        expected = torch.zeros_like(
            weight
        )

        idx = torch.arange(
            channel
        )

        expected[
            idx,
            idx,
            0,
            0,
        ] = 1.0

        diff = float(
            (weight - expected)
            .abs()
            .max()
            .item()
        )

        if diff != 0.0:
            raise RuntimeError(
                f"Adapter {index} is not identity. "
                f"diff={diff}"
            )

        if bias_key in state_dict:
            bias = (
                state_dict[bias_key]
                .detach()
                .cpu()
            )

            if max_abs(bias) != 0.0:
                raise RuntimeError(
                    f"Adapter {index} bias is not zero."
                )


def main():

    print("=" * 72)
    print("AIC2026 C04 FORMAL INITIALIZATION PREPARATION")
    print("=" * 72)

    print("Source :", SRC)
    print("Output :", DST)

    if not SRC.exists():
        raise FileNotFoundError(
            f"C03 initialization checkpoint not found:\n{SRC}"
        )

    checkpoint = torch.load(
        SRC,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print(
        "Top-level checkpoint keys:",
        (
            sorted(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else type(checkpoint)
        ),
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    print(
        "State dict parameters:",
        len(state_dict),
    )

    # --------------------------------------------------------
    # Critical source audit
    # --------------------------------------------------------

    n_backbone_keys = (
        check_rgb_ir_exact(
            state_dict
        )
    )

    print(
        "RGB/IR exact-copy keys:",
        n_backbone_keys,
    )

    print(
        "PASS: RGB and IR public initialization "
        "branches are bit-identical"
    )

    check_gates(
        state_dict
    )

    print(
        "PASS: source gates are exactly zero"
    )

    check_adapters(
        state_dict
    )

    print(
        "PASS: source adapters are exact identity"
    )

    # --------------------------------------------------------
    # Component inventory
    # --------------------------------------------------------

    groups = {
        "RGB backbone":
            "backbone.rgb_backbone.",

        "IR backbone":
            "backbone.ir_backbone.",

        "Adapters":
            "backbone.adapters.",

        "Gates":
            "backbone.gates",

        "FPN":
            "neck.",

        "RPN":
            "rpn_head.",

        "Cascade":
            "roi_head.",
    }

    print()
    print("Source parameter inventory:")

    for name, prefix in groups.items():
        count = sum(
            1
            for key in state_dict
            if key.startswith(prefix)
        )

        print(
            f"  {name:<14s}: {count}"
        )

    # --------------------------------------------------------
    # Rename IR branch -> Depth branch
    # --------------------------------------------------------

    new_state = OrderedDict()

    old_prefix = (
        "backbone.ir_backbone."
    )

    new_prefix = (
        "backbone.depth_backbone."
    )

    renamed = 0

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            old_prefix
        ):
            suffix = key[
                len(old_prefix):
            ]

            new_key = (
                new_prefix
                + suffix
            )

            renamed += 1
        else:
            new_key = key

        if new_key in new_state:
            raise RuntimeError(
                f"Duplicate output key: {new_key}"
            )

        new_state[
            new_key
        ] = value

    if renamed != n_backbone_keys:
        raise RuntimeError(
            "Unexpected number of renamed "
            f"IR keys: {renamed} "
            f"(expected {n_backbone_keys})"
        )

    if any(
        key.startswith(
            old_prefix
        )
        for key in new_state
    ):
        raise RuntimeError(
            "IR backbone keys remain "
            "after conversion."
        )

    # Verify RGB/Depth exact equality.
    rgb_prefix = (
        "backbone.rgb_backbone."
    )

    depth_prefix = (
        "backbone.depth_backbone."
    )

    rgb = {
        k[len(rgb_prefix):]: v
        for k, v in new_state.items()
        if k.startswith(rgb_prefix)
    }

    depth = {
        k[len(depth_prefix):]: v
        for k, v in new_state.items()
        if k.startswith(depth_prefix)
    }

    if set(rgb) != set(depth):
        raise RuntimeError(
            "RGB/Depth output key sets differ."
        )

    for suffix in rgb:
        if not torch.equal(
            rgb[suffix],
            depth[suffix],
        ):
            raise RuntimeError(
                "RGB/Depth output branches "
                f"differ at {suffix}"
            )

    print()
    print(
        "PASS: IR branch renamed to "
        "Depth branch"
    )

    print(
        "PASS: RGB and Depth initialization "
        "are bit-identical"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    DST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        output = dict(
            checkpoint
        )

        output[
            "state_dict"
        ] = new_state

        meta = dict(
            output.get(
                "meta",
                {},
            )
            or {}
        )

        meta.update(
            {
                "experiment":
                    "AIC2026 C04 RGB+Depth",

                "initialization":
                    "C03 formal public-COCO init mirrored IR->Depth",

                "source_checkpoint":
                    str(SRC),

                "not_from":
                    "C02/C03 trained best checkpoint",
            }
        )

        output[
            "meta"
        ] = meta

    else:
        output = {
            "state_dict":
                new_state,

            "meta": {
                "experiment":
                    "AIC2026 C04 RGB+Depth",

                "initialization":
                    "C03 formal public-COCO init mirrored IR->Depth",

                "source_checkpoint":
                    str(SRC),

                "not_from":
                    "C02/C03 trained best checkpoint",
            },
        }

    torch.save(
        output,
        DST,
    )

    # --------------------------------------------------------
    # Reload verification
    # --------------------------------------------------------

    reloaded = torch.load(
        DST,
        map_location="cpu",
        weights_only=False,
    )

    reloaded_state = extract_state_dict(
        reloaded
    )

    if list(
        reloaded_state.keys()
    ) != list(
        new_state.keys()
    ):
        raise RuntimeError(
            "Reloaded state-dict key ordering differs."
        )

    for key in new_state:
        if not torch.equal(
            new_state[key],
            reloaded_state[key],
        ):
            raise RuntimeError(
                f"Reload verification failed: {key}"
            )

    print()
    print(
        "PASS: saved checkpoint reload "
        "is bit-identical"
    )

    print()
    print("=" * 72)
    print("C04 FORMAL INIT READY")
    print("=" * 72)
    print("Output:", DST)
    print()
    print("This checkpoint is allowed for:")
    print("  C04 smoke training")
    print("  C04 formal training")
    print()
    print("It is NOT:")
    print("  C02 best")
    print("  C03 best")
    print("  parity-only checkpoint")
    print()
    print("STATUS = PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

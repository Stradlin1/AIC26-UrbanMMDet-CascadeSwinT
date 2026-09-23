# AIC2026
# Cascade R-CNN + Swin-T architecture smoke config
#
# 当前只验证模型组合能否正常构建和 CUDA 推理。
# 仍然使用 COCO 80 类。
# 后续再建立正式的 AIC2026 12 类配置。

_base_ = [
    '../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py',
]

# ------------------------------------------------------------
# Swin-T pretrained backbone
# ------------------------------------------------------------

pretrained = (
    'https://github.com/SwinTransformer/storage/releases/download/'
    'v1.0.0/swin_tiny_patch4_window7_224.pth'
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = dict(

    # 保留 CascadeRCNN / RPN / CascadeRoIHead，
    # 仅把 ResNet-50 backbone 替换成 Swin-T。
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
        ),
    ),

    # Swin-T 四级 feature channels
    neck=dict(
        in_channels=[96, 192, 384, 768],
    ),
)

# ------------------------------------------------------------
# Swin 推荐 AdamW
# 目前 smoke 不训练，但先把组合配置写正确。
# ------------------------------------------------------------

optim_wrapper = dict(
    type='OptimWrapper',

    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(
                decay_mult=0.0,
            ),
            'relative_position_bias_table': dict(
                decay_mult=0.0,
            ),
            'norm': dict(
                decay_mult=0.0,
            ),
        },
    ),

    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)

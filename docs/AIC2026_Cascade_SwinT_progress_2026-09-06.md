# AIC2026 多模态目标检测项目进展记录
## Cascade R-CNN + Swin-T 路线

**日期：2026-09-06**

---

# 1. 当前目标

在已经完成 YOLO11s RGB / IR / Depth / 多模态早期融合实验的基础上，新增一条与 YOLO 检测范式明显不同的路线：

**Cascade R-CNN + Swin Transformer Tiny（Swin-T）+ FPN**

当前先做 **RGB-only baseline**，用于判断：

1. 换成 Two-Stage + Proposal + Cascade Refinement 后，是否能超过现有 YOLO RGB baseline。
2. AIC2026 的瓶颈究竟主要来自检测器架构，还是来自数据规模、长尾、小目标和多模态融合。
3. 如果 RGB Cascade 路线有效，后续再加入 Depth / IR 做多模态特征融合。

---

# 2. 已有 YOLO 基线

| 实验 | 模型 | 模态 | Val mAP50-95 | 备注 |
|---|---|---|---:|---|
| Exp01 | YOLO11s | RGB | 约 0.3808 | 固定 val400 |
| Exp02 | YOLO11s | IR | 约 0.2093 | 单红外 |
| Exp03 | YOLO11s | Depth | 约 0.1771 | 单深度 |
| Exp04 | YOLO11s 5ch early fusion | RGB+IR+Depth | 约 0.379~0.381 | 公开榜约 45~46 |

已有多模态消融结论：

- RGB 是主语义模态。
- Depth 是明显有效的辅助模态。
- IR 是弱正增益模态。
- 直接做 5 通道 early fusion，提升有限。
- 因此继续只在 YOLO 第一层输入通道上微调，边际收益已经比较小。

---

# 3. 新路线选择

选择：

**Cascade R-CNN + Swin-T + FPN**

核心原因：

- 与 YOLO 的 One-Stage Dense Prediction 路线差异较大。
- Cascade R-CNN 通过多阶段 bbox refinement 强化高 IoU 定位。
- AIC2026 指标为 mAP50:95，高 IoU 定位质量很重要。
- Swin Transformer 与 YOLO CNN backbone 的建模方式不同。
- 后续适合扩展 RGB + Depth Feature Fusion。

---

# 4. 当前工程目录规范

当前工作根目录：

```text
/home/xhm/Desktop/ldds/
```

统一目录：

```text
ldds/
├── datasets/
│   └── AIC2026_Train_2000/
│       ├── visible/
│       ├── infrared/
│       ├── depth/
│       └── labels/
├── generated/
│   └── aic2026_coco/
│       ├── instances_train.json
│       └── instances_val.json
├── pretrained/
├── runs/
├── scripts/
│   └── 正式训练脚本
├── yolo_solution/
│   ├── scripts/
│   │   └── smoke / 数据检查 / 转换 / 划分 / 验证 / 一次性工具脚本
│   └── splits/
│       └── aic2026_group_stratified_v2/
├── mmcv/
└── mmdetection/
```

## 脚本放置规范

正式训练脚本统一放：

```text
/home/xhm/Desktop/ldds/scripts/
```

启动方式：

```bash
cd /home/xhm/Desktop/ldds
python scripts/train_xxxx.py
```

一次性 / 工具类脚本统一放：

```text
/home/xhm/Desktop/ldds/yolo_solution/scripts/
```

包括 smoke test、数据转换、数据检查、split、val、checkpoint 兼容性检查和临时分析工具。

---

# 5. MMDetection 环境

Conda 环境：

```text
mmdet_swin
```

关键版本：

```text
Python      3.10
PyTorch     2.8.0+cu128
CUDA        12.8
GPU         RTX 5060 Laptop GPU
MMCV        2.1.0
MMEngine    0.10.7
MMDetection 3.3.0
```

MMCV 为本机针对 CUDA 12.8 / RTX 5060 源码编译。

已验证：

```text
mmcv._ext PASS
mmcv.ops.nms PASS
mmcv.ops.roi_align PASS
CUDA forward PASS
```

后续不要随意执行：

```bash
pip install mmcv
mim install mmcv
```

避免覆盖已经编译好的 MMCV。

---

# 6. Swin / Cascade 架构验证

已经完成：

- Cascade R-CNN R50 官方 smoke：PASS
- Swin-T 官方 smoke：PASS
- 自定义 Cascade R-CNN + Swin-T + FPN：Config build PASS / CUDA forward PASS

---

# 7. AIC2026 数据状态

当前使用：

```text
/home/xhm/Desktop/ldds/datasets/AIC2026_Train_2000
```

包含：

```text
visible/
infrared/
depth/
labels/
```

当前 labels 为更新后的新标签，作为数据源真值。

---

# 8. 固定 train / val 划分

继续使用：

```text
aic2026_group_stratified_v2
```

路径：

```text
/home/xhm/Desktop/ldds/yolo_solution/splits/aic2026_group_stratified_v2/
```

数量：

```text
train = 1600
val   = 400
total = 2000
```

检查结果：

```text
Train unique stems : 1600
Val unique stems   : 400
Train/val overlap  : 0
Union              : 2000
Missing images     : 0
Missing labels     : 0
```

---

# 9. YOLO 标签转换为 COCO

转换脚本：

```text
yolo_solution/scripts/convert_aic2026_yolo_to_coco.py
```

输出：

```text
/home/xhm/Desktop/ldds/generated/aic2026_coco/
├── instances_train.json
└── instances_val.json
```

## 9.1 边界框越界审计

```text
Total images : 2000
Total boxes  : 15195
OOB boxes    : 412
OOB ratio    : 2.7114%
```

分布：

```text
<= 0.01 px      : 348
0.1 - 1 px      : 6
1 - 5 px        : 17
5 - 10 px       : 10
10 - 50 px      : 19
> 50 px         : 12
Maximum overflow: 120.999960 px
```

处理原则：

- 不修改原始 YOLO TXT。
- 大多数是小数舍入或边缘截断目标。
- COCO 转换时裁剪到图像可见区域。

## 9.2 转换结果

Train：

```text
Images          : 1600
Annotations     : 11863
Empty label file: 1
```

Val：

```text
Images          : 400
Annotations     : 3332
Empty label file: 0
```

总计：

```text
15195 annotations
```

与原始 YOLO 标签框总数完全一致，无 bbox 丢失。

---

# 10. 类别定义

类别顺序：

```text
0  person
1  boat
2  animal
3  seat
4  sign
5  bicycle
6  car
7  ball
8  light
9  garbage_can
10 uav
11 tricycle
```

COCO category_id 使用 1~12，MMDetection 内部仍映射为 0~11。

---

# 11. COCO API 最终验证

```text
Images      : train 1600 / val 400
Annotations : total 15195
Categories  : 12
Invalid bbox: 0
COCO validation: PASS
```

数据转换阶段已经封板。

---

# 12. 类别分布

## Train

```text
person       4345
boat          109
animal       2399
seat          532
sign          636
bicycle       532
car          1480
ball           74
light         1336
garbage_can   224
uav           176
tricycle       20
```

## Val

```text
person       1204
boat           26
animal        621
seat          109
sign          152
bicycle       134
car           469
ball           21
light         499
garbage_can    58
uav            32
tricycle        7
```

存在明显长尾。当前 baseline 阶段暂不加入 class weight、oversampling 或 repeat factor sampling，避免与架构因素混杂。

---

# 13. C01：真实训练 Smoke Test

脚本：

```text
/home/xhm/Desktop/ldds/yolo_solution/scripts/
train_c01_cascade_swin_t_rgb_smoke.py
```

配置：

```text
Train images : 64
Val images   : 32
Epoch        : 1
RGB only
Swin-T
Cascade R-CNN
FPN
AMP
batch=1
with_cp=True
```

第一次遇到 deterministic CuBLAS 报错，增加：

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

之后 smoke 完整通过：

```text
COCO dataset read      PASS
Swin-T pretrained load PASS
forward                PASS
backward               PASS
AMP                    PASS
checkpoint             PASS
COCO val               PASS
CUDA                    PASS
```

Smoke mAP 不作为正式实验结果。

---

# 14. COCO Detector 级预训练兼容性检查

检查脚本：

```text
/home/xhm/Desktop/ldds/yolo_solution/scripts/
check_cascade_swin_t_coco_pretrain_compat.py
```

使用官方 Mask R-CNN + Swin-T + FPN COCO 1x checkpoint。

结果：

```text
Backbone full compatible : True
FPN full compatible      : True
RPN full compatible      : True

Missing keys after partial load : 24
Unexpected keys after load      : 0
```

最终初始化策略：

```text
Swin-T backbone   ← COCO detector pretrained
FPN               ← COCO detector pretrained
RPN               ← COCO detector pretrained

Cascade stage 1   ← AIC2026 random init
Cascade stage 2   ← AIC2026 random init
Cascade stage 3   ← AIC2026 random init
```

24 个 missing keys 与三个 Cascade bbox head 的随机初始化一致。

---

# 15. C02：正式 RGB Cascade Swin-T Baseline

正式训练脚本：

```text
/home/xhm/Desktop/ldds/scripts/
train_c02_cascade_swin_t_rgb_1333x800.py
```

输出目录：

```text
/home/xhm/Desktop/ldds/runs/
c02_cascade_swin_t_rgb_1333x800/
```

模型：

```text
Cascade R-CNN
+
Swin Transformer Tiny
+
FPN
```

检测流程：

```text
RGB
 ↓
Swin-T Backbone
 ↓
FPN
 ↓
RPN
 ↓
RoIAlign
 ↓
Cascade Stage 1 (IoU 0.5)
 ↓
Cascade Stage 2 (IoU 0.6)
 ↓
Cascade Stage 3 (IoU 0.7)
 ↓
NMS
 ↓
max 100 detections
```

---

# 16. C02 正式训练参数

```text
Modality        : RGB
Train           : 1600
Val             : 400

Input scale     : (1333, 800)
keep_ratio      : True

Batch size      : 1
AMP             : True
with_cp         : True

Optimizer       : AdamW
LR              : 2.5e-5
Weight decay    : 0.05

Epochs          : 12
Warmup          : 500 iterations
LR milestones   : [8, 11]
gamma           : 0.1

Seed            : 2026
Deterministic   : True

score_thr       : 0.001
max_det         : 100
```

`(1333, 800), keep_ratio=True` 表示等比例缩放。例如 1920×1080 图像大约变为 1333×750，再做必要 padding。

---

# 17. C02 当前训练状态

目前至少训练到：

```text
Epoch 6
```

训练状态：

```text
无 OOM
无 NaN
无 loss 爆炸
AMP 正常
checkpoint 正常
val 正常
```

训练显存大约：

```text
1.6 ~ 1.9 GB
```

---

# 18. C02 Epoch 5

```text
mAP50-95 : 0.372
mAP50    : 0.722
mAP75    : 0.338

small AP : 0.168
medium AP: 0.338
large AP : 0.512
```

---

# 19. C02 Epoch 6

当前最佳：

```text
mAP50-95 : 0.377
mAP50    : 0.706
mAP75    : 0.351

small AP : 0.195
medium AP: 0.338
large AP : 0.525
```

best checkpoint：

```text
best_coco_bbox_mAP_epoch_6.pth
```

---

# 20. Epoch 5 → Epoch 6 变化

| Metric | Epoch 5 | Epoch 6 | Change |
|---|---:|---:|---:|
| mAP50-95 | 0.372 | 0.377 | +0.005 |
| mAP50 | 0.722 | 0.706 | -0.016 |
| mAP75 | 0.338 | 0.351 | +0.013 |
| small AP | 0.168 | 0.195 | +0.027 |
| medium AP | 0.338 | 0.338 | 0 |
| large AP | 0.512 | 0.525 | +0.013 |

重要现象：

```text
mAP50 下降
mAP75 上升
mAP50-95 上升
```

说明模型仍在改善更高 IoU 条件下的 bbox 定位质量，符合 Cascade R-CNN 的设计目标。

---

# 21. Epoch 6 各类别 AP

```text
person       0.402
boat         0.273
animal       0.418
seat         0.680
sign         0.312
bicycle      0.321
car          0.381
ball         0.463
light        0.277
garbage_can  0.350
uav          0.567
tricycle     0.073
```

观察：

- seat、uav 表现较好。
- ball 在 Epoch 6 提升明显。
- tricycle 波动很大，但 val 只有 7 个实例，不宜过度解读。
- small-object AP 明显低于 large-object AP，是后续重要优化方向。

---

# 22. 当前与 YOLO RGB Baseline 对比

已有 RGB YOLO11s：

```text
mAP50-95 ≈ 0.3808
```

C02 Epoch 6：

```text
mAP50-95 = 0.3770
```

差距约：

```text
0.0038
```

因此 Cascade R-CNN + Swin-T 在只训练到第 6 个 epoch 时，已经基本追平 YOLO11s RGB baseline。

当前 scheduler 的第一次 LR decay 在 epoch 8，尚未发生，所以应继续完整跑到 12 epoch。

---

# 23. Cascade R-CNN 与 YOLO11s 核心区别

## YOLO11s

One-Stage Dense Detector：

```text
Image
 ↓
Backbone
 ↓
Neck
 ↓
Dense Detection Head
 ↓
bbox + class + confidence
 ↓
NMS
```

特点：

- 一阶段
- 直接预测
- 推理快
- 密集检测
- 无 RPN
- 无 RoIAlign
- 无 Cascade 多阶段 bbox refinement

## Cascade R-CNN + Swin-T

Two-Stage Proposal-Based Detector：

```text
Image
 ↓
Swin-T
 ↓
FPN
 ↓
RPN
 ↓
Proposal
 ↓
RoIAlign
 ↓
Stage 1
 ↓
Stage 2
 ↓
Stage 3
 ↓
bbox + class
```

特点：

- 两阶段
- 先 proposal，再分类与回归
- 三次 bbox refinement
- IoU threshold 逐级提高：0.5 → 0.6 → 0.7
- 更重视高质量定位
- 推理速度比 YOLO 慢
- 更适合验证 mAP50:95 / 高 IoU 定位能力

---

# 24. 当前阶段结论

1. MMDetection / MMCV / Swin-T / Cascade R-CNN 工程链路已完全跑通。
2. YOLO → COCO 数据转换正确，15195 个 bbox 全部保留。
3. COCO detector checkpoint 能完整初始化 Swin-T + FPN + RPN。
4. Cascade 三阶段 12 类 head 可正常从随机初始化学习。
5. C02 在 Epoch 6 已达到 0.377 mAP50-95，接近 YOLO RGB baseline 0.381。
6. C02 尚未经历 Epoch 8 的第一次 LR decay。
7. 当前结果已经证明 Cascade + Swin-T 是有效替代检测路线，不只是“能跑”。

---

# 25. 当前最重要的下一步

第一优先级：**完成 C02 到 Epoch 12。**

不要中断，不修改参数。

重点记录：

```text
Epoch 7
Epoch 8
Epoch 9
Epoch 10
Epoch 11
Epoch 12
```

每个 epoch 保存：

```text
mAP50-95
mAP50
mAP75
small AP
medium AP
large AP
```

最终确定：

```text
best epoch
best mAP50-95
best checkpoint
```

---

# 26. C02 完成后的实验方向

如果最终明显超过 YOLO RGB，例如 mAP50-95 达到 0.40+，则优先继续：

```text
RGB + Depth feature fusion
```

如果最终仍约 0.38，则说明架构不是唯一瓶颈，下一步重点考虑：

- 小目标
- 长尾类别
- Depth 融合
- 数据增强
- proposal / anchor 尺度
- input resolution
- sampler
- class imbalance

---

# 27. 多模态后续策略

现有 YOLO 消融已经说明：

```text
RGB   : 主模态
Depth : 强辅助模态
IR    : 弱辅助模态
```

因此 Cascade 路线建议：

```text
C02 RGB baseline
 ↓
RGB + Depth
 ↓
RGB + Depth + IR
```

优先尝试：

```text
feature-level fusion
```

而不是继续使用：

```text
5-channel raw early fusion
```

---

# 28. 一句话总结

**AIC2026 新路线已经从“环境搭建与 smoke”正式进入“有效模型训练阶段”：Cascade R-CNN + Swin-T RGB baseline 在 Epoch 6 达到 0.377 mAP50-95，已基本追平此前 YOLO11s RGB baseline，当前应继续完整训练到 12 epoch，再决定是否进入 RGB+Depth 多模态特征融合。**

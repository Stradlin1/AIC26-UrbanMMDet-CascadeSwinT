# AIC2026 面向城市场景的视觉多模态目标检测
## 总体实验计划（2026-09-06）

---

# 0. 当前状态

项目已经完成两条主要路线的前期验证：

1. **YOLO11s 多模态 Early Fusion 路线**：已完成 RGB / IR / Depth 单模态、RGB+IR+Depth 五通道训练、消融、Rect 推理和测试集提交闭环。
2. **MMDetection Two-Stage 路线**：已完成 **Cascade R-CNN + Swin-T + FPN RGB baseline（C02）**，固定 val400 最佳 **mAP50-95 = 0.396**，超过原 RGB YOLO11s 的约 0.381，说明该路线值得继续做多模态融合。

当前下一阶段主线：

```text
C02 RGB
  ↓
C03 RGB + IR
  ↓
C04 RGB + Depth
  ↓
C05 RGB + IR + Depth
  ↓
最优结构 Longer Schedule
  ↓
Full-data 2000 Final Fit
  ↓
Phase-1 Test Inference / Submission
```

核心原则：

> **先分别验证 IR 和 Depth 在 Two-Stage + Swin-T 框架中的独立贡献，再做三模态；先验证“加什么”，再验证“训多久”。**

---

# 1. 赛题约束与评价目标

任务输入为三种空间对齐模态：

```text
RGB
Infrared / IR
Depth
```

检测 12 类目标：

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

比赛核心指标：

```text
mAP@50-95
```

每张图最多：

```text
100 个预测框
```

测试输出格式：

```text
class_id norm_center_x norm_center_y norm_w norm_h confidence
```

若某张测试图无检测结果，也必须输出同名空 TXT。

数据与算法约束：

- 仅使用官方训练数据。
- 测试集不得参与训练。
- 可使用公开 ImageNet / COCO / Objects365 等预训练权重。
- 不采用简单投票 / 平均形式的多模型集成作为主方法。
- 初赛 / 复赛提交预测结果；半决赛还需提交算法代码、模型、环境和技术报告。

---

# 2. 当前数据与固定划分

官方训练数据：

```text
2000 组三模态样本
```

固定开发划分：

```text
train = 1600
val   = 400
```

固定 split：

```text
yolo_solution/splits/aic2026_group_stratified_v2/
```

原则：

```text
所有开发实验必须使用同一个 1600 / 400 split
```

最终架构和超参数确定以后，再做：

```text
2000 / 0
full-data final fit
```

全量训练只用于最终提交，不用于继续调参。

---

# 3. MMDetection 数据转换状态

MMDetection 路线使用 COCO 格式。

当前已经完成：

```text
generated/aic2026_coco/
├── instances_train.json
└── instances_val.json
```

结果：

```text
train images      = 1600
train annotations = 11863

val images        = 400
val annotations   = 3332

total annotations = 15195
```

检查：

```text
COCO categories = 12
invalid bbox     = 0
bbox total       = 15195
```

原始 YOLO label 不修改；边缘越界 bbox 在 COCO 序列化时裁剪到图像可见区域。

---

# 4. 当前工程目录规范

当前工程根目录：

```text
/home/xhm/Desktop/ldds
```

当前规范：

```text
ldds/
├── datasets/
├── generated/
├── pretrained/
├── runs/
├── scripts/
│   └── 正式训练脚本
├── yolo_solution/
│   ├── scripts/
│   │   └── 数据检查 / 转换 / smoke / val / inference / 临时工具
│   └── splits/
├── mmcv/
└── mmdetection/
```

正式训练脚本：

```text
/home/xhm/Desktop/ldds/scripts/train_xxxx.py
```

启动：

```bash
cd /home/xhm/Desktop/ldds
python scripts/train_xxxx.py
```

工具脚本统一放：

```text
/home/xhm/Desktop/ldds/yolo_solution/scripts/
```

正式脚本继续遵守可迁移路径原则：根据脚本位置推导项目根目录，不在代码中硬编码某台机器路径。

---

# 5. 历史 YOLO 路线结论

## 5.1 RGB YOLO11s baseline

固定 val400：

```text
mAP50-95 ≈ 0.380843
```

Leaderboard：

```text
43.611
```

## 5.2 RGB + IR + Depth YOLO Early Fusion

输入：

```text
[R, G, B, IR, Depth]
```

正式结果：

```text
Local Rect val400 ≈ 0.380711
Leaderboard       = 45.614
```

说明多模态在真实测试集上有价值，但输入级 Early Fusion 已接近瓶颈。

## 5.3 YOLO 模态消融

```text
Full RGB+IR+Depth = 0.375390
No IR             = 0.372447
No Depth          = 0.354958
RGB only in 5ch   = 0.355652
```

在该 YOLO 模型中：

```text
RGB   = 主语义模态
Depth = 强辅助模态
IR    = 弱正辅助模态
```

说明辅助模态有价值，但简单输入拼接不是理想最终结构。

---

# 6. 其他已有实验对后续设计的启示

另一组已有多模态实验结果：

```text
R0 RGB                    0.5712
M1 RGB + IR               0.5786
M2 RGB + Depth            0.5773
M3 RGB + IR + Depth       0.5656
M4 RGB + IR + 多Depth特征 0.5694
```

该组结果同样提示：

```text
单独加入 IR 或 Depth 可以增益
但直接同时堆叠多个模态可能出现模态干扰
```

因此下一阶段主假设改为：

```text
RGB 保留稳定主干
IR / Depth 独立提取
中层 Feature Fusion
可学习 Gate 控制辅助模态贡献
```

---

# 7. C02：Cascade R-CNN + Swin-T RGB Baseline

## 7.1 模型

```text
RGB
 ↓
Swin Transformer Tiny
 ↓
FPN
 ↓
RPN
 ↓
RoIAlign
 ↓
Cascade Stage 1
 ↓
Cascade Stage 2
 ↓
Cascade Stage 3
 ↓
Detection
```

Cascade 三阶段 IoU：

```text
Stage 1 = 0.5
Stage 2 = 0.6
Stage 3 = 0.7
```

## 7.2 初始化

已验证 COCO detector checkpoint 可完整初始化：

```text
Swin-T backbone
FPN
RPN
```

Cascade 三个 12 类 bbox head 随机初始化。

兼容性：

```text
Backbone full compatible : True
FPN full compatible      : True
RPN full compatible      : True
Missing keys             : 24
Unexpected keys          : 0
```

## 7.3 正式训练配置

```text
train        = 1600
val          = 400
input        = 1333 x 800
keep_ratio   = True
batch        = 1
AMP          = True
with_cp      = True
optimizer    = AdamW
lr           = 2.5e-5
weight_decay = 0.05
epochs       = 12
warmup       = 500 iterations
milestones   = [8, 11]
seed         = 2026
deterministic= True
score_thr    = 0.001
NMS IoU      = 0.5
max_det      = 100
```

## 7.4 最终结果

最佳：

```text
Epoch 11
mAP50-95 = 0.396
mAP50    = 0.746
mAP75    = 0.345
AP small  = 0.233
AP medium = 0.338
AP large  = 0.547
```

Epoch 12：

```text
mAP50-95 = 0.395
```

最终 checkpoint：

```text
runs/c02_cascade_swin_t_rgb_1333x800/
best_coco_bbox_mAP_epoch_11.pth
```

阶段结论：

> **Cascade R-CNN + Swin-T RGB 已经超过 RGB YOLO11s baseline，因此正式进入 Cascade 多模态阶段。**

---

# 8. C03：RGB + IR Feature Fusion

C03 是当前下一实验。

目标只回答一个问题：

> **在 C02 的 Cascade + Swin-T 框架里，IR 独立特征是否能提升 RGB baseline 0.396？**

因此 C03 相对 C02 只改变：

```text
RGB
↓
RGB + IR
```

其余尽量保持不变。

## 8.1 推荐结构

```text
RGB ----------------→ RGB Swin-T --------┐
                                         ├→ Feature Fusion
IR -----------------→ IR Swin-T ---------┘
                                             ↓
                                            FPN
                                             ↓
                                            RPN
                                             ↓
                                      Cascade Head ×3
```

Swin-T 四层输出：

```text
C2 = 96 channels
C3 = 192 channels
C4 = 384 channels
C5 = 768 channels
```

逐层融合：

```text
F2 = RGB2 + alpha2 * Adapter(IR2)
F3 = RGB3 + alpha3 * Adapter(IR3)
F4 = RGB4 + alpha4 * Adapter(IR4)
F5 = RGB5 + alpha5 * Adapter(IR5)
```

其中：

```text
Adapter    = 1x1 Conv
alpha      = learnable parameter
alpha init = 0
```

初始化时：

```text
F_fused ≈ F_rgb
```

这样可以最大程度保护 C02 已验证的 RGB representation。

## 8.2 IR 输入

官方 IR 为 3-channel uint8，三个通道视觉特征高度一致，本质上是热辐射灰度信息的三通道堆叠。

C03-v1 不改变 Swin-T 3-channel patch embedding，直接使用统一的 3-channel IR 输入，从而保留标准 Swin-T 结构并复用预训练权重。

## 8.3 初始化

```text
RGB Swin-T
    ← COCO detector pretrained

IR Swin-T
    ← Swin-T pretrained

Fusion Adapter
    ← 新初始化

alpha2~alpha5
    ← 0

FPN
    ← COCO detector pretrained

RPN
    ← COCO detector pretrained

Cascade bbox heads
    ← random init / 12 classes
```

## 8.4 训练协议

为保证与 C02 公平：

```text
train        = 1600
val          = 400
input        = 1333 x 800
keep_ratio   = True
batch        = 1
AMP          = True
with_cp      = True
optimizer    = AdamW
lr           = 2.5e-5
epochs       = 12
milestones   = [8, 11]
seed         = 2026
score_thr    = 0.001
NMS IoU      = 0.5
max_det      = 100
```

不同时修改 epoch、resolution、augmentation、optimizer、scheduler。

---

# 9. C03 开发步骤

```text
Step 1  RGB / IR 对齐检查
Step 2  建立双模态 Dataset / Pipeline
Step 3  建立 Dual-Swin backbone
Step 4  实现逐层 gated residual fusion
Step 5  模型 Unit Test
Step 6  64 train / 32 val / 1 epoch smoke
Step 7  正式 1600 / 400 训练
Step 8  固定 val400 评估
Step 9  有明确增益再跑 Phase-1 test
```

Unit Test 必须检查：

```text
RGB input shape 正确
IR input shape 正确
RGB C2/C3/C4/C5 shape 正确
IR  C2/C3/C4/C5 shape 正确
Fusion shape 正确
FPN 输入兼容
forward PASS
loss PASS
backward PASS
RGB branch gradient != 0
IR branch gradient  != 0
adapter gradient    != 0
alpha gradient      != 0
```

---

# 10. C03 评价标准

固定基线：

```text
C02 RGB = 0.396
```

建议判定：

```text
C03 <= 0.396
    无明确增益

0.397 ~ 0.402
    弱正增益

0.403 ~ 0.410
    明确有效

> 0.410
    非常值得继续
```

除了总 mAP，还必须对比：

```text
mAP50
mAP75
AP small
AP medium
AP large
12 类 per-class AP
```

---

# 11. C04：RGB + Depth Feature Fusion

C03 完成后做 C04。

原则：

```text
结构与 C03 尽量一致
只把 IR branch 替换为 Depth branch
```

结构：

```text
RGB ----------------→ RGB Swin-T --------┐
                                         ├→ Gated Fusion → FPN → RPN → Cascade
Depth ---------------→ Depth Swin-T -----┘
```

这样：

```text
C02 vs C03 = IR contribution
C02 vs C04 = Depth contribution
```

形成真正的模态消融。

---

# 12. C05：RGB + IR + Depth

只有 C03 和 C04 完成后，才进入三模态。

推荐结构：

```text
RGB → RGB Swin-T -----------------------------┐
                                              │
IR → IR Swin-T → Gate_IR ---------------------┤
                                              ├→ Fused Features
Depth → Depth Swin-T → Gate_Depth ------------┤
                                              │
                                             FPN
                                              ↓
                                             RPN
                                              ↓
                                           Cascade
```

核心：

```text
F_fused =
F_rgb
+ alpha_ir * Adapter(F_ir)
+ alpha_depth * Adapter(F_depth)
```

原则：

```text
RGB 永远是主路径
IR / Depth 是 residual auxiliary paths
```

不再让三种模态在输入层直接 concat。

---

# 13. 复杂 Attention 的位置

C03 / C04 第一版不加入：

```text
Cross Attention
Transformer Fusion
Spatial Attention
Channel Attention
Quality-aware Gate
Modality Dropout
```

第一轮只验证：

```text
独立 backbone + gated residual feature fusion
```

如果第一版证明有效，再进入更复杂的融合优化。

---

# 14. C06：Quality-aware Fusion（条件实验）

只有 C05 或某个双模态模型证明 gated fusion 有价值后，再考虑 C06。

可加入：

```text
Depth valid ratio
IR quality score
Modality Dropout
sample-adaptive gate
```

目标：

```text
辅助模态质量差时自动降权
质量好时充分利用
```

---

# 15. Longer Schedule 实验

当前 C02 使用 12 epoch。

开发阶段先保持：

```text
C02 / C03 / C04 / C05 = 12 epoch
```

完成公平结构比较后，只对最优模型做：

```text
24 epoch
或
36 epoch
```

原则：

> **先解决“结构哪个好”，再解决“训练多久”。**

---

# 16. Full-data 2000 全量训练

开发实验阶段：

```text
1600 train
400 val
```

最终架构和训练参数确定以后：

```text
2000 train
0 val
```

进行 final fit。

相对 1600 train，训练数据增加：

```text
25%
```

但因为没有独立 val，full-data final fit 之后不再继续调参。

---

# 17. 最终测试集推理流程

```text
best checkpoint
↓
Phase-1 test 1000 组
↓
MMDetection inference
↓
pred_instances
↓
xyxy + label + score
↓
转 normalized xywh
↓
每图最多 100
↓
1000 个 TXT
↓
ZIP
↓
Leaderboard
```

强制检查：

```text
images = 1000
TXT    = 1000
class id        ∈ [0, 11]
confidence      ∈ [0, 1]
normalized bbox ∈ [0, 1]
boxes/image     <= 100
NaN / Inf       = 0
```

---

# 18. 当前 C02 Phase-1 推理基线

最佳 checkpoint：

```text
best_coco_bbox_mAP_epoch_11.pth
```

推理配置：

```text
1333 x 800
keep_ratio = True
score_thr  = 0.001
NMS IoU    = 0.5
max_det    = 100
```

Phase-1 测试集：

```text
datasets/AIC2026_PHASE_1_1000/
├── visible/
├── infrared/
└── depth/
```

后续 C03 / C04 / C05 必须使用与各自 val 对应的同一推理协议。

---

# 19. 最终实验表目标

| ID | 模态 | 模型 | 融合 | Epoch | Val mAP50-95 | LB |
|---|---|---|---|---:|---:|---:|
| C02 | RGB | Cascade + Swin-T | - | 12 | 0.396 | 待记录 |
| C03 | RGB+IR | Dual Swin-T | gated feature fusion | 12 | ? | ? |
| C04 | RGB+Depth | Dual Swin-T | gated feature fusion | 12 | ? | ? |
| C05 | RGB+IR+Depth | Multi-branch Swin-T | gated feature fusion | 12 | ? | ? |
| Cxx | 最优模态 | 最优结构 | longer schedule | 24/36 | ? | ? |
| Final | 最优模态 | 最优结构 | full 2000 | final | 无独立 val | ? |

---

# 20. 技术报告同步规划

官方技术方案大纲要求最终报告至少包含：

```text
1. 算法概述
2. 实现方案
3. 算法创新点
4. 问题与思考
5. 过程进度
6. 团队分工
7. 解题参考
```

因此从现在开始，每个实验都必须记录：

```text
实验目的
唯一主要变量
数据划分
模型结构
初始化策略
训练参数
best epoch
mAP50-95
mAP50
mAP75
AP small / medium / large
per-class AP
测试集 leaderboard
失败原因 / 成功原因
下一步决策
```

这样后续技术报告可直接从实验记录汇总。

---

# 21. 近期执行顺序

## P0 — C02 收尾

状态：

```text
DONE
```

记录：

```text
best epoch = 11
val mAP50-95 = 0.396
checkpoint = best_coco_bbox_mAP_epoch_11.pth
Phase-1 submission / leaderboard = 待记录
```

## P1 — C03 RGB+IR 数据与结构设计

状态：

```text
NEXT
```

完成：

```text
RGB / IR alignment
Dataset design
Dual-Swin input design
Feature Fusion module
Pretrain initialization design
```

## P2 — C03 Unit Test

```text
forward
backward
gradient
checkpoint
```

## P3 — C03 Smoke

```text
64 train
32 val
1 epoch
```

只验证工程闭环，不看性能。

## P4 — C03 Formal

```text
1600 / 400
12 epoch
```

与 C02 = 0.396 严格比较。

## P5 — C04 RGB+Depth

结构尽量复制 C03，只替换辅助模态。

## P6 — C05 RGB+IR+Depth

基于 C03 / C04 结果决定是否保留两个辅助模态，以及 gate 是否需要非对称设计。

## P7 — 最优结构 Longer Schedule

```text
24 / 36 epoch
```

## P8 — Full 2000 Final Fit

不再调参。

## P9 — Final Test + Report

最终输出：

```text
submission ZIP
模型文件
环境说明
推理脚本
训练脚本
技术报告
```

---

# 22. 当前不应该做的事情

暂时不要：

```text
同时改融合 + epoch + resolution
同时加入 IR + Depth + attention
修改固定 split
修改 COCO 标签转换
修改 C02 baseline
用 full 2000 继续调参
频繁提交训练中途 checkpoint
继续做输入级简单通道堆叠
```

所有新实验尽量做到：

```text
一次只验证一个核心假设
```

---

# 23. 当前核心科研问题

## Q1

```text
IR 在 Two-Stage + Swin-T 中是否有独立增益？
```

由 C03 回答。

## Q2

```text
Depth 在 Two-Stage + Swin-T 中是否有独立增益？
```

由 C04 回答。

## Q3

```text
IR + Depth 同时加入后，
feature-level gated fusion 能否避免 Early Fusion 的模态干扰？
```

由 C05 回答。

如果 C03 / C04 有增益而 C05 无增益：

```text
说明三模态融合机制仍存在干扰问题
```

如果 C05 明显最高：

```text
说明独立分支 + 中层 gated residual fusion 的方向成立
```

---

# 24. 最终决策树

```text
C02 RGB = 0.396
      │
      ├───────────────┐
      ↓               ↓
C03 RGB+IR        C04 RGB+Depth
      │               │
      └───────┬───────┘
              ↓
      分析两个辅助模态
              ↓
      C05 RGB+IR+Depth
              ↓
        是否明确提升？
          /         \
        否           是
        │            │
分析 gate /       保留结构
fusion 位置          │
                     ↓
             24/36 epoch
                     ↓
                 选 best
                     ↓
              Full 2000 train
                     ↓
              Phase-1 inference
                     ↓
                 Submission
```

---

# 25. 最终总体策略

当前项目不再追求：

```text
不断堆更多输入通道
```

而是围绕：

```text
稳定 RGB 主干
+
IR / Depth 独立表示
+
中层 Feature Fusion
+
可学习辅助模态 Gate
+
严格单变量消融
```

建立一条清晰、可复现、可直接写入技术报告的实验链。

当前立即执行的下一步：

```text
C03 RGB + IR
Dual Swin-T
+
Gated Residual Feature Fusion
```

正式训练前先完成：

```text
数据读取
结构设计
Unit Test
1-epoch Smoke
```

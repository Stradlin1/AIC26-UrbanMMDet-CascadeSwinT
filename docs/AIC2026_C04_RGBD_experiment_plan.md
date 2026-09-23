# AIC2026 C04 RGB+Depth 整体实验思路与具体路线

> 目标：在已经验证成功的 C02 RGB / C03 RGB+IR 路线上，严格控制变量地验证 **Depth 是否能够作为独立辅助模态提升 Cascade R-CNN + Swin-T 的目标检测性能**。  
> 核心原则：**C04 只把 C03 的辅助模态从 IR 替换为 Depth，其他结构、初始化、训练协议、验证协议尽量完全一致。**

---

## 0. 当前基线

当前 MMDetection 路线已经得到：

| 实验 | 模态 | mAP50-95 | mAP50 | mAP75 | APs | APm | APl |
|---|---|---:|---:|---:|---:|---:|---:|
| C02 | RGB | 0.396 | 0.746 | 0.345 | 0.233 | 0.338 | 0.547 |
| C03 | RGB + IR | **0.408** | 0.744 | **0.362** | 0.221 | 0.339 | **0.568** |

C03 相对 C02：

- mAP50-95：**+0.012**
- mAP75：**+0.017**
- APl：**+0.021**
- APs：**-0.012**

C03 已经证明以下设计有效：

1. 保持 RGB 主干稳定；
2. 辅助模态使用独立 Swin-T；
3. 四阶段特征分别融合；
4. 使用 identity-init 1×1 Adapter；
5. 使用 zero-init learnable gate；
6. 融合形式：

```text
F_i = RGB_i + tanh(g_i) * Adapter_i(AUX_i)
```

C03 最优 epoch 为 epoch 9，IR gate 大致呈现：

```text
C4 > C5 > C3 >>> C2
```

因此 C04 不重新发明结构，而是做 **严格镜像的 RGB+Depth 对照实验**。

---

# 1. C04 的核心研究问题

C04 需要回答：

> 在完全相同的 Cascade R-CNN + Swin-T 检测框架中，将 C03 的 IR 辅助分支替换为 Depth 后，Depth 是否能够比 RGB baseline 带来正增益？Depth 与 IR 哪一个辅助模态更有效？

最终重点比较：

```text
C02 RGB       = 0.396
C03 RGB + IR  = 0.408
C04 RGB + D   = ?
```

需要计算：

```text
ΔDepth_vs_RGB = C04 - C02
ΔDepth_vs_IR  = C04 - C03
```

---

# 2. C04 总体结构

C04 v1 结构固定为：

```text
RGB --------------------> RGB Swin-T ----------------------┐
                                                           │
                                                           ├─> F2
Depth -> Depth preprocess -> Depth Swin-T -> Adapter2 -> α2┤
                                                           │
RGB stage 3 -----------------------------------------------├─> F3
Depth stage 3 -> Adapter3 -> α3 ---------------------------┤
                                                           │
RGB stage 4 -----------------------------------------------├─> F4
Depth stage 4 -> Adapter4 -> α4 ---------------------------┤
                                                           │
RGB stage 5 -----------------------------------------------├─> F5
Depth stage 5 -> Adapter5 -> α5 ---------------------------┘

F2 ~ F5
   ↓
FPN
   ↓
RPN
   ↓
Cascade RoI Head
```

每一层融合：

```text
F_i = RGB_i + α_i * Adapter_i(Depth_i)

α_i = tanh(g_i)
```

初始化：

```text
Adapter_i = Identity
g_i       = 0
α_i       = 0
```

因此：

```text
初始状态：
F_i = RGB_i
```

这保证 C04 在初始时严格退化成 RGB baseline。

---

# 3. C04 必须保持不变的控制变量

除辅助模态外，以下内容全部保持与 C03 一致。

## 3.1 检测器

```text
Cascade R-CNN
Swin-T
FPN
RPN
3-stage Cascade RoI Head
IoU stages = 0.5 / 0.6 / 0.7
```

## 3.2 输入规格

```text
input scale = 1333 × 800
keep_ratio  = True
batch       = 1
AMP         = True
with_cp     = True
```

## 3.3 训练协议

```text
train = 1600
val   = 400

optimizer      = AdamW
lr             = 2.5e-5
weight_decay   = 0.05
warmup         = 500 iterations
epochs         = 12
milestones     = [8, 11]
gamma          = 0.1

seed           = 2026
deterministic  = True
```

## 3.4 推理协议

```text
score_thr = 0.001
NMS IoU   = 0.5
max_det   = 100
```

## 3.5 初始化协议

正式 C04：

```text
RGB Swin-T   <- 与 C02/C03 相同的 public COCO detector backbone
Depth Swin-T <- 同一个 public COCO detector backbone

FPN          <- public COCO detector
RPN          <- public COCO detector
Cascade Head <- random 12-class initialization

Adapter      <- identity
Gate         <- zero
```

禁止：

```text
C04 <- C02 best
C04 <- C03 best
```

这样才能保证 C02 / C03 / C04 是公平实验。

---

# 4. C04 最大的新问题：Depth 表示

C04 与 C03 最大区别不在网络，而在输入。

IR 本质上仍然是 8-bit 三通道图像。

Depth 是：

```text
单通道
uint16
实际距离
单位约为 mm
```

所以 C04 第一步不能直接开训，必须先冻结 Depth representation。

---

# 5. Depth Representation v1 设计原则

## 5.1 推荐第一版

C04 v1 推荐：

```text
raw uint16 depth
       ↓
无效深度处理
       ↓
固定全局距离范围 clip
       ↓
固定映射到 uint8
       ↓
near = bright
far  = dark
       ↓
复制三通道 [D, D, D]
       ↓
标准 Swin-T
```

最终 Depth 分支输入：

```text
H × W × 3
```

这使 C03 与 C04 网络结构完全一致。

---

## 5.2 禁止每张图单独 min-max

不要：

```python
depth8 = (depth - depth.min()) / (depth.max() - depth.min())
```

原因：

同一个真实距离在不同图片中可能映射成完全不同的灰度值，会破坏绝对距离语义。

C04 应采用：

```text
全数据集统一 D_near
全数据集统一 D_far
```

例如：

```text
D = clip(D, D_near, D_far)

Depth8 =
255 * (D_far - D) / (D_far - D_near)
```

得到：

```text
近 -> 亮
远 -> 暗
```

---

# 6. C04-01：Depth 数据审计

## 目标

先搞清楚训练集 2000 张 Depth 的真实数据特性。

需要检查：

```text
文件数量
RGB / Depth 文件名是否一一对应
RGB / Depth 尺寸是否一致
文件后缀
Depth dtype
全局 min / max
有效像素 min / max
0 值比例
无效值比例
P0.1
P1
P5
P50
P95
P99
P99.9
异常极值
```

建议脚本：

```text
yolo_solution/scripts/analyze_c04_depth_dataset.py
```

输出：

```text
runs/c04_depth_analysis/
├── depth_stats.json
└── depth_stats.txt
```

## 通过标准

最终必须冻结：

```text
invalid definition = ?
D_near             = ?
D_far              = ?
mapping             = fixed inverse linear
output dtype        = uint8
output channels     = 3
```

在这一步结束之前，不进入模型实验。

---

# 7. C04-02：Depth 表示实现

建议新增：

```text
yolo_solution/scripts/c04_rgbd_components.py
```

组件至少包含：

```text
LoadRGBDepthFromFile
RGBDepthDetDataPreprocessor
DualSwinGatedBackbone（可复用 C03 逻辑）
```

原则：

```text
visible/xxx.png
depth/xxx.png
```

读取后构造：

```text
[B, G, R, D, D, D]
```

这里的 6 channel 只是 **transport tensor**。

不是 Early Fusion。

其目的只是保证：

```text
Resize
RandomFlip
PackDetInputs
```

对 RGB 和 Depth 使用完全相同的几何变换。

---

# 8. C04-03：RGBD Pipeline Parity

建议脚本：

```text
yolo_solution/scripts/test_c04_rgbd_pipeline.py
```

检查：

```text
RGB / Depth 原图尺寸一致
6-channel Resize 一致
6-channel RandomFlip 一致
PackDetInputs 一致

RGB preprocessing 与 C02 一致
Depth preprocessing 可重复、无异常
Depth 映射值域正确
```

目标输出：

```text
PASS: raw RGB / Depth alignment
PASS: 6-channel Resize
PASS: 6-channel RandomFlip
PASS: 6-channel PackDetInputs
PASS: RGB preprocessing parity with C02
PASS: Depth preprocessing
STATUS = PASS
```

如果这一阶段不 PASS，不允许继续。

---

# 9. C04-04：Dual-Swin Backbone 单元测试

建议脚本：

```text
yolo_solution/scripts/test_c04_dual_swin_backbone.py
```

测试：

```text
Gate zero-init
Adapter identity-init

4-stage feature shape:
C2 = 96
C3 = 192
C4 = 384
C5 = 768

gate=0:
fused == RGB

第一次 backward:
RGB grad     > 0
Gate grad    > 0
Depth grad   = 0
Adapter grad = 0

人工设置 gate=0.01 后:
Depth grad   > 0
Adapter grad > 0
```

目标：

```text
STATUS = PASS
```

---

# 10. C04-05：C02 -> C04 Detector Parity

这一步非常重要。

仅用于 parity test：

```text
C02 best backbone
    ↓
复制给 C04 RGB Swin
复制给 C04 Depth Swin

C02 FPN
    ↓
C04 FPN

C02 RPN
    ↓
C04 RPN

C02 Cascade
    ↓
C04 Cascade

Adapter = identity
Gate    = 0
```

注意：

> 这个 checkpoint 只能用于 parity，绝不能作为正式 C04 初始化。

建议脚本：

```text
yolo_solution/scripts/test_c04_c02_detector_parity.py
```

要求：

```text
RGB preprocess max diff = 0

C2 diff = 0
C3 diff = 0
C4 diff = 0
C5 diff = 0

P2 diff = 0
P3 diff = 0
P4 diff = 0
P5 diff = 0
P6 diff = 0

bbox diff  = 0
score diff = 0
labels     = identical
```

目标：

```text
STATUS = PASS
```

---

# 11. C04-06：完整 val400 gate=0 parity

建议脚本：

```text
yolo_solution/scripts/val_c04_gate0_parity_val400.py
```

必须重新得到 C02 baseline：

```text
mAP50-95 = 0.396
mAP50    = 0.746
mAP75    = 0.345
APs      = 0.233
APm      = 0.338
APl      = 0.547
```

如果不是：

```text
STOP
```

优先排查：

```text
Depth loader
RGB preprocessing
dataset pipeline
checkpoint mapping
数据索引
```

不要直接训练。

---

# 12. C04-07：完整 Detector Backward Test

建议：

```text
yolo_solution/scripts/test_c04_full_detector_backward.py
```

真实训练样本下：

## gate = 0

要求：

```text
RGB backbone grad   > 0
Depth backbone grad = 0
Adapters grad       = 0
Gates grad          > 0

FPN grad            > 0
RPN grad            > 0
Cascade grad        > 0
```

## gate = 0.01

要求：

```text
Depth backbone grad > 0
Adapter grad        > 0
Gate grad           > 0
```

目标：

```text
STATUS = PASS
```

---

# 13. C04-08：Smoke Training

建议训练脚本：

```text
scripts/train_c04_rgbd_gated_smoke_1333x800.py
```

配置：

```text
train = 64
val   = 32

epochs = 1

1333 × 800
batch = 1
AMP   = True
with_cp = True
```

初始化仍然必须是：

```text
public COCO
```

不是：

```text
C02 best
C03 best
```

## Smoke 主要检查

```text
loss finite

gate moved > 0

RGB backbone updated
Depth backbone updated
Adapter updated
Gate updated

checkpoint strict reload PASS

显存正常
```

不要把 smoke 的 mAP 当作性能结果。

---

# 14. C04-09：正式训练

正式实验名：

```text
c04_cascade_swin_t_rgbd_gated_1333x800
```

正式训练脚本：

```text
scripts/train_c04_cascade_swin_t_rgbd_gated_1333x800.py
```

完整规格：

```text
train                = 1600
val                  = 400

input                = 1333 × 800
keep_ratio           = True

batch                = 1
AMP                  = True
with_cp              = True

optimizer            = AdamW
lr                   = 2.5e-5
weight_decay         = 0.05

warmup               = 500 iter
epochs               = 12
milestones           = [8, 11]
gamma                = 0.1

seed                 = 2026
deterministic        = True

score_thr            = 0.001
NMS IoU              = 0.5
max_det              = 100
```

C04 v1 不做：

```text
Depth branch 特殊 LR
Adapter 特殊 LR
Gate 特殊 LR
冻结 RGB backbone
特殊 attention
三模态融合
```

保持实验干净。

---

# 15. C04 Gate 监控

继续记录：

```text
runs/c04_cascade_swin_t_rgbd_gated_1333x800/
└── gate_history.jsonl
```

每 epoch：

```text
epoch
raw_gates
alpha_gates
peak memory
```

重点观察：

```text
α2
α3
α4
α5
```

然后和 C03 IR gate 对比。

C03 best epoch 9：

```text
IR:
C2 ≈ 0.0013
C3 ≈ 0.0120
C4 ≈ 0.0321
C5 ≈ 0.0169
```

C04 结束后可以研究：

```text
Depth 更偏浅层还是中高层？
Depth gate 是否比 IR gate 更大？
Depth gate 是否持续增长？
```

注意：

> Gate 大小表示融合门控强度，不等于单独的“真实性能贡献”。

---

# 16. C04-10：best checkpoint 独立 val400

训练完成后：

```text
找到 best_coco_bbox_mAP_epoch_X.pth
```

不要只看训练日志。

独立验证：

```text
yolo_solution/scripts/val_c04_best_val400.py
```

最终形成：

| Metric | C02 RGB | C03 RGB+IR | C04 RGB+Depth |
|---|---:|---:|---:|
| mAP50-95 | 0.396 | 0.408 | ? |
| mAP50 | 0.746 | 0.744 | ? |
| mAP75 | 0.345 | 0.362 | ? |
| APs | 0.233 | 0.221 | ? |
| APm | 0.338 | 0.339 | ? |
| APl | 0.547 | 0.568 | ? |

---

# 17. C04 成功判据

定义：

```text
ΔDepth = C04_mAP - 0.396
```

判断：

| ΔmAP50-95 | 结论 |
|---:|---|
| ≥ +0.005 | Depth 明确有效 |
| +0.002 ~ +0.005 | 弱正增益，可考虑第二 seed |
| -0.002 ~ +0.002 | 基本打平 |
| -0.005 ~ -0.002 | 弱负增益 |
| ≤ -0.005 | 当前 Depth 方案明显失败 |

再比较：

```text
ΔDepth_vs_IR = C04 - 0.408
```

用于判断：

```text
IR 和 Depth 谁是更强辅助模态
```

---

# 18. 对结果的进一步分析

C04 结束后不能只看总 mAP。

需要重点分析：

## 18.1 IoU 维度

```text
mAP50
mAP75
```

如果：

```text
mAP50 变化小
mAP75 明显提高
```

说明 Depth 更可能帮助定位精度。

---

## 18.2 尺度维度

```text
APs
APm
APl
```

重点和 C03 对比。

C03：

```text
APs -0.012
APm +0.001
APl +0.021
```

观察 Depth 是否：

```text
继续提升 APl
改善 APs
或出现不同尺度偏好
```

---

## 18.3 类别维度

比较 12 类 precision / AP。

观察：

```text
vehicle 类
person
light
sign
seat
tricycle
boat
```

哪些类别真正从 Depth 受益。

---

# 19. 如果 C04 成功，C05 怎么设计

如果：

```text
C03 > C02
C04 > C02
```

则自然进入 C05：

```text
RGB + IR + Depth
```

推荐形式：

```text
F_i =
RGB_i
+ α_IR_i * Adapter_IR_i(IR_i)
+ α_D_i  * Adapter_D_i(Depth_i)
```

即：

```text
                IR Swin -> Adapter -> α_IR --┐
                                               │
RGB Swin --------------------------------------┼-> fused
                                               │
Depth Swin -> Adapter -> α_D -----------------┘
```

不要优先做：

```text
RGB/IR/Depth naive concat
```

因为之前实验已经提示直接堆叠多模态可能产生模态干扰。

---

# 20. 如果 C04 失败，下一步怎么判断

如果 C04 明显低于 C02：

```text
C04 <= 0.391
```

不要立即得出：

```text
Depth 没用
```

首先检查：

```text
Depth representation 是否合理
```

优先做一次表示消融：

```text
D1 = fixed inverse-linear depth
D2 = log-depth
D3 = inverse-depth
```

此时只改变 Depth representation，不改变检测器。

如果多个合理 Depth representation 都没有增益，才更有依据判断当前架构下 Depth 无明显帮助。

---

# 21. 推荐脚本清单

整个 C04 建议最终形成：

```text
yolo_solution/scripts/
├── analyze_c04_depth_dataset.py
├── c04_rgbd_components.py
├── test_c04_rgbd_pipeline.py
├── test_c04_dual_swin_backbone.py
├── test_c04_c02_detector_parity.py
├── val_c04_gate0_parity_val400.py
├── test_c04_full_detector_backward.py
├── val_c04_best_val400.py
└── infer_c04_rgbd_gated_phase1_submit.py

scripts/
├── train_c04_rgbd_gated_smoke_1333x800.py
└── train_c04_cascade_swin_t_rgbd_gated_1333x800.py
```

所有新脚本继续遵守项目规范：

```text
1. 终端 cat 创建；
2. Python 训练脚本启动；
3. 路径基于 __file__ / 项目根目录推导；
4. 不在 Python 脚本里硬编码 /home/xhm/...；
5. 每次正式训练独立 train_xxxx.py；
6. 训练后先独立 val，再 test 推理提交。
```

---

# 22. 推荐执行顺序

完整路线固定为：

```text
C04-01 Depth 数据审计
        ↓
冻结 Depth representation
        ↓
C04-02 RGBD components
        ↓
C04-03 Pipeline parity
        ↓
C04-04 Dual-Swin backbone test
        ↓
C04-05 C02 -> C04 full detector parity
        ↓
C04-06 gate=0 val400 parity
        ↓
C04-07 full detector backward test
        ↓
C04-08 64/32 × 1 epoch smoke
        ↓
C04-09 1600/400 × 12 epoch formal
        ↓
C04-10 independent best val400
        ↓
C02 / C03 / C04 总对比
        ↓
决定 C05 RGB+IR+Depth
```

---

# 23. Stop / Go 条件

## C04-01

```text
Depth 数据定义不清楚
=> STOP
```

## C04-03

```text
Pipeline alignment != PASS
=> STOP
```

## C04-04

```text
gate=0 RGB parity != PASS
=> STOP
```

## C04-05

```text
完整 detector bbox/score parity != PASS
=> STOP
```

## C04-06

```text
val400 gate=0 != C02 0.396
=> STOP
```

## C04-07

```text
Depth 学习路径没有梯度
=> STOP
```

## C04-08

```text
1333×800 smoke OOM / loss abnormal / gate 不动
=> STOP 并排查
```

只有全部通过以后：

```text
GO -> C04 正式 12 epoch
```

---

# 24. 当前最优执行建议

现在不需要先写正式训练脚本。

下一步直接开始：

```text
C04-01：Depth 数据审计
```

首先确定：

```text
Depth dtype
无效值
全局距离分布
D_near
D_far
固定映射策略
```

只要 Depth representation 冻结，后面的 C04 pipeline / parity / backward / smoke / formal 可以基本沿用 C03 已经验证成功的工程框架快速推进。

---

# 25. C04 最终要回答的三个问题

实验完成后，应能够清楚回答：

### Q1

```text
Depth 是否比 RGB baseline 更好？
```

### Q2

```text
Depth 和 IR 谁是更强的辅助模态？
```

### Q3

```text
IR / Depth 分别在哪些 feature stage、IoU 精度、目标尺度和类别上产生作用？
```

回答完这三个问题后，再设计 C05 三模态才有充分依据。

---

## 当前路线总结

```text
C02 RGB
0.396
   │
   ├──────────────────────────────┐
   │                              │
   ▼                              ▼
C03 RGB+IR                     C04 RGB+Depth
0.408                          ?
+0.012                         ?
   │                              │
   └──────────────┬───────────────┘
                  ▼
            分析模态贡献
                  │
                  ▼
        决定 C05 RGB+IR+Depth
```

当前下一步：

```text
C04-01 Depth 数据审计
```

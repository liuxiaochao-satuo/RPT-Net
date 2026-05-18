# RPT-Net 实验规划文档

## 1. 项目背景

本项目面向 **基于骨架的人体动作识别（Skeleton-based Action Recognition）** 任务，计划在 **NTU RGB+D 60** 和 **NTU RGB+D 120** 数据集上验证所提出方法的有效性。

现有骨架动作识别方法主要包括以下几类：

1. **基于 GCN 的方法**  
   典型方法如 ST-GCN、2s-AGCN、MS-G3D 等。这类方法将人体关节点建模为图结构，通过空间图卷积和时间卷积学习动作特征。

2. **基于 Transformer 的方法**  
   典型方法如 ST-TR、DSTA-Net、STST、IIP-Transformer 等。这类方法利用自注意力机制建模关节之间或身体部位之间的长程依赖关系。

3. **基于热图表示的方法**  
   典型思想如 PoseConv3D，将骨架坐标转换为热图或 3D heatmap volume，再利用 CNN 提取时空特征。相比直接使用坐标，热图表示能够保留一定空间响应信息，并对姿态估计噪声具有更强鲁棒性。

本项目提出的 **RPT-Net** 计划以“关节坐标 → 热图 → CNN → 部位特征 → 注意力计算”为基础流程，在此基础上重点引入两个核心模块：

1. **Uncertainty Encoding，不确定性编码模块**  
   用于让部位级 Token 显式感知姿态响应质量。

2. **Reliability Attention，可靠性注意力模块**  
   用于利用可靠性信息调节注意力分配，使模型更加关注动作关键阶段中可靠的身体部位。

整体目标不是构建一个特别庞大的实验系统，而是面向三区期刊论文，完成一套实验量适中、逻辑完整、能够支撑方法有效性的验证方案。

---

## 2. 实验目的

本实验规划主要服务于以下几个目的：

### 2.1 与先进方法进行性能对比

在 NTU-60 和 NTU-120 的标准划分上，将 RPT-Net 与主流先进方法进行对比，证明所提出方法具有竞争力。

建议对比的数据集划分包括：

| 数据集 | 评价协议 |
|---|---|
| NTU RGB+D 60 | Cross-Subject, X-Sub |
| NTU RGB+D 60 | Cross-View, X-View |
| NTU RGB+D 120 | Cross-Subject, X-Sub |
| NTU RGB+D 120 | Cross-Setup, X-Set |

建议对比方法包括：

| 类型 | 方法 |
|---|---|
| GCN 方法 | ST-GCN |
| GCN 方法 | 2s-AGCN |
| GCN 方法 | MS-G3D |
| Transformer 方法 | ST-TR |
| Transformer 方法 | IIP-Transformer / STST |
| 本文方法 | RPT-Net |

主实验的目标是证明：

```text
RPT-Net 在标准数据集上具有较好的识别性能；
在保证实验量适中的情况下，能够与典型先进方法形成有效对比。
```

---

### 2.2 验证核心模块的有效性

本项目重点验证两个模块：

```text
1. 不确定性编码模块是否有效？
2. 可靠性注意力模块是否有效？
3. 两个模块是否具有互补性？
```

其中：

- 不确定性编码主要解决 **Token 表征质量问题**；
- 可靠性注意力主要解决 **Token 交互过程中的注意力分配问题**。

换言之：

```text
Uncertainty Encoding:
    让每个部位 Token 知道自身姿态响应是否可靠。

Reliability Attention:
    在注意力计算过程中，利用可靠性信息抑制低质量 Token 的干扰。
```

---

### 2.3 验证模型在低质量骨架输入下的鲁棒性

由于骨架数据可能存在以下问题：

```text
1. 关节点缺失
2. 关节点抖动
3. 姿态估计置信度低
4. 热图响应扩散
5. 连续帧之间姿态不稳定
```

因此，需要通过小规模鲁棒性实验说明：

```text
RPT-Net 在存在骨架噪声时，性能下降更慢；
不确定性编码和可靠性注意力可以提高模型对低质量输入的适应能力。
```

---

### 2.4 展示模型的可解释性

可靠性注意力不仅用于提高准确率，也可以用于分析模型关注的动作关键帧和关键身体部位。

可解释性实验目标是展示：

```text
模型是否更加关注动作相关的部位？
模型是否减少了对低质量部位的错误关注？
可靠性注意力是否能产生更合理的时空部位关注分布？
```

---

## 3. 整体实验思路

本项目的基础模型流程设定为：

```text
Joint Coordinates
    ↓
Heatmap Generation
    ↓
CNN Feature Extraction
    ↓
Part Feature / Part Token
    ↓
Standard Attention
    ↓
Classifier
```

完整 RPT-Net 流程为：

```text
Joint Coordinates
    ↓
Heatmap Generation
    ↓
CNN Feature Extraction
    ↓
Part Feature / Part Token
    ↓
Uncertainty Encoding
    ↓
Reliability Attention
    ↓
Classifier
```

其中 baseline 已经包含：

```text
1. 坐标到热图
2. CNN 特征提取
3. 部位级特征生成
4. 标准注意力计算
```

所以消融实验不再重点讨论“热图是否有效”或“部位特征是否有效”，而是集中证明：

```text
1. 在 heatmap-part-attention baseline 上加入不确定性编码是否有效；
2. 在 heatmap-part-attention baseline 上加入可靠性注意力是否有效；
3. 两者结合是否进一步提升性能；
4. 在低质量骨架输入下，完整模型是否更稳健；
5. 可靠性注意力是否能带来更合理的注意力分布。
```

---

## 4. 数据集规划

### 4.1 使用数据集

本项目使用：

```text
NTU RGB+D 60
NTU RGB+D 120
```
数据集路径：/data/lxc/datasets/ntu/nturgbd_skeletons_s001_to_s017
        /data/lxc/datasets/ntu/nturgbd_skeletons_s018_to_s032
### 4.2 数据格式

建议将原始 skeleton 数据预处理为统一格式：

```python
data.shape = [N, C, T, V, M]
```

其中：

| 符号 | 含义 |
|---|---|
| N | 样本数量 |
| C | 坐标通道数，例如 x, y, z 或 x, y, confidence |
| T | 时间帧数 |
| V | 关节点数量，NTU 通常为 25 |
| M | 人数，NTU 通常最多 2 人 |

单个 batch 输入格式建议为：

```python
x.shape = [B, C, T, V, M]
label.shape = [B]
```

### 4.3 数据预处理内容

预处理脚本建议完成以下功能：

```text
1. 读取 NTU 原始 skeleton 文件；
2. 过滤官方错误样本；
3. 提取每个样本的关节坐标；
4. 处理单人/双人 skeleton；
5. 对时间长度进行采样、补齐或裁剪；
6. 进行坐标归一化；
7. 按 X-Sub / X-View / X-Set 生成训练集和测试集；
8. 保存为 npy、pkl 或 pt 格式。
```

### 4.4 推荐目录结构

```text
data/
└── ntu/
    ├── raw/
    │   ├── ntu60_skeletons/
    │   └── ntu120_skeletons/
    │
    ├── processed/
    │   ├── NTU60_XSub/
    │   ├── NTU60_XView/
    │   ├── NTU120_XSub/
    │   └── NTU120_XSet/
    │
    └── split/
        ├── ntu60_xsub_train.txt
        ├── ntu60_xsub_val.txt
        ├── ntu60_xview_train.txt
        ├── ntu60_xview_val.txt
        ├── ntu120_xsub_train.txt
        ├── ntu120_xsub_val.txt
        ├── ntu120_xset_train.txt
        └── ntu120_xset_val.txt
```

---

## 5. 模型设计规划

### 5.1 Baseline 设计

Baseline 定义为：

```text
Heatmap-Part Attention Baseline
```

其流程为：

```text
关节坐标
    ↓
生成关节热图
    ↓
CNN 提取局部空间响应特征
    ↓
根据人体部位划分聚合为部位特征
    ↓
形成 Part Token
    ↓
标准空间-时间注意力
    ↓
分类器输出动作类别
```

可以命名为：

```text
HP-Att
Heatmap-Part Attention Baseline
```

### 5.2 完整模型 RPT-Net

完整模型在 HP-Att 基础上加入：

```text
1. Uncertainty Encoding
2. Reliability Attention
```

完整流程为：

```text
Joint Coordinates
    ↓
Heatmap Generation
    ↓
CNN Feature Extraction
    ↓
Part Token Generation
    ↓
Uncertainty Encoding
    ↓
Reliability-guided Spatial-Temporal Attention
    ↓
Global Pooling / CLS Token
    ↓
Classifier
```

---

## 6. 核心模块说明

### 6.1 Heatmap Generation

输入：

```python
skeleton: [B, C, T, V, M]
```

输出：

```python
heatmap: [B, V, T, H, W]
```

功能：

```text
1. 将关节坐标转换为二维高斯热图；
2. 每个关节点对应一个 heatmap channel；
3. 沿时间维度堆叠得到时序热图表示；
4. 为后续 CNN 提供空间响应信息。
```

### 6.2 CNN Feature Extraction

输入：

```python
heatmap: [B, V, T, H, W]
```

输出：

```python
joint_features 或 part_features
```

功能：

```text
1. 从热图中提取局部空间响应特征；
2. 保留关节附近的局部空间结构；
3. 为后续部位级 Token 提供基础特征。
```

### 6.3 Part Token Generation

建议将 25 个关节点划分为若干身体部位，例如：

```text
1. torso
2. left arm
3. right arm
4. left leg
5. right leg
```

输入：

```python
joint_features: [B, T, V, D]
```

输出：

```python
part_tokens: [B, T, P, D]
```

其中：

| 符号 | 含义 |
|---|---|
| B | batch size |
| T | 时间帧数 |
| P | 身体部位数 |
| D | token 维度 |

功能：

```text
1. 将关节点级特征聚合为部位级特征；
2. 降低单个关节点噪声对模型的影响；
3. 提高局部人体结构表达能力；
4. 为不确定性编码提供部位级输入。
```

---

### 6.4 Uncertainty Encoding

不确定性编码模块用于构造不确定性感知 Token。

建议包含三个指标：

```text
1. Confidence
2. Dispersion
3. Temporal Stability
```

#### 6.4.1 Confidence

含义：

```text
表示当前关节或部位热图响应强度。
```

可用于描述：

```text
该部位是否被姿态估计器稳定检测到；
该部位当前响应是否明显。
```

#### 6.4.2 Dispersion

含义：

```text
表示热图响应的空间扩散程度。
```

可用于描述：

```text
热图响应是集中还是分散；
响应越分散，说明位置不确定性越强。
```

#### 6.4.3 Temporal Stability

含义：

```text
表示部位响应在连续帧之间是否稳定。
```

可用于描述：

```text
该部位是否出现明显抖动；
该部位在时间维度上是否具有连续性。
```

#### 6.4.4 模块输入输出

输入：

```python
part_tokens: [B, T, P, D]
uncertainty: [B, T, P, 3]
```

输出：

```python
uncertainty_tokens: [B, T, P, D]
```

作用：

```text
1. 让每个部位 Token 显式感知自身姿态质量；
2. 使 Token 不仅包含动作特征，也包含可靠性描述；
3. 为后续可靠性注意力提供依据。
```

---

### 6.5 Reliability Attention

可靠性注意力模块用于利用可靠性信息调节注意力计算。

普通注意力只根据特征相似性计算：

```text
Attention = Softmax(QK^T / sqrt(d)) V
```

可靠性注意力可以引入 reliability bias 或 reliability weight：

```text
Attention = Softmax(QK^T / sqrt(d) + R) V
```

或者：

```text
Attention = Softmax(QK^T / sqrt(d)) · reliability_weight · V
```

其中 R 或 reliability_weight 由不确定性信息计算得到。

功能：

```text
1. 引导模型关注动作相关且可靠的部位；
2. 抑制低质量 Token 对注意力交互的干扰；
3. 提升模型对姿态噪声、关节点缺失和时序抖动的鲁棒性；
4. 提高模型的可解释性。
```

---

## 7. 实验设计

### 7.1 主实验：与先进方法对比

#### 7.1.1 实验目的

证明 RPT-Net 在标准骨架动作识别数据集上具有竞争力。

#### 7.1.2 实验数据集

```text
NTU60 X-Sub
NTU60 X-View
NTU120 X-Sub
NTU120 X-Set
```

#### 7.1.3 表格设计

| Method | NTU60 X-Sub | NTU60 X-View | NTU120 X-Sub | NTU120 X-Set |
|---|---:|---:|---:|---:|
| ST-GCN | - | - | - | - |
| 2s-AGCN | - | - | - | - |
| MS-G3D | - | - | - | - |
| ST-TR | - | - | - | - |
| IIP-Transformer / STST | - | - | - | - |
| RPT-Net | - | - | - | - |

#### 7.1.4 可选补充指标

如果实现方便，可以额外报告：

```text
1. Params
2. FLOPs
3. Inference Time
```

但对于三区期刊，准确率对比是核心。

---

### 7.2 消融实验一：核心模块有效性

#### 7.2.1 实验目的

验证：

```text
1. 不确定性编码模块是否有效；
2. 可靠性注意力模块是否有效；
3. 两者结合是否具有互补效果。
```

#### 7.2.2 实验设置

建议只在：

```text
NTU60 X-Sub
```

上进行。

#### 7.2.3 表格设计

| Method | Uncertainty Encoding | Reliability Attention | Accuracy |
|---|---:|---:|---:|
| HP-Att Baseline | ✗ | ✗ | - |
| HP-Att + UE | ✓ | ✗ | - |
| HP-Att + RA | ✗ | ✓ | - |
| RPT-Net | ✓ | ✓ | - |

#### 7.2.4 预期结论

```text
1. 加入 UE 后性能提升，说明不确定性信息增强了 Token 表征；
2. 加入 RA 后性能提升，说明可靠性约束改善了注意力分配；
3. UE 和 RA 同时使用时性能最好，说明二者具有互补性。
```

---

### 7.3 消融实验二：不确定性指标组成分析

#### 7.3.1 实验目的

验证不确定性编码中三个指标的作用：

```text
1. Confidence
2. Dispersion
3. Temporal Stability
```

#### 7.3.2 实验设置

建议只在：

```text
NTU60 X-Sub
```

上进行。

#### 7.3.3 表格设计

| Method | Confidence | Dispersion | Temporal Stability | Accuracy |
|---|---:|---:|---:|---:|
| w/o UE | ✗ | ✗ | ✗ | - |
| C only | ✓ | ✗ | ✗ | - |
| C + D | ✓ | ✓ | ✗ | - |
| C + D + S | ✓ | ✓ | ✓ | - |

#### 7.3.4 预期结论

```text
1. Confidence 描述热图响应强度；
2. Dispersion 补充空间分布不确定性；
3. Temporal Stability 补充时间连续性信息；
4. 三者结合时可以更完整地描述部位级姿态质量。
```

---

### 7.4 消融实验三：噪声鲁棒性实验

#### 7.4.1 实验目的

验证 RPT-Net 在低质量骨架输入下是否更稳健。

#### 7.4.2 实验设置

建议只在：

```text
NTU60 X-Sub
```

上进行。

#### 7.4.3 噪声设置

推荐使用简单的关节点随机丢弃策略：

```text
Clean
Drop 10%
Drop 20%
Drop 30%
```

实现方式：

```text
随机选择一定比例的关节点，将其坐标置零或将其置信度置零。
```

#### 7.4.4 表格设计

| Method | Clean | Drop 10% | Drop 20% | Drop 30% |
|---|---:|---:|---:|---:|
| HP-Att Baseline | - | - | - | - |
| HP-Att + UE | - | - | - | - |
| HP-Att + RA | - | - | - | - |
| RPT-Net | - | - | - | - |

#### 7.4.5 预期结论

```text
1. 在无噪声条件下，RPT-Net 应取得较好性能；
2. 在关节点缺失条件下，RPT-Net 性能下降应更慢；
3. UE 可以提升模型对低质量 Token 的感知能力；
4. RA 可以减少不可靠 Token 对注意力交互的影响。
```

---

### 7.5 可解释性实验：可靠性注意力可视化

#### 7.5.1 实验目的

展示可靠性注意力是否能够让模型关注更加合理的时空部位。

#### 7.5.2 可视化对象

建议选择 2 到 3 类动作，例如：

```text
1. clapping
2. kicking
3. drinking water
4. falling down
5. sitting down / standing up
```

#### 7.5.3 可视化内容

每个样本展示：

```text
1. Skeleton sequence 或关键帧；
2. Standard Attention 的 T × P 热力图；
3. Reliability Attention 的 T × P 热力图；
4. Reliability Score 的 T × P 热力图。
```

其中：

```text
T 表示时间帧；
P 表示身体部位；
颜色深浅表示注意力权重或可靠性分数。
```

#### 7.5.4 预期分析

对于不同动作，模型应关注不同身体部位：

| 动作 | 预期关注部位 |
|---|---|
| clapping | left arm, right arm |
| kicking | left leg 或 right leg |
| drinking water | arm, head / torso |
| falling down | torso, legs, temporal transition |
| standing up | torso, legs |

预期结论：

```text
1. Standard Attention 的关注可能较分散；
2. Reliability Attention 更倾向于关注动作关键帧和关键部位；
3. 可靠性较低的部位权重会被适当抑制；
4. 模型具有更好的可解释性。
```

---

## 8. 实验步骤规划

### 阶段一：数据处理

需要实现：

```text
tools/preprocess_ntu.py
datasets/ntu_dataset.py
datasets/transforms.py
```

任务：

```text
1. 读取原始 NTU skeleton 文件；
2. 过滤错误样本；
3. 生成统一 [N, C, T, V, M] 数据；
4. 生成 X-Sub / X-View / X-Set split；
5. 验证 dataloader 输出是否正确。
```

验收标准：

```text
1. 能够成功加载一个 batch；
2. 输入 shape 正确；
3. label 正确；
4. train / val split 数量符合预期。
```

---

### 阶段二：Baseline 实现

需要实现：

```text
models/baseline.py
models/modules/heatmap_generator.py
models/modules/part_tokenizer.py
models/modules/spatial_attention.py
models/modules/temporal_attention.py
```

任务：

```text
1. 将 skeleton 坐标转换为 heatmap；
2. 使用 CNN 提取 heatmap feature；
3. 聚合为 part token；
4. 使用标准注意力进行时空建模；
5. 完成分类输出。
```

验收标准：

```text
1. 前向传播无报错；
2. 输出 shape 为 [B, num_classes]；
3. 能够在少量 batch 上完成 overfit 测试；
4. 能够在 NTU60 X-Sub 上完整训练。
```

---

### 阶段三：训练与评估框架

需要实现：

```text
train.py
test.py
engine/trainer.py
engine/evaluator.py
engine/losses.py
engine/metrics.py
utils/logger.py
utils/checkpoint.py
```

任务：

```text
1. 支持配置文件训练；
2. 支持保存 checkpoint；
3. 支持恢复训练；
4. 支持验证集评估；
5. 支持记录 best accuracy；
6. 支持保存实验日志。
```

验收标准：

```text
1. 可以通过 yaml 配置启动训练；
2. 每轮输出 loss 和 accuracy；
3. 自动保存 best checkpoint；
4. 测试脚本可以加载 checkpoint 并输出最终准确率。
```

---

### 阶段四：不确定性编码模块实现

需要实现：

```text
models/modules/uncertainty_encoder.py
```

任务：

```text
1. 计算 confidence；
2. 计算 dispersion；
3. 计算 temporal stability；
4. 将三种不确定性指标编码到 part token 中。
```

输入输出：

```python
part_tokens: [B, T, P, D]
uncertainty: [B, T, P, 3]
output: [B, T, P, D]
```

验收标准：

```text
1. 可以通过配置打开或关闭 UE；
2. UE 打开后模型前向传播正常；
3. 可以单独保存 uncertainty score 用于分析。
```

---

### 阶段五：可靠性注意力模块实现

需要实现：

```text
models/modules/reliability_attention.py
```

任务：

```text
1. 根据不确定性信息计算 reliability score；
2. 将 reliability score 融入 attention score；
3. 支持标准注意力和可靠性注意力切换。
```

验收标准：

```text
1. 可以通过配置打开或关闭 RA；
2. RA 打开后模型前向传播正常；
3. 可以导出 attention map 和 reliability map。
```

---

### 阶段六：主实验

执行：

```text
NTU60 X-Sub
NTU60 X-View
NTU120 X-Sub
NTU120 X-Set
```

输出：

```text
1. Top-1 Accuracy；
2. 日志文件；
3. checkpoint；
4. 实验结果表格。
```

---

### 阶段七：消融实验

执行：

```text
1. 核心模块消融；
2. 不确定性指标组成消融；
3. 噪声鲁棒性实验。
```

建议全部只在：

```text
NTU60 X-Sub
```

上进行。

---

### 阶段八：可视化实验

需要实现：

```text
tools/visualize_attention.py
tools/visualize_reliability.py
```

输出：

```text
1. T × P attention heatmap；
2. T × P reliability heatmap；
3. 关键帧 skeleton 可视化；
4. 可用于论文展示的图片。
```

---

## 9. 推荐代码目录结构

```text
RPT-Net/
│
├── configs/
│   ├── ntu60/
│   │   ├── hpat_baseline_xsub.yaml
│   │   ├── rptnet_xsub.yaml
│   │   ├── rptnet_xview.yaml
│   │   └── ablation/
│   │       ├── baseline.yaml
│   │       ├── baseline_ue.yaml
│   │       ├── baseline_ra.yaml
│   │       ├── full_rptnet.yaml
│   │       ├── uncertainty_c.yaml
│   │       ├── uncertainty_cd.yaml
│   │       └── uncertainty_cds.yaml
│   │
│   └── ntu120/
│       ├── rptnet_xsub.yaml
│       └── rptnet_xset.yaml
│
├── data/
│   └── ntu/
│       ├── raw/
│       ├── processed/
│       └── split/
│
├── datasets/
│   ├── __init__.py
│   ├── ntu_dataset.py
│   ├── feeder.py
│   ├── transforms.py
│   └── graph.py
│
├── models/
│   ├── __init__.py
│   ├── baseline.py
│   ├── rptnet.py
│   │
│   └── modules/
│       ├── heatmap_generator.py
│       ├── cnn_encoder.py
│       ├── part_tokenizer.py
│       ├── uncertainty_encoder.py
│       ├── reliability_attention.py
│       ├── spatial_attention.py
│       ├── temporal_attention.py
│       └── classifier.py
│
├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   ├── losses.py
│   └── metrics.py
│
├── tools/
│   ├── preprocess_ntu.py
│   ├── generate_split.py
│   ├── compute_flops.py
│   ├── visualize_skeleton.py
│   ├── visualize_attention.py
│   └── visualize_reliability.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── checkpoint.py
│   ├── seed.py
│   └── io.py
│
├── outputs/
│   ├── ntu60_xsub/
│   ├── ntu60_xview/
│   ├── ntu120_xsub/
│   └── ntu120_xset/
│
├── train.py
├── test.py
├── infer.py
├── requirements.txt
└── README.md
```

---

## 10. 配置文件建议

### 10.1 Baseline 配置

```yaml
model:
  name: HP-Att
  use_uncertainty_encoding: false
  use_reliability_attention: false
  use_confidence: false
  use_dispersion: false
  use_temporal_stability: false
```

### 10.2 Baseline + UE 配置

```yaml
model:
  name: HP-Att-UE
  use_uncertainty_encoding: true
  use_reliability_attention: false
  use_confidence: true
  use_dispersion: true
  use_temporal_stability: true
```

### 10.3 Baseline + RA 配置

```yaml
model:
  name: HP-Att-RA
  use_uncertainty_encoding: false
  use_reliability_attention: true
  use_confidence: true
  use_dispersion: true
  use_temporal_stability: true
```

### 10.4 Full RPT-Net 配置

```yaml
model:
  name: RPT-Net
  use_uncertainty_encoding: true
  use_reliability_attention: true
  use_confidence: true
  use_dispersion: true
  use_temporal_stability: true
```

---

## 11. 实验优先级

### 最高优先级

```text
1. NTU 数据预处理
2. Baseline 跑通
3. Full RPT-Net 跑通
4. 核心模块消融
```

### 中等优先级

```text
1. 不确定性指标组成消融
2. 噪声鲁棒性实验
3. 主实验四个 benchmark
```

### 较低优先级

```text
1. FLOPs 和 Params 统计
2. 注意力可视化
3. 不同动作类别的案例分析
```

---

## 12. 最终论文实验结构建议

论文实验部分可以按照以下结构组织：

```text
4. Experiments

4.1 Datasets and Evaluation Protocols
    - NTU RGB+D 60
    - NTU RGB+D 120
    - X-Sub, X-View, X-Set

4.2 Implementation Details
    - input shape
    - optimizer
    - learning rate
    - batch size
    - epoch
    - heatmap size
    - part division

4.3 Comparison with State-of-the-Art Methods
    - 主实验结果表

4.4 Ablation Study
    - 核心模块消融
    - 不确定性指标组成消融

4.5 Robustness Analysis
    - 关节点丢弃或噪声扰动实验

4.6 Visualization and Interpretability
    - attention map
    - reliability map
```

---

## 13. 关键结论预期

最终实验希望支撑以下结论：

```text
1. RPT-Net 在 NTU-60 和 NTU-120 上取得具有竞争力的识别性能；
2. 不确定性编码能够增强部位级 Token 的姿态质量感知能力；
3. 可靠性注意力能够改善时空部位之间的注意力分配；
4. 两个模块具有互补性；
5. 在关节点缺失或姿态扰动条件下，RPT-Net 比 baseline 更鲁棒；
6. 可靠性注意力能够产生更合理的动作关键部位关注分布。
```

---

## 14. 后续代码智能体执行建议

后续指导代码智能体时，应优先让其按照以下顺序完成：

```text
Step 1: 构建项目目录结构
Step 2: 实现 NTU 数据预处理
Step 3: 实现 NTU Dataset 和 DataLoader
Step 4: 实现 HeatmapGenerator
Step 5: 实现 CNNEncoder
Step 6: 实现 PartTokenizer
Step 7: 实现 HP-Att Baseline
Step 8: 实现训练和验证框架
Step 9: 实现 UncertaintyEncoder
Step 10: 实现 ReliabilityAttention
Step 11: 实现 RPT-Net
Step 12: 实现配置文件控制模块开关
Step 13: 跑通 NTU60 X-Sub baseline
Step 14: 跑通 NTU60 X-Sub full model
Step 15: 进行核心模块消融
Step 16: 进行主实验和鲁棒性实验
Step 17: 实现可视化脚本
```

---

## 15. 简要总结

本实验规划以 **Heatmap-Part Attention Baseline** 为基础，围绕 **不确定性编码** 和 **可靠性注意力** 两个核心模块进行验证。

实验设计遵循以下原则：

```text
1. 实验量适中；
2. 逻辑链条完整；
3. 能够支撑论文创新点；
4. 便于后续代码实现；
5. 便于撰写实验分析。
```

最终目标是形成一套适合三区期刊论文的实验体系：  
既有与先进方法的标准对比，又有针对核心模块的消融实验，同时通过鲁棒性和可解释性分析进一步增强方法说服力。

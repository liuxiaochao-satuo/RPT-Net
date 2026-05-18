# PH-Att Baseline 模型设计文档

## 1. Baseline 名称

推荐将当前 baseline 命名为：

```text
PH-Att Baseline
Part Heatmap Attention Baseline
```

其核心思想是：

```text
坐标直接生成部位热图；
每个部位热图经过共享 CNN；
得到部位 Token；
再进行空间-时间注意力建模；
最后完成动作分类。
```

该 baseline 作为后续 RPT-Net 的基础模型，后续将在其上加入：

```text
1. Uncertainty Encoding，不确定性编码模块
2. Reliability Attention，可靠性注意力模块
```

因此，baseline 的设计应当保持结构清晰、接口明确、易于扩展。

---

## 2. 整体模型流程

PH-Att Baseline 的完整流程如下：

```text
Input Skeleton
[B, C, T, V, M]
    ↓
Coordinate Normalization
    ↓
Part Heatmap Generation
[B, T, P, H, W]
    ↓
Shared CNN Encoder
[B*T*P, 1, H, W] → [B*T*P, D]
    ↓
Part Token Reshape
[B, T, P, D]
    ↓
Part Embedding + Temporal Embedding
[B, T, P, D]
    ↓
Spatial-Temporal Attention Blocks
[B, T, P, D]
    ↓
Global Average Pooling
[B, D]
    ↓
Classifier
[B, num_classes]
```

其中：

| 符号 | 含义 | 推荐值 |
|---|---|---|
| B | batch size | 16 / 32 |
| C | 坐标通道数 | 3，即 x, y, z |
| T | 输入帧数 | 64，显存不足可用 32 |
| V | 关节点数 | NTU 为 25 |
| M | 人数 | NTU 最多 2 |
| P | 身体部位数 | 推荐 5 |
| H, W | 热图尺寸 | 56 × 56 |
| D | Token 维度 | 128 |

---

## 3. 输入数据格式

模型输入为 NTU skeleton 数据，建议统一为：

```python
x.shape = [B, C, T, V, M]
```

例如：

```python
x.shape = [32, 3, 64, 25, 2]
```

其中：

```text
C = 3，表示 x, y, z 坐标；
T = 64，表示采样后的帧数；
V = 25，表示 NTU 的 25 个关节点；
M = 2，表示最多两个人。
```

分类标签为：

```python
label.shape = [B]
```

输出为：

```python
logits.shape = [B, num_classes]
```

其中：

```text
NTU-60: num_classes = 60
NTU-120: num_classes = 120
```

---

## 4. 模块一：坐标归一化

### 4.1 模块目的

NTU 原始骨架坐标是 3D 坐标，数值范围不一定固定。为了将坐标映射到二维热图，需要先进行归一化。

### 4.2 推荐处理方式

第一版 baseline 建议使用 `x-y` 平面生成二维热图：

```text
使用 skeleton 中的 x 和 y 坐标；
暂时不使用 z 坐标生成热图；
z 坐标可以在后续增强版本中扩展为多视角热图。
```

### 4.3 坐标归一化方式

对每个样本或每个 batch 的 x、y 坐标做 min-max 归一化：

```python
x_norm = (x - x_min) / (x_max - x_min + eps)
y_norm = (y - y_min) / (y_max - y_min + eps)
```

然后映射到热图坐标：

```python
u = x_norm * (W - 1)
v = y_norm * (H - 1)
```

其中：

```text
u 表示热图中的横坐标；
v 表示热图中的纵坐标；
H = W = 56。
```

### 4.4 需要注意的问题

如果一个样本中存在无效关节点，例如坐标全为 0，需要避免其影响 min-max 计算。

建议：

```text
1. 识别无效关节点；
2. 对有效关节点计算 min 和 max；
3. 无效关节点生成全零热图。
```

---

## 5. 模块二：身体部位划分

### 5.1 模块目的

将 25 个关节点划分为若干身体部位，使模型从一开始就围绕部位级结构建模。

### 5.2 推荐部位数量

第一版推荐划分为 5 个部位：

```text
1. torso
2. left_arm
3. right_arm
4. left_leg
5. right_leg
```

### 5.3 推荐部位索引

以下索引为建议版本，实际实现时需要与 NTU 数据预处理后的 joint index 保持一致：

```python
PARTS = {
    "torso":      [0, 1, 2, 3, 20],
    "left_arm":  [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_leg":  [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19],
}
```

### 5.4 后续可扩展方案

如果 5 个部位过于粗糙，可以扩展为 6 或 7 个部位，例如：

```text
torso
head
left_arm
right_arm
left_leg
right_leg
```

或者：

```text
torso
left_arm
right_arm
left_hand
right_hand
left_leg
right_leg
```

但第一版 baseline 建议保持 5 个部位，避免 token 数量增加和实验复杂化。

---

## 6. 模块三：部位热图生成

### 6.1 模块目的

直接根据关节坐标生成部位级热图，而不是先生成关节级热图再聚合特征。

对于每个身体部位，将其包含的多个关节点热图融合为一个 part heatmap。

### 6.2 输入输出

输入：

```python
skeleton.shape = [B, C, T, V, M]
```

输出：

```python
part_heatmaps.shape = [B, T, P, H, W]
```

例如：

```python
part_heatmaps.shape = [32, 64, 5, 56, 56]
```

### 6.3 单个关节热图计算

对于某一帧中某个关节，其热图中心为：

```text
(u, v)
```

则热图中位置 `(i, j)` 的响应值为：

```text
H(i, j) = exp(-((j - u)^2 + (i - v)^2) / (2 * sigma^2))
```

其中：

| 参数 | 含义 | 推荐值 |
|---|---|---|
| H, W | 热图高宽 | 56 |
| sigma | 高斯核标准差 | 2.0 |
| u, v | 归一化后映射的热图坐标 | 0 到 55 |

### 6.4 多人处理方式

NTU 中最多存在 2 个人。对于同一关节、同一部位，可以对不同人的热图进行融合。

推荐使用：

```text
max fusion
```

即：

```python
joint_heatmap = max(joint_heatmap_person_1, joint_heatmap_person_2)
```

优点：

```text
1. 避免多人响应相加导致数值过大；
2. 保留最强空间响应；
3. 实现简单稳定。
```

### 6.5 部位内关节融合方式

一个部位包含多个关节。推荐使用：

```text
Sum + Normalize
```

流程：

```python
part_heatmap = sum(joint_heatmaps_in_this_part)
part_heatmap = part_heatmap / (part_heatmap.max() + eps)
```

最终将每个部位热图归一化到：

```text
[0, 1]
```

### 6.6 为什么使用部位热图

直接生成部位热图具有以下优点：

```text
1. 部位 Token 来源更加明确；
2. 模型从输入阶段就进行部位级建模；
3. 计算量小于 25 个关节热图输入；
4. 后续不确定性指标可以直接从部位热图计算；
5. 后续可靠性注意力可以自然作用在 T × P 个部位 Token 上；
6. 可视化更直观。
```

---

## 7. 模块四：共享 CNN 编码器

### 7.1 模块目的

共享 CNN 用于从每个部位热图中提取局部空间响应特征，并将每个部位热图编码为一个部位 Token。

### 7.2 输入输出

部位热图：

```python
part_heatmaps.shape = [B, T, P, H, W]
```

送入 CNN 前 reshape：

```python
cnn_input = part_heatmaps.reshape(B * T * P, 1, H, W)
```

CNN 输入：

```python
cnn_input.shape = [B*T*P, 1, 56, 56]
```

CNN 输出：

```python
part_token_flat.shape = [B*T*P, D]
```

reshape 回：

```python
part_tokens.shape = [B, T, P, D]
```

例如：

```python
part_tokens.shape = [32, 64, 5, 128]
```

### 7.3 推荐 CNN 结构

第一版建议使用轻量 2D CNN：

```text
Conv2d(1, 32, kernel_size=3, padding=1)
BatchNorm2d(32)
ReLU

Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
BatchNorm2d(64)
ReLU

Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
BatchNorm2d(128)
ReLU

AdaptiveAvgPool2d(1)
Flatten
```

如果输出通道为 128，则得到：

```python
part_token_flat.shape = [B*T*P, 128]
```

如果需要不同 token 维度，可以增加：

```python
Linear(128, token_dim)
```

### 7.4 推荐参数

| 参数 | 推荐值 |
|---|---|
| CNN 输入通道 | 1 |
| CNN 输出维度 | 128 |
| 卷积核大小 | 3 |
| stride | 1, 2, 2 |
| normalization | BatchNorm2d |
| activation | ReLU |
| pooling | AdaptiveAvgPool2d(1) |

### 7.5 为什么使用共享 CNN

所有部位使用同一个 CNN 编码器，而不是每个部位单独一个 CNN。

优点：

```text
1. 参数量更小；
2. 不同部位共享空间响应提取规则；
3. 避免小数据条件下过拟合；
4. 实现简单；
5. 后续可以通过 part embedding 区分不同部位语义。
```

---

## 8. 模块五：Part Embedding 与 Temporal Embedding

### 8.1 模块目的

部位 Token 本身只是从热图中提取的特征。为了让 attention 知道每个 token 的语义位置和时间位置，需要加入嵌入信息。

### 8.2 Part Embedding

定义：

```python
part_embedding.shape = [1, 1, P, D]
```

加入方式：

```python
part_tokens = part_tokens + part_embedding
```

作用：

```text
告诉模型当前 token 对应 torso、left_arm、right_arm、left_leg 还是 right_leg。
```

### 8.3 Temporal Embedding

定义：

```python
time_embedding.shape = [1, T, 1, D]
```

加入方式：

```python
part_tokens = part_tokens + time_embedding
```

作用：

```text
告诉模型当前 token 属于第几帧。
```

### 8.4 输出

加入嵌入后，shape 不变：

```python
part_tokens.shape = [B, T, P, D]
```

---

## 9. 模块六：空间-时间注意力模块

### 9.1 模块目的

对时序部位 Token 进行关系建模。

建议采用空间注意力和时间注意力分离的方式：

```text
Spatial Attention:
    建模同一帧内不同身体部位之间的关系。

Temporal Attention:
    建模同一身体部位在不同时间帧之间的动态变化。
```

输入输出：

```python
x.shape = [B, T, P, D]
```

经过注意力模块后：

```python
x.shape = [B, T, P, D]
```

### 9.2 Spatial Attention

输入：

```python
x.shape = [B, T, P, D]
```

reshape：

```python
x_s = x.reshape(B * T, P, D)
```

执行 Multi-Head Self-Attention：

```python
x_s = SpatialMHSA(x_s)
```

reshape 回：

```python
x.shape = [B, T, P, D]
```

作用：

```text
在每一帧内部，让 torso、arms、legs 等部位进行信息交互。
```

### 9.3 Temporal Attention

输入：

```python
x.shape = [B, T, P, D]
```

reshape：

```python
x_t = x.permute(0, 2, 1, 3)
x_t = x_t.reshape(B * P, T, D)
```

执行 Multi-Head Self-Attention：

```python
x_t = TemporalMHSA(x_t)
```

reshape 回：

```python
x.shape = [B, T, P, D]
```

作用：

```text
对每个身体部位单独建模其跨帧运动变化。
```

### 9.4 推荐 Attention Block 结构

推荐采用：

```python
class STAttentionBlock(nn.Module):
    def forward(self, x):
        x = x + self.spatial_attn(self.norm1(x))
        x = x + self.temporal_attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x
```

其中：

```text
norm 使用 LayerNorm；
spatial_attn 和 temporal_attn 使用 Multi-Head Self-Attention；
mlp 使用两层全连接网络。
```

### 9.5 推荐参数

| 参数 | 推荐值 |
|---|---|
| token_dim | 128 |
| attention depth | 2 或 4 |
| num_heads | 4 |
| mlp_ratio | 4.0 |
| dropout | 0.1 |
| attention dropout | 0.1 |
| normalization | LayerNorm |

第一版建议：

```text
depth = 4
num_heads = 4
dim = 128
```

如果显存或训练速度有压力，可以改为：

```text
depth = 2
dim = 96
```

---

## 10. 模块七：分类器

### 10.1 模块目的

将经过注意力增强后的时空部位 Token 聚合为视频级特征，并输出动作类别。

### 10.2 输入输出

输入：

```python
x.shape = [B, T, P, D]
```

使用全局平均池化：

```python
feat = x.mean(dim=(1, 2))
```

得到：

```python
feat.shape = [B, D]
```

分类器：

```python
logits = classifier(feat)
```

输出：

```python
logits.shape = [B, num_classes]
```

### 10.3 推荐分类器结构

```text
LayerNorm(D)
Dropout(0.5)
Linear(D, num_classes)
```

对应 PyTorch 结构：

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Dropout(0.5),
    nn.Linear(dim, num_classes)
)
```

### 10.4 为什么使用 Global Average Pooling

第一版 baseline 建议使用 Global Average Pooling，而不是 CLS token。

原因：

```text
1. 实现简单；
2. 训练稳定；
3. 不引入额外复杂因素；
4. 便于突出后续 UE 和 RA 模块的作用。
```

---

## 11. Baseline 前向传播伪代码

```python
class PHAttBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.part_heatmap_generator = PartHeatmapGenerator(
            parts=cfg.parts,
            heatmap_size=cfg.heatmap_size,
            sigma=cfg.sigma
        )

        self.cnn_encoder = SharedCNNEncoder(
            in_channels=1,
            out_dim=cfg.token_dim
        )

        self.part_embedding = nn.Parameter(
            torch.zeros(1, 1, cfg.num_parts, cfg.token_dim)
        )

        self.time_embedding = nn.Parameter(
            torch.zeros(1, cfg.num_frames, 1, cfg.token_dim)
        )

        self.blocks = nn.ModuleList([
            STAttentionBlock(
                dim=cfg.token_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout
            )
            for _ in range(cfg.depth)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.token_dim, cfg.num_classes)
        )

    def forward(self, x):
        # x: [B, C, T, V, M]

        part_heatmaps = self.part_heatmap_generator(x)
        # [B, T, P, H, W]

        B, T, P, H, W = part_heatmaps.shape

        cnn_input = part_heatmaps.reshape(B * T * P, 1, H, W)
        # [B*T*P, 1, H, W]

        part_tokens = self.cnn_encoder(cnn_input)
        # [B*T*P, D]

        part_tokens = part_tokens.reshape(B, T, P, -1)
        # [B, T, P, D]

        part_tokens = part_tokens + self.part_embedding + self.time_embedding[:, :T]

        x = part_tokens

        for block in self.blocks:
            x = block(x)

        feat = x.mean(dim=(1, 2))
        # [B, D]

        logits = self.classifier(feat)
        # [B, num_classes]

        return logits
```

---

## 12. 推荐配置文件

```yaml
model:
  name: PH-Att

  input:
    num_joints: 25
    num_person: 2
    in_channels: 3
    num_frames: 64

  heatmap:
    type: part_heatmap
    size: 56
    sigma: 2.0
    coordinate_plane: xy
    multi_person_fusion: max
    part_fusion: sum_norm

  parts:
    num_parts: 5
    layout:
      torso: [0, 1, 2, 3, 20]
      left_arm: [4, 5, 6, 7, 21, 22]
      right_arm: [8, 9, 10, 11, 23, 24]
      left_leg: [12, 13, 14, 15]
      right_leg: [16, 17, 18, 19]

  cnn:
    type: shared_2d_cnn
    in_channels: 1
    channels: [32, 64, 128]
    kernel_size: 3
    strides: [1, 2, 2]
    out_dim: 128
    norm: batchnorm
    activation: relu
    pooling: adaptive_avg_pool

  token:
    dim: 128
    use_part_embedding: true
    use_time_embedding: true

  attention:
    type: separated_spatial_temporal
    depth: 4
    num_heads: 4
    mlp_ratio: 4.0
    dropout: 0.1
    attn_dropout: 0.1
    norm: layernorm

  classifier:
    pooling: mean
    dropout: 0.5
    num_classes: 60
```

---

## 13. 推荐代码文件划分

建议将 baseline 拆分为以下文件：

```text
models/
├── phatt_baseline.py
│
└── modules/
    ├── part_heatmap_generator.py
    ├── shared_cnn_encoder.py
    ├── st_attention_block.py
    ├── spatial_attention.py
    ├── temporal_attention.py
    └── classifier.py
```

### 13.1 `part_heatmap_generator.py`

负责：

```text
1. 接收 skeleton 坐标；
2. 进行坐标归一化；
3. 根据 part layout 生成部位热图；
4. 输出 [B, T, P, H, W]。
```

### 13.2 `shared_cnn_encoder.py`

负责：

```text
1. 接收 [B*T*P, 1, H, W]；
2. 通过共享 CNN 提取空间特征；
3. 输出 [B*T*P, D]。
```

### 13.3 `spatial_attention.py`

负责：

```text
1. 接收 [B, T, P, D]；
2. reshape 为 [B*T, P, D]；
3. 计算同一帧内的部位关系；
4. 输出 [B, T, P, D]。
```

### 13.4 `temporal_attention.py`

负责：

```text
1. 接收 [B, T, P, D]；
2. reshape 为 [B*P, T, D]；
3. 计算同一部位跨帧关系；
4. 输出 [B, T, P, D]。
```

### 13.5 `st_attention_block.py`

负责：

```text
1. 串联 spatial attention；
2. 串联 temporal attention；
3. 加入 FFN；
4. 使用 residual connection 和 LayerNorm。
```

### 13.6 `phatt_baseline.py`

负责：

```text
1. 调用 PartHeatmapGenerator；
2. 调用 SharedCNNEncoder；
3. 加入 part/time embedding；
4. 堆叠 STAttentionBlock；
5. 完成分类。
```

---

## 14. 后续扩展接口

baseline 实现时需要预留 UE 和 RA 接口。

### 14.1 不确定性编码接口

在得到 part heatmaps 和 part tokens 后预留：

```python
if self.use_uncertainty_encoding:
    uncertainty = self.compute_uncertainty(part_heatmaps)
    part_tokens = self.uncertainty_encoder(part_tokens, uncertainty)
```

其中：

```python
uncertainty.shape = [B, T, P, 3]
```

包含：

```text
confidence
dispersion
temporal_stability
```

### 14.2 可靠性注意力接口

attention block 支持传入 reliability：

```python
for block in self.blocks:
    x = block(x, reliability=reliability)
```

baseline 中：

```python
reliability = None
```

完整 RPT-Net 中：

```python
reliability.shape = [B, T, P]
```

### 14.3 可视化接口

建议 attention 模块支持返回 attention map：

```python
logits, aux = model(x, return_attention=True)
```

其中 aux 可以包含：

```python
aux = {
    "part_heatmaps": part_heatmaps,
    "spatial_attention": spatial_attention_maps,
    "temporal_attention": temporal_attention_maps,
    "part_tokens": part_tokens
}
```

后续加入 RPT-Net 后还可以包含：

```python
aux = {
    "uncertainty": uncertainty,
    "reliability": reliability
}
```

---

## 15. 实现优先级

### 第一阶段：最低可运行版本

优先实现：

```text
1. PartHeatmapGenerator
2. SharedCNNEncoder
3. PartEmbedding + TimeEmbedding
4. SpatialAttention
5. TemporalAttention
6. Classifier
7. PHAttBaseline forward
```

验收标准：

```text
输入 [B, 3, T, 25, 2]；
输出 [B, num_classes]；
前向传播不报错；
可以进行 loss.backward()。
```

### 第二阶段：训练验证

实现：

```text
1. train.py
2. test.py
3. dataloader
4. loss
5. accuracy
6. checkpoint
```

验收标准：

```text
能在少量数据上过拟合；
能在 NTU60 X-Sub 上完整训练；
能保存 best checkpoint。
```

### 第三阶段：扩展 RPT-Net

在 baseline 上加入：

```text
1. UncertaintyEncoder
2. ReliabilityAttention
3. 模块开关配置
4. 消融实验配置
```

---

## 16. 需要避免的问题

### 16.1 不要让 baseline 过于复杂

第一版 baseline 不建议加入：

```text
1. 三视角 xy/xz/yz 热图；
2. limb heatmap；
3. 3D CNN；
4. CLS token；
5. 多流 joint/bone/motion fusion；
6. 每个部位单独 CNN；
7. 复杂数据增强。
```

这些可以作为后续增强，不应作为第一版 baseline 的必选项。

### 16.2 注意部位热图数值归一化

不同部位包含关节数量不同，如果直接 sum，可能导致不同部位热图强度不公平。

必须进行：

```text
part heatmap normalization
```

推荐：

```python
part_heatmap = part_heatmap / (part_heatmap.max() + eps)
```

### 16.3 注意无效关节点

如果关节点坐标无效，应避免生成错误热图。

建议：

```text
无效关节点对应热图为全零；
无效部位如果所有关节都无效，则该部位热图为全零。
```

### 16.4 注意张量维度

核心维度必须保持一致：

```text
PartHeatmapGenerator 输出:
[B, T, P, H, W]

SharedCNNEncoder 输入:
[B*T*P, 1, H, W]

SharedCNNEncoder 输出:
[B*T*P, D]

Attention 输入:
[B, T, P, D]

Classifier 输出:
[B, num_classes]
```

---

## 17. 最终总结

PH-Att Baseline 是 RPT-Net 的基础模型，核心流程为：

```text
坐标 → 部位热图 → 共享 CNN → 部位 Token → 空间-时间注意力 → 分类器
```

该 baseline 的优势是：

```text
1. 结构简单；
2. 模块边界清楚；
3. 计算量适中；
4. 直接服务于部位级建模；
5. 便于计算部位级不确定性；
6. 便于后续加入可靠性注意力；
7. 便于进行注意力和可靠性可视化。
```

后续完整 RPT-Net 可以在该 baseline 上自然扩展为：

```text
PH-Att Baseline
+ Uncertainty Encoding
+ Reliability Attention
```

该文档可作为代码智能体实现 baseline 模型的技术规格说明。

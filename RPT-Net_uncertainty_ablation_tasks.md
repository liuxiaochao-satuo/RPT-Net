# RPT-Net 不确定性编码与消融实验任务文档

## 1. 当前阶段判断

当前 PH-Att Baseline 已经在 NTU-60 X-Sub 上跑通。下一步不应直接开始完整消融训练，而应先实现并检查不确定性指标。

推荐顺序：

```text
PH-Att Baseline 已跑通
→ 实现 UncertaintyEstimator
→ 检查 confidence / dispersion / temporal_instability
→ 实现 UncertaintyEncoder
→ 跑 PH-Att + UE
→ 实现 ReliabilityEstimator
→ 改造 Attention 支持 reliability
→ 跑 PH-Att + RA
→ 跑完整 RPT-Net
→ 整理消融实验
```

这样做的原因：

```text
1. UE 和 RA 都依赖不确定性指标；
2. 如果指标本身异常，后续训练结果没有解释意义；
3. 先做数值检查可以降低调试难度；
4. 消融实验结果更可信。
```

---

## 2. 输入输出约定

不确定性指标从部位热图中计算。

输入：

```python
part_heatmaps.shape = [B, T, P, H, W]
```

输出：

```python
uncertainty.shape = [B, T, P, K]
```

第一版建议：

```text
K = 3
0: confidence
1: dispersion
2: temporal_instability
```

三类指标方向：

| 指标 | 数值越大表示 | 可靠性方向 |
|---|---|---|
| confidence | 热图响应越强 | 越可靠 |
| dispersion | 热图越分散 | 越不可靠 |
| temporal_instability | 帧间抖动越大 | 越不可靠 |

---

## 3. Confidence 实现方式

### 3.1 含义

Confidence 表示当前部位热图响应强度。

```text
热图峰值越高，说明该部位响应越明确；
热图峰值越低，说明该部位可能缺失或响应较弱。
```

### 3.2 推荐实现

```python
confidence = part_heatmaps.amax(dim=(-2, -1))
```

输出：

```python
confidence.shape = [B, T, P]
```

### 3.3 检查项

```text
1. 是否存在 NaN / Inf；
2. 是否基本位于 [0, 1]；
3. 是否所有值都接近 1；
4. 是否所有值都接近 0；
5. 不同部位、不同帧之间是否有差异。
```

如果每个 part heatmap 都被单独归一化到峰值为 1，则 confidence 可能区分度不足。此时可以考虑：

```python
confidence = part_heatmaps.mean(dim=(-2, -1))
```

---

## 4. Dispersion 实现方式

### 4.1 含义

Dispersion 表示部位热图的空间扩散程度。

```text
热图越集中，位置越明确；
热图越分散，空间不确定性越强。
```

### 4.2 推荐实现

使用热图响应的加权方差。

```python
def compute_dispersion(part_heatmaps, eps=1e-6):
    B, T, P, H, W = part_heatmaps.shape
    device = part_heatmaps.device

    ys = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)

    weights = part_heatmaps
    norm = weights.sum(dim=(-2, -1), keepdim=True) + eps

    mu_x = (weights * xs).sum(dim=(-2, -1), keepdim=True) / norm
    mu_y = (weights * ys).sum(dim=(-2, -1), keepdim=True) / norm

    dist2 = (xs - mu_x) ** 2 + (ys - mu_y) ** 2
    dispersion = (weights * dist2).sum(dim=(-2, -1)) / norm.squeeze(-1).squeeze(-1)

    return dispersion
```

输出：

```python
dispersion.shape = [B, T, P]
```

### 4.3 空热图处理

建议增加有效掩码：

```python
valid = part_heatmaps.sum(dim=(-2, -1)) > eps
```

如果某个部位热图全为 0，建议设置：

```text
confidence = 0
dispersion = 1.0
temporal_instability = 1.0
```

表示该部位高度不可靠。

### 4.4 检查项

```text
1. 是否存在 NaN / Inf；
2. 是否非负；
3. 数值是否在合理范围；
4. 集中热图 dispersion 是否较小；
5. 扩散热图 dispersion 是否较大。
```

---

## 5. Temporal Instability 实现方式

### 5.1 含义

Temporal Instability 表示同一部位热图中心在连续帧之间的变化幅度。

```text
中心变化越大，说明该部位越不稳定；
中心变化越小，说明该部位越稳定。
```

### 5.2 加权中心计算

```python
def compute_heatmap_center(part_heatmaps, eps=1e-6):
    B, T, P, H, W = part_heatmaps.shape
    device = part_heatmaps.device

    ys = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)

    weights = part_heatmaps
    norm = weights.sum(dim=(-2, -1), keepdim=True) + eps

    mu_x = (weights * xs).sum(dim=(-2, -1), keepdim=False) / norm.squeeze(-1).squeeze(-1)
    mu_y = (weights * ys).sum(dim=(-2, -1), keepdim=False) / norm.squeeze(-1).squeeze(-1)

    centers = torch.stack([mu_x, mu_y], dim=-1)
    return centers
```

输出：

```python
centers.shape = [B, T, P, 2]
```

### 5.3 Temporal Instability 计算

```python
def compute_temporal_instability(part_heatmaps, eps=1e-6):
    centers = compute_heatmap_center(part_heatmaps, eps=eps)

    delta = torch.norm(centers[:, 1:] - centers[:, :-1], dim=-1)
    first = torch.zeros_like(delta[:, :1])
    instability = torch.cat([first, delta], dim=1)

    return instability
```

输出：

```python
temporal_instability.shape = [B, T, P]
```

### 5.4 检查项

```text
1. 是否存在 NaN / Inf；
2. 是否非负；
3. 第一帧是否为 0；
4. 静止动作是否相对较小；
5. 剧烈动作或抖动样本是否相对较大。
```

---

## 6. UncertaintyEstimator 模块

建议文件：

```text
models/modules/uncertainty_estimator.py
```

推荐结构：

```python
class UncertaintyEstimator(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        use_confidence=True,
        use_dispersion=True,
        use_temporal_instability=True,
    ):
        super().__init__()
        self.eps = eps
        self.use_confidence = use_confidence
        self.use_dispersion = use_dispersion
        self.use_temporal_instability = use_temporal_instability

    def forward(self, part_heatmaps):
        features = []

        if self.use_confidence:
            confidence = self.compute_confidence(part_heatmaps)
            features.append(confidence)

        if self.use_dispersion:
            dispersion = self.compute_dispersion(part_heatmaps)
            features.append(dispersion)

        if self.use_temporal_instability:
            instability = self.compute_temporal_instability(part_heatmaps)
            features.append(instability)

        uncertainty = torch.stack(features, dim=-1)
        return uncertainty
```

输出：

```python
uncertainty.shape = [B, T, P, K]
```

其中 `K` 为启用的不确定性指标数量。

---

## 7. 数值检查脚本

建议文件：

```text
tools/check_uncertainty_stats.py
```

脚本功能：

```text
1. 加载一个 batch；
2. 调用 PartHeatmapGenerator；
3. 调用 UncertaintyEstimator；
4. 打印 min / max / mean / std；
5. 检查 NaN / Inf；
6. 可选保存若干样本结果。
```

推荐打印：

```python
print("uncertainty shape:", uncertainty.shape)

names = ["confidence", "dispersion", "instability"]

for i, name in enumerate(names):
    value = uncertainty[..., i]
    print(name)
    print("  min:", value.min().item())
    print("  max:", value.max().item())
    print("  mean:", value.mean().item())
    print("  std:", value.std().item())
    print("  has_nan:", torch.isnan(value).any().item())
    print("  has_inf:", torch.isinf(value).any().item())
```

通过标准：

```text
1. 无 NaN；
2. 无 Inf；
3. confidence 基本位于 [0, 1]；
4. dispersion 非负；
5. instability 非负；
6. 三类指标不是常数；
7. 不同部位和样本之间存在差异。
```

---

## 8. 可视化检查脚本

建议文件：

```text
tools/visualize_uncertainty.py
```

建议输出：

```text
1. part heatmap；
2. confidence 的 T × P 热力图；
3. dispersion 的 T × P 热力图；
4. temporal_instability 的 T × P 热力图。
```

可视化格式：

```text
横轴：时间帧 T
纵轴：身体部位 P
颜色：指标数值
```

---

## 9. UncertaintyEncoder 实现

建议文件：

```text
models/modules/uncertainty_encoder.py
```

输入：

```python
part_tokens.shape = [B, T, P, D]
uncertainty.shape = [B, T, P, K]
```

输出：

```python
enhanced_tokens.shape = [B, T, P, D]
```

推荐实现：

```python
class UncertaintyEncoder(nn.Module):
    def __init__(self, token_dim=128, uncertainty_dim=3):
        super().__init__()

        self.uncertainty_proj = nn.Sequential(
            nn.Linear(uncertainty_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

    def forward(self, tokens, uncertainty):
        u = self.uncertainty_proj(uncertainty)
        fused = self.fusion(torch.cat([tokens, u], dim=-1))
        return tokens + fused
```

建议使用残差：

```python
return tokens + fused
```

原因：

```text
1. 保留 baseline 原始 token 表达；
2. 减少训练不稳定；
3. 方便与 baseline 对比。
```

---

## 10. ReliabilityEstimator 实现

建议文件：

```text
models/modules/reliability_estimator.py
```

输入：

```python
uncertainty.shape = [B, T, P, K]
```

输出：

```python
reliability.shape = [B, T, P]
```

推荐实现：

```python
class ReliabilityEstimator(nn.Module):
    def __init__(self, uncertainty_dim=3, hidden_dim=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(uncertainty_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, uncertainty):
        reliability = self.mlp(uncertainty).squeeze(-1)
        return reliability
```

说明：

```text
confidence、dispersion、instability 的方向不一致；
因此不建议手工写死 reliability 公式；
建议使用 MLP 端到端学习。
```

---

## 11. Reliability Attention 接入方式

推荐使用 `attention bias`。

普通 attention：

```python
attn_score = (q @ k.transpose(-2, -1)) * scale
attn = softmax(attn_score)
```

加入 reliability bias：

```python
rel_bias = torch.log(reliability.clamp(min=1e-6))
rel_bias = rel_bias.unsqueeze(1).unsqueeze(2)

attn_score = attn_score + alpha * rel_bias
attn = softmax(attn_score)
```

推荐：

```text
alpha = 0.5
```

### Spatial Attention 中的 reliability

```python
x.shape = [B, T, P, D]
reliability.shape = [B, T, P]

x_s = x.reshape(B * T, P, D)
rel_s = reliability.reshape(B * T, P)
```

### Temporal Attention 中的 reliability

```python
x.shape = [B, T, P, D]
reliability.shape = [B, T, P]

x_t = x.permute(0, 2, 1, 3).reshape(B * P, T, D)
rel_t = reliability.permute(0, 2, 1).reshape(B * P, T)
```

---

## 12. 与 Baseline 对接方式

不要写多个独立模型。建议在统一模型中通过配置控制模块开关。

核心逻辑：

```python
part_heatmaps = self.part_heatmap_generator(x)
part_tokens = self.cnn_encoder(part_heatmaps)

uncertainty = None
reliability = None

if self.need_uncertainty:
    uncertainty = self.uncertainty_estimator(part_heatmaps)

if self.use_uncertainty_encoding:
    part_tokens = self.uncertainty_encoder(part_tokens, uncertainty)

part_tokens = part_tokens + self.part_embedding + self.time_embedding[:, :T]

if self.use_reliability_attention:
    reliability = self.reliability_estimator(uncertainty)

for block in self.blocks:
    part_tokens = block(part_tokens, reliability=reliability)

feat = part_tokens.mean(dim=(1, 2))
logits = self.classifier(feat)
```

关键开关：

```python
self.use_uncertainty_encoding = cfg.model.use_uncertainty_encoding
self.use_reliability_attention = cfg.model.use_reliability_attention

self.need_uncertainty = (
    self.use_uncertainty_encoding or self.use_reliability_attention
)
```

---

## 13. 核心模块消融实验设置

消融实验建议只在：

```text
NTU-60 X-Sub
```

进行。

核心消融表：

| Method | UE | RA | Accuracy |
|---|---:|---:|---:|
| PH-Att Baseline | ✗ | ✗ | - |
| PH-Att + UE | ✓ | ✗ | - |
| PH-Att + RA | ✗ | ✓ | - |
| RPT-Net | ✓ | ✓ | - |

配置文件建议：

```text
configs/ntu60/ablation/
├── 00_phatt_baseline_xsub.yaml
├── 01_phatt_ue_xsub.yaml
├── 02_phatt_ra_xsub.yaml
└── 03_rptnet_full_xsub.yaml
```

配置开关：

| 配置文件 | use_uncertainty_encoding | use_reliability_attention |
|---|---:|---:|
| 00_phatt_baseline_xsub.yaml | false | false |
| 01_phatt_ue_xsub.yaml | true | false |
| 02_phatt_ra_xsub.yaml | false | true |
| 03_rptnet_full_xsub.yaml | true | true |

---

## 14. 不确定性指标组成消融

实验目的：

```text
验证 confidence、dispersion、temporal instability 是否逐步带来收益。
```

表格：

| Method | Confidence | Dispersion | Temporal Instability | Accuracy |
|---|---:|---:|---:|---:|
| w/o UE | ✗ | ✗ | ✗ | - |
| C only | ✓ | ✗ | ✗ | - |
| C + D | ✓ | ✓ | ✗ | - |
| C + D + I | ✓ | ✓ | ✓ | - |

配置文件建议：

```text
configs/ntu60/ablation/
├── 04_uncertainty_c_xsub.yaml
├── 05_uncertainty_cd_xsub.yaml
└── 06_uncertainty_cdi_xsub.yaml
```

配置示例：

```yaml
model:
  uncertainty:
    use_confidence: true
    use_dispersion: true
    use_temporal_instability: false
```

注意：

```text
UncertaintyEncoder 和 ReliabilityEstimator 的 uncertainty_dim 必须与实际启用指标数量一致。
```

---

## 15. 训练时必须保持一致的参数

为了保证消融公平，以下参数必须一致：

```text
1. 数据集：NTU-60 X-Sub
2. 输入帧数 T
3. heatmap size
4. sigma
5. part layout
6. CNN 结构
7. token dim
8. attention depth
9. num_heads
10. optimizer
11. learning rate
12. scheduler
13. batch size
14. epoch 数
15. random seed
16. 数据增强方式
```

唯一改变：

```text
1. 是否启用 UE；
2. 是否启用 RA；
3. 使用哪些不确定性指标。
```

---

## 16. 需要记录的关键指标

每个实验记录：

```text
1. train loss
2. val loss
3. val top-1 accuracy
4. best top-1 accuracy
5. best epoch
6. Params
7. FLOPs
```

不确定性相关统计：

```text
1. confidence mean / std
2. dispersion mean / std
3. temporal_instability mean / std
4. reliability mean / std
```

每个实验建议保存：

```text
1. config.yaml
2. train.log
3. best_model.pth
4. last_model.pth
5. metrics.json
```

---

## 17. 推荐运行顺序

先做数值检查：

```bash
python tools/check_uncertainty_stats.py --config configs/ntu60/ablation/01_phatt_ue_xsub.yaml
```

核心模块消融：

```bash
python train.py --config configs/ntu60/ablation/00_phatt_baseline_xsub.yaml
python train.py --config configs/ntu60/ablation/01_phatt_ue_xsub.yaml
python train.py --config configs/ntu60/ablation/02_phatt_ra_xsub.yaml
python train.py --config configs/ntu60/ablation/03_rptnet_full_xsub.yaml
```

不确定性指标组成消融：

```bash
python train.py --config configs/ntu60/ablation/04_uncertainty_c_xsub.yaml
python train.py --config configs/ntu60/ablation/05_uncertainty_cd_xsub.yaml
python train.py --config configs/ntu60/ablation/06_uncertainty_cdi_xsub.yaml
```

---

## 18. 结果合理性判断

核心模块消融理想趋势：

```text
PH-Att Baseline < PH-Att + UE
PH-Att Baseline < PH-Att + RA
RPT-Net >= PH-Att + UE
RPT-Net >= PH-Att + RA
```

不确定性组成消融理想趋势：

```text
C only < C + D < C + D + I
```

如果不是严格递增，可以从以下角度分析：

```text
1. 不同指标之间可能存在冗余；
2. 数据集骨架质量较高，时序不稳定性收益有限；
3. 指标归一化方式需要调整；
4. RA 可能比 UE 更适合利用不确定性信息。
```

---

## 19. 当前阶段交付清单

代码智能体当前应优先完成：

```text
1. models/modules/uncertainty_estimator.py
2. models/modules/uncertainty_encoder.py
3. models/modules/reliability_estimator.py
4. Attention 模块 reliability 参数接口
5. tools/check_uncertainty_stats.py
6. tools/visualize_uncertainty.py
7. 四组核心消融配置文件
8. 三组不确定性组成消融配置文件
```

最先验收：

```text
1. UncertaintyEstimator 是否正确输出 [B, T, P, K]；
2. 三类指标是否无 NaN 和 Inf；
3. 数值范围是否合理；
4. baseline 在 reliability=None 时结果不变；
5. 开启 UE 或 RA 后 forward 和 backward 是否正常。
```

---

## 20. 总结

当前阶段正确路线：

```text
先实现不确定性指标；
先做数值检查；
再接入 UE；
再接入 RA；
最后做消融实验。
```

三个指标作用：

```text
Confidence:
    描述热图响应强度。

Dispersion:
    描述空间响应是否集中。

Temporal Instability:
    描述部位响应在时间维度上是否稳定。
```

UE 和 RA 的关系：

```text
UE:
    将不确定性信息编码进 token，增强 token 表征质量。

RA:
    将不确定性信息转化为 reliability score，用于调节 attention 分配。
```

最终消融实验应证明：

```text
1. UE 有助于提升 token 表征；
2. RA 有助于优化注意力交互；
3. 二者结合具有互补作用；
4. 三类不确定性指标能够从不同角度描述部位级姿态质量。
```

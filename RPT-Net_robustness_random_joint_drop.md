# RPT-Net 鲁棒性实验：随机关节点丢弃实施文档

## 1. 实验目的

本实验用于验证 PH-Att Baseline、PH-Att + UE、PH-Att + RA 和完整 RPT-Net 在低质量骨架输入下的鲁棒性。

核心问题：

```text
当测试样本中的部分关节点被随机丢弃时，
RPT-Net 是否比 baseline 具有更小的准确率下降？
```

本实验主要支撑论文中的 Robustness Analysis 部分，证明：

```text
1. 不确定性编码 UE 能增强模型对低质量部位 token 的感知能力；
2. 可靠性注意力 RA 能抑制不可靠 token 对注意力交互的干扰；
3. 完整 RPT-Net 在关节点缺失场景下更稳定。
```

---

## 2. 实验设置

### 2.1 数据集

建议只在：

```text
NTU RGB+D 60
X-Sub protocol
```

上进行鲁棒性实验。

### 2.2 是否重新训练

不重新训练。

本实验是 **测试阶段扰动评估**：

```text
训练阶段：不加入随机丢弃；
测试阶段：加载已训练好的 checkpoint，并对输入 skeleton 加入随机丢弃。
```

### 2.3 参与评估模型

建议评估 4 个模型：

| Method | UE | RA | 目的 |
|---|---:|---:|---|
| PH-Att Baseline | ✗ | ✗ | 基础对照 |
| PH-Att + UE | ✓ | ✗ | 验证不确定性编码 |
| PH-Att + RA | ✗ | ✓ | 验证可靠性注意力 |
| RPT-Net | ✓ | ✓ | 完整模型 |

如果时间有限，最低限度评估：

```text
PH-Att Baseline
RPT-Net
```

---

## 3. 随机关节点丢弃策略

### 3.1 扰动定义

在测试阶段随机选择一定比例的关节点，将其坐标置零，模拟姿态估计失败、遮挡或关键点缺失。

输入数据形状：

```python
x.shape = [B, C, T, V, M]
```

其中 NTU 中：

```text
V = 25
M = 2
```

### 3.2 丢弃比例

本实验设置：

```text
Clean: 不丢弃
Drop 10%: 随机丢弃 10% 关节点
Drop 20%: 随机丢弃 20% 关节点
```

对于 NTU 的 25 个关节点：

```text
Drop 10%: ceil(25 × 0.10) = 3 个关节点
Drop 20%: ceil(25 × 0.20) = 5 个关节点
```

推荐使用：

```python
num_drop = max(1, math.ceil(V * drop_ratio))
```

### 3.3 丢弃粒度

推荐采用 **样本级固定关节丢弃**：

```text
对每个测试样本随机选择若干关节点；
这些关节点在该样本的所有帧中都被置零。
```

例如某个样本随机丢弃：

```text
left_wrist
right_elbow
left_knee
```

那么这几个关节在该样本所有 T 帧中都置零。

这样做的优点：

```text
1. 模拟持续遮挡或持续检测失败；
2. 实现简单；
3. 实验可复现；
4. 结果波动较小。
```

### 3.4 多人 skeleton 处理

推荐对所有人使用同一组丢弃关节索引：

```python
x[b, :, :, drop_idx, :] = 0
```

即：

```text
如果丢弃 left_wrist，则两个人的 left_wrist 都置零。
```

这样便于解释，也便于控制扰动强度。

---

## 4. 扰动发生位置

推荐在 **坐标层** 进行丢弃：

```python
x.shape = [B, C, T, V, M]
x[:, :, :, drop_joint_indices, :] = 0
```

原因：

```text
1. 更贴近真实姿态估计中的关节点缺失；
2. 会影响后续 part heatmap generation；
3. 能影响 UE / RA 所依赖的不确定性指标；
4. 更适合作为 skeleton robustness 实验。
```

注意：坐标置零后，`PartHeatmapGenerator` 必须能识别无效关节点，避免将 `(0, 0, 0)` 当作有效位置生成热图。

推荐规则：

```text
如果某个关节点三个坐标全为 0，则认为该关节点无效；
无效关节点不生成高斯热图。
```

---

## 5. 随机种子与公平性

### 5.1 固定随机种子

推荐：

```text
seed = 42
```

### 5.2 不同模型必须使用相同 mask

为了公平比较，所有模型在同一个测试样本、同一个 drop ratio 下必须使用相同的丢弃关节。

推荐用：

```text
global seed + sample_index
```

生成每个样本的随机 mask。

### 5.3 Dataset 返回 sample_index

建议 dataset 从：

```python
return data, label
```

改为：

```python
return data, label, index
```

这样不同模型、不同 batch size 下都可以复现相同 mask。

---

## 6. 核心函数实现

建议新增文件：

```text
utils/robustness.py
```

实现：

```python
import math
import torch

def apply_random_joint_drop_batch(x, drop_ratio=0.1, seed=42, sample_indices=None):
    """
    Args:
        x: Tensor, [B, C, T, V, M]
        drop_ratio: float, e.g. 0.1 or 0.2
        seed: int
        sample_indices: Tensor or list, [B]

    Returns:
        x_drop: Tensor, [B, C, T, V, M]
    """
    B, C, T, V, M = x.shape

    if drop_ratio <= 0:
        return x

    x_drop = x.clone()
    num_drop = max(1, math.ceil(V * drop_ratio))

    for b in range(B):
        if sample_indices is not None:
            cur_seed = seed + int(sample_indices[b])
        else:
            cur_seed = seed + b

        g = torch.Generator(device=x.device)
        g.manual_seed(cur_seed)

        perm = torch.randperm(V, generator=g, device=x.device)
        drop_idx = perm[:num_drop]

        # Drop selected joints across all channels, frames, and persons.
        x_drop[b, :, :, drop_idx, :] = 0

    return x_drop
```

---

## 7. Evaluator 接入方式

在 `test.py` 或 `engine/evaluator.py` 中加入鲁棒性评估逻辑。

### 7.1 配置文件字段

建议配置：

```yaml
robustness:
  enable: true
  drop_type: random_joint
  drop_ratios: [0.0, 0.1, 0.2]
  seed: 42
  mode: sample_fixed
  apply_stage: test_only
```

### 7.2 测试循环逻辑

```python
for drop_ratio in cfg.robustness.drop_ratios:
    acc = evaluate(
        model=model,
        dataloader=test_loader,
        drop_ratio=drop_ratio,
        seed=cfg.robustness.seed
    )
    print(f"drop_ratio={drop_ratio}, acc={acc}")
```

### 7.3 evaluator 内部逻辑

```python
for batch in dataloader:
    x, y, sample_index = batch

    if drop_ratio > 0:
        x = apply_random_joint_drop_batch(
            x,
            drop_ratio=drop_ratio,
            seed=seed,
            sample_indices=sample_index
        )

    logits = model(x)
    update_metrics(logits, y)
```

---

## 8. 输出指标

建议报告：

```text
Clean Accuracy
Drop 10% Accuracy
Drop 20% Accuracy
Drop10 Δ
Drop20 Δ
Avg Drop
```

定义：

```text
Drop10 Δ = Clean Acc - Drop10 Acc
Drop20 Δ = Clean Acc - Drop20 Acc
Avg Drop = (Drop10 Δ + Drop20 Δ) / 2
```

其中：

```text
Avg Drop 越小，表示模型鲁棒性越强。
```

---

## 9. 推荐结果表格

| Method | Clean | Drop 10% | Drop 20% | Drop10 Δ | Drop20 Δ | Avg Drop |
|---|---:|---:|---:|---:|---:|---:|
| PH-Att Baseline | - | - | - | - | - | - |
| PH-Att + UE | - | - | - | - | - | - |
| PH-Att + RA | - | - | - | - | - | - |
| RPT-Net | - | - | - | - | - | - |

论文中重点分析：

```text
1. Drop 10% 和 Drop 20% 下所有模型性能都会下降；
2. RPT-Net 的 Avg Drop 应最小；
3. PH-Att + UE 若优于 baseline，说明不确定性编码增强了低质量输入下的 token 表征；
4. PH-Att + RA 若优于 baseline，说明可靠性注意力能够降低缺失关节带来的干扰；
5. RPT-Net 若最好，说明 UE 和 RA 具有互补性。
```

---

## 10. 结果保存格式

建议每个模型保存一个 JSON 文件：

```text
outputs/robustness/ntu60_xsub/phatt_baseline.json
outputs/robustness/ntu60_xsub/phatt_ue.json
outputs/robustness/ntu60_xsub/phatt_ra.json
outputs/robustness/ntu60_xsub/rptnet_full.json
```

JSON 示例：

```json
{
  "method": "RPT-Net",
  "dataset": "NTU60",
  "protocol": "X-Sub",
  "drop_type": "random_joint_drop",
  "seed": 42,
  "results": {
    "clean": 88.50,
    "drop_10": 86.90,
    "drop_20": 84.70
  },
  "drops": {
    "drop10_delta": 1.60,
    "drop20_delta": 3.80,
    "avg_drop": 2.70
  }
}
```

---

## 11. 实验运行建议

对每个模型 checkpoint 分别运行：

```bash
python test.py --config configs/ntu60/ablation/00_phatt_baseline_xsub.yaml --robustness
python test.py --config configs/ntu60/ablation/01_phatt_ue_xsub.yaml --robustness
python test.py --config configs/ntu60/ablation/02_phatt_ra_xsub.yaml --robustness
python test.py --config configs/ntu60/ablation/03_rptnet_full_xsub.yaml --robustness
```

或者统一写一个脚本：

```text
tools/run_robustness_eval.py
```

负责循环评估多个 checkpoint 和多个 drop ratio。

---

## 12. 代码智能体任务清单

代码智能体需要完成：

```text
1. 新增 utils/robustness.py；
2. 实现 apply_random_joint_drop_batch；
3. 修改 Dataset，使其返回 sample_index；
4. 修改 evaluator，使其支持 drop_ratio；
5. 修改 test.py，使其支持 drop_ratios = [0.0, 0.1, 0.2]；
6. 确保不同模型使用相同 seed 和 sample_index；
7. 自动计算 Clean、Drop10、Drop20；
8. 自动计算 Drop10 Δ、Drop20 Δ、Avg Drop；
9. 保存 JSON 或 CSV 结果；
10. 汇总鲁棒性实验表格。
```

---

## 13. 验收标准

### 13.1 功能验收

```text
1. drop_ratio=0.0 时，结果应与 clean test 一致；
2. drop_ratio=0.1 时，每个样本丢弃 3 个关节点；
3. drop_ratio=0.2 时，每个样本丢弃 5 个关节点；
4. 不同模型对同一 sample_index 使用相同 drop_idx；
5. 输入 x 被 clone，不修改原始 batch；
6. 测试流程不触发训练或参数更新。
```

### 13.2 数值验收

```text
1. Drop 10% accuracy 通常低于 Clean；
2. Drop 20% accuracy 通常低于 Drop 10%；
3. 如果某个模型 Drop 后准确率异常升高，需要检查随机 mask 或无效点处理；
4. RPT-Net 期望具有更小 Avg Drop。
```

---

## 14. 注意事项

### 14.1 不要在训练阶段加入随机丢弃

本实验是测试阶段鲁棒性评估，不是数据增强实验。

### 14.2 不同模型必须使用相同 mask

否则模型之间的鲁棒性比较不公平。

### 14.3 不要只报告 noisy accuracy

应同时报告：

```text
Clean
Drop 10%
Drop 20%
Accuracy Drop
Avg Drop
```

### 14.4 坐标置零后的无效点处理

`PartHeatmapGenerator` 需要识别全零关节点：

```text
如果 x, y, z 坐标全为 0，则不生成对应关节点热图。
```

否则置零点可能被错误映射到热图边界或中心，影响实验结果。

---

## 15. 最终论文表述建议

可以在论文中这样描述：

```text
To evaluate the robustness of the proposed method under incomplete skeleton inputs,
we randomly drop 10% and 20% of body joints during the testing stage.
The same dropping masks are used for all compared models to ensure fair comparison.
No model is retrained under noisy inputs.
```

中文表述：

```text
为验证模型在骨架信息缺失情况下的鲁棒性，本文在测试阶段随机丢弃 10% 和 20% 的关节点，并在相同随机掩码下比较不同模型的识别性能。所有模型均不使用带噪声样本重新训练，以保证实验能够反映模型本身对低质量骨架输入的适应能力。
```

---

## 16. 总结

本鲁棒性实验推荐采用：

```text
测试阶段坐标层随机关节点丢弃；
丢弃比例为 10% 和 20%；
每个样本固定随机丢弃关节；
所有模型使用相同随机 mask；
评估 Clean、Drop 10%、Drop 20%；
报告 Accuracy 和 Avg Drop。
```

预期结论：

```text
RPT-Net 在关节点随机缺失情况下比 PH-Att Baseline 具有更小的性能下降，
说明不确定性编码与可靠性注意力能够提升模型对低质量骨架输入的鲁棒性。
```

# RPT-Net 完整训练设置（仅 Full RPT，不含 baseline/模块消融）

本文档基于 `configs/ntu60/ablation/03_rptnet_full_xsub.yaml` 与训练代码实现，整理论文可直接描述的 **完整 RPT-Net 训练配置**。

## 1. 训练入口与配置文件

- 训练脚本：`train.py`
- Full RPT 配置：`configs/ntu60/ablation/03_rptnet_full_xsub.yaml`
- 典型命令：

```bash
python train.py --config configs/ntu60/ablation/03_rptnet_full_xsub.yaml
```

## 2. 数据与输入设置（NTU60 xsub）

- 训练集：`data/ntu/processed/ntu60_xsub/train_data.npy` + `train_label.pkl`
- 验证集：`data/ntu/processed/ntu60_xsub/val_data.npy` + `val_label.pkl`
- 输入帧数：`num_frames = 64`
- batch size：`32`
- DataLoader workers：`8`
- 训练时增强开关：
  - `random_shift = true`
  - `random_mirror = true`

### 实际数据处理策略（代码行为）

- 先移除全零尾帧，再均匀采样到 64 帧。
- 若 `random_shift=true` 且原序列长于 64 帧：对均匀采样索引做小幅随机抖动（jitter）。
- 若 `random_mirror=true`：以 0.5 概率做左右镜像（x 轴取反 + 左右关节交换）。
- `drop_last=True`（仅训练集），验证集不打乱。

## 3. Full RPT 模型设置（核心）

以下均来自 Full RPT 配置：

- `name: RPTNet`
- `num_classes: 60`
- `num_frames: 64`
- `num_parts: 5`
- `heatmap_size: 32`
- `sigma: 1.5`
- `token_dim: 128`
- `depth: 4`
- `num_heads: 4`
- `mlp_ratio: 4.0`
- `dropout: 0.1`
- `attn_drop: 0.1`
- `classifier_drop: 0.5`

### Full RPT 模块开关（确保是完整模型）

- `use_uncertainty_encoding: true`
- `use_reliability_attention: true`
- `use_confidence: true`
- `use_dispersion: true`
- `use_temporal_instability: true`

> 上述组合对应注释 “RPT-Net Full (UE=on, RA=on)”，即完整 RPT 配置。

## 4. 优化与学习率策略

配置给定：

- `epochs: 100`
- `optimizer: adam`
- `lr: 1e-3`
- `weight_decay: 1e-4`
- `lr_scheduler: cosine`
- `warmup_epochs: 5`
- `min_lr: 1e-5`
- `label_smoothing: 0.1`

代码实现细节（论文中可说明）：

1. 优化器由 `build_optimizer` 构建；当配置为 `adam` 时使用 `torch.optim.Adam`。
2. 学习率调度由 `build_scheduler` 构建：
   - 主调度器：`CosineAnnealingLR(T_max=epochs-warmup, eta_min=min_lr)`；
   - 预热：若 warmup>0，先使用 `LinearLR(start_factor=0.01, total_iters=warmup)`；
   - 两者通过 `SequentialLR` 串联。
3. scheduler 在每个 epoch 训练结束后 `step()`（按 epoch 更新）。

## 5. 损失函数与训练机制

- 损失函数：`CrossEntropyLoss(label_smoothing=0.1)`。
- 默认精度：FP32。
- 可选 AMP：通过 `--amp` 开启，使用 `torch.amp.autocast('cuda')` + `GradScaler`。
- 每个 iteration 执行：`zero_grad -> backward -> optimizer.step`。

## 6. 评估、保存与日志

- 每个 epoch 后都会在验证集评估（Top-1 accuracy + loss）。
- 当验证集 Top-1 刷新最佳时保存 `best.pth`。
- 每 `save_freq=10` 个 epoch 额外保存一次 `epoch_k.pth`，最后一轮必保存。
- 日志打印频率：`print_freq=20`。
- 默认开启 TensorBoard（若环境可用）：记录 train/val loss、acc 与 lr。

## 7. 设备与复现设置

- 配置 GPU 列表：`[0,1,2,3]`。
- 若检测到多卡，则使用 `nn.DataParallel`。
- 随机种子：`seed=42`（由 `set_seed` 在训练开始时设置）。

## 8. 可直接放论文的方法段落（示例）

我们在 NTU RGB+D 60 的 cross-subject 划分上训练完整 RPT-Net。输入骨架序列先去除无效零帧，再均匀采样为 64 帧；训练阶段使用随机时间抖动与左右镜像增强。模型采用 5-part 表示，token 维度为 128，Transformer 深度 4、head 数 4，dropout/attention dropout 分别为 0.1，分类头 dropout 为 0.5。完整模型中不确定性编码与可靠性注意力均启用，且三类不确定性指标（confidence、dispersion、temporal instability）全部开启。优化器为 Adam（初始学习率 1e-3，weight decay 1e-4），训练 100 epoch；学习率策略为 5 epoch 线性 warmup（起始因子 0.01）后接余弦退火至 1e-5。损失函数为带 0.1 label smoothing 的交叉熵。我们在每个 epoch 后进行验证，并依据验证集 Top-1 准确率保存最优模型。

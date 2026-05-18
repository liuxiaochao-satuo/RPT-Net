"""
Shared CNN Encoder: extracts spatial features from each part heatmap.

Input:  [B*T*P, 1, H, W]
Output: [B*T*P, D]
"""

import torch
import torch.nn as nn


class SharedCNNEncoder(nn.Module):
    """Lightweight 2D CNN shared across all body parts.

    3-layer conv with stride-based downsampling + adaptive avg pool.
    """

    def __init__(self, in_channels=1, channels=(32, 64, 128), out_dim=128):
        super().__init__()

        layers = []
        ch_in = in_channels
        strides = [1, 2, 2]
        for i, ch_out in enumerate(channels):
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3,
                          stride=strides[i], padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = None
        if channels[-1] != out_dim:
            self.proj = nn.Linear(channels[-1], out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [N, 1, H, W] where N = B*T*P

        Returns:
            tokens: [N, D]
        """
        x = self.features(x)    # [N, C_last, H', W']
        x = self.pool(x)        # [N, C_last, 1, 1]
        x = x.flatten(1)        # [N, C_last]
        if self.proj is not None:
            x = self.proj(x)    # [N, D]
        return x

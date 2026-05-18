"""
Spatial-Temporal Attention Block: one layer of spatial attention,
followed by temporal attention, followed by FFN, all with residual connections.

Input:  [B, T, P, D]
Output: [B, T, P, D]
"""

import torch
import torch.nn as nn
from models.modules.spatial_attention import SpatialAttention
from models.modules.temporal_attention import TemporalAttention


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class STAttentionBlock(nn.Module):
    """Single spatial-temporal attention block.

    Structure:
        x = x + SpatialAttn(Norm(x))
        x = x + TemporalAttn(Norm(x))
        x = x + FFN(Norm(x))
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0,
                 dropout=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=dropout,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, reliability=None, return_attn=False):
        """
        Args:
            x: [B, T, P, D]
            reliability: optional [B, T, P] for reliability attention
            return_attn: whether to collect attention maps

        Returns:
            x: [B, T, P, D]
            attn_dict: dict with spatial/temporal attention if return_attn
        """
        attn_dict = {}

        # spatial attention
        if return_attn:
            sa_out, sa_weights = self.spatial_attn(
                self.norm1(x), reliability=reliability, return_attn=True)
            attn_dict['spatial'] = sa_weights
        else:
            sa_out = self.spatial_attn(self.norm1(x), reliability=reliability)
        x = x + sa_out

        # temporal attention
        if return_attn:
            ta_out, ta_weights = self.temporal_attn(
                self.norm2(x), reliability=reliability, return_attn=True)
            attn_dict['temporal'] = ta_weights
        else:
            ta_out = self.temporal_attn(self.norm2(x), reliability=reliability)
        x = x + ta_out

        # feed-forward
        x = x + self.ffn(self.norm3(x))

        if return_attn:
            return x, attn_dict
        return x

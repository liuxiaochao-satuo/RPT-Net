"""
Temporal Multi-Head Self-Attention: models dynamics of each body part
across time frames.

Input:  [B, T, P, D]
Output: [B, T, P, D]
"""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Multi-head self-attention over time frames (temporal dimension).

    Reshapes [B, T, P, D] → [B*P, T, D], applies MHSA, reshapes back.
    Supports returning attention weights for visualization.
    """

    def __init__(self, dim, num_heads=4, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, reliability=None, return_attn=False):
        """
        Args:
            x: [B, T, P, D]
            reliability: optional [B, T, P] bias for attention (reserved for RA)
            return_attn: whether to return attention weights

        Returns:
            out: [B, T, P, D]
            attn_weights: [B*P, num_heads, T, T] if return_attn else None
        """
        B, T, P, D = x.shape
        # reshape: group by part → [B*P, T, D]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * P, T, D)

        qkv = self.qkv(x_flat).reshape(B * P, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*P, H, T, d]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*P, H, T, T]

        if reliability is not None:
            # reliability bias: [B, T, P] → [B*P, 1, 1, T] broadcast to key dim
            r_bias = reliability.permute(0, 2, 1).reshape(B * P, 1, 1, T)
            attn = attn + r_bias

        attn_weights = attn.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B * P, T, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        # reshape back: [B*P, T, D] → [B, P, T, D] → [B, T, P, D]
        out = out.reshape(B, P, T, D).permute(0, 2, 1, 3)

        if return_attn:
            return out, attn_weights
        return out

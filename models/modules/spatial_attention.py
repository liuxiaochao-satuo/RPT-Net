"""
Spatial Multi-Head Self-Attention: models relationships between body parts
within the same frame.

Input:  [B, T, P, D]
Output: [B, T, P, D]
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Multi-head self-attention over body parts (spatial dimension).

    Reshapes [B, T, P, D] → [B*T, P, D], applies MHSA, reshapes back.
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
            attn_weights: [B*T, num_heads, P, P] if return_attn else None
        """
        B, T, P, D = x.shape
        x_flat = x.reshape(B * T, P, D)

        qkv = self.qkv(x_flat).reshape(B * T, P, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*T, H, P, d]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*T, H, P, P]

        if reliability is not None:
            # log-space reliability bias: maps (0,1) → (-inf, 0]
            # low reliability → large negative bias → suppressed attention
            r_bias = torch.log(reliability.clamp(min=1e-6))
            r_bias = r_bias.reshape(B * T, 1, 1, P)  # broadcast to key dim
            attn = attn + r_bias

        attn_weights = attn.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B * T, P, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, T, P, D)

        if return_attn:
            return out, attn_weights
        return out

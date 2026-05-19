"""
Uncertainty Encoder: fuses uncertainty information into part tokens.

Input:  part_tokens [B, T, P, D], uncertainty [B, T, P, K]
Output: enhanced_tokens [B, T, P, D]
"""

import torch
import torch.nn as nn


class UncertaintyEncoder(nn.Module):
    """Projects uncertainty indicators and fuses them into part tokens via residual."""

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
        """
        Args:
            tokens: [B, T, P, D]
            uncertainty: [B, T, P, K]

        Returns:
            enhanced_tokens: [B, T, P, D]
        """
        u_proj = self.uncertainty_proj(uncertainty)  # [B, T, P, D]
        fused = self.fusion(torch.cat([tokens, u_proj], dim=-1))  # [B, T, P, D]
        return tokens + fused

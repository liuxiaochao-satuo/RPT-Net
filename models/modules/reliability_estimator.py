"""
Reliability Estimator: converts uncertainty indicators to a single reliability score.

Input:  uncertainty [B, T, P, K]
Output: reliability [B, T, P]  in (0, 1)
"""

import torch
import torch.nn as nn


class ReliabilityEstimator(nn.Module):
    """Learned mapping from uncertainty indicators to reliability score."""

    def __init__(self, uncertainty_dim=3, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(uncertainty_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, uncertainty):
        """
        Args:
            uncertainty: [B, T, P, K]

        Returns:
            reliability: [B, T, P] in (0, 1)
        """
        return self.mlp(uncertainty).squeeze(-1)

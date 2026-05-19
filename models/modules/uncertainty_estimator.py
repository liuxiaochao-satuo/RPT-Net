"""
Uncertainty Estimator: computes per-part uncertainty indicators from heatmaps.

Indicators:
  - confidence: raw heatmap peak strength (before normalization)
  - dispersion: weighted spatial variance of heatmap response
  - temporal_instability: frame-to-frame center displacement

Input:  part_heatmaps [B, T, P, H, W], raw_peaks [B, T, P]
Output: uncertainty [B, T, P, K]  where K = number of enabled indicators
"""

import torch
import torch.nn as nn


class UncertaintyEstimator(nn.Module):

    def __init__(self, use_confidence=True, use_dispersion=True,
                 use_temporal_instability=True, eps=1e-6):
        super().__init__()
        self.use_confidence = use_confidence
        self.use_dispersion = use_dispersion
        self.use_temporal_instability = use_temporal_instability
        self.eps = eps

        self.num_indicators = sum([
            use_confidence, use_dispersion, use_temporal_instability])

    def forward(self, part_heatmaps, raw_peaks):
        """
        Args:
            part_heatmaps: [B, T, P, H, W] (normalized)
            raw_peaks: [B, T, P] (pre-normalization peak values)

        Returns:
            uncertainty: [B, T, P, K]
        """
        features = []

        if self.use_confidence:
            conf = self._compute_confidence(raw_peaks)
            features.append(conf)

        if self.use_dispersion:
            disp = self._compute_dispersion(part_heatmaps)
            features.append(disp)

        if self.use_temporal_instability:
            inst = self._compute_temporal_instability(part_heatmaps)
            features.append(inst)

        uncertainty = torch.stack(features, dim=-1)  # [B, T, P, K]
        return uncertainty

    def _compute_confidence(self, raw_peaks):
        """Confidence from pre-normalization peak values.

        Higher peak = stronger response = more reliable.
        Normalize to [0, 1] per sample for stability.
        """
        # raw_peaks: [B, T, P]
        B = raw_peaks.shape[0]
        p_min = raw_peaks.reshape(B, -1).min(dim=1).values.reshape(B, 1, 1)
        p_max = raw_peaks.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1)
        confidence = (raw_peaks - p_min) / (p_max - p_min + self.eps)
        return confidence  # [B, T, P]

    def _compute_dispersion(self, part_heatmaps):
        """Weighted spatial variance of heatmap response.

        Higher dispersion = more spread out = less reliable.
        """
        B, T, P, H, W = part_heatmaps.shape
        device = part_heatmaps.device

        ys = torch.linspace(0, 1, H, device=device).reshape(1, 1, 1, H, 1)
        xs = torch.linspace(0, 1, W, device=device).reshape(1, 1, 1, 1, W)

        weights = part_heatmaps
        norm = weights.sum(dim=(-2, -1), keepdim=True) + self.eps

        mu_x = (weights * xs).sum(dim=(-2, -1), keepdim=True) / norm
        mu_y = (weights * ys).sum(dim=(-2, -1), keepdim=True) / norm

        dist_sq = (xs - mu_x) ** 2 + (ys - mu_y) ** 2
        dispersion = (weights * dist_sq).sum(dim=(-2, -1)) / norm.squeeze(-1).squeeze(-1)

        return dispersion  # [B, T, P]

    def _compute_temporal_instability(self, part_heatmaps):
        """Frame-to-frame displacement of heatmap center.

        Higher instability = more jitter = less reliable.
        """
        centers = self._compute_centers(part_heatmaps)  # [B, T, P, 2]
        delta = torch.norm(centers[:, 1:] - centers[:, :-1], dim=-1)  # [B, T-1, P]
        first = torch.zeros_like(delta[:, :1])
        instability = torch.cat([first, delta], dim=1)  # [B, T, P]
        return instability

    def _compute_centers(self, part_heatmaps):
        """Weighted center (centroid) of each part heatmap."""
        B, T, P, H, W = part_heatmaps.shape
        device = part_heatmaps.device

        ys = torch.linspace(0, 1, H, device=device).reshape(1, 1, 1, H, 1)
        xs = torch.linspace(0, 1, W, device=device).reshape(1, 1, 1, 1, W)

        weights = part_heatmaps
        norm = weights.sum(dim=(-2, -1), keepdim=True) + self.eps

        mu_x = (weights * xs).sum(dim=(-2, -1)) / norm.squeeze(-1).squeeze(-1)
        mu_y = (weights * ys).sum(dim=(-2, -1)) / norm.squeeze(-1).squeeze(-1)

        centers = torch.stack([mu_x, mu_y], dim=-1)  # [B, T, P, 2]
        return centers

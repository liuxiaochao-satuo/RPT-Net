"""
Part Heatmap Generator: converts skeleton coordinates to part-level 2D heatmaps.

Input:  skeleton [B, C, T, V, M]  (C=3: x,y,z)
Output: part_heatmaps [B, T, P, H, W]
"""

import torch
import torch.nn as nn

# NTU 25-joint body part layout (0-indexed)
DEFAULT_PARTS = {
    "torso":     [0, 1, 2, 3, 20],
    "left_arm":  [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_leg":  [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19],
}


class PartHeatmapGenerator(nn.Module):
    """Generate part-level 2D Gaussian heatmaps from skeleton coordinates.

    Uses x-y plane coordinates, normalizes per-sample, and renders
    Gaussian blobs for each joint, then fuses within each body part.
    """

    def __init__(self, heatmap_size=32, sigma=1.5, parts=None):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        if parts is None:
            parts = DEFAULT_PARTS
        self.part_names = list(parts.keys())
        self.num_parts = len(self.part_names)

        # store joint indices per part as a list of tensors
        self.part_indices = []
        for name in self.part_names:
            self.part_indices.append(parts[name])

        # pre-compute coordinate grids (registered as buffer for device transfer)
        coords = torch.arange(heatmap_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('grid_x', grid_x)  # [H, W]
        self.register_buffer('grid_y', grid_y)  # [H, W]

    def forward(self, skeleton):
        """
        Args:
            skeleton: [B, C, T, V, M] where C >= 2 (x, y, ...)

        Returns:
            part_heatmaps: [B, T, P, H, W]
        """
        B, C, T, V, M = skeleton.shape
        H = W = self.heatmap_size

        # extract x, y coordinates: [B, T, V, M]
        x_coord = skeleton[:, 0]  # [B, T, V, M]
        y_coord = skeleton[:, 1]  # [B, T, V, M]

        # build validity mask: a joint is valid if its coordinates are non-zero
        valid = (x_coord.abs() + y_coord.abs()) > 1e-6  # [B, T, V, M]

        # per-sample min-max normalization using only valid joints
        # replace invalid joints with inf/−inf so they don't affect min/max
        x_for_min = x_coord.clone()
        y_for_min = y_coord.clone()
        x_for_min[~valid] = float('inf')
        y_for_min[~valid] = float('inf')
        x_for_max = x_coord.clone()
        y_for_max = y_coord.clone()
        x_for_max[~valid] = float('-inf')
        y_for_max[~valid] = float('-inf')

        # min/max over T, V, M dims → [B, 1, 1, 1]
        x_min = x_for_min.reshape(B, -1).min(dim=1).values.reshape(B, 1, 1, 1)
        x_max = x_for_max.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1, 1)
        y_min = y_for_min.reshape(B, -1).min(dim=1).values.reshape(B, 1, 1, 1)
        y_max = y_for_max.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1, 1)

        eps = 1e-6
        x_norm = (x_coord - x_min) / (x_max - x_min + eps)  # [B, T, V, M] in [0, 1]
        y_norm = (y_coord - y_min) / (y_max - y_min + eps)

        # map to heatmap pixel coordinates
        u = x_norm * (W - 1)  # [B, T, V, M]
        v = y_norm * (H - 1)  # [B, T, V, M]

        # generate per-joint Gaussian heatmaps (vectorized)
        # u, v: [B, T, V, M] → [B, T, V, M, 1, 1]
        u = u.unsqueeze(-1).unsqueeze(-1)  # [B, T, V, M, 1, 1]
        v = v.unsqueeze(-1).unsqueeze(-1)

        # grid_x, grid_y: [H, W] → broadcast with [B, T, V, M, H, W]
        dist_sq = (self.grid_x - u) ** 2 + (self.grid_y - v) ** 2
        joint_heatmaps = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # [B, T, V, M, H, W]

        # zero out invalid joints
        valid_mask = valid.unsqueeze(-1).unsqueeze(-1)  # [B, T, V, M, 1, 1]
        joint_heatmaps = joint_heatmaps * valid_mask

        # multi-person fusion: max over M dimension
        joint_heatmaps, _ = joint_heatmaps.max(dim=3)  # [B, T, V, H, W]

        # part-level fusion: sum joints within each part, then normalize
        part_heatmaps = torch.zeros(B, T, self.num_parts, H, W,
                                    device=skeleton.device, dtype=skeleton.dtype)

        for p_idx, joint_indices in enumerate(self.part_indices):
            part_map = joint_heatmaps[:, :, joint_indices].sum(dim=2)  # [B, T, H, W]
            # normalize to [0, 1] per part per frame
            p_max = part_map.reshape(B, T, -1).max(dim=-1).values.reshape(B, T, 1, 1)
            part_map = part_map / (p_max + eps)
            part_heatmaps[:, :, p_idx] = part_map

        return part_heatmaps

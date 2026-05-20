"""Robustness evaluation utilities: random joint drop for test-time perturbation."""

import math
import torch


def apply_random_joint_drop_batch(x, drop_ratio=0.1, seed=42, sample_indices=None):
    """Apply random joint drop to a batch of skeleton data.

    For each sample, randomly selects joints to zero out across all frames
    and all persons. Uses deterministic per-sample seeding for reproducibility.

    Args:
        x: [B, C, T, V, M] skeleton tensor
        drop_ratio: fraction of joints to drop (e.g. 0.1 = 10%)
        seed: base random seed
        sample_indices: [B] tensor/list of dataset indices for reproducible masks

    Returns:
        x_drop: [B, C, T, V, M] with selected joints zeroed out
    """
    if drop_ratio <= 0:
        return x

    B, C, T, V, M = x.shape
    x_drop = x.clone()
    num_drop = max(1, math.ceil(V * drop_ratio))

    for b in range(B):
        cur_seed = seed + int(sample_indices[b]) if sample_indices is not None else seed + b
        g = torch.Generator(device='cpu')
        g.manual_seed(cur_seed)
        perm = torch.randperm(V, generator=g)
        drop_idx = perm[:num_drop]
        x_drop[b, :, :, drop_idx, :] = 0

    return x_drop

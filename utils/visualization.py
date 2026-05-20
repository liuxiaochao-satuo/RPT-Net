"""Visualization utilities for attention and reliability heatmaps."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

PART_NAMES = ['Torso', 'L-Arm', 'R-Arm', 'L-Leg', 'R-Leg']

# NTU skeleton parent-child pairs (0-indexed), from ntu_dataset.py
NTU_BONE_PAIRS = [
    (0, 0), (1, 0), (2, 20), (3, 2), (4, 20),
    (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
    (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
    (20, 1), (21, 7), (22, 7), (23, 11), (24, 11),
]

# Joint index -> part index for coloring
JOINT_TO_PART = {}
PART_JOINTS = {
    0: [0, 1, 2, 3, 20],
    1: [4, 5, 6, 7, 21, 22],
    2: [8, 9, 10, 11, 23, 24],
    3: [12, 13, 14, 15],
    4: [16, 17, 18, 19],
}
for part_idx, joints in PART_JOINTS.items():
    for joint in joints:
        JOINT_TO_PART[joint] = part_idx

PART_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']


def _normalize_map(arr):
    """Min-max normalize to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def aggregate_tp_attention(attention_maps, batch_idx=0, layer='last'):
    """Aggregate spatial/temporal attention weights into [T, P] map.

    Args:
        attention_maps: list of dicts with 'spatial' and 'temporal' tensors
        batch_idx: which sample in the batch
        layer: 'last' or 'mean' across ST blocks

    Returns:
        np.ndarray [T, P] normalized attention importance
    """
    if layer == 'last':
        layers = [attention_maps[-1]]
    elif layer == 'mean':
        layers = attention_maps
    else:
        raise ValueError(f"Unknown layer mode: {layer}")

    spatial_maps = []
    temporal_maps = []

    for attn_dict in layers:
        spatial = attn_dict['spatial'].detach().cpu()  # [B*T, H, P, P]
        temporal = attn_dict['temporal'].detach().cpu()  # [B*P, H, T, T]

        _, num_heads, p, _ = spatial.shape
        bp, _, t, _ = temporal.shape
        b = bp // p

        # spatial: importance of part p at time t = mean attention received as key
        spatial = spatial.reshape(b, t, num_heads, p, p)
        spatial_tp = spatial[batch_idx].mean(dim=1).mean(dim=1)  # [T, P]
        spatial_maps.append(spatial_tp)

        # temporal: importance at time t for part p = mean attention received at t
        temporal = temporal.reshape(b, p, num_heads, t, t)
        temporal_tp = temporal[batch_idx].mean(dim=1).mean(dim=1).transpose(0, 1)  # [T, P]
        temporal_maps.append(temporal_tp)

    spatial_tp = torch.stack(spatial_maps, dim=0).mean(dim=0).numpy()
    temporal_tp = torch.stack(temporal_maps, dim=0).mean(dim=0).numpy()
    combined = (spatial_tp + temporal_tp) / 2.0
    return _normalize_map(combined)


def plot_tp_heatmap(ax, tp_map, title, cmap='YlOrRd', vmin=0.0, vmax=1.0,
                    show_ylabel=True, time_stride=1):
    """Plot a T x P heatmap on the given axes."""
    data = np.asarray(tp_map)[::time_stride]
    im = ax.imshow(
        data, aspect='auto', origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax, interpolation='nearest',
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(PART_NAMES)))
    ax.set_xticklabels(PART_NAMES, rotation=30, ha='right', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Time frame', fontsize=10)
    else:
        ax.set_yticks([])
    return im


def _draw_skeleton_frame(ax, skeleton_xy, frame_idx, person_idx=0):
    """Draw one skeleton frame on x-y plane."""
    # skeleton_xy: [T, V, 2]
    joints = skeleton_xy[frame_idx]

    for child, parent in NTU_BONE_PAIRS:
        if child == parent:
            continue
        x_vals = [joints[child, 0], joints[parent, 0]]
        y_vals = [joints[child, 1], joints[parent, 1]]
        ax.plot(x_vals, y_vals, color='#555555', linewidth=1.5, zorder=1)

    for joint_idx, (x, y) in enumerate(joints):
        part_idx = JOINT_TO_PART.get(joint_idx, 0)
        ax.scatter(
            x, y, s=18, color=PART_COLORS[part_idx],
            edgecolors='white', linewidths=0.4, zorder=2,
        )

    ax.set_aspect('equal')
    ax.axis('off')


def extract_skeleton_xy(skeleton_tensor, person_idx=0):
    """Extract x-y coordinates from skeleton tensor [C, T, V, M]."""
    data = skeleton_tensor.detach().cpu().numpy()
    x_coord = data[0, :, :, person_idx]
    y_coord = data[1, :, :, person_idx]
    return np.stack([x_coord, y_coord], axis=-1)  # [T, V, 2]


def plot_skeleton_keyframes(ax, skeleton_tensor, num_keyframes=5, person_idx=0):
    """Plot evenly spaced skeleton keyframes in a horizontal strip."""
    skeleton_xy = extract_skeleton_xy(skeleton_tensor, person_idx=person_idx)
    t = skeleton_xy.shape[0]
    if num_keyframes <= 1:
        frame_indices = [0]
    else:
        frame_indices = np.linspace(0, t - 1, num_keyframes, dtype=int)

    ax.set_title('Skeleton Keyframes', fontsize=11)
    ax.axis('off')

    n = len(frame_indices)
    for i, frame_idx in enumerate(frame_indices):
        left = 0.02 + i * (0.96 / n)
        width = 0.96 / n - 0.01
        inset = ax.inset_axes([left, 0.08, width, 0.84])
        _draw_skeleton_frame(inset, skeleton_xy, frame_idx, person_idx=person_idx)
        inset.set_title(f't={frame_idx}', fontsize=8, pad=2)


def save_figure_panel(
    skeleton_tensor,
    standard_tp,
    reliability_tp,
    reliability_score,
    save_path,
    action_name='',
    sample_name='',
    time_stride=1,
):
    """Save 4-column panel: skeleton | standard attn | RA attn | reliability."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(
        1, 4, figsize=(14, 4.5),
        gridspec_kw={'width_ratios': [1.4, 1.0, 1.0, 1.0]},
    )

    title = action_name.replace('_', ' ').title()
    if sample_name:
        title = f'{title} ({sample_name})'
    fig.suptitle(title, fontsize=13, y=1.02)

    plot_skeleton_keyframes(axes[0], skeleton_tensor)
    plot_tp_heatmap(
        axes[1], standard_tp, 'Standard Attention',
        show_ylabel=True, time_stride=time_stride,
    )
    im2 = plot_tp_heatmap(
        axes[2], reliability_tp, 'Reliability Attention',
        show_ylabel=False, time_stride=time_stride,
    )
    im3 = plot_tp_heatmap(
        axes[3], reliability_score, 'Reliability Score',
        cmap='Blues', show_ylabel=False, time_stride=time_stride,
    )

    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

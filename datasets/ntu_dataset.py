"""
NTU RGB+D Dataset for skeleton-based action recognition.

Loads preprocessed [N, C, T, V, M] .npy data and corresponding labels.
Supports temporal sampling, random crop, and data augmentation at training time.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class NTUDataset(Dataset):
    """NTU RGB+D skeleton dataset.

    Args:
        data_path: Path to {split}_data.npy
        label_path: Path to {split}_label.pkl
        num_frames: Number of frames to sample per clip
        split: 'train' or 'val'
        random_shift: Whether to use random temporal shift (train only)
        random_mirror: Whether to randomly mirror left/right joints (train only)
        bone: Whether to also compute bone features (joint differences)
        noise_drop_ratio: Ratio of joints to randomly zero out (for robustness experiments)
    """

    # NTU skeleton left-right joint pairs for mirroring
    MIRROR_PAIRS = [
        (1, 1), (2, 2), (3, 3), (4, 4),  # spine chain (no swap)
        (5, 9), (6, 10), (7, 11), (8, 12),  # arms
        (13, 17), (14, 18), (15, 19), (16, 20),  # legs
        (21, 21), (22, 22), (23, 24), (25, 25),  # extra
    ]

    # NTU 25 joints: mapping from 1-indexed to 0-indexed swap pairs
    SWAP_INDICES = None  # built lazily

    def __init__(
        self,
        data_path,
        label_path,
        num_frames=64,
        split='train',
        random_shift=True,
        random_mirror=True,
        bone=False,
        noise_drop_ratio=0.0,
    ):
        self.num_frames = num_frames
        self.split = split
        self.random_shift = random_shift and (split == 'train')
        self.random_mirror = random_mirror and (split == 'train')
        self.bone = bone
        self.noise_drop_ratio = noise_drop_ratio

        self.data = np.load(data_path, mmap_mode='r')  # [N, C, T, V, M]
        with open(label_path, 'rb') as f:
            self.sample_names, self.labels = pickle.load(f)

        self._build_swap_indices()

    def _build_swap_indices(self):
        """Build left-right swap index array for mirroring."""
        if NTUDataset.SWAP_INDICES is not None:
            return
        swap = list(range(25))
        for l, r in self.MIRROR_PAIRS:
            li, ri = l - 1, r - 1
            swap[li] = ri
            swap[ri] = li
        NTUDataset.SWAP_INDICES = np.array(swap)

    def __len__(self):
        return len(self.labels)

    def _temporal_sample(self, data_numpy):
        """Uniform temporal sampling to num_frames.

        1. Strip trailing zero-frames (keep only valid portion)
        2. Uniformly sample to exactly num_frames
        3. During training, optionally add small random temporal perturbation

        data_numpy: [C, T, V, M]
        """
        C, T, V, M = data_numpy.shape

        # strip zero-frames: keep only the valid portion
        valid_mask = np.abs(data_numpy).sum(axis=(0, 2, 3)) > 0  # [T]
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return np.zeros((C, self.num_frames, V, M), dtype=np.float32)

        data_numpy = data_numpy[:, valid_indices]
        actual_len = data_numpy.shape[1]

        # uniform sampling: works for both longer and shorter sequences
        if actual_len == self.num_frames:
            return data_numpy

        interval = actual_len / self.num_frames
        uniform_indices = [int(i * interval) for i in range(self.num_frames)]

        if self.random_shift and actual_len > self.num_frames:
            # small random jitter: each index can shift by +-1 within valid range
            max_jitter = max(1, int(interval * 0.5))
            jittered = []
            for idx in uniform_indices:
                offset = np.random.randint(-max_jitter, max_jitter + 1)
                jittered.append(np.clip(idx + offset, 0, actual_len - 1))
            return data_numpy[:, jittered]

        return data_numpy[:, uniform_indices]

    def _mirror(self, data_numpy):
        """Randomly mirror left-right joints with x-axis flip."""
        if np.random.random() > 0.5:
            return data_numpy
        # flip x coordinate
        data_numpy[0] = -data_numpy[0]
        # swap left-right joints
        data_numpy = data_numpy[:, :, self.SWAP_INDICES]
        return data_numpy

    def _noise_drop(self, data_numpy):
        """Randomly zero out joints for robustness experiments."""
        if self.noise_drop_ratio <= 0:
            return data_numpy
        C, T, V, M = data_numpy.shape
        mask = np.random.random((T, V)) > self.noise_drop_ratio
        mask = mask.reshape(1, T, V, 1).astype(np.float32)
        return data_numpy * mask

    def _compute_bone(self, data_numpy):
        """Compute bone features: differences between connected joints."""
        # NTU skeleton bone connections (child → parent, 0-indexed)
        bone_pairs = [
            (0, 0), (1, 0), (2, 20), (3, 2), (4, 20),
            (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
            (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
            (20, 1), (21, 7), (22, 7), (23, 11), (24, 11),
        ]
        bone = np.zeros_like(data_numpy)
        for child, parent in bone_pairs:
            bone[:, :, child] = data_numpy[:, :, child] - data_numpy[:, :, parent]
        return bone

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])  # [C, T, V, M]
        label = self.labels[index]

        # temporal sampling
        data_numpy = self._temporal_sample(data_numpy)

        # augmentations (train only)
        if self.split == 'train':
            if self.random_mirror:
                data_numpy = self._mirror(data_numpy)
            data_numpy = self._noise_drop(data_numpy)

        if self.bone:
            bone = self._compute_bone(data_numpy)
            data_numpy = np.concatenate([data_numpy, bone], axis=0)  # [2C, T, V, M]

        return torch.from_numpy(data_numpy.astype(np.float32)), label, index


def build_dataloader(cfg, split='train'):
    """Build dataloader from config dict.

    Args:
        cfg: dict with keys 'data_path', 'label_path', 'num_frames',
             'batch_size', 'num_workers', etc.
        split: 'train' or 'val'
    """
    from torch.utils.data import DataLoader

    dataset = NTUDataset(
        data_path=cfg[f'{split}_data'],
        label_path=cfg[f'{split}_label'],
        num_frames=cfg.get('num_frames', 64),
        split=split,
        random_shift=cfg.get('random_shift', True),
        random_mirror=cfg.get('random_mirror', True),
        bone=cfg.get('bone', False),
        noise_drop_ratio=cfg.get('noise_drop_ratio', 0.0),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=(split == 'train'),
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return loader

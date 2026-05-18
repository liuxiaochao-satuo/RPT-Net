"""
Skeleton data augmentation transforms for training.

All transforms operate on numpy arrays with shape [C, T, V, M].
"""

import numpy as np


class RandomRotation:
    """Random rotation around y-axis (vertical) by a small angle."""

    def __init__(self, max_angle=15.0):
        self.max_angle = max_angle

    def __call__(self, data):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=np.float32)
        C, T, V, M = data.shape
        data_reshaped = data[:3].transpose(1, 2, 3, 0).reshape(-1, 3)  # [T*V*M, 3]
        data_rotated = data_reshaped @ rot.T
        data[:3] = data_rotated.reshape(T, V, M, 3).transpose(3, 0, 1, 2)
        return data


class RandomScale:
    """Random uniform scaling."""

    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, data):
        scale = np.random.uniform(self.low, self.high)
        data[:3] *= scale
        return data


class RandomShift:
    """Random spatial translation."""

    def __init__(self, max_shift=0.05):
        self.max_shift = max_shift

    def __call__(self, data):
        shift = np.random.uniform(-self.max_shift, self.max_shift, size=(3, 1, 1, 1))
        data[:3] += shift.astype(np.float32)
        return data


class RandomGaussianNoise:
    """Add Gaussian noise to joint coordinates."""

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        noise = np.random.randn(*data[:3].shape).astype(np.float32) * self.std
        mask = np.abs(data[:3]).sum(0, keepdims=True) > 0
        data[:3] += noise * mask
        return data


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def get_train_transforms():
    return Compose([
        RandomRotation(max_angle=10.0),
        RandomScale(low=0.95, high=1.05),
        RandomShift(max_shift=0.03),
        RandomGaussianNoise(std=0.005),
    ])

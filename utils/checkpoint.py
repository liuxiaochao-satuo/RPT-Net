"""Checkpoint save/load utilities."""

import os
import torch


def save_checkpoint(state, work_dir, filename='checkpoint.pth'):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return the epoch and best accuracy."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', 0)
    best_acc = ckpt.get('best_acc', 0.0)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return start_epoch, best_acc

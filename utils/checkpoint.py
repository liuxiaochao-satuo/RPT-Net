"""Checkpoint save/load utilities."""

import os
import torch


def save_checkpoint(state, work_dir, filename='checkpoint.pth'):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, filename)
    torch.save(state, path)
    return path


def _adapt_state_dict(model, state_dict):
    """Align checkpoint keys with model (handles DataParallel module. prefix)."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys == ckpt_keys:
        return state_dict

    model_has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)

    if model_has_module and not ckpt_has_module:
        return {f'module.{k}': v for k, v in state_dict.items()}
    if ckpt_has_module and not model_has_module:
        return {k[len('module.'):]: v for k, v in state_dict.items() if k.startswith('module.')}
    return state_dict


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return the epoch and best accuracy."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = _adapt_state_dict(model, ckpt['model'])
    model.load_state_dict(state_dict)
    start_epoch = ckpt.get('epoch', 0)
    best_acc = ckpt.get('best_acc', 0.0)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return start_epoch, best_acc

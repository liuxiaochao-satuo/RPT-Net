"""
RPT-Net / PH-Att Testing Script.

Usage:
    python test.py --config configs/ntu60/phatt_baseline_xsub.yaml --checkpoint outputs/.../best.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint
from datasets.ntu_dataset import build_dataloader
from models.phatt_baseline import build_phatt_baseline
from engine.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description='RPT-Net Testing')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))

    work_dir = cfg.get('work_dir', 'outputs/default')
    logger = setup_logger(work_dir, name='test')
    logger.info(f'Config: {args.config}')
    logger.info(f'Checkpoint: {args.checkpoint}')

    gpu_ids = cfg.device.get('gpu', [0])
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # data
    val_loader = build_dataloader(cfg.data, split='val')
    logger.info(f'Val: {len(val_loader.dataset)} samples')

    # model
    model_cfg = dict(cfg.model)
    model = build_phatt_baseline(model_cfg)
    model = model.to(device)

    # load weights
    _, best_acc = load_checkpoint(args.checkpoint, model)
    logger.info(f'Loaded checkpoint (reported best_acc={best_acc:.2f}%)')

    # evaluate
    criterion = nn.CrossEntropyLoss()
    val_stats = evaluate(model, val_loader, criterion, device, logger)
    logger.info(f'Test Acc@1: {val_stats["top1"]:.2f}%')


if __name__ == '__main__':
    main()

"""
RPT-Net / PH-Att Training Script.

Usage:
    python train.py --config configs/ntu60/phatt_baseline_xsub.yaml
    python train.py --config configs/ntu60/phatt_baseline_xsub.yaml --resume outputs/.../best.pth
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from datasets.ntu_dataset import build_dataloader
from models.phatt_baseline import build_phatt_baseline
from engine.trainer import train_one_epoch, evaluate


def build_optimizer(model, cfg):
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = cfg.training.get('optimizer', 'adam').lower()
    lr = cfg.training.lr
    wd = cfg.training.get('weight_decay', 1e-4)

    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=wd, nesterov=True)
    else:
        raise ValueError(f'Unknown optimizer: {name}')


def build_scheduler(optimizer, cfg):
    name = cfg.training.get('lr_scheduler', 'cosine')
    epochs = cfg.training.epochs
    warmup = cfg.training.get('warmup_epochs', 5)
    min_lr = cfg.training.get('min_lr', 1e-5)

    if name == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=min_lr)
    elif name == 'step':
        milestones = cfg.training.get('milestones', [60, 80])
        gamma = cfg.training.get('gamma', 0.1)
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f'Unknown scheduler: {name}')

    if warmup > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup],
        )
    else:
        scheduler = main_scheduler

    return scheduler


def main():
    parser = argparse.ArgumentParser(description='RPT-Net Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # seed
    set_seed(cfg.get('seed', 42))

    # work dir
    work_dir = cfg.get('work_dir', 'outputs/default')
    os.makedirs(work_dir, exist_ok=True)

    # logger
    logger = setup_logger(work_dir, name='train')
    logger.info(f'Config: {args.config}')
    logger.info(f'Work dir: {work_dir}')

    # device
    gpu_ids = cfg.device.get('gpu', [0])
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # data
    logger.info('Building dataloaders...')
    train_loader = build_dataloader(cfg.data, split='train')
    val_loader = build_dataloader(cfg.data, split='val')
    logger.info(f'Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches')
    logger.info(f'Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches')

    # model
    logger.info('Building model...')
    model_cfg = dict(cfg.model)
    model = build_phatt_baseline(model_cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model: {cfg.model.name}  Params: {n_params:,}')

    # multi-GPU
    if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f'Using DataParallel on GPUs: {gpu_ids}')

    # loss
    label_smoothing = cfg.training.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        logger.info('Using AMP mixed precision training')

    # resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        logger.info(f'Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%')

    # tensorboard
    tb_writer = None
    if cfg.logging.get('use_tensorboard', False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(os.path.join(work_dir, 'tb'))
        except ImportError:
            logger.info('TensorBoard not available, skipping')

    # training loop
    print_freq = cfg.logging.get('print_freq', 20)
    save_freq = cfg.logging.get('save_freq', 10)
    total_epochs = cfg.training.epochs

    logger.info(f'Start training for {total_epochs} epochs')
    logger.info('=' * 60)

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch + 1, device, logger, print_freq=print_freq, scaler=scaler,
        )

        # evaluate
        val_stats = evaluate(model, val_loader, criterion, device, logger)

        epoch_time = time.time() - epoch_start
        logger.info(
            f'Epoch {epoch + 1}/{total_epochs} done in {epoch_time:.0f}s  '
            f'Train Loss={train_stats["loss"]:.4f} Acc={train_stats["top1"]:.2f}%  '
            f'Val Loss={val_stats["loss"]:.4f} Acc={val_stats["top1"]:.2f}%'
        )
        logger.info('-' * 60)

        # tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', train_stats['loss'], epoch + 1)
            tb_writer.add_scalar('train/acc', train_stats['top1'], epoch + 1)
            tb_writer.add_scalar('val/loss', val_stats['loss'], epoch + 1)
            tb_writer.add_scalar('val/acc', val_stats['top1'], epoch + 1)
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)

        # save checkpoint
        is_best = val_stats['top1'] > best_acc
        if is_best:
            best_acc = val_stats['top1']

        raw_model = model.module if hasattr(model, 'module') else model
        state = {
            'epoch': epoch + 1,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'config': dict(cfg),
        }

        if is_best:
            save_checkpoint(state, work_dir, 'best.pth')
            logger.info(f'  ** New best: {best_acc:.2f}% **')

        if (epoch + 1) % save_freq == 0 or (epoch + 1) == total_epochs:
            save_checkpoint(state, work_dir, f'epoch_{epoch + 1}.pth')

    logger.info('=' * 60)
    logger.info(f'Training finished. Best Val Acc: {best_acc:.2f}%')

    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()

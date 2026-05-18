"""Training and evaluation engine."""

import time
import torch
import torch.nn as nn


class AverageMeter:
    """Computes and stores running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch,
                    device, logger, print_freq=20, scaler=None):
    """Train for one epoch.

    Returns:
        dict with 'loss' and 'top1' averages.
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for i, (data, label, _) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(data)
                loss = criterion(logits, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1 = accuracy(logits, label, topk=(1,))[0]
        loss_meter.update(loss.item(), data.size(0))
        acc_meter.update(top1, data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0 or (i + 1) == len(loader):
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{i + 1}/{len(loader)}]  '
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})  '
                f'LR {lr:.6f}  '
                f'Time {batch_time.avg:.3f}s'
            )

    if scheduler is not None:
        scheduler.step()

    return {'loss': loss_meter.avg, 'top1': acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, logger):
    """Evaluate model on validation set.

    Returns:
        dict with 'loss' and 'top1' averages.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for data, label, _ in loader:
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(data)
        loss = criterion(logits, label)

        top1 = accuracy(logits, label, topk=(1,))[0]
        loss_meter.update(loss.item(), data.size(0))
        acc_meter.update(top1, data.size(0))

    logger.info(
        f'  Val Loss {loss_meter.avg:.4f}  '
        f'Val Acc@1 {acc_meter.avg:.2f}%'
    )
    return {'loss': loss_meter.avg, 'top1': acc_meter.avg}

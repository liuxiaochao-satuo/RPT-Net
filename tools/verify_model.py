"""
Verify PH-Att Baseline: forward pass, backward pass, and parameter count.

Usage:
    conda run -n RPT-Net python tools/verify_model.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.phatt_baseline import PHAttBaseline


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_forward_backward():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # build model
    model = PHAttBaseline(
        num_classes=60,
        num_frames=64,
        num_parts=5,
        heatmap_size=32,
        sigma=1.5,
        token_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_drop=0.1,
        classifier_drop=0.5,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f'\nModel Parameters:')
    print(f'  Total:     {total:,}')
    print(f'  Trainable: {trainable:,}')
    print(f'  Size:      {total * 4 / 1024 / 1024:.1f} MB (fp32)')

    # per-module parameter count
    print(f'\nPer-module breakdown:')
    for name, module in [
        ('heatmap_generator', model.heatmap_generator),
        ('cnn_encoder', model.cnn_encoder),
        ('part_embedding', model.part_embedding),
        ('time_embedding', model.time_embedding),
        ('attention_blocks', model.blocks),
        ('classifier', model.classifier),
    ]:
        if isinstance(module, nn.Parameter):
            n = module.numel()
        else:
            n = sum(p.numel() for p in module.parameters())
        print(f'  {name:25s}: {n:>10,}')

    # test different batch sizes
    for B in [2, 8, 16]:
        print(f'\n--- Batch size = {B} ---')
        x = torch.randn(B, 3, 64, 25, 2, device=device)
        labels = torch.randint(0, 60, (B,), device=device)

        # forward
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        model.train()
        logits = model(x)
        print(f'  Input:  {list(x.shape)}')
        print(f'  Output: {list(logits.shape)}')

        # loss + backward
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        print(f'  Loss:   {loss.item():.4f}')
        print(f'  Grad OK: {model.classifier[-1].weight.grad is not None}')

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f'  Peak GPU mem: {peak_mb:.0f} MB')

        model.zero_grad()

    # test return_attention
    print(f'\n--- Test return_attention ---')
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 64, 25, 2, device=device)
        logits, aux = model(x, return_attention=True)
        print(f'  logits: {list(logits.shape)}')
        print(f'  aux keys: {list(aux.keys())}')
        print(f'  part_heatmaps: {list(aux["part_heatmaps"].shape)}')
        print(f'  part_tokens: {list(aux["part_tokens"].shape)}')
        print(f'  num attention layers: {len(aux["attention_maps"])}')

    print(f'\n=== All verification passed! ===')


if __name__ == '__main__':
    test_forward_backward()

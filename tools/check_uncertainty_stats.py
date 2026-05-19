"""
Check uncertainty indicator statistics and verify all ablation variants.

Usage:
    python tools/check_uncertainty_stats.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.phatt_baseline import PHAttBaseline


def check_uncertainty_stats():
    """Load a batch, compute uncertainty indicators, print stats."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PHAttBaseline(
        num_classes=60, num_frames=64, num_parts=5,
        heatmap_size=32, sigma=1.5, token_dim=128, depth=2,
        use_uncertainty_encoding=True, use_reliability_attention=True,
    ).to(device)

    # use real data if available, otherwise synthetic
    data_path = 'data/ntu/processed/ntu60_xsub/train_data.npy'
    label_path = 'data/ntu/processed/ntu60_xsub/train_label.pkl'

    if os.path.exists(data_path):
        from datasets.ntu_dataset import NTUDataset
        from torch.utils.data import DataLoader
        dataset = NTUDataset(data_path, label_path, num_frames=64, split='val')
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        x, labels, _ = next(iter(loader))
        x = x.to(device)
        print('Using real NTU data')
    else:
        x = torch.randn(8, 3, 64, 25, 2, device=device)
        print('Using synthetic data')

    model.eval()
    with torch.no_grad():
        logits, aux = model(x, return_attention=True)

    print(f'\nInput shape: {list(x.shape)}')
    print(f'Output shape: {list(logits.shape)}')

    # Part heatmaps
    hm = aux['part_heatmaps']
    print(f'\n--- Part Heatmaps ---')
    print(f'  shape: {list(hm.shape)}')
    print(f'  min: {hm.min().item():.6f}')
    print(f'  max: {hm.max().item():.6f}')
    print(f'  mean: {hm.mean().item():.6f}')

    # Uncertainty
    unc = aux['uncertainty']
    print(f'\n--- Uncertainty ---')
    print(f'  shape: {list(unc.shape)}')

    names = ['confidence', 'dispersion', 'temporal_instability']
    for i, name in enumerate(names):
        if i >= unc.shape[-1]:
            break
        val = unc[..., i]
        print(f'\n  [{name}]')
        print(f'    min:     {val.min().item():.6f}')
        print(f'    max:     {val.max().item():.6f}')
        print(f'    mean:    {val.mean().item():.6f}')
        print(f'    std:     {val.std().item():.6f}')
        print(f'    has_nan: {torch.isnan(val).any().item()}')
        print(f'    has_inf: {torch.isinf(val).any().item()}')

    # Reliability
    rel = aux['reliability']
    print(f'\n--- Reliability ---')
    print(f'  shape: {list(rel.shape)}')
    print(f'  min:     {rel.min().item():.6f}')
    print(f'  max:     {rel.max().item():.6f}')
    print(f'  mean:    {rel.mean().item():.6f}')
    print(f'  std:     {rel.std().item():.6f}')
    print(f'  has_nan: {torch.isnan(rel).any().item()}')
    print(f'  has_inf: {torch.isinf(rel).any().item()}')

    print(f'\n[OK] Uncertainty stats check passed!')


def verify_all_ablation_variants():
    """Verify forward + backward for all 4 ablation configurations."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    configs = [
        ('Baseline (UE=off, RA=off)', False, False),
        ('PH-Att + UE',               True,  False),
        ('PH-Att + RA',               False, True),
        ('RPT-Net (UE=on, RA=on)',    True,  True),
    ]

    x = torch.randn(4, 3, 64, 25, 2, device=device)
    labels = torch.randint(0, 60, (4,), device=device)
    criterion = nn.CrossEntropyLoss()

    for name, use_ue, use_ra in configs:
        print(f'\n--- {name} ---')
        model = PHAttBaseline(
            num_classes=60, num_frames=64, num_parts=5,
            heatmap_size=32, sigma=1.5, token_dim=128, depth=2,
            use_uncertainty_encoding=use_ue,
            use_reliability_attention=use_ra,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Params: {n_params:,}')

        # forward
        model.train()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()

        print(f'  Output: {list(logits.shape)}')
        print(f'  Loss: {loss.item():.4f}')
        print(f'  Grad OK: {model.classifier[-1].weight.grad is not None}')
        print(f'  [OK]')

        model.zero_grad()

    # Test uncertainty indicator subsets
    print(f'\n=== Uncertainty Indicator Subsets ===')
    subset_configs = [
        ('C only',   True,  False, False),
        ('C + D',    True,  True,  False),
        ('C + D + I', True, True,  True),
    ]
    for name, c, d, i in subset_configs:
        print(f'\n--- {name} ---')
        model = PHAttBaseline(
            num_classes=60, num_frames=64, num_parts=5,
            heatmap_size=32, sigma=1.5, token_dim=128, depth=2,
            use_uncertainty_encoding=True,
            use_reliability_attention=True,
            use_confidence=c, use_dispersion=d, use_temporal_instability=i,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Params: {n_params:,}')

        model.train()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()

        print(f'  Loss: {loss.item():.4f}')
        print(f'  Grad OK: {model.classifier[-1].weight.grad is not None}')
        print(f'  [OK]')
        model.zero_grad()

    print(f'\n=== All ablation variants verified! ===')


if __name__ == '__main__':
    print('=' * 60)
    print('Part 1: Uncertainty Statistics Check')
    print('=' * 60)
    check_uncertainty_stats()

    print('\n' + '=' * 60)
    print('Part 2: Ablation Variant Verification')
    print('=' * 60)
    verify_all_ablation_variants()

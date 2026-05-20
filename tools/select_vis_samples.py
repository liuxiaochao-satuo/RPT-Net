"""
Select high-confidence correctly-predicted samples for attention visualization.

Usage:
    python tools/select_vis_samples.py \
        --config configs/ntu60/ablation/03_rptnet_full_xsub.yaml \
        --checkpoint outputs/ablation/03_rptnet_full_xsub/best.pth \
        --actions 0,9,23,42 \
        --output outputs/visualization/ntu60_xsub/sample_manifest.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.checkpoint import load_checkpoint
from datasets.ntu_dataset import NTUDataset
from models.phatt_baseline import build_phatt_baseline

ACTION_NAMES = {
    0: 'drink_water',
    9: 'clapping',
    23: 'kicking',
    42: 'falling_down',
}


@torch.no_grad()
def scan_val_set(model, loader, target_labels, top_k=2, device='cuda'):
    """Find top-k highest-confidence correct samples per target label."""
    model.eval()
    candidates = {label: [] for label in target_labels}

    for data, labels, indices in loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(data)
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

        for i in range(data.size(0)):
            label = labels[i].item()
            if label not in target_labels:
                continue
            if preds[i].item() != label:
                continue
            candidates[label].append({
                'label': label,
                'index': indices[i].item(),
                'confidence': conf[i].item(),
                'pred': preds[i].item(),
            })

    manifest = []
    for label in target_labels:
        items = sorted(candidates[label], key=lambda x: x['confidence'], reverse=True)
        action = ACTION_NAMES.get(label, f'action_{label}')
        for rank, item in enumerate(items[:top_k]):
            manifest.append({
                'action': action,
                'label': label,
                'index': item['index'],
                'confidence': round(item['confidence'], 4),
                'rank': rank + 1,
                'role': 'primary' if rank == 0 else 'backup',
            })

    return manifest


def attach_sample_names(manifest, dataset):
    """Add sample_name field from dataset."""
    for entry in manifest:
        idx = entry['index']
        entry['sample_name'] = dataset.sample_names[idx]
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Select visualization samples')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--actions', type=str, default='0,9,23,42')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    target_labels = [int(x.strip()) for x in args.actions.split(',')]

    cfg = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = NTUDataset(
        data_path=cfg.data.val_data,
        label_path=cfg.data.val_label,
        num_frames=cfg.data.num_frames,
        split='val',
        random_shift=False,
        random_mirror=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.data.get('num_workers', 4), pin_memory=True,
    )

    model = build_phatt_baseline(dict(cfg.model)).to(device)
    load_checkpoint(args.checkpoint, model)
    print(f'Loaded checkpoint: {args.checkpoint}')

    manifest = scan_val_set(
        model, loader, target_labels, top_k=args.top_k, device=device,
    )
    manifest = attach_sample_names(manifest, dataset)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Selected {len(manifest)} samples -> {args.output}')
    for entry in manifest:
        print(f"  [{entry['role']}] {entry['action']}: idx={entry['index']}, "
              f"conf={entry['confidence']:.4f}, name={entry['sample_name']}")


if __name__ == '__main__':
    main()

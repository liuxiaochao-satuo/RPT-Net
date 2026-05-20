"""
Generate attention visualization panels for selected samples.

Usage:
    python tools/visualize_attention.py \
        --config configs/ntu60/ablation/03_rptnet_full_xsub.yaml \
        --checkpoint outputs/ablation/03_rptnet_full_xsub/best.pth \
        --manifest outputs/visualization/ntu60_xsub/sample_manifest.json \
        --output_dir outputs/visualization/ntu60_xsub
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.checkpoint import load_checkpoint
from utils.visualization import (
    aggregate_tp_attention,
    save_figure_panel,
)
from datasets.ntu_dataset import NTUDataset
from models.phatt_baseline import build_phatt_baseline


@torch.no_grad()
def extract_maps(model, x, layer='last'):
    """Run dual forward pass and return standard/RA/reliability T×P maps."""
    _, aux_std = model(
        x, return_attention=True, apply_reliability_bias=False,
    )
    _, aux_ra = model(
        x, return_attention=True, apply_reliability_bias=True,
    )

    standard_tp = aggregate_tp_attention(
        aux_std['attention_maps'], batch_idx=0, layer=layer,
    )
    reliability_tp = aggregate_tp_attention(
        aux_ra['attention_maps'], batch_idx=0, layer=layer,
    )
    reliability_score = aux_ra['reliability'][0].detach().cpu().numpy()

    return standard_tp, reliability_tp, reliability_score


def load_sample(dataset, index):
    """Load a single sample by dataset index."""
    data, label, idx = dataset[index]
    return data, label, idx


def main():
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--layer', type=str, default='last', choices=['last', 'mean'])
    parser.add_argument('--time_stride', type=int, default=1,
                        help='Downsample time axis for display (1=no downsample)')
    parser.add_argument('--role', type=str, default='primary',
                        choices=['primary', 'backup', 'all'])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.role != 'all':
        manifest = [e for e in manifest if e.get('role', 'primary') == args.role]

    dataset = NTUDataset(
        data_path=cfg.data.val_data,
        label_path=cfg.data.val_label,
        num_frames=cfg.data.num_frames,
        split='val',
        random_shift=False,
        random_mirror=False,
    )

    model = build_phatt_baseline(dict(cfg.model)).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Visualizing {len(manifest)} samples')

    for entry in manifest:
        action = entry['action']
        index = entry['index']
        sample_name = entry.get('sample_name', f'idx_{index}')

        data, label, _ = load_sample(dataset, index)
        x = data.unsqueeze(0).to(device)

        standard_tp, reliability_tp, reliability_score = extract_maps(
            model, x, layer=args.layer,
        )

        out_dir = os.path.join(args.output_dir, action)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, f'sample_{index}_standard.npy'), standard_tp)
        np.save(os.path.join(out_dir, f'sample_{index}_reliability_attn.npy'), reliability_tp)
        np.save(os.path.join(out_dir, f'sample_{index}_reliability_score.npy'), reliability_score)

        panel_path = os.path.join(out_dir, f'sample_{index}_panel.png')
        save_figure_panel(
            skeleton_tensor=data,
            standard_tp=standard_tp,
            reliability_tp=reliability_tp,
            reliability_score=reliability_score,
            save_path=panel_path,
            action_name=action,
            sample_name=sample_name,
            time_stride=args.time_stride,
        )

        print(f'  [{action}] idx={index}, label={label}, saved -> {panel_path}')


if __name__ == '__main__':
    main()

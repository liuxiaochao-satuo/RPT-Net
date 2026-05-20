"""
Robustness evaluation: test models under random joint drop.

Loads trained checkpoints, evaluates at drop_ratio = 0.0 / 0.1 / 0.2,
and outputs a summary table + JSON results.

Usage:
    python tools/run_robustness_eval.py
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.robustness import apply_random_joint_drop_batch
from datasets.ntu_dataset import NTUDataset
from models.phatt_baseline import PHAttBaseline
from engine.trainer import accuracy, AverageMeter


# ──────────────────────────────────────────────
# Model configurations and checkpoint paths
# ──────────────────────────────────────────────
EVAL_MODELS = [
    {
        "name": "PH-Att Baseline",
        "checkpoint": "outputs/ntu60_xsub_phatt/best.pth",
        "config": {
            "use_uncertainty_encoding": False,
            "use_reliability_attention": False,
        },
    },
    {
        "name": "PH-Att + UE",
        "checkpoint": "outputs/ablation/01_phatt_ue_xsub/best.pth",
        "config": {
            "use_uncertainty_encoding": True,
            "use_reliability_attention": False,
        },
    },
    {
        "name": "PH-Att + RA",
        "checkpoint": "outputs/ablation/02_phatt_ra_xsub/best.pth",
        "config": {
            "use_uncertainty_encoding": False,
            "use_reliability_attention": True,
        },
    },
    {
        "name": "RPT-Net",
        "checkpoint": "outputs/ablation/03_rptnet_full_xsub/best.pth",
        "config": {
            "use_uncertainty_encoding": True,
            "use_reliability_attention": True,
        },
    },
]

DROP_RATIOS = [0.0, 0.1, 0.2]
SEED = 42
DATA_PATH = "data/ntu/processed/ntu60_xsub/val_data.npy"
LABEL_PATH = "data/ntu/processed/ntu60_xsub/val_label.pkl"
OUTPUT_DIR = "outputs/robustness/ntu60_xsub"

# shared model hyperparams (must match training)
MODEL_BASE = dict(
    num_classes=60, num_frames=64, num_parts=5,
    heatmap_size=32, sigma=1.5, token_dim=128,
    depth=4, num_heads=4, mlp_ratio=4.0,
    dropout=0.1, attn_drop=0.1, classifier_drop=0.5,
)


def build_model(model_cfg, device):
    cfg = {**MODEL_BASE, **model_cfg}
    model = PHAttBaseline(**cfg).to(device)
    return model


@torch.no_grad()
def evaluate_with_drop(model, loader, drop_ratio, seed, device):
    """Evaluate model with random joint drop applied to inputs."""
    model.eval()
    acc_meter = AverageMeter()

    for data, label, indices in loader:
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if drop_ratio > 0:
            data = apply_random_joint_drop_batch(
                data, drop_ratio=drop_ratio,
                seed=seed, sample_indices=indices,
            )

        logits = model(data)
        top1 = accuracy(logits, label, topk=(1,))[0]
        acc_meter.update(top1, data.size(0))

    return acc_meter.avg


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # dataloader
    dataset = NTUDataset(
        data_path=DATA_PATH, label_path=LABEL_PATH,
        num_frames=64, split='val',
        random_shift=False, random_mirror=False,
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f'Val samples: {len(dataset)}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # results table
    all_results = []

    for model_info in EVAL_MODELS:
        name = model_info["name"]
        ckpt_path = model_info["checkpoint"]

        if not os.path.exists(ckpt_path):
            print(f'\n[SKIP] {name}: checkpoint not found at {ckpt_path}')
            continue

        print(f'\n{"=" * 60}')
        print(f'Evaluating: {name}')
        print(f'Checkpoint: {ckpt_path}')

        model = build_model(model_info["config"], device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded (reported best_acc={ckpt.get("best_acc", "N/A")})')

        results = {}
        for dr in DROP_RATIOS:
            t0 = time.time()
            acc = evaluate_with_drop(model, loader, dr, SEED, device)
            elapsed = time.time() - t0
            label = f'drop_{int(dr * 100)}' if dr > 0 else 'clean'
            results[label] = round(acc, 2)
            print(f'  {label:>10s}: {acc:.2f}%  ({elapsed:.1f}s)')

        # compute deltas
        clean = results['clean']
        drop10_delta = round(clean - results['drop_10'], 2)
        drop20_delta = round(clean - results['drop_20'], 2)
        avg_drop = round((drop10_delta + drop20_delta) / 2, 2)

        result_entry = {
            "method": name,
            "dataset": "NTU60",
            "protocol": "X-Sub",
            "drop_type": "random_joint_drop",
            "seed": SEED,
            "results": results,
            "drops": {
                "drop10_delta": drop10_delta,
                "drop20_delta": drop20_delta,
                "avg_drop": avg_drop,
            },
        }
        all_results.append(result_entry)

        # save per-model JSON
        safe_name = name.lower().replace(' ', '_').replace('+', '_')
        json_path = os.path.join(OUTPUT_DIR, f'{safe_name}.json')
        with open(json_path, 'w') as f:
            json.dump(result_entry, f, indent=2)

    # print summary table
    print(f'\n{"=" * 60}')
    print(f'ROBUSTNESS SUMMARY (NTU60 X-Sub)')
    print(f'{"=" * 60}')
    print(f'{"Method":<20s} {"Clean":>7s} {"Drop10":>7s} {"Drop20":>7s} {"Δ10":>6s} {"Δ20":>6s} {"AvgΔ":>6s}')
    print('-' * 60)
    for r in all_results:
        res = r['results']
        drp = r['drops']
        print(f'{r["method"]:<20s} {res["clean"]:>7.2f} {res["drop_10"]:>7.2f} '
              f'{res["drop_20"]:>7.2f} {drp["drop10_delta"]:>6.2f} '
              f'{drp["drop20_delta"]:>6.2f} {drp["avg_drop"]:>6.2f}')
    print('=' * 60)

    # save all results
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()

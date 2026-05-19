#!/bin/bash
# Run all ablation experiments sequentially on 4 GPUs.
# Baseline (00) is skipped since we already have the result (66.78%).
#
# Usage: bash scripts/run_ablation.sh

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/hello/miniconda3/envs/RPT-Net/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "========================================"
echo "RPT-Net Ablation Experiments"
echo "Start time: $(date)"
echo "========================================"

# Core module ablation (skip 00_baseline, reuse existing result)
configs=(
    "configs/ntu60/ablation/01_phatt_ue_xsub.yaml"
    "configs/ntu60/ablation/02_phatt_ra_xsub.yaml"
    "configs/ntu60/ablation/03_rptnet_full_xsub.yaml"
    "configs/ntu60/ablation/04_uncertainty_c_xsub.yaml"
    "configs/ntu60/ablation/05_uncertainty_cd_xsub.yaml"
    "configs/ntu60/ablation/06_uncertainty_cdi_xsub.yaml"
)

for cfg in "${configs[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $cfg"
    echo "Time: $(date)"
    echo "========================================"
    $PYTHON -u train.py --config "$cfg"
    echo "Finished: $cfg at $(date)"
done

echo ""
echo "========================================"
echo "All ablation experiments completed!"
echo "End time: $(date)"
echo "========================================"

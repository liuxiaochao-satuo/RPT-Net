#!/bin/bash
# Uncertainty composition ablation: 04 (resume) -> 05
# 06 (C+D+I) is identical to 03_rptnet_full; reuse outputs/ablation/03_rptnet_full_xsub
set -e
cd "$(dirname "$0")/.."
PYTHON=/home/hello/miniconda3/envs/RPT-Net/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
LOG=outputs/ablation/uncertainty_ablation_runner.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Uncertainty composition ablation (04 resume -> 05) ==="

log "Resuming 04_uncertainty_c_xsub from best.pth (epoch 56, best 70.12%)"
$PYTHON -u train.py \
  --config configs/ntu60/ablation/04_uncertainty_c_xsub.yaml \
  --resume outputs/ablation/04_uncertainty_c_xsub/best.pth \
  2>&1 | tee -a "$LOG"

log "Starting 05_uncertainty_cd_xsub (from scratch)"
$PYTHON -u train.py \
  --config configs/ntu60/ablation/05_uncertainty_cd_xsub.yaml \
  2>&1 | tee -a "$LOG"

log "Skipping 06: same as 03_rptnet_full (reuse outputs/ablation/03_rptnet_full_xsub)"
log "=== Uncertainty ablation runs completed ==="

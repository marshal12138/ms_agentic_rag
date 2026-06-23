#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU CoSearch validation-only smoke derived from the official VERL trainer.
# This runs CoSearchRayTrainer.fit() with trainer.val_only=True.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints/official_verl_multi_gpu_val_smoke}"
EXP_NAME="${EXP_NAME:-official_verl_qwen3_0_6b_val_smoke}"
TOTAL_STEPS=1
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"

export OUT_DIR EXP_NAME TOTAL_STEPS TRAIN_MAX_SAMPLES VAL_MAX_SAMPLES TRAIN_BATCH_SIZE VAL_BATCH_SIZE

bash "${ROOT}/scripts/cosearch_local/train_cosearch_verl_base.sh" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1

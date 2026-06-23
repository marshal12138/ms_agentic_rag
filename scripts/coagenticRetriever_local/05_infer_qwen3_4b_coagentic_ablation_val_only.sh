#!/usr/bin/env bash
set -euo pipefail

# Same inference entry as 02_infer_qwen3_4b_ablation_val_only.sh, but points
# at the Qwen3-style CoAgenticRetriever ablation eval parquet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${TRAIN_DATA:=${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet}"
: "${VAL_DATA:=${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet}"
: "${GROUP_NAME:=coAgenticRetriever}"

export TRAIN_DATA VAL_DATA GROUP_NAME
exec bash "${SCRIPT_DIR}/02_infer_qwen3_4b_ablation_val_only.sh" "$@"

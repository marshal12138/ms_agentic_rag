#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
PORT="${PORT:-8010}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export GPU_IDS

cd "${ROOT}"
LOG_DIR="${LOG_DIR:-${ROOT}/log/train_logs/smoke}"
mkdir -p "${LOG_DIR}"

bash scripts/cosearch_local/00_prepare_assets.sh

PORT="${PORT}" RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${GPU_IDS}}" \
  bash scripts/cosearch_local/02b_start_dense_retriever_server.sh > "${LOG_DIR}/dense_retriever_${PORT}.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
sleep 8

curl -fsS "http://127.0.0.1:${PORT}/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["capital of France"],"topk":1,"return_scores":true}' >/dev/null

bash scripts/cosearch_local/train_cosearch_verl_base.sh
bash scripts/cosearch_local/04_test_official_verl_multi_gpu_smoke.sh

echo "Official VERL smoke outputs: checkpoints/official_verl_multi_gpu_smoke and checkpoints/official_verl_multi_gpu_val_smoke"

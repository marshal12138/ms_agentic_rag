#!/usr/bin/env bash
set -euo pipefail

# One-command paper-path smoke:
# check assets, start dense retriever, run joint training, then validation-only eval.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
PORT="${PORT:-8010}"
RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${RETRIEVER_GPU_ID:-0,1,2,3,4,5,6,7}}"
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-data/retrieval/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"

cd "${ROOT}"
LOG_DIR="${LOG_DIR:-${ROOT}/log/train_logs/paper_qwen3_dense_smoke}"
mkdir -p "${LOG_DIR}"

bash scripts/cosearch_local/00_prepare_assets.sh

if [[ ! -f "${INDEX_FILE}" || ! -f "${CORPUS_FILE}" ]]; then
  echo "ERROR: retrieval assets not found: ${INDEX_FILE} or ${CORPUS_FILE}" >&2
  echo "Run scripts/cosearch_local/01b_download_e5_and_build_dense_retriever.sh before training or inference." >&2
  exit 2
fi

RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR}" INDEX_FILE="${INDEX_FILE}" CORPUS_FILE="${CORPUS_FILE}" \
  PORT="${PORT}" RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS}" \
  bash scripts/cosearch_local/02b_start_dense_retriever_server.sh > "${LOG_DIR}/dense_retriever_${PORT}.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
sleep 8

curl -fsS "http://127.0.0.1:${PORT}/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["capital of France"],"topk":1}' >/dev/null

PORT="${PORT}" bash scripts/cosearch_local/05_train_paper_qwen3_dense_smoke.sh
PORT="${PORT}" bash scripts/cosearch_local/06_eval_paper_qwen3_dense_smoke.sh

echo "Paper-path smoke complete."
echo "Training output: checkpoints/paper_qwen3_dense_smoke"
echo "Evaluation output: checkpoints/paper_qwen3_dense_eval_smoke"

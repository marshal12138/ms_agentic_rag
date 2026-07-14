#!/usr/bin/env bash
set -euo pipefail

# AIR SPAD agent search evaluation.
# This entry calls only AgenticIterRag v1 inference code and does not call CAR scripts.

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/AgenticIterRag:${ROOT}/AgenticIterRag/verl:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage:
  eval_spad_agent_search_350.sh --agent-model PATH [options]

Options:
  --agent-model PATH      Final SPAD actor checkpoint or HF model directory. Required.
  --data-path PATH        Eval parquet. Default: data/AgenticIterRag/source/co_search_ablation.infer.parquet
  --max-samples N         Eval sample count. Default: 350
  --task-name NAME        Output task name. Default: timestamped spad_agent_search_350
  --repeat-id N           Independent repeat identifier recorded in the run manifest.
  --agent-gpu-ids IDS     Agent vLLM data-parallel replica devices. Default: 0,1,2,3,4,5
  --agent-instance-count N Agent replica count. Default: number of --agent-gpu-ids
  --agent-backend-base-port PORT First agent replica port. Default: agent-port + 1
  --agent-max-num-seqs N  Per-replica vLLM max_num_seqs. Default: 64
  --recall-gpu-ids IDS    Recall retriever devices. Default: 6,7
  --infer-batch-size N    Async eval concurrency. Default: 384
  --flush-every-n N       Rewrite partial traces every N completions. Default: 500
  --agent-port PORT       Agent data-parallel proxy port. Default: 8240
  --proxy-port PORT       Recall proxy port. Default: 8230
  --help                  Show this message.
EOF
}

AGENT_MODEL="${AGENT_MODEL:-}"
DATA_PATH="${DATA_PATH:-${ROOT}/data/AgenticIterRag/source/co_search_ablation.infer.parquet}"
MAX_SAMPLES="${MAX_SAMPLES:-350}"
TASK_NAME="${TASK_NAME:-$(date +%y%m%d-%H%M%S)-spad_agent_search_350}"
REPEAT_ID="${REPEAT_ID:-}"
AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3,4,5}"
AGENT_INSTANCE_COUNT="${AGENT_INSTANCE_COUNT:-}"
AGENT_BACKEND_BASE_PORT="${AGENT_BACKEND_BASE_PORT:-}"
AGENT_MAX_NUM_SEQS="${AGENT_MAX_NUM_SEQS:-64}"
RECALL_GPU_IDS="${RECALL_GPU_IDS:-6,7}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-384}"
FLUSH_EVERY_N="${FLUSH_EVERY_N:-500}"
AGENT_PORT="${AGENT_PORT:-8240}"
PROXY_PORT="${PROXY_PORT:-8230}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --agent-model)
      AGENT_MODEL="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --repeat-id)
      REPEAT_ID="$2"
      shift 2
      ;;
    --agent-gpu-ids)
      AGENT_GPU_IDS="$2"
      shift 2
      ;;
    --agent-instance-count)
      AGENT_INSTANCE_COUNT="$2"
      shift 2
      ;;
    --agent-backend-base-port)
      AGENT_BACKEND_BASE_PORT="$2"
      shift 2
      ;;
    --agent-max-num-seqs)
      AGENT_MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --recall-gpu-ids)
      RECALL_GPU_IDS="$2"
      shift 2
      ;;
    --infer-batch-size)
      INFER_BATCH_SIZE="$2"
      shift 2
      ;;
    --flush-every-n)
      FLUSH_EVERY_N="$2"
      shift 2
      ;;
    --agent-port)
      AGENT_PORT="$2"
      shift 2
      ;;
    --proxy-port)
      PROXY_PORT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${AGENT_MODEL}" ]]; then
  echo "ERROR: --agent-model is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -e "${DATA_PATH}" ]]; then
  echo "ERROR: eval data not found: ${DATA_PATH}" >&2
  exit 2
fi
if [[ ! "${MAX_SAMPLES}" =~ ^[0-9]+$ ]] || (( MAX_SAMPLES < 1 )); then
  echo "ERROR: --max-samples must be a positive integer; got ${MAX_SAMPLES}" >&2
  exit 2
fi
if [[ ! "${INFER_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( INFER_BATCH_SIZE < 1 )); then
  echo "ERROR: --infer-batch-size must be a positive integer; got ${INFER_BATCH_SIZE}" >&2
  exit 2
fi
if [[ ! "${FLUSH_EVERY_N}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --flush-every-n must be a non-negative integer; got ${FLUSH_EVERY_N}" >&2
  exit 2
fi
if [[ ! "${AGENT_MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || (( AGENT_MAX_NUM_SEQS < 1 )); then
  echo "ERROR: --agent-max-num-seqs must be a positive integer; got ${AGENT_MAX_NUM_SEQS}" >&2
  exit 2
fi
if [[ -n "${REPEAT_ID}" ]] && [[ ! "${REPEAT_ID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --repeat-id must be a positive integer; got ${REPEAT_ID}" >&2
  exit 2
fi
if [[ ! -e "${AGENT_MODEL}" ]]; then
  echo "ERROR: agent model not found: ${AGENT_MODEL}" >&2
  exit 2
fi

csv_count() {
  local value="$1"
  python - "$value" <<'PY'
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(items))
PY
}

if [[ -z "${AGENT_INSTANCE_COUNT}" ]]; then
  AGENT_INSTANCE_COUNT="$(csv_count "${AGENT_GPU_IDS}")"
fi
if [[ -z "${AGENT_BACKEND_BASE_PORT}" ]]; then
  AGENT_BACKEND_BASE_PORT="$((AGENT_PORT + 1))"
fi
RECALL_INSTANCE_COUNT="${RECALL_INSTANCE_COUNT:-$(csv_count "${RECALL_GPU_IDS}")}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/reports/eval/agenticIterRag}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/eval/agenticIterRag}"
TRACE_DIR="${TRACE_DIR:-${LOG_ROOT}/${TASK_NAME}/trace}"
RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-${LOG_ROOT}/${TASK_NAME}/runtime_logs}"
REPORT_PATH="${REPORT_PATH:-${OUT_ROOT}/${TASK_NAME}.report.md}"
RUN_MANIFEST="${RUNTIME_LOG_DIR}/eval_run_manifest.json"

for output_path in "${TRACE_DIR}" "${RUNTIME_LOG_DIR}"; do
  if [[ -d "${output_path}" ]] && [[ -n "$(find "${output_path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: refusing to reuse non-empty eval output directory: ${output_path}" >&2
    echo "Use a unique --task-name for every model and repeat." >&2
    exit 2
  fi
done
if [[ -e "${REPORT_PATH}" ]]; then
  echo "ERROR: refusing to overwrite existing eval report: ${REPORT_PATH}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${TRACE_DIR}" "${RUNTIME_LOG_DIR}"

python "${ROOT}/scripts/cosearch_local/write_eval_run_manifest.py" \
  --output "${RUN_MANIFEST}" \
  --task-name "${TASK_NAME}" \
  --repeat-id "${REPEAT_ID}" \
  --data-path "${DATA_PATH}" \
  --model-path "${AGENT_MODEL}" \
  --max-samples "${MAX_SAMPLES}" \
  --temperature 0.0 \
  --top-p 1.0

AIR_INFER_PRECOMPILED_ENV=1 \
GROUP_NAME=agenticIterRag \
INFER_TASK_NAME=spad_agent_search_eval \
TASK_NAME="${TASK_NAME}" \
RUN_NAME=spad_agent_search_eval \
EXP_NAME="${TASK_NAME}" \
DATA_PATH="${DATA_PATH}" \
AGENT_MODEL="${AGENT_MODEL}" \
MAX_INFER_NUM="${MAX_SAMPLES}" \
INFER_BATCH_SIZE="${INFER_BATCH_SIZE}" \
FLUSH_EVERY_N="${FLUSH_EVERY_N}" \
RUN_MODE=no-ranker \
RERANKER=dense_e5 \
KEEP_TRACE=full \
FAIL_ON_INFER_ERROR=false \
AGENT_GPU_IDS="${AGENT_GPU_IDS}" \
AGENT_TP_SIZE=1 \
AGENT_INSTANCE_COUNT="${AGENT_INSTANCE_COUNT}" \
AGENT_PORT="${AGENT_PORT}" \
AGENT_BACKEND_BASE_PORT="${AGENT_BACKEND_BASE_PORT}" \
AGENT_PROXY_STRATEGY=least_inflight \
AGENT_SERVED_MODEL=air-spad-agent-eval \
GPU_MEMORY_UTILIZATION=0.70 \
MAX_NUM_SEQS="${AGENT_MAX_NUM_SEQS}" \
MAX_MODEL_LEN=12288 \
RECALL_BACKEND_TYPE=npu \
RECALL_GPU_ID="${RECALL_GPU_IDS}" \
RECALL_INSTANCE_COUNT="${RECALL_INSTANCE_COUNT}" \
PROXY_PORT="${PROXY_PORT}" \
RECALL_BACKEND_BASE_PORT="$((PROXY_PORT + 1))" \
RETRIEVAL_SERVICE_URL="http://127.0.0.1:${PROXY_PORT}/retrieve" \
RECALL_ASSET_PRECHECK=1 \
RECALL_QUERY_PREFLIGHT=1 \
RECALL_FINAL_TOP_N=50 \
SEARCH_TOOL_FINAL_TOP_M=5 \
MAX_ASSISTANT_TURNS=6 \
MAX_USER_TURNS=6 \
MAX_RESPONSE_LENGTH=1024 \
MAX_TOOL_RESPONSE_LENGTH=4096 \
TEMPERATURE=0.0 \
TOP_P=1.0 \
TRACE_DIR="${TRACE_DIR}" \
OUT_DIR="${TRACE_DIR}" \
RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR}" \
LOG_DIR="${RUNTIME_LOG_DIR}" \
REPORT_PATH="${REPORT_PATH}" \
METRICS_JSONL="${TRACE_DIR}/metrics.jsonl" \
AGENT_TIMING_JSONL="${RUNTIME_LOG_DIR}/agent_timing.jsonl" \
SEARCH_TIMING_JSONL="${RUNTIME_LOG_DIR}/search_timing.jsonl" \
LLM_IO_JSONL="${RUNTIME_LOG_DIR}/llm_io.jsonl" \
bash "${ROOT}/scripts/agenticIterRag_v1/assets/infer_backend/02_air_infer_launcher.sh"

echo "eval report: ${REPORT_PATH}"
echo "eval trace: ${TRACE_DIR}"

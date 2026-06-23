#!/usr/bin/env bash
set -euo pipefail

# vLLM-only CoSearch evaluation entry.
# This script starts dense retrievers and an agent vLLM server. In full mode it
# also starts a reranker vLLM server. It does not use VERL for model loading or
# evaluation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
GROUP_NAME="${GROUP_NAME:-cosearch}"
resolve_cosearch_group_identity "${GROUP_NAME}"

STRATEGY_NAME="${STRATEGY_NAME:-default}"
RUN_MODE="${RUN_MODE:-full}"
case "${RUN_MODE}" in
  full|no-ranker) ;;
  *)
    echo "ERROR: unsupported RUN_MODE=${RUN_MODE}; use full or no-ranker" >&2
    exit 2
    ;;
esac
AGENT_MODEL="${AGENT_MODEL:-/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B}"
RERANKER_MODEL="${RERANKER_MODEL:-/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-${ROOT}/data/co_search/local_flashrag/co_search_ablation.eval.parquet}"
MAX_EVAL_NUM="${MAX_EVAL_NUM:--1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
KEEP_TRACE="${KEEP_TRACE:-partial}"
LLM_IO_JSONL="${LLM_IO_JSONL:-}"

AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1}"
RERANKER_GPU_IDS="${RERANKER_GPU_IDS:-2,3}"
AGENT_TP_SIZE="${AGENT_TP_SIZE:-2}"
RERANKER_TP_SIZE="${RERANKER_TP_SIZE:-2}"
AGENT_PORT="${AGENT_PORT:-8040}"
RERANKER_PORT="${RERANKER_PORT:-8041}"
AGENT_SERVED_MODEL="${AGENT_SERVED_MODEL:-cosearch-agent}"
RERANKER_SERVED_MODEL="${RERANKER_SERVED_MODEL:-cosearch-reranker}"

RETRIEVER_INSTANCES="${RETRIEVER_INSTANCES:-1}"
RETRIEVER_PORT_BASE="${RETRIEVER_PORT_BASE:-8020}"
PROXY_PORT="${PROXY_PORT:-8030}"
PROXY_TIMEOUT="${PROXY_TIMEOUT:-180}"
RETRIEVER_MODE="${RETRIEVER_MODE:-gpu}"
RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda}"
RETRIEVER_GPU_ID="${RETRIEVER_GPU_ID:-5}"
RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${RETRIEVER_GPU_ID}}"
RETRIEVER_DOC_DTYPE="${RETRIEVER_DOC_DTYPE:-float16}"
RETRIEVER_QUERY_BATCH_SIZE="${RETRIEVER_QUERY_BATCH_SIZE:-32}"
RETRIEVER_OMP_NUM_THREADS="${RETRIEVER_OMP_NUM_THREADS:-1}"
RETRIEVER_MKL_NUM_THREADS="${RETRIEVER_MKL_NUM_THREADS:-1}"
RETRIEVER_STARTUP_TIMEOUT="${RETRIEVER_STARTUP_TIMEOUT:-900}"

TOP_N="${TOP_N:-50}"
TOP_M="${TOP_M:-5}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-6}"
MAX_USER_TURNS="${MAX_USER_TURNS:-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-11264}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
RERANKER_MAX_PROMPT_LENGTH="${RERANKER_MAX_PROMPT_LENGTH:-16384}"
RERANKER_MAX_RESPONSE_LENGTH="${RERANKER_MAX_RESPONSE_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RERANKER_TEMPERATURE="${RERANKER_TEMPERATURE:-0.0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

setup_cosearch_eval_artifact_defaults "${ROOT}" "${STRATEGY_NAME}"

RETRIEVER_PGIDS=()
PROXY_PGID=""
AGENT_PGID=""
RERANKER_PGID=""

cleanup() {
  if [[ -n "${AGENT_PGID}" ]] && kill -0 "-${AGENT_PGID}" 2>/dev/null; then
    kill -TERM "-${AGENT_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${RERANKER_PGID}" ]] && kill -0 "-${RERANKER_PGID}" 2>/dev/null; then
    kill -TERM "-${RERANKER_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${PROXY_PGID}" ]] && kill -0 "-${PROXY_PGID}" 2>/dev/null; then
    kill -TERM "-${PROXY_PGID}" 2>/dev/null || true
  fi
  for pgid in "${RETRIEVER_PGIDS[@]}"; do
    if [[ -n "${pgid}" ]] && kill -0 "-${pgid}" 2>/dev/null; then
      kill -TERM "-${pgid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

check_retriever_url() {
  local url="$1"
  curl -fsS "${url}" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["who got the first nobel prize in physics?"],"topk":1,"return_scores":true}' >/dev/null 2>&1
}

wait_for_retriever() {
  local url="$1"
  local pgid="$2"
  local log="$3"
  local start_ts now
  start_ts="$(date +%s)"
  while true; do
    if check_retriever_url "${url}"; then
      echo "retriever ready: ${url}"
      return
    fi
    if ! kill -0 "${pgid}" 2>/dev/null; then
      echo "ERROR: retriever exited before ready: ${url}; log=${log}" >&2
      tail -80 "${log}" >&2 || true
      exit 3
    fi
    now="$(date +%s)"
    if (( now - start_ts > RETRIEVER_STARTUP_TIMEOUT )); then
      echo "ERROR: retriever startup timed out: ${url}; log=${log}" >&2
      tail -80 "${log}" >&2 || true
      exit 3
    fi
    sleep 5
  done
}

start_retrievers() {
  local i port url log pgid
  local wait_urls=()
  local wait_pgids=()
  local wait_logs=()
  for ((i = 0; i < RETRIEVER_INSTANCES; i++)); do
    port=$((RETRIEVER_PORT_BASE + i))
    url="http://127.0.0.1:${port}/retrieve"
    log="${RUNTIME_LOG_DIR}/retriever_${port}.log"
    if check_retriever_url "${url}"; then
      echo "using existing retriever: ${url}"
      continue
    fi
    echo "starting dense retriever ${i}/${RETRIEVER_INSTANCES} on port ${port}; log=${log}"
    setsid env PY="${PY}" PORT="${port}" MODE="${RETRIEVER_MODE}" DEVICE="${RETRIEVER_DEVICE}" \
      GPU_ID="${RETRIEVER_GPU_ID}" RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS}" \
      DOC_DTYPE="${RETRIEVER_DOC_DTYPE}" QUERY_BATCH_SIZE="${RETRIEVER_QUERY_BATCH_SIZE}" FAISS_GPU=0 \
      OMP_NUM_THREADS="${RETRIEVER_OMP_NUM_THREADS}" MKL_NUM_THREADS="${RETRIEVER_MKL_NUM_THREADS}" \
      bash "${ROOT}/src/retrievers/start_dense_retriever_server.sh" > "${log}" 2>&1 &
    pgid="$!"
    RETRIEVER_PGIDS+=("${pgid}")
    wait_urls+=("${url}")
    wait_pgids+=("${pgid}")
    wait_logs+=("${log}")
  done
  for ((i = 0; i < ${#wait_urls[@]}; i++)); do
    wait_for_retriever "${wait_urls[$i]}" "${wait_pgids[$i]}" "${wait_logs[$i]}"
  done
}

start_proxy() {
  local args=()
  local i port
  local log="${RUNTIME_LOG_DIR}/retrieval_proxy.log"
  for ((i = 0; i < RETRIEVER_INSTANCES; i++)); do
    port=$((RETRIEVER_PORT_BASE + i))
    args+=(--backend "http://127.0.0.1:${port}/retrieve")
  done
  if check_retriever_url "http://127.0.0.1:${PROXY_PORT}/retrieve"; then
    echo "using existing retrieval proxy: http://127.0.0.1:${PROXY_PORT}/retrieve"
    return
  fi
  echo "starting retrieval proxy on port ${PROXY_PORT}; backends=${RETRIEVER_INSTANCES}"
  setsid "${PY}" "${ROOT}/src/retrievers/retrieval_round_robin_proxy.py" \
    --host 127.0.0.1 \
    --port "${PROXY_PORT}" \
    --timeout "${PROXY_TIMEOUT}" \
    "${args[@]}" > "${log}" 2>&1 &
  PROXY_PGID="$!"
  wait_for_retriever "http://127.0.0.1:${PROXY_PORT}/retrieve" "${PROXY_PGID}" "${log}"
}

check_vllm() {
  local port="$1"
  curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

wait_for_vllm() {
  local port="$1"
  local pgid="$2"
  local log="$3"
  local start_ts now
  start_ts="$(date +%s)"
  while true; do
    if check_vllm "${port}"; then
      echo "vLLM ready: http://127.0.0.1:${port}"
      return
    fi
    if ! kill -0 "${pgid}" 2>/dev/null; then
      echo "ERROR: vLLM server exited before ready: port=${port}; log=${log}" >&2
      tail -120 "${log}" >&2 || true
      exit 4
    fi
    now="$(date +%s)"
    if (( now - start_ts > VLLM_STARTUP_TIMEOUT )); then
      echo "ERROR: vLLM startup timed out: port=${port}; log=${log}" >&2
      tail -120 "${log}" >&2 || true
      exit 4
    fi
    sleep 10
  done
}

start_vllm_server() {
  local name="$1"
  local model_path="$2"
  local served_name="$3"
  local port="$4"
  local gpu_ids="$5"
  local tp_size="$6"
  local log="${RUNTIME_LOG_DIR}/${name}_vllm_${port}.log"
  local pgid

  if check_vllm "${port}"; then
    echo "ERROR: ${name} vLLM port ${port} is already serving a model; vLLM reuse is disabled for eval." >&2
    echo "       Stop the existing service or choose another ${name^^}_PORT before rerunning." >&2
    exit 4
  fi

  echo "starting ${name} vLLM server on GPUs ${gpu_ids}; model=${model_path}; log=${log}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    VLLM_DISABLE_FLASHINFER=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
    TOKENIZERS_PARALLELISM=false \
    "${PY}" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "${port}" \
      --model "${model_path}" \
      --served-model-name "${served_name}" \
      --tensor-parallel-size "${tp_size}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --trust-remote-code \
      --dtype bfloat16 \
      --enforce-eager > "${log}" 2>&1 &
  pgid="$!"
  if [[ "${name}" == "agent" ]]; then
    AGENT_PGID="${pgid}"
  else
    RERANKER_PGID="${pgid}"
  fi
  wait_for_vllm "${port}" "${pgid}" "${log}"
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "TASK_NAME=${TASK_NAME}"
  echo "TRACE_DIR=${TRACE_DIR}"
  echo "RUNTIME_LOG_DIR=${RUNTIME_LOG_DIR}"
  echo "REPORT_PATH=${REPORT_PATH}"
  echo "LLM_IO_JSONL=${LLM_IO_JSONL}"
  echo "RUN_MODE=${RUN_MODE}"
  echo "AGENT_MODEL=${AGENT_MODEL}"
  if [[ "${RUN_MODE}" == "no-ranker" ]]; then
    echo "RERANKER_MODEL=disabled"
  else
    echo "RERANKER_MODEL=${RERANKER_MODEL}"
  fi
  echo "AGENT_GPU_IDS=${AGENT_GPU_IDS}"
  echo "AGENT_TP_SIZE=${AGENT_TP_SIZE}"
  echo "RERANKER_GPU_IDS=${RERANKER_GPU_IDS}"
  echo "RERANKER_TP_SIZE=${RERANKER_TP_SIZE}"
  echo "RETRIEVER_MODE=${RETRIEVER_MODE}"
  echo "RETRIEVER_DEVICE=${RETRIEVER_DEVICE}"
  echo "RETRIEVER_GPU_ID=${RETRIEVER_GPU_ID}"
  echo "RETRIEVER_GPU_IDS=${RETRIEVER_GPU_IDS}"
  echo "RETRIEVER_LAUNCHER=${ROOT}/src/retrievers/start_dense_retriever_server.sh"
  echo "RETRIEVAL_PROXY=${ROOT}/src/retrievers/retrieval_round_robin_proxy.py"
  exit 0
fi

AGENT_MODEL_RESOLVED="$("${PY}" "${ROOT}/scripts/cosearch_local/evaluate_cosearch_vllm.py" resolve-model --path "${AGENT_MODEL}" --role agent)"
RERANKER_MODEL_RESOLVED=""
if [[ "${RUN_MODE}" != "no-ranker" ]]; then
  RERANKER_MODEL_RESOLVED="$("${PY}" "${ROOT}/scripts/cosearch_local/evaluate_cosearch_vllm.py" resolve-model --path "${RERANKER_MODEL}" --role reranker)"
fi

echo "resolved agent model: ${AGENT_MODEL_RESOLVED}"
if [[ "${RUN_MODE}" == "no-ranker" ]]; then
  echo "reranker disabled: RUN_MODE=no-ranker"
else
  echo "resolved reranker model: ${RERANKER_MODEL_RESOLVED}"
fi
echo "trace dir: ${TRACE_DIR}"
echo "runtime logs: ${RUNTIME_LOG_DIR}"
echo "report: ${REPORT_PATH}"

start_retrievers
start_proxy
start_vllm_server "agent" "${AGENT_MODEL_RESOLVED}" "${AGENT_SERVED_MODEL}" "${AGENT_PORT}" "${AGENT_GPU_IDS}" "${AGENT_TP_SIZE}"
if [[ "${RUN_MODE}" != "no-ranker" ]]; then
  start_vllm_server "reranker" "${RERANKER_MODEL_RESOLVED}" "${RERANKER_SERVED_MODEL}" "${RERANKER_PORT}" "${RERANKER_GPU_IDS}" "${RERANKER_TP_SIZE}"
fi

llm_io_args=()
if [[ -n "${LLM_IO_JSONL}" ]]; then
  llm_io_args+=(--llm-io-jsonl "${LLM_IO_JSONL}")
fi
eval_args=(
  run
  --run-mode "${RUN_MODE}"
  --agent-model "${AGENT_MODEL_RESOLVED}"
  --data-path "${DATA_PATH}"
  --max-eval-num "${MAX_EVAL_NUM}"
  --batch-size "${BATCH_SIZE}"
  --keep-trace "${KEEP_TRACE}"
  --trace-dir "${TRACE_DIR}"
  --report-path "${REPORT_PATH}"
  --strategy-name "${STRATEGY_NAME}"
  --retrieval-url "http://127.0.0.1:${PROXY_PORT}/retrieve"
  --agent-base-url "http://127.0.0.1:${AGENT_PORT}"
  --agent-served-model "${AGENT_SERVED_MODEL}"
  --reranker-served-model "${RERANKER_SERVED_MODEL}"
  --top-n "${TOP_N}"
  --top-m "${TOP_M}"
  --max-assistant-turns "${MAX_ASSISTANT_TURNS}"
  --max-user-turns "${MAX_USER_TURNS}"
  --max-tool-response-length "${MAX_TOOL_RESPONSE_LENGTH}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-response-length "${MAX_RESPONSE_LENGTH}"
  --reranker-max-prompt-length "${RERANKER_MAX_PROMPT_LENGTH}"
  --reranker-max-response-length "${RERANKER_MAX_RESPONSE_LENGTH}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --reranker-temperature "${RERANKER_TEMPERATURE}"
  --request-timeout "${REQUEST_TIMEOUT}"
)
if [[ "${RUN_MODE}" != "no-ranker" ]]; then
  eval_args+=(
    --reranker-model "${RERANKER_MODEL_RESOLVED}"
    --reranker-base-url "http://127.0.0.1:${RERANKER_PORT}"
  )
fi
eval_args+=("${llm_io_args[@]}")

"${PY}" "${ROOT}/scripts/cosearch_local/evaluate_cosearch_vllm.py" "${eval_args[@]}"

echo "evaluation complete"
echo "report: ${REPORT_PATH}"
echo "trace: ${TRACE_DIR}"
echo "runtime logs: ${RUNTIME_LOG_DIR}"

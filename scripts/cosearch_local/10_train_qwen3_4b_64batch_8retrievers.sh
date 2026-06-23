#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_SCHEMA_PATH="${REPORT_SCHEMA_PATH:-${SCRIPT_DIR}/assets/report_schema.py}"
EXP_NAME="${EXP_NAME:-}"
GROUP_NAME="${GROUP_NAME:-cosearch}"
resolve_cosearch_training_run_identity "${ROOT}" "" 1 "${GROUP_NAME}"
setup_cosearch_logging_defaults "${ROOT}" "${RUN_NAME}"

PORT_BASE="${PORT_BASE:-8020}"
PROXY_PORT="${PROXY_PORT:-8030}"
RETRIEVER_INSTANCES="${RETRIEVER_INSTANCES:-8}"
RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cpu}"
RETRIEVER_OMP_NUM_THREADS="${RETRIEVER_OMP_NUM_THREADS:-1}"
RETRIEVER_MKL_NUM_THREADS="${RETRIEVER_MKL_NUM_THREADS:-1}"
RETRIEVER_STARTUP_TIMEOUT="${RETRIEVER_STARTUP_TIMEOUT:-900}"
PROXY_TIMEOUT="${PROXY_TIMEOUT:-180}"
RETRIEVER_MIN_INSTANCES="${RETRIEVER_MIN_INSTANCES:-4}"
RETRIEVER_MEMORY_LIMIT_PCT="${RETRIEVER_MEMORY_LIMIT_PCT:-70}"
RETRIEVER_RSS_ESTIMATE_KB="${RETRIEVER_RSS_ESTIMATE_KB:-66870370}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-64}"
TOTAL_STEPS="${TOTAL_STEPS:-10}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-10}"
MAIN_GPU_IDS="${MAIN_GPU_IDS:-0,1,2,3}"
RERANKER_GPU_IDS="${RERANKER_GPU_IDS:-4,5,6,7}"
REPORT_STEPS="${REPORT_STEPS:-10}"

mkdir -p "${ROOT}/checkpoints/qwen3_4b_probe/${GROUP_SLUG}"

OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints/qwen3_4b_probe/${GROUP_SLUG}/${CONFIG_NAME}}"
cosearch_assert_safe_run_target "${LOG_DIR}" "log dir"
cosearch_assert_safe_run_target "${OUT_DIR}" "checkpoint dir"

RETRIEVER_PGIDS=()
PROXY_PGID=""
REPORTER_PGID=""
NVIDIA_SMI_PGID=""

cleanup() {
  if [[ -n "${NVIDIA_SMI_PGID}" ]] && kill -0 "${NVIDIA_SMI_PGID}" 2>/dev/null; then
    kill -TERM "${NVIDIA_SMI_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${REPORTER_PGID}" ]] && kill -0 "${REPORTER_PGID}" 2>/dev/null; then
    kill -TERM "${REPORTER_PGID}" 2>/dev/null || true
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

check_url() {
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
    if check_url "${url}"; then
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
    if (( (now - start_ts) % 60 < 5 )); then
      echo "waiting for retriever: ${url} elapsed=$((now - start_ts))s log=${log}"
    fi
    sleep 5
  done
}

projected_memory_pct_after_retriever_starts() {
  local pending_starts="${1:-1}"
  awk -v rss_kb="${RETRIEVER_RSS_ESTIMATE_KB}" -v pending_starts="${pending_starts}" '
    $1 == "MemTotal:" { total = $2 }
    $1 == "MemAvailable:" { available = $2 }
    END {
      if (total <= 0 || available <= 0) {
        print "100.00"
        exit
      }
      printf "%.2f", ((total - available + (rss_kb * pending_starts)) / total) * 100
    }
  ' /proc/meminfo
}

start_retrievers() {
  local i port log target_instances min_instances projected_pct pgid missing_count
  local wait_urls=()
  local wait_pgids=()
  local wait_logs=()
  target_instances="${RETRIEVER_INSTANCES}"
  min_instances="${RETRIEVER_MIN_INSTANCES}"
  if (( target_instances < min_instances )); then
    min_instances="${target_instances}"
  fi

  if (( target_instances > min_instances )); then
    missing_count=0
    for ((i = 0; i < target_instances; i++)); do
      port=$((PORT_BASE + i))
      if ! check_url "http://127.0.0.1:${port}/retrieve"; then
        missing_count=$((missing_count + 1))
      fi
    done
    projected_pct="$(projected_memory_pct_after_retriever_starts "${missing_count}")"
    if ! awk -v pct="${projected_pct}" -v limit="${RETRIEVER_MEMORY_LIMIT_PCT}" 'BEGIN { exit !(pct <= limit) }'; then
      echo "retriever memory guard: projected memory ${projected_pct}% exceeds limit ${RETRIEVER_MEMORY_LIMIT_PCT}% before starting ${missing_count} missing retrievers; using ${min_instances} retriever instances"
      target_instances="${min_instances}"
      RETRIEVER_INSTANCES="${min_instances}"
    fi
  fi

  for ((i = 0; i < target_instances; i++)); do
    port=$((PORT_BASE + i))
    log="${LOG_DIR}/${RUN_NAME}.retriever_${port}.log"
    if check_url "http://127.0.0.1:${port}/retrieve"; then
      echo "using existing retriever: http://127.0.0.1:${port}/retrieve"
      continue
    fi
    echo "starting retriever ${i}/${RETRIEVER_INSTANCES} on port ${port}; log=${log}"
    setsid env PY="${PY}" PORT="${port}" DEVICE="${RETRIEVER_DEVICE}" FAISS_GPU=0 \
      OMP_NUM_THREADS="${RETRIEVER_OMP_NUM_THREADS}" MKL_NUM_THREADS="${RETRIEVER_MKL_NUM_THREADS}" \
      bash "${ROOT}/src/retrievers/start_dense_retriever_server.sh" > "${log}" 2>&1 &
    pgid="$!"
    RETRIEVER_PGIDS+=("${pgid}")
    wait_urls+=("http://127.0.0.1:${port}/retrieve")
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
  for ((i = 0; i < RETRIEVER_INSTANCES; i++)); do
    port=$((PORT_BASE + i))
    args+=(--backend "http://127.0.0.1:${port}/retrieve")
  done
  echo "starting retrieval proxy on port ${PROXY_PORT}; backends=${RETRIEVER_INSTANCES}"
  setsid "${PY}" "${ROOT}/src/retrievers/retrieval_round_robin_proxy.py" \
    --host 127.0.0.1 \
    --port "${PROXY_PORT}" \
    --timeout "${PROXY_TIMEOUT}" \
    "${args[@]}" > "${LOG_DIR}/${RUN_NAME}.retrieval_proxy.log" 2>&1 &
  PROXY_PGID=$!

  local start_ts now
  start_ts="$(date +%s)"
  while true; do
    if check_url "http://127.0.0.1:${PROXY_PORT}/retrieve"; then
      echo "retrieval proxy ready: http://127.0.0.1:${PROXY_PORT}/retrieve"
      return
    fi
    if ! kill -0 "${PROXY_PGID}" 2>/dev/null; then
      echo "ERROR: retrieval proxy exited before ready" >&2
      tail -80 "${LOG_DIR}/${RUN_NAME}.retrieval_proxy.log" >&2 || true
      exit 3
    fi
    now="$(date +%s)"
    if (( now - start_ts > 120 )); then
      echo "ERROR: retrieval proxy startup timed out" >&2
      tail -80 "${LOG_DIR}/${RUN_NAME}.retrieval_proxy.log" >&2 || true
      exit 3
    fi
    if (( (now - start_ts) % 20 < 2 )); then
      echo "waiting for retrieval proxy: http://127.0.0.1:${PROXY_PORT}/retrieve elapsed=$((now - start_ts))s"
    fi
    sleep 2
  done
}

start_reporter() {
  cosearch_start_training_reporter "${ROOT}"
}

start_nvidia_smi_sampler() {
  cosearch_start_nvidia_smi_sampler
}

write_run_config() {
  cat > "${LOG_DIR}/${RUN_NAME}.env" <<EOF
EXP_NAME=${EXP_NAME}
GROUP_NAME=${GROUP_NAME}
GROUP_SLUG=${GROUP_SLUG}
RUN_STAMP=${RUN_STAMP:-}
RUN_NAME=${RUN_NAME}
CONFIG_NAME=${CONFIG_NAME}
GPU_IDS=${GPU_IDS}
LOG_DIR=${LOG_DIR}
PORT_BASE=${PORT_BASE}
PROXY_PORT=${PROXY_PORT}
RETRIEVER_INSTANCES=${RETRIEVER_INSTANCES}
RETRIEVER_MIN_INSTANCES=${RETRIEVER_MIN_INSTANCES}
RETRIEVER_MEMORY_LIMIT_PCT=${RETRIEVER_MEMORY_LIMIT_PCT}
RETRIEVER_RSS_ESTIMATE_KB=${RETRIEVER_RSS_ESTIMATE_KB}
RETRIEVER_DEVICE=${RETRIEVER_DEVICE}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE}
TOTAL_STEPS=${TOTAL_STEPS}
N_ROLLOUTS=${N_ROLLOUTS}
TOOL_MAX_CONCURRENT_PER_WORKER=${TOOL_MAX_CONCURRENT_PER_WORKER:-8}
TRAIN_LOG=${TRAIN_LOG}
METRICS_JSONL=${METRICS_JSONL}
SEARCH_TIMING_JSONL=${SEARCH_TIMING_JSONL}
NVIDIA_SMI_CSV=${NVIDIA_SMI_CSV}
NVIDIA_SMI_INTERVAL=${NVIDIA_SMI_INTERVAL}
MAIN_GPU_IDS=${MAIN_GPU_IDS}
RERANKER_GPU_IDS=${RERANKER_GPU_IDS}
REPORT_STEPS=${REPORT_STEPS}
REPORT_SCHEMA_PATH=${REPORT_SCHEMA_PATH}
HYDRA_OVERRIDE_YAMLS=${HYDRA_OVERRIDE_YAMLS:-}
COSEARCH_STRATEGY_YAML=${COSEARCH_STRATEGY_YAML:-}
COSEARCH_EXTRA_ARGS=${COSEARCH_EXTRA_ARGS:-}
OUT_DIR=${OUT_DIR}
ALLOW_RUN_REUSE=${ALLOW_RUN_REUSE:-0}
ALLOW_DIR_REUSE=${ALLOW_DIR_REUSE:-0}
RETRIEVER_LAUNCHER=${ROOT}/src/retrievers/start_dense_retriever_server.sh
RETRIEVAL_PROXY=${ROOT}/src/retrievers/retrieval_round_robin_proxy.py
EOF
}

write_run_config

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; configuration written to ${LOG_DIR}/${RUN_NAME}.env"
  echo "training log: ${TRAIN_LOG}"
  echo "metrics jsonl: ${METRICS_JSONL}"
  echo "search timing jsonl: ${SEARCH_TIMING_JSONL}"
  echo "reports: ${REPORT_STEPS}"
  echo "retriever launcher: ${ROOT}/src/retrievers/start_dense_retriever_server.sh"
  echo "retrieval proxy: ${ROOT}/src/retrievers/retrieval_round_robin_proxy.py"
  exit 0
fi

start_retrievers
start_proxy
start_reporter
start_nvidia_smi_sampler

export PY
export PORT="${PROXY_PORT}"
export GPU_IDS
export CONFIG_NAME
export MODEL_PATH="${MODEL_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B}"
export RETRIEVAL_SERVICE_URL="http://127.0.0.1:${PROXY_PORT}/retrieve"
export RERANKER_TRAINABLE="${RERANKER_TRAINABLE:-true}"
export OUT_DIR
export EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
export TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_rl_51k.train.parquet}"
export VAL_DATA="${VAL_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_7bench_smoke.eval.parquet}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-51200}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
export TRAIN_BATCH_SIZE
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export TOTAL_STEPS
export N_ROLLOUTS
export MAX_TURNS="${MAX_TURNS:-6}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-6}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-6}"
export TOP_N="${TOP_N:-50}"
export TOP_M="${TOP_M:-5}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-11264}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
export AGENT_WORKERS="${AGENT_WORKERS:-8}"
export TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-8}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export ACTOR_BATCH_SIZE
export ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export COSEARCH_ROLLOUT_PROGRESS_INTERVAL="${COSEARCH_ROLLOUT_PROGRESS_INTERVAL:-60}"
export COSEARCH_ROLLOUT_ITEM_PROGRESS_INTERVAL="${COSEARCH_ROLLOUT_ITEM_PROGRESS_INTERVAL:-32}"
export COSEARCH_ACTOR_PROGRESS_INTERVAL="${COSEARCH_ACTOR_PROGRESS_INTERVAL:-32}"
export COSEARCH_PROGRESS_ROLLOUT_N="${N_ROLLOUTS}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:--1}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollout_data}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUT_DIR}/validation_data}"
export TRAINER_LOGGER="['console','file']"
export VERL_FILE_LOGGER_PATH="${METRICS_JSONL}"
export COSEARCH_SEARCH_TIMING_JSONL="${SEARCH_TIMING_JSONL}"
export HYDRA_OVERRIDE_YAMLS="${HYDRA_OVERRIDE_YAMLS:-}"
export COSEARCH_STRATEGY_YAML="${COSEARCH_STRATEGY_YAML:-}"
export COSEARCH_EXTRA_ARGS="${COSEARCH_EXTRA_ARGS:-}"

set +e
bash "${ROOT}/scripts/cosearch_local/train_cosearch_verl_base.sh" "$@" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e

cosearch_generate_final_training_reports "${ROOT}"

exit "${TRAIN_STATUS}"

#!/usr/bin/env bash
set -euo pipefail

# AgenticIterRag v1 独立推理入口。
# 该脚本只消费 AIR pipeline compiler/runner 注入的 runtime env，不调用 CAR shell 链路。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}"
source "${ASSETS_DIR}/00_project_paths.sh"
setup_agent_iteration_paths "${ROOT}"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
source "${ASSETS_DIR}/00_air_accelerator.sh"
PROJECT_ROOT="${AGENTIC_ITER_RAG_PROJECT_ROOT:-${ROOT}/AgenticIterRag}"

if [[ "${AIR_INFER_PRECOMPILED_ENV:-0}" != "1" ]]; then
  echo "ERROR: AIR infer launcher requires AIR_INFER_PRECOMPILED_ENV=1 from AgenticIterRag pipeline runner." >&2
  echo "       Please start it via scripts/agenticIterRag_v1/01_pipeline_launcher.sh." >&2
  exit 2
fi
set --
INFER_ENGINE="${INFER_ENGINE:-${SCRIPT_DIR}/infer_air_vllm.py}"
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_ROOT}/config/agentic_iter_rag_tool_config.yaml}"


# common-use
GROUP_NAME="${GROUP_NAME:-agenticIterRag}"
resolve_air_group_identity "${GROUP_NAME}"
INFER_TASK_NAME="${INFER_TASK_NAME:-default}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
DEFAULT_AIR_INFER_DATA_PATH="${ROOT}/data/AgenticIterRag/source/co_search_ablation.infer.parquet"
DATA_PATH="${DATA_PATH:-${DEFAULT_AIR_INFER_DATA_PATH}}"
AGENT_MODEL="${AGENT_MODEL:-}"
MAX_INFER_NUM="${MAX_INFER_NUM:--1}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"



INFER_TASK_SLUG="${INFER_TASK_SLUG:-$(slugify_air_name "${INFER_TASK_NAME}")}"
TASK_NAME="${TASK_NAME:-$(date +%y%m%d-%H%M)-${INFER_TASK_SLUG}}"
INFER_LOG_ROOT="${INFER_LOG_ROOT:-${ROOT}/log/infer_res/${GROUP_SLUG}}"
INFER_REPORT_ROOT="${INFER_REPORT_ROOT:-${ROOT}/reports/infer/${GROUP_SLUG}}"
TRACE_DIR="${TRACE_DIR:-${INFER_LOG_ROOT}/${TASK_NAME}}"
REPORT_PATH="${REPORT_PATH:-${INFER_REPORT_ROOT}/${TASK_NAME}.report.md}"
RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-${TRACE_DIR}/runtime_logs}"
RUN_NAME="${RUN_NAME:-${INFER_TASK_SLUG}}"
EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
mkdir -p "${INFER_REPORT_ROOT}" "${TRACE_DIR}" "${RUNTIME_LOG_DIR}"

RUN_MODE="${RUN_MODE:-full}"
case "${RUN_MODE}" in
  ranker-only|full|no-ranker) ;;
  co-training)
    RUN_MODE="full"
    ;;
  *)
    echo "ERROR: unsupported RUN_MODE=${RUN_MODE}; use ranker-only, full, or no-ranker" >&2
    exit 2
    ;;
esac
RERANKER="${RERANKER:-dense_e5}"
case "${RERANKER}" in
  dense|e5|dense-e5)
    RERANKER="dense_e5"
    ;;
  llm-as-judge|llm_judge|judge)
    RERANKER="llm_as_judge"
    ;;
  dense_e5|llm_as_judge) ;;
  *)
    echo "ERROR: unsupported RERANKER=${RERANKER}; use dense_e5 or llm_as_judge" >&2
    exit 2
    ;;
esac

RECALL_MODEL_PATH="${RECALL_MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
RANKER_MODEL="${RANKER_MODEL:-}"
RANKER_BASE_MODEL="${RANKER_BASE_MODEL:-}"
RANKER_ENCODER_PATH="${RANKER_ENCODER_PATH:-}"

CORPUS_JSONL="${CORPUS_JSONL:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18/wiki-18.jsonl}"

MAX_INFER_STEPS="${MAX_INFER_STEPS:-1}"
MAX_RANKER_STEPS="${MAX_RANKER_STEPS:-${MAX_INFER_STEPS}}"
# 是否保持完整轨迹：full/partial
KEEP_TRACE="${KEEP_TRACE:-partial}"
# 增量落盘频率；默认每 10 条样本写一次前缀结果，降低长任务中断损失。
FLUSH_EVERY_N="${FLUSH_EVERY_N:-10}"

RECALL_FINAL_TOP_N="${RECALL_FINAL_TOP_N:-50}"
SEARCH_TOOL_FINAL_TOP_M="${SEARCH_TOOL_FINAL_TOP_M:-5}"
RANKER_FINAL_TOP_K="${RANKER_FINAL_TOP_K:-${RECALL_FINAL_TOP_N}}"

PROXY_PORT="${PROXY_PORT:-8030}"
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PROXY_PORT}/retrieve}"
RECALL_GPU_ID="${RECALL_GPU_ID:-5}"
RANK_GPU_ID="${RANK_GPU_ID:-4}"
RANKER_CUDA_VISIBLE_DEVICES="${RANKER_CUDA_VISIBLE_DEVICES:-${RANK_GPU_ID}}"
AGENT_GPU_IDS="${AGENT_GPU_IDS:-6}"
AGENT_TP_SIZE="${AGENT_TP_SIZE:-$(awk -F',' '{print NF}' <<< "${AGENT_GPU_IDS}")}"
AGENT_PORT="${AGENT_PORT:-8040}"
AGENT_SERVED_MODEL="${AGENT_SERVED_MODEL:-air-agent}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${INFER_BATCH_SIZE}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-6}"
MAX_USER_TURNS="${MAX_USER_TURNS:-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-11264}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"

AGENT_MAX_RETRIES="${AGENT_MAX_RETRIES:-3}"
AGENT_RETRY_DELAY="${AGENT_RETRY_DELAY:-1.0}"
AGENT_RETRY_BACKOFF="${AGENT_RETRY_BACKOFF:-2.0}"
AGENT_HTTP_FORCE_CLOSE="${AGENT_HTTP_FORCE_CLOSE:-true}"
FAIL_ON_INFER_ERROR="${FAIL_ON_INFER_ERROR:-true}"

RETRIEVAL_MAX_RETRIES="${RETRIEVAL_MAX_RETRIES:-1}"
RETRIEVAL_RETRY_DELAY="${RETRIEVAL_RETRY_DELAY:-0.5}"
RETRIEVAL_RETRY_BACKOFF="${RETRIEVAL_RETRY_BACKOFF:-1.0}"
AUTO_START_RECALL_SERVICE="${AUTO_START_RECALL_SERVICE:-1}"
AUTO_STOP_RECALL_SERVICE="${AUTO_STOP_RECALL_SERVICE:-1}"
RECALL_SERVICE_WAIT_SECONDS="${RECALL_SERVICE_WAIT_SECONDS:-240}"
RECALL_BACKEND_BASE_PORT="${RECALL_BACKEND_BASE_PORT:-}"
RETRIEVAL_PREFLIGHT_QUERY="${RETRIEVAL_PREFLIGHT_QUERY:-who got the first nobel prize in physics?}"
RETRIEVAL_PREFLIGHT_EXPECT="${RETRIEVAL_PREFLIGHT_EXPECT:-}"

RANKER_DEVICE="${RANKER_DEVICE:-$(air_accel_device_spec 0)}"
RANKER_MAX_QUERY_LENGTH="${RANKER_MAX_QUERY_LENGTH:-192}"
RANKER_MAX_DOC_LENGTH="${RANKER_MAX_DOC_LENGTH:-256}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-http://127.0.0.1:8067/v1/chat/completions}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-DeepSeek-V4-Flash}"
LLM_JUDGE_PROMPT_PATH="${LLM_JUDGE_PROMPT_PATH:-${PROJECT_ROOT}/agentic_iter_rag/llm_reranker/prompts/air_rerank_tags_v1.md}"
LLM_JUDGE_MAX_CHUNK_CHARS="${LLM_JUDGE_MAX_CHUNK_CHARS:-512}"
LLM_JUDGE_MAX_TOKENS="${LLM_JUDGE_MAX_TOKENS:-1024}"
LLM_JUDGE_TEMPERATURE="${LLM_JUDGE_TEMPERATURE:-0.0}"
LLM_JUDGE_REQUEST_TIMEOUT="${LLM_JUDGE_REQUEST_TIMEOUT:-600}"
LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"
LLM_JUDGE_RETRY_DELAY="${LLM_JUDGE_RETRY_DELAY:-2.0}"
LLM_JUDGE_RETRY_BACKOFF="${LLM_JUDGE_RETRY_BACKOFF:-2.0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
INJECT_TOOL_SCHEMA="${INJECT_TOOL_SCHEMA:-false}"
RANKER_CONFIG_DEVICE="${RANKER_CONFIG_DEVICE:-${RANKER_DEVICE}}"
STOP_SEQUENCES="${STOP_SEQUENCES:-}"
LLM_IO_MAX_RECORDS="${LLM_IO_MAX_RECORDS:-20}"


OUT_DIR="${OUT_DIR:-${TRACE_DIR}}"
LOG_DIR="${LOG_DIR:-${RUNTIME_LOG_DIR}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUT_DIR}/validation_data}"
METRICS_JSONL="${METRICS_JSONL:-${LOG_DIR}/${RUN_NAME}.metrics.jsonl}"
AGENT_TIMING_JSONL="${AGENT_TIMING_JSONL:-${LOG_DIR}/${RUN_NAME}.agent_timing.jsonl}"
SEARCH_TIMING_JSONL="${SEARCH_TIMING_JSONL:-${LOG_DIR}/${RUN_NAME}.search_timing.jsonl}"
LLM_IO_JSONL="${LLM_IO_JSONL:-${LOG_DIR}/${RUN_NAME}.llm_io.jsonl}"
INFER_LOG="${INFER_LOG:-${LOG_DIR}/${RUN_NAME}.infer.log}"
RECALL_SERVICE_LOG="${RECALL_SERVICE_LOG:-${LOG_DIR}/${RUN_NAME}.recall_retriever_server.log}"
RECALL_PROXY_LOG="${RECALL_PROXY_LOG:-${RECALL_SERVICE_LOG%.log}.proxy.log}"
RANKER_OUTPUT_JSONL="${RANKER_OUTPUT_JSONL:-${OUT_DIR}/ranker_infer_smoke.jsonl}"
ENV_PATH="${ENV_PATH:-${LOG_DIR}/${RUN_NAME}.env}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}"

RECALL_SERVICE_PID=""
RECALL_PROXY_PID=""
RECALL_BACKEND_PIDS=()
RECALL_BACKEND_LOGS=()
AGENT_PGID=""

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  if [[ -n "${AGENT_PGID}" ]] && kill -0 "-${AGENT_PGID}" 2>/dev/null; then
    kill -TERM "-${AGENT_PGID}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "-${AGENT_PGID}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "-${AGENT_PGID}" 2>/dev/null; then
      kill -KILL "-${AGENT_PGID}" 2>/dev/null || true
    fi
    wait "${AGENT_PGID}" 2>/dev/null || true
  fi
  if is_truthy "${AUTO_STOP_RECALL_SERVICE}"; then
    if [[ -n "${RECALL_PROXY_PID}" ]] && kill -0 "${RECALL_PROXY_PID}" 2>/dev/null; then
      kill -TERM "${RECALL_PROXY_PID}" 2>/dev/null || true
      wait "${RECALL_PROXY_PID}" 2>/dev/null || true
    fi
    for pid in "${RECALL_BACKEND_PIDS[@]:-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
      fi
    done
  fi
  if [[ -n "${RECALL_SERVICE_PID}" ]] && is_truthy "${AUTO_STOP_RECALL_SERVICE}"; then
    if kill -0 "${RECALL_SERVICE_PID}" 2>/dev/null; then
      kill -TERM "${RECALL_SERVICE_PID}" 2>/dev/null || true
      wait "${RECALL_SERVICE_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

validate_recall_preflight_args() {
  if ! [[ "${RECALL_FINAL_TOP_N}" =~ ^[0-9]+$ ]] || (( RECALL_FINAL_TOP_N < 1 )); then
    echo "ERROR: RECALL_FINAL_TOP_N must be a positive integer; got ${RECALL_FINAL_TOP_N}" >&2
    exit 2
  fi
  if ! [[ "${SEARCH_TOOL_FINAL_TOP_M}" =~ ^[0-9]+$ ]] || (( SEARCH_TOOL_FINAL_TOP_M < 1 )); then
    echo "ERROR: SEARCH_TOOL_FINAL_TOP_M must be a positive integer; got ${SEARCH_TOOL_FINAL_TOP_M}" >&2
    exit 2
  fi
  if (( SEARCH_TOOL_FINAL_TOP_M > RECALL_FINAL_TOP_N )); then
    echo "ERROR: SEARCH_TOOL_FINAL_TOP_M=${SEARCH_TOOL_FINAL_TOP_M} exceeds RECALL_FINAL_TOP_N=${RECALL_FINAL_TOP_N}" >&2
    exit 2
  fi
  if (( SEARCH_TOOL_FINAL_TOP_M > 5 )); then
    echo "ERROR: SEARCH_TOOL_FINAL_TOP_M=${SEARCH_TOOL_FINAL_TOP_M} is invalid for current reward preflight; answer_match_reward supports at most 5 visible documents." >&2
    echo "       SEARCH_TOOL_FINAL_TOP_M is agent-visible docs. Do not pass ranker cutoffs here." >&2
    exit 2
  fi
}

check_recall_http_ready() {
  check_recall_url_ready "${RETRIEVAL_SERVICE_URL}"
}

check_recall_url_ready() {
  local url="$1"
  "${PY}" - "${url}" "${RETRIEVAL_PREFLIGHT_QUERY}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url, query = sys.argv[1:3]
payload = json.dumps({"queries": [query], "topk": 1, "return_scores": False}).encode("utf-8")
request = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status >= 500:
            print(f"recall service returned HTTP {response.status}", file=sys.stderr)
            raise SystemExit(2)
        data = json.loads(response.read().decode("utf-8"))
        if "result" not in data:
            print("recall service response missing result", file=sys.stderr)
            raise SystemExit(2)
except urllib.error.HTTPError as exc:
    if exc.code >= 500:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        print(f"recall service returned HTTP {exc.code}: {body}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_recall_url() {
  local url="$1"
  local label="$2"
  local waited=0 status
  while [[ "${waited}" -lt "${RECALL_SERVICE_WAIT_SECONDS}" ]]; do
    if check_recall_url_ready "${url}"; then
      status=0
    else
      status=$?
    fi
    if (( status == 0 )); then
      echo "recall retrieval ${label} ready: ${url}"
      return 0
    fi
    if (( status == 2 )); then
      echo "ERROR: recall retrieval ${label} returned a fatal readiness error: ${url}" >&2
      return 2
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "ERROR: timed out waiting for recall retrieval ${label} after ${RECALL_SERVICE_WAIT_SECONDS}s: ${url}" >&2
  return 1
}

run_recall_preflight() {
  local output status
  if output="$("${PY}" "${ASSETS_DIR}/00_check_air_tool_retrieval.py" \
      --url "${RETRIEVAL_SERVICE_URL}" \
      --query "${RETRIEVAL_PREFLIGHT_QUERY}" \
      --top-n "${RECALL_FINAL_TOP_N}" \
      --top-m "${SEARCH_TOOL_FINAL_TOP_M}" \
      --expect-contains "${RETRIEVAL_PREFLIGHT_EXPECT}" 2>&1)"; then
    echo "recall retrieval semantic preflight passed: top_n=${RECALL_FINAL_TOP_N} top_m=${SEARCH_TOOL_FINAL_TOP_M}"
    return 0
  fi
  status=$?
  printf '%s\n' "${output}" >&2
  return "${status}"
}

start_single_recall_service() {
  echo "starting recall retrieval service; accelerator=${AIR_ACCELERATOR}; device_id=${RECALL_GPU_ID}; log=${RECALL_SERVICE_LOG}"
  PORT="${PROXY_PORT}" \
  RECALL_GPU_ID="${RECALL_GPU_ID}" \
  RETRIEVER_GPU_IDS="${RECALL_GPU_ID}" \
  RETRIEVER_MODEL="${RECALL_MODEL_PATH}" \
  RECALL_FINAL_TOP_N="${RECALL_FINAL_TOP_N}" \
  DEVICE="${RETRIEVER_DEVICE:-$(air_accel_device_prefix)}" \
  AIR_ACCELERATOR="${AIR_ACCELERATOR}" \
  PY="${PY}" \
    bash "${SCRIPT_DIR}/00_start_dense_retriever_server.sh" >"${RECALL_SERVICE_LOG}" 2>&1 &
  RECALL_SERVICE_PID=$!

  local waited=0
  while [[ "${waited}" -lt "${RECALL_SERVICE_WAIT_SECONDS}" ]]; do
    if check_recall_http_ready; then
      if ! run_recall_preflight; then
        echo "ERROR: recall retrieval semantic preflight failed; aborting instead of retrying readiness." >&2
        exit 2
      fi
      echo "recall retrieval service ready: ${RETRIEVAL_SERVICE_URL}"
      return 0
    else
      ready_status=$?
      if (( ready_status == 2 )); then
        echo "ERROR: recall retrieval service returned a fatal readiness error; aborting instead of waiting." >&2
        tail -80 "${RECALL_SERVICE_LOG}" >&2 || true
        exit 2
      fi
    fi
    if ! kill -0 "${RECALL_SERVICE_PID}" 2>/dev/null; then
      echo "ERROR: recall retrieval service exited before ready. Log tail:" >&2
      tail -80 "${RECALL_SERVICE_LOG}" >&2 || true
      exit 2
    fi
    sleep 2
    waited=$((waited + 2))
  done

  echo "ERROR: timed out waiting for recall retrieval service after ${RECALL_SERVICE_WAIT_SECONDS}s. Log tail:" >&2
  tail -80 "${RECALL_SERVICE_LOG}" >&2 || true
  exit 2
}

start_multi_recall_service() {
  local raw_gpu_ids="$1"
  local -a gpu_ids backend_urls
  local backend_base_port backend_count idx gpu_id backend_port backend_url backend_log
  IFS=',' read -r -a gpu_ids <<< "${raw_gpu_ids}"
  backend_count="${#gpu_ids[@]}"
  backend_base_port="${RECALL_BACKEND_BASE_PORT:-$((PROXY_PORT + 1))}"
  : > "${RECALL_SERVICE_LOG}"
  echo "starting ${backend_count} recall retrieval backends in parallel; accelerator=${AIR_ACCELERATOR}; devices=${raw_gpu_ids}; proxy_port=${PROXY_PORT}; backend_base_port=${backend_base_port}" | tee -a "${RECALL_SERVICE_LOG}"

  for idx in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[$idx]}"
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ -z "${gpu_id}" ]]; then
      echo "ERROR: empty gpu id in RECALL_GPU_ID=${raw_gpu_ids}" >&2
      exit 2
    fi
    backend_port=$((backend_base_port + idx))
    backend_url="http://127.0.0.1:${backend_port}/retrieve"
    backend_log="${RECALL_SERVICE_LOG%.log}.gpu${gpu_id}.port${backend_port}.log"
    backend_urls+=("${backend_url}")
    RECALL_BACKEND_LOGS+=("${backend_log}")
    echo "starting recall backend: gpu=${gpu_id}; port=${backend_port}; url=${backend_url}; log=${backend_log}" | tee -a "${RECALL_SERVICE_LOG}"
    PORT="${backend_port}" \
    RECALL_GPU_ID="${gpu_id}" \
    RETRIEVER_GPU_IDS="${gpu_id}" \
    RETRIEVER_MODEL="${RECALL_MODEL_PATH}" \
    RECALL_FINAL_TOP_N="${RECALL_FINAL_TOP_N}" \
    DEVICE="${RETRIEVER_DEVICE:-$(air_accel_device_prefix)}" \
    AIR_ACCELERATOR="${AIR_ACCELERATOR}" \
    PY="${PY}" \
      bash "${SCRIPT_DIR}/00_start_dense_retriever_server.sh" >"${backend_log}" 2>&1 &
    RECALL_BACKEND_PIDS+=("$!")
  done

  local waited=0 all_ready status
  while [[ "${waited}" -lt "${RECALL_SERVICE_WAIT_SECONDS}" ]]; do
    all_ready=1
    for idx in "${!backend_urls[@]}"; do
      if ! kill -0 "${RECALL_BACKEND_PIDS[$idx]}" 2>/dev/null; then
        echo "ERROR: recall backend exited before ready: url=${backend_urls[$idx]}; log=${RECALL_BACKEND_LOGS[$idx]}" >&2
        tail -80 "${RECALL_BACKEND_LOGS[$idx]}" >&2 || true
        exit 2
      fi
      if check_recall_url_ready "${backend_urls[$idx]}"; then
        status=0
      else
        status=$?
      fi
      if (( status != 0 )); then
        if (( status == 2 )); then
          echo "ERROR: recall backend returned a fatal readiness error: url=${backend_urls[$idx]}; log=${RECALL_BACKEND_LOGS[$idx]}" >&2
          tail -80 "${RECALL_BACKEND_LOGS[$idx]}" >&2 || true
          exit 2
        fi
        all_ready=0
      fi
    done
    if (( all_ready == 1 )); then
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done
  if (( all_ready != 1 )); then
    echo "ERROR: timed out waiting for recall backends after ${RECALL_SERVICE_WAIT_SECONDS}s" >&2
    for idx in "${!backend_urls[@]}"; do
      echo "--- backend ${backend_urls[$idx]} log tail: ${RECALL_BACKEND_LOGS[$idx]}" >&2
      tail -80 "${RECALL_BACKEND_LOGS[$idx]}" >&2 || true
    done
    exit 2
  fi

  echo "starting recall round-robin proxy: port=${PROXY_PORT}; log=${RECALL_PROXY_LOG}" | tee -a "${RECALL_SERVICE_LOG}"
  local -a proxy_args
  proxy_args=(--host 127.0.0.1 --port "${PROXY_PORT}" --timeout "${REQUEST_TIMEOUT}")
  for backend_url in "${backend_urls[@]}"; do
    proxy_args+=(--backend "${backend_url}")
  done
  "${PY}" "${ROOT}/src/retrievers/retrieval_round_robin_proxy.py" "${proxy_args[@]}" >"${RECALL_PROXY_LOG}" 2>&1 &
  RECALL_PROXY_PID=$!

  if ! wait_for_recall_url "${RETRIEVAL_SERVICE_URL}" "proxy"; then
    echo "ERROR: recall proxy failed to become ready. Log tail:" >&2
    tail -80 "${RECALL_PROXY_LOG}" >&2 || true
    exit 2
  fi
  if ! run_recall_preflight; then
    echo "ERROR: recall retrieval semantic preflight failed through proxy; aborting." >&2
    exit 2
  fi
  echo "recall retrieval proxy ready: ${RETRIEVAL_SERVICE_URL}"
}

ensure_recall_service() {
  validate_recall_preflight_args
  if check_recall_http_ready; then
    if ! run_recall_preflight; then
      echo "ERROR: recall retrieval semantic preflight failed; aborting instead of retrying readiness." >&2
      exit 2
    fi
    echo "recall retrieval service ready: ${RETRIEVAL_SERVICE_URL}"
    return 0
  else
    ready_status=$?
    if (( ready_status == 2 )); then
      echo "ERROR: recall retrieval service returned a fatal readiness error; aborting instead of waiting." >&2
      tail -80 "${RECALL_SERVICE_LOG}" >&2 || true
      exit 2
    fi
  fi
  if ! is_truthy "${AUTO_START_RECALL_SERVICE}"; then
    echo "ERROR: recall retrieval service is unavailable and AUTO_START_RECALL_SERVICE=${AUTO_START_RECALL_SERVICE}" >&2
    echo "       url=${RETRIEVAL_SERVICE_URL}" >&2
    exit 2
  fi

  if [[ "${RECALL_GPU_ID}" == *,* ]]; then
    start_multi_recall_service "${RECALL_GPU_ID}"
  else
    start_single_recall_service
  fi
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
      return 0
    fi
    if ! kill -0 "-${pgid}" 2>/dev/null; then
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

start_agent_vllm() {
  local model_path="$1"
  local log="${RUNTIME_LOG_DIR}/agent_vllm_${AGENT_PORT}.log"
  if check_vllm "${AGENT_PORT}"; then
    echo "ERROR: agent vLLM port ${AGENT_PORT} is already serving a model; vLLM reuse is disabled for infer." >&2
    echo "       Stop the existing service or choose another AGENT_PORT before rerunning." >&2
    exit 4
  fi
  echo "starting agent vLLM server on ${AIR_ACCELERATOR} devices ${AGENT_GPU_IDS}; model=${model_path}; log=${log}"
  setsid env $(air_accel_env_visible_devices_cmd "${AGENT_GPU_IDS}") \
    VLLM_DISABLE_FLASHINFER=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
    TOKENIZERS_PARALLELISM=false \
    "${PY}" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "${AGENT_PORT}" \
      --model "${model_path}" \
      --served-model-name "${AGENT_SERVED_MODEL}" \
      --tensor-parallel-size "${AGENT_TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --trust-remote-code \
      --dtype bfloat16 \
      --enforce-eager > "${log}" 2>&1 &
  AGENT_PGID="$!"
  wait_for_vllm "${AGENT_PORT}" "${AGENT_PGID}" "${log}"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required ${label} not found: ${path}" >&2
    exit 2
  fi
}

check_paths() {
  require_path "${PROJECT_ROOT}" "project root"
  require_path "${INFER_ENGINE}" "infer engine"
  require_path "${DATA_PATH}" "infer data"
  require_path "${CORPUS_JSONL}" "retrieval corpus"
  require_path "${RECALL_MODEL_PATH}" "recall model"
  if [[ "${RUN_MODE}" != "ranker-only" ]]; then
    if [[ -z "${AGENT_MODEL}" ]]; then
      echo "ERROR: AGENT_MODEL must be explicitly set for RUN_MODE=${RUN_MODE}; no default agent model is allowed in infer." >&2
      exit 2
    fi
    require_path "${AGENT_MODEL}" "agent model"
  fi
  if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "dense_e5" ]]; then
    if [[ -z "${RANKER_MODEL}" ]]; then
      echo "ERROR: RANKER_MODEL must be explicitly set for RUN_MODE=${RUN_MODE}; no default ranker model is allowed in infer." >&2
      exit 2
    fi
    if [[ -z "${RANKER_BASE_MODEL}" ]]; then
      echo "ERROR: RANKER_BASE_MODEL must be explicitly set for RUN_MODE=${RUN_MODE}; use the tokenizer/base model such as e5-base-v2." >&2
      exit 2
    fi
    require_path "${RANKER_MODEL}" "ranker model"
    require_path "${RANKER_BASE_MODEL}" "ranker base model"
    if [[ -n "${RANKER_ENCODER_PATH}" ]]; then
      require_path "${RANKER_ENCODER_PATH}" "ranker encoder"
    fi
  fi
  if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "llm_as_judge" ]]; then
    if [[ -z "${LLM_JUDGE_ENDPOINT}" || -z "${LLM_JUDGE_MODEL}" ]]; then
      echo "ERROR: RERANKER=llm_as_judge requires LLM_JUDGE_ENDPOINT and LLM_JUDGE_MODEL." >&2
      exit 2
    fi
    require_path "${LLM_JUDGE_PROMPT_PATH}" "LLM judge prompt"
  fi
}

count_jsonl_rows() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

write_shell_report() {
  local status="${1:-unknown}"
  local metrics_rows agent_timing_rows timing_rows llm_io_rows ranker_rows ranker_enabled_label ranker_model_label ranker_base_model_label ranker_encoder_label
  metrics_rows="$(count_jsonl_rows "${METRICS_JSONL}")"
  agent_timing_rows="$(count_jsonl_rows "${AGENT_TIMING_JSONL}")"
  timing_rows="$(count_jsonl_rows "${SEARCH_TIMING_JSONL}")"
  llm_io_rows="$(count_jsonl_rows "${LLM_IO_JSONL}")"
  ranker_rows="$(count_jsonl_rows "${RANKER_OUTPUT_JSONL}")"
  ranker_enabled_label="true"
  ranker_model_label="${RANKER_MODEL}"
  ranker_base_model_label="${RANKER_BASE_MODEL}"
  ranker_encoder_label="${RANKER_ENCODER_PATH:-auto}"
  if [[ "${RUN_MODE}" == "no-ranker" ]]; then
    ranker_enabled_label="false"
    ranker_model_label="not used"
    ranker_base_model_label="not used"
    ranker_encoder_label="not used"
  fi
  if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "llm_as_judge" ]]; then
    ranker_model_label="not used"
    ranker_base_model_label="not used"
    ranker_encoder_label="not used"
  fi
  mkdir -p "$(dirname "${REPORT_PATH}")"
  cat > "${REPORT_PATH}" <<EOF
# AgenticIterRag v1 Infer Report

## Run

- Status: ${status}
- Group: ${GROUP_NAME}
- Group slug: ${GROUP_SLUG}
- Task: ${TASK_NAME}
- Infer task: ${INFER_TASK_NAME}
- Infer task slug: ${INFER_TASK_SLUG}
- Run name: ${RUN_NAME}
- Run mode: ${RUN_MODE}
- Reranker: ${RERANKER}
- Dataset: ${DATA_PATH}
- Trace dir: ${TRACE_DIR}
- Runtime logs: ${RUNTIME_LOG_DIR}

## Models

- Agent model: ${AGENT_MODEL}
- Recall model: ${RECALL_MODEL_PATH}
- Ranker enabled: ${ranker_enabled_label}
- Ranker model: ${ranker_model_label}
- Ranker base model: ${ranker_base_model_label}
- Ranker encoder path: ${ranker_encoder_label}
- LLM judge endpoint: ${LLM_JUDGE_ENDPOINT}
- LLM judge model: ${LLM_JUDGE_MODEL}

## Artifacts

- Config env: ${ENV_PATH}
- Infer log: ${INFER_LOG}
- Recall service log: ${RECALL_SERVICE_LOG}
- Metrics JSONL: ${METRICS_JSONL} (${metrics_rows} rows)
- Agent timing JSONL: ${AGENT_TIMING_JSONL} (${agent_timing_rows} rows)
- Search timing JSONL: ${SEARCH_TIMING_JSONL} (${timing_rows} rows)
- LLM IO JSONL: ${LLM_IO_JSONL} (${llm_io_rows} rows)
- LLM IO max records: ${LLM_IO_MAX_RECORDS}
- Ranker output JSONL: ${RANKER_OUTPUT_JSONL} (${ranker_rows} rows)
- Validation data dir: ${VALIDATION_DATA_DIR}
- Rollout data dir: ${ROLLOUT_DATA_DIR}
- Tool config: ${TOOL_CONFIG}
- Infer budget config: ${INFER_BUDGET_CONFIG:-unknown}

## Key Config

- RECALL_FINAL_TOP_N: ${RECALL_FINAL_TOP_N}
- SEARCH_TOOL_FINAL_TOP_M: ${SEARCH_TOOL_FINAL_TOP_M}
- RANKER_FINAL_TOP_K: ${RANKER_FINAL_TOP_K}
- MAX_INFER_NUM: ${MAX_INFER_NUM}
- INFER_BATCH_SIZE: ${INFER_BATCH_SIZE}
- FLUSH_EVERY_N: ${FLUSH_EVERY_N}
- ENABLE_THINKING: ${ENABLE_THINKING}
- MAX_MODEL_LEN: ${MAX_MODEL_LEN}
- STOP_SEQUENCES: ${STOP_SEQUENCES:-none}
- AIR_ACCELERATOR: ${AIR_ACCELERATOR}
- $(air_accel_visible_devices_var): ${AGENT_GPU_IDS}
- AGENT_GPU_IDS: ${AGENT_GPU_IDS}
- RANK_GPU_ID: ${RANK_GPU_ID}
- RANKER_CUDA_VISIBLE_DEVICES: ${RANKER_CUDA_VISIBLE_DEVICES}
- RANKER_DEVICE: ${RANKER_DEVICE}
- LLM_JUDGE_ENDPOINT: ${LLM_JUDGE_ENDPOINT}
- LLM_JUDGE_MODEL: ${LLM_JUDGE_MODEL}
- RECALL_GPU_ID: ${RECALL_GPU_ID}
- RECALL_BACKEND_BASE_PORT: ${RECALL_BACKEND_BASE_PORT}
- RECALL_PROXY_LOG: ${RECALL_PROXY_LOG}
- RETRIEVAL_SERVICE_URL: ${RETRIEVAL_SERVICE_URL}
- AGENT_MAX_RETRIES: ${AGENT_MAX_RETRIES}
- AGENT_RETRY_DELAY: ${AGENT_RETRY_DELAY}
- AGENT_RETRY_BACKOFF: ${AGENT_RETRY_BACKOFF}
- AGENT_HTTP_FORCE_CLOSE: ${AGENT_HTTP_FORCE_CLOSE}
- FAIL_ON_INFER_ERROR: ${FAIL_ON_INFER_ERROR}
- RETRIEVAL_MAX_RETRIES: ${RETRIEVAL_MAX_RETRIES}
- RETRIEVAL_RETRY_DELAY: ${RETRIEVAL_RETRY_DELAY}
- RETRIEVAL_RETRY_BACKOFF: ${RETRIEVAL_RETRY_BACKOFF}
EOF
}

write_env_file() {
  cat > "${ENV_PATH}" <<EOF
TASK_NAME=${TASK_NAME}
RUN_NAME=${RUN_NAME}
EXP_NAME=${EXP_NAME}
GROUP_NAME=${GROUP_NAME}
GROUP_SLUG=${GROUP_SLUG}
INFER_TASK_NAME=${INFER_TASK_NAME}
INFER_TASK_SLUG=${INFER_TASK_SLUG}
RUN_MODE=${RUN_MODE}
RERANKER=${RERANKER}
AIR_ACCELERATOR=${AIR_ACCELERATOR}
VISIBLE_DEVICES_VAR=$(air_accel_visible_devices_var)
$(air_accel_visible_devices_var)=${AGENT_GPU_IDS}
PROJECT_ROOT=${PROJECT_ROOT}
PY=${PY}
INFER_ENGINE=${INFER_ENGINE}
AGENT_MODEL=${AGENT_MODEL}
RECALL_MODEL_PATH=${RECALL_MODEL_PATH}
RANKER_MODEL=${RANKER_MODEL}
RANKER_BASE_MODEL=${RANKER_BASE_MODEL}
RANKER_ENCODER_PATH=${RANKER_ENCODER_PATH}
DATA_PATH=${DATA_PATH}
MAX_INFER_NUM=${MAX_INFER_NUM}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE}
MAX_RANKER_STEPS=${MAX_RANKER_STEPS}
KEEP_TRACE=${KEEP_TRACE}
RECALL_FINAL_TOP_N=${RECALL_FINAL_TOP_N}
SEARCH_TOOL_FINAL_TOP_M=${SEARCH_TOOL_FINAL_TOP_M}
RANKER_FINAL_TOP_K=${RANKER_FINAL_TOP_K}
TRACE_DIR=${TRACE_DIR}
OUT_DIR=${OUT_DIR}
REPORT_PATH=${REPORT_PATH}
RUNTIME_LOG_DIR=${RUNTIME_LOG_DIR}
LOG_DIR=${LOG_DIR}
INFER_LOG=${INFER_LOG}
ENV_PATH=${ENV_PATH}
METRICS_JSONL=${METRICS_JSONL}
AGENT_TIMING_JSONL=${AGENT_TIMING_JSONL}
SEARCH_TIMING_JSONL=${SEARCH_TIMING_JSONL}
LLM_IO_JSONL=${LLM_IO_JSONL}
LLM_IO_MAX_RECORDS=${LLM_IO_MAX_RECORDS}
RANKER_OUTPUT_JSONL=${RANKER_OUTPUT_JSONL}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}
RETRIEVAL_SERVICE_URL=${RETRIEVAL_SERVICE_URL}
RECALL_SERVICE_LOG=${RECALL_SERVICE_LOG}
AUTO_START_RECALL_SERVICE=${AUTO_START_RECALL_SERVICE}
AUTO_STOP_RECALL_SERVICE=${AUTO_STOP_RECALL_SERVICE}
AGENT_GPU_IDS=${AGENT_GPU_IDS}
AGENT_TP_SIZE=${AGENT_TP_SIZE}
AGENT_PORT=${AGENT_PORT}
AGENT_SERVED_MODEL=${AGENT_SERVED_MODEL}
RANK_GPU_ID=${RANK_GPU_ID}
RANKER_CUDA_VISIBLE_DEVICES=${RANKER_CUDA_VISIBLE_DEVICES}
RECALL_GPU_ID=${RECALL_GPU_ID}
RECALL_BACKEND_BASE_PORT=${RECALL_BACKEND_BASE_PORT}
RECALL_PROXY_LOG=${RECALL_PROXY_LOG}
RANKER_DEVICE=${RANKER_DEVICE}
RANKER_MAX_QUERY_LENGTH=${RANKER_MAX_QUERY_LENGTH}
RANKER_MAX_DOC_LENGTH=${RANKER_MAX_DOC_LENGTH}
LLM_JUDGE_ENDPOINT=${LLM_JUDGE_ENDPOINT}
LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}
LLM_JUDGE_PROMPT_PATH=${LLM_JUDGE_PROMPT_PATH}
LLM_JUDGE_MAX_CHUNK_CHARS=${LLM_JUDGE_MAX_CHUNK_CHARS}
LLM_JUDGE_MAX_TOKENS=${LLM_JUDGE_MAX_TOKENS}
LLM_JUDGE_TEMPERATURE=${LLM_JUDGE_TEMPERATURE}
LLM_JUDGE_REQUEST_TIMEOUT=${LLM_JUDGE_REQUEST_TIMEOUT}
LLM_JUDGE_MAX_RETRIES=${LLM_JUDGE_MAX_RETRIES}
LLM_JUDGE_RETRY_DELAY=${LLM_JUDGE_RETRY_DELAY}
LLM_JUDGE_RETRY_BACKOFF=${LLM_JUDGE_RETRY_BACKOFF}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
ENABLE_THINKING=${ENABLE_THINKING}
TOOL_CONFIG=${TOOL_CONFIG}
INFER_RUNTIME_CONFIG=${INFER_RUNTIME_CONFIG:-}
INFER_BUDGET_CONFIG=${INFER_BUDGET_CONFIG:-}
STOP_SEQUENCES=${STOP_SEQUENCES}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS}
MAX_USER_TURNS=${MAX_USER_TURNS}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT}
AGENT_MAX_RETRIES=${AGENT_MAX_RETRIES}
AGENT_RETRY_DELAY=${AGENT_RETRY_DELAY}
AGENT_RETRY_BACKOFF=${AGENT_RETRY_BACKOFF}
AGENT_HTTP_FORCE_CLOSE=${AGENT_HTTP_FORCE_CLOSE}
FAIL_ON_INFER_ERROR=${FAIL_ON_INFER_ERROR}
RETRIEVAL_MAX_RETRIES=${RETRIEVAL_MAX_RETRIES}
RETRIEVAL_RETRY_DELAY=${RETRIEVAL_RETRY_DELAY}
RETRIEVAL_RETRY_BACKOFF=${RETRIEVAL_RETRY_BACKOFF}
FLUSH_EVERY_N=${FLUSH_EVERY_N}
EOF
}

ranker_args=()
if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "dense_e5" ]]; then
  ranker_args+=(--ranker-model "${RANKER_MODEL}")
  ranker_args+=(--ranker-base-model "${RANKER_BASE_MODEL}")
  if [[ -n "${RANKER_ENCODER_PATH}" ]]; then
    ranker_args+=(--ranker-encoder "${RANKER_ENCODER_PATH}")
  fi
fi

llm_judge_args=()
if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "llm_as_judge" ]]; then
  llm_judge_args+=(--llm-judge-endpoint "${LLM_JUDGE_ENDPOINT}")
  llm_judge_args+=(--llm-judge-model "${LLM_JUDGE_MODEL}")
  llm_judge_args+=(--llm-judge-prompt-path "${LLM_JUDGE_PROMPT_PATH}")
  llm_judge_args+=(--llm-judge-max-chunk-chars "${LLM_JUDGE_MAX_CHUNK_CHARS}")
  llm_judge_args+=(--llm-judge-max-tokens "${LLM_JUDGE_MAX_TOKENS}")
  llm_judge_args+=(--llm-judge-temperature "${LLM_JUDGE_TEMPERATURE}")
  llm_judge_args+=(--llm-judge-request-timeout "${LLM_JUDGE_REQUEST_TIMEOUT}")
  llm_judge_args+=(--llm-judge-max-retries "${LLM_JUDGE_MAX_RETRIES}")
  llm_judge_args+=(--llm-judge-retry-delay "${LLM_JUDGE_RETRY_DELAY}")
  llm_judge_args+=(--llm-judge-retry-backoff "${LLM_JUDGE_RETRY_BACKOFF}")
fi

llm_io_args=()
if [[ -n "${LLM_IO_JSONL}" ]]; then
  llm_io_args+=(--llm-io-jsonl "${LLM_IO_JSONL}")
fi
llm_io_args+=(--llm-io-max-records "${LLM_IO_MAX_RECORDS}")

check_paths
write_env_file

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "TASK_NAME=${TASK_NAME}"
  echo "TRACE_DIR=${TRACE_DIR}"
  echo "RUNTIME_LOG_DIR=${RUNTIME_LOG_DIR}"
  echo "REPORT_PATH=${REPORT_PATH}"
  echo "INFER_TASK_NAME=${INFER_TASK_NAME}"
  echo "INFER_TASK_SLUG=${INFER_TASK_SLUG}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "RUN_MODE=${RUN_MODE}"
  echo "RERANKER=${RERANKER}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "MAX_INFER_NUM=${MAX_INFER_NUM}"
  echo "INFER_BATCH_SIZE=${INFER_BATCH_SIZE}"
  echo "FLUSH_EVERY_N=${FLUSH_EVERY_N}"
  echo "ENABLE_THINKING=${ENABLE_THINKING}"
  echo "INJECT_TOOL_SCHEMA=${INJECT_TOOL_SCHEMA}"
  echo "MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS}"
  echo "MAX_USER_TURNS=${MAX_USER_TURNS}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
  echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
  echo "MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH}"
  echo "AGENT_MODEL=${AGENT_MODEL}"
  echo "RECALL_MODEL_PATH=${RECALL_MODEL_PATH}"
  echo "RANKER_MODEL=${RANKER_MODEL}"
  echo "RANKER_ENCODER_PATH=${RANKER_ENCODER_PATH:-auto}"
  echo "LLM_JUDGE_ENDPOINT=${LLM_JUDGE_ENDPOINT}"
  echo "LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}"
  echo "LLM_JUDGE_PROMPT_PATH=${LLM_JUDGE_PROMPT_PATH}"
  echo "AIR_ACCELERATOR=${AIR_ACCELERATOR}"
  echo "$(air_accel_visible_devices_var)=${AGENT_GPU_IDS}"
  echo "AGENT_GPU_IDS=${AGENT_GPU_IDS}"
  echo "RANK_GPU_ID=${RANK_GPU_ID}"
  echo "RANKER_CUDA_VISIBLE_DEVICES=${RANKER_CUDA_VISIBLE_DEVICES}"
  echo "RANKER_DEVICE=${RANKER_DEVICE}"
  echo "RECALL_GPU_ID=${RECALL_GPU_ID}"
  echo "METRICS_JSONL=${METRICS_JSONL}"
  echo "AGENT_TIMING_JSONL=${AGENT_TIMING_JSONL}"
  echo "SEARCH_TIMING_JSONL=${SEARCH_TIMING_JSONL}"
  echo "LLM_IO_JSONL=${LLM_IO_JSONL}"
  echo "LLM_IO_MAX_RECORDS=${LLM_IO_MAX_RECORDS}"
  echo "TOOL_CONFIG=${TOOL_CONFIG}"
  echo "INFER_BUDGET_CONFIG=${INFER_BUDGET_CONFIG:-unknown}"
  echo "STOP_SEQUENCES=${STOP_SEQUENCES:-none}"
  write_shell_report "dry-run"
  exit 0
fi

AGENT_MODEL_RESOLVED=""
if [[ "${RUN_MODE}" != "ranker-only" ]]; then
  AGENT_MODEL_RESOLVED="$("${PY}" "${INFER_ENGINE}" resolve-model --path "${AGENT_MODEL}" --role agent)"
  echo "resolved agent model: ${AGENT_MODEL_RESOLVED}"
fi
if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "dense_e5" ]]; then
  RANKER_RESOLVED_JSON="$("${PY}" "${INFER_ENGINE}" resolve-ranker "${ranker_args[@]}")"
  echo "resolved ranker: ${RANKER_RESOLVED_JSON}"
fi

echo "trace dir: ${TRACE_DIR}"
echo "runtime logs: ${RUNTIME_LOG_DIR}"
echo "report: ${REPORT_PATH}"

ensure_recall_service
if [[ "${RUN_MODE}" != "ranker-only" ]]; then
  start_agent_vllm "${AGENT_MODEL_RESOLVED}"
fi

agent_args=()
if [[ "${RUN_MODE}" != "ranker-only" ]]; then
  agent_args+=(--agent-model "${AGENT_MODEL_RESOLVED}")
  agent_args+=(--agent-base-url "http://127.0.0.1:${AGENT_PORT}")
fi

trust_remote_code_arg="--trust-remote-code"
if ! is_truthy "${TRUST_REMOTE_CODE}"; then
  trust_remote_code_arg="--no-trust-remote-code"
fi

enable_thinking_arg="--enable-thinking"
if ! is_truthy "${ENABLE_THINKING}"; then
  enable_thinking_arg="--no-enable-thinking"
fi

inject_tool_schema_arg="--inject-tool-schema"
if ! is_truthy "${INJECT_TOOL_SCHEMA}"; then
  inject_tool_schema_arg="--no-inject-tool-schema"
fi

agent_http_force_close_arg="--agent-http-force-close"
if ! is_truthy "${AGENT_HTTP_FORCE_CLOSE}"; then
  agent_http_force_close_arg="--no-agent-http-force-close"
fi

fail_on_error_arg="--fail-on-error"
if ! is_truthy "${FAIL_ON_INFER_ERROR}"; then
  fail_on_error_arg="--no-fail-on-error"
fi

stop_sequence_args=()
if [[ -n "${STOP_SEQUENCES}" ]]; then
  IFS=',' read -r -a _stop_sequences <<< "${STOP_SEQUENCES}"
  for stop_sequence in "${_stop_sequences[@]}"; do
    if [[ -n "${stop_sequence}" ]]; then
      stop_sequence_args+=(--stop-sequence "${stop_sequence}")
    fi
  done
fi

infer_engine_env=()
infer_engine_env+=(PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}")

set +e
env "${infer_engine_env[@]}" "${PY}" "${INFER_ENGINE}" run \
  --run-mode "${RUN_MODE}" \
  --reranker "${RERANKER}" \
  "${agent_args[@]}" \
  "${ranker_args[@]}" \
  "${llm_judge_args[@]}" \
  --data-path "${DATA_PATH}" \
  --max-infer-num "${MAX_INFER_NUM}" \
  --max-ranker-steps "${MAX_RANKER_STEPS}" \
  --batch-size "${INFER_BATCH_SIZE}" \
  --keep-trace "${KEEP_TRACE}" \
  --flush-every-n "${FLUSH_EVERY_N}" \
  --trace-dir "${TRACE_DIR}" \
  --report-path "${REPORT_PATH}" \
  --infer-task-name "${INFER_TASK_NAME}" \
  --retrieval-url "${RETRIEVAL_SERVICE_URL}" \
  --agent-served-model "${AGENT_SERVED_MODEL}" \
  --top-n "${RECALL_FINAL_TOP_N}" \
  --top-m "${SEARCH_TOOL_FINAL_TOP_M}" \
  --ranker-top-k "${RANKER_FINAL_TOP_K}" \
  --max-assistant-turns "${MAX_ASSISTANT_TURNS}" \
  --max-user-turns "${MAX_USER_TURNS}" \
  --max-tool-response-length "${MAX_TOOL_RESPONSE_LENGTH}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --max-model-len "${MAX_MODEL_LEN}" \
	  --temperature "${TEMPERATURE}" \
	  --top-p "${TOP_P}" \
	  --request-timeout "${REQUEST_TIMEOUT}" \
	  --agent-max-retries "${AGENT_MAX_RETRIES}" \
	  --agent-retry-delay "${AGENT_RETRY_DELAY}" \
	  --agent-retry-backoff "${AGENT_RETRY_BACKOFF}" \
	  "${agent_http_force_close_arg}" \
	  --max-retries "${RETRIEVAL_MAX_RETRIES}" \
	  --retry-delay "${RETRIEVAL_RETRY_DELAY}" \
	  --retry-backoff "${RETRIEVAL_RETRY_BACKOFF}" \
	  --metrics-jsonl "${METRICS_JSONL}" \
	  --agent-timing-jsonl "${AGENT_TIMING_JSONL}" \
	  --search-timing-jsonl "${SEARCH_TIMING_JSONL}" \
  --ranker-output-jsonl "${RANKER_OUTPUT_JSONL}" \
  --validation-data-dir "${VALIDATION_DATA_DIR}" \
  --rollout-data-dir "${ROLLOUT_DATA_DIR}" \
  --ranker-device "${RANKER_DEVICE}" \
  --ranker-max-query-length "${RANKER_MAX_QUERY_LENGTH}" \
  --ranker-max-doc-length "${RANKER_MAX_DOC_LENGTH}" \
  --tool-config-path "${TOOL_CONFIG}" \
  "${llm_io_args[@]}" \
  "${stop_sequence_args[@]}" \
	  "${trust_remote_code_arg}" \
	  "${enable_thinking_arg}" \
	  "${inject_tool_schema_arg}" \
	  "${fail_on_error_arg}" \
	  "$@" 2>&1 | tee "${INFER_LOG}"
INFER_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${INFER_STATUS}" -ne 0 ]]; then
  write_shell_report "exit_${INFER_STATUS}"
fi

echo "inference complete"
echo "report: ${REPORT_PATH}"
echo "trace: ${TRACE_DIR}"
echo "runtime logs: ${RUNTIME_LOG_DIR}"

exit "${INFER_STATUS}"

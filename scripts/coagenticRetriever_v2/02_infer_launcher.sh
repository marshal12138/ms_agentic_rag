#!/usr/bin/env bash
set -euo pipefail

# vLLM-only CoAgenticRetriever local evaluation entry.
# This script starts the recall retriever and, when needed, an agent vLLM
# server. It does not run VERL and does not require checkpoint resume.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
source "${ASSETS_DIR}/00_project_paths.sh"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
setup_agent_iteration_paths "${ROOT}"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_accelerator.sh"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}"

CONFIG_COMPILER="${ASSETS_DIR}/eval_launcher/compile_config.py"
if [[ ! -f "${CONFIG_COMPILER}" ]]; then
  echo "ERROR: eval launcher config compiler not found: ${CONFIG_COMPILER}" >&2
  exit 2
fi

EVAL_RUNTIME_ENV_SH="$("${PY}" "${CONFIG_COMPILER}" \
  --repo-root "${ROOT}" \
  --script-dir "${SCRIPT_DIR}" \
  --assets-dir "${ASSETS_DIR}" \
  --project-root "${PROJECT_ROOT}" \
  --external-model-root "${EXTERNAL_MODEL_ROOT}" \
  --external-retrieval-root "${EXTERNAL_RETRIEVAL_ROOT}" \
  --device-prefix "$(co_accel_device_prefix)" \
  --visible-devices-var "$(co_accel_visible_devices_var)" \
  --accelerator "${COSEARCH_ACCELERATOR}" \
  -- "$@")"

if [[ -z "${EVAL_RUNTIME_ENV_SH}" || ! -f "${EVAL_RUNTIME_ENV_SH}" ]]; then
  echo "ERROR: eval config compiler did not produce a source-able runtime env file." >&2
  echo "       output=${EVAL_RUNTIME_ENV_SH}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${EVAL_RUNTIME_ENV_SH}"
if [[ -n "${EVAL_PASSTHROUGH_ARGS_FILE:-}" && -f "${EVAL_PASSTHROUGH_ARGS_FILE}" ]]; then
  mapfile -t EVAL_PASSTHROUGH_ARGS < "${EVAL_PASSTHROUGH_ARGS_FILE}"
  set -- "${EVAL_PASSTHROUGH_ARGS[@]}"
else
  set --
fi
EVALUATOR="${EVALUATOR:-${SCRIPT_DIR}/evaluate_coagentic_vllm.py}"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}}"
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml}"


# common-use
GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"
resolve_coagentic_group_identity "${GROUP_NAME}"
EVAL_TASK_NAME="${EVAL_TASK_NAME:-default}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
DEFAULT_COAGENTIC_DATA_PATH="${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet"
DATA_PATH="${DATA_PATH:-${DEFAULT_COAGENTIC_DATA_PATH}}"
AGENT_MODEL="${AGENT_MODEL:-}"
MAX_EVAL_NUM="${MAX_EVAL_NUM:--1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"



EVAL_TASK_SLUG="${EVAL_TASK_SLUG:-$(slugify_cosearch_name "${EVAL_TASK_NAME}")}"
TASK_NAME="${TASK_NAME:-$(date +%y%m%d-%H%M)-${EVAL_TASK_SLUG}}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${ROOT}/log/eval_res/${GROUP_SLUG}}"
EVAL_REPORT_ROOT="${EVAL_REPORT_ROOT:-${ROOT}/reports/eval/${GROUP_SLUG}}"
TRACE_DIR="${TRACE_DIR:-${EVAL_LOG_ROOT}/${TASK_NAME}}"
REPORT_PATH="${REPORT_PATH:-${EVAL_REPORT_ROOT}/${TASK_NAME}.report.md}"
RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-${TRACE_DIR}/runtime_logs}"
RUN_NAME="${RUN_NAME:-${EVAL_TASK_SLUG}}"
EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
mkdir -p "${EVAL_REPORT_ROOT}" "${TRACE_DIR}" "${RUNTIME_LOG_DIR}"

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

MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-1}"
MAX_RANKER_STEPS="${MAX_RANKER_STEPS:-${MAX_EVAL_STEPS}}"
# 是否保持完整轨迹：full/partial
KEEP_TRACE="${KEEP_TRACE:-partial}"

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
AGENT_SERVED_MODEL="${AGENT_SERVED_MODEL:-cosearch-agent}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${EVAL_BATCH_SIZE}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-6}"
MAX_USER_TURNS="${MAX_USER_TURNS:-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-11264}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"

RETRIEVAL_MAX_RETRIES="${RETRIEVAL_MAX_RETRIES:-1}"
RETRIEVAL_RETRY_DELAY="${RETRIEVAL_RETRY_DELAY:-0.5}"
RETRIEVAL_RETRY_BACKOFF="${RETRIEVAL_RETRY_BACKOFF:-1.0}"
AUTO_START_RECALL_SERVICE="${AUTO_START_RECALL_SERVICE:-1}"
AUTO_STOP_RECALL_SERVICE="${AUTO_STOP_RECALL_SERVICE:-1}"
RECALL_SERVICE_WAIT_SECONDS="${RECALL_SERVICE_WAIT_SECONDS:-240}"
RETRIEVAL_PREFLIGHT_QUERY="${RETRIEVAL_PREFLIGHT_QUERY:-who got the first nobel prize in physics?}"
RETRIEVAL_PREFLIGHT_EXPECT="${RETRIEVAL_PREFLIGHT_EXPECT:-}"

RANKER_DEVICE="${RANKER_DEVICE:-$(co_accel_device_spec 0)}"
RANKER_MAX_QUERY_LENGTH="${RANKER_MAX_QUERY_LENGTH:-192}"
RANKER_MAX_DOC_LENGTH="${RANKER_MAX_DOC_LENGTH:-256}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-http://127.0.0.1:8067/v1/chat/completions}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-DeepSeek-V4-Flash}"
LLM_JUDGE_PROMPT_PATH="${LLM_JUDGE_PROMPT_PATH:-${PROJECT_ROOT}/async_ranker_training/prompts/llm_judge_rank50_v1.md}"
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
SEARCH_TIMING_JSONL="${SEARCH_TIMING_JSONL:-${LOG_DIR}/${RUN_NAME}.search_timing.jsonl}"
LLM_IO_JSONL="${LLM_IO_JSONL:-${LOG_DIR}/${RUN_NAME}.llm_io.jsonl}"
INFER_LOG="${INFER_LOG:-${LOG_DIR}/${RUN_NAME}.infer.log}"
RECALL_SERVICE_LOG="${RECALL_SERVICE_LOG:-${LOG_DIR}/${RUN_NAME}.recall_retriever_server.log}"
RANKER_OUTPUT_JSONL="${RANKER_OUTPUT_JSONL:-${OUT_DIR}/ranker_infer_smoke.jsonl}"
ENV_PATH="${ENV_PATH:-${LOG_DIR}/${RUN_NAME}.env}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}"

RECALL_SERVICE_PID=""
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
    wait "${AGENT_PGID}" 2>/dev/null || true
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
  "${PY}" - "${RETRIEVAL_SERVICE_URL}" "${RETRIEVAL_PREFLIGHT_QUERY}" <<'PY'
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

run_recall_preflight() {
  local output status
  if output="$("${PY}" "${ASSETS_DIR}/00_check_coagentic_tool_retrieval.py" \
      --project-root "${PROJECT_ROOT}" \
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

  echo "starting recall retrieval service; accelerator=${COSEARCH_ACCELERATOR}; device_id=${RECALL_GPU_ID}; log=${RECALL_SERVICE_LOG}"
  PORT="${PROXY_PORT}" \
  RECALL_GPU_ID="${RECALL_GPU_ID}" \
  RETRIEVER_GPU_IDS="${RECALL_GPU_ID}" \
  RETRIEVER_MODEL="${RECALL_MODEL_PATH}" \
  RECALL_FINAL_TOP_N="${RECALL_FINAL_TOP_N}" \
  DEVICE="${RETRIEVER_DEVICE:-$(co_accel_device_prefix)}" \
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
    echo "ERROR: agent vLLM port ${AGENT_PORT} is already serving a model; vLLM reuse is disabled for eval." >&2
    echo "       Stop the existing service or choose another AGENT_PORT before rerunning." >&2
    exit 4
  fi
  echo "starting agent vLLM server on ${COSEARCH_ACCELERATOR} devices ${AGENT_GPU_IDS}; model=${model_path}; log=${log}"
  setsid env $(co_accel_env_visible_devices_cmd "${AGENT_GPU_IDS}") \
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
  require_path "${EVALUATOR}" "evaluator"
  require_path "${DATA_PATH}" "eval data"
  require_path "${CORPUS_JSONL}" "retrieval corpus"
  require_path "${RECALL_MODEL_PATH}" "recall model"
  if [[ "${RUN_MODE}" != "ranker-only" ]]; then
    if [[ -z "${AGENT_MODEL}" ]]; then
      echo "ERROR: AGENT_MODEL must be explicitly set for RUN_MODE=${RUN_MODE}; no default agent model is allowed in eval." >&2
      exit 2
    fi
    require_path "${AGENT_MODEL}" "agent model"
  fi
  if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "dense_e5" ]]; then
    if [[ -z "${RANKER_MODEL}" ]]; then
      echo "ERROR: RANKER_MODEL must be explicitly set for RUN_MODE=${RUN_MODE}; no default ranker model is allowed in eval." >&2
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
  local metrics_rows timing_rows llm_io_rows ranker_rows ranker_enabled_label ranker_model_label ranker_base_model_label ranker_encoder_label
  metrics_rows="$(count_jsonl_rows "${METRICS_JSONL}")"
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
# CoAgenticRetriever vLLM Evaluation Report

## Run

- Status: ${status}
- Group: ${GROUP_NAME}
- Group slug: ${GROUP_SLUG}
- Task: ${TASK_NAME}
- Eval task: ${EVAL_TASK_NAME}
- Eval task slug: ${EVAL_TASK_SLUG}
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
- Search timing JSONL: ${SEARCH_TIMING_JSONL} (${timing_rows} rows)
- LLM IO JSONL: ${LLM_IO_JSONL} (${llm_io_rows} rows)
- LLM IO max records: ${LLM_IO_MAX_RECORDS}
- Ranker output JSONL: ${RANKER_OUTPUT_JSONL} (${ranker_rows} rows)
- Validation data dir: ${VALIDATION_DATA_DIR}
- Rollout data dir: ${ROLLOUT_DATA_DIR}
- Tool config: ${TOOL_CONFIG}
- Eval budget config: ${EVAL_BUDGET_CONFIG:-unknown}

## Key Config

- RECALL_FINAL_TOP_N: ${RECALL_FINAL_TOP_N}
- SEARCH_TOOL_FINAL_TOP_M: ${SEARCH_TOOL_FINAL_TOP_M}
- RANKER_FINAL_TOP_K: ${RANKER_FINAL_TOP_K}
- MAX_EVAL_NUM: ${MAX_EVAL_NUM}
- EVAL_BATCH_SIZE: ${EVAL_BATCH_SIZE}
- ENABLE_THINKING: ${ENABLE_THINKING}
- MAX_MODEL_LEN: ${MAX_MODEL_LEN}
- STOP_SEQUENCES: ${STOP_SEQUENCES:-none}
- COSEARCH_ACCELERATOR: ${COSEARCH_ACCELERATOR}
- $(co_accel_visible_devices_var): ${AGENT_GPU_IDS}
- AGENT_GPU_IDS: ${AGENT_GPU_IDS}
- RANK_GPU_ID: ${RANK_GPU_ID}
- RANKER_CUDA_VISIBLE_DEVICES: ${RANKER_CUDA_VISIBLE_DEVICES}
- RANKER_DEVICE: ${RANKER_DEVICE}
- LLM_JUDGE_ENDPOINT: ${LLM_JUDGE_ENDPOINT}
- LLM_JUDGE_MODEL: ${LLM_JUDGE_MODEL}
- RECALL_GPU_ID: ${RECALL_GPU_ID}
- RETRIEVAL_SERVICE_URL: ${RETRIEVAL_SERVICE_URL}
EOF
}

write_env_file() {
  cat > "${ENV_PATH}" <<EOF
TASK_NAME=${TASK_NAME}
RUN_NAME=${RUN_NAME}
EXP_NAME=${EXP_NAME}
GROUP_NAME=${GROUP_NAME}
GROUP_SLUG=${GROUP_SLUG}
EVAL_TASK_NAME=${EVAL_TASK_NAME}
EVAL_TASK_SLUG=${EVAL_TASK_SLUG}
RUN_MODE=${RUN_MODE}
RERANKER=${RERANKER}
COSEARCH_ACCELERATOR=${COSEARCH_ACCELERATOR}
VISIBLE_DEVICES_VAR=$(co_accel_visible_devices_var)
$(co_accel_visible_devices_var)=${AGENT_GPU_IDS}
PROJECT_ROOT=${PROJECT_ROOT}
PY=${PY}
EVALUATOR=${EVALUATOR}
AGENT_MODEL=${AGENT_MODEL}
RECALL_MODEL_PATH=${RECALL_MODEL_PATH}
RANKER_MODEL=${RANKER_MODEL}
RANKER_BASE_MODEL=${RANKER_BASE_MODEL}
RANKER_ENCODER_PATH=${RANKER_ENCODER_PATH}
DATA_PATH=${DATA_PATH}
MAX_EVAL_NUM=${MAX_EVAL_NUM}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}
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
EVAL_BUDGET_CONFIG=${EVAL_BUDGET_CONFIG}
EVAL_BUDGET_CONFIG_FILE=${EVAL_BUDGET_CONFIG_FILE}
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
  echo "EVAL_TASK_NAME=${EVAL_TASK_NAME}"
  echo "EVAL_TASK_SLUG=${EVAL_TASK_SLUG}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "RUN_MODE=${RUN_MODE}"
  echo "RERANKER=${RERANKER}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "MAX_EVAL_NUM=${MAX_EVAL_NUM}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
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
  echo "COSEARCH_ACCELERATOR=${COSEARCH_ACCELERATOR}"
  echo "$(co_accel_visible_devices_var)=${AGENT_GPU_IDS}"
  echo "AGENT_GPU_IDS=${AGENT_GPU_IDS}"
  echo "RANK_GPU_ID=${RANK_GPU_ID}"
  echo "RANKER_CUDA_VISIBLE_DEVICES=${RANKER_CUDA_VISIBLE_DEVICES}"
  echo "RANKER_DEVICE=${RANKER_DEVICE}"
  echo "RECALL_GPU_ID=${RECALL_GPU_ID}"
  echo "METRICS_JSONL=${METRICS_JSONL}"
  echo "SEARCH_TIMING_JSONL=${SEARCH_TIMING_JSONL}"
  echo "LLM_IO_JSONL=${LLM_IO_JSONL}"
  echo "LLM_IO_MAX_RECORDS=${LLM_IO_MAX_RECORDS}"
  echo "TOOL_CONFIG=${TOOL_CONFIG}"
  echo "EVAL_BUDGET_CONFIG=${EVAL_BUDGET_CONFIG}"
  echo "STOP_SEQUENCES=${STOP_SEQUENCES:-none}"
  write_shell_report "dry-run"
  exit 0
fi

AGENT_MODEL_RESOLVED=""
if [[ "${RUN_MODE}" != "ranker-only" ]]; then
  AGENT_MODEL_RESOLVED="$("${PY}" "${EVALUATOR}" resolve-model --path "${AGENT_MODEL}" --role agent)"
  echo "resolved agent model: ${AGENT_MODEL_RESOLVED}"
fi
if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "dense_e5" ]]; then
  RANKER_RESOLVED_JSON="$("${PY}" "${EVALUATOR}" resolve-ranker "${ranker_args[@]}")"
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

stop_sequence_args=()
if [[ -n "${STOP_SEQUENCES}" ]]; then
  IFS=',' read -r -a _stop_sequences <<< "${STOP_SEQUENCES}"
  for stop_sequence in "${_stop_sequences[@]}"; do
    if [[ -n "${stop_sequence}" ]]; then
      stop_sequence_args+=(--stop-sequence "${stop_sequence}")
    fi
  done
fi

evaluator_env=()
evaluator_env+=(PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}")

set +e
env "${evaluator_env[@]}" "${PY}" "${EVALUATOR}" run \
  --run-mode "${RUN_MODE}" \
  --reranker "${RERANKER}" \
  "${agent_args[@]}" \
  "${ranker_args[@]}" \
  "${llm_judge_args[@]}" \
  --data-path "${DATA_PATH}" \
  --max-eval-num "${MAX_EVAL_NUM}" \
  --max-ranker-steps "${MAX_RANKER_STEPS}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --keep-trace "${KEEP_TRACE}" \
  --trace-dir "${TRACE_DIR}" \
  --report-path "${REPORT_PATH}" \
  --eval-task-name "${EVAL_TASK_NAME}" \
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
  --max-retries "${RETRIEVAL_MAX_RETRIES}" \
  --retry-delay "${RETRIEVAL_RETRY_DELAY}" \
  --retry-backoff "${RETRIEVAL_RETRY_BACKOFF}" \
  --metrics-jsonl "${METRICS_JSONL}" \
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
  "$@" 2>&1 | tee "${INFER_LOG}"
INFER_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${INFER_STATUS}" -ne 0 ]]; then
  write_shell_report "exit_${INFER_STATUS}"
fi

echo "evaluation complete"
echo "report: ${REPORT_PATH}"
echo "trace: ${TRACE_DIR}"
echo "runtime logs: ${RUNTIME_LOG_DIR}"

exit "${INFER_STATUS}"

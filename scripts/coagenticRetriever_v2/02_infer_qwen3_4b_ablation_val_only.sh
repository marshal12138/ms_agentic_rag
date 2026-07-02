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
EVALUATOR="${EVALUATOR:-${SCRIPT_DIR}/evaluate_coagentic_vllm.py}"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}"
TOOL_CONFIG="${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml"
EVAL_BUDGET_YAML="${EVAL_BUDGET_YAML:-${ROOT}/scripts/coagenticRetriever_local/strategies_yaml/rollout_cosearch_aligned_budget.yaml}"

load_eval_budget_config() {
  if [[ -z "${EVAL_BUDGET_YAML}" ]]; then
    return 0
  fi
  if [[ ! -f "${EVAL_BUDGET_YAML}" ]]; then
    echo "ERROR: EVAL_BUDGET_YAML not found: ${EVAL_BUDGET_YAML}" >&2
    exit 2
  fi

  local parsed
  parsed="$("${PY}" - "${EVAL_BUDGET_YAML}" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except ModuleNotFoundError:
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}

def get(path, default=""):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def emit(name, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = ""
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")

emit("BUDGET_MAX_PROMPT_LENGTH", get("data.max_prompt_length"))
emit("BUDGET_MAX_RESPONSE_LENGTH", get("data.max_response_length"))
emit("BUDGET_ENABLE_THINKING", get("data.apply_chat_template_kwargs.enable_thinking"))
emit("BUDGET_MAX_MODEL_LEN", get("actor_rollout_ref.rollout.max_model_len"))
emit("BUDGET_MAX_ASSISTANT_TURNS", get("actor_rollout_ref.rollout.multi_turn.max_assistant_turns"))
emit("BUDGET_MAX_USER_TURNS", get("actor_rollout_ref.rollout.multi_turn.max_user_turns"))
emit("BUDGET_MAX_TOOL_RESPONSE_LENGTH", get("actor_rollout_ref.rollout.multi_turn.max_tool_response_length"))
PY
)"
  eval "${parsed}"
}

load_eval_budget_config


# common-use
GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"
resolve_coagentic_group_identity "${GROUP_NAME}"
STRATEGY_NAME="${STRATEGY_NAME:-default}" # 评估对象的策略名，作为评估任务的核心id
ENABLE_THINKING="${ENABLE_THINKING:-${BUDGET_ENABLE_THINKING:-false}}"
DEFAULT_COAGENTIC_DATA_PATH="${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet"
DATA_PATH="${DATA_PATH:-${DEFAULT_COAGENTIC_DATA_PATH}}"
AGENT_MODEL="${AGENT_MODEL:-${MODEL_PATH:-}}"
MAX_EVAL_NUM="${MAX_EVAL_NUM:-${VAL_MAX_SAMPLES:--1}}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${MAX_EVAL_NUM}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"



setup_coagentic_eval_artifact_defaults "${ROOT}" "${STRATEGY_NAME}"
RUN_NAME="${RUN_NAME:-${STRATEGY_NAME}}"
EXP_NAME="${EXP_NAME:-${RUN_NAME}}"

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
RERANKER="${RERANKER:-${reranker:-dense_e5}}"
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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
MODEL_SOURCE_FROM_CHECKPOINT="${RESUME_FROM_PATH:-${CHECKPOINT_DIR:-}}"

MODEL_PATH="${MODEL_PATH:-${AGENT_MODEL}}"
RECALL_MODEL_PATH="${RECALL_MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
RANKER_MODEL="${RANKER_MODEL:-${RANKER_MODEL_PATH:-}}"
RANKER_BASE_MODEL="${RANKER_BASE_MODEL:-${RANKER_BASE_MODEL_PATH:-}}"
RANKER_ENCODER_PATH="${RANKER_ENCODER_PATH:-${RANK_ENCODER_PATH:-}}"

CORPUS_JSONL="${CORPUS_JSONL:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18/wiki-18.jsonl}"

VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${MAX_EVAL_NUM}}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-1}"
MAX_RANKER_STEPS="${MAX_RANKER_STEPS:-${MAX_EVAL_STEPS}}"
# 是否保持完整轨迹：full/partial
KEEP_TRACE="${KEEP_TRACE:-partial}"

TOP_N="${TOP_N:-${RECALL_TOP_K:-50}}"
RECALL_TOP_K="${RECALL_TOP_K:-${TOP_N}}"
TOP_M="${TOP_M:-${TOP_K:-5}}"
TOP_K="${TOP_K:-${TOP_M}}"
RANK_TOP_K="${RANK_TOP_K:-${RANKER_TOP_K:-${TOP_M}}}"
RANKER_TOP_K="${RANKER_TOP_K:-${RANK_TOP_K}}"

EXPLICIT_PROXY_PORT="${PROXY_PORT+x}"
EXPLICIT_RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL+x}"
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${BUDGET_MAX_MODEL_LEN:-12288}}"

MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-${BUDGET_MAX_ASSISTANT_TURNS:-6}}"
MAX_USER_TURNS="${MAX_USER_TURNS:-${BUDGET_MAX_USER_TURNS:-6}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-${BUDGET_MAX_PROMPT_LENGTH:-11264}}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${BUDGET_MAX_RESPONSE_LENGTH:-1024}}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-${BUDGET_MAX_TOOL_RESPONSE_LENGTH:-4096}}"
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

RANKER_DEVICE="${RANKER_DEVICE:-cuda:0}"
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
FORMAT_PENALTY="${FORMAT_PENALTY:--0.2}"
SAVE_TOP_N_DOCUMENTS="${SAVE_TOP_N_DOCUMENTS:-false}"
COAGENTIC_TOOL_CLASS_NAME="${COAGENTIC_TOOL_CLASS_NAME:-verl.tools.coagentic_retriever_tool.CoAgenticRetrieverTool}"
COAGENTIC_RANKER_ENABLED="${COAGENTIC_RANKER_ENABLED:-true}"
if [[ "${RUN_MODE}" == "no-ranker" ]]; then
  COAGENTIC_RANKER_ENABLED=false
fi
TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-2}"
RANKER_CONFIG_DEVICE="${RANKER_CONFIG_DEVICE:-${RANKER_DEVICE}}"
STOP_SEQUENCES="${STOP_SEQUENCES:-}"


OUT_DIR="${OUT_DIR:-${TRACE_DIR}}"
LOG_DIR="${LOG_DIR:-${RUNTIME_LOG_DIR}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUT_DIR}/validation_data}"
METRICS_JSONL="${METRICS_JSONL:-${LOG_DIR}/${RUN_NAME}.metrics.jsonl}"
SEARCH_TIMING_JSONL="${SEARCH_TIMING_JSONL:-${LOG_DIR}/${RUN_NAME}.search_timing.jsonl}"
LLM_IO_JSONL="${LLM_IO_JSONL:-${COAGENTIC_RETRIEVER_LLM_IO_JSONL:-${LOG_DIR}/${RUN_NAME}.llm_io.jsonl}}"
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

load_static_tool_config() {
  local parsed
  parsed="$("${PY}" -c '
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except ModuleNotFoundError:
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}

tool = (data.get("tools") or [{}])[0]
config = tool.get("config") or {}
ranker = config.get("ranker") or {}

def emit(name, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = ""
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")

emit("STATIC_TOOL_CLASS_NAME", tool.get("class_name", ""))
emit("STATIC_RETRIEVAL_SERVICE_URL", config.get("retrieval_service_url", ""))
try:
    from urllib.parse import urlparse
    emit("STATIC_RETRIEVAL_PORT", urlparse(str(config.get("retrieval_service_url", ""))).port or "")
except Exception:
    emit("STATIC_RETRIEVAL_PORT", "")
emit("STATIC_DEFAULT_TOP_N", config.get("default_top_n", ""))
emit("STATIC_DEFAULT_TOP_M", config.get("default_top_m", ""))
emit("STATIC_MAX_RETRIES", config.get("max_retries", ""))
emit("STATIC_RETRY_DELAY", config.get("retry_delay", ""))
emit("STATIC_RETRY_BACKOFF", config.get("retry_backoff", ""))
emit("STATIC_FORMAT_PENALTY", config.get("format_penalty", ""))
emit("STATIC_MAX_CONCURRENT_PER_WORKER", config.get("max_concurrent_per_worker", ""))
emit("STATIC_RANKER_ENABLED", config.get("ranker_enabled", ""))
emit("STATIC_RANKER_MODEL_PATH", ranker.get("model_path", ""))
emit("STATIC_RANKER_ENCODER_PATH", ranker.get("encoder_path", ""))
emit("STATIC_RANKER_DEVICE", ranker.get("device", ""))
emit("STATIC_RANKER_TOP_K", ranker.get("top_k", ""))
emit("STATIC_RANKER_MAX_QUERY_LENGTH", ranker.get("max_query_length", ""))
emit("STATIC_RANKER_MAX_DOC_LENGTH", ranker.get("max_doc_length", ""))
emit("STATIC_TRUST_REMOTE_CODE", ranker.get("trust_remote_code", ""))
' "${TOOL_CONFIG}")"
  eval "${parsed}"

  COAGENTIC_TOOL_CLASS_NAME="${STATIC_TOOL_CLASS_NAME}"
  if [[ -z "${EXPLICIT_RETRIEVAL_SERVICE_URL}" ]]; then
    RETRIEVAL_SERVICE_URL="${STATIC_RETRIEVAL_SERVICE_URL}"
  fi
  if [[ -z "${EXPLICIT_PROXY_PORT}" && -n "${STATIC_RETRIEVAL_PORT}" ]]; then
    PROXY_PORT="${STATIC_RETRIEVAL_PORT}"
  elif [[ -n "${EXPLICIT_PROXY_PORT}" && -z "${EXPLICIT_RETRIEVAL_SERVICE_URL}" ]]; then
    RETRIEVAL_SERVICE_URL="http://127.0.0.1:${PROXY_PORT}/retrieve"
  fi
  TOP_N="${STATIC_DEFAULT_TOP_N}"
  RECALL_TOP_K="${STATIC_DEFAULT_TOP_N}"
  TOP_M="${STATIC_DEFAULT_TOP_M}"
  TOP_K="${STATIC_DEFAULT_TOP_M}"
  RANK_TOP_K="${STATIC_RANKER_TOP_K}"
  RANKER_TOP_K="${STATIC_RANKER_TOP_K}"
  RETRIEVAL_MAX_RETRIES="${STATIC_MAX_RETRIES}"
  RETRIEVAL_RETRY_DELAY="${STATIC_RETRY_DELAY}"
  RETRIEVAL_RETRY_BACKOFF="${STATIC_RETRY_BACKOFF}"
  FORMAT_PENALTY="${STATIC_FORMAT_PENALTY}"
  COAGENTIC_RANKER_ENABLED="${STATIC_RANKER_ENABLED}"
  RANKER_DEVICE="${STATIC_RANKER_DEVICE}"
  RANKER_CONFIG_DEVICE="${STATIC_RANKER_DEVICE}"
  RANKER_MAX_QUERY_LENGTH="${STATIC_RANKER_MAX_QUERY_LENGTH}"
  RANKER_MAX_DOC_LENGTH="${STATIC_RANKER_MAX_DOC_LENGTH}"
  TOOL_MAX_CONCURRENT_PER_WORKER="${STATIC_MAX_CONCURRENT_PER_WORKER}"
  TRUST_REMOTE_CODE="${STATIC_TRUST_REMOTE_CODE}"
  if [[ "${RUN_MODE}" == "no-ranker" ]]; then
    COAGENTIC_RANKER_ENABLED=false
  elif [[ "${RUN_MODE}" == "full" || "${RUN_MODE}" == "ranker-only" ]]; then
    COAGENTIC_RANKER_ENABLED=true
  fi
}

load_static_tool_config

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
  if ! [[ "${RECALL_TOP_K}" =~ ^[0-9]+$ ]] || (( RECALL_TOP_K < 1 )); then
    echo "ERROR: RECALL_TOP_K must be a positive integer; got ${RECALL_TOP_K}" >&2
    exit 2
  fi
  if ! [[ "${TOP_M}" =~ ^[0-9]+$ ]] || (( TOP_M < 1 )); then
    echo "ERROR: TOP_M must be a positive integer; got ${TOP_M}" >&2
    exit 2
  fi
  if (( TOP_M > RECALL_TOP_K )); then
    echo "ERROR: TOP_M=${TOP_M} exceeds RECALL_TOP_K=${RECALL_TOP_K}" >&2
    exit 2
  fi
  if (( TOP_M > 5 )); then
    echo "ERROR: TOP_M=${TOP_M} is invalid for current reward preflight; answer_match_reward supports at most 5 visible documents." >&2
    echo "       TOP_M is agent-visible docs. Do not pass RANK_TOP_K/ranker.top_k here." >&2
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
      --top-n "${RECALL_TOP_K}" \
      --top-m "${TOP_M}" \
      --expect-contains "${RETRIEVAL_PREFLIGHT_EXPECT}" 2>&1)"; then
    echo "recall retrieval semantic preflight passed: top_n=${RECALL_TOP_K} top_m=${TOP_M}"
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

  echo "starting recall retrieval service; gpu=${RECALL_GPU_ID}; log=${RECALL_SERVICE_LOG}"
  PORT="${PROXY_PORT}" \
  RECALL_GPU_ID="${RECALL_GPU_ID}" \
  RETRIEVER_GPU_IDS="${RECALL_GPU_ID}" \
  RETRIEVER_MODEL="${RECALL_MODEL_PATH}" \
  DEVICE=cuda \
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
  echo "starting agent vLLM server on GPUs ${AGENT_GPU_IDS}; model=${model_path}; log=${log}"
  setsid env CUDA_VISIBLE_DEVICES="${AGENT_GPU_IDS}" \
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
      echo "ERROR: AGENT_MODEL or MODEL_PATH must be explicitly set for RUN_MODE=${RUN_MODE}; no default agent model is allowed in eval." >&2
      exit 2
    fi
    require_path "${AGENT_MODEL}" "agent model"
  fi
  if [[ "${RUN_MODE}" != "no-ranker" && "${RERANKER}" == "dense_e5" ]]; then
    if [[ -z "${RANKER_MODEL}" ]]; then
      echo "ERROR: RANKER_MODEL or RANKER_MODEL_PATH must be explicitly set for RUN_MODE=${RUN_MODE}; no default ranker model is allowed in eval." >&2
      exit 2
    fi
    if [[ -z "${RANKER_BASE_MODEL}" ]]; then
      echo "ERROR: RANKER_BASE_MODEL or RANKER_BASE_MODEL_PATH must be explicitly set for RUN_MODE=${RUN_MODE}; use the tokenizer/base model such as e5-base-v2." >&2
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
- Strategy: ${STRATEGY_NAME}
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
- Ranker output JSONL: ${RANKER_OUTPUT_JSONL} (${ranker_rows} rows)
- Validation data dir: ${VALIDATION_DATA_DIR}
- Rollout data dir: ${ROLLOUT_DATA_DIR}
- Tool config: ${TOOL_CONFIG}
- Eval budget YAML: ${EVAL_BUDGET_YAML:-none}

## Key Config

- TOP_N: ${TOP_N}
- TOP_M: ${TOP_M}
- RANKER_TOP_K: ${RANKER_TOP_K}
- MAX_EVAL_NUM: ${MAX_EVAL_NUM}
- EVAL_BATCH_SIZE: ${EVAL_BATCH_SIZE}
- ENABLE_THINKING: ${ENABLE_THINKING}
- MAX_MODEL_LEN: ${MAX_MODEL_LEN}
- STOP_SEQUENCES: ${STOP_SEQUENCES:-none}
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
STRATEGY_NAME=${STRATEGY_NAME}
STRATEGY_SLUG=${STRATEGY_SLUG}
RUN_MODE=${RUN_MODE}
RERANKER=${RERANKER}
PROJECT_ROOT=${PROJECT_ROOT}
PY=${PY}
EVALUATOR=${EVALUATOR}
AGENT_MODEL=${AGENT_MODEL}
MODEL_PATH=${MODEL_PATH}
RECALL_MODEL_PATH=${RECALL_MODEL_PATH}
RANKER_MODEL=${RANKER_MODEL}
RANKER_BASE_MODEL=${RANKER_BASE_MODEL}
RANKER_ENCODER_PATH=${RANKER_ENCODER_PATH}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
RESUME_FROM_PATH=${RESUME_FROM_PATH}
DATA_PATH=${DATA_PATH}
MAX_EVAL_NUM=${MAX_EVAL_NUM}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}
MAX_RANKER_STEPS=${MAX_RANKER_STEPS}
KEEP_TRACE=${KEEP_TRACE}
TOP_N=${TOP_N}
TOP_M=${TOP_M}
TOP_K=${TOP_K}
RECALL_TOP_K=${RECALL_TOP_K}
RANKER_TOP_K=${RANKER_TOP_K}
RANK_TOP_K=${RANK_TOP_K}
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
EVAL_BUDGET_YAML=${EVAL_BUDGET_YAML}
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

check_paths
write_env_file

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "TASK_NAME=${TASK_NAME}"
  echo "TRACE_DIR=${TRACE_DIR}"
  echo "RUNTIME_LOG_DIR=${RUNTIME_LOG_DIR}"
  echo "REPORT_PATH=${REPORT_PATH}"
  echo "STRATEGY_NAME=${STRATEGY_NAME}"
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
  echo "AGENT_GPU_IDS=${AGENT_GPU_IDS}"
  echo "RANK_GPU_ID=${RANK_GPU_ID}"
  echo "RANKER_CUDA_VISIBLE_DEVICES=${RANKER_CUDA_VISIBLE_DEVICES}"
  echo "RANKER_DEVICE=${RANKER_DEVICE}"
  echo "RECALL_GPU_ID=${RECALL_GPU_ID}"
  echo "METRICS_JSONL=${METRICS_JSONL}"
  echo "SEARCH_TIMING_JSONL=${SEARCH_TIMING_JSONL}"
  echo "LLM_IO_JSONL=${LLM_IO_JSONL}"
  echo "TOOL_CONFIG=${TOOL_CONFIG}"
  echo "EVAL_BUDGET_YAML=${EVAL_BUDGET_YAML}"
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
  --strategy-name "${STRATEGY_NAME}" \
  --retrieval-url "${RETRIEVAL_SERVICE_URL}" \
  --agent-served-model "${AGENT_SERVED_MODEL}" \
  --top-n "${TOP_N}" \
  --top-m "${TOP_M}" \
  --ranker-top-k "${RANKER_TOP_K}" \
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

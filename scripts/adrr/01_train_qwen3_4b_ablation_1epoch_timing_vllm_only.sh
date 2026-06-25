#!/usr/bin/env bash
set -euo pipefail

# CoAgenticRetriever local training entry.
#
# RUN_MODE=full launches the full CoAgenticRetriever/VERL trainer with agent
# LLM updates and dense ranker contrastive updates enabled.
# RUN_MODE=ranker-only validates only the dense ranker contrastive framework
# without starting full LLM rollout.
# 默认使用cosearch原prompt进行训练；

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
source "${ASSETS_DIR}/00_project_paths.sh"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
source "${ROOT}/src/checkpoints/checkpoint_conversion.sh"
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"
setup_agent_iteration_paths "${ROOT}"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/AgenticDynamicRecallRag}"
EXP_NAME="${EXP_NAME:-}" # 必须给出新的 EXP_NAME 来区分不同实验； 测试运行可以写test,不知道怎么写就写default；
GROUP_NAME="${GROUP_NAME:-adr}"
resolve_coagentic_training_run_identity "${ROOT}" "" 1 "${GROUP_NAME}"
setup_coagentic_logging_defaults "${ROOT}" "${RUN_NAME}"
: "${TRAIN_LOG:=${LOG_DIR}/${RUN_NAME}.train.log}"
: "${METRICS_JSONL:=${LOG_DIR}/${RUN_NAME}.metrics.jsonl}"
: "${SEARCH_TIMING_JSONL:=${LOG_DIR}/${RUN_NAME}.search_timing.jsonl}"
: "${NVIDIA_SMI_CSV:=${LOG_DIR}/${RUN_NAME}.nvidia_smi.csv}"
: "${CHECKPOINT_CONVERSION_LOG:=${LOG_DIR}/${RUN_NAME}.checkpoint_conversion.log}"
: "${REPORT_PREFIX:=${LOG_DIR}/${RUN_NAME}.timing_report}"
: "${REPORT_SCHEMA_PATH:=${ASSETS_DIR}/report_schema.py}"
RUN_MODE="${RUN_MODE:-full}"
case "${RUN_MODE}" in
  full)
    RUN_MODE="full"
    EFFECTIVE_RUN_MODE="full"
    ;;
  co-training)
    RUN_MODE="full"
    EFFECTIVE_RUN_MODE="full"
    ;;
  ranker-only)
    EFFECTIVE_RUN_MODE="ranker-only"
    ;;
  *)
    echo "ERROR: unsupported RUN_MODE=${RUN_MODE}; use full or ranker-only" >&2
    exit 2
    ;;
esac
AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1}"
AGENT_N_GPUS_PER_NODE="${AGENT_N_GPUS_PER_NODE:-$(awk -F',' '{print NF}' <<< "${AGENT_GPU_IDS}")}"
RECALL_GPU_ID="${RECALL_GPU_ID:-3}"
RANK_GPU_ID="${RANK_GPU_ID:-2}"
GPU_IDS="${GPU_IDS:-${AGENT_GPU_IDS},${RANK_GPU_ID}}"
RANKER_VISIBLE_DEVICE_INDEX="${RANKER_VISIBLE_DEVICE_INDEX:-${AGENT_N_GPUS_PER_NODE}}"
MAIN_GPU_IDS="${MAIN_GPU_IDS:-${AGENT_GPU_IDS}}"
RANKER_GPU_IDS="${RANKER_GPU_IDS:-${RANK_GPU_ID}}"
RERANKER_GPU_IDS="${RERANKER_GPU_IDS:-${RANKER_GPU_IDS}}"
REPORT_STEPS="${REPORT_STEPS:-10}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-10}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-60}"
COAGENTIC_ROLLOUT_PROGRESS_INTERVAL="${COAGENTIC_ROLLOUT_PROGRESS_INTERVAL:-60}"
COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL="${COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL:-32}"
CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS="${CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS:-1}"
CHECKPOINT_DELETE_OLD_GLOBAL_STEPS="${CHECKPOINT_DELETE_OLD_GLOBAL_STEPS:-1}"
CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS="${CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS:-1}"
CHECKPOINT_TRAINABLE_ROLES="${CHECKPOINT_TRAINABLE_ROLES:-actor ranker}"
CHECKPOINT_REMOVE_ROOT_DIRS="${CHECKPOINT_REMOVE_ROOT_DIRS:-ranker retriever rollout_data validation_data}"
CHECKPOINT_REMOVE_ROOT_GLOBS="${CHECKPOINT_REMOVE_ROOT_GLOBS:-ranker_contrastive_smoke_metrics.jsonl}"
REPORTER_PGID=""
NVIDIA_SMI_PGID=""

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-64}"
TOTAL_STEPS="${TOTAL_STEPS:-auto}" # 默认auto，即不做训练步数限制；
N_ROLLOUTS="${N_ROLLOUTS:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-5100}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"

MODEL_PATH="${MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/llm/Qwen3-4B}"
RECALL_MODEL_PATH="${RECALL_MODEL_PATH:-${RETRIEVER_MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}}"
RANKER_BASE_MODEL_PATH="${RANKER_BASE_MODEL_PATH:-}"
RANKER_ENCODER_PATH="${RANKER_ENCODER_PATH:-}"

TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet}"
VAL_DATA="${VAL_DATA:-${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet}"
CORPUS_JSONL="${CORPUS_JSONL:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18/wiki-18.jsonl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/checkpoints/qwen3_4b_probe}"
mkdir -p "${CHECKPOINT_ROOT}/${GROUP_SLUG}"
OUT_DIR="${OUT_DIR:-${CHECKPOINT_ROOT}/${GROUP_SLUG}/${RUN_NAME}}"
coagentic_assert_safe_run_target "${LOG_DIR}" "log dir"
coagentic_assert_safe_run_target "${OUT_DIR}" "checkpoint dir"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${LOG_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${LOG_DIR}/validation_data}"
DUMP_ROLLOUT_EVERY_STEP_NUM="${DUMP_ROLLOUT_EVERY_STEP_NUM:-10}"
DUMP_ROLLOUT_NUM_EVERYTIME="${DUMP_ROLLOUT_NUM_EVERYTIME:-1}"
MAX_ROLLOUT_DUMP_NUM="${MAX_ROLLOUT_DUMP_NUM:--1}"
ROLLOUT_TRACE_MODE="${ROLLOUT_TRACE_MODE:-full}"
case "${ROLLOUT_TRACE_MODE}" in
  full|partial)
    ;;
  *)
    echo "ERROR: unsupported ROLLOUT_TRACE_MODE=${ROLLOUT_TRACE_MODE}; use full or partial" >&2
    exit 2
    ;;
esac

PROXY_PORT="${PROXY_PORT:-8030}"
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PROXY_PORT}/retrieve}"
RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda}"
AUTO_START_RECALL_SERVICE="${AUTO_START_RECALL_SERVICE:-1}"
AUTO_STOP_RECALL_SERVICE="${AUTO_STOP_RECALL_SERVICE:-1}"
RECALL_SERVICE_WAIT_SECONDS="${RECALL_SERVICE_WAIT_SECONDS:-240}"
RETRIEVAL_PREFLIGHT_QUERY="${RETRIEVAL_PREFLIGHT_QUERY:-who got the first nobel prize in physics?}"
RETRIEVAL_PREFLIGHT_EXPECT="${RETRIEVAL_PREFLIGHT_EXPECT:-}"
ENABLE_ASYNC_LABELING="${ENABLE_ASYNC_LABELING:-0}"
ASYNC_LABELING_YAML="${ASYNC_LABELING_YAML:-}"
AUTO_START_LLM_JUDGE="${AUTO_START_LLM_JUDGE:-0}"
AUTO_STOP_LLM_JUDGE="${AUTO_STOP_LLM_JUDGE:-0}"
LLM_JUDGE_SERVICE_CONFIG="${LLM_JUDGE_SERVICE_CONFIG:-${PROJECT_ROOT}/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-http://127.0.0.1:8067/v1/chat/completions}"
LLM_JUDGE_PREFLIGHT="${LLM_JUDGE_PREFLIGHT:-1}"
LLM_JUDGE_WAIT_SECONDS="${LLM_JUDGE_WAIT_SECONDS:-600}"
ASYNC_LABELING_LOG_DIR="${ASYNC_LABELING_LOG_DIR:-${LOG_DIR}/async_labeling}"

RANKER_CONTRASTIVE_BATCH_SIZE="${RANKER_CONTRASTIVE_BATCH_SIZE:-}"
RANKER_GRADIENT_ACCUMULATION_STEPS="${RANKER_GRADIENT_ACCUMULATION_STEPS:-}"
RANKER_NUM_GROUPS_PER_STEP="${RANKER_NUM_GROUPS_PER_STEP:-}"
RANKER_STEPS_PER_GLOBAL_STEP="${RANKER_STEPS_PER_GLOBAL_STEP:-}"
RANKER_INFERENCE_SYNC_INTERVAL="${RANKER_INFERENCE_SYNC_INTERVAL:-}"
RANKER_INFERENCE_ACTOR_NAME="${RANKER_INFERENCE_ACTOR_NAME:-}"
RANKER_NEG_PER_POS="${RANKER_NEG_PER_POS:-}"
RANKER_POSITIVE_TOP_K="${RANKER_POSITIVE_TOP_K:-}"
RANKER_TEMPERATURE="${RANKER_TEMPERATURE:-}"
RANKER_MAX_QUERY_LENGTH="${RANKER_MAX_QUERY_LENGTH:-}"
RANKER_MAX_DOC_LENGTH="${RANKER_MAX_DOC_LENGTH:-}"
RECALL_TOP_K="${RECALL_TOP_K:-50}"
RANK_TOP_K="${RANK_TOP_K:-}"
TOP_N="${TOP_N:-${RECALL_TOP_K}}"
TOP_M="${TOP_M:-5}"
RECALL_RETRIEVER_CONFIG_DEVICE="${RECALL_RETRIEVER_CONFIG_DEVICE:-cuda:${RECALL_GPU_ID}}"
RANKER_CONFIG_DEVICE="${RANKER_CONFIG_DEVICE:-}"
TOOL_CONFIG="${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml"

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

def emit(name, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = ""
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")

emit("STATIC_RETRIEVAL_SERVICE_URL", config.get("retrieval_service_url", ""))
try:
    from urllib.parse import urlparse
    emit("STATIC_RETRIEVAL_PORT", urlparse(str(config.get("retrieval_service_url", ""))).port or "")
except Exception:
    emit("STATIC_RETRIEVAL_PORT", "")
emit("STATIC_DEFAULT_TOP_N", config.get("default_top_n", ""))
emit("STATIC_DEFAULT_TOP_M", config.get("default_top_m", ""))
emit("STATIC_FORMAT_PENALTY", config.get("format_penalty", ""))
' "${TOOL_CONFIG}")"
  eval "${parsed}"

  RETRIEVAL_SERVICE_URL="${STATIC_RETRIEVAL_SERVICE_URL}"
  if [[ -n "${STATIC_RETRIEVAL_PORT}" ]]; then
    PROXY_PORT="${STATIC_RETRIEVAL_PORT}"
  fi
  RECALL_TOP_K="${STATIC_DEFAULT_TOP_N}"
  TOP_N="${STATIC_DEFAULT_TOP_N}"
  TOP_M="${STATIC_DEFAULT_TOP_M}"
  FORMAT_PENALTY="${STATIC_FORMAT_PENALTY}"
  RECALL_RETRIEVER_CONFIG_DEVICE="cuda:${RECALL_GPU_ID}"
  GPU_IDS="${AGENT_GPU_IDS},${RANK_GPU_ID}"
}

load_static_tool_config
RANKER_DEVICE_TRAIN="${RANKER_DEVICE_TRAIN:-}"
RECALL_RETRIEVER_DEVICE="${RECALL_RETRIEVER_DEVICE:-cuda:1}"

RECALL_SERVICE_LOG="${RECALL_SERVICE_LOG:-${LOG_DIR}/${RUN_NAME}.recall_retriever_server.log}"
RECALL_SERVICE_PID=""
LLM_JUDGE_PID=""

cleanup_background_tasks() {
  cleanup_llm_judge_service
  coagentic_stop_background_pid "${NVIDIA_SMI_PGID}"
  coagentic_stop_background_pid "${REPORTER_PGID}"
  cleanup_recall_service
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

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
  "${PY}" - "${RETRIEVAL_SERVICE_URL}" "${RETRIEVAL_PREFLIGHT_QUERY}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

url, query = sys.argv[1:3]
payload = json.dumps({"queries": [query], "bm25_weight":0.3, "dense_weight":0.4, "graph_weight":0.3, "topk": 1, "return_scores": False}).encode("utf-8")

request = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=5) as response:
    if response.status >= 500:
        raise SystemExit(1)
    data = json.loads(response.read().decode("utf-8"))
    if "result" not in data:
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
    echo "Recall retrieval semantic preflight passed: top_n=${RECALL_TOP_K} top_m=${TOP_M}"
    return 0
  fi
  status=$?
  printf '%s\n' "${output}" >&2
  return "${status}"
}

cleanup_recall_service() {
  if [[ -n "${RECALL_SERVICE_PID}" ]] && is_truthy "${AUTO_STOP_RECALL_SERVICE}"; then
    if kill -0 "${RECALL_SERVICE_PID}" 2>/dev/null; then
      kill -TERM "${RECALL_SERVICE_PID}" 2>/dev/null || true
      wait "${RECALL_SERVICE_PID}" 2>/dev/null || true
    fi
  fi
}

check_llm_judge_service() {
  local models_url
  models_url="${LLM_JUDGE_ENDPOINT%/v1/chat/completions}/v1/models"
  "${PY}" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('${models_url}', timeout=5).status < 500 else 1)" >/dev/null 2>&1
}

cleanup_llm_judge_service() {
  if [[ -n "${LLM_JUDGE_PID}" ]] && is_truthy "${AUTO_STOP_LLM_JUDGE}"; then
    if kill -0 "${LLM_JUDGE_PID}" 2>/dev/null; then
      kill -TERM "${LLM_JUDGE_PID}" 2>/dev/null || true
      wait "${LLM_JUDGE_PID}" 2>/dev/null || true
    fi
  fi
}

validate_async_labeling_config() {
  if ! is_truthy "${ENABLE_ASYNC_LABELING}"; then
    return 0
  fi
  if [[ -z "${ASYNC_LABELING_YAML}" ]]; then
    echo "ERROR: ENABLE_ASYNC_LABELING=1 requires ASYNC_LABELING_YAML." >&2
    exit 2
  fi
  if [[ ! -f "${ASYNC_LABELING_YAML}" ]]; then
    echo "ERROR: async labeling YAML not found: ${ASYNC_LABELING_YAML}" >&2
    exit 2
  fi
  if [[ ! -f "${LLM_JUDGE_SERVICE_CONFIG}" ]]; then
    echo "ERROR: LLM judge service config not found: ${LLM_JUDGE_SERVICE_CONFIG}" >&2
    exit 2
  fi
  hydra_yaml_overrides_to_array async_labeling_dryrun_args "${PY}" "${ASYNC_LABELING_YAML}" >/dev/null
  "${PY}" - "${ASYNC_LABELING_YAML}" "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

project_root = Path(sys.argv[2])
sys.path.insert(0, str(project_root))
from async_labeling.config import validate_prompt_path

cfg = OmegaConf.load(sys.argv[1])
stages = OmegaConf.select(cfg, "ranker_training.async_labeling.stages") or []
for stage in stages:
    if stage.get("type") != "llm_as_judge":
        continue
    prompt_path = stage.get("prompt", {}).get("path")
    if not prompt_path:
        raise SystemExit("ERROR: llm_as_judge stage is missing prompt.path")
    validate_prompt_path(str(prompt_path), project_root=project_root)
PY
  LLM_JUDGE_LOG_DIR="${ASYNC_LABELING_LOG_DIR}/judge_server" \
    bash "${PROJECT_ROOT}/scripts/launch_llm_as_judge.sh" --config "${LLM_JUDGE_SERVICE_CONFIG}" --dry-run >/dev/null
  mkdir -p "${ASYNC_LABELING_LOG_DIR}" "${ASYNC_LABELING_LOG_DIR}/judge_server"
}

ensure_llm_judge_service() {
  if ! is_truthy "${ENABLE_ASYNC_LABELING}"; then
    return 0
  fi
  validate_async_labeling_config
  if is_truthy "${LLM_JUDGE_PREFLIGHT}" && check_llm_judge_service; then
    echo "LLM judge service already available: ${LLM_JUDGE_ENDPOINT}"
    return 0
  fi
  if ! is_truthy "${AUTO_START_LLM_JUDGE}"; then
    if is_truthy "${LLM_JUDGE_PREFLIGHT}"; then
      echo "ERROR: LLM judge service is unavailable and AUTO_START_LLM_JUDGE=${AUTO_START_LLM_JUDGE}" >&2
      echo "       endpoint=${LLM_JUDGE_ENDPOINT}" >&2
      exit 2
    fi
    echo "LLM judge preflight disabled; skipping service availability check."
    return 0
  fi

  echo "Starting LLM judge service"
  echo "  config=${LLM_JUDGE_SERVICE_CONFIG}"
  echo "  endpoint=${LLM_JUDGE_ENDPOINT}"
  LLM_JUDGE_LOG_DIR="${ASYNC_LABELING_LOG_DIR}/judge_server" \
    bash "${PROJECT_ROOT}/scripts/launch_llm_as_judge.sh" --config "${LLM_JUDGE_SERVICE_CONFIG}"
  local judge_pid_file="${ASYNC_LABELING_LOG_DIR}/judge_server/vllm_gpu06_07_8067.pid"
  if [[ -f "${judge_pid_file}" ]]; then
    LLM_JUDGE_PID="$(cat "${judge_pid_file}")"
  fi

  local waited=0
  while [[ "${waited}" -lt "${LLM_JUDGE_WAIT_SECONDS}" ]]; do
    if check_llm_judge_service; then
      echo "LLM judge service is ready: ${LLM_JUDGE_ENDPOINT}"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "ERROR: timed out waiting for LLM judge service after ${LLM_JUDGE_WAIT_SECONDS}s." >&2
  exit 2
}

ensure_recall_service() {
  validate_recall_preflight_args
  if check_recall_http_ready; then
    echo "Recall retrieval HTTP endpoint already available: ${RETRIEVAL_SERVICE_URL}"
    if ! run_recall_preflight; then
      echo "ERROR: recall retrieval semantic preflight failed; aborting instead of retrying readiness." >&2
      exit 2
    fi
    echo "Recall retrieval service already available: ${RETRIEVAL_SERVICE_URL}"
    return 0
  fi
  if ! is_truthy "${AUTO_START_RECALL_SERVICE}"; then
    echo "ERROR: recall retrieval service is unavailable and AUTO_START_RECALL_SERVICE=${AUTO_START_RECALL_SERVICE}" >&2
    echo "       url=${RETRIEVAL_SERVICE_URL}" >&2
    exit 2
  fi

  echo "Starting recall retrieval service via 00_start_dense_retriever_server.sh"
  echo "  gpu=${RECALL_GPU_ID} url=${RETRIEVAL_SERVICE_URL} log=${RECALL_SERVICE_LOG}"
  PORT="${PROXY_PORT}" \
  RECALL_GPU_ID="${RECALL_GPU_ID}" \
  RETRIEVER_GPU_IDS="${RECALL_GPU_ID}" \
  DEVICE="${RETRIEVER_DEVICE}" \
  PY="${PY}" \
    bash "${SCRIPT_DIR}/00_start_dense_retriever_server.sh" >"${RECALL_SERVICE_LOG}" 2>&1 &
  RECALL_SERVICE_PID=$!
  trap cleanup_background_tasks EXIT INT TERM

  local waited=0
  while [[ "${waited}" -lt "${RECALL_SERVICE_WAIT_SECONDS}" ]]; do
    if check_recall_http_ready; then
      if ! run_recall_preflight; then
        echo "ERROR: recall retrieval semantic preflight failed; aborting instead of retrying readiness." >&2
        exit 2
      fi
      echo "Recall retrieval service is ready: ${RETRIEVAL_SERVICE_URL}"
      return 0
    fi
    if ! kill -0 "${RECALL_SERVICE_PID}" 2>/dev/null; then
      echo "ERROR: recall retrieval service exited before becoming ready. Log tail:" >&2
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

write_env() {
  cat > "${LOG_DIR}/${RUN_NAME}.env" <<EOF
RUN_NAME=${RUN_NAME}
EXP_NAME=${EXP_NAME}
GROUP_NAME=${GROUP_NAME}
GROUP_SLUG=${GROUP_SLUG}
RUN_STAMP=${RUN_STAMP:-}
RUN_MODE=${RUN_MODE}
EFFECTIVE_RUN_MODE=${EFFECTIVE_RUN_MODE}
PROJECT_ROOT=${PROJECT_ROOT}
CONFIG_NAME=${CONFIG_NAME}
GPU_IDS=${GPU_IDS}
AGENT_GPU_IDS=${AGENT_GPU_IDS}
AGENT_N_GPUS_PER_NODE=${AGENT_N_GPUS_PER_NODE}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
ACTOR_BATCH_SIZE=${ACTOR_BATCH_SIZE}
TOTAL_STEPS=${TOTAL_STEPS}
N_ROLLOUTS=${N_ROLLOUTS}
LORA_RANK=${LORA_RANK}
LORA_ALPHA=${LORA_ALPHA}
MAIN_GPU_IDS=${MAIN_GPU_IDS}
RANKER_GPU_IDS=${RANKER_GPU_IDS}
REPORT_STEPS=${REPORT_STEPS}
NVIDIA_SMI_INTERVAL=${NVIDIA_SMI_INTERVAL}
REPORT_INTERVAL_SECONDS=${REPORT_INTERVAL_SECONDS}
COAGENTIC_ROLLOUT_PROGRESS_INTERVAL=${COAGENTIC_ROLLOUT_PROGRESS_INTERVAL}
COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL=${COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL}
CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS=${CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS}
CHECKPOINT_DELETE_OLD_GLOBAL_STEPS=${CHECKPOINT_DELETE_OLD_GLOBAL_STEPS}
CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS=${CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS}
CHECKPOINT_REMOVE_ROOT_DIRS=${CHECKPOINT_REMOVE_ROOT_DIRS}
CHECKPOINT_REMOVE_ROOT_GLOBS=${CHECKPOINT_REMOVE_ROOT_GLOBS}
ALLOW_RUN_REUSE=${ALLOW_RUN_REUSE:-0}
ALLOW_DIR_REUSE=${ALLOW_DIR_REUSE:-0}
MODEL_PATH=${MODEL_PATH}
RECALL_MODEL_PATH=${RECALL_MODEL_PATH}
RANKER_BASE_MODEL_PATH=${RANKER_BASE_MODEL_PATH}
TRAIN_DATA=${TRAIN_DATA}
VAL_DATA=${VAL_DATA}
CORPUS_JSONL=${CORPUS_JSONL}
OUT_DIR=${OUT_DIR}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR}
DUMP_ROLLOUT_EVERY_STEP_NUM=${DUMP_ROLLOUT_EVERY_STEP_NUM}
DUMP_ROLLOUT_NUM_EVERYTIME=${DUMP_ROLLOUT_NUM_EVERYTIME}
MAX_ROLLOUT_DUMP_NUM=${MAX_ROLLOUT_DUMP_NUM}
ROLLOUT_TRACE_MODE=${ROLLOUT_TRACE_MODE}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}
LOG_DIR=${LOG_DIR}
TRAIN_LOG=${TRAIN_LOG}
METRICS_JSONL=${METRICS_JSONL}
SEARCH_TIMING_JSONL=${SEARCH_TIMING_JSONL}
NVIDIA_SMI_CSV=${NVIDIA_SMI_CSV}
CHECKPOINT_CONVERSION_LOG=${CHECKPOINT_CONVERSION_LOG}
REPORT_PREFIX=${REPORT_PREFIX}
REPORT_SCHEMA_PATH=${REPORT_SCHEMA_PATH}
RETRIEVAL_SERVICE_URL=${RETRIEVAL_SERVICE_URL}
AUTO_START_RECALL_SERVICE=${AUTO_START_RECALL_SERVICE}
AUTO_STOP_RECALL_SERVICE=${AUTO_STOP_RECALL_SERVICE}
RANKER_CONTRASTIVE_BATCH_SIZE=${RANKER_CONTRASTIVE_BATCH_SIZE}
RANKER_GRADIENT_ACCUMULATION_STEPS=${RANKER_GRADIENT_ACCUMULATION_STEPS}
RANKER_NUM_GROUPS_PER_STEP=${RANKER_NUM_GROUPS_PER_STEP}
RANKER_STEPS_PER_GLOBAL_STEP=${RANKER_STEPS_PER_GLOBAL_STEP}
RANKER_INFERENCE_SYNC_INTERVAL=${RANKER_INFERENCE_SYNC_INTERVAL}
RANKER_INFERENCE_ACTOR_NAME=${RANKER_INFERENCE_ACTOR_NAME}
RANKER_NEG_PER_POS=${RANKER_NEG_PER_POS}
RANKER_POSITIVE_TOP_K=${RANKER_POSITIVE_TOP_K}
RANKER_TEMPERATURE=${RANKER_TEMPERATURE}
RANKER_MAX_QUERY_LENGTH=${RANKER_MAX_QUERY_LENGTH}
RANKER_MAX_DOC_LENGTH=${RANKER_MAX_DOC_LENGTH}
HYDRA_OVERRIDE_YAMLS=${HYDRA_OVERRIDE_YAMLS:-}
RANKER_STRATEGY_YAML=${RANKER_STRATEGY_YAML:-}
ENABLE_ASYNC_LABELING=${ENABLE_ASYNC_LABELING}
ASYNC_LABELING_YAML=${ASYNC_LABELING_YAML}
AUTO_START_LLM_JUDGE=${AUTO_START_LLM_JUDGE}
AUTO_STOP_LLM_JUDGE=${AUTO_STOP_LLM_JUDGE}
LLM_JUDGE_SERVICE_CONFIG=${LLM_JUDGE_SERVICE_CONFIG}
LLM_JUDGE_ENDPOINT=${LLM_JUDGE_ENDPOINT}
LLM_JUDGE_PREFLIGHT=${LLM_JUDGE_PREFLIGHT}
LLM_JUDGE_WAIT_SECONDS=${LLM_JUDGE_WAIT_SECONDS}
ASYNC_LABELING_LOG_DIR=${ASYNC_LABELING_LOG_DIR}
RECALL_GPU_ID=${RECALL_GPU_ID}
RANK_GPU_ID=${RANK_GPU_ID}
RANKER_VISIBLE_DEVICE_INDEX=${RANKER_VISIBLE_DEVICE_INDEX}
TOOL_CONFIG=${TOOL_CONFIG}
RECALL_TOP_K=${RECALL_TOP_K}
RANK_TOP_K=${RANK_TOP_K}
TOP_N=${TOP_N}
TOP_M=${TOP_M}
RECALL_RETRIEVER_CONFIG_DEVICE=${RECALL_RETRIEVER_CONFIG_DEVICE}
RANKER_CONFIG_DEVICE=${RANKER_CONFIG_DEVICE}
RANKER_DEVICE_TRAIN=${RANKER_DEVICE_TRAIN}
RECALL_RETRIEVER_DEVICE=${RECALL_RETRIEVER_DEVICE}
EOF
}

check_paths() {
  local required_paths=("${PROJECT_ROOT}" "${TRAIN_DATA}" "${VAL_DATA}" "${CORPUS_JSONL}" "${RECALL_MODEL_PATH}")
  if [[ "${EFFECTIVE_RUN_MODE}" == "ranker-only" ]]; then
    if [[ -z "${RANKER_BASE_MODEL_PATH}" ]]; then
      echo "ERROR: RUN_MODE=ranker-only requires explicit RANKER_BASE_MODEL_PATH." >&2
      exit 2
    fi
    required_paths+=("${RANKER_BASE_MODEL_PATH}")
  elif [[ -n "${RANKER_BASE_MODEL_PATH}" ]]; then
    required_paths+=("${RANKER_BASE_MODEL_PATH}")
  fi
  for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "ERROR: required path not found: ${path}" >&2
      exit 2
    fi
  done
}

write_env
check_paths
validate_async_labeling_config

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; configuration written to ${LOG_DIR}/${RUN_NAME}.env"
  echo "training log: ${TRAIN_LOG}"
  echo "metrics jsonl: ${METRICS_JSONL}"
  echo "search timing jsonl: ${SEARCH_TIMING_JSONL}"
  echo "nvidia-smi csv: ${NVIDIA_SMI_CSV}"
  echo "report prefix: ${REPORT_PREFIX}"
  echo "rollout trace mode: ${ROLLOUT_TRACE_MODE}"
  echo "checkpoint dir is reserved for actual model checkpoint writes: ${OUT_DIR}"
  if is_truthy "${ENABLE_ASYNC_LABELING}"; then
    echo "async labeling yaml: ${ASYNC_LABELING_YAML}"
    echo "async labeling log dir: ${ASYNC_LABELING_LOG_DIR}"
    echo "llm judge service config: ${LLM_JUDGE_SERVICE_CONFIG}"
    echo "llm judge endpoint: ${LLM_JUDGE_ENDPOINT}"
  fi
  exit 0
fi

trap cleanup_background_tasks EXIT INT TERM

if [[ "${EFFECTIVE_RUN_MODE}" == "ranker-only" ]]; then
  if [[ -n "${HYDRA_OVERRIDE_YAMLS:-}" || -n "${RANKER_STRATEGY_YAML:-}" ]]; then
    echo "ERROR: HYDRA_OVERRIDE_YAMLS/RANKER_STRATEGY_YAML are only supported in RUN_MODE=full." >&2
    echo "       ranker-only mode uses the standalone smoke script, not Hydra config composition." >&2
    exit 2
  fi
  ensure_recall_service
  coagentic_start_nvidia_smi_sampler
  coagentic_start_training_reporter "${ROOT}"
  export CUDA_VISIBLE_DEVICES="${RANKER_ONLY_CUDA_VISIBLE_DEVICES:-${RANK_GPU_ID},${RECALL_GPU_ID}}"
  for name in \
    RANKER_CONTRASTIVE_BATCH_SIZE \
    RANKER_GRADIENT_ACCUMULATION_STEPS \
    RANKER_NUM_GROUPS_PER_STEP \
    RANKER_NEG_PER_POS \
    RANKER_POSITIVE_TOP_K \
    RANKER_TEMPERATURE \
    RANKER_DEVICE_TRAIN \
    RANKER_MAX_QUERY_LENGTH \
    RANKER_MAX_DOC_LENGTH \
    RANK_TOP_K; do
    if [[ -z "${!name:-}" ]]; then
      echo "ERROR: RUN_MODE=ranker-only requires explicit ${name}; no ranker fallback is allowed." >&2
      exit 2
    fi
  done
  set +e
  "${PY}" "${ASSETS_DIR}/01_ranker_contrastive_smoke.py" \
    --project-root "${PROJECT_ROOT}" \
    --train-data "${TRAIN_DATA}" \
    --corpus-jsonl "${CORPUS_JSONL}" \
    --model-path "${RANKER_BASE_MODEL_PATH}" \
    --output-dir "${OUT_DIR}" \
    --max-steps "${TOTAL_STEPS}" \
    --batch-size "${RANKER_CONTRASTIVE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${RANKER_GRADIENT_ACCUMULATION_STEPS}" \
    --num-groups-per-step "${RANKER_NUM_GROUPS_PER_STEP}" \
    --neg-per-pos "${RANKER_NEG_PER_POS}" \
    --positive-top-k "${RANKER_POSITIVE_TOP_K}" \
    --temperature "${RANKER_TEMPERATURE}" \
    --max-query-length "${RANKER_MAX_QUERY_LENGTH}" \
    --max-doc-length "${RANKER_MAX_DOC_LENGTH}" \
    --device "${RANKER_DEVICE_TRAIN}" \
    --recall-device "${RECALL_RETRIEVER_DEVICE}" \
    --recall-top-k "${RECALL_TOP_K}" \
    --rank-top-k "${RANK_TOP_K}" \
    --construction-log-jsonl "${LOG_DIR}/${RUN_NAME}.contrastive_construction.jsonl" \
    --metrics-jsonl "${METRICS_JSONL}" \
    --retrieval-service-url "${RETRIEVAL_SERVICE_URL}" \
    2>&1 | tee "${TRAIN_LOG}"
  TRAIN_STATUS="${PIPESTATUS[0]}"
  set -e
  if [[ "${TRAIN_STATUS}" == "0" ]]; then
    run_checkpoint_cleanup "${ROOT}" "${OUT_DIR}"
  fi
  coagentic_generate_final_training_reports "${ROOT}" || true
  exit "${TRAIN_STATUS}"
fi

ensure_llm_judge_service
ensure_recall_service
coagentic_start_nvidia_smi_sampler
coagentic_start_training_reporter "${ROOT}"

export PY
export COAGENTIC_PROJECT_ROOT="${PROJECT_ROOT}"
export CHECKPOINT_VERL_ROOT="${CHECKPOINT_VERL_ROOT:-${PROJECT_ROOT}/verl}"
export GPU_IDS
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${AGENT_N_GPUS_PER_NODE}}"
export CONFIG_NAME="${CONFIG_NAME:-${RUN_NAME}}"
export MODEL_PATH
export RETRIEVAL_SERVICE_URL
export OUT_DIR
export EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
export TRAIN_DATA
export VAL_DATA
export ROLLOUT_DATA_DIR
export VALIDATION_DATA_DIR
export TRAIN_MAX_SAMPLES
export VAL_MAX_SAMPLES
export TRAIN_BATCH_SIZE
export VAL_BATCH_SIZE
export TOTAL_STEPS
export N_ROLLOUTS
export LORA_RANK
export LORA_ALPHA
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','file']}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${METRICS_JSONL}}"
export DUMP_ROLLOUT_EVERY_STEP_NUM
export DUMP_ROLLOUT_NUM_EVERYTIME
export MAX_ROLLOUT_DUMP_NUM
export ROLLOUT_TRACE_MODE
export COAGENTIC_ROLLOUT_PROGRESS_INTERVAL
export COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL
export COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL="${COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL:-${SEARCH_TIMING_JSONL}}"
export RETRIEVAL_PREFLIGHT_QUERY
export RETRIEVAL_PREFLIGHT_EXPECT
export TOP_N
export TOP_M
export ENABLE_ASYNC_LABELING
export ASYNC_LABELING_YAML
export LLM_JUDGE_ENDPOINT
export ASYNC_LABELING_LOG_DIR
export SAVE_TOP_N_DOCUMENTS=true
export COAGENTIC_RETRIEVER_LLM_IO_JSONL="${COAGENTIC_RETRIEVER_LLM_IO_JSONL:-${LOG_DIR}/${RUN_NAME}.llm_io.jsonl}"
export COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS="${COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS:-20}"

export COAGENTIC_MAIN="${PROJECT_ROOT}/main_coagentic_retriever.py"
USER_COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-}"
DEFAULT_COAGENTIC_EXTRA_ARGS="trainer.ranker_trainable=false trainer.ranker_update_mode=contrastive trainer.disable_reranker_rollout=true recall_retriever.model_path=${RECALL_MODEL_PATH} recall_retriever.device=${RECALL_RETRIEVER_CONFIG_DEVICE} recall_retriever.service_url=${RETRIEVAL_SERVICE_URL} recall_retriever.top_k=${RECALL_TOP_K} recall_retriever.trainable=false recall_retriever.index_refresh=false ranker_training.construction_log_jsonl=${LOG_DIR}/${RUN_NAME}.contrastive_construction.jsonl"
if [[ -n "${RANKER_STEPS_PER_GLOBAL_STEP}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" trainer.ranker_steps_per_global_step=${RANKER_STEPS_PER_GLOBAL_STEP}"
fi
if [[ -n "${RANKER_BASE_MODEL_PATH}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker.model_path=${RANKER_BASE_MODEL_PATH}"
fi
if [[ -n "${RANKER_ENCODER_PATH}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker.encoder_path=${RANKER_ENCODER_PATH}"
fi
if [[ -n "${RANKER_CONFIG_DEVICE}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker.device=${RANKER_CONFIG_DEVICE}"
fi
if [[ -n "${RANK_TOP_K}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker.top_k=${RANK_TOP_K}"
fi
if [[ -n "${RANKER_INFERENCE_ACTOR_NAME}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.shared_inference_ranker.actor_name=${RANKER_INFERENCE_ACTOR_NAME}"
fi
if [[ -n "${RANKER_INFERENCE_SYNC_INTERVAL}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.shared_inference_ranker.sync_interval=${RANKER_INFERENCE_SYNC_INTERVAL}"
fi
if [[ -n "${RANKER_CONTRASTIVE_BATCH_SIZE}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.batch_size=${RANKER_CONTRASTIVE_BATCH_SIZE}"
fi
if [[ -n "${RANKER_GRADIENT_ACCUMULATION_STEPS}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.gradient_accumulation_steps=${RANKER_GRADIENT_ACCUMULATION_STEPS}"
fi
if [[ -n "${RANKER_TEMPERATURE}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.loss.temperature=${RANKER_TEMPERATURE}"
fi
if [[ -n "${RANKER_NEG_PER_POS}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.sample_builder.neg_per_pos=${RANKER_NEG_PER_POS}"
fi
if [[ -n "${RANKER_POSITIVE_TOP_K}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.signal_builder.positive_top_k=${RANKER_POSITIVE_TOP_K}"
fi
if [[ -n "${RANKER_NUM_GROUPS_PER_STEP}" ]]; then
  DEFAULT_COAGENTIC_EXTRA_ARGS+=" ranker_training.sample_builder.num_groups_per_step=${RANKER_NUM_GROUPS_PER_STEP}"
fi
export COAGENTIC_DEFAULT_EXTRA_ARGS="${DEFAULT_COAGENTIC_EXTRA_ARGS}"
export COAGENTIC_EXTRA_ARGS="${USER_COAGENTIC_EXTRA_ARGS}"
export ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"


set +e
bash "${ASSETS_DIR}/00_run_agentic_iter_rag_verl.sh" "$@" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e

CHECKPOINT_CONVERSION_STATUS=0
if latest_role_fsdp_checkpoint "${OUT_DIR}" actor >/dev/null; then
  echo "Starting checkpoint conversion and validation. Log: ${CHECKPOINT_CONVERSION_LOG}" | tee -a "${TRAIN_LOG}"
  set +e
  run_verl_fsdp_checkpoint_conversion "${ROOT}" "${OUT_DIR}" 2>&1 | tee "${CHECKPOINT_CONVERSION_LOG}" | tee -a "${TRAIN_LOG}"
  CHECKPOINT_CONVERSION_STATUS="${PIPESTATUS[0]}"
  set -e
else
  echo "Checkpoint conversion skipped: no actor FSDP checkpoint found under ${OUT_DIR}" | tee -a "${TRAIN_LOG}"
fi

coagentic_generate_final_training_reports "${ROOT}" || true
if [[ "${TRAIN_STATUS}" != "0" ]]; then
  exit "${TRAIN_STATUS}"
fi
exit "${CHECKPOINT_CONVERSION_STATUS}"

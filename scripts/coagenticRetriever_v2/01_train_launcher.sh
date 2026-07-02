#!/usr/bin/env bash
set -euo pipefail

# CoAgenticRetriever v2 本地训练 launcher。
#
# 这个 Bash 脚本现在只负责“有副作用的执行编排”：
#   1. 初始化仓库路径、Python 和 accelerator helper。
#   2. 调用 Python config compiler，把配置合并结果物化为 env/Hydra 审计文件。
#   3. 按编译后的 env 等待 GPU、启动/检查 LLM judge 和 recall retriever 服务。
#   4. 启动 nvidia-smi sampler、训练报告器和 canonical training runner。
#   5. 训练结束后执行 checkpoint 转换和最终报告生成。
#
# 配置解析、main_run/resource/overlay 合并、Hydra 参数构造、运行态 override YAML
# 生成都已经下沉到：
#   scripts/coagenticRetriever_v2/assets/trainer_launcher/compile_config.py

# ========== 1. 基础路径和公共工具初始化 ==========
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}"

source "${ASSETS_DIR}/00_project_paths.sh"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
source "${ROOT}/src/checkpoints/checkpoint_conversion.sh"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_accelerator.sh"
source "${ROOT}/src/runtime/wait_for_gpus.sh"
setup_agent_iteration_paths "${ROOT}"

# ========== 2. Python 配置编译模块 ==========
# Python compiler 是 launcher 配置的唯一编译入口。它会读取 main_run/resource/overlay，
# 生成本次 run 的 runtime env、.env 审计文件、runtime override YAML 和 hydra_args.txt。
# Bash 只 source 编译出的 runtime env，不再手写这些配置默认值。
CONFIG_COMPILER="${ASSETS_DIR}/trainer_launcher/compile_config.py"
if [[ ! -f "${CONFIG_COMPILER}" ]]; then
  echo "ERROR: launcher config compiler not found: ${CONFIG_COMPILER}" >&2
  exit 2
fi
RECALL_PREFLIGHT_PY="${ASSETS_DIR}/trainer_launcher/recall_preflight.py"
if [[ ! -f "${RECALL_PREFLIGHT_PY}" ]]; then
  echo "ERROR: recall preflight helper not found: ${RECALL_PREFLIGHT_PY}" >&2
  exit 2
fi
CANONICAL_TRAINING_RUNNER="${ASSETS_DIR}/trainer_launcher/run_canonical_training.py"
if [[ ! -f "${CANONICAL_TRAINING_RUNNER}" ]]; then
  echo "ERROR: canonical training runner not found: ${CANONICAL_TRAINING_RUNNER}" >&2
  exit 2
fi

LAUNCHER_RUNTIME_ENV_SH="$("${PY}" "${CONFIG_COMPILER}" \
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

if [[ -z "${LAUNCHER_RUNTIME_ENV_SH}" || ! -f "${LAUNCHER_RUNTIME_ENV_SH}" ]]; then
  echo "ERROR: config compiler did not produce a source-able runtime env file." >&2
  echo "       output=${LAUNCHER_RUNTIME_ENV_SH}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${LAUNCHER_RUNTIME_ENV_SH}"
if [[ "${CANONICAL_CONFIG_MODE}" != "1" ]]; then
  echo "ERROR: 01_train_launcher.sh only supports canonical config mode." >&2
  echo "       Use task scripts with --main_run_config / trainer config groups / overlay YAML." >&2
  exit 2
fi

# ========== 3. 后台进程清理模块 ==========
# 只清理本 launcher 拉起的后台任务。外部已有服务不会因为本脚本退出而被停止。
RECALL_SERVICE_PID=""
LLM_JUDGE_PID=""
REPORTER_PGID=""
NVIDIA_SMI_PGID=""

cleanup_background_tasks() {
  cleanup_llm_judge_service
  coagentic_stop_background_pid "${NVIDIA_SMI_PGID}"
  coagentic_stop_background_pid "${REPORTER_PGID}"
  cleanup_recall_service
}

trap cleanup_background_tasks EXIT INT TERM

# ========== 4. Shell bool 解析工具 ==========
# 服务开关、dry-run 和 GPU wait 都复用同一组 truthy 规则。
is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# ========== 5. recall retriever 预检模块 ==========
# recall retriever 是训练工具链的外部 HTTP 服务。参数校验、HTTP 探活和语义预检
# 已下沉到 Python runtime helper；Bash 这里只保留服务启动、PID 管理和日志 tail。
#
# 本模块分成五个小职责：
#   1. validate_recall_preflight_args：校验 top_n/top_m 这类纯参数约束。
#   2. check_recall_http_ready：轻量 HTTP 探活，用于判断 endpoint 是否已经可用。
#   3. run_recall_preflight：严格语义预检，确认返回数量和可见文档内容符合训练假设。
#   4. cleanup_recall_service：只清理由本 launcher 启动的 recall 服务。
#   5. ensure_recall_service：整体编排，复用已有服务或按需启动服务并等待 ready。
validate_recall_preflight_args() {
  # 纯参数校验，不访问 HTTP 服务。
  #
  # RECALL_TOP_K 是 retriever 返回候选数量，TOP_M 是 agent/reward 当前可见文档数。
  # 这些约束由 Python helper 统一维护，避免 Bash 和旧 standalone 检查脚本出现两套规则。
  "${PY}" "${RECALL_PREFLIGHT_PY}" validate \
    --top-n "${RECALL_TOP_K}" \
    --top-m "${TOP_M}"
}

check_recall_http_ready() {
  # 轻量 HTTP 探活，只验证 endpoint 能返回基本 JSON 结构。
  #
  # 这个函数用于等待循环，所以失败不是 fatal；调用方会根据 AUTO_START_RECALL_SERVICE
  # 决定是启动服务、继续等待，还是最终报错。
  "${PY}" "${RECALL_PREFLIGHT_PY}" http-ready \
    --url "${RETRIEVAL_SERVICE_URL}" \
    --query "${RETRIEVAL_PREFLIGHT_QUERY}" \
    --timeout 5 >/dev/null 2>&1
}

run_recall_preflight() {
  # 严格语义预检。
  #
  # HTTP ready 只说明服务“活着”；semantic preflight 会进一步检查：
  #   - retriever 返回文档数是否等于 RECALL_TOP_K。
  #   - 如果设置了 RETRIEVAL_PREFLIGHT_EXPECT，TOP_M 可见文档中是否包含期望子串。
  # endpoint 已经 ready 但 semantic 失败，说明训练工具链假设不成立，必须中止。
  local output status
  if output="$("${PY}" "${RECALL_PREFLIGHT_PY}" semantic \
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
  # recall 服务清理。
  #
  # 只清理当前 launcher 启动并记录在 RECALL_SERVICE_PID 中的服务。若 endpoint 是外部
  # 已存在服务，RECALL_SERVICE_PID 为空，本函数不会停止它。
  if [[ -n "${RECALL_SERVICE_PID}" ]] && is_truthy "${AUTO_STOP_RECALL_SERVICE}"; then
    if kill -0 "${RECALL_SERVICE_PID}" 2>/dev/null; then
      kill -TERM "${RECALL_SERVICE_PID}" 2>/dev/null || true
      wait "${RECALL_SERVICE_PID}" 2>/dev/null || true
    fi
  fi
}

ensure_recall_service() {
  # recall 服务准备总入口。
  #
  # 顺序是：
  #   1. 先校验参数，参数错误立即失败。
  #   2. 如果已有 endpoint 可用，直接做 semantic preflight 并复用。
  #   3. 如果 endpoint 不可用且禁止自动启动，立即报错。
  #   4. 如果允许自动启动，由 Bash 启动 dense retriever server，记录 PID，并循环等待。
  #
  # 启动、PID、tail log 仍留在 Bash，是因为 Bash launcher 是后台进程 owner。
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
  # 这里启动的是本 launcher 拥有的后台服务。RECALL_SERVICE_PID 只在这种情况下设置，
  # cleanup_recall_service 也只会停止这个 PID。
  PORT="${PROXY_PORT}" \
  RECALL_GPU_ID="${RECALL_GPU_ID}" \
  RETRIEVER_GPU_IDS="${RECALL_GPU_ID}" \
  DEVICE="${RETRIEVER_DEVICE}" \
  PY="${PY}" \
    bash "${SCRIPT_DIR}/00_start_dense_retriever_server.sh" >"${RECALL_SERVICE_LOG}" 2>&1 &
  RECALL_SERVICE_PID=$!

  local waited=0
  while [[ "${waited}" -lt "${RECALL_SERVICE_WAIT_SECONDS}" ]]; do
    # 服务启动后仍然先做轻量 HTTP ready，再做 semantic preflight。这样可以区分
    # “服务还没起来”和“服务起来了但返回内容不符合训练假设”。
    if check_recall_http_ready; then
      if ! run_recall_preflight; then
        echo "ERROR: recall retrieval semantic preflight failed; aborting instead of retrying readiness." >&2
        exit 2
      fi
      echo "Recall retrieval service is ready: ${RETRIEVAL_SERVICE_URL}"
      return 0
    fi
    # 如果子进程提前退出，继续等待没有意义；直接打印服务日志尾部帮助定位启动失败原因。
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

# ========== 6. LLM judge 服务准备模块 ==========
# LLM judge 是否需要启动由 Python compiler 从最终 Hydra 配置推导：
# 只有 async ranker training 真实启用，且 stages 中存在 llm_as_judge 时，
# 才会派生 NEEDS_LLM_JUDGE_SERVICE=1。
# Bash 这里只消费这个派生结果，负责 endpoint 检查、按需启动和退出时清理。
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

ensure_llm_judge_service() {
  if ! is_truthy "${NEEDS_LLM_JUDGE_SERVICE}"; then
    return 0
  fi
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
  LLM_JUDGE_LOG_DIR="${ASYNC_RANKER_TRAINING_LOG_DIR}/judge_server" \
    bash "${PROJECT_ROOT}/scripts/launch_llm_as_judge.sh" --config "${LLM_JUDGE_SERVICE_CONFIG}"

  local judge_pid_file="${ASYNC_RANKER_TRAINING_LOG_DIR}/judge_server/vllm_gpu06_07_8067.pid"
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

# ========== 7. dry-run 输出模块 ==========
# DRY_RUN=1 只验证配置编译链路和必要路径，不启动服务、不执行训练。
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; configuration written to ${LOG_DIR}/${RUN_NAME}.env"
  echo "training log: ${TRAIN_LOG}"
  echo "metrics jsonl: ${METRICS_JSONL}"
  echo "search timing jsonl: ${SEARCH_TIMING_JSONL}"
  echo "nvidia-smi csv: ${NVIDIA_SMI_CSV}"
  echo "report prefix: ${REPORT_PREFIX}"
  echo "rollout trace mode: ${ROLLOUT_TRACE_MODE}"
  echo "checkpoint dir is reserved for actual model checkpoint writes: ${OUT_DIR}"
  if [[ "${CANONICAL_CONFIG_MODE}" == "1" ]]; then
    echo "main run config: ${MAIN_RUN_CONFIG_FILE}"
    echo "trainer main hydra config: ${CANONICAL_TRAINER_MAIN_HYDRA_CONFIG_FILE}"
    echo "hydra groups: ${CANONICAL_HYDRA_GROUPS_FILE}"
    echo "hydra cli overrides: ${CANONICAL_CLI_OVERRIDES_FILE}"
    echo "overlay yamls: ${CANONICAL_OVERLAY_YAMLS_FILE}"
    echo "run mode overrides: ${CANONICAL_RUN_MODE_OVERRIDE_YAML}"
    echo "runtime overrides: ${CANONICAL_RUNTIME_OVERRIDE_YAML}"
    echo "hydra args: ${CANONICAL_HYDRA_ARGS_FILE}"
    echo "final config yaml: ${CANONICAL_FINAL_CONFIG_YAML}"
    echo "final config json: ${CANONICAL_FINAL_CONFIG_JSON}"
  fi
  if is_truthy "${NEEDS_LLM_JUDGE_SERVICE}"; then
    echo "async ranker training yaml: ${ASYNC_RANKER_TRAINING_YAML}"
    echo "async ranker training log dir: ${ASYNC_RANKER_TRAINING_LOG_DIR}"
    echo "llm judge service config: ${LLM_JUDGE_SERVICE_CONFIG}"
    echo "llm judge endpoint: ${LLM_JUDGE_ENDPOINT}"
  fi
  exit 0
fi

# ========== 8. 训练前运行态准备模块 ==========
# 这里开始进入真正有副作用的部分：等待设备、准备服务、启动采样器和报告器。
wait_for_gpus_if_enabled
ensure_llm_judge_service
ensure_recall_service
coagentic_start_nvidia_smi_sampler
coagentic_start_training_reporter "${ROOT}"

# ========== 9. canonical training runner 环境导出模块 ==========
# 这些变量已经由 Python config compiler 物化。Bash 这里只把后续 Python runner 需要
# 继承的运行态变量显式导出，不再拼 legacy env-to-Hydra 参数。
export PY
export COAGENTIC_PROJECT_ROOT="${PROJECT_ROOT}"
export CHECKPOINT_VERL_ROOT
export GPU_IDS
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${AGENT_N_GPUS_PER_NODE}}"
export CONFIG_NAME="${CONFIG_NAME:-${RUN_NAME}}"
export RETRIEVAL_SERVICE_URL
export OUT_DIR
export EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
export ROLLOUT_DATA_DIR
export VALIDATION_DATA_DIR
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${METRICS_JSONL}}"
export COAGENTIC_ROLLOUT_PROGRESS_INTERVAL
export COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL
export COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL
export RETRIEVAL_PREFLIGHT_QUERY
export RETRIEVAL_PREFLIGHT_EXPECT
export TOP_N
export TOP_M
export NEEDS_LLM_JUDGE_SERVICE
export ASYNC_RANKER_TRAINING_YAML
export LLM_JUDGE_ENDPOINT
export ASYNC_RANKER_TRAINING_LOG_DIR
export CANONICAL_CONFIG_MODE
export CANONICAL_HYDRA_ARGS_FILE
export SAVE_TOP_N_DOCUMENTS
export COAGENTIC_RETRIEVER_LLM_IO_JSONL
export COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS
export COAGENTIC_MAIN
export ACTOR_MICRO_BATCH_SIZE_PER_GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU

# ========== 10. 训练执行模块 ==========
# v2 launcher 不再调用 legacy `00_run_agentic_iter_rag_verl.sh`。canonical runner 只读取
# compile_config.py 生成的 hydra_args.txt，然后直接 exec main_coagentic_retriever.py。
set +e
"${PY}" "${CANONICAL_TRAINING_RUNNER}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e

# ========== 11. 训练收尾模块 ==========
# 训练进程结束后转换 actor FSDP checkpoint，生成最终训练报告，并返回真实退出码。
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

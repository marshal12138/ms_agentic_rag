#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT}/scripts/agenticIterRag_v1"
PROJECT_ROOT="${ROOT}/AgenticIterRag"
COMPILER="${SCRIPT_DIR}/assets/compile_config.py"
RUNNER="${SCRIPT_DIR}/assets/run_pipeline.py"
PY="${PY:-/data05/conda/envs/ms/ms_agt_rag/bin/python}"

LOCAL_DRY_RUN=0
ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      LOCAL_DRY_RUN=1
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

RUNTIME_ENV="$("${PY}" "${COMPILER}" \
  --repo-root "${ROOT}" \
  --project-root "${PROJECT_ROOT}" \
  --script-dir "${SCRIPT_DIR}" \
  "${ARGS[@]}")"

# shellcheck disable=SC1090
source "${RUNTIME_ENV}"

PIPELINE_TERMINAL_LOG="${PIPELINE_TERMINAL_LOG:-${LOG_DIR}/pipeline.terminal.log}"
mkdir -p "$(dirname "${PIPELINE_TERMINAL_LOG}")"

RUNNER_ARGS=(
  --config "${FINAL_CONFIG_YAML}"
  --manifest "${MANIFEST_PATH}"
  --execution-plan "${EXECUTION_PLAN_PATH}"
)
if [[ "${LOCAL_DRY_RUN}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  RUNNER_ARGS+=(--dry-run)
fi

# pipeline 主进程日志默认同时输出到 terminal 和磁盘，便于长任务现场观察和事后复盘。
AIR_LOG_MESSAGE="[AIR pipeline] terminal log will also be written to: ${PIPELINE_TERMINAL_LOG}"
echo "${AIR_LOG_MESSAGE}" | tee -a "${PIPELINE_TERMINAL_LOG}"
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PY}" "${RUNNER}" "${RUNNER_ARGS[@]}" 2>&1 | tee -a "${PIPELINE_TERMINAL_LOG}"

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

RUNNER_ARGS=(
  --config "${FINAL_CONFIG_YAML}"
  --manifest "${MANIFEST_PATH}"
  --execution-plan "${EXECUTION_PLAN_PATH}"
)
if [[ "${LOCAL_DRY_RUN}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  RUNNER_ARGS+=(--dry-run)
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PY}" "${RUNNER}" "${RUNNER_ARGS[@]}"

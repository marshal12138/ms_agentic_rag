#!/usr/bin/env bash

slugify_cosearch_name() {
  local raw="${1:-default}"
  local max_len="${2:-0}"
  local value
  if [[ "${max_len}" =~ ^[0-9]+$ ]] && (( max_len > 0 )); then
    raw="${raw:0:max_len}"
  fi
  value="$(printf '%s' "${raw}" | tr -c 'A-Za-z0-9._-' '_')"
  value="${value##[._-]}"
  value="${value%%[._-]}"
  if [[ -z "${value}" ]]; then
    value="default"
  fi
  printf '%s\n' "${value}"
}

resolve_cosearch_group_identity() {
  local default_group_name="${1:-defaultGroup}"
  GROUP_NAME="${GROUP_NAME:-${default_group_name}}"
  GROUP_SLUG="${GROUP_SLUG:-$(slugify_cosearch_name "${GROUP_NAME}")}"
  export GROUP_NAME GROUP_SLUG
}

cosearch_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cosearch_path_has_payload() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    return 1
  fi
  if [[ -d "${path}" ]]; then
    find "${path}" -mindepth 1 -print -quit | grep -q .
    return $?
  fi
  return 0
}

cosearch_run_reuse_allowed() {
  if cosearch_truthy "${ALLOW_RUN_REUSE:-0}" || cosearch_truthy "${ALLOW_DIR_REUSE:-0}"; then
    return 0
  fi
  if [[ -n "${RESUME_MODE:-}" && "${RESUME_MODE}" != "disable" ]]; then
    return 0
  fi
  return 1
}

resolve_cosearch_training_run_identity() {
  local root="$1"
  local default_exp_name="${2:-}"
  local require_exp_name="${3:-1}"
  local default_group_name="${4:-defaultGroup}"
  local safe_exp_name

  resolve_cosearch_group_identity "${default_group_name}"
  TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${root}/log/train_logs/${GROUP_SLUG}}"

  if [[ -n "${RUN_NAME:-}" ]]; then
    RUN_NAME="$(slugify_cosearch_name "${RUN_NAME}")"
    LOG_DIR="${LOG_DIR:-${TRAIN_LOG_ROOT}/${RUN_NAME}}"
    CONFIG_NAME="${CONFIG_NAME:-${RUN_NAME}}"
    export TRAIN_LOG_ROOT LOG_DIR RUN_NAME CONFIG_NAME
    return 0
  fi

  EXP_NAME="${EXP_NAME:-${default_exp_name}}"
  if [[ "${require_exp_name}" == "1" && -z "${EXP_NAME}" ]]; then
    echo "ERROR: EXP_NAME is required to build a unique RUN_NAME. Example: EXP_NAME=my_rule_v1 bash <script>" >&2
    return 2
  fi
  if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="default"
  fi

  safe_exp_name="$(slugify_cosearch_name "${EXP_NAME}")"
  RUN_STAMP="${RUN_STAMP:-$(date +%y%m%d-%H%M%S)}"
  RUN_NAME="${RUN_STAMP}-${safe_exp_name}"
  LOG_DIR="${LOG_DIR:-${TRAIN_LOG_ROOT}/${RUN_NAME}}"
  CONFIG_NAME="${CONFIG_NAME:-${RUN_NAME}}"
  export TRAIN_LOG_ROOT EXP_NAME RUN_STAMP LOG_DIR RUN_NAME CONFIG_NAME
}

cosearch_assert_safe_run_target() {
  local path="$1"
  local label="${2:-target}"

  if cosearch_run_reuse_allowed; then
    return 0
  fi
  if cosearch_path_has_payload "${path}"; then
    echo "ERROR: ${label} already exists and is non-empty: ${path}" >&2
    echo "       Refusing to reuse it by default because this may overwrite checkpoints or logs." >&2
    echo "       Set ALLOW_RUN_REUSE=1 (or ALLOW_DIR_REUSE=1), or use a new EXP_NAME/RUN_NAME." >&2
    return 2
  fi
}

setup_cosearch_training_log_defaults() {
  local root="$1"
  local default_experiment="${2:-default}"
  local experiment_name="${LOG_EXPERIMENT_NAME:-${EXPERIMENT_NAME:-${default_experiment}}}"

  resolve_cosearch_group_identity "${GROUP_NAME:-defaultGroup}"

  if [[ -z "${experiment_name}" ]]; then
    experiment_name="default"
  fi

  local safe_experiment
  safe_experiment="$(slugify_cosearch_name "${experiment_name}")"
  local log_timestamp
  log_timestamp="${LOG_TIMESTAMP:-$(date +%y%m%d-%H%M)}"

  TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${root}/log/train_logs/${GROUP_SLUG}}"
  if [[ -z "${LOG_DIR:-}" ]]; then
    LOG_DIR="${TRAIN_LOG_ROOT}/${log_timestamp}-${safe_experiment}"
  fi

  RUN_NAME="${RUN_NAME:-${safe_experiment}}"
  TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/${RUN_NAME}.train.log}"
  METRICS_JSONL="${METRICS_JSONL:-${LOG_DIR}/${RUN_NAME}.metrics.jsonl}"
  SEARCH_TIMING_JSONL="${SEARCH_TIMING_JSONL:-${LOG_DIR}/${RUN_NAME}.search_timing.jsonl}"
  NVIDIA_SMI_CSV="${NVIDIA_SMI_CSV:-${LOG_DIR}/${RUN_NAME}.nvidia_smi.csv}"
  REPORT_PREFIX="${REPORT_PREFIX:-${LOG_DIR}/${RUN_NAME}.timing_report}"
  VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${METRICS_JSONL}}"
  TRAINER_LOGGER="${TRAINER_LOGGER:-['console','file']}"

  mkdir -p "${LOG_DIR}"
  export TRAIN_LOG_ROOT LOG_DIR RUN_NAME TRAIN_LOG METRICS_JSONL SEARCH_TIMING_JSONL NVIDIA_SMI_CSV REPORT_PREFIX
  export VERL_FILE_LOGGER_PATH TRAINER_LOGGER
}

setup_cosearch_training_report_defaults() {
  TIMING_REPORT="${TIMING_REPORT:-${REPORT_PREFIX}.latest.md}"
  METRICS_REPORT="${METRICS_REPORT:-${LOG_DIR}/${RUN_NAME}.training_metrics_report.latest.md}"
  DETAILED_METRICS_REPORT="${DETAILED_METRICS_REPORT:-${LOG_DIR}/${RUN_NAME}.detailed_metrics_report.latest.md}"
  METRICS_PLOT_PREFIX="${METRICS_PLOT_PREFIX:-${LOG_DIR}/${RUN_NAME}.metrics.latest}"
  export TIMING_REPORT METRICS_REPORT DETAILED_METRICS_REPORT
  export METRICS_PLOT_PREFIX
}

train_report_system_schema_path() {
  local root="$1"
  if [[ -n "${REPORT_SCHEMA_PATH:-}" ]]; then
    printf '%s\n' "${REPORT_SCHEMA_PATH}"
    return 0
  fi
  case "${GROUP_SLUG:-${GROUP_NAME:-}}" in
    coAgenticRetriever|coagenticRetriever|coagenticretriever)
      printf '%s\n' "${root}/scripts/coagenticRetriever_local/assets/report_schema.py"
      ;;
    AgenticIterRag|agenticIterRag|agenticiterrag|iterRag|iterrag)
      printf '%s\n' "${root}/scripts/iterRag_scripts/assets/report_schema.py"
      ;;
    *)
      printf '%s\n' "${root}/scripts/cosearch_local/assets/report_schema.py"
      ;;
  esac
}

train_report_system_max_metric_step() {
  local root="$1"
  "${PY}" "${root}/src/logs/report_system/train_max_metric_step.py" \
    --metrics-jsonl "${METRICS_JSONL}"
}

train_report_system_generate_reports() {
  local root="$1"
  local mode="${2:-snapshot}"
  local schema_path
  local step_limit_args=()
  local plot_prefix

  schema_path="$(train_report_system_schema_path "${root}")"
  setup_cosearch_training_report_defaults
  plot_prefix="${METRICS_PLOT_PREFIX:-${LOG_DIR}/${RUN_NAME}.metrics.latest}"

  if [[ "${mode}" == "snapshot" ]]; then
    local max_step=0
    local step=0
    max_step="$(train_report_system_max_metric_step "${root}" 2>/dev/null || echo 0)"
    if [[ "${max_step}" =~ ^[0-9]+$ ]] && (( max_step > 0 )); then
      if [[ "${TRAIN_REPORT_SNAPSHOT_MODE:-latest}" == "scheduled" ]]; then
        for step in ${REPORT_STEPS:-10}; do
          if [[ "${step}" =~ ^[0-9]+$ ]] && (( step <= max_step )); then
            step_limit_args=(--step-limit "${step}")
          fi
        done
      fi
      if (( ${#step_limit_args[@]} == 0 )); then
        step_limit_args=(--step-limit "${max_step}")
      fi
    else
      for step in ${REPORT_STEPS:-10}; do
        if [[ "${step}" =~ ^[0-9]+$ ]]; then
          step_limit_args=(--step-limit "${step}")
        fi
      done
    fi
  fi

  "${PY}" "${root}/src/logs/report_system/train_timing_report.py" \
    --metrics-jsonl "${METRICS_JSONL}" \
    --search-jsonl "${SEARCH_TIMING_JSONL}" \
    --train-log "${TRAIN_LOG}" \
    --nvidia-smi-csv "${NVIDIA_SMI_CSV}" \
    --main-gpu-ids "${MAIN_GPU_IDS:-}" \
    --reranker-gpu-ids "${RERANKER_GPU_IDS:-}" \
    --ranker-gpu-ids "${RANKER_GPU_IDS:-}" \
    --report-schema "${schema_path}" \
    "${step_limit_args[@]}" \
    --out "${TIMING_REPORT}"

  "${PY}" "${root}/src/logs/report_system/train_metrics_report.py" \
    --metrics-jsonl "${METRICS_JSONL}" \
    --train-log "${TRAIN_LOG}" \
    --report-schema "${schema_path}" \
    "${step_limit_args[@]}" \
    --out "${METRICS_REPORT}"

  "${PY}" "${root}/src/logs/report_system/train_metrics_report.py" \
    --metrics-jsonl "${METRICS_JSONL}" \
    --train-log "${TRAIN_LOG}" \
    --report-schema "${schema_path}" \
    --detailed \
    "${step_limit_args[@]}" \
    --out "${DETAILED_METRICS_REPORT}"

  "${PY}" "${root}/src/logs/report_system/train_metrics_plots.py" \
    --metrics-jsonl "${METRICS_JSONL}" \
    --report-schema "${schema_path}" \
    "${step_limit_args[@]}" \
    --out-prefix "${plot_prefix}" >/dev/null
}

cosearch_generate_training_reports() {
  train_report_system_generate_reports "$@"
}

cosearch_generate_final_training_reports() {
  local root="$1"
  train_report_system_generate_reports "${root}" final
}

report_system_schema_path() {
  train_report_system_schema_path "$@"
}

report_system_max_metric_step() {
  train_report_system_max_metric_step "$@"
}

report_system_generate_training_reports() {
  train_report_system_generate_reports "$@"
}

cosearch_start_training_reporter() {
  local root="$1"

  (
    while true; do
      cosearch_generate_training_reports "${root}" >/dev/null 2>&1 || true
      sleep "${REPORT_INTERVAL_SECONDS:-60}"
    done
  ) &
  REPORTER_PGID=$!
  export REPORTER_PGID
}

cosearch_start_nvidia_smi_sampler() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found; GPU utilization sampling disabled" >&2
    NVIDIA_SMI_PGID=""
    export NVIDIA_SMI_PGID
    return
  fi

  echo "timestamp,index,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],power.draw [W]" > "${NVIDIA_SMI_CSV}"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits >> "${NVIDIA_SMI_CSV}" 2>/dev/null || true
      sleep "${NVIDIA_SMI_INTERVAL:-10}"
    done
  ) &
  NVIDIA_SMI_PGID=$!
  export NVIDIA_SMI_PGID
  echo "nvidia-smi sampling enabled: interval=${NVIDIA_SMI_INTERVAL:-10}s csv=${NVIDIA_SMI_CSV}"
}

cosearch_stop_background_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
}

setup_cosearch_summary_report_defaults() {
  local root="$1"
  local report_name="${2:-${RUN_NAME:-summary}}"
  resolve_cosearch_group_identity "${GROUP_NAME:-defaultGroup}"
  REPORT_ROOT="${REPORT_ROOT:-${root}/reports/train/${GROUP_SLUG}}"
  REPORT_PATH="${REPORT_PATH:-${REPORT_ROOT}/${report_name}.md}"

  mkdir -p "${REPORT_ROOT}"
  export REPORT_ROOT REPORT_PATH
}

setup_cosearch_eval_artifact_defaults() {
  local root="$1"
  local strategy_name="${2:-${STRATEGY_NAME:-default}}"
  resolve_cosearch_group_identity "${GROUP_NAME:-defaultGroup}"

  STRATEGY_SLUG="${STRATEGY_SLUG:-$(slugify_cosearch_name "${strategy_name}")}"
  TASK_NAME="${TASK_NAME:-$(date +%y%m%d-%H%M)-${STRATEGY_SLUG}}"
  EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${root}/log/eval_res/${GROUP_SLUG}}"
  EVAL_REPORT_ROOT="${EVAL_REPORT_ROOT:-${root}/reports/eval/${GROUP_SLUG}}"
  TRACE_DIR="${TRACE_DIR:-${EVAL_LOG_ROOT}/${TASK_NAME}}"
  REPORT_PATH="${REPORT_PATH:-${EVAL_REPORT_ROOT}/${TASK_NAME}.report.md}"
  RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-${TRACE_DIR}/runtime_logs}"

  mkdir -p "${EVAL_REPORT_ROOT}" "${TRACE_DIR}" "${RUNTIME_LOG_DIR}"
  export STRATEGY_SLUG TASK_NAME EVAL_LOG_ROOT EVAL_REPORT_ROOT TRACE_DIR REPORT_PATH RUNTIME_LOG_DIR
}

# Backward-compatible function name used by existing scripts.
setup_cosearch_logging_defaults() {
  setup_cosearch_training_log_defaults "$@"
}

#!/usr/bin/env bash

# AIR 路径初始化：只设置 AgenticIterRag 推理后端需要的根目录。
setup_agent_iteration_paths() {
  local root="$1"
  local parent
  parent="$(cd "${root}/.." && pwd)"

  AGENTIC_ITER_RAG_LEARN_ROOT="${AGENTIC_ITER_RAG_LEARN_ROOT:-${parent}/CoSearch_derevitives}"
  EXTERNAL_MODEL_ROOT="${EXTERNAL_MODEL_ROOT:-${parent}/models}"
  EXTERNAL_RETRIEVAL_ROOT="${EXTERNAL_RETRIEVAL_ROOT:-${AGENTIC_ITER_RAG_LEARN_ROOT}/data/retrieval}"
  LOCAL_FLASHRAG_ROOT="${LOCAL_FLASHRAG_ROOT:-${root}/data/AgenticIterRag/local_flashrag}"

  export AGENTIC_ITER_RAG_LEARN_ROOT EXTERNAL_MODEL_ROOT EXTERNAL_RETRIEVAL_ROOT LOCAL_FLASHRAG_ROOT
}

# AIR slug 规则：用于目录名、报告名和任务名，避免 shell 特殊字符进入路径。
slugify_air_name() {
  local raw="${1:-run}"
  local slug
  slug="$(printf '%s' "${raw}" | sed -E 's/[^A-Za-z0-9_.-]+/-/g; s/^-+//; s/-+$//')"
  if [[ -z "${slug}" ]]; then
    slug="run"
  fi
  printf '%s\n' "${slug}"
}

# AIR group 标识：生成 group slug，供日志和报告目录使用。
resolve_air_group_identity() {
  GROUP_NAME="${1:-${GROUP_NAME:-agenticIterRag}}"
  GROUP_SLUG="${GROUP_SLUG:-$(slugify_air_name "${GROUP_NAME}")}"
  export GROUP_NAME GROUP_SLUG
}

# AIR 训练运行标识：训练入口后续接入时使用，当前 infer backend 保留独立实现。
resolve_air_training_run_identity() {
  local default_group_name="${1:-agenticIterRag}"
  resolve_air_group_identity "${default_group_name}"
  RUN_NAME="${RUN_NAME:-${EXP_NAME:-air_run}}"
  RUN_SLUG="${RUN_SLUG:-$(slugify_air_name "${RUN_NAME}")}"
  export RUN_NAME RUN_SLUG
}

# AIR 日志默认目录：只给需要训练型 reporter 的入口使用，infer 入口会自行设置 TRACE_DIR。
setup_air_logging_defaults() {
  local root="${1:-${PWD}}"
  local experiment_name="${2:-${EXP_NAME:-air_run}}"
  resolve_air_group_identity "${GROUP_NAME:-agenticIterRag}"
  local safe_experiment
  safe_experiment="$(slugify_air_name "${experiment_name}")"
  LOG_ROOT="${LOG_ROOT:-${root}/log/agenticIterRag/${GROUP_SLUG}}"
  LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${safe_experiment}}"
  REPORT_ROOT="${REPORT_ROOT:-${root}/reports/agenticIterRag/${GROUP_SLUG}}"
  mkdir -p "${LOG_DIR}" "${REPORT_ROOT}"
  export LOG_ROOT LOG_DIR REPORT_ROOT
}

# AIR 安全检查：避免把运行目录误设成仓库根或空路径。
air_assert_safe_run_target() {
  local path="${1:-}"
  if [[ -z "${path}" || "${path}" == "/" ]]; then
    echo "ERROR: unsafe AIR run target path=${path}" >&2
    return 2
  fi
}

# AIR 训练报告占位：data produce 当前不依赖训练报告生成。
air_generate_training_reports() {
  return 0
}

# AIR 最终训练报告占位：data produce 当前不依赖训练报告生成。
air_generate_final_training_reports() {
  return 0
}

# AIR 后台训练报告占位：返回空 pid，避免调用方误等不存在的进程。
air_start_training_reporter() {
  return 0
}

# AIR GPU 采样占位：后续如需要再接入 AIR 自己的采样器。
air_start_nvidia_smi_sampler() {
  return 0
}

# AIR 后台进程停止：只按 pid 停止调用方显式传入的进程。
air_stop_background_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

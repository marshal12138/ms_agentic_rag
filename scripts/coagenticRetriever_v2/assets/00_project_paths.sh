#!/usr/bin/env bash

setup_agent_iteration_paths() {
  local root="$1"
  local parent
  parent="$(cd "${root}/.." && pwd)"

  COAGENTIC_LEARN_ROOT="${COAGENTIC_LEARN_ROOT:-${parent}/CoSearch_derevitives}"
  EXTERNAL_MODEL_ROOT="${EXTERNAL_MODEL_ROOT:-${parent}/models}"
  EXTERNAL_RETRIEVAL_ROOT="${EXTERNAL_RETRIEVAL_ROOT:-${COAGENTIC_LEARN_ROOT}/data/retrieval}"
  LOCAL_FLASHRAG_ROOT="${LOCAL_FLASHRAG_ROOT:-${root}/data/co_search/local_flashrag}"

  export COAGENTIC_LEARN_ROOT EXTERNAL_MODEL_ROOT EXTERNAL_RETRIEVAL_ROOT LOCAL_FLASHRAG_ROOT
}

slugify_coagentic_name() {
  slugify_cosearch_name "$@"
}

resolve_coagentic_group_identity() {
  resolve_cosearch_group_identity "$@"
}

resolve_coagentic_training_run_identity() {
  resolve_cosearch_training_run_identity "$@"
}

setup_coagentic_logging_defaults() {
  setup_cosearch_logging_defaults "$@"
}

setup_coagentic_eval_artifact_defaults() {
  setup_cosearch_eval_artifact_defaults "$@"
}

coagentic_assert_safe_run_target() {
  cosearch_assert_safe_run_target "$@"
}

coagentic_generate_training_reports() {
  cosearch_generate_training_reports "$@"
}

coagentic_generate_final_training_reports() {
  cosearch_generate_final_training_reports "$@"
}

coagentic_start_training_reporter() {
  cosearch_start_training_reporter "$@"
}

coagentic_start_nvidia_smi_sampler() {
  cosearch_start_nvidia_smi_sampler "$@"
}

coagentic_stop_background_pid() {
  cosearch_stop_background_pid "$@"
}

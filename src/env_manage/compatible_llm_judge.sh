#!/usr/bin/env bash

# Source this file from LLM judge launchers to resolve the vLLM runtime
# environment. Caller-provided LLM_JUDGE_* values are preserved; otherwise the
# historical /data04 environment is preferred, with /data05 as the compatible
# fallback.

LLM_JUDGE_COMPAT_ENV_DIRS=(
  "/data04/envs/ms/deepseek_v4"
  "/data05/conda/envs/ms/deepseek_v4"
)

_llm_judge_env_from_bin_path() {
  local path="$1"
  local bin_dir

  if [[ -z "${path}" ]]; then
    return 1
  fi
  bin_dir="$(dirname "${path}")"
  if [[ "$(basename "${bin_dir}")" != "bin" ]]; then
    return 1
  fi
  dirname "${bin_dir}"
}

_llm_judge_find_site_packages() {
  local env_dir="$1"
  local py_dir=""

  py_dir="$(find "${env_dir}/lib" -maxdepth 1 -type d -name 'python3*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${py_dir}" && -d "${py_dir}/site-packages" ]]; then
    printf '%s\n' "${py_dir}/site-packages"
    return 0
  fi

  printf '%s\n' "${env_dir}/lib/python3.11/site-packages"
}

_llm_judge_env_is_usable() {
  local env_dir="$1"
  [[ -x "${env_dir}/bin/python" && -x "${env_dir}/bin/vllm" ]]
}

_llm_judge_resolved_env_dir=""

if [[ -n "${LLM_JUDGE_COMPAT_ENV_DIR:-}" ]]; then
  _llm_judge_resolved_env_dir="${LLM_JUDGE_COMPAT_ENV_DIR}"
elif [[ -n "${LLM_JUDGE_PYTHON:-}" ]]; then
  _llm_judge_resolved_env_dir="$(_llm_judge_env_from_bin_path "${LLM_JUDGE_PYTHON}" || true)"
elif [[ -n "${LLM_JUDGE_VLLM:-}" ]]; then
  _llm_judge_resolved_env_dir="$(_llm_judge_env_from_bin_path "${LLM_JUDGE_VLLM}" || true)"
else
  for _llm_judge_env_dir in "${LLM_JUDGE_COMPAT_ENV_DIRS[@]}"; do
    if _llm_judge_env_is_usable "${_llm_judge_env_dir}"; then
      _llm_judge_resolved_env_dir="${_llm_judge_env_dir}"
      break
    fi
  done
fi

if [[ -n "${_llm_judge_resolved_env_dir}" ]]; then
  if [[ -z "${LLM_JUDGE_PYTHON:-}" && -x "${_llm_judge_resolved_env_dir}/bin/python" ]]; then
    export LLM_JUDGE_PYTHON="${_llm_judge_resolved_env_dir}/bin/python"
  fi
  if [[ -z "${LLM_JUDGE_VLLM:-}" && -x "${_llm_judge_resolved_env_dir}/bin/vllm" ]]; then
    export LLM_JUDGE_VLLM="${_llm_judge_resolved_env_dir}/bin/vllm"
  fi
  if [[ -z "${LLM_JUDGE_ENV_SITE_PACKAGES:-}" ]]; then
    export LLM_JUDGE_ENV_SITE_PACKAGES="$(_llm_judge_find_site_packages "${_llm_judge_resolved_env_dir}")"
  fi
fi

unset _llm_judge_resolved_env_dir
unset _llm_judge_env_dir
unset -f _llm_judge_env_from_bin_path
unset -f _llm_judge_find_site_packages
unset -f _llm_judge_env_is_usable

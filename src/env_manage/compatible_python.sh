#!/usr/bin/env bash

# Source this file from bash scripts to resolve their Python executable.
# A caller-provided PY is preserved; otherwise the historical default is tried
# first, then the compatible environment list below is searched.

COSEARCH_DEFAULT_PYTHON="/data04/envs/ms/ms_cosearch_official/bin/python"
COSEARCH_COMPAT_PYTHON_ENVS=(
  "/data05/conda/envs/ms/ms_agt_rag"
)

_cosearch_resolve_python() {
  local requested="${PY:-}"
  local candidate=""
  local env_dir=""

  if [[ -n "${requested}" ]]; then
    if [[ -d "${requested}" && -x "${requested}/bin/python" ]]; then
      PY="${requested}/bin/python"
    elif [[ -x "${requested}" ]]; then
      PY="${requested}"
    elif [[ "${requested}" != "${COSEARCH_DEFAULT_PYTHON}" ]]; then
      PY="${requested}"
    else
      requested=""
    fi
  fi

  if [[ -z "${requested}" ]]; then
    if [[ -x "${COSEARCH_DEFAULT_PYTHON}" ]]; then
      PY="${COSEARCH_DEFAULT_PYTHON}"
    else
      for env_dir in "${COSEARCH_COMPAT_PYTHON_ENVS[@]}"; do
        candidate="${env_dir}/bin/python"
        if [[ -x "${candidate}" ]]; then
          PY="${candidate}"
          break
        fi
      done
      PY="${PY:-${COSEARCH_DEFAULT_PYTHON}}"
    fi
  fi

  export PY
}

_cosearch_resolve_python
unset -f _cosearch_resolve_python

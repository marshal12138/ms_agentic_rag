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

_cosearch_prepend_path_entry() {
  local var_name="$1"
  local entry="$2"
  local current="${!var_name:-}"

  [[ -n "${entry}" && -d "${entry}" ]] || return 0
  case ":${current}:" in
    *":${entry}:"*) ;;
    *) export "${var_name}=${entry}${current:+:${current}}" ;;
  esac
}

_cosearch_export_python_libdir() {
  local py_libdir=""

  case "${COSEARCH_PREPEND_PYTHON_LIBDIR:-1}" in
    0|false|FALSE|no|NO|off|OFF) return 0 ;;
  esac

  py_libdir="$("${PY}" - <<'PY' 2>/dev/null || true
import sysconfig

print(sysconfig.get_config_var("LIBDIR") or "")
PY
)"
  [[ -n "${py_libdir}" ]] || return 0

  export COSEARCH_PYTHON_LIBDIR="${py_libdir}"
  _cosearch_prepend_path_entry LD_LIBRARY_PATH "${py_libdir}"
}

_cosearch_resolve_python
_cosearch_export_python_libdir
unset -f _cosearch_resolve_python
unset -f _cosearch_prepend_path_entry
unset -f _cosearch_export_python_libdir

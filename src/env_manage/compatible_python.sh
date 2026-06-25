#!/usr/bin/env bash

# Source this file from bash scripts to resolve their Python executable.
# A caller-provided PY is preserved; otherwise the historical /data04 default is
# tried first, then the compatible /data05 environment list below is searched.
# When the /data05 base env is selected, repo-local overlay packages are added
# automatically so direct script execution gets the same dependencies used by
# the NPU-compatible runtime without mutating the base conda env.

COSEARCH_DEFAULT_PYTHON="/data04/envs/ms/ms_cosearch_official/bin/python"
COSEARCH_COMPAT_PYTHON_ENVS=(
  "/data05/conda/envs/ms/ms_agt_rag"
)
COSEARCH_ENV_MANAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSEARCH_ROOT="$(cd "${COSEARCH_ENV_MANAGE_DIR}/../.." && pwd)"
COSEARCH_COMPAT_PYTHON_OVERLAY_DIRS=(
  "${COSEARCH_ROOT}/.venvs/ms_agt_rag_overlay"
)
COSEARCH_PYTHON_ENV_DIR=""

_cosearch_resolve_python() {
  local requested="${PY:-}"
  local candidate=""
  local env_dir=""

  if [[ -n "${requested}" ]]; then
    if [[ -d "${requested}" && -x "${requested}/bin/python" ]]; then
      PY="${requested}/bin/python"
      COSEARCH_PYTHON_ENV_DIR="${requested}"
    elif [[ -x "${requested}" ]]; then
      PY="${requested}"
      COSEARCH_PYTHON_ENV_DIR="$(_cosearch_env_from_python "${requested}" || true)"
    elif [[ "${requested}" != "${COSEARCH_DEFAULT_PYTHON}" ]]; then
      PY="${requested}"
    else
      requested=""
    fi
  fi

  if [[ -z "${requested}" ]]; then
    if [[ -x "${COSEARCH_DEFAULT_PYTHON}" ]]; then
      PY="${COSEARCH_DEFAULT_PYTHON}"
      COSEARCH_PYTHON_ENV_DIR="$(_cosearch_env_from_python "${PY}" || true)"
    else
      for env_dir in "${COSEARCH_COMPAT_PYTHON_ENVS[@]}"; do
        candidate="${env_dir}/bin/python"
        if [[ -x "${candidate}" ]]; then
          PY="${candidate}"
          COSEARCH_PYTHON_ENV_DIR="${env_dir}"
          break
        fi
      done
      PY="${PY:-${COSEARCH_DEFAULT_PYTHON}}"
    fi
  fi

  export PY
  export COSEARCH_PYTHON_ENV_DIR
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

_cosearch_env_from_python() {
  local path="$1"
  local bin_dir

  [[ -n "${path}" ]] || return 1
  bin_dir="$(dirname "${path}")"
  if [[ "$(basename "${bin_dir}")" != "bin" ]]; then
    return 1
  fi
  dirname "${bin_dir}"
}

_cosearch_find_site_packages() {
  local env_dir="$1"
  local py_dir=""

  py_dir="$(find "${env_dir}/lib" -maxdepth 1 -type d -name 'python3*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${py_dir}" && -d "${py_dir}/site-packages" ]]; then
    printf '%s\n' "${py_dir}/site-packages"
    return 0
  fi

  printf '%s\n' "${env_dir}/lib/python3.11/site-packages"
}

_cosearch_path_has_entry() {
  local value="$1"
  local entry="$2"
  case ":${value}:" in
    *":${entry}:"*) return 0 ;;
    *) return 1 ;;
  esac
}

_cosearch_should_use_overlay() {
  local overlay_dir="$1"

  case "${COSEARCH_ENABLE_PYTHON_OVERLAY:-1}" in
    0|false|FALSE|no|NO|off|OFF) return 1 ;;
  esac

  [[ -d "${overlay_dir}" && -x "${overlay_dir}/bin/python" ]] || return 1

  if [[ -n "${COSEARCH_PYTHON_OVERLAY_DIR:-}" ]]; then
    [[ "${overlay_dir}" == "${COSEARCH_PYTHON_OVERLAY_DIR}" ]]
    return $?
  fi

  # Keep /data04 pristine. The overlay is a dependency patch for the /data05
  # compatibility env only.
  [[ "${COSEARCH_PYTHON_ENV_DIR}" == "/data05/conda/envs/ms/ms_agt_rag" ]]
}

_cosearch_export_python_overlay() {
  local overlay_dir=""
  local site_packages=""

  for overlay_dir in "${COSEARCH_PYTHON_OVERLAY_DIR:-}" "${COSEARCH_COMPAT_PYTHON_OVERLAY_DIRS[@]}"; do
    [[ -n "${overlay_dir}" ]] || continue
    if _cosearch_should_use_overlay "${overlay_dir}"; then
      site_packages="$(_cosearch_find_site_packages "${overlay_dir}")"
      [[ -d "${site_packages}" ]] || continue

      export COSEARCH_PYTHON_OVERLAY_DIR="${overlay_dir}"
      export COSEARCH_PYTHON_OVERLAY_SITE_PACKAGES="${site_packages}"
      if ! _cosearch_path_has_entry "${PYTHONPATH:-}" "${site_packages}"; then
        export PYTHONPATH="${site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
      fi
      _cosearch_prepend_path_entry PATH "${overlay_dir}/bin"
      break
    fi
  done
}

_cosearch_export_python_path() {
  local env_dir="${COSEARCH_PYTHON_ENV_DIR:-}"

  if [[ -z "${env_dir}" ]]; then
    env_dir="$(_cosearch_env_from_python "${PY}" || true)"
  fi
  if [[ -n "${env_dir}" ]]; then
    _cosearch_prepend_path_entry PATH "${env_dir}/bin"
  fi
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
_cosearch_export_python_overlay
_cosearch_export_python_path
_cosearch_export_python_libdir
unset COSEARCH_ENV_MANAGE_DIR
unset COSEARCH_ROOT
unset -f _cosearch_resolve_python
unset -f _cosearch_prepend_path_entry
unset -f _cosearch_env_from_python
unset -f _cosearch_find_site_packages
unset -f _cosearch_path_has_entry
unset -f _cosearch_should_use_overlay
unset -f _cosearch_export_python_overlay
unset -f _cosearch_export_python_path
unset -f _cosearch_export_python_libdir

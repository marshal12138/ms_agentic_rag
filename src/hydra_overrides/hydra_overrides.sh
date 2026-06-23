#!/usr/bin/env bash

hydra_yaml_overrides_to_array() {
  local output_array_name="$1"
  shift

  local py="${1:-${PY:-python}}"
  shift || true

  local -a yaml_files=()
  local item
  for item in "$@"; do
    [[ -n "${item}" ]] || continue
    yaml_files+=("${item}")
  done

  eval "${output_array_name}=()"
  if [[ "${#yaml_files[@]}" -eq 0 ]]; then
    return 0
  fi

  local helper_dir
  helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local converter="${helper_dir}/yaml_to_dotlist.py"
  if [[ ! -f "${converter}" ]]; then
    echo "ERROR: Hydra override converter not found: ${converter}" >&2
    return 2
  fi

  local output status
  output="$("${py}" "${converter}" "${yaml_files[@]}")"
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi

  local -a overrides=()
  if [[ -n "${output}" ]]; then
    mapfile -t overrides <<< "${output}"
  fi

  local quoted=""
  if [[ "${#overrides[@]}" -gt 0 ]]; then
    printf -v quoted '%q ' "${overrides[@]}"
  fi
  eval "${output_array_name}=(${quoted})"
}

hydra_collect_yaml_override_files() {
  local output_array_name="$1"
  shift

  local -a yaml_files=()
  local -a _hydra_yaml_items=()
  local value item
  for value in "$@"; do
    [[ -n "${value}" ]] || continue
    read -r -a _hydra_yaml_items <<< "${value}"
    for item in "${_hydra_yaml_items[@]}"; do
      [[ -n "${item}" ]] || continue
      yaml_files+=("${item}")
    done
  done

  local quoted=""
  if [[ "${#yaml_files[@]}" -gt 0 ]]; then
    printf -v quoted '%q ' "${yaml_files[@]}"
  fi
  eval "${output_array_name}=(${quoted})"
}

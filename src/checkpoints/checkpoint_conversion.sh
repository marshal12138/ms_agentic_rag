#!/usr/bin/env bash

run_checkpoint_cleanup() {
  local root="$1"
  local out_dir="$2"
  local py="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
  local script="${root}/src/checkpoints/cleanup_checkpoint_layout.py"
  local -a trainable_roles=()
  local -a remove_root_dirs=()
  local -a remove_root_globs=()

  if [[ "${CHECKPOINT_RETENTION_AFTER_TRAIN:-1}" != "1" ]]; then
    echo "checkpoint cleanup disabled: CHECKPOINT_RETENTION_AFTER_TRAIN=${CHECKPOINT_RETENTION_AFTER_TRAIN:-0}"
    return 0
  fi
  if [[ ! -f "${script}" ]]; then
    echo "ERROR: checkpoint cleanup script not found: ${script}" >&2
    return 2
  fi
  if [[ ! -d "${out_dir}" ]]; then
    echo "checkpoint cleanup skipped: OUT_DIR not found: ${out_dir}" >&2
    return 0
  fi

  local -a args=(
    --checkpoint-root "${out_dir}"
  )
  read -r -a trainable_roles <<< "${CHECKPOINT_TRAINABLE_ROLES:-actor reranker_actor_rollout retriever}"
  args+=(--trainable-roles "${trainable_roles[@]}")

  if [[ "${CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS:-1}" != "" ]]; then
    args+=(--keep-latest-global-steps "${CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS:-1}")
  fi
  if [[ "${CHECKPOINT_DELETE_OLD_GLOBAL_STEPS:-1}" == "1" ]]; then
    args+=(--delete-old-global-steps)
  fi
  if [[ "${CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS:-1}" == "1" ]]; then
    args+=(--delete-empty-global-steps)
  fi
  if [[ -n "${CHECKPOINT_REMOVE_ROOT_DIRS:-}" ]]; then
    read -r -a remove_root_dirs <<< "${CHECKPOINT_REMOVE_ROOT_DIRS}"
    args+=(--remove-root-dirs "${remove_root_dirs[@]}")
  fi
  if [[ -n "${CHECKPOINT_REMOVE_ROOT_GLOBS:-}" ]]; then
    read -r -a remove_root_globs <<< "${CHECKPOINT_REMOVE_ROOT_GLOBS}"
    args+=(--remove-root-globs "${remove_root_globs[@]}")
  fi

  "${py}" "${script}" "${args[@]}"
}

latest_role_fsdp_checkpoint() {
  local out_dir="$1"
  local role="${2:-actor}"
  local latest_dir=""
  local latest_step=-1
  local dir base step role_dir

  [[ -d "${out_dir}" ]] || return 1
  for dir in "${out_dir}"/global_step_*; do
    [[ -d "${dir}" ]] || continue
    base="${dir##*/}"
    step="${base#global_step_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    role_dir="${dir}/${role}"
    [[ -d "${role_dir}" ]] || continue
    [[ -f "${role_dir}/fsdp_config.json" ]] || continue
    [[ -f "${role_dir}/huggingface/config.json" ]] || continue
    compgen -G "${role_dir}/model_world_size_*_rank_*.pt" >/dev/null || continue
    if (( step > latest_step )); then
      latest_step="${step}"
      latest_dir="${dir}"
    fi
  done

  [[ -n "${latest_dir}" ]] || return 1
  printf '%s\n' "${latest_dir}"
}

validate_role_hf_safetensors() {
  local out_dir="$1"
  local role="${2:-actor}"
  local target_subdir="${3:-hf_safetensors}"
  local latest_role_step=""
  local role_hf_dir=""

  if ! latest_role_step="$(latest_role_fsdp_checkpoint "${out_dir}" "${role}")"; then
    echo "checkpoint conversion validation skipped: no ${role} FSDP checkpoint found under ${out_dir}"
    return 0
  fi

  role_hf_dir="${latest_role_step}/${target_subdir}/${role}"
  if [[ ! -d "${role_hf_dir}" ]]; then
    echo "ERROR: ${role} HF safetensors directory not found: ${role_hf_dir}" >&2
    return 3
  fi
  if [[ ! -f "${role_hf_dir}/model.safetensors" && ! -f "${role_hf_dir}/model.safetensors.index.json" ]]; then
    echo "ERROR: ${role} HF safetensors model file/index not found under: ${role_hf_dir}" >&2
    return 3
  fi
  if ! compgen -G "${role_hf_dir}/*.safetensors" >/dev/null; then
    echo "ERROR: no *.safetensors shard found under: ${role_hf_dir}" >&2
    return 3
  fi
  if [[ ! -f "${role_hf_dir}/config.json" ]]; then
    echo "ERROR: ${role} HF config.json not found under: ${role_hf_dir}" >&2
    return 3
  fi

  echo "${role} HF safetensors validation passed: ${role_hf_dir}"
}

run_verl_fsdp_checkpoint_conversion() {
  local root="$1"
  local out_dir="$2"
  local py="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
  local script="${root}/src/checkpoints/convert_verl_fsdp_checkpoint.py"
  local -a convert_roles=()
  local required_hf_role="${REQUIRE_HF_SAFETENSORS_ROLE:-}"
  local target_subdir="${CHECKPOINT_CONVERT_TARGET_SUBDIR:-hf_safetensors}"
  local convert_status=0
  local role=""
  local found_role=""
  local -a converter_args=()

  if [[ ! -d "${out_dir}" ]]; then
    echo "checkpoint postprocess skipped: OUT_DIR not found: ${out_dir}" >&2
    return 0
  fi

  read -r -a convert_roles <<< "${CHECKPOINT_CONVERT_ROLES:-actor reranker_actor_rollout}"
  if [[ -z "${required_hf_role}" ]]; then
    for role in "${convert_roles[@]}"; do
      if latest_role_fsdp_checkpoint "${out_dir}" "${role}" >/dev/null; then
        found_role="${role}"
        break
      fi
    done
    required_hf_role="${found_role}"
  fi

  if [[ -z "${required_hf_role}" ]] || ! latest_role_fsdp_checkpoint "${out_dir}" "${required_hf_role}" >/dev/null; then
    echo "checkpoint conversion skipped: no requested FSDP checkpoint found under ${out_dir}; roles=${convert_roles[*]}"
    run_checkpoint_cleanup "${root}" "${out_dir}"
    return $?
  fi

  if [[ "${CONVERT_CHECKPOINT_AFTER_TRAIN:-1}" == "1" ]]; then
    if [[ ! -f "${script}" ]]; then
      echo "ERROR: checkpoint conversion script not found: ${script}" >&2
      return 2
    fi
    converter_args=(
      --checkpoint-root "${out_dir}" \
      --roles "${convert_roles[@]}" \
      --target-subdir "${target_subdir}" \
      --keep "${CHECKPOINT_CONVERT_KEEP:-1}" \
      --delete-empty
    )
    if [[ -n "${CHECKPOINT_VERL_ROOT:-}" ]]; then
      converter_args+=(--verl-root "${CHECKPOINT_VERL_ROOT}")
    fi
    "${py}" "${script}" "${converter_args[@]}" || convert_status=$?
    if [[ "${convert_status}" != "0" ]]; then
      echo "ERROR: checkpoint conversion failed with status ${convert_status}" >&2
      return "${convert_status}"
    fi
  else
    echo "checkpoint conversion disabled: CONVERT_CHECKPOINT_AFTER_TRAIN=${CONVERT_CHECKPOINT_AFTER_TRAIN:-0}"
  fi

  if [[ "${REQUIRE_HF_SAFETENSORS_AFTER_TRAIN:-1}" == "1" ]]; then
    validate_role_hf_safetensors "${out_dir}" "${required_hf_role}" "${target_subdir}" || return $?
  else
    echo "HF safetensors validation disabled: REQUIRE_HF_SAFETENSORS_AFTER_TRAIN=${REQUIRE_HF_SAFETENSORS_AFTER_TRAIN:-0}"
  fi

  run_checkpoint_cleanup "${root}" "${out_dir}"
}

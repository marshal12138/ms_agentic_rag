#!/usr/bin/env bash

# Generic serial task orchestration helpers.
# Source this file from eval/train task sequence scripts.

TASK_SEQUENCE_RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SEQUENCE_ROOT="$(cd "${TASK_SEQUENCE_RUNTIME_DIR}/../.." && pwd)"

source "${TASK_SEQUENCE_RUNTIME_DIR}/wait_for_gpus.sh"

task_sequence_is_truthy() {
  gpu_wait_is_truthy "${1:-}"
}

task_sequence_sanitize_label() {
  local label="${1:-task}"
  printf '%s' "${label}" | tr -cs 'A-Za-z0-9_.-' '-' | sed 's/^-//; s/-$//'
}

task_sequence_command_string() {
  local out="" arg
  for arg in "$@"; do
    out+="$(printf '%q' "${arg}") "
  done
  printf '%s' "${out% }"
}

task_sequence_now() {
  date '+%Y-%m-%d %H:%M:%S'
}

task_sequence_init() {
  if [[ -n "${TASK_SEQUENCE_INITIALIZED:-}" ]]; then
    return 0
  fi

  TASK_SEQUENCE_NAME="${TASK_SEQUENCE_NAME:-task_sequence}"
  TASK_SEQUENCE_STAMP="${TASK_SEQUENCE_STAMP:-$(date '+%y%m%d-%H%M%S')}"
  TASK_SEQUENCE_LOG_ROOT="${TASK_SEQUENCE_LOG_ROOT:-${TASK_SEQUENCE_ROOT}/log/task_sequences}"
  TASK_SEQUENCE_LOG_DIR="${TASK_SEQUENCE_LOG_DIR:-${TASK_SEQUENCE_LOG_ROOT}/${TASK_SEQUENCE_STAMP}-$(task_sequence_sanitize_label "${TASK_SEQUENCE_NAME}")}"
  TASK_SEQUENCE_SUMMARY="${TASK_SEQUENCE_SUMMARY:-${TASK_SEQUENCE_LOG_DIR}/summary.tsv}"
  TASK_SEQUENCE_INDEX="${TASK_SEQUENCE_INDEX:-0}"
  TASK_SEQUENCE_START_INDEX="${TASK_SEQUENCE_START_INDEX:-1}"
  TASK_SEQUENCE_CONTINUE_ON_FAIL="${TASK_SEQUENCE_CONTINUE_ON_FAIL:-0}"
  TASK_SEQUENCE_DRY_RUN="${TASK_SEQUENCE_DRY_RUN:-0}"
  TASK_SEQUENCE_WAIT_FOR_GPUS="${TASK_SEQUENCE_WAIT_FOR_GPUS:-1}"
  TASK_SEQUENCE_RELEASE_GPUS="${TASK_SEQUENCE_RELEASE_GPUS:-0}"
  TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY="${TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY:-1}"
  TASK_SEQUENCE_RELEASE_GRACE_SECONDS="${TASK_SEQUENCE_RELEASE_GRACE_SECONDS:-20}"

  mkdir -p "${TASK_SEQUENCE_LOG_DIR}"
  if [[ ! -f "${TASK_SEQUENCE_SUMMARY}" ]]; then
    printf 'index\taction\tlabel\tgpu_list\tcommand\tstart_time\tend_time\texit_code\tstatus\tlog_path\n' > "${TASK_SEQUENCE_SUMMARY}"
  fi
  TASK_SEQUENCE_INITIALIZED=1

  echo "task sequence log dir: ${TASK_SEQUENCE_LOG_DIR}"
}

task_sequence_summary_append() {
  local index="$1"
  local action="$2"
  local label="$3"
  local gpu_list="$4"
  local command="$5"
  local start_time="$6"
  local end_time="$7"
  local exit_code="$8"
  local status="$9"
  local log_path="${10}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${index}" "${action}" "${label}" "${gpu_list}" "${command}" \
    "${start_time}" "${end_time}" "${exit_code}" "${status}" "${log_path}" \
    >> "${TASK_SEQUENCE_SUMMARY}"
}

task_sequence_next_index() {
  task_sequence_init
  TASK_SEQUENCE_INDEX=$((TASK_SEQUENCE_INDEX + 1))
}

task_sequence_should_skip_index() {
  local index="$1"
  (( index < TASK_SEQUENCE_START_INDEX ))
}

task_sequence_restore_errexit() {
  local errexit_was_set="$1"
  if [[ "${errexit_was_set}" == "1" ]]; then
    set -e
  fi
}

task_sequence_maybe_stop_on_failure() {
  local status="$1"
  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi
  if task_sequence_is_truthy "${TASK_SEQUENCE_CONTINUE_ON_FAIL}"; then
    echo "task sequence: continuing after failure because TASK_SEQUENCE_CONTINUE_ON_FAIL=${TASK_SEQUENCE_CONTINUE_ON_FAIL}"
    return 0
  fi
  return "${status}"
}

task_sequence_wait_gpus() {
  local label="$1"
  local gpu_list="$2"
  task_sequence_init

  if [[ -z "${gpu_list//[[:space:],]/}" ]]; then
    echo "task sequence: no GPU wait requested for ${label}"
    return 0
  fi
  if ! task_sequence_is_truthy "${TASK_SEQUENCE_WAIT_FOR_GPUS}"; then
    echo "task sequence: GPU wait skipped for ${label}; TASK_SEQUENCE_WAIT_FOR_GPUS=${TASK_SEQUENCE_WAIT_FOR_GPUS}"
    return 0
  fi

  wait_for_gpu_release \
    "${gpu_list}" \
    "${WAIT_FOR_GPU_INTERVAL_SECONDS:-300}" \
    "${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}" \
    "${label}"
}

task_sequence_run() {
  local label="$1"
  local gpu_list="$2"
  shift 2
  if [[ "$#" -eq 0 ]]; then
    echo "ERROR: task_sequence_run requires a command." >&2
    return 2
  fi

  local index safe_label log_path command start_time end_time status errexit_was_set
  task_sequence_next_index
  index="${TASK_SEQUENCE_INDEX}"
  safe_label="$(task_sequence_sanitize_label "${label}")"
  log_path="${TASK_SEQUENCE_LOG_DIR}/$(printf '%03d' "${index}")-${safe_label}.log"
  command="$(task_sequence_command_string "$@")"

  if task_sequence_should_skip_index "${index}"; then
    echo "task sequence: skipping #${index} ${label}; TASK_SEQUENCE_START_INDEX=${TASK_SEQUENCE_START_INDEX}"
    task_sequence_summary_append "${index}" "run" "${label}" "${gpu_list}" "${command}" "" "" 0 "skipped" "${log_path}"
    return 0
  fi

  start_time="$(task_sequence_now)"
  echo "task sequence: starting #${index} ${label}"
  echo "  gpus: ${gpu_list:-none}"
  echo "  command: ${command}"
  echo "  log: ${log_path}"

  if task_sequence_is_truthy "${TASK_SEQUENCE_DRY_RUN}"; then
    end_time="$(task_sequence_now)"
    task_sequence_summary_append "${index}" "run" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" 0 "dry_run" "${log_path}"
    echo "task sequence: dry-run; command not executed."
    return 0
  fi

  if ! task_sequence_wait_gpus "${label}" "${gpu_list}"; then
    status=$?
    end_time="$(task_sequence_now)"
    task_sequence_summary_append "${index}" "run" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" "${status}" "gpu_wait_failed" "${log_path}"
    task_sequence_maybe_stop_on_failure "${status}"
    return $?
  fi

  errexit_was_set=0
  case "$-" in
    *e*) errexit_was_set=1; set +e ;;
  esac
  "$@" 2>&1 | tee "${log_path}"
  status="${PIPESTATUS[0]}"
  task_sequence_restore_errexit "${errexit_was_set}"

  end_time="$(task_sequence_now)"
  if [[ "${status}" -eq 0 ]]; then
    task_sequence_summary_append "${index}" "run" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" "${status}" "ok" "${log_path}"
    echo "task sequence: finished #${index} ${label}"
    return 0
  fi

  task_sequence_summary_append "${index}" "run" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" "${status}" "failed" "${log_path}"
  echo "task sequence: failed #${index} ${label}; exit_code=${status}" >&2
  task_sequence_maybe_stop_on_failure "${status}"
}

task_sequence_collect_gpu_processes() {
  local gpu_list="$1"
  local output_file="$2"
  local tmp_dir gpu_file gpu_info_file gpu_uuid_file gpu_apps_file

  gpu_wait_validate_gpu_list "${gpu_list}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required for GPU process collection." >&2
    return 2
  fi

  tmp_dir="$(mktemp -d)"
  gpu_file="${tmp_dir}/gpus.txt"
  gpu_info_file="${tmp_dir}/gpu_info.txt"
  gpu_uuid_file="${tmp_dir}/gpu_uuid.txt"
  gpu_apps_file="${tmp_dir}/gpu_apps.txt"

  gpu_wait_csv_to_lines "${gpu_list}" > "${gpu_file}"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits > "${gpu_info_file}"
  awk -F', ' 'NR==FNR {want[$1]=1; next} ($1 in want) {print $1 "\t" $2}' \
    "${gpu_file}" "${gpu_info_file}" > "${gpu_uuid_file}"

  local wanted_count mapped_count
  wanted_count="$(wc -l < "${gpu_file}" | tr -d ' ')"
  mapped_count="$(wc -l < "${gpu_uuid_file}" | tr -d ' ')"
  if [[ "${wanted_count}" != "${mapped_count}" ]]; then
    echo "ERROR: some requested GPU ids do not exist: ${gpu_list}" >&2
    rm -rf "${tmp_dir}"
    return 2
  fi

  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits \
    > "${gpu_apps_file}" 2>/dev/null || true

  awk -F'\t' '
    NR==FNR {uuid_to_gpu[$2]=$1; next}
    {
      split($0, fields, ", ")
      uuid=fields[1]
      if (uuid in uuid_to_gpu) {
        print uuid_to_gpu[uuid] "\t" fields[2] "\t" fields[3] "\t" fields[4]
      }
    }
  ' "${gpu_uuid_file}" "${gpu_apps_file}" > "${output_file}"

  rm -rf "${tmp_dir}"
}

task_sequence_release_gpus() {
  local label="$1"
  local gpu_list="$2"
  local index safe_label log_path command start_time end_time status
  local process_file filtered_file pid_file current_user

  task_sequence_next_index
  index="${TASK_SEQUENCE_INDEX}"
  safe_label="$(task_sequence_sanitize_label "${label}")"
  log_path="${TASK_SEQUENCE_LOG_DIR}/$(printf '%03d' "${index}")-${safe_label}.log"
  command="release ${gpu_list}"
  start_time="$(task_sequence_now)"

  if task_sequence_should_skip_index "${index}"; then
    echo "task sequence: skipping #${index} ${label}; TASK_SEQUENCE_START_INDEX=${TASK_SEQUENCE_START_INDEX}"
    task_sequence_summary_append "${index}" "release" "${label}" "${gpu_list}" "${command}" "" "" 0 "skipped" "${log_path}"
    return 0
  fi

  echo "task sequence: release #${index} ${label}"
  echo "  gpus: ${gpu_list}"
  echo "  log: ${log_path}"

  if task_sequence_is_truthy "${TASK_SEQUENCE_DRY_RUN}"; then
    end_time="$(task_sequence_now)"
    task_sequence_summary_append "${index}" "release" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" 0 "dry_run" "${log_path}"
    echo "task sequence: dry-run; GPU release not executed."
    return 0
  fi

  process_file="$(mktemp)"
  filtered_file="$(mktemp)"
  pid_file="$(mktemp)"
  current_user="$(id -un)"
  status=0

  {
    echo "target GPUs: ${gpu_list}"
    echo "release enabled: ${TASK_SEQUENCE_RELEASE_GPUS}"
    echo "current-user-only: ${TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY}"
    echo "grace seconds: ${TASK_SEQUENCE_RELEASE_GRACE_SECONDS}"
    echo

    if ! task_sequence_collect_gpu_processes "${gpu_list}" "${process_file}"; then
      status=$?
      echo "failed to collect GPU processes; status=${status}"
    elif [[ ! -s "${process_file}" ]]; then
      echo "no compute processes found on target GPUs."
    else
      echo "matched GPU processes:"
      echo -e "gpu\tpid\towner\tused_memory_mib\tprocess_name"
      while IFS=$'\t' read -r gpu pid process_name used_memory; do
        [[ -z "${pid}" ]] && continue
        owner="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}')"
        [[ -z "${owner}" ]] && owner="unknown"
        echo -e "${gpu}\t${pid}\t${owner}\t${used_memory}\t${process_name}"
        if task_sequence_is_truthy "${TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY}" && [[ "${owner}" != "${current_user}" ]]; then
          continue
        fi
        echo -e "${pid}\t${owner}\t${process_name}" >> "${filtered_file}"
      done < "${process_file}"

      cut -f1 "${filtered_file}" 2>/dev/null | sed '/^$/d' | sort -n | uniq > "${pid_file}"
      if [[ ! -s "${pid_file}" ]]; then
        echo
        echo "no releasable processes matched the current release policy."
      elif ! task_sequence_is_truthy "${TASK_SEQUENCE_RELEASE_GPUS}"; then
        echo
        echo "release is disabled. Set TASK_SEQUENCE_RELEASE_GPUS=1 to send signals."
        echo "candidate PIDs:"
        cat "${pid_file}"
      else
        echo
        echo "sending SIGTERM to PIDs:"
        cat "${pid_file}"
        while IFS= read -r pid; do
          kill -TERM "${pid}" 2>/dev/null || true
        done < "${pid_file}"

        sleep "${TASK_SEQUENCE_RELEASE_GRACE_SECONDS}"

        local remaining_file
        remaining_file="$(mktemp)"
        while IFS= read -r pid; do
          if kill -0 "${pid}" 2>/dev/null; then
            echo "${pid}" >> "${remaining_file}"
          fi
        done < "${pid_file}"

        if [[ -s "${remaining_file}" ]]; then
          echo
          echo "PIDs still alive after grace period; sending SIGKILL:"
          cat "${remaining_file}"
          while IFS= read -r pid; do
            kill -KILL "${pid}" 2>/dev/null || true
          done < "${remaining_file}"
          sleep 1
        fi
        rm -f "${remaining_file}"

        local still_alive_file
        still_alive_file="$(mktemp)"
        while IFS= read -r pid; do
          if kill -0 "${pid}" 2>/dev/null; then
            echo "${pid}" >> "${still_alive_file}"
          fi
        done < "${pid_file}"
        if [[ -s "${still_alive_file}" ]]; then
          echo
          echo "ERROR: some PIDs are still alive after SIGKILL:"
          cat "${still_alive_file}"
          status=1
        else
          echo
          echo "release completed."
        fi
        rm -f "${still_alive_file}"
      fi
    fi
  } 2>&1 | tee "${log_path}"

  rm -f "${process_file}" "${filtered_file}" "${pid_file}"
  end_time="$(task_sequence_now)"

  if [[ "${status}" -eq 0 ]]; then
    if task_sequence_is_truthy "${TASK_SEQUENCE_RELEASE_GPUS}"; then
      task_sequence_summary_append "${index}" "release" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" 0 "ok" "${log_path}"
    else
      task_sequence_summary_append "${index}" "release" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" 0 "disabled" "${log_path}"
    fi
    return 0
  fi

  task_sequence_summary_append "${index}" "release" "${label}" "${gpu_list}" "${command}" "${start_time}" "${end_time}" "${status}" "failed" "${log_path}"
  task_sequence_maybe_stop_on_failure "${status}"
}

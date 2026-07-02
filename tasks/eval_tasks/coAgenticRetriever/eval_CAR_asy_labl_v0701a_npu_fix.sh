#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

cd "${ROOT}"
export GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"
export EVAL_TASK_NAME="${EVAL_TASK_NAME:-async_label_dpskv4f_v0702_no_ranker}"

bash "${ROOT}/scripts/coagenticRetriever_v2/02_infer_launcher.sh" \
  --main_run_config=coAgenticRetriever_main \
  --EVAL_RUNTIME_CONFIG=coagentic_retriever_vllm \
  --EVAL_BUDGET_CONFIG=coagentic_retriever_aligned_budget \
  --RESOURCE_CONFIG=local_eval_4gpu_0_3 \
  --OVERLAY_YAML=tasks/eval_tasks/coAgenticRetriever/configs/eval_CAR_asy_labl_v0701a_npu_fix_overlay.yaml \
  "$@"

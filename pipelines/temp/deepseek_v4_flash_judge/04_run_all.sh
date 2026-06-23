#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/data04/envs/ms/deepseek_v4/bin/python}"
ENV_SITE_PACKAGES="${ENV_SITE_PACKAGES:-/data04/envs/ms/deepseek_v4/lib/python3.11/site-packages}"
export LD_LIBRARY_PATH="${ENV_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${ENV_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PORT="${PORT:-8067}"
MODEL="${MODEL:-DeepSeek-V4-Flash}"
URL="${URL:-http://127.0.0.1:${PORT}/v1/chat/completions}"
DATA="${DATA:-/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/llm_judge/chunk_ranking/examples/chunk_ranking_judge_examples_100.jsonl}"
LIMIT="${LIMIT:-100}"
CONCURRENCY="${CONCURRENCY:-1}"
RESULT_DIR="${RESULT_DIR:-${PIPELINE_DIR}/results/$(date '+%Y%m%d_%H%M%S')}"

mkdir -p "${RESULT_DIR}" "${PIPELINE_DIR}/logs"

bash "${PIPELINE_DIR}/00_start_vllm_gpu06_07.sh"

echo "waiting for vLLM: http://127.0.0.1:${PORT}/v1/models"
for i in $(seq 1 240); do
  if "${PY}" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5).status < 500 else 1)" >/dev/null 2>&1; then
    echo "vLLM ready"
    break
  fi
  if [[ "${i}" == "240" ]]; then
    echo "ERROR: vLLM did not become ready; see logs" >&2
    exit 2
  fi
  sleep 5
done

"${PY}" "${PIPELINE_DIR}/01_prepare_chunk_ranking_prompts.py" \
  --input "${DATA}" \
  --output-dir "${RESULT_DIR}/intermediate" \
  --limit "${LIMIT}" | tee "${RESULT_DIR}/prepare.log"

"${PY}" "${PIPELINE_DIR}/02_run_judge_benchmark.py" \
  --requests "${RESULT_DIR}/intermediate/requests_think.jsonl" \
  --output "${RESULT_DIR}/outputs_think.jsonl" \
  --summary "${RESULT_DIR}/summary_think.json" \
  --url "${URL}" \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --limit "${LIMIT}" | tee "${RESULT_DIR}/run_think.log"

"${PY}" "${PIPELINE_DIR}/02_run_judge_benchmark.py" \
  --requests "${RESULT_DIR}/intermediate/requests_no_think.jsonl" \
  --output "${RESULT_DIR}/outputs_no_think.jsonl" \
  --summary "${RESULT_DIR}/summary_no_think.json" \
  --url "${URL}" \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --limit "${LIMIT}" | tee "${RESULT_DIR}/run_no_think.log"

"${PY}" "${PIPELINE_DIR}/03_summarize_report.py" \
  --result-dir "${RESULT_DIR}" \
  --output "${RESULT_DIR}/report.md"

ln -sfn "${RESULT_DIR}" "${PIPELINE_DIR}/results/latest"
echo "report: ${RESULT_DIR}/report.md"

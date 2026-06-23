#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JUDGE_SCRIPT="${SCRIPT_DIR}/llm_judge_dpskv4f.sh"
CONFIG_PATH="${LLM_JUDGE_CONFIG:-${PROJECT_ROOT}/CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"
CLIENT_PYTHON="${LLM_JUDGE_CLIENT_PYTHON:-/data04/envs/ms/ms_cosearch_official/bin/python}"

EXAMPLE_DATA="${EXAMPLE_DATA:-${PROJECT_ROOT}/data/llm_judge/chunk_ranking/examples/chunk_ranking_judge_examples_100.jsonl}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/run_examples}"
INPUT_JSON="${INPUT_JSON:-${WORK_DIR}/single_request.json}"
OUTPUT_JSON="${OUTPUT_JSON:-${WORK_DIR}/single_result.json}"

mkdir -p "${WORK_DIR}"

if [[ -z "${REQUEST_JSON:-}" ]]; then
  "${CLIENT_PYTHON}" - "${EXAMPLE_DATA}" "${INPUT_JSON}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.is_file():
    raise SystemExit(f"example data not found: {src}")
for line in src.read_text(encoding="utf-8").splitlines():
    if line.strip():
        dst.write_text(line.strip() + "\n", encoding="utf-8")
        break
else:
    raise SystemExit(f"no JSONL rows found: {src}")
PY
else
  INPUT_JSON="${REQUEST_JSON}"
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

"${JUDGE_SCRIPT}" call \
  --config "${CONFIG_PATH}" \
  --input "${INPUT_JSON}" \
  --output "${OUTPUT_JSON}" \
  "${DRY_RUN_ARGS[@]}"

echo "single result: ${OUTPUT_JSON}"

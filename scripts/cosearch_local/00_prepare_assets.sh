#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"

cd "${ROOT}"
mkdir -p log

for path in \
  "/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-0.6B" \
  "/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B" \
  "/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required model asset not found: ${path}" >&2
    echo "Run the standalone download script before training or inference." >&2
    exit 2
  fi
done

"${PY}" scripts/cosearch_local/prepare_cosearch_data.py \
  --out-root data/co_search/local_flashrag \
  --smoke-train-per-source 16 \
  --smoke-eval-per-source 8

"${PY}" scripts/cosearch_local/check_paper_mechanics.py | tee log/check_paper_mechanics.json

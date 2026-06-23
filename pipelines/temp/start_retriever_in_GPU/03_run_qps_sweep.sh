#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PIPELINE_DIR}/../../.." && pwd)"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"

GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-8050}"
URL="${URL:-http://127.0.0.1:${PORT}/retrieve}"
READY_URL="${READY_URL:-http://127.0.0.1:${PORT}/docs}"
TOPK="${TOPK:-50}"
QUERY_LIMIT="${QUERY_LIMIT:-5000}"
BENCH_DURATION="${BENCH_DURATION:-20}"
BENCH_WARMUP="${BENCH_WARMUP:-5}"

mkdir -p "${PIPELINE_DIR}/logs" "${PIPELINE_DIR}/queries" "${PIPELINE_DIR}/results" "${PIPELINE_DIR}/run"

QUERY_FILE="${QUERY_FILE:-${PIPELINE_DIR}/queries/co_search_ablation_queries.jsonl}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${ROOT}/data/co_search/local_flashrag/co_search_ablation.train.parquet}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
RESULT_DIR="${PIPELINE_DIR}/results/${STAMP}"
mkdir -p "${RESULT_DIR}"

"${PY}" "${PIPELINE_DIR}/01_prepare_queries.py" \
  --input "${TRAIN_PARQUET}" \
  --output "${QUERY_FILE}" \
  --limit "${QUERY_LIMIT}" | tee "${RESULT_DIR}/prepare_queries.log"

GPU_ID="${GPU_ID}" PORT="${PORT}" START_BACKGROUND=1 \
  bash "${PIPELINE_DIR}/00_start_gpu05_retriever.sh" | tee "${RESULT_DIR}/start_retriever.log"

echo "waiting for retriever: ${READY_URL}"
for i in $(seq 1 180); do
  if "${PY}" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('${READY_URL}', timeout=5).status < 500 else 1)" >/dev/null 2>&1; then
    echo "retriever ready"
    break
  fi
  if [[ "${i}" == "180" ]]; then
    echo "ERROR: retriever did not become ready; see logs" >&2
    exit 2
  fi
  sleep 2
done

{
  echo "gpu before benchmark:"
  nvidia-smi -i "${GPU_ID}" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
} | tee "${RESULT_DIR}/gpu_before.log"

BATCH_SIZES="${BATCH_SIZES:-1 4 8 16 32}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"

for bs in ${BATCH_SIZES}; do
  for cc in ${CONCURRENCIES}; do
    out="${RESULT_DIR}/bench_b${bs}_c${cc}.json"
    echo "benchmark batch=${bs} concurrency=${cc}"
    "${PY}" "${PIPELINE_DIR}/02_benchmark_retriever_qps.py" \
      --url "${URL}" \
      --queries "${QUERY_FILE}" \
      --output "${out}" \
      --topk "${TOPK}" \
      --batch-size "${bs}" \
      --concurrency "${cc}" \
      --duration "${BENCH_DURATION}" \
      --warmup "${BENCH_WARMUP}" | tee "${out}.stdout"
    nvidia-smi -i "${GPU_ID}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "${RESULT_DIR}/gpu_samples.csv" || true
  done
done

"${PY}" "${PIPELINE_DIR}/04_summarize_results.py" \
  --results-dir "${RESULT_DIR}" \
  --output "${RESULT_DIR}/summary.md"

ln -sfn "${RESULT_DIR}" "${PIPELINE_DIR}/results/latest"
echo "summary: ${RESULT_DIR}/summary.md"

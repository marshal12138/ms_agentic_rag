#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
PORT="${PORT:-8010}"
URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PORT}/retrieve}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
QUERY="${QUERY:-who got the first nobel prize in physics?}"
EXPECT_CONTAINS="${EXPECT_CONTAINS:-Röntgen}"
TOPK="${TOPK:-5}"
WARMUP="${WARMUP:-1}"
REQUESTS="${REQUESTS:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
DEVICE="${DEVICE:-cpu}"
FAISS_GPU="${FAISS_GPU:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
LOG="${LOG:-${ROOT}/log/retrieval_verify/bench_server_${PORT}_${DEVICE}_faissgpu${FAISS_GPU}.log}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT}/log/retrieval_verify/bench_${PORT}_${DEVICE}_faissgpu${FAISS_GPU}.json}"
RESOURCE_JSON="${RESOURCE_JSON:-${ROOT}/log/retrieval_verify/resources_${PORT}_${DEVICE}_faissgpu${FAISS_GPU}.json}"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")" "$(dirname "${OUTPUT_JSON}")" "$(dirname "${RESOURCE_JSON}")"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

: > "${LOG}"
PORT="${PORT}" \
DEVICE="${DEVICE}" \
FAISS_GPU="${FAISS_GPU}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
bash scripts/cosearch_local/02b_start_dense_retriever_server.sh > "${LOG}" 2>&1 &
SERVER_PID=$!
echo "Started dense retriever server pid=${SERVER_PID}; log=${LOG}"

start_ts="$(date +%s)"
while true; do
  if "${PY}" scripts/cosearch_local/check_dense_retriever_http.py \
      --url "${URL}" \
      --query "${QUERY}" \
      --topk "${TOPK}" \
      --expect-contains "${EXPECT_CONTAINS}" \
      --quiet >/dev/null 2>&1; then
    break
  fi

  if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: dense retriever server exited before benchmark. Log: ${LOG}" >&2
    tail -120 "${LOG}" >&2 || true
    exit 1
  fi

  now="$(date +%s)"
  if (( now - start_ts > STARTUP_TIMEOUT )); then
    echo "ERROR: dense retriever did not verify in ${STARTUP_TIMEOUT}s. Log: ${LOG}" >&2
    tail -120 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Resource snapshot before benchmark:"
ps -o pid,ppid,etime,pcpu,pmem,rss,vsz,nlwp,cmd -p "${SERVER_PID}"

"${PY}" scripts/cosearch_local/benchmark_dense_retriever_http.py \
  --url "${URL}" \
  --query "${QUERY}" \
  --topk "${TOPK}" \
  --warmup "${WARMUP}" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --output-json "${OUTPUT_JSON}"

echo "Resource snapshot after benchmark:"
ps -o pid,ppid,etime,pcpu,pmem,rss,vsz,nlwp,cmd -p "${SERVER_PID}"

"${PY}" - "${SERVER_PID}" "${RESOURCE_JSON}" <<'PY'
import json
import os
import resource
import sys
from pathlib import Path

pid = int(sys.argv[1])
out = Path(sys.argv[2])

page_size = os.sysconf("SC_PAGE_SIZE")
phys_pages = os.sysconf("SC_PHYS_PAGES")
total_mem_bytes = page_size * phys_pages
cpu_count = os.cpu_count() or 1

status = {}
with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
    for line in fh:
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()

statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
resident_pages = int(statm[1])
rss_bytes = resident_pages * page_size

summary = {
    "pid": pid,
    "cpu_count": cpu_count,
    "total_mem_bytes": total_mem_bytes,
    "rss_bytes": rss_bytes,
    "rss_gib": rss_bytes / (1024**3),
    "rss_percent_of_total_mem": rss_bytes * 100 / total_mem_bytes,
    "threads": int(status.get("Threads", "0")),
    "voluntary_context_switches": status.get("voluntary_ctxt_switches"),
    "nonvoluntary_context_switches": status.get("nonvoluntary_ctxt_switches"),
    "ru_maxrss_kib_for_sampler": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Resource JSON:", out)
print(json.dumps(summary, indent=2))
PY

echo "Benchmark JSON: ${OUTPUT_JSON}"
echo "Server log: ${LOG}"

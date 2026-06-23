#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
PORT="${PORT:-8010}"
URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PORT}/retrieve}"
START_SERVER="${START_SERVER:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
QUERY="${QUERY:-who got the first nobel prize in physics?}"
EXPECT_CONTAINS="${EXPECT_CONTAINS:-Röntgen}"
TOPK="${TOPK:-5}"
LOG="${LOG:-${ROOT}/log/retrieval_verify/server_${PORT}.log}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT}/log/retrieval_verify/response_${PORT}.json}"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")" "$(dirname "${OUTPUT_JSON}")"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${START_SERVER}" == "1" || "${START_SERVER}" == "true" || "${START_SERVER}" == "yes" ]]; then
  : > "${LOG}"
  PORT="${PORT}" bash scripts/cosearch_local/02b_start_dense_retriever_server.sh > "${LOG}" 2>&1 &
  SERVER_PID=$!
  echo "Started dense retriever server pid=${SERVER_PID}; log=${LOG}"
else
  echo "Using existing dense retriever at ${URL}"
fi

start_ts="$(date +%s)"
while true; do
  if "${PY}" scripts/cosearch_local/check_dense_retriever_http.py \
      --url "${URL}" \
      --query "${QUERY}" \
      --topk "${TOPK}" \
      --expect-contains "${EXPECT_CONTAINS}" \
      --output-json "${OUTPUT_JSON}" \
      --quiet >/dev/null 2>&1; then
    break
  fi

  if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: dense retriever server exited before verification succeeded. Log: ${LOG}" >&2
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

"${PY}" scripts/cosearch_local/check_dense_retriever_http.py \
  --url "${URL}" \
  --query "${QUERY}" \
  --topk "${TOPK}" \
  --expect-contains "${EXPECT_CONTAINS}" \
  --output-json "${OUTPUT_JSON}"

echo "Response JSON: ${OUTPUT_JSON}"

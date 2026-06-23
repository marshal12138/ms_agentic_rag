#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-data/retrieval/wiki-18}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"

cd "${ROOT}"

"${PY}" scripts/cosearch_local/download_modelscope_assets.py --download-e5

mkdir -p "${RETRIEVAL_DATA_DIR}"

if [[ -L "${CORPUS_FILE}" ]]; then
  rm "${CORPUS_FILE}"
fi

if [[ ! -f "${RETRIEVAL_DATA_DIR}/part_aa" || ! -f "${RETRIEVAL_DATA_DIR}/part_ab" || \
      ( ! -f "${CORPUS_FILE}" && ! -f "${RETRIEVAL_DATA_DIR}/wiki-18.jsonl.gz" ) ]]; then
  "${PY}" scripts/cosearch_local/download_modelscope_retrieval_assets.py \
    --out-dir "${RETRIEVAL_DATA_DIR}"
fi

if [[ ! -f "${INDEX_FILE}" ]]; then
  if [[ ! -f "${RETRIEVAL_DATA_DIR}/part_aa" || ! -f "${RETRIEVAL_DATA_DIR}/part_ab" ]]; then
    echo "ERROR: official Search-R1 index parts are missing under ${RETRIEVAL_DATA_DIR}" >&2
    exit 2
  fi
  cat "${RETRIEVAL_DATA_DIR}/part_aa" "${RETRIEVAL_DATA_DIR}/part_ab" > "${INDEX_FILE}"
  echo "Merged official Search-R1 index parts -> ${INDEX_FILE}"
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  if [[ ! -f "${RETRIEVAL_DATA_DIR}/wiki-18.jsonl.gz" ]]; then
    echo "ERROR: official Search-R1 corpus archive is missing: ${RETRIEVAL_DATA_DIR}/wiki-18.jsonl.gz" >&2
    exit 2
  fi
  gzip -dk "${RETRIEVAL_DATA_DIR}/wiki-18.jsonl.gz"
  echo "Decompressed official Search-R1 corpus -> ${CORPUS_FILE}"
fi

if file "${CORPUS_FILE}" | grep -q "tar archive"; then
  tmp_corpus="${CORPUS_FILE}.tmp"
  tar -xOf "${CORPUS_FILE}" > "${tmp_corpus}"
  mv "${tmp_corpus}" "${CORPUS_FILE}"
  echo "Extracted official Search-R1 corpus tar payload -> ${CORPUS_FILE}"
fi

"${PY}" scripts/cosearch_local/verify_official_retrieval_assets.py \
  --index "${INDEX_FILE}" \
  --corpus "${CORPUS_FILE}"

echo "Official Search-R1 retrieval assets are ready:"
echo "  index:  ${INDEX_FILE}"
echo "  corpus: ${CORPUS_FILE}"

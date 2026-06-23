#!/usr/bin/env bash
set -euo pipefail

# Paper-path CoSearch training smoke:
# E5 dense retriever + official CoSearch VERL trainer + Qwen3-0.6B replacement base model.
# This keeps the core paper mechanics but shrinks batch/rollout/context for limited GPU memory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-8010}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

export PORT GPU_IDS
export RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PORT}/retrieve}"
export RERANKER_TRAINABLE="${RERANKER_TRAINABLE:-true}"
export OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints/paper_qwen3_dense_smoke}"
export EXP_NAME="${EXP_NAME:-paper_qwen3_0_6b_dense_joint_smoke}"

export TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_rl_smoke.train.parquet}"
export VAL_DATA="${VAL_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_7bench_smoke.eval.parquet}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export TOTAL_STEPS="${TOTAL_STEPS:-1}"
export N_ROLLOUTS="${N_ROLLOUTS:-2}"
export MAX_TURNS="${MAX_TURNS:-2}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-1}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-2}"
export TOP_N="${TOP_N:-50}"
export TOP_M="${TOP_M:-5}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-11264}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
export AGENT_WORKERS="${AGENT_WORKERS:-1}"
export TEMPERATURE="${TEMPERATURE:-0.7}"

bash "${ROOT}/scripts/cosearch_local/train_cosearch_verl_base.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR="$HERE/runtime"
LOG_DIR="$RUNTIME_DIR/logs"
MODEL_PATH=${MODEL_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/GLM-4.7-Flash}
MODEL_NAME=${MODEL_NAME:-GLM-4.7-Flash}
IMAGE=${IMAGE:-m.daocloud.io/quay.io/ascend/vllm-ascend:v0.21.0rc1}
WAIT_SECONDS=${WAIT_SECONDS:-900}

mkdir -p "$LOG_DIR"
rm -f "$RUNTIME_DIR/replicas.pending.tsv"

ports=(8067 8068 8069 8070)
cards=("0,1" "2,3" "4,5" "6,7")

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

docker image inspect "$IMAGE" >/dev/null

for index in "${!ports[@]}"; do
  port=${ports[$index]}
  card_pair=${cards[$index]}
  name="teacher_pe_glm47_${port}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d \
    --name "$name" \
    --privileged --net=host --ipc=host \
    -e ASCEND_RT_VISIBLE_DEVICES="$card_pair" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_DISABLE_FLASHINFER=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=0 \
    -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
    -e VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}" \
    -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    -v /data01:/data01 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    "$IMAGE" \
    bash -lc "exec vllm serve '$MODEL_PATH' \
      --served-model-name '$MODEL_NAME' \
      --host 0.0.0.0 \
      --port '$port' \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 32000 \
      --max-num-seqs 128 \
      --max-num-batched-tokens 65536 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --kv-cache-dtype auto \
      --moe-backend auto \
      --dtype bfloat16 \
      --disable-custom-all-reduce" >/dev/null
  echo -e "$name\t$port\t$card_pair" >> "$RUNTIME_DIR/replicas.pending.tsv"
done

mv "$RUNTIME_DIR/replicas.pending.tsv" "$RUNTIME_DIR/replicas.tsv"

deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  ready=0
  for port in "${ports[@]}"; do
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=$((ready + 1))
    fi
  done
  echo "teacher replicas ready=${ready}/4 elapsed=${SECONDS}s"
  if (( ready == 4 )); then
    for port in "${ports[@]}"; do
      docker logs "teacher_pe_glm47_${port}" > "$LOG_DIR/teacher_pe_glm47_${port}.log" 2>&1 || true
    done
    exit 0
  fi
  for port in "${ports[@]}"; do
    name="teacher_pe_glm47_${port}"
    if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" != "true" ]]; then
      echo "container exited before readiness: $name" >&2
      docker logs "$name" >&2 || true
      exit 1
    fi
  done
  sleep 15
done

for port in "${ports[@]}"; do
  name="teacher_pe_glm47_${port}"
  echo "===== $name =====" >&2
  docker logs --tail 120 "$name" >&2 || true
done
echo "teacher replicas did not become ready within ${WAIT_SECONDS}s" >&2
exit 1

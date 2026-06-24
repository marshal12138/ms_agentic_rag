# CoAgenticRetriever Local Scripts

This directory contains local entries for the retriever contrastive framework.

## Files

```text
00_start_dense_retriever_server.sh
01_train_qwen3_4b_ablation_1epoch_timing.sh
02_infer_qwen3_4b_ablation_val_only.sh
assets/
```

The scripts default to:

```text
PROJECT_ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever
TRAIN_DATA=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/local_flashrag/co_search_ablation.train.parquet
VAL_DATA=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/local_flashrag/co_search_ablation.eval.parquet
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
```

The retriever contrastive worker and local dense retrieval server are
CUDA-only. If CUDA is not visible to PyTorch, the local entries fail
immediately.

`00_start_dense_retriever_server.sh` is the frozen recall service launcher. It
calls `src/retrievers/gpu_dense_retriever_server.py` directly, binds the service
to GPU05 by default through `CUDA_VISIBLE_DEVICES=5`, and loads the FAISS flat
doc embedding matrix into GPU memory as a torch tensor (`DOC_DTYPE=float16` by
default). It does not use the legacy Search-R1 CPU retrieval server.

## Full Mode

Default training mode is full mode. It launches the CoAgenticRetriever/VERL
trainer, updates the agent LLM through the normal PPO/GRPO path, enriches
rollout tool traces with the current dense rank retriever's top50 ordering, and
updates the trainable shared-encoder E5 rank retriever through contrastive loss.

Default GPU intent:

```text
GPU00-03: agent LLM VERL workers
GPU04: trainable dense rank retriever
GPU05: frozen recall retrieval service
```

This physical GPU mapping is represented in two places: YAML defaults under
`CoAgenticRetriever/config/retriever_contrastive.yaml`, and override variables
in the local scheduling scripts. The `CoAgenticRetriever` core Python code only
consumes configured device strings and does not hard-code GPU IDs.

In `01_train_qwen3_4b_ablation_1epoch_timing.sh`, `AGENT_GPU_IDS` is the
explicit agent-LLM physical GPU list. `GPU_IDS` is the full-mode process
visible GPU list and defaults to `${AGENT_GPU_IDS},${RANK_GPU_ID}`.

Training scripts no longer use a fixed default `RUN_NAME`. Pass `EXP_NAME`,
and the script constructs a unique `RUN_NAME=<timestamp>-<EXP_NAME>`. By
default, a non-empty existing log/checkpoint target causes the script to abort
instead of reusing that directory.

```bash
EXP_NAME=qwen3_4b_probe_rule_v1 \
TOTAL_STEPS=2 \
RETRIEVER_CONTRASTIVE_BATCH_SIZE=4 \
RETRIEVER_NEG_PER_POS=3 \
bash scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

Outputs:

```text
log/train_logs/<YYMMDD-HHMM>-coagentic_ranker_contrastive_smoke/
log/train_logs/<YYMMDD-HHMM>-coagentic_ranker_contrastive_smoke/coagentic_ranker_contrastive_smoke.train.log
log/train_logs/<YYMMDD-HHMM>-coagentic_ranker_contrastive_smoke/coagentic_ranker_contrastive_smoke.metrics.jsonl
log/train_logs/<YYMMDD-HHMM>-coagentic_ranker_contrastive_smoke/coagentic_ranker_contrastive_smoke.contrastive_construction.jsonl
checkpoints/qwen3_4b_probe/coagentic_ranker_contrastive_smoke/
```

The checkpoint directory is not created by dry-run or logging setup. Rollout
and validation traces default to the train log directory; the checkpoint
directory is reserved for actual model checkpoint writes. Retained trainable
checkpoint content is kept under `global_step_*/`, and old root-level
`retriever/`, `rollout_data/`, and similar legacy residue is cleaned after a
successful run.

Canonical full training mode is `RUN_MODE=full`.

## Dense Reranker Only

This mode validates only the dense rank retriever contrastive framework. It
does not start full LLM rollout and does not update the agent LLM. It still
starts the same frozen recall service through `00_start_dense_retriever_server.sh`
by default, then trains the dense rank retriever from recall top50.

```bash
EXP_NAME=qwen3_4b_dense_only_rule_v1 \
RUN_MODE=ranker-only \
TOTAL_STEPS=2 \
RETRIEVER_CONTRASTIVE_BATCH_SIZE=4 \
RETRIEVER_NEG_PER_POS=3 \
bash scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

## Dense Reranker Only Inference

Loads the trained `rank_encoder`, starts the frozen recall service by default,
and uses the shared E5 rank encoder to rerank top50 -> top5.
`CHECKPOINT_DIR` can point either to the run root or directly to
`global_step_*/ranker`.

```bash
MAX_EVAL_STEPS=1 \
CHECKPOINT_DIR=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_ranker_contrastive_smoke \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

Output:

```text
log/eval_res/<TASK_NAME>/ranker_infer_smoke.jsonl
log/eval_res/<TASK_NAME>/runtime_logs/coagentic_ranker_infer_smoke.infer.log
```

## Full VERL Entry

The full-mode path uses `CoAgenticRetriever/main_coagentic_retriever.py`
and enables:

```text
trainer.ranker_trainable=true
trainer.ranker_update_mode=contrastive
trainer.ranker_steps_per_global_step=2
save_top_n_documents=true
```

`save_top_n_documents=true` is required because retriever training consumes the
top-N documents saved by `CoAgenticRetrieverTool` into rollout `tool_call_details`.

## Key Overrides

```bash
RUN_MODE=full|ranker-only
AUTO_START_RECALL_SERVICE=1
AUTO_STOP_RECALL_SERVICE=1
TOTAL_STEPS=2
TRAIN_BATCH_SIZE=64
ACTOR_BATCH_SIZE=64
N_ROLLOUTS=8
RETRIEVER_CONTRASTIVE_BATCH_SIZE=32
RETRIEVER_NEG_PER_POS=15
RETRIEVER_POSITIVE_TOP_K=5
RETRIEVER_TEMPERATURE=0.05
AGENT_GPU_IDS=0,1,2,3
AGENT_N_GPUS_PER_NODE=4
GPU_IDS=0,1,2,3,4
RECALL_GPU_ID=5
RANK_GPU_ID=4
RANKER_VISIBLE_DEVICE_INDEX=4
RECALL_RETRIEVER_CONFIG_DEVICE=cuda:5
RANKER_CONFIG_DEVICE=cuda:4
PROXY_PORT=8030
RETRIEVAL_SERVICE_URL=http://127.0.0.1:8030/retrieve
RECALL_TOP_K=50
RANK_TOP_K=5
RANKER_DEVICE_TRAIN=cuda:0
RECALL_RETRIEVER_DEVICE=cuda:1
RETRIEVER_DEVICE=cuda
DENSE_RERANKER_ONLY_CUDA_VISIBLE_DEVICES=4,5
DRY_RUN=1
```

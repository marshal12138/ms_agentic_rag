# IterRAG Scripts

This directory keeps the clean entry points for the new `Agent_Iteration_Rag`
project.

The top-level directory should only expose this README plus:

```text
00_start_dense_retriever_server.sh
01_train_qwen3_4b_ablation_1epoch_timing.sh
02_infer_qwen3_4b_ablation_val_only.sh
```

Helper scripts are kept under:

```text
assets/
```

Both scripts default to the copied core framework:

```text
AgenticIterRag/
```

Override it with `COSEARCH_PROJECT_ROOT=/path/to/core` when needed.

## Training

```bash
bash scripts/iterRag_scripts/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

`01_train_qwen3_4b_ablation_1epoch_timing.sh` starts retrievers, starts
the retrieval proxy, runs periodic reports, and calls
`assets/00_run_agentic_iter_rag_verl.sh`.

Retrieval service startup is implemented in this directory:

```text
00_start_dense_retriever_server.sh
```

Its header contains the external resource dependency block for the shared model, index, corpus, and Search-R1 retrieval server paths.

## Inference / Validation

```bash
bash scripts/iterRag_scripts/02_infer_qwen3_4b_ablation_val_only.sh
```

`02_infer_qwen3_4b_ablation_val_only.sh` starts CPU dense retrievers, starts a round-robin proxy, then runs the AgenticIterRag trainer in `val_only` mode on:

```text
data/co_search/local_flashrag/co_search_ablation.eval.parquet
```

Useful overrides:

```bash
RESUME_FROM_PATH=/path/to/global_step_xxx bash scripts/iterRag_scripts/02_infer_qwen3_4b_ablation_val_only.sh
VAL_MAX_SAMPLES=8 bash scripts/iterRag_scripts/02_infer_qwen3_4b_ablation_val_only.sh
DRY_RUN=1 bash scripts/iterRag_scripts/02_infer_qwen3_4b_ablation_val_only.sh
```

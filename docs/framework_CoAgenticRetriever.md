# CoAgenticRetriever Framework Implementation

This document describes the current CoAgenticRetriever implementation in
`CoAgenticRetriever/`. It follows the Fix Ver2 design in
`docs/planning/260610_retriever_contrastive_step_plan_fix_ver2 copy.md`, with
the current code naming normalized around `ranker` rather than the older
`retriever` wording.

The framework keeps the agent LLM PPO/GRPO path intact and adds a trainable E5
dense ranker. The frozen recall retriever serves recall top-50 documents. The
trainable ranker reranks that fixed pool and is updated with contrastive loss.
The ranker uses one shared encoder for query and document encoding, following
the Agentic-R / Tevatron dense retriever pattern.

## Core Data Flow

```text
agent rollout
  -> CoAgenticRetrieverTool
      -> frozen recall retriever HTTP service returns recall_top50_docs
      -> local E5 ranker reranks recall top50
      -> rank_top5_docs are formatted into the tool response for the agent
      -> recall_top50_docs / rank_top50_docs / rank_top5_docs are saved in tool_call_details
  -> main_batch
      -> process_main_agent_ppo_step
      -> actor_rollout_wg.update_actor
  -> tool_call_details / fresh trajectories
      -> trajectory_selector
      -> signal_builder
      -> sample_builder
      -> replay_buffer
      -> collator
      -> ranker_wg.update_ranker_contrastive
```

The ranker update path is separate from PPO/GRPO. Ranker updates do not compute
LLM log-prob, advantage, return, KL, or actor updates. The agent global step is
still the outer training clock; `ranker_steps_per_global_step` controls how many
ranker contrastive updates are attempted per agent step.

## Directory Structure

```text
CoAgenticRetriever/
  config/
    coagentic_retriever_trainer.yaml
    ranker_contrastive.yaml
    coagentic_retriever_tool_config.yaml

  ranker_strategies/
    __init__.py
    schemas.py
    config.py
    collator.py
    replay_buffer.py
    logging_utils.py

    trajectory_selector/
      __init__.py
      top_f1.py
      best_and_worst_f1.py

    signal_builder/
      __init__.py
      topk_pseudo_rank.py

    sample_builder/
      __init__.py
      random_negative_repeat.py

  verl/verl/tools/
    coagentic_retriever_tool.py

  verl/verl/trainer/ppo/
    coagentic_retriever_ray_trainer.py
    coagentic_ranker_contrastive_ray_trainer.py
    ranker_contrastive_step.py

  verl/verl/workers/ranker/
    e5_ranker_worker.py

  main_coagentic_retriever.py
```

Local execution wrappers live outside the framework directory:

```text
scripts/coagenticRetriever_local/
  00_start_dense_retriever_server.sh
  01_train_qwen3_4b_ablation_1epoch_timing.sh
  02_infer_qwen3_4b_ablation_val_only.sh
  04_train_qwen3_4b_coagentic_ablation_1epoch_timing.sh
  05_infer_qwen3_4b_coagentic_ablation_val_only.sh
  strategies_yaml/ranker_contrastive_new_sampling.yaml
```

## Directory Roles

`config/coagentic_retriever_trainer.yaml`

Hydra entry config for full CoAgenticRetriever training. It loads the standard
VERL actor/data/ref/rollout/model/critic/reward configs and then includes
`ranker_contrastive.yaml`.

`config/ranker_contrastive.yaml`

Defines the frozen recall retriever, trainable ranker, ranker strategy knobs,
loss, replay buffer, optimizer, and trainer switches:

```yaml
trainer:
  ranker_trainable: true
  ranker_update_mode: contrastive
  ranker_steps_per_global_step: 2

recall_retriever:
  top_k: 50
  trainable: false
  index_refresh: false
  service_url: http://127.0.0.1:8030/retrieve

ranker:
  shared_encoder: true
  top_k: 5

ranker_training:
  batch_size: 32
```

`config/coagentic_retriever_tool_config.yaml`

Configures `CoAgenticRetrieverTool`. The tool calls the frozen recall retriever
service, optionally runs the local dense ranker, returns only the final top-M
documents to the agent, and records recall/rank traces for later ranker
training.

`ranker_strategies/`

Contains CPU/driver-side strategy code for building ranker contrastive training
batches from rollout traces. This package does not own model forward/backward
logic.

`ranker_strategies/schemas.py`

Defines plain dataclasses used across strategy modules:

```text
RankedPassage
ToolCallContext
RankerTrajectory
LabeledPassage
LabeledRankingContext
ContrastiveSample
```

The current trace fields are `recall_top50_docs`, `rank_top50_docs`,
`rank_top5_docs`, and `ranked_passages`. `rank_top50_docs` and
`ranked_passages` represent the full ranker-sorted recall pool used for signal
and sample construction. `rank_top5_docs` is the view passed to the agent.

`ranker_strategies/config.py`

Factory layer for strategy objects. It reads `ranker_training.*` fields and
constructs the selector, signal builder, sample builder, replay buffer,
collator, and construction logger.

`ranker_strategies/trajectory_selector/`

Trajectory selection strategy package.

- `top_f1.py`: selects high-F1 valid trajectories and normalizes rollout tool
  call traces into `ToolCallContext`.
- `best_and_worst_f1.py`: selects top-k and bottom-n valid trajectories by final
  F1/reward. The local override
  `scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml`
  enables this strategy.

`ranker_strategies/signal_builder/`

Supervision signal strategy package. The current implementation is
`topk_pseudo_rank.py`, which marks the ranker-sorted top-k passages as positives
and the remaining ranker-sorted passages as negatives.

`ranker_strategies/sample_builder/`

Contrastive sample construction strategy package. The current implementation is
`random_negative_repeat.py`, which builds repeated positive groups and samples
negatives from the recall top-50 pool.

`ranker_strategies/collator.py`

Tokenizes `ContrastiveSample` objects into a ranker `DataProto` batch: query
tokens, grouped doc tokens, positive index labels, loss weights, and non-tensor
trace metadata.

`ranker_strategies/replay_buffer.py`

Stores fresh and historical contrastive samples. Fresh samples are added only on
`ranker_step_idx == 0`; each ranker update samples a fresh/replay mixture using
`ranker_training.replay_buffer.fresh_ratio`.

`ranker_strategies/logging_utils.py`

Prints one complete contrastive construction example every configured interval
and on the first smoke step. The log includes query, recall top50 sample,
ranker top5 sample, positive doc, negatives, ranks, scores, and label source.

`verl/verl/tools/coagentic_retriever_tool.py`

Search tool used during async rollout. It calls the frozen recall HTTP service,
normalizes recall docs, runs `LocalE5RankerWorker.rank_topk` when
`ranker_enabled=true`, formats final docs for the agent, computes tool reward,
and stores trace fields in `tool_call_details`.

`verl/verl/trainer/ppo/coagentic_retriever_ray_trainer.py`

Base CoAgenticRetriever trainer for the agent path. It provides the normal
agent rollout and PPO/GRPO update behavior that the ranker trainer extends.

`verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py`

Full integration point for joint agent/ranker training. It initializes the
local ranker worker and ranker strategy components, runs async rollout, enriches
tool calls with current ranker top-50 traces, launches the main agent PPO step,
and inserts ranker contrastive updates before collecting the agent update
result.

`verl/verl/trainer/ppo/ranker_contrastive_step.py`

Pure orchestration for ranker contrastive updates. It selects trajectories,
builds signals, builds contrastive samples, updates the replay buffer, collates
a batch, calls `ranker_wg.update_ranker_contrastive`, and returns ranker timing
and metric keys. It does not reuse PPO/GRPO actor-update semantics.

`verl/verl/workers/ranker/e5_ranker_worker.py`

GPU-side shared-encoder E5 ranker worker. It loads one trainable `rank_encoder`,
uses the same encoder for query and docs, supports recall top-50 reranking,
computes InfoNCE loss, updates the shared encoder, and saves checkpoints under
`ranker/rank_encoder/`.

`main_coagentic_retriever.py`

Hydra/Ray entry point for launching `CoAgenticRankerContrastiveRayTrainer`.

## Configuration Points

Main trainer switches:

```yaml
trainer:
  ranker_trainable: true
  ranker_update_mode: contrastive
  ranker_steps_per_global_step: 2
  ranker_start_step: 0
  ranker_stop_step: null
```

Recall retriever:

```yaml
recall_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:5
  top_k: 50
  trainable: false
  index_refresh: false
  service_url: http://127.0.0.1:8030/retrieve
```

`recall_retriever.device` is a configuration default, not a Python code
constant. Local scripts start the frozen recall service on physical GPU05 by
default and expose it through `:8030`.

Ranker:

```yaml
ranker:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  encoder_path: null
  device: cuda:4
  shared_encoder: true
  top_k: 5
  max_query_length: 256
  max_doc_length: 512
```

The ranker worker is CUDA-only. It fails fast when CUDA is unavailable or when
`ranker.device` / `ranker_training.device` is set to a non-CUDA device.

Trajectory selection:

```yaml
ranker_training:
  trajectory_selector:
    type: top_f1_trajectories
    max_selected_trajectories: 1
    min_final_reward: 0.0
```

Alternative local sampling:

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
    min_final_reward: 0.0
```

Supervision signal:

```yaml
ranker_training:
  signal_builder:
    type: topk_pseudo_rank
    positive_top_k: 50
    allow_all_negative: false
```

Sample construction:

```yaml
ranker_training:
  sample_builder:
    type: random_negative_repeat
    num_groups_per_step: 32
    neg_per_pos: 15
    allow_repeat_negative_sampling: true
    use_in_batch_negatives: false
```

Loss and optimization:

```yaml
ranker_training:
  loss:
    type: info_nce
    temperature: 0.05
  max_grad_norm: 1.0
  optim:
    lr: 2.0e-5
    weight_decay: 0.01
    warmup_steps: 0
    total_steps: 1000
```

Replay buffer:

```yaml
ranker_training:
  replay_buffer:
    enable: true
    max_size: 200000
    fresh_ratio: 0.5
```

Construction logging:

```yaml
ranker_training:
  log_every_n_steps: 10
  log_first_sample: true
  construction_log_jsonl: null
```

## Runtime Modes

Full mode:

```text
RUN_MODE=full
```

This launches the CoAgenticRetriever/VERL trainer, updates the agent LLM through
the normal PPO/GRPO path, enriches rollout traces with the current dense ranker
ordering, and updates the trainable E5 ranker through contrastive loss.

Ranker-only mode:

```text
RUN_MODE=ranker-only
```

This validates the dense ranker contrastive framework without full LLM rollout
or agent LLM update. It still starts the frozen recall service and trains the
ranker from recall top-50 candidates.

Dense-ranker inference:

```text
scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
scripts/coagenticRetriever_local/05_infer_qwen3_4b_coagentic_ablation_val_only.sh
```

These entries load a trained `rank_encoder`, start the frozen recall service by
default, and rerank recall top-50 to top-5 for evaluation.

## First-Version Constraints

- Recall retriever is frozen and serves top-50 from a fixed E5/FAISS space.
- Ranker is trainable and uses one shared E5 encoder for query/docs.
- Ranker reranks recall top-50; it does not perform full-corpus ANN search.
- Recall FAISS doc index is not refreshed by ranker training.
- Ranker contrastive batches use their own schema instead of PPO fields.
- Fresh samples are added to replay buffer only on `ranker_step_idx == 0`.
- `query_input = origin_query + " [SEP] " + sub_query`.
- Full training currently asserts async rollout mode in
  `CoAgenticRankerContrastiveRayTrainer.fit`.

## Validation Commands

Physical GPU placement is not hard-coded in framework Python code. The YAML
defaults and local scheduling scripts use this intended placement:

```text
GPU00-03: agent LLM VERL workers
GPU04: trainable dense ranker
GPU05: frozen recall retrieval service
```

Both local train and inference entries start the frozen recall service through
`scripts/coagenticRetriever_local/00_start_dense_retriever_server.sh` by default.
This launcher calls `src/retrievers/gpu_dense_retriever_server.py`, exposes the
service on `:8030`, and loads the E5 flat doc embedding matrix into GPU memory
by default.

Full smoke:

```bash
EXP_NAME=qwen3_4b_probe_rule_v1 \
RUN_MODE=full \
TOTAL_STEPS=2 \
RETRIEVER_CONTRASTIVE_BATCH_SIZE=4 \
RETRIEVER_NEG_PER_POS=3 \
bash scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

CoAgenticRetriever ablation data wrapper:

```bash
EXP_NAME=coagentic_qwen3_probe_rule_v1 \
RUN_MODE=full \
TOTAL_STEPS=2 \
bash scripts/coagenticRetriever_local/04_train_qwen3_4b_coagentic_ablation_1epoch_timing.sh
```

Ranker-only smoke:

```bash
EXP_NAME=qwen3_4b_dense_only_rule_v1 \
RUN_MODE=ranker-only \
TOTAL_STEPS=2 \
RETRIEVER_CONTRASTIVE_BATCH_SIZE=4 \
RETRIEVER_NEG_PER_POS=3 \
bash scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

Ranker inference smoke:

```bash
MAX_EVAL_STEPS=1 \
CHECKPOINT_DIR=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_ranker_contrastive_smoke \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

## Key Local Overrides

```text
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
RECALL_RETRIEVER_CONFIG_DEVICE=cuda:5
RANKER_CONFIG_DEVICE=cuda:4
PROXY_PORT=8030
RETRIEVAL_SERVICE_URL=http://127.0.0.1:8030/retrieve
RECALL_TOP_K=50
RANK_TOP_K=5
RANKER_DEVICE_TRAIN=cuda:0
DRY_RUN=1
```

The local scripts still expose some `RETRIEVER_*` environment variable names for
backward compatibility. Inside the framework, the current canonical config
fields are `ranker.*` and `ranker_training.*`.

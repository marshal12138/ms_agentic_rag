# 04 训练脚本到训练完成的全流程说明

本文梳理 `scripts/coagenticRetriever_local/04_train_qwen3_4b_coagentic_ablation_1epoch_timing.sh` 从入口到训练完成的完整链路。目标是把 shell 参数、服务启动、Hydra 配置、VERL/Ray worker 初始化、agent rollout、tool 调用、reward、ranker contrastive 更新、actor 更新、checkpoint 和日志产物串起来，便于后续对齐和排查。

当前说明对应 2026-06-11 的代码状态。重点路径：

- 入口 wrapper：`scripts/coagenticRetriever_local/04_train_qwen3_4b_coagentic_ablation_1epoch_timing.sh`
- 训练主脚本：`scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
- 共享 VERL launcher：`scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh`
- Python 入口：`CoAgenticRetriever/main_coagentic_retriever.py`
- Trainer：`CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py`
- Agent loop：`CoAgenticRetriever/verl/verl/experimental/agent_loop/coagentic_retriever_agent_loop.py`
- Search tool：`CoAgenticRetriever/verl/verl/tools/coagentic_retriever_tool.py`
- Ranker worker：`CoAgenticRetriever/verl/verl/workers/ranker/e5_ranker_worker.py`
- Reward：`CoAgenticRetriever/rewards/search_qa_f1_with_format_penalty.py`

## 1. 总体链路

04 训练不是一个独立训练实现，而是一个数据路径 wrapper。它只把训练/验证数据切到 `data/coAgenticRetriever/albation_1`，然后复用 01 的完整训练链路。

完整调用链：

```text
04_train_qwen3_4b_coagentic_ablation_1epoch_timing.sh
  -> 01_train_qwen3_4b_ablation_1epoch_timing.sh
    -> ensure_recall_service()
      -> 00_start_dense_retriever_server.sh
    -> assets/00_run_agentic_iter_rag_verl.sh
      -> CoAgenticRetriever/main_coagentic_retriever.py
        -> CoAgenticRankerTaskRunner.run()
          -> CoAgenticRankerContrastiveRayTrainer.init_workers()
          -> CoAgenticRankerContrastiveRayTrainer.fit()
            -> async rollout manager generate_sequences()
              -> CoAgenticRetrieverAgentLoop.run()
                -> CoAgenticRetrieverTool.execute()
            -> process_main_agent_ppo_step.remote()
            -> process_ranker_contrastive_step()
            -> _save_checkpoint()
```

从训练语义看，一个 global step 内有两条训练线：

1. main agent 线：模型 rollout 生成多轮工具调用轨迹，然后按 QA reward 做 GRPO/PPO actor update。
2. ranker 线：从 rollout 里提取 search tool 的 `tool_call_details`，构造 ranker contrastive 样本，用 E5 shared encoder 做 InfoNCE 更新。

这两条线共享同一个 rollout 轨迹来源。main agent 的目标是学会正确调用工具并最终回答；ranker 的目标是把 recall top-N 文档重排成更适合回答的 top-M 文档。

## 2. 04 wrapper 做了什么

文件：`scripts/coagenticRetriever_local/04_train_qwen3_4b_coagentic_ablation_1epoch_timing.sh`

核心逻辑：

```bash
: "${TRAIN_DATA:=${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet}"
: "${VAL_DATA:=${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet}"
: "${GROUP_NAME:=coAgenticRetriever}"

export TRAIN_DATA VAL_DATA GROUP_NAME
exec bash "${SCRIPT_DIR}/01_train_qwen3_4b_ablation_1epoch_timing.sh" "$@"
```

作用：

- `TRAIN_DATA` 指向 `data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet`
- `VAL_DATA` 指向 `data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- `GROUP_NAME=coAgenticRetriever`
- 之后完全交给 01 脚本。

因此，04 和 01 的训练过程应该保持一致，差异只应该来自数据路径。若 04 和 01 出现行为差异，优先检查 wrapper 传入的数据内容、prompt 格式和样本字段，而不是训练框架本身。

## 3. 01 主训练脚本阶段

文件：`scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`

01 是真正的本地训练入口，主要负责运行身份、路径、GPU、服务、日志、ranker 参数和最终 launcher 调用。

### 3.1 初始化路径和运行身份

脚本开头设置：

```bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
source "${ASSETS_DIR}/00_project_paths.sh"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
source "${ROOT}/src/checkpoints/checkpoint_conversion.sh"
setup_agent_iteration_paths "${ROOT}"
```

然后设置：

```bash
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}"
EXP_NAME="${EXP_NAME:-}"
GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"
resolve_coagentic_training_run_identity "${ROOT}" "" 1 "${GROUP_NAME}"
setup_coagentic_logging_defaults "${ROOT}" "${RUN_NAME}"
```

关键结果：

- `PROJECT_ROOT` 是 CoAgenticRetriever 子工程。
- `RUN_NAME`、`GROUP_SLUG`、`LOG_DIR`、`OUT_DIR` 等由 helper 统一生成。
- 训练日志默认在：
  `log/train_logs/coAgenticRetriever/<RUN_NAME>/`
- checkpoint 默认在：
  `checkpoints/qwen3_4b_probe/coAgenticRetriever/<RUN_NAME>/`

### 3.2 RUN_MODE

01 支持：

```bash
RUN_MODE=full
RUN_MODE=co-training
RUN_MODE=ranker-only
```

当前 04 的目标是 full 训练：

- `RUN_MODE=full` 和 `RUN_MODE=co-training` 都映射到 `EFFECTIVE_RUN_MODE=full`
- `ranker-only` 只跑 dense ranker contrastive smoke，不启动完整 LLM rollout

本文只描述 `EFFECTIVE_RUN_MODE=full` 的主流程。

### 3.3 GPU 切分

01 默认：

```bash
AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3}"
RECALL_GPU_ID="${RECALL_GPU_ID:-5}"
RANK_GPU_ID="${RANK_GPU_ID:-4}"
GPU_IDS="${GPU_IDS:-${AGENT_GPU_IDS},${RANK_GPU_ID}}"
RANKER_VISIBLE_DEVICE_INDEX="${RANKER_VISIBLE_DEVICE_INDEX:-${AGENT_N_GPUS_PER_NODE}}"
```

默认资源含义：

- main agent actor/rollout 用 `AGENT_GPU_IDS`
- trainable ranker worker 用 `RANK_GPU_ID`
- frozen recall retriever service 用 `RECALL_GPU_ID`
- `GPU_IDS` 传给 launcher 后成为 `CUDA_VISIBLE_DEVICES`
- `RANKER_VISIBLE_DEVICE_INDEX` 是在 `CUDA_VISIBLE_DEVICES` 内的相对 index，通常等于 agent GPU 数量。例如 `GPU_IDS=0,1,2,3,4` 时，ranker 在 visible index `4` 上，即物理 GPU 4。

### 3.4 关键训练参数

01 顶层暴露的关键参数：

```bash
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-64}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-5100}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
```

注意：

- 当前默认 `LORA_RANK=0`，即默认不使用 LoRA。
- 如果需要 LoRA，必须显式传入，例如 `LORA_RANK=8 LORA_ALPHA=16`。
- GRPO smoke 不应把 `TRAIN_BATCH_SIZE/ACTOR_BATCH_SIZE` 设得小于参与训练的 GPU 数；否则容易触发 mini-batch 归一化为 0 或分组异常。
- 多轮工具闭环验证不应压低 `MAX_RESPONSE_LENGTH` 或 `MAX_TOOL_RESPONSE_LENGTH`，这些参数在共享 launcher 里默认较宽。

### 3.5 数据和模型路径

04 wrapper 已经覆盖了 `TRAIN_DATA/VAL_DATA`。进入 01 后，默认不会再改掉它们：

```bash
TRAIN_DATA="${TRAIN_DATA:-${LOCAL_FLASHRAG_ROOT}/co_search_ablation.train.parquet}"
VAL_DATA="${VAL_DATA:-${LOCAL_FLASHRAG_ROOT}/co_search_ablation.eval.parquet}"
```

04 场景下有效值为：

```text
TRAIN_DATA=data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet
VAL_DATA=data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet
```

模型默认：

```bash
MODEL_PATH="${MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/llm/Qwen3-4B}"
RECALL_MODEL_PATH="${RECALL_MODEL_PATH:-${RETRIEVER_MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}}"
RANKER_BASE_MODEL_PATH="${RANKER_BASE_MODEL_PATH:-${RECALL_MODEL_PATH}}"
```

含义：

- main agent actor/ref/rollout 模型是 Qwen3-4B。
- frozen recall retriever service 使用 E5。
- trainable ranker encoder 初始也来自 E5。

### 3.6 recall service 启动和 preflight

01 里 `ensure_recall_service` 先调用 `check_recall_service`：

```bash
"${PY}" "${ASSETS_DIR}/00_check_coagentic_tool_retrieval.py" \
  --project-root "${PROJECT_ROOT}" \
  --url "${RETRIEVAL_SERVICE_URL}" \
  --query "${RETRIEVAL_PREFLIGHT_QUERY}" \
  --top-n "${RECALL_TOP_K}" \
  --top-m "${RANK_TOP_K}" \
  --expect-contains "${RETRIEVAL_PREFLIGHT_EXPECT}"
```

如果服务不可用且 `AUTO_START_RECALL_SERVICE=1`，则启动：

```bash
bash "${SCRIPT_DIR}/00_start_dense_retriever_server.sh"
```

这个服务是 frozen recall retriever，训练过程中只负责提供 recall top-N 文档，不参与梯度更新。

### 3.7 full 模式导出环境并进入共享 launcher

full 模式下，01 会导出大量环境变量给 `assets/00_run_agentic_iter_rag_verl.sh`，包括：

- `COAGENTIC_PROJECT_ROOT`
- `GPU_IDS`
- `N_GPUS_PER_NODE`
- `MODEL_PATH`
- `TRAIN_DATA/VAL_DATA`
- `TRAIN_BATCH_SIZE/VAL_BATCH_SIZE/TOTAL_STEPS/N_ROLLOUTS`
- `LORA_RANK/LORA_ALPHA`
- `ROLLOUT_DATA_DIR/VALIDATION_DATA_DIR`
- `COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL`
- `COAGENTIC_RETRIEVER_LLM_IO_JSONL`

同时构造 `COAGENTIC_EXTRA_ARGS`：

```bash
trainer.ranker_trainable=true
trainer.ranker_update_mode=contrastive
trainer.ranker_steps_per_global_step=${RANKER_STEPS_PER_GLOBAL_STEP}
trainer.disable_reranker_rollout=true
recall_retriever.model_path=${RECALL_MODEL_PATH}
recall_retriever.device=${RECALL_RETRIEVER_CONFIG_DEVICE}
recall_retriever.service_url=${RETRIEVAL_SERVICE_URL}
recall_retriever.top_k=${RECALL_TOP_K}
recall_retriever.trainable=false
ranker.model_path=${RANKER_BASE_MODEL_PATH}
ranker.device=${RANKER_CONFIG_DEVICE}
ranker.shared_encoder=true
ranker.top_k=${RANK_TOP_K}
ranker_training.batch_size=${RANKER_CONTRASTIVE_BATCH_SIZE}
ranker_training.loss.temperature=${RANKER_TEMPERATURE}
ranker_training.sample_builder.neg_per_pos=${RANKER_NEG_PER_POS}
ranker_training.sample_builder.num_groups_per_step=${RANKER_CONTRASTIVE_BATCH_SIZE}
ranker_training.signal_builder.positive_top_k=${RANKER_POSITIVE_TOP_K}
ranker_training.construction_log_jsonl=${LOG_DIR}/${RUN_NAME}.contrastive_construction.jsonl
```

这一步是 ranker 训练打开的关键：`trainer.ranker_trainable=true` 且 `trainer.ranker_update_mode=contrastive`。

最后执行：

```bash
bash "${ASSETS_DIR}/00_run_agentic_iter_rag_verl.sh" "$@" 2>&1 | tee "${TRAIN_LOG}"
```

训练完成且返回 0 后，01 会调用：

```bash
run_verl_fsdp_checkpoint_conversion "${ROOT}" "${OUT_DIR}"
coagentic_generate_training_reports "${ROOT}" || true
```

也就是保存 checkpoint 后再导出 hf/safetensors 形式，并生成 timing/metrics 报告。

## 4. 共享 launcher 阶段

文件：`scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh`

这个脚本是真正把 shell 环境变成 Hydra override 的地方。

### 4.1 默认 rollout/agent 参数

关键默认值：

```bash
N_ROLLOUTS="${N_ROLLOUTS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TURNS="${MAX_TURNS:-2}"
MAX_USER_TURNS="${MAX_USER_TURNS:-${MAX_TURNS}}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-${MAX_TURNS}}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
```

这些参数直接决定 agent 是否能完成完整闭环：

- `MAX_RESPONSE_LENGTH` 是整段多轮 response budget，包括 assistant 输出和 tool response mask，不是单轮 assistant length。
- `MAX_ASSISTANT_TURNS>=2` 才能支持“先 tool_call，再 observation，再 final answer”。
- `MAX_TOOL_RESPONSE_LENGTH` 过小会让 observation 信息不足。

### 4.2 LoRA 默认

当前：

```bash
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
```

并传给 Hydra：

```bash
actor_rollout_ref.model.lora_rank="${LORA_RANK}"
actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
```

所以默认训练全参/非 LoRA；如需 LoRA 必须显式设置 `LORA_RANK>0`。

### 4.3 tool config 生成

launcher 会写出一个运行时 tool config：

```yaml
tools:
  - class_name: verl.tools.coagentic_retriever_tool.CoAgenticRetrieverTool
    config:
      type: native
      retrieval_service_url: "<RETRIEVAL_SERVICE_URL>"
      default_top_n: <TOP_N>
      default_top_m: <TOP_M>
      ranker_enabled: true
      ranker:
        model_path: "<RANKER_BASE_MODEL_PATH>"
        encoder_path: "<RANKER_ENCODER_PATH>"
        device: "<RANKER_CONFIG_DEVICE>"
        shared_encoder: true
        top_k: <TOP_M>
```

这份 tool config 有两个用途：

1. Agent loop 初始化 `search` 工具。
2. Tool 内部加载本地 dense ranker，用于把 recall top-N 重排为 top-M 后返回给 agent。

注意：tool 里的 ranker 用于 rollout 时重排；trainer 里的 ranker worker 用于 contrastive 更新。训练时二者初始来自同一个 E5 路径，但当前 rollout tool 实例不会在同一个 global step 内自动热更新为刚训练后的 ranker；checkpoint 保存时会保存训练后的 `ranker/rank_encoder`，推理脚本会加载它。

### 4.4 Hydra override

launcher 最终执行：

```bash
exec "${PY}" "${COAGENTIC_MAIN}" \
  algorithm.use_kl_in_reward=False \
  algorithm.adv_estimator=grpo \
  data.train_files="['${TRAIN_DATA}']" \
  data.val_files="['${VAL_DATA}']" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.multi_turn.format=search_r1 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop="${COAGENTIC_AGENT_LOOP_NAME}" \
  custom_reward_function.path="${PROJECT_ROOT}/rewards/search_qa_f1_with_format_penalty.py" \
  custom_reward_function.name=search_qa_f1_penalty_compute_score \
  trainer.default_local_dir="${OUT_DIR}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  "${coagentic_extra_args[@]}"
```

重要结论：

- 训练配置最终以 Hydra override 为准。
- `coagentic_retriever_trainer.yaml` 是基线配置。
- shell 脚本覆盖了大部分运行参数。
- `COAGENTIC_EXTRA_ARGS` 再覆盖 ranker 训练相关配置。

## 5. Python 入口阶段

文件：`CoAgenticRetriever/main_coagentic_retriever.py`

Hydra 入口：

```python
@hydra.main(config_path="config", config_name="coagentic_retriever_trainer", version_base=None)
def main(config):
    run_coagentic_retriever(config)
```

`run_coagentic_retriever` 做：

1. 初始化 Ray。
2. 创建 `CoAgenticRankerTaskRunner` Ray actor。
3. 调用 `runner.run(config)`。

`CoAgenticRankerTaskRunner.run()` 做：

1. 注册 actor/rollout worker。
2. 如果需要，注册 critic/reward/ref worker。
3. `validate_config(config)`。
4. 加载 tokenizer/processor。
5. 创建 train/val dataset。
6. 创建 `CoAgenticRankerContrastiveRayTrainer`。
7. `trainer.init_workers()`。
8. `trainer.fit()`。

当前配置里：

- `algorithm.adv_estimator=grpo`
- `critic.enable=False`
- `reward_model.enable=False`
- `reward_model.use_reward_loop=True`
- `actor_rollout_ref.actor.use_kl_loss=True`

因此训练使用 actor + ref policy + reward loop，不启用 critic。

## 6. Worker 初始化

### 6.1 main agent worker/resource pool

文件：`CoAgenticRetriever/main_coagentic_base.py`

`CoAgenticRetrieverTaskRunnerBase.add_actor_rollout_worker()` 根据 config 选择：

- `AsyncActorRolloutRefWorker`，因为 `actor_rollout_ref.rollout.mode=async`
- role 是 `Role.ActorRollout`

`add_ref_policy_worker()` 会在以下条件加入 ref policy：

```python
if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
    self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
```

当前 `actor.use_kl_loss=True`，所以 ref policy 会存在，用于 actor loss 中的 KL。

resource pool 是单 global pool：

```python
resource_pool_spec = {
    "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
```

### 6.2 ranker worker

文件：`CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py`

`CoAgenticRankerContrastiveRayTrainer.init_workers()` 先调用父类初始化 actor/rollout/ref worker，再检查：

```python
trainer.ranker_trainable == true
trainer.ranker_update_mode == "contrastive"
```

满足后调用 `_init_ranker_components()`：

```python
self.ranker_wg = _LocalRankerWG(self.config)
self.ranker_wg.init_model()
self.ranker_replay_buffer = build_replay_buffer(self.config)
self.ranker_selector = build_selector(self.config)
self.ranker_signal_builder = build_signal_builder(self.config)
self.ranker_sample_builder = build_sample_builder(self.config)
self.ranker_collator = build_collator(self.config, self.ranker_wg.worker.tokenizer)
self.ranker_construction_logger = build_construction_logger(self.config)
```

`_LocalRankerWG` 包装 `LocalRankerContrastiveWorker`，实际是 `LocalE5RankerWorker`。

### 6.3 E5 ranker 初始化

文件：`CoAgenticRetriever/verl/verl/workers/ranker/e5_ranker_worker.py`

ranker worker：

- 加载 tokenizer：`AutoTokenizer.from_pretrained(ranker.model_path)`
- 加载 encoder：`AutoModel.from_pretrained(ranker.encoder_path or ranker.model_path)`
- 放到 `ranker.device`
- 设置 `encoder.train()`
- 优化器：`AdamW`
- scheduler：linear warmup
- 训练损失：InfoNCE/CrossEntropy over 1 positive + N negatives

关键训练函数：

```python
def update_ranker_contrastive(self, data: DataProto) -> DataProto:
    query_outputs = self.encoder(...)
    doc_outputs = self.encoder(...)
    scores = torch.einsum("bh,bkh->bk", query_emb, doc_emb)
    logits = scores / self.temperature
    per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
    loss.backward()
    optimizer.step()
```

checkpoint 保存：

```python
self.encoder.save_pretrained(os.path.join(path, "rank_encoder"))
self.tokenizer.save_pretrained(path)
```

因此训练产物中 ranker 的有效 encoder 在：

```text
global_step_<N>/ranker/rank_encoder/
```

## 7. 每个 global step 的训练流程

核心文件：`CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py`

`fit()` 的一个 step 顺序如下。

### 7.1 读取 batch 并生成 uid

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)
batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
gen_batch = self._get_gen_batch(batch)
gen_batch.meta_info["global_steps"] = self.global_steps
gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
```

含义：

- 每条训练样本分配 uid，用于 GRPO 分组和后续轨迹跟踪。
- `N_ROLLOUTS` 决定每个 prompt 重复采样多少条 rollout。
- `gen_batch_output` 是实际送入 agent rollout 的 batch。

### 7.2 agent rollout

```python
main_batch = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

这是整个训练最关键的行为采样阶段。它会调用注册的 `coagentic_retriever_agent` agent loop。

rollout 返回的 `main_batch` 包含：

- `prompt_ids`
- `response_ids`
- `response_mask`
- `rm_scores`
- `tool_call_details`
- `messages`
- `initial_query`
- `answers`
- `json_correct`
- `one_tool_call_per_assistant`
- reward extra info

### 7.3 用当前 trainable ranker 补充 ranker traces

rollout tool 已经返回 recall/rank 信息，但 trainer 在 ranker 训练前会再用当前 ranker worker 对 recall docs 重排一次：

```python
self._enrich_tool_calls_with_ranker(main_batch)
```

逻辑：

1. 从 `main_batch.non_tensor_batch["tool_call_details"]` 读取每次 search 的 recall docs。
2. 对每个 `sub_query` 调用 `self.ranker_wg.worker.rank_topk(...)`。
3. 写回：
   - `rank_top50_docs`
   - `rank_top5_docs`
   - `ranked_passages`
4. 记录指标：
   - `ranker/trace_enriched_tool_calls`
   - `ranker/trace_ranked_docs`

这一步让 contrastive 样本基于当前可训练 ranker 的排序结果，而不是仅依赖 rollout tool 初始排序。

### 7.4 main agent PPO/GRPO 更新任务启动

trainer 立即启动一个 Ray remote task：

```python
main_futures = process_main_agent_ppo_step.remote(...)
```

这个 remote task 内部做：

1. 从 rollout 阶段得到 `rm_scores`。
2. 重新计算 old log prob。
3. 计算 ref log prob。
4. 如启用 critic 则计算 values；当前 critic disabled。
5. 根据 reward 计算 token-level rewards。
6. 用 `adv_estimator=grpo` 计算 advantage。
7. 调用 `actor_rollout_wg.update_actor(batch)` 更新 main agent。

注意：trainer 在启动 main agent update 后，会继续执行 ranker contrastive 更新；随后 `ray.get(main_futures)` 等 main agent 更新结束。

### 7.5 ranker contrastive 更新

如果 ranker train enabled：

```python
fresh_trajectories = build_fresh_trajectories_from_dataproto(main_batch, self.global_steps)
for ranker_step_idx in range(steps_per_global):
    ranker_metrics = process_ranker_contrastive_step(...)
```

默认 `ranker_steps_per_global_step=2`，即每个 global step 更新 ranker 两次。

ranker 样本构造链路：

```text
main_batch.tool_call_details
  -> build_fresh_trajectories_from_dataproto()
  -> TopF1TrajectorySelector.select()
  -> TopKPseudoRankSignalBuilder.build()
  -> RandomNegativeRepeatSampleBuilder.build()
  -> RankerContrastiveReplayBuffer.add/sample()
  -> RankerContrastiveCollator()
  -> LocalE5RankerWorker.update_ranker_contrastive()
```

#### 7.5.1 从 DataProto 抽轨迹

`build_fresh_trajectories_from_dataproto()` 读取：

- `tool_call_details`
- `raw_prompt`
- `initial_query`
- `answers`
- `messages`
- `uid`
- `reward`

并整理为 ranker trajectory：

- `trajectory_id`
- `origin_query`
- `golden_answers`
- `score`
- `tool_calls`
  - `sub_query`
  - `recall_top50_docs`
  - `rank_top50_docs`
  - `rank_top5_docs`

#### 7.5.2 trajectory selector

`TopF1TrajectorySelector`：

- 只选 valid 且分数满足 `min_final_reward` 的轨迹。
- 按 trajectory score 降序。
- 默认最多选 `max_selected_trajectories=1` 条。
- 返回选中轨迹里的 search tool contexts。

#### 7.5.3 signal builder

`TopKPseudoRankSignalBuilder`：

- 将当前 ranker 排名前 `positive_top_k` 的 passage 标为 positive。
- 其他 passage 标为 negative。
- 默认 `positive_top_k=5`。

这是伪标签策略：ranker 使用自身 top-k 作为正例来源，而不是人工标注。

#### 7.5.4 sample builder

`RandomNegativeRepeatSampleBuilder`：

- 对每个 positive 构造一个样本。
- 每个样本包含 1 个 positive + `neg_per_pos` 个 negative。
- 默认 `neg_per_pos=15`。
- 默认每个 ranker step 构造 `num_groups_per_step=32` 个 group。
- `query_input = origin_query + " [SEP] " + sub_query`

#### 7.5.5 replay buffer

`process_ranker_contrastive_step()` 中：

```python
if ranker_step_idx == 0:
    added_samples = replay_buffer.add(contrastive_samples, source_step=global_steps)
train_samples = replay_buffer.sample(batch_size=batch_size, fresh_ratio=fresh_ratio)
```

含义：

- 每个 global step 的第 0 个 ranker step 会把 fresh contrastive samples 加入 replay buffer。
- 每次 ranker 更新从 replay buffer 采样，默认 fresh ratio 0.5。

#### 7.5.6 collator 和 InfoNCE

collator 输出：

- `query_input_ids`
- `query_attention_mask`
- `doc_input_ids`
- `doc_attention_mask`
- `positive_doc_index`
- `loss_weights`

ranker worker 计算：

```python
scores = query_emb @ doc_emb.T
logits = scores / temperature
loss = cross_entropy(logits, positive_doc_index)
```

更新指标包括：

- `ranker/loss`
- `ranker/acc@1`
- `ranker/mrr`
- `ranker/pos_score_mean`
- `ranker/neg_score_mean`
- `ranker/score_margin`
- `ranker/grad_norm`

### 7.6 等待 main agent 更新完成

ranker 更新完成后：

```python
main_results = ray.get(main_futures)
main_batch, main_metrics, main_timing_raw, main_reward_extra_infos_dict = main_results
```

然后合并：

- main agent actor metrics
- timing metrics
- reward extra info
- rollout stats

典型指标：

- `main_agent/valid_rate`
- `main_agent/score_mean`
- `main_agent/f1_mean`
- `main_agent_response_length/*`
- `main_agent_num_turns/*`
- `main_agent_actor/*`

### 7.7 rollout dump

如果配置了 `trainer.rollout_data_dir`：

```python
self._log_rollout_data(
    main_batch,
    main_reward_extra_infos_dict,
    timing_raw,
    os.path.join(rollout_data_dir, "main_agent"),
)
```

04 默认落在：

```text
log/train_logs/coAgenticRetriever/<RUN_NAME>/rollout_data/main_agent/<global_step>.jsonl
```

这个文件是验证完整闭环的关键证据。应直接检查每条 `output` 是否包含：

- `<tool_call>`
- `<tool_response>`
- `<answer>`

### 7.8 checkpoint

如果到达 save 条件：

```python
self._save_checkpoint()
```

`CoAgenticRankerContrastiveRayTrainer._save_checkpoint()` 先调用父类保存 actor，再保存 ranker：

```python
ranker_path = default_local_dir/global_step_<N>/ranker
self.ranker_wg.save_checkpoint(ranker_path)
```

产物结构大致为：

```text
checkpoints/qwen3_4b_probe/coAgenticRetriever/<RUN_NAME>/global_step_<N>/
  actor/
  hf_safetensors/actor/
  ranker/
    rank_encoder/
    tokenizer files
```

01 脚本在训练进程退出后会检查 `OUT_DIR` 下是否存在有效的 actor FSDP checkpoint。
只要存在 `global_step_<N>/actor/model_world_size_*_rank_*.pt`、`fsdp_config.json`
和 `actor/huggingface/config.json`，就会执行 FSDP checkpoint conversion，将 actor 转为
HF/safetensors 形式。转换日志写入 `${RUN_NAME}.checkpoint_conversion.log`，并追加到
`${RUN_NAME}.train.log`。01 会通过 `CHECKPOINT_VERL_ROOT=${PROJECT_ROOT}/verl`
显式指定 CoAgenticRetriever 的 VERL package；共享转换器本身仍支持 CoSearch、AgenticIterRag
或后续子项目通过各自入口传入 `CHECKPOINT_VERL_ROOT` 和 `CHECKPOINT_CONVERT_ROLES`。
如果训练进程返回 0 但 actor HF/safetensors 校验失败，01 脚本最终返回失败。

## 8. Agent loop 内部：完整 tool-call 闭环

文件：`CoAgenticRetriever/verl/verl/experimental/agent_loop/coagentic_retriever_agent_loop.py`

注册名：

```python
@register("coagentic_retriever_agent")
class CoAgenticRetrieverAgentLoop(AgentLoopBase):
```

launcher 用：

```bash
actor_rollout_ref.rollout.agent.default_agent_loop="${COAGENTIC_AGENT_LOOP_NAME}"
```

默认 `COAGENTIC_AGENT_LOOP_NAME=coagentic_retriever_agent`。

### 8.1 class 初始化

`init_class()` 做：

1. 读取 multi-turn 参数：
   - `max_user_turns`
   - `max_assistant_turns`
   - `max_tool_response_length`
2. 读取 `tool_config_path`
3. 初始化 `CoAgenticRetrieverTool`
4. 初始化 Qwen tool parser
5. 检查只允许一个 `search` tool

### 8.2 状态机

`run()` 使用状态机：

```text
PENDING
  -> GENERATING
    -> PROCESSING_TOOLS
      -> GENERATING
        -> TERMINATED
```

实际完整闭环：

1. `PENDING`：用 tokenizer chat template 组装 prompt，并带上 tool schema。
2. `GENERATING`：调用 vLLM server 生成 assistant 输出。
3. 解析 `<tool_call>...</tool_call>`。
4. 如果刚好一个 tool call，则截断 tool call 之后的多余 token，进入 `PROCESSING_TOOLS`。
5. `PROCESSING_TOOLS`：执行 `search` tool，生成 tool response。
6. 将 tool response 用 chat template 编码后追加到 prompt，`response_mask` 对 tool response 置 0。
7. 回到 `GENERATING` 继续生成最终 `<answer>...</answer>`。
8. 如果检测到 `<answer>`，终止并返回 `AgentLoopOutput`。

关键截断逻辑：

```python
self._truncate_after_first_tool_call(agent_data)
```

这用于避免模型在 tool call 后继续输出 premature answer，确保 search tool 仍会执行。

### 8.3 response length 的含义

Agent loop 中有：

```python
if len(agent_data.response_mask) >= self.response_length:
    return AgentState.TERMINATED
```

以及在追加 tool response 前：

```python
if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
    return AgentState.TERMINATED
```

因此 `MAX_RESPONSE_LENGTH` 限制的是整段多轮轨迹预算，包含：

- assistant 首轮 tool-call 输出
- tool response 追加部分
- assistant 最终 answer 输出

这就是为什么 smoke 验证不能把 `MAX_RESPONSE_LENGTH` 压得过低。过低时会出现已经 tool-call，但 observation 或 final answer 被截断的情况。

## 9. Search tool 内部

文件：`CoAgenticRetriever/verl/verl/tools/coagentic_retriever_tool.py`

`CoAgenticRetrieverTool.execute()` 做：

1. 读取模型生成的 `query`。
2. 调用 frozen recall retriever service：

```python
recall_docs = await self._call_retrieval_api(query, top_n)
```

3. 规范化 recall docs，写入：

```python
metrics["recall_top50_docs"] = recall_docs[:top_n]
```

4. 用 local dense ranker 对 recall docs 重排：

```python
ranked_docs = self.ranker.rank_topk(query=query, docs=recall_docs, top_k=len(recall_docs))
```

5. 只把 top-M 返回给 agent：

```python
final_docs = ranked_docs[:agent_top_k]
response_text = format_tool_response(final_docs)
```

6. 计算 tool reward：

```python
compute_average_hit_at_ks(answers=answers, documents=final_docs, hit_cutoffs=hit_cutoffs)
```

7. 返回：

```python
AgentToolResponse(text=response_text), reward, metrics
```

tool 的 metrics 会被 agent loop 整理为 `tool_call_details`，用于 ranker contrastive 训练。

## 10. Reward 逻辑

文件：`CoAgenticRetriever/rewards/search_qa_f1_with_format_penalty.py`

reward 函数：

```python
search_qa_f1_penalty_compute_score(...)
```

核心逻辑：

1. 从最终输出中抽取最后一个 `<answer>...</answer>`。
2. 对 ground truth target 计算 F1。
3. 检查格式。
4. 格式正确则 reward = F1。
5. 格式错误则 reward = `format_penalty`，默认负值。

格式要求：

- 每个 assistant block 必须有一个 reasoning block：
  - `<think>...</think>` 或 `<reason>...</reason>`
- 非最终 assistant turn 必须包含 `<tool_call>...</tool_call>`
- 最终 assistant turn 必须包含 `<answer>...</answer>`

这解释了为什么训练验证必须检查完整闭环：只有 `tool_call -> tool_response -> answer` 都出现，reward/valid 才能代表真实训练路径。

## 11. 训练完成后的产物

04 full 训练完成后，主要产物包括：

```text
log/train_logs/coAgenticRetriever/<RUN_NAME>/
  <RUN_NAME>.env
  <RUN_NAME>.train.log
  <RUN_NAME>.metrics.jsonl
  <RUN_NAME>.search_timing.jsonl
  <RUN_NAME>.contrastive_construction.jsonl
  <RUN_NAME>.llm_io.jsonl
  rollout_data/main_agent/<step>.jsonl
  <RUN_NAME>.timing_report.step<N>.md
  <RUN_NAME>.detailed_metrics_report.step<N>.md

checkpoints/qwen3_4b_probe/coAgenticRetriever/<RUN_NAME>/
  global_step_<N>/
    actor/
    hf_safetensors/actor/
    ranker/
      rank_encoder/
```

最重要的对齐文件：

- `.env`：确认本次训练实际参数。
- `.train.log`：看完整控制台输出、worker 初始化、metric 打印。
- `.metrics.jsonl`：结构化指标。
- `rollout_data/main_agent/<step>.jsonl`：逐条轨迹输出，验证闭环。
- `.contrastive_construction.jsonl`：ranker contrastive 样本构造细节。
- `global_step_<N>/ranker/rank_encoder/`：训练后的 ranker encoder。
- `global_step_<N>/hf_safetensors/actor/`：转换后的 actor。

## 12. 已验证的 04 smoke 证据

此前 04 fullchain smoke 运行：

```text
RUN_NAME=smoke_coagentic_ablation_fullchain_260610a
```

rollout dump：

```text
log/train_logs/coAgenticRetriever/260610-233829-smoke_coagentic_ablation_fullchain_260610a/rollout_data/main_agent/1.jsonl
```

文件级校验结论：

```text
lines=4
sample=1 tool_call=1 tool_response=1 answer=1
sample=2 tool_call=1 tool_response=1 answer=1
sample=3 tool_call=1 tool_response=1 answer=1
sample=4 tool_call=1 tool_response=1 answer=1
FULL_CHAIN_OK=True
```

训练日志中也出现：

```text
main_agent/valid_rate: 1.0
main_agent_response_length/clip_ratio: 0.0
main_agent_num_turns/min: 4
main_agent_num_turns/max: 4
main_agent_num_turns/mean: 4.0
ranker/trace_enriched_tool_calls: 4
ranker/trace_ranked_docs: 40
```

这说明该 smoke 不只是“程序没有报错”，而是证明了 4 条样本都完成了：

```text
assistant tool_call
  -> search tool 执行
  -> tool_response observation 回灌
  -> assistant final answer
  -> reward/ranker/actor/checkpoint 链路
```

## 13. 对齐时重点看哪些参数

建议每次对齐先看 `<RUN_NAME>.env` 和 launcher 参数，重点确认：

```text
TRAIN_DATA
VAL_DATA
MODEL_PATH
TRAIN_BATCH_SIZE
ACTOR_BATCH_SIZE
N_ROLLOUTS
TOTAL_STEPS
LORA_RANK
MAX_RESPONSE_LENGTH
MAX_TOOL_RESPONSE_LENGTH
MAX_ASSISTANT_TURNS
RECALL_TOP_K
RANK_TOP_K
RANKER_CONTRASTIVE_BATCH_SIZE
RANKER_STEPS_PER_GLOBAL_STEP
RANKER_NEG_PER_POS
RANKER_POSITIVE_TOP_K
RANKER_TEMPERATURE
RANKER_CONFIG_DEVICE
RANKER_DEVICE_TRAIN
```

关键约束：

- 默认 `LORA_RANK=0`，不会再隐式 LoRA 训练。
- GRPO batch size 不应低于参与 actor 训练 GPU 数。
- `MAX_ASSISTANT_TURNS` 至少为 2，否则很难覆盖 tool call 后继续 answer 的流程。
- `MAX_RESPONSE_LENGTH` 是整段多轮轨迹预算，不能为了 smoke 过度压缩。
- `RANK_TOP_K` 决定 agent 看到的文档数。
- `RECALL_TOP_K` 决定 ranker 的候选池。
- `RANKER_STEPS_PER_GLOBAL_STEP` 决定每个 actor step 中 ranker 更新次数。

## 14. 常见问题定位

### 14.1 只有 tool_call，没有 tool_response 或 answer

优先检查：

- `MAX_RESPONSE_LENGTH` 是否过小。
- `MAX_TOOL_RESPONSE_LENGTH` 是否过小。
- `MAX_ASSISTANT_TURNS` 是否小于 2。
- `rollout_data/main_agent/<step>.jsonl` 中 output 是否在 tool call 后截断。

### 14.2 reward valid_rate 低

优先检查：

- assistant block 是否满足 `<think>/<reason>` + `<tool_call>` 或 `<answer>`。
- 是否只有 `<tool_call>` 没有最终 `<answer>`。
- reward 文件中的 `validate_response_structure()` 对标签数量和顺序要求严格。

### 14.3 ranker 没有更新

优先检查：

- `trainer.ranker_trainable=true`
- `trainer.ranker_update_mode=contrastive`
- `ranker/selected_contexts`
- `ranker/fresh_samples`
- `ranker/train_samples`
- `.contrastive_construction.jsonl`
- `tool_call_details` 是否存在于 rollout dump。

### 14.4 推理没有用训练后的 ranker

训练时 ranker checkpoint 在：

```text
global_step_<N>/ranker/rank_encoder/
```

推理脚本必须把 `ranker.encoder_path` 指到这个目录。否则 tool 会回退到 base E5 或错误路径。

### 14.5 checkpoint 里是否 LoRA

当前默认：

```text
LORA_RANK=0
```

因此新训练默认不是 LoRA。若看到 LoRA adapter，说明运行时显式传了 `LORA_RANK>0`，或使用的是旧 checkpoint。

## 15. 一句话流程总结

04 训练脚本只切换到 `coAgenticRetriever/albation_1` 数据集，然后复用 01 的 full 训练链路：先启动 frozen recall retriever service，再由共享 launcher 把参数注入 Hydra，Python 端创建 Ray/VERL trainer；每个 global step 中 main agent 通过多轮 agent loop 完成 `tool_call -> search -> tool_response -> answer` 的 rollout，reward 函数基于最终 answer 和格式给 main agent 训练信号，同时 trainer 从 rollout 的 search traces 构造 ranker contrastive 样本更新 E5 ranker，最后保存 actor 和 ranker checkpoint 并生成日志报告。

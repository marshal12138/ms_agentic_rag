# 08 Canonical Mix-Signal 配置构造流程

本文说明 canonical `mix_signal_fix` 训练入口从 task shell 到最终 Python 训练代码的配置构造流程：哪些文件参与、每层配置的职责、Hydra override 的顺序，以及运行时审计文件如何生成。

当前说明对应 canonical 入口：

```text
tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

## 1. 总体链路

整体流程：

```text
task shell
  -> v2 launcher
  -> runtime override YAML + hydra_args.txt
  -> asset runner
  -> main_coagentic_retriever.py
  -> Hydra 组合最终 config
  -> CoAgenticRankerContrastiveRayTrainer
```

配置优先级从低到高：

```text
Trainer Hydra 主配置 + config groups
< launcher minimal defaults
< reusable overlay YAML
< task overlay YAML
< runtime env override YAML
< canonical CLI temporary overrides
```

同一个 key 后面覆盖前面。例如 reusable overlay 中：

```text
ranker_training.async_ranker_training.sample_builder_request_batch=1
```

task overlay 中：

```text
ranker_training.async_ranker_training.sample_builder_request_batch=3
```

最终生效的是 `3`。

## 2. Task 入口脚本

入口文件：

```text
tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

这个脚本只声明实验身份、资源和配置选择：

```bash
export EXP_NAME="${EXP_NAME:-CAR_async_ranker_training_ds_flash_mix_signal_b3_v1_select_all}"
export GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"

export AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3}"
export RANK_GPU_ID="${RANK_GPU_ID:-4}"
export RECALL_GPU_ID="${RECALL_GPU_ID:-5}"
export LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"
```

然后调用 v2 launcher：

```bash
bash "${ROOT}/scripts/coagenticRetriever_v2/01_train_launcher.sh" \
  --main_run_config=coAgenticRetriever_main \
  --trainer_main_hydra_config=coagentic_retriever_trainer \
  --DATA_CONFIG=co_search_ablation \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=cosearch_async_qwen3_4b \
  --RANKER_BASE_CONFIG=ranker_contrastive \
  --ASYNC_RANKER_TRAINING_BASE_CONFIG=async_ranker_training \
  --OVERLAY_YAML=scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_deepseek_flash_rank50_select_all.yaml \
  --OVERLAY_YAML=tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml \
  --LLM_JUDGE_SERVICE_CONFIG=CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

这个入口不再使用：

```text
COAGENTIC_EXTRA_ARGS
COAGENTIC_DEFAULT_EXTRA_ARGS
HYDRA_OVERRIDE_YAMLS
SAMPLE_BUILDER_REQUEST_BATCH
"$@" 透传
```

这保证 canonical 训练参数都能进入审计文件。

## 3. Config Group 选择

launcher 先接收 Trainer Hydra 主配置参数：

```text
--trainer_main_hydra_config
```

当前 task 显式传入：

```bash
--main_run_config=coAgenticRetriever_main \
  --trainer_main_hydra_config=coagentic_retriever_trainer
```

它会被校验为顶层主配置文件：

```text
CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
```

最终写入 `hydra_args.txt` 的 Trainer Hydra 主配置参数是：

```text
--config-name=coagentic_retriever_trainer
```

随后 launcher 接收五个 config group 参数：

```text
--DATA_CONFIG
--MODEL_CONFIG
--ROLLOUT_CONFIG
--RANKER_BASE_CONFIG
--ASYNC_RANKER_TRAINING_BASE_CONFIG
```

它们可以使用短名：

```bash
--MODEL_CONFIG=qwen3_4b
```

也可以使用对应 group 目录内的 YAML 路径：

```bash
--MODEL_CONFIG=CoAgenticRetriever/config/model/qwen3_4b.yaml
```

launcher 会把路径归一化回 Hydra group 短名。最终传给 Hydra 的 group selection 是：

```text
data=co_search_ablation
model@actor_rollout_ref.model=qwen3_4b
rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b
experimental/ranker_base@_global_=ranker_contrastive
experimental/async_ranker_training_base@_global_=async_ranker_training
```

对应文件：

```text
CoAgenticRetriever/config/data/co_search_ablation.yaml
CoAgenticRetriever/config/model/qwen3_4b.yaml
CoAgenticRetriever/config/rollout/cosearch_async_qwen3_4b.yaml
CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml
CoAgenticRetriever/config/experimental/async_ranker_training_base/async_ranker_training.yaml
```

五者职责：

- `data/co_search_ablation.yaml`：训练/验证 parquet、样本数、batch size、prompt/response 长度、chat template thinking 开关。
- `model/qwen3_4b.yaml`：Qwen3-4B 模型路径、remote code、remove padding、LoRA 默认值、attention override。
- `rollout/cosearch_async_qwen3_4b.yaml`：vLLM async rollout、上下文长度、采样条数、multi-turn/tool budget、AgentLoop 配置路径。
- `experimental/ranker_base/ranker_contrastive.yaml`：ranker/recall/contrastive 训练基础结构和值。
- `experimental/async_ranker_training_base/async_ranker_training.yaml`：异步 ranker training基础结构和值。

## 4. Trainer Hydra 主配置

Python 训练入口：

```text
CoAgenticRetriever/main_coagentic_retriever.py
```

入口使用 Hydra：

```python
@hydra.main(config_path="config", config_name="coagentic_retriever_trainer")
```

主配置文件：

```text
CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
```

它的 `defaults` 先加载基础配置：

```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - data@data: legacy_data
  - ref@actor_rollout_ref.ref: dp_ref
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  - critic@critic: dp_critic
  - reward_model@reward_model: dp_reward_model
  - algorithm@algorithm.rollout_correction: rollout_correction
  - experimental/ranker_base@_global_: ranker_contrastive
  - experimental/async_ranker_training_base@_global_: async_ranker_training
  - _self_
```

canonical group selection 会替换默认的 `legacy_data`、`hf_model`、`rollout`，并显式记录 ranker base 与 async ranker training base 的选择。

## 5. Overlay YAML

canonical 入口传入两个 overlay，按顺序生效。

### 5.1 Reusable Overlay

文件：

```text
scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_deepseek_flash_rank50_select_all.yaml
```

作用：

- 启用 `ranker_training.signal_source=async_ranker_training`
- 启用 async ranker training
- 配置 DeepSeek-Flash LLM judge stage
- 配置 rank50 judge schema
- 配置 `select_all` trajectory selector
- 配置 async sample builder、logging、shared inference ranker

这层是可复用策略，不放 run name、GPU、checkpoint 路径。

### 5.2 Task Overlay

文件：

```text
tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml
```

作用：

- 设置 canonical b3 实验差异：

```yaml
ranker_training:
  async_ranker_training:
    sample_builder_request_batch: 3
```

- 固定该实验的 FSDP offload 行为。
- 设置 `coagentic_retriever.agent.inject_tool_schema=false`。
- 声明默认资源布局：

```yaml
resources:
  agent_gpu_ids: "0,1,2,3"
  rank_gpu_id: "4"
  recall_gpu_id: "5"
  llm_judge_gpu_ids: "6,7"
```

这些资源默认值会被 launcher 读取；如果外部环境变量显式设置 GPU，则环境变量优先。

## 6. YAML Overlay 如何变成 Hydra 参数

转换工具：

```text
src/hydra_overrides/yaml_to_dotlist.py
src/hydra_overrides/hydra_overrides.sh
```

launcher 调用 `hydra_yaml_overrides_to_array`，把 overlay YAML 转成 dotlist。

示例：

```yaml
ranker_training:
  async_ranker_training:
    sample_builder_request_batch: 3
```

转成：

```text
++ranker_training.async_ranker_training.sample_builder_request_batch=3
```

注意：这个工具拒绝 top-level `defaults:`，所以 overlay 只能表达值覆盖，不能做 Hydra composition。

## 7. Runtime Override

launcher 会生成运行时覆盖文件：

```text
${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml
```

它由当前 shell 环境和 run identity 生成，包含：

```yaml
trainer:
  experiment_name: "${EXP_NAME}"
  default_local_dir: "${OUT_DIR}"
  n_gpus_per_node: ${AGENT_N_GPUS_PER_NODE}
  rollout_data_dir: "${ROLLOUT_DATA_DIR}"
  validation_data_dir: "${VALIDATION_DATA_DIR}"

recall_retriever:
  device: "cuda:${RECALL_GPU_ID}"
  service_url: "${RETRIEVAL_SERVICE_URL}"

ranker:
  device: "cuda:${RANK_GPU_ID}"

ranker_training:
  construction_log_jsonl: "${LOG_DIR}/${RUN_NAME}.contrastive_construction.jsonl"

resources:
  agent_gpu_ids: "${AGENT_GPU_IDS}"
  rank_gpu_id: "${RANK_GPU_ID}"
  recall_gpu_id: "${RECALL_GPU_ID}"
  llm_judge_gpu_ids: "${LLM_JUDGE_GPU_IDS}"
```

runtime override 排在所有 overlay 之后，因此 GPU/device、run path、日志路径最终以 runtime 为准。

task 脚本末尾可以追加临时 Hydra 覆盖，例如：

```bash
--actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
```

launcher 会把可选前导 `--` 去掉，归一化为：

```text
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
```

这类临时覆盖排在 `hydra_args.txt` 最后，因此优先级最高。

## 8. 最终 hydra_args.txt

launcher 生成主配置审计文件和最终参数审计文件：

```text
${LOG_DIR}/${RUN_NAME}.trainer_main_hydra_config.txt
${LOG_DIR}/${RUN_NAME}.hydra_cli_overrides.txt
${LOG_DIR}/${RUN_NAME}.hydra_args.txt
```

`.trainer_main_hydra_config.txt` 记录实际使用的 Trainer Hydra 主配置短名：

```text
coagentic_retriever_trainer
```

`.hydra_args.txt` 记录真实传给 Python 的参数序列：

```text
--config-name=coagentic_retriever_trainer
data=co_search_ablation
model@actor_rollout_ref.model=qwen3_4b
rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b
...
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
```

写入顺序：

```text
1. Hydra main config argument
2. Hydra group selection
3. launcher minimal defaults
4. reusable overlay dotlist
5. task overlay dotlist
6. runtime override dotlist
7. canonical CLI temporary overrides
```

asset runner 在 canonical 模式下只读取这个文件：

```bash
mapfile -t canonical_hydra_args < "${CANONICAL_HYDRA_ARGS_FILE}"
exec "${PY}" "${COAGENTIC_MAIN}" "${canonical_hydra_args[@]}"
```

因此 canonical 路径不会再插入 legacy 的 `COAGENTIC_EXTRA_ARGS` 或 CLI passthrough。

## 9. Asset Runner 分流

asset runner 文件：

```text
scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh
```

canonical 模式：

```text
CANONICAL_CONFIG_MODE=1
```

行为：

- 禁止 `"$@"` 透传。
- 要求 `CANONICAL_HYDRA_ARGS_FILE` 存在。
- 逐行读取最终 Hydra 参数。
- 直接执行 `main_coagentic_retriever.py`。

legacy 模式：

- 保留 `HYDRA_OVERRIDE_YAMLS`
- 保留 `RANKER_STRATEGY_YAML`
- 保留 `ASYNC_RANKER_TRAINING_YAML`
- 保留 `COAGENTIC_DEFAULT_EXTRA_ARGS`
- 保留 `COAGENTIC_EXTRA_ARGS`
- 保留 `"$@"`

这保证 canonical 改造不破坏旧任务。

## 10. 服务配置与训练配置的边界

LLM judge 服务配置：

```text
CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

这个文件只用于：

- dry-run 校验 judge 服务启动参数
- 自动启动 LLM judge vLLM 服务
- 设置 judge 模型、GPU、端口、max model len 等服务参数

它不是训练 Hydra config 的一部分。

Recall retriever 服务由 launcher 控制生命周期，训练 Hydra 中只记录：

```yaml
recall_retriever:
  service_url: ...
  device: ...
```

## 11. Tool Config 与 AgentLoop Config

搜索工具配置：

```text
CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
```

作用：

- 定义 search tool 类。
- 配置 retrieval service URL。
- 配置 topN/topM、format penalty、ranker actor、ranker device 等。
- launcher 会读取其中部分静态值，例如 retrieval URL、topN/topM、format penalty。

AgentLoop 配置：

```text
CoAgenticRetriever/config/coagentic_retriever_agent_loop_config.yaml
```

作用：

- 注册 `coagentic_retriever_agent`。
- 指向 Python 类：

```text
verl.experimental.agent_loop.coagentic_retriever_agent_loop.CoAgenticRetrieverAgentLoop
```

rollout 配置中的：

```yaml
agent:
  default_agent_loop: coagentic_retriever_agent
  agent_loop_config_path: ...
```

会让 AgentLoop manager 加载这个文件，并在 rollout 时实例化对应 agent loop。

## 12. 最终进入训练代码

最终执行：

```text
CoAgenticRetriever/main_coagentic_retriever.py
```

Hydra 完成配置组合后，`CoAgenticRankerTaskRunner` 拿到完整 `config`。

训练代码使用该 config：

- 初始化 Ray。
- 创建 tokenizer/processor。
- 创建 train/val dataset。
- 创建 actor rollout worker。
- 创建 ranker worker。
- 创建 recall/ranker/async ranker training 配置。
- 启动 `CoAgenticRankerContrastiveRayTrainer`。

训练代码看到的是已经组合完成的配置对象，而不是 shell 变量。

## 13. 审计产物

dry-run 和正式运行都会写出：

```text
${LOG_DIR}/${RUN_NAME}.env
${LOG_DIR}/${RUN_NAME}.trainer_main_hydra_config.txt
${LOG_DIR}/${RUN_NAME}.hydra_groups.txt
${LOG_DIR}/${RUN_NAME}.hydra_cli_overrides.txt
${LOG_DIR}/${RUN_NAME}.overlay_yamls.txt
${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml
${LOG_DIR}/${RUN_NAME}.hydra_args.txt
```

用途：

- `.env`：记录运行环境、run identity、资源、服务开关。
- `.trainer_main_hydra_config.txt`：记录实际 Trainer Hydra 主配置。
- `.hydra_groups.txt`：记录实际 Hydra group selection。
- `.hydra_cli_overrides.txt`：记录 task 中追加的临时 Hydra 覆盖，已去掉可选前导 `--`。
- `.overlay_yamls.txt`：记录 overlay YAML 顺序。
- `.runtime_env_overrides.yaml`：记录最终 runtime 覆盖。
- `.hydra_args.txt`：记录真实传给 Python 主程序的 Hydra 参数顺序。

排查配置问题时，优先看 `.hydra_args.txt` 和 `main_coagentic_retriever.py --cfg job` 输出。

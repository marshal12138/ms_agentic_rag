# CoAgenticRetriever Train Task Records

更新时间：2026-06-18

这个文件用于记录 `tasks/train_tasks/` 下每个训练脚本对应的实验意图、关键变量和与其它实验的差异。后续新增脚本时，建议按同样格式追加。

## 通用入口与覆盖顺序

当前 task 脚本最终都会调用：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

底层训练入口会再启动 CoAgenticRetriever/VERL full training，默认启用：

- `trainer.ranker_trainable=true`
- `trainer.ranker_update_mode=contrastive`
- `trainer.ranker_steps_per_global_step=2`
- `ranker.model_path=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- `recall_retriever.model_path=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`

Hydra 覆盖顺序需要特别注意：

```text
base config
< COAGENTIC_DEFAULT_EXTRA_ARGS
< HYDRA_OVERRIDE_YAMLS / RANKER_STRATEGY_YAML / ASYNC_LABELING_YAML
< COAGENTIC_EXTRA_ARGS
< 脚本命令行最后传入的 "$@"
```

因此：

- `ASYNC_LABELING_YAML` 会覆盖 base 里的 async_labeling 默认配置。
- `COAGENTIC_EXTRA_ARGS` 会覆盖 YAML 中的同路径参数。
- `train_CAR_async_labeling_ds_flash_mix_signal.sh` 就是通过 `COAGENTIC_EXTRA_ARGS` 把 YAML 里的 `sample_builder_request_batch: 1` 覆盖成 3。

## 当前任务脚本总览

| 脚本 | 默认 EXP_NAME | 实验类型 | 核心差异 |
| --- | --- | --- | --- |
| `train_CAR_naive_acce.sh` | `CAR_mem_speed_no_think_v1` | 原始/非 async labeling 的加速训练配置 | 不启用 `ENABLE_ASYNC_LABELING`，不启动 LLM judge；主要验证 no-thinking、并发和显存配置下的基础训练吞吐 |
| `train_CAR_async_labeling_ds_flash.sh` | `CAR_async_labeling_ds_flash_v1` | DeepSeek-V4-Flash async labeling | 启用 LLM-as-judge 异步标注，每次 ranker update 默认消费 1 条 completed judge signal |
| `train_CAR_async_labeling_ds_flash_mix_signal.sh` | `CAR_async_labeling_ds_flash_mix_signal_b3_v1` | DeepSeek-V4-Flash async labeling + 多 signal 混合 | 在 async labeling 基础上，每次 ranker update 默认消费 3 条 completed judge signals，并等待目标 GPU 空闲后启动 |

## `train_CAR_naive_acce.sh`

路径：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_naive_acce.sh
```

实验目的：

- 作为当前 CoAgenticRetriever full training 的基础加速版本。
- 关闭 Qwen3 thinking 模式，提升 rollout/训练速度并保持 prompt 格式稳定。
- 不启用 async labeling，因此不会启动 DeepSeek-V4-Flash judge 服务。

关键默认参数：

```bash
EXP_NAME=CAR_mem_speed_no_think_v1
GPU_MEMORY_UTILIZATION=0.55
MAX_NUM_SEQS=16
AGENT_WORKERS=4
TOOL_MAX_CONCURRENT_PER_WORKER=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
ACTOR_MICRO_BATCH_SIZE_PER_GPU=4
```

关键 Hydra 覆盖：

```bash
actor_rollout_ref.rollout.max_num_batched_tokens=32768
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.ref.fsdp_config.param_offload=False
++data.apply_chat_template_kwargs.enable_thinking=False
```

与 async labeling 脚本的主要差异：

- 未设置 `ENABLE_ASYNC_LABELING=1`，底层默认是 0。
- 未设置 `ASYNC_LABELING_YAML`。
- 未设置 `AUTO_START_LLM_JUDGE` / `AUTO_STOP_LLM_JUDGE`。
- ranker 仍走底层默认 contrastive update，但训练信号不是 DeepSeek judge 异步标注信号。

适合用途：

- 测基础训练速度和显存。
- 做无 LLM judge 的对照实验。
- 排查 actor rollout、recall/ranker 服务、VERL 训练链路本身的问题。

## `train_CAR_async_labeling_ds_flash.sh`

路径：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
```

实验目的：

- 启用 async labeling 框架。
- 使用 DeepSeek-V4-Flash 作为 LLM-as-judge，对 ranker 排序后的 top50 chunks 产生排序/伪标签信号。
- 用 judge 结果构造 ranker contrastive samples，而不是只依赖原始 rollout 伪标签。

关键默认参数：

```bash
EXP_NAME=CAR_async_labeling_ds_flash_v1
ENABLE_ASYNC_LABELING=1
ASYNC_LABELING_YAML=scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
AUTO_START_LLM_JUDGE=1
AUTO_STOP_LLM_JUDGE=1
LLM_JUDGE_ENDPOINT=http://127.0.0.1:8067/v1/chat/completions
LLM_JUDGE_PREFLIGHT=1
```

默认资源分配：

```bash
AGENT_GPU_IDS=0,1,2,3
RANK_GPU_ID=4
RECALL_GPU_ID=5
LLM judge GPU=6,7
```

检索与可见文档：

```bash
TOP_N=50
TOP_M=5
```

含义：

- `TOP_N=50`：检索工具保留 dense ranker 排序后的 top50，供 judge 标注使用。
- `TOP_M=5`：agent LLM 最终可见 top5 文档。
- judge 标注 top50 和 agent 可见 top5 是两件事，不能混为一谈。

对应 async YAML 的关键特征：

```yaml
ranker_training:
  signal_source: async_labeling
  async_labeling:
    enable: true
    max_sub_query: 10
    sample_builder_request_batch: 1
    num_workers: 4
    stages:
      - type: llm_as_judge
        model: DeepSeek-V4-Flash
        score_schema: ranked_ids_top50
        max_docs_per_request: 50
    sample_builder:
      type: random_negative_repeat_from_signal
      num_groups_per_step: 32
      neg_per_pos: 15
      strategy_kwargs:
        signal_builder_type: llm_judge_topk
        positive_top_k: 5
        label_source: llm_judge_top5
```

当前实验假设：

- DeepSeek-V4-Flash judge 对 top50 候选的排序信号，比单纯的 dense/ranker 自身 top-k 伪标签更能提供训练监督。
- 每次 ranker update 消费 1 条 completed judge signal，信号更“新”，但单次更新的 label 来源更窄。

适合用途：

- async labeling 主线实验。
- 评估 LLM judge 信号是否改善 ranker。
- 和 `train_CAR_naive_acce.sh` 做有无 judge label 的对照。

## `train_CAR_async_labeling_ds_flash_mix_signal.sh`

路径：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash_mix_signal.sh
```

实验目的：

- 在 DeepSeek-V4-Flash async labeling 基础上，混合多条 completed judge signals 再构造 ranker contrastive samples。
- 降低单条 sub-query / 单条轨迹 judge label 的偶然性。
- 让每次 ranker update 的训练信号覆盖更多 query/trajectory。

与 `train_CAR_async_labeling_ds_flash.sh` 共有的默认配置：

```bash
ENABLE_ASYNC_LABELING=1
ASYNC_LABELING_YAML=scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
AUTO_START_LLM_JUDGE=1
AUTO_STOP_LLM_JUDGE=1
LLM_JUDGE_ENDPOINT=http://127.0.0.1:8067/v1/chat/completions
TOP_N=50
TOP_M=5
AGENT_GPU_IDS=0,1,2,3
RANK_GPU_ID=4
RECALL_GPU_ID=5
```

本脚本特有参数：

```bash
EXP_NAME=CAR_async_labeling_ds_flash_mix_signal_b3_v1
SAMPLE_BUILDER_REQUEST_BATCH=3
WAIT_FOR_GPUS=${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},6,7
WAIT_FOR_GPU_RELEASE=1
WAIT_FOR_GPU_INTERVAL_SECONDS=600
```

关键覆盖：

```bash
ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}
```

注意：

- `async_labeling_deepseek_flash.yaml` 中默认 `sample_builder_request_batch: 1`。
- 本脚本会在 `COAGENTIC_EXTRA_ARGS` 末尾追加 `sample_builder_request_batch=3`，因此最终生效值是 3。
- `sample_builder_request_batch=3` 表示每次 ranker async update 从 completed buffer 中取 3 条 judge signals。
- 它不表示每次产生 3 倍 contrastive groups；当前 sample_builder 仍默认输出 `num_groups_per_step=32` 组。

GPU 等待逻辑：

- 默认等待 agent/ranker/recall/judge 所需 GPU 全部空闲后再启动。
- 如果只是临时 smoke 或已经确认资源可用，可以显式设置：

```bash
WAIT_FOR_GPU_RELEASE=0 bash tasks/train_tasks/train_CAR_async_labeling_ds_flash_mix_signal.sh
```

当前实验假设：

- 相比每次只用 1 条 judge signal，混合 3 条 signals 可以提升 ranker update 的样本多样性。
- 可能降低 label 噪声，但也可能引入更旧的 completed signals，需观察 `async_labeler/completed_buffer_size`、`ranker/async_consumed_signals`、`ranker/loss`、`ranker/acc@1`、下游 eval 指标。

适合用途：

- 与 `train_CAR_async_labeling_ds_flash.sh` 做 `sample_builder_request_batch=1 vs 3` 对照。
- 验证多 signal 混合是否改善 ranker 训练稳定性和最终 QA 指标。

## 相关策略 YAML

### `async_labeling_deepseek_flash.yaml`

路径：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
```

作用：

- async labeling 主配置。
- 设置 `ranker_training.signal_source=async_labeling`。
- 配置 DeepSeek-V4-Flash judge endpoint、prompt、top50 打分格式、sample_builder 和日志。

关键字段：

```yaml
ranker_training.signal_source: async_labeling
ranker_training.async_labeling.enable: true
ranker_training.async_labeling.max_sub_query: 10
ranker_training.async_labeling.sample_builder_request_batch: 1
ranker_training.async_labeling.trajectory_selector.type: best_and_worst_f1
ranker_training.async_labeling.trajectory_selector.top_k: 1
ranker_training.async_labeling.trajectory_selector.bottom_n: 2
ranker_training.async_labeling.stages[0].type: llm_as_judge
ranker_training.async_labeling.stages[0].model: DeepSeek-V4-Flash
ranker_training.async_labeling.stages[0].score_schema: ranked_ids_top50
ranker_training.async_labeling.sample_builder.type: random_negative_repeat_from_signal
ranker_training.async_labeling.sample_builder.strategy_kwargs.positive_top_k: 5
```

覆盖关系：

- 会覆盖 base async_labeling 配置。
- 会被 `COAGENTIC_EXTRA_ARGS` 里的同路径参数覆盖。
- 在 `mix_signal` 实验中，`sample_builder_request_batch` 最终从 1 被覆盖为 3。

### `ranker_contrastive_new_sampling.yaml`

路径：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml
```

作用：

- 可选 ranker strategy override。
- 当前三个 task 脚本默认没有显式设置 `RANKER_STRATEGY_YAML`，因此除非外部手动传入，否则这个 YAML 不会自动生效。

关键字段：

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
    min_final_reward: 0.0
```

适合用途：

- 对非 async labeling 的 ranker pseudo label 构造路径做采样策略改动。
- 与 async labeling 中的 `ranker_training.async_labeling.trajectory_selector` 区分：二者路径不同，一个是普通 ranker training selector，一个是 async labeling 内部 selector。

## 推荐记录格式

后续每次新实验可以追加：

```markdown
## YYYY-MM-DD / RUN_STAMP - 实验名

- 启动脚本：
- EXP_NAME：
- 代码版本/重要改动：
- 关键假设：
- 相比上一个实验的唯一变量：
- 关键覆盖参数：
- 预期观察指标：
- 实际输出目录：
- Eval 结果：
- 结论：
```

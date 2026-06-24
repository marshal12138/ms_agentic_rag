# Mix-Signal Fix 训练任务脚本配置治理计划

日期：2026-06-23

更新：2026-06-24

目标脚本：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix.sh
```

本计划只治理这个 canonical mix-signal 训练入口，不治理：

- `train_CAR_async_labeling_ds_flash.sh`
- `train_CAR_async_labeling_ds_flash_mix_signal.sh`
- `train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh`
- `train_CAR_async_labeling_ds_flash_mix_signal_fix_exp02.sh`
- `train_CAR_naive_acce.sh`

## 1. 核心判断

前一版计划把 `tasks/train_tasks/coAgenticRetriever/configs/train_mix_signal_ds_flash_b3.yaml` 设计成“大而全的主 recipe YAML”。这个方向需要修正。

原因是当前训练入口已经有 Hydra 原生配置体系：

```python
@hydra.main(config_path="config", config_name="coagentic_retriever_trainer", version_base=None)
```

主配置是：

```bash
CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
```

它通过 `defaults` 组合了 VERL/CoAgenticRetriever 的基础 config groups：

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
  - ranker_contrastive
  - async_labeling
  - _self_
```

因此：

- `actor/`、`data/`、`model/`、`rollout/`、`ref/` 等目录不是闲置 YAML，而是 Hydra 原生配置组。
- 这些配置组大体来自 VERL 原生配置体系，CoAgenticRetriever 在其上增加了 `coagentic_retriever_trainer.yaml`、`ranker_contrastive.yaml`、`async_labeling.yaml`、tool config、agent loop config 等扩展。
- 基础配置治理应该回到 Hydra 原生 config group，不应该用 tasks 下的一个大 YAML 替代这些组。

本计划采用新的分工：

```text
Hydra 原生 config groups:
  管基础配置体系、组件组合、可复用默认值。

yaml_to_dotlist.py:
  只管 partial YAML overlay，把少量实验覆盖参数转成 Hydra dotlist。

task shell:
  只管任务身份、实际运行资源、服务生命周期，以及显式选择 config group / overlay YAML。
```

## 2. 治理目标

本次治理的核心目标是让 task script 像一个可读的训练任务声明，而不是半个训练 launcher。

目标：

- task 脚本可读：上半段写环境字段，下半段显式选择 Hydra config group 和 overlay YAML。
- 环境配置和训练/推理参数配置分开。
- 公共训练/推理默认值优先进入 Hydra 原生 config group。
- 实验性、局部、不完整的配置进入 overlay YAML，再由 `yaml_to_dotlist.py` 转成 Hydra override。
- 环境到 Hydra 的桥接由 launcher 生成 runtime override YAML，并写入审计文件。
- 不再使用 `COAGENTIC_DEFAULT_EXTRA_ARGS`、`DEFAULT_COAGENTIC_EXTRA_ARGS`、task 中的 `COAGENTIC_EXTRA_ARGS`。
- 不再使用 `"$@"` 做隐式用户参数透传。
- dry-run 能审计最终环境、Hydra group selections、overlay YAML 列表、runtime override YAML 和最终 dotlist。

非目标：

- 不重写 CoAgenticRetriever/VERL 主程序。
- 不一次性治理所有 train/eval task。
- 不把服务启动、GPU wait、日志/checkpoint 生命周期迁入训练 YAML。
- 不把 `yaml_to_dotlist.py` 扩展成 Hydra composition 替代品。

## 3. 当前问题判断

当前配置链路混合了四种机制：

1. task 脚本直接 export 环境参数，例如 GPU、judge endpoint、batch、rollout、micro batch。
2. task 脚本通过 `COAGENTIC_EXTRA_ARGS` 拼 Hydra override 字符串。
3. v2 launcher 动态拼 `DEFAULT_COAGENTIC_EXTRA_ARGS`，再 export 成 `COAGENTIC_DEFAULT_EXTRA_ARGS`。
4. asset runner 在 `exec python main...` 时硬编码大量 env -> Hydra 参数。

这导致：

- task 脚本看不出哪些参数是运行环境，哪些是训练逻辑。
- launcher 变成隐式配置生成器。
- `COAGENTIC_DEFAULT_EXTRA_ARGS` / `COAGENTIC_EXTRA_ARGS` 名字不表达真实语义。
- 用户和维护者很难判断最终 Hydra 参数来源。
- 大量参数绕开了 `CoAgenticRetriever/config/actor`、`data`、`model`、`rollout` 等已有配置组，形成“字段兼容但体系不兼容”的风险。

新的方向是：

```text
task script:
  声明任务身份、GPU/服务资源 override、服务生命周期。
  显式选择 Hydra config groups。
  显式选择少量 overlay YAML。

Hydra config:
  继续以 CoAgenticRetriever/config/coagentic_retriever_trainer.yaml 为唯一主 config。
  通过 actor/data/model/rollout/ref/... 等 config groups 管基础配置。

overlay YAML:
  只保存局部实验覆盖，不写 defaults。
  由 yaml_to_dotlist.py 转成 Hydra CLI override。

launcher:
  准备运行时路径、校验配置、启动服务。
  将 env GPU override 同步到 runtime override YAML。
  审计并执行。
```

## 4. yaml_to_dotlist.py 的角色

`src/hydra_overrides/yaml_to_dotlist.py` 的职责是把 partial YAML 转成 Hydra dotlist。

例如：

```yaml
ranker_training:
  async_labeling:
    sample_builder_request_batch: 3
```

转换后成为：

```bash
++ranker_training.async_labeling.sample_builder_request_batch=3
```

这适合管理“不完整、实验性、只覆盖少数字段”的 YAML。

它不负责：

- Hydra `defaults` 组合。
- 选择 `actor/data/model/rollout` 这种 config group。
- 校验某个 YAML 是否是完整 recipe。
- 理解哪些参数是环境参数、哪些参数是训练参数。

因此：

- `yaml_to_dotlist.py` 作为 overlay 工具保留。
- 基础配置体系必须使用 Hydra 原生 config group。
- overlay YAML 中不允许写 `defaults` 是合理约束；否则会把 Hydra composition 语义误压成普通字段。

## 5. YAML 放置原则

`src/` 是公共代码目录，本计划不在 `src` 下新增 YAML。

配置文件按“复用量”和“策略敏捷性”分层：

```text
复用量：CoAgenticRetriever/ > scripts/ > tasks/
策略敏捷性：tasks/ > scripts/ > CoAgenticRetriever/
```

放置规则：

| 目录 | 放什么 | 不放什么 |
| --- | --- | --- |
| `CoAgenticRetriever/config/` | Hydra 主 config、VERL/CoAgenticRetriever 原生 config groups、长期稳定基础配置。 | 单个 task 的临时实验覆盖。 |
| `CoAgenticRetriever/config/data/` | 可复用数据配置组，例如 co-search ablation 数据集。 | run-name、GPU、服务生命周期。 |
| `CoAgenticRetriever/config/model/` | 可复用模型配置组，例如 Qwen3-4B。 | task 实验差异。 |
| `CoAgenticRetriever/config/rollout/` | 可复用 rollout 配置组，例如 cosearch async 多轮检索预算。 | 单次运行资源占用。 |
| `CoAgenticRetriever/async_labeling/configs/` | LLM judge 服务启动配置。 | 训练 task overlay。 |
| `scripts/coagenticRetriever_v2/strategies_yaml/` | 多个脚本可复用的 partial overlay，例如 DeepSeek-Flash async labeling 策略覆盖。 | 完整基础配置体系。 |
| `tasks/train_tasks/coAgenticRetriever/configs/` | 单个 task 的 partial overlay，例如 mix-signal b3 实验差异和默认资源布局。 | 通用框架默认值、大而全 recipe。 |

本次治理中：

- `coagentic_retriever_trainer.yaml` 继续作为唯一 Hydra 主 config。
- `actor/data/model/rollout/ref/...` 继续作为基础 config group。
- 可复用基础配置优先新增到 `CoAgenticRetriever/config/<group>/`。
- scripts/tasks 下的 YAML 降级为 overlay，不写 `defaults`。
- `sample_builder_request_batch=3` 属于 mix-signal b3 实验差异，进入 task overlay，而不是 task env。

## 6. 参数分类与覆盖原则

### 6.1 Task 环境参数

定义：任务身份、本机资源 override、服务生命周期和 dry-run 相关参数。它们是“这次在哪台机器上怎么跑”的声明。

管理方式：

- 保留在 task 脚本中。
- GPU 相关 env 是对 YAML 默认资源配置的 override，必须最后生效。
- 允许调用 `src/runtime/wait_for_gpus.sh`。
- 不放训练 batch、rollout budget、sample builder、loss、prompt budget 等算法参数。

| 参数 | 说明 |
| --- | --- |
| `ROOT` | 仓库根目录。 |
| `EXP_NAME` | run-name/log/checkpoint 身份。 |
| `GROUP_NAME` | 实验组名，可显式写出或沿用 launcher 默认。 |
| `RUN_STAMP` | 可选固定 run 时间戳。 |
| `PY` | Python 环境选择，通常由 launcher 兜底。 |
| `AGENT_GPU_IDS` | agent rollout/update GPU；覆盖 YAML 默认资源布局。 |
| `RANK_GPU_ID` | dense ranker GPU；覆盖 YAML 默认资源布局和 ranker device。 |
| `RECALL_GPU_ID` | recall retriever 服务 GPU；覆盖 YAML 默认资源布局和 recall device。 |
| `LLM_JUDGE_GPU_IDS` | LLM judge vLLM 服务 GPU；覆盖 judge service YAML 默认 GPU。 |
| `WAIT_FOR_GPUS` | GPU wait 目标列表。 |
| `WAIT_FOR_GPU_RELEASE` | 是否等待 GPU 释放。 |
| `WAIT_FOR_GPU_INTERVAL_SECONDS` | GPU wait 轮询间隔。 |
| `WAIT_FOR_GPU_TIMEOUT_SECONDS` | 可选 GPU wait 超时。 |
| `WAIT_FOR_GPU_LABEL` | GPU wait 日志标签。 |
| `AUTO_START_RECALL_SERVICE` | recall 服务是否自动启动。 |
| `AUTO_STOP_RECALL_SERVICE` | recall 服务是否自动停止。 |
| `RECALL_SERVICE_WAIT_SECONDS` | recall 服务启动等待时间。 |
| `AUTO_START_LLM_JUDGE` | judge 服务是否自动启动。 |
| `AUTO_STOP_LLM_JUDGE` | judge 服务是否自动停止。 |
| `LLM_JUDGE_SERVICE_CONFIG` | judge vLLM 服务启动 YAML。 |
| `LLM_JUDGE_PREFLIGHT` | judge endpoint 检查开关。 |
| `LLM_JUDGE_WAIT_SECONDS` | judge 服务启动等待时间。 |
| `CHECKPOINT_ROOT` | checkpoint 根目录。 |
| `DRY_RUN` | dry-run 开关。 |

### 6.2 Hydra config group 参数

定义：基础、可复用、组件级训练/推理配置。它们是“这个系统默认如何组成”的声明。

管理方式：

- 放入 `CoAgenticRetriever/config/<group>/`。
- 通过 Hydra 原生命令行选择 config group。
- 适合多个训练任务复用。

建议从当前 shell 默认值中迁出的稳定配置：

| 参数域 | 目标 |
| --- | --- |
| data | `CoAgenticRetriever/config/data/co_search_ablation.yaml` |
| model | `CoAgenticRetriever/config/model/qwen3_4b.yaml` |
| rollout | `CoAgenticRetriever/config/rollout/cosearch_async_qwen3_4b.yaml` |
| actor/ref | 如需要，新增 actor/ref profile；否则保留现有 `dp_actor` / `dp_ref` 后用 overlay 覆盖少量字段。 |
| reward | 保留在主 config 或稳定 overlay，避免 task shell 写 reward path/name。 |
| ranker/recall base | 优先复用 `ranker_contrastive.yaml`；只有确实需要可复用变体时再升级为 config group。 |

示例 group selection：

```bash
data@data=co_search_ablation
model@actor_rollout_ref.model=qwen3_4b
rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b
```

### 6.3 Overlay YAML 参数

定义：实验性质、只含部分字段、不完整的 YAML，用来覆盖 Hydra 组合后的最终配置。

管理方式：

- scripts/tasks 下的 overlay YAML 不写 `defaults`。
- 由 `yaml_to_dotlist.py` 转成 Hydra dotlist。
- 不用于替代 `actor/data/model/rollout` 等 config group。

适合放入 overlay 的内容：

| 参数 | 目标位置 |
| --- | --- |
| DeepSeek-Flash async labeling 策略覆盖 | `scripts/coagenticRetriever_v2/strategies_yaml/async_labeling_deepseek_flash_rank50_select_all.yaml` |
| `sample_builder_request_batch=3` | `tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml` |
| task 默认 `resources` 布局 | `tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml` |
| 临时训练 profile 覆盖 | task overlay；若复用稳定，再提升到 `CoAgenticRetriever/config/` config group。 |

不应继续作为 task env 主配置的参数：

| 参数 | 目标 |
| --- | --- |
| `MODEL_PATH` | `model` config group |
| `TRAIN_DATA` / `VAL_DATA` | `data` config group |
| `GPU_MEMORY_UTILIZATION` | `rollout` config group or overlay |
| `MAX_NUM_SEQS` | `rollout` config group or overlay |
| `AGENT_WORKERS` | `rollout` config group or overlay |
| `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU` | actor/ref/rollout config group or overlay |
| `ACTOR_MICRO_BATCH_SIZE_PER_GPU` | actor config group or overlay |
| `SAMPLE_BUILDER_REQUEST_BATCH` | task overlay |

### 6.4 GPU 双层配置与覆盖

GPU 分配必须同时存在于 YAML 和 task env：

- YAML 中写默认资源布局，保证 task 自描述、可审计、可 dry-run。
- task env 中写本次运行的实际资源 override，方便换机器、排队、抢资源。
- env override 的优先级必须高于 YAML 默认值。

默认资源字段建议放在 task overlay 的 `resources` 段：

```yaml
resources:
  agent_gpu_ids: "0,1,2,3"
  rank_gpu_id: "4"
  recall_gpu_id: "5"
  llm_judge_gpu_ids: "6,7"
```

launcher 读取 overlay 默认值后，再用 task env 覆盖：

```text
overlay resources
< task env AGENT_GPU_IDS / RANK_GPU_ID / RECALL_GPU_ID / LLM_JUDGE_GPU_IDS
```

覆盖后必须同步到所有实际使用处：

| 目标 | 来源 |
| --- | --- |
| `GPU_IDS` / `CUDA_VISIBLE_DEVICES` | `AGENT_GPU_IDS,RANK_GPU_ID` |
| `trainer.n_gpus_per_node` | `AGENT_GPU_IDS` 数量 |
| `recall_retriever.device` | `cuda:${RECALL_GPU_ID}` |
| `ranker.device` | `cuda:${RANK_GPU_ID}` |
| tool config `ranker.device` | `cuda:${RANK_GPU_ID}` |
| recall service start env | `RECALL_GPU_ID` |
| judge service config `runtime.cuda_visible_devices` | `LLM_JUDGE_GPU_IDS` |
| wait list | `AGENT_GPU_IDS,RANK_GPU_ID,RECALL_GPU_ID,LLM_JUDGE_GPU_IDS` |

这一步由 launcher 生成最后生效的 runtime override 文件：

```bash
${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml
```

该文件写入审计目录，内容类似：

```yaml
trainer:
  n_gpus_per_node: 4

recall_retriever:
  device: cuda:5

ranker:
  device: cuda:4

resources:
  agent_gpu_ids: "0,1,2,3"
  rank_gpu_id: "4"
  recall_gpu_id: "5"
  llm_judge_gpu_ids: "6,7"
```

runtime override 排在所有 overlay YAML 之后，保证 env GPU override 最后生效。

## 7. YAML 设计

### 7.1 Hydra 原生 config group

优先新增可复用配置组，而不是写大而全 task recipe。

建议新增：

```bash
CoAgenticRetriever/config/data/co_search_ablation.yaml
CoAgenticRetriever/config/model/qwen3_4b.yaml
CoAgenticRetriever/config/rollout/cosearch_async_qwen3_4b.yaml
```

这些文件可以通过 `defaults` 继承现有基础配置，例如：

```yaml
defaults:
  - legacy_data
  - _self_

train_files:
  - /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet
val_files:
  - /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet
```

说明：

- `data/model/rollout` 属于已有 Hydra config group，可以通过 group selection 原生切换。
- 不需要把这些字段复制进 tasks 下的完整 recipe。
- 如果 async labeling 后续也要彻底 config group 化，应先把当前顶层 `async_labeling.yaml` 重构为 group base，再调整 `coagentic_retriever_trainer.yaml` 的 defaults；这属于较大变更，本轮可以先用 overlay。

### 7.2 Async labeling overlay

建议路径：

```bash
scripts/coagenticRetriever_v2/strategies_yaml/async_labeling_deepseek_flash_rank50_select_all.yaml
```

职责：描述可被多个 task 复用的 DeepSeek-Flash async labeling 策略覆盖。

内容来源：

- `CoAgenticRetriever/config/async_labeling.yaml`
- `scripts/coagenticRetriever_v2/strategies_yaml/async_labeling_deepseek_flash.yaml`

管理：

- `ranker_training.signal_source`
- `ranker_training.shared_inference_ranker`
- `ranker_training.async_labeling.*`
- judge stage endpoint/model/prompt/schema
- async sample builder
- async logging

约束：

- 不写 `defaults`。
- 不放 `sample_builder_request_batch=3` 这种单个 task 的 mix-signal 身份。
- 如果该策略长期稳定且被多个入口使用，再考虑升级为 `CoAgenticRetriever/config/async_labeling/` config group。

### 7.3 Task overlay

建议路径：

```bash
tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml
```

职责：只描述 canonical mix-signal b3 相比基础 async labeling 训练的实验差异。

应包含：

```yaml
ranker_training:
  async_labeling:
    sample_builder_request_batch: 3

resources:
  agent_gpu_ids: "0,1,2,3"
  rank_gpu_id: "4"
  recall_gpu_id: "5"
  llm_judge_gpu_ids: "6,7"
```

原则：

- 不写模型路径、数据路径、rollout 大段参数。
- 不写 `defaults`。
- 不承担完整 recipe 职责。
- 如果以后新增 b5/b10，新增同类 task overlay：

```bash
tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b5_overlay.yaml
```

不要通过 `SAMPLE_BUILDER_REQUEST_BATCH` env 隐式改变实验身份。

### 7.4 Judge service YAML

保留当前路径，本轮只要求 task 显式选择：

```bash
CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

职责：描述 judge vLLM 服务如何启动。

注意：

- `runtime.cuda_visible_devices` 是 YAML 默认值。
- `LLM_JUDGE_GPU_IDS` env 必须能覆盖它。
- `server.endpoint` 应与 async labeling overlay 中的 stage endpoint 一致；launcher dry-run 要校验二者一致。

### 7.5 Runtime override YAML

建议由 launcher 生成，不由用户手写：

```bash
${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml
```

职责：把 task env 中的实际运行资源、运行时路径和日志路径转换为最终 Hydra override。

管理：

- GPU/env override 后的 device。
- `trainer.experiment_name`
- `trainer.default_local_dir`
- rollout/validation data dir。
- `ranker_training.construction_log_jsonl`
- 必要时的 `recall_retriever.service_url`。

该 YAML 是桥接层，但它是审计产物，不是 task 人手维护的配置碎片。

## 8. Task 脚本目标形态

目标 task 脚本应接近：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

# 环境配置：任务身份、实际资源 override、服务生命周期、等待策略。
export EXP_NAME="CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all"
export AGENT_GPU_IDS="0,1,2,3"
export RANK_GPU_ID="4"
export RECALL_GPU_ID="5"
export LLM_JUDGE_GPU_IDS="6,7"
export AUTO_START_LLM_JUDGE="1"
export AUTO_STOP_LLM_JUDGE="1"
export LLM_JUDGE_SERVICE_CONFIG="${ROOT}/CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml"
export LLM_JUDGE_PREFLIGHT="1"
export WAIT_FOR_GPUS="${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}"
export WAIT_FOR_GPU_RELEASE="1"
export WAIT_FOR_GPU_INTERVAL_SECONDS="30"
export WAIT_FOR_GPU_LABEL="mix-signal experiment GPU wait"

source "${ROOT}/src/runtime/wait_for_gpus.sh"
wait_for_gpus_if_enabled

bash "${ROOT}/scripts/coagenticRetriever_v2/01_train_qwen3_4b_ablation_1epoch_timing.sh" \
  --DATA_CONFIG="co_search_ablation" \
  --MODEL_CONFIG="qwen3_4b" \
  --ROLLOUT_CONFIG="cosearch_async_qwen3_4b" \
  --OVERLAY_YAML="${ROOT}/scripts/coagenticRetriever_v2/strategies_yaml/async_labeling_deepseek_flash_rank50_select_all.yaml" \
  --OVERLAY_YAML="${ROOT}/tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml" \
  --LLM_JUDGE_SERVICE_CONFIG="${LLM_JUDGE_SERVICE_CONFIG}"
```

原则：

- task 脚本不使用 `"$@"`。
- task 脚本不设置 `HYDRA_OVERRIDE_YAMLS`。
- task 脚本不设置 `COAGENTIC_EXTRA_ARGS`。
- task 脚本不设置训练过程参数，例如 `GPU_MEMORY_UTILIZATION`、`MAX_NUM_SEQS`、micro batch。
- task 脚本通过显式参数选择 Hydra config group 和 overlay YAML。
- task 脚本可以设置 GPU env override；这些 override 最后覆盖 overlay YAML 中的 `resources` 和 device。

## 9. v2 Launcher 目标接口

目标接口：

```bash
bash 01_train_qwen3_4b_ablation_1epoch_timing.sh \
  --DATA_CONFIG=co_search_ablation \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=cosearch_async_qwen3_4b \
  --OVERLAY_YAML=/path/to/async_labeling_deepseek_flash_rank50_select_all.yaml \
  --OVERLAY_YAML=/path/to/mix_signal_b3_overlay.yaml \
  --LLM_JUDGE_SERVICE_CONFIG=/path/to/llm_judge_vllm.yaml
```

解析规则：

- `--DATA_CONFIG=name`
  - 转成 Hydra group override：`data@data=name`。
  - 校验 `CoAgenticRetriever/config/data/${name}.yaml` 存在。
- `--MODEL_CONFIG=name`
  - 转成 Hydra group override：`model@actor_rollout_ref.model=name`。
  - 校验 `CoAgenticRetriever/config/model/${name}.yaml` 存在。
- `--ROLLOUT_CONFIG=name`
  - 转成 Hydra group override：`rollout@actor_rollout_ref.rollout=name`。
  - 校验 `CoAgenticRetriever/config/rollout/${name}.yaml` 存在。
- `--OVERLAY_YAML=path`
  - 校验文件存在。
  - 校验不包含 `defaults`。
  - 通过 `yaml_to_dotlist.py` 转为 Hydra field overrides。
  - 多个 overlay 按传入顺序生效。
- `--LLM_JUDGE_SERVICE_CONFIG=path`
  - 校验文件存在。
  - 用于启动 judge 服务。
  - 校验 service endpoint 与 async labeling overlay 中的 stage endpoint 一致。

launcher 还要：

- 从 task overlay 读取默认 `resources`。
- 用 task env 的 GPU 变量覆盖 overlay resources。
- 生成 `${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml`。
- 把 runtime override YAML 转成 dotlist，并放在所有 overlay dotlist 之后，保证 env GPU override 最后生效。

兼容策略：

- v2 launcher 可以暂时保留读取旧环境变量 `HYDRA_OVERRIDE_YAMLS` / `RANKER_STRATEGY_YAML` / `ASYNC_LABELING_YAML` 的能力，避免影响其它脚本。
- canonical mix-signal task 不再使用这些旧入口。
- canonical 路径中必须移除 `COAGENTIC_DEFAULT_EXTRA_ARGS`、`DEFAULT_COAGENTIC_EXTRA_ARGS`、task `COAGENTIC_EXTRA_ARGS`。

## 10. Asset Runner 调整

目标文件：

```bash
scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh
```

调整方向：

- 移除：

```bash
read -r -a coagentic_default_args <<< "${COAGENTIC_DEFAULT_EXTRA_ARGS:-}"
read -r -a coagentic_extra_args <<< "${COAGENTIC_EXTRA_ARGS}"
```

- 不再插入：

```bash
"${coagentic_default_args[@]}"
"${coagentic_extra_args[@]}"
"$@"
```

- 明确分开：

```text
Hydra config group selections:
  data@data=...
  model@actor_rollout_ref.model=...
  rollout@actor_rollout_ref.rollout=...

Overlay YAML dotlist:
  scripts overlay
  task overlay
  runtime env override
```

- 最终执行顺序：

```text
base hardcoded minimal args, only if truly required
< Hydra config group selections
< reusable overlay YAML dotlist
< task overlay YAML dotlist
< runtime env override YAML dotlist
```

说明：

- asset runner 仍可保留最底层运行必须参数，例如 Python main、CUDA 环境变量、Ray 启动必要项。
- 能进入 Hydra config group 的基础参数应从 `exec` 命令中移走。
- 能进入 overlay YAML 或 runtime override YAML 的字段覆盖应从 `exec` 命令中移走。
- 如果一次性迁移所有 hardcoded Hydra args 风险太大，第一步至少迁移 canonical mix-signal 涉及的字段，并留下 TODO。

## 11. 最终配置审计

dry-run 和正式运行都应写出：

| 文件 | 内容 |
| --- | --- |
| `${LOG_DIR}/${RUN_NAME}.env` | 环境参数快照。 |
| `${LOG_DIR}/${RUN_NAME}.hydra_groups.txt` | 最终 Hydra config group selections。 |
| `${LOG_DIR}/${RUN_NAME}.overlay_yamls.txt` | 最终参与转换的 overlay YAML 文件，按优先级顺序排列。 |
| `${LOG_DIR}/${RUN_NAME}.hydra_args.txt` | 最终传给 Python 主程序的 Hydra dotlist，按传入顺序排列。 |
| `${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml` | 由 task env 派生的最终运行时覆盖。 |

审计要求：

- 能看出基础配置来自 `coagentic_retriever_trainer.yaml`。
- 能看出选中了哪些 Hydra config group。
- 能看出哪些 YAML 是 overlay，而不是完整 recipe。
- 能看出 GPU env override 覆盖了哪些 YAML 默认 device。
- 能看出 `sample_builder_request_batch=3` 来自 task overlay。
- 不出现 `COAGENTIC_DEFAULT_EXTRA_ARGS` / `DEFAULT_COAGENTIC_EXTRA_ARGS` / task `COAGENTIC_EXTRA_ARGS`。

## 12. 实施顺序

1. 梳理当前 hardcoded Hydra args，按 config group / reusable overlay / task overlay / runtime override 分类。
2. 新增可复用 Hydra config group：
   - `CoAgenticRetriever/config/data/co_search_ablation.yaml`
   - `CoAgenticRetriever/config/model/qwen3_4b.yaml`
   - `CoAgenticRetriever/config/rollout/cosearch_async_qwen3_4b.yaml`
3. 新增或迁移 reusable overlay：
   - `scripts/coagenticRetriever_v2/strategies_yaml/async_labeling_deepseek_flash_rank50_select_all.yaml`
4. 新增 task overlay：
   - `tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml`
5. 修改 v2 launcher，支持 `--DATA_CONFIG` / `--MODEL_CONFIG` / `--ROLLOUT_CONFIG` / repeated `--OVERLAY_YAML` / `--LLM_JUDGE_SERVICE_CONFIG`。
6. 修改 v2 launcher：读取 overlay `resources`，用 task env GPU 覆盖，生成 runtime env override YAML。
7. 修改 launcher 的 async config 读取逻辑，使 judge preflight endpoint 来自 async labeling overlay，并校验 judge service endpoint 一致。
8. 修改 asset runner，移除 `COAGENTIC_DEFAULT_EXTRA_ARGS` 和 `COAGENTIC_EXTRA_ARGS` 插入路径。
9. 将 canonical mix-signal 相关 hardcoded Hydra args 迁入 Hydra config group、overlay YAML 或 runtime override YAML。
10. 重写 `train_CAR_async_labeling_ds_flash_mix_signal_fix.sh` 为目标形态。
11. 增加 Hydra group、overlay YAML、runtime override YAML 和最终 dotlist 审计输出。
12. 更新相关记录文档，标注 canonical 入口和旧入口边界。

## 13. 验证计划

静态验证：

```bash
bash -n tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix.sh
bash -n scripts/coagenticRetriever_v2/01_train_qwen3_4b_ablation_1epoch_timing.sh
bash -n scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh
```

Hydra config group 验证：

```bash
cd CoAgenticRetriever
/data04/envs/ms/ms_cosearch_official/bin/python main_coagentic_retriever.py \
  --cfg job \
  data@data=co_search_ablation \
  model@actor_rollout_ref.model=qwen3_4b \
  rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b
```

Overlay YAML 验证：

```bash
/data04/envs/ms/ms_cosearch_official/bin/python \
  src/hydra_overrides/yaml_to_dotlist.py \
  scripts/coagenticRetriever_v2/strategies_yaml/async_labeling_deepseek_flash_rank50_select_all.yaml \
  tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml
```

Dry-run 验证：

```bash
DRY_RUN=1 bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix.sh
```

验收条件：

- dry-run 能成功生成 env/config 审计文件。
- `hydra_groups.txt` 中包含 `data@data=co_search_ablation`、`model@actor_rollout_ref.model=qwen3_4b`、`rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b`。
- `overlay_yamls.txt` 中 YAML 顺序和 task 脚本一致。
- `runtime_env_overrides.yaml` 中包含 env override 后的 GPU/device。
- `hydra_args.txt` 中包含 env override 后的 ranker/recall device。
- `hydra_args.txt` 中包含 `ranker_training.async_labeling.sample_builder_request_batch=3`。
- task 脚本中没有 `COAGENTIC_EXTRA_ARGS`、`DEFAULT_COAGENTIC_EXTRA_ARGS`、`HYDRA_OVERRIDE_YAMLS`、`"$@"`。
- canonical v2 路径中没有 `COAGENTIC_DEFAULT_EXTRA_ARGS`。

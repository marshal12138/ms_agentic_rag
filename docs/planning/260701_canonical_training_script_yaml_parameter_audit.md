# Canonical 训练入口 Shell/YAML 参数归属与覆盖审计

更新日期：2026-07-01

## 文档目的

本文用于审计 CoAgenticRetriever canonical 训练入口中“参数到底归谁管”：

- 训练语义应来自 Hydra YAML、overlay YAML 或显式 Hydra CLI override。
- 运行态资源、服务、日志、checkpoint 路径等由 launcher/runtime 编译层物化。
- 旧式 shell env 到 Hydra key 的隐式训练超参覆盖必须继续被拒绝。

框架已经从 Bash 内部拼接配置，迁移为 Python config compiler 统一编译。因此本文以当前代码为准，记录最新调用链、配置优先级、仍允许的 runtime override，以及当前 task wrapper 的实际效果。

## 当前结论

当前状态：

- 训练入口仍是 `scripts/coagenticRetriever_v2/01_train_launcher.sh`，但它现在只做副作用编排：初始化路径、调用 Python compiler、等待设备、启动/检查服务、启动训练、收尾转换 checkpoint。
- 配置编译的唯一入口已经下沉到 `scripts/coagenticRetriever_v2/assets/trainer_launcher/compile_config.py`。
- canonical 训练执行不再通过 `scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh`；当前链路直接调用 `run_canonical_training.py`，再 exec `CoAgenticRetriever/main_coagentic_retriever.py`。
- `01_train_launcher.sh` 当前只接受 canonical config mode。compiler 中仍保留 legacy writer，但该 launcher 会拒绝 `CANONICAL_CONFIG_MODE!=1`。
- 当前训练 launcher 支持 `run_mode=full` 和 `run_mode=no-ranker`；`co-training/co_training` 归一为 `full`。`ranker-only` 不属于当前训练 launcher 支持范围。
- `run_mode` 是 launcher-only 字段，不直接传给 Hydra trainer。compiler 会把它转换为 `*.run_mode_overrides.yaml`，再展开进 `hydra_args.txt`。
- 当前审计 task wrapper 默认在末尾追加 `--run_mode=no-ranker`，所以虽然脚本名仍带 `async_ranker_training`，默认实际训练形态是 no-ranker。若需要 full 路径，可在 task 命令末尾追加 `run_mode=full` 覆盖。
- `ENABLE_ASYNC_RANKER_TRAINING` 已废弃，并在 canonical 模式下被 `CANONICAL_DEPRECATED_ENV_OVERRIDES` 拒绝；是否需要 LLM judge 服务由最终 Hydra 配置推导。

## 当前审计入口和调用链

当前审计入口：

```text
tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

真实调用链：

```text
task wrapper
  -> scripts/coagenticRetriever_v2/01_train_launcher.sh
  -> scripts/coagenticRetriever_v2/assets/trainer_launcher/compile_config.py
     -> main_run_config.py
     -> resource.py
     -> run_mode.py
     -> runtime_env.py
     -> runtime_overrides.py
     -> hydra_args.py
     -> validators.py
     -> audit_files.py
  -> source *.launcher_runtime_env.sh
  -> wait GPU / prepare recall service / prepare LLM judge service / reporter
  -> scripts/coagenticRetriever_v2/assets/trainer_launcher/run_canonical_training.py
  -> CoAgenticRetriever/main_coagentic_retriever.py
```

`assets/00_run_agentic_iter_rag_verl.sh` 仍保留 legacy 和旧 canonical 分支，但已不在当前 `01_train_launcher.sh` 的 canonical 主链路上。

## 当前审计依据

基于当前源码执行 dry-run：

```bash
WAIT_FOR_GPU_RELEASE=0 DRY_RUN=1 EXP_NAME=doc_audit_current_212941 \
  bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

生成审计目录：

```text
log/train_logs/coAgenticRetriever/260701-212941-doc_audit_current_212941/
```

该 run 的关键结果：

- `RUN_MODE=no-ranker`
- `RUN_MODE_SOURCE=trainer_cli_override`
- `NEEDS_LLM_JUDGE_SERVICE=0`
- `TOOL_CONFIG=CoAgenticRetriever/config/coagentic_retriever_tool_config_no_ranker.yaml`
- `GPU_IDS=0,1,2,3`

另外验证 full 路径：

```bash
WAIT_FOR_GPU_RELEASE=0 DRY_RUN=1 EXP_NAME=doc_audit_full_213235 \
  bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh run_mode=full
```

生成审计目录：

```text
log/train_logs/coAgenticRetriever/260701-213235-doc_audit_full_213235/
```

该 run 的关键结果：

- `RUN_MODE=full`
- `RUN_MODE_SOURCE=trainer_cli_override`
- `NEEDS_LLM_JUDGE_SERVICE=1`
- `TOOL_CONFIG=CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`
- `GPU_IDS=0,1,2,3,4`

重点检查文件：

- `*.env`
- `*.launcher_runtime_env.sh`
- `*.hydra_args.txt`
- `*.trainer_main_hydra_config.txt`
- `*.hydra_groups.txt`
- `*.hydra_cli_overrides.txt`
- `*.overlay_yamls.txt`
- `*.run_mode_overrides.yaml`
- `*.runtime_env_overrides.yaml`
- `CoAgenticRetriever/config/**/*.yaml`
- `tasks/train_tasks/coAgenticRetriever/configs/*.yaml`
- `scripts/coagenticRetriever_v2/strategies_yaml/*.yaml`
- `scripts/coagenticRetriever_v2/assets/trainer_launcher/*.py`

当前 `hydra_args.txt` 的固定顺序：

```text
trainer main Hydra config
  -> Hydra config group selection
  -> 普通 overlay YAML 展开结果，过滤掉顶层 resource env 和 launcher-only run_mode
  -> run_mode_overrides.yaml 展开结果
  -> runtime_env_overrides.yaml 展开结果
  -> task/user 末尾显式 Hydra CLI override
```

这个顺序就是覆盖优先级，越靠后优先级越高。

## 1. 当前配置来源和优先级

### 1.1 main_run_config

当前 main-run manifest：

```text
CoAgenticRetriever/config/main_run/coAgenticRetriever_main.yaml
```

它是 launcher 级 manifest，不是 Hydra defaults。当前可提供：

- `trainer_main_hydra_config`
- `trainer_config_groups.data`
- `trainer_config_groups.model`
- `trainer_config_groups.rollout`
- `trainer_config_groups.ranker_base`
- `trainer_config_groups.async_ranker_training_base`
- `run_mode`
- `resource_config`
- `service_configs.llm_judge_service_config`
- `runtime_configs.tool_config`
- `overlay_yamls`
- `trainer_cli_overrides`

合并规则：

```text
main_run_config defaults < task wrapper 显式 launcher 参数
```

当前 task wrapper 显式传入了 main config、全部 Hydra group、resource、三个 overlay 和 judge service config，因此 main-run manifest 主要起默认基线和审计锚点作用。

### 1.2 resource 配置

当前 resource 配置：

```text
CoAgenticRetriever/config/resource/base.yaml
CoAgenticRetriever/config/resource/local_8gpu_0_7.yaml
```

resource 管理的是 launcher 运行环境变量，不是训练超参。允许字段由 `resource.py` 的 `RESOURCE_KEYS` 定义：

| 字段 | 用途 |
|---|---|
| `GROUP_NAME` | 实验分组。 |
| `AGENT_GPU_IDS` | actor/rollout 主训练设备。 |
| `RANK_GPU_ID` | dense ranker 设备。 |
| `RECALL_GPU_ID` | recall retriever 服务设备。 |
| `LLM_JUDGE_GPU_IDS` | LLM judge 服务设备。 |
| `AUTO_START_RECALL_SERVICE`、`AUTO_STOP_RECALL_SERVICE`、`RECALL_SERVICE_WAIT_SECONDS` | recall 服务编排。 |
| `AUTO_START_LLM_JUDGE`、`AUTO_STOP_LLM_JUDGE`、`LLM_JUDGE_PREFLIGHT`、`LLM_JUDGE_WAIT_SECONDS` | judge 服务编排。 |
| `WAIT_FOR_GPUS`、`WAIT_FOR_GPU_RELEASE`、`WAIT_FOR_GPU_INTERVAL_SECONDS`、`WAIT_FOR_GPU_LABEL` | 训练前设备等待。 |

resource 合并优先级：

```text
resource/base.yaml
  < resource/<selected>.yaml
  < 普通 OVERLAY_YAML 中的顶层 resource env 或 resources.*
  < 显式外部 env
```

普通 overlay 中允许写顶层 `AGENT_GPU_IDS` 等 resource env，也兼容 `resources.agent_gpu_ids` 等字段。compiler 会把这些字段用于 shell runtime 合并，并从 Hydra overlay dotlist 中过滤掉，避免传入不存在的 Hydra 顶层 key。

当前 task wrapper 在脚本顶部 export 了一组默认 resource env。这些 env 是显式外部 env，优先级高于 resource YAML；它们仍然只属于资源/服务/等待编排，不属于训练语义超参。

### 1.3 run_mode

`run_mode` 是 launcher-only 训练形态字段，可出现在：

- `main_run_config`
- 普通 overlay YAML
- task/user 末尾 CLI override，如 `run_mode=no-ranker` 或 `--run_mode=no-ranker`

优先级：

```text
main_run_config < overlay YAML < task/user CLI
```

当前支持值：

| run_mode | 当前效果 |
|---|---|
| `full` / `co-training` / `co_training` | 标准 CoAgenticRetriever 训练形态；ranker、shared inference ranker 和 async ranker training 的具体行为来自 ranker/async base 与 overlay。run-mode override 文件内容为空 `{}`。 |
| `no-ranker` / `no_ranker` | recall-only 训练形态；compiler 生成低层 Hydra override：关闭 ranker 训练、关闭 shared inference ranker、关闭 async ranker training，并默认切到 `coagentic_retriever_tool_config_no_ranker.yaml`。 |

当前 task wrapper 的固定参数中包含：

```text
--run_mode=no-ranker
```

因此当前默认 dry-run 结果是 no-ranker。这个参数不会出现在最终 `hydra_cli_overrides.txt` 中；它会先被 compiler 消费，然后写成 `*.run_mode_overrides.yaml`。

### 1.4 Hydra 训练配置

当前 canonical Hydra group：

```text
--config-name=coagentic_retriever_trainer
data=co_search_ablation
model@actor_rollout_ref.model=qwen3_4b
rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b
experimental/ranker_base@_global_=ranker_contrastive
experimental/async_ranker_training_base@_global_=async_ranker_training
```

训练语义默认值位于：

- `CoAgenticRetriever/config/coagentic_retriever_trainer.yaml`
- `CoAgenticRetriever/config/data/co_search_ablation.yaml`
- `CoAgenticRetriever/config/model/qwen3_4b.yaml`
- `CoAgenticRetriever/config/rollout/cosearch_async_qwen3_4b.yaml`
- `CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml`
- `CoAgenticRetriever/config/experimental/async_ranker_training_base/async_ranker_training.yaml`
- task/strategy overlay YAML

当前 task wrapper 显式选择三个 overlay：

```text
scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_deepseek_flash_rank50_select_all.yaml
tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml
tasks/train_tasks/coAgenticRetriever/configs/train_args_overlay.yaml
```

## 2. Shell 中存在但不属于训练 YAML 的运行变量

这些变量仍在 shell/env 中存在，但它们不应被理解为 Hydra 训练超参默认值。

### 2.1 Task wrapper

所在脚本：

```text
tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

| 类别 | 变量 / 参数 | 当前用途 |
|---|---|---|
| run 身份 | `EXP_NAME`、`GROUP_NAME` | 构造 `RUN_NAME`、日志目录、checkpoint 目录，并写入 runtime `trainer.experiment_name`。 |
| 资源分配 | `AGENT_GPU_IDS`、`RANK_GPU_ID`、`RECALL_GPU_ID`、`LLM_JUDGE_GPU_IDS` | 作为显式外部 resource env，覆盖 resource YAML 默认值。 |
| recall service 编排 | `AUTO_START_RECALL_SERVICE`、`AUTO_STOP_RECALL_SERVICE`、`RECALL_SERVICE_WAIT_SECONDS` | 控制 recall 检索服务是否启动、是否清理和等待时间。 |
| LLM judge service 编排 | `AUTO_START_LLM_JUDGE`、`AUTO_STOP_LLM_JUDGE`、`LLM_JUDGE_PREFLIGHT`、`LLM_JUDGE_WAIT_SECONDS` | 控制 judge 服务启动、清理和预检；是否需要服务由最终 Hydra 配置推导。 |
| 设备等待 | `WAIT_FOR_GPUS`、`WAIT_FOR_GPU_RELEASE`、`WAIT_FOR_GPU_INTERVAL_SECONDS`、`WAIT_FOR_GPU_LABEL` | 训练前等待设备释放。当前 wrapper 默认等待 agent、ranker、recall、judge 全部设备。 |
| 配置选择参数 | `--main_run_config`、`--trainer_main_hydra_config`、`--DATA_CONFIG`、`--MODEL_CONFIG`、`--ROLLOUT_CONFIG`、`--RANKER_BASE_CONFIG`、`--ASYNC_RANKER_TRAINING_BASE_CONFIG`、`--RESOURCE_CONFIG`、`--OVERLAY_YAML`、`--LLM_JUDGE_SERVICE_CONFIG` | 传给 compiler，用于选择 main-run manifest、Hydra group、resource、overlay 和 judge service config。 |
| run mode | `--run_mode=no-ranker` | 当前 task 固定的 launcher-only 训练形态覆盖；会生成 `run_mode_overrides.yaml`，不会作为普通 Hydra key 传入。 |
| 显式 Hydra CLI override | `--actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1`、`"$@"` | 进入最终 `hydra_args.txt` 末尾；用户追加参数优先级最高。 |

### 2.2 Canonical launcher

所在脚本：

```text
scripts/coagenticRetriever_v2/01_train_launcher.sh
```

| 类别 | 变量 / 参数 | 当前用途 |
|---|---|---|
| 路径定位 | `ROOT`、`PROJECT_ROOT`、`SCRIPT_DIR`、`ASSETS_DIR` | 定位仓库、项目和 helper。 |
| Python compiler 输入 | `CONFIG_COMPILER`、`RECALL_PREFLIGHT_PY`、`CANONICAL_TRAINING_RUNNER`、`PY`、`COSEARCH_ACCELERATOR`、device prefix / visible device var | 传给 compiler 或后续 runner。 |
| compiler 生成文件 | `LAUNCHER_RUNTIME_ENV_SH`、`CANONICAL_HYDRA_ARGS_FILE`、`CANONICAL_RUN_MODE_OVERRIDE_YAML`、`CANONICAL_RUNTIME_OVERRIDE_YAML`、`CANONICAL_*_FILE` | Bash source 或 dry-run 审计输出。 |
| run 身份和目录 | `RUN_NAME`、`EXP_NAME`、`GROUP_NAME`、`LOG_DIR`、`OUT_DIR`、`TRAIN_LOG`、`METRICS_JSONL`、`SEARCH_TIMING_JSONL`、`NVIDIA_SMI_CSV`、`REPORT_PREFIX` | 由 compiler 物化，Bash 只消费。 |
| 资源和设备 | `GPU_IDS`、`N_GPUS_PER_NODE`、`AGENT_GPU_IDS`、`RANK_GPU_ID`、`RECALL_GPU_ID`、`LLM_JUDGE_GPU_IDS`、`WAIT_FOR_GPUS` | 等待设备、设置训练主进程可见设备、启动外部服务。 |
| 服务编排 | `NEEDS_LLM_JUDGE_SERVICE`、`AUTO_START_RECALL_SERVICE`、`AUTO_STOP_RECALL_SERVICE`、`AUTO_START_LLM_JUDGE`、`AUTO_STOP_LLM_JUDGE`、`LLM_JUDGE_PREFLIGHT`、`LLM_JUDGE_ENDPOINT`、`LLM_JUDGE_SERVICE_CONFIG` | 启动、清理和预检 recall / judge 服务。 |
| recall 预检 | `RETRIEVAL_SERVICE_URL`、`RETRIEVAL_PREFLIGHT_QUERY`、`RETRIEVAL_PREFLIGHT_EXPECT`、`RECALL_TOP_K`、`TOP_N`、`TOP_M` | 由 tool config/runtime 同步，用于 HTTP ready 和 semantic preflight。 |
| 报告和清理 | `REPORT_STEPS`、`NVIDIA_SMI_INTERVAL`、`REPORT_INTERVAL_SECONDS`、`CHECKPOINT_*` | nvidia-smi 采样、训练报告、checkpoint conversion 和清理策略。 |

`01_train_launcher.sh` 不再手写训练超参默认值，也不再拼接长串 Hydra 参数。

### 2.3 Python compiler

所在目录：

```text
scripts/coagenticRetriever_v2/assets/trainer_launcher/
```

关键模块职责：

| 模块 | 职责 |
|---|---|
| `cli.py` | 解析 launcher 参数，把已知 launcher selector 与未知 Hydra override 分流。 |
| `main_run_config.py` | 读取 main-run manifest，并按 main_run < task CLI 合并默认选择。 |
| `resource.py` | 合并 resource env，过滤 overlay 中的顶层 resource env。 |
| `run_mode.py` | 解析 launcher-only run mode，并生成低层 Hydra override。 |
| `runtime_env.py` | 物化 run identity、日志路径、服务参数、tool config 同步值。 |
| `runtime_overrides.py` | 写 `runtime_env_overrides.yaml`。 |
| `hydra_args.py` | 按固定优先级生成 `hydra_args.txt`。 |
| `validators.py` | 拒绝 deprecated env、compose 最终 Hydra、检查路径、推导是否需要 judge 服务。 |
| `audit_files.py` | 写 `.env`、`hydra_args.txt`、`launcher_runtime_env.sh` 等审计/执行文件。 |

### 2.4 Canonical training runner

所在脚本：

```text
scripts/coagenticRetriever_v2/assets/trainer_launcher/run_canonical_training.py
```

当前职责：

- 只接受 canonical 模式。
- 只读取 `CANONICAL_HYDRA_ARGS_FILE`。
- 设置训练进程必需环境变量，例如 `PYTHONPATH`、visible devices、vLLM/NCCL/GLOO/WANDB 相关变量。
- 直接 `execvpe` 执行 `CoAgenticRetriever/main_coagentic_retriever.py`。

它不解析 task CLI、不合并 YAML、不启动服务、不等待 GPU、不做 checkpoint conversion。

## 3. 当前仍存在的 shell / CLI 覆盖

这里的“CLI 覆盖”指最终进入 `hydra_args.txt` 末尾的显式 Hydra override。

当前 task wrapper 固定保留一项：

| Hydra key | YAML 中的位置 | 当前 CLI override | 说明 |
|---|---|---|---|
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | `tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml` 已设置为 `1` | `actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1` | 值不变；这是当前固定保留的重复 Hydra CLI override。 |

当前 task wrapper 还固定传入：

```text
--run_mode=no-ranker
```

这不是普通 Hydra override。compiler 会消费它，生成 `run_mode_overrides.yaml`，并从 `hydra_cli_overrides.txt` 中移除。

用户如果在 task wrapper 之后显式追加 `key=value` 或 `--key=value`，launcher 会归一化为最高优先级 Hydra CLI override。用户也可以追加 `run_mode=full` / `run_mode=no-ranker` 覆盖当前固定 run mode；该类参数仍然由 run-mode 编译层消费。

## 4. 当前 runtime 覆盖

runtime override 只指 compiler 生成 `*.runtime_env_overrides.yaml` 后展开进 `hydra_args.txt` 的内容。这些值依赖本次 run 的目录、设备、服务 URL 或当前工具配置，不适合写进实验 overlay。

当前 runtime override 字段：

| Hydra key | runtime 来源 | 保留原因 |
|---|---|---|
| `trainer.experiment_name` | `EXP_NAME` / run identity | 当前 run 身份。 |
| `trainer.default_local_dir` | `OUT_DIR` | 当前 run checkpoint 目录。 |
| `trainer.device` | accelerator device prefix | CUDA / NPU 设备类型。 |
| `trainer.n_gpus_per_node` | `AGENT_N_GPUS_PER_NODE` | 当前 agent 资源分配。 |
| `trainer.nnodes` | `NNODES` | 当前分布式节点数。 |
| `trainer.rollout_data_dir` | 当前 run 的 `rollout_data` 目录 | 当前 run 产物路径。 |
| `trainer.validation_data_dir` | 当前 run 的 `validation_data` 目录 | 当前 run 产物路径。 |
| `actor_rollout_ref.nccl_timeout` | `NCCL_TIMEOUT` / `HCCL_TIMEOUT` 派生值 | NPU/CUDA 通信兼容项。 |
| `actor_rollout_ref.actor.use_torch_compile` | `ACTOR_USE_TORCH_COMPILE`，NPU 下默认 `False` | 硬件兼容项。 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | 最终 `TOOL_CONFIG` | 保证训练进程和 launcher preflight 使用同一份 tool YAML；no-ranker 时默认切换到 no-ranker tool config。 |
| `recall_retriever.model_path` | `RECALL_MODEL_PATH` | 本地 recall 模型路径。 |
| `recall_retriever.device` | `RECALL_GPU_ID` 映射出的设备 | 当前 recall 服务设备。 |
| `recall_retriever.service_url` | `RETRIEVAL_SERVICE_URL` | 当前 recall 服务 URL。 |
| `ranker.device` | `RANK_GPU_ID` 映射出的设备 | 当前 ranker 设备；no-ranker 下仍作为审计/配置值写入，但 run-mode override 会关闭 ranker 使用路径。 |
| `ranker.model_path` | `RANKER_BASE_MODEL_PATH` 非空时写入 | 条件本地路径覆盖。 |
| `ranker.encoder_path` | `RANKER_ENCODER_PATH` 非空时写入 | 条件本地路径覆盖。 |
| `ranker_training.construction_log_jsonl` | 当前 run 的 construction JSONL 路径 | 当前 run 产物路径。 |
| `ranker_training.async_ranker_training.logging.log_dir` | 当前 run 的 async ranker 日志目录 | 当前 run 产物路径。 |
| `resources.agent_gpu_ids` | 最终 `AGENT_GPU_IDS` | 训练配置中的资源审计视图。 |
| `resources.rank_gpu_id` | 最终 `RANK_GPU_ID` | 训练配置中的资源审计视图。 |
| `resources.recall_gpu_id` | 最终 `RECALL_GPU_ID` | 训练配置中的资源审计视图。 |
| `resources.llm_judge_gpu_ids` | 最终 `LLM_JUDGE_GPU_IDS` | 训练配置中的资源审计视图。 |

## 5. 当前 run_mode 覆盖

`*.run_mode_overrides.yaml` 是当前新增的 canonical 编译产物。

`run_mode=full` 时内容为空：

```yaml
{}
```

`run_mode=no-ranker` 时内容为：

```yaml
trainer:
  ranker_trainable: false
  ranker_update_mode: disabled
  disable_reranker_rollout: true
ranker_training:
  signal_source: pseudo_rank
  shared_inference_ranker:
    enable: false
  async_ranker_training:
    enable: false
```

这些低层 Hydra key 不需要在 task overlay 中重复维护；task 只声明高层 `run_mode` 即可。

## 6. 当前 deprecated env guard

canonical 模式下仍拒绝历史 shell 训练语义 env。当前拒绝列表位于：

```text
scripts/coagenticRetriever_v2/assets/trainer_launcher/validators.py
```

变量名：

```text
CANONICAL_DEPRECATED_ENV_OVERRIDES
```

当前包含：

```text
TRAINER_LOGGER
SAVE_FREQ
TEST_FREQ
RESUME_MODE
MAX_ACTOR_CKPT_TO_KEEP
DUMP_ROLLOUT_EVERY_STEP_NUM
DUMP_ROLLOUT_NUM_EVERYTIME
MAX_ROLLOUT_DUMP_NUM
ROLLOUT_TRACE_MODE
RECALL_TOP_K
RANK_TOP_K
ACTOR_BATCH_SIZE
ACTOR_MICRO_BATCH_SIZE_PER_GPU
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
ACTOR_LR
KL_LOSS_COEF
TRAIN_BATCH_SIZE
VAL_BATCH_SIZE
TRAIN_MAX_SAMPLES
VAL_MAX_SAMPLES
N_ROLLOUTS
MODEL_PATH
TRAIN_DATA
VAL_DATA
LORA_RANK
LORA_ALPHA
TOTAL_STEPS
ENABLE_ASYNC_RANKER_TRAINING
```

这些值已经迁入 Hydra YAML/overlay 或 run-mode 编译链路。需要临时改变时，应修改 YAML/overlay，或在 task 末尾显式追加 Hydra CLI override。

## 7. 当前 path check 和静态校验

canonical compiler 在 dry-run/正式训练前做静态校验，不启动训练、不启动服务：

1. 拒绝 deprecated shell env 训练语义覆盖。
2. 生成 run identity、审计文件路径和 runtime env。
3. 写 `run_mode_overrides.yaml`、`runtime_env_overrides.yaml` 和 `hydra_args.txt`。
4. compose 最终 Hydra 配置，推导 `NEEDS_LLM_JUDGE_SERVICE`：
   - `ranker_training.async_ranker_training.enable == true`
   - 且 stages 中存在 `type: llm_as_judge`
5. 如果需要 LLM judge：
   - 检查 overlay / prompt / judge service config。
   - 调用 `CoAgenticRetriever/scripts/launch_llm_as_judge.sh --dry-run` 校验服务配置。
6. 从最终 Hydra 配置抽取并检查：
   - `actor_rollout_ref.model.path`
   - `data.train_files`
   - `data.val_files`
   - `actor_rollout_ref.rollout.multi_turn.tool_config_path`
   - `actor_rollout_ref.rollout.agent.agent_loop_config_path`
7. 继续检查运行态路径：
   - `PROJECT_ROOT`
   - `CORPUS_JSONL`
   - `RECALL_MODEL_PATH`
   - 条件 `RANKER_BASE_MODEL_PATH`

这意味着 canonical 训练的模型、数据、LoRA、batch、sample、rollout、schedule 等实验语义仍应来自 YAML/overlay 或显式 CLI override，而不是 shell 默认值。

## 8. 当前维护规则

- `01_train_launcher.sh` 只做执行编排，不再新增 Bash env 到 Hydra key 的训练超参映射。
- 训练语义默认值进入 `CoAgenticRetriever/config/**/*.yaml`、task overlay 或 strategy overlay。
- task 级训练形态用 `run_mode` 表达，由 compiler 统一转换为低层 Hydra override。
- 资源、服务、设备等待可以进入 `CoAgenticRetriever/config/resource/*.yaml`；task wrapper 中 export 的 resource env 属于显式外部覆盖。
- 每次运行才确定的路径、设备、服务 URL、tool config path、资源审计视图和硬件兼容项可以保留在 runtime override。
- `ENABLE_ASYNC_RANKER_TRAINING` 不再使用；full/no-ranker 由 `run_mode` 和最终 Hydra 配置推导。
- `ranker-only` 不应加入当前训练 launcher 文档；评估脚本中若仍出现 ranker-only，属于 eval 路径，不属于本文审计范围。
- 新增 compiler 生成文件时，应同步更新 `.env` 审计、`launcher_runtime_env.sh` 白名单和本文档。

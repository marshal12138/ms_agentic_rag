# CoAgenticRetriever 训练任务脚本评审记录

日期：2026-06-18

评审对象：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash_mix_signal.sh
```

相关文件：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_tasks_records.md
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
```

## 总体评价

`train_CAR_async_labeling_ds_flash_mix_signal.sh` 作为一个“能跑的实验入口”是合格的。它清楚表达了本次实验的核心变量：

```bash
SAMPLE_BUILDER_REQUEST_BATCH=3
ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}
```

这说明该实验的意图是：在 DeepSeek-V4-Flash async labeling 基础上，每次 ranker async update 从 completed judge signal buffer 中取 3 条信号，而不是基础 async labeling 实验中的 1 条。

但是作为长期维护的 `tasks/train_tasks/` 实验配方脚本，它已经出现维护品味上的坏味道：真正的实验差异很小，脚本里却复制了大量通用训练默认值、服务配置、资源等待逻辑和 Hydra 参数拼接。继续按这个方式新增实验，后续会导致脚本漂移、默认参数不一致、覆盖顺序难以追踪。

当前判断：短期可继续使用；中期应重构为“薄 task script + 公共 helper/launcher”的结构。

## 好的部分

### 实验核心变量明确

脚本明确设置：

```bash
export SAMPLE_BUILDER_REQUEST_BATCH="${SAMPLE_BUILDER_REQUEST_BATCH:-3}"
```

并在 `COAGENTIC_EXTRA_ARGS` 末尾追加：

```bash
ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}
```

这保证它会覆盖 `async_labeling_deepseek_flash.yaml` 中的：

```yaml
sample_builder_request_batch: 1
```

因此实验差异在最终 Hydra 配置中是确定生效的。

### 保留了外部覆盖能力

多数变量使用：

```bash
VAR="${VAR:-default}"
```

外部可以通过环境变量覆盖，例如：

```bash
SAMPLE_BUILDER_REQUEST_BATCH=5 bash tasks/train_tasks/train_CAR_async_labeling_ds_flash_mix_signal.sh
```

这对 smoke test、资源临时调整、ablation 扩展都有用。

### 与项目分层方向基本一致

脚本最后仍调用公共训练入口：

```bash
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

recall service、LLM judge preflight、训练日志、checkpoint 目录、report 生成等核心执行逻辑没有完全复制到 task 脚本里。这一点是正确的。

## 主要问题

### 问题 1：实验差异很小，但脚本复制了大量公共默认值

`train_CAR_async_labeling_ds_flash_mix_signal.sh` 和 `train_CAR_async_labeling_ds_flash.sh` 的主体高度相似。真正差异主要是：

```bash
EXP_NAME=CAR_async_labeling_ds_flash_mix_signal_b3_v1
SAMPLE_BUILDER_REQUEST_BATCH=3
WAIT_FOR_GPU_RELEASE=1
WAIT_FOR_GPUS=...
ranker_training.async_labeling.sample_builder_request_batch=3
```

其余大量变量是公共 async-labeling 训练默认值：

```bash
GPU_MEMORY_UTILIZATION=0.55
MAX_NUM_SEQS=16
AGENT_WORKERS=4
TOOL_MAX_CONCURRENT_PER_WORKER=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
ACTOR_MICRO_BATCH_SIZE_PER_GPU=4
TOP_N=50
TOP_M=5
AGENT_GPU_IDS=0,1,2,3
RANK_GPU_ID=4
RECALL_GPU_ID=5
ENABLE_ASYNC_LABELING=1
ASYNC_LABELING_YAML=...
AUTO_START_LLM_JUDGE=1
AUTO_STOP_LLM_JUDGE=1
LLM_JUDGE_ENDPOINT=...
LLM_JUDGE_PREFLIGHT=1
```

这些变量一旦在某个脚本中更新，另一个 task 脚本很容易忘记同步，造成实验不可比。

处理建议：

- 新建公共 task helper，例如：

```bash
tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh
```

- 把 async-labeling 共同默认值放进去。
- `train_CAR_async_labeling_ds_flash.sh` 和 `train_CAR_async_labeling_ds_flash_mix_signal.sh` 只保留各自实验差异。

理想形态：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
source "${ROOT}/tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh"

export EXP_NAME="${EXP_NAME:-CAR_async_labeling_ds_flash_mix_signal_b3_v1}"
export SAMPLE_BUILDER_REQUEST_BATCH="${SAMPLE_BUILDER_REQUEST_BATCH:-3}"
append_coagentic_extra_args \
  "ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"

bash "${ROOT}/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh" "$@"
```

### 问题 2：GPU wait 是资源调度逻辑，长期不应放在 task 脚本里

当前脚本包含：

```bash
gpu_list_csv_to_lines() { ... }
wait_for_gpu_release() { ... }
```

这已经是一个小型资源调度器。短期放在 mix-signal 实验里可以接受，因为它确实是这个实验启动时的资源需求；但如果后续多个实验都需要等待 GPU 空闲，这段逻辑就会被复制，最终变成多个版本。

处理建议：

- 短期：如果只有这个脚本用，可以暂时保留。
- 中期：下沉到公共位置，例如：

```bash
scripts/coagenticRetriever_local/assets/wait_for_gpus.sh
```

或直接并入：

```bash
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

task 脚本只声明：

```bash
export WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},6,7}"
export WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"
```

公共 launcher 负责真正检查和等待。

### 问题 3：`COAGENTIC_EXTRA_ARGS` 的组合方式容易误伤默认配置

当前写法：

```bash
DEFAULT_COAGENTIC_EXTRA_ARGS="..."
export COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-${DEFAULT_COAGENTIC_EXTRA_ARGS}} ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"
```

问题在于：如果用户外部设置了 `COAGENTIC_EXTRA_ARGS`，它会完全替代 `DEFAULT_COAGENTIC_EXTRA_ARGS`。用户可能只是想额外加一个 override，但实际会丢掉这些默认加速/稳定配置：

```bash
actor_rollout_ref.rollout.max_num_batched_tokens=32768
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.ref.fsdp_config.param_offload=False
++data.apply_chat_template_kwargs.enable_thinking=False
```

这会让实验行为和脚本默认假设不一致，而且不容易从命令行看出来。

处理建议：

- 区分“脚本默认 extra args”和“用户追加 extra args”。
- 例如引入：

```bash
TASK_DEFAULT_COAGENTIC_EXTRA_ARGS="..."
TASK_EXTRA_COAGENTIC_ARGS="ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"
USER_COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-}"
export COAGENTIC_EXTRA_ARGS="${TASK_DEFAULT_COAGENTIC_EXTRA_ARGS} ${USER_COAGENTIC_EXTRA_ARGS} ${TASK_EXTRA_COAGENTIC_ARGS}"
```

这样用户的 override 仍可覆盖默认值，同时不会无意丢失默认加速配置。

如果希望用户 override 具有最高优先级，需要把用户参数放最后：

```bash
export COAGENTIC_EXTRA_ARGS="${TASK_DEFAULT_COAGENTIC_EXTRA_ARGS} ${TASK_EXTRA_COAGENTIC_ARGS} ${USER_COAGENTIC_EXTRA_ARGS}"
```

这里要明确策略：实验关键变量 `sample_builder_request_batch` 是否允许被用户覆盖。如果允许，用户参数放最后；如果不允许，实验关键变量放最后。

### 问题 4：judge GPU `6,7` 被硬编码，可能与 judge config 漂移

当前默认：

```bash
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},6,7}"
```

但 LLM judge GPU 实际来自：

```bash
LLM_JUDGE_SERVICE_CONFIG=CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

如果之后 judge config 改成 GPU 6 或 GPU 2,3，`WAIT_FOR_GPUS` 不会自动同步。表现上可能是等待了不必要的 GPU，或没有等待真正需要的 judge GPU。

处理建议：

- 短期：保留硬编码，但注释明确它必须和 judge service config 同步。
- 中期：增加 `LLM_JUDGE_GPU_IDS` 变量：

```bash
export LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"
```

- 长期：由公共 launcher 从 judge config 中解析 GPU 设置，再决定 wait list。

### 问题 5：task 脚本承担的注释解释过多，记录与代码职责边界变模糊

当前脚本内有大量解释性注释。这对当前阶段理解有帮助，但长期看，实验背景、假设和结果更适合放在：

```bash
tasks/train_tasks/train_tasks_records.md
docs/pre_works/
reports/train/
```

task 脚本本身应尽量短，只保留运行必要注释。否则脚本会变成半文档半 launcher，维护时容易混乱。

处理建议：

- task 脚本保留短注释：说明变量含义和危险点。
- 实验目的、假设、结果、对照关系放入 `train_tasks_records.md`。
- 阶段性反思和设计判断放入 `docs/pre_works/`。

## 推荐处理顺序

### 第一步：先修 `COAGENTIC_EXTRA_ARGS` 组合方式

这是最容易引入隐性实验偏差的问题。建议优先处理。

目标：

- 保留脚本默认 extra args。
- 允许用户追加或覆盖。
- 明确 `sample_builder_request_batch` 是实验强制变量还是可覆盖变量。

建议本实验中把 `sample_builder_request_batch` 放最后，确保该脚本语义稳定：

```bash
USER_COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-}"
export COAGENTIC_EXTRA_ARGS="${DEFAULT_COAGENTIC_EXTRA_ARGS} ${USER_COAGENTIC_EXTRA_ARGS} ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"
```

### 第二步：抽出 async-labeling 公共 task helper

目标：

- 减少 `train_CAR_async_labeling_ds_flash.sh` 和 `mix_signal` 的重复。
- 让两个脚本只表达实验差异。
- 后续新增 b5、b10、不同 judge、不同 sample_builder 策略时不会复制一整份脚本。

候选路径：

```bash
tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh
```

### 第三步：处理 GPU wait 逻辑

如果短期只有 `mix_signal` 用，可以先保留。

如果后续新增实验也要等待 GPU，立即下沉到：

```bash
scripts/coagenticRetriever_local/assets/wait_for_gpus.sh
```

或并入公共训练 launcher。

### 第四步：拆出 `LLM_JUDGE_GPU_IDS`

目标：

- 消除 `WAIT_FOR_GPUS` 中的裸 `6,7`。
- 让资源声明更显式。

建议：

```bash
export LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"
```

### 第五步：精简脚本注释，把解释沉淀到记录文档

保留必要运行注释，其它实验解释放到：

```bash
tasks/train_tasks/train_tasks_records.md
docs/pre_works/20260618_coagentic_retriever_train_task_script_review.md
```

## 理想目标

最终 `train_tasks` 里的脚本应该一眼能看出“这次实验只改了什么”。例如 mix-signal 实验最终应该接近：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
source "${ROOT}/tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh"

export EXP_NAME="${EXP_NAME:-CAR_async_labeling_ds_flash_mix_signal_b3_v1}"
export SAMPLE_BUILDER_REQUEST_BATCH="${SAMPLE_BUILDER_REQUEST_BATCH:-3}"
export WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"

append_coagentic_extra_args \
  "ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"

run_car_training "$@"
```

这时：

- task script 负责实验差异。
- common helper 负责同类实验默认值。
- `scripts/coagenticRetriever_local/` 负责真实训练执行。
- `train_tasks_records.md` 负责实验台账。
- `docs/pre_works/` 负责阶段性设计和反思。

## 2026-06-18 fix 进展记录

当前新增修复脚本：

```bash
tasks/train_tasks/train_CAR_async_labeling_ds_flash_mix_signal_fix.sh
```

### 已解决：GPU wait 下沉

GPU wait 已从 task 脚本内联函数下沉到项目级公共工具：

```bash
src/runtime/wait_for_gpus.sh
```

`fix.sh` 现在只声明等待策略：

```bash
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"
export WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"
export WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"
export WAIT_FOR_GPU_LABEL="${WAIT_FOR_GPU_LABEL:-mix-signal experiment GPU wait}"
```

然后调用公共函数：

```bash
source "${ROOT}/src/runtime/wait_for_gpus.sh"
wait_for_gpus_if_enabled
```

公共工具支持两种用法：

```bash
source "${ROOT}/src/runtime/wait_for_gpus.sh"
wait_for_gpus_if_enabled
```

或：

```bash
bash src/runtime/wait_for_gpus.sh --gpus "0,1,2" --interval 30 --timeout 0
```

相比原始内联实现，公共工具额外补充了：

- GPU id 格式校验。
- GPU id 存在性校验。
- 可选 timeout。
- 可复用 label。
- 跳过开关 `WAIT_FOR_GPU_RELEASE=0`。

### 已解决：`LLM_JUDGE_GPU_IDS` 解硬编码

`WAIT_FOR_GPUS` 中不再直接写裸 `6,7`，而是拆出：

```bash
export LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"
```

然后组合：

```bash
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"
```

这还不是完全自动解析 judge config，但已经比裸写 `6,7` 更清楚。后续如果 judge GPU 改动，可以通过 `LLM_JUDGE_GPU_IDS=...` 覆盖。

### 已解决：GPU wait 工具内部 cleanup 风格

`src/runtime/wait_for_gpus.sh` 不再在 `wait_for_gpu_release` 函数内部动态定义 `gpu_wait_cleanup_tmp`，改为全局 helper：

```bash
gpu_wait_cleanup_tmp() {
  local tmp_dir="$1"
  if [[ -n "${tmp_dir}" && -d "${tmp_dir}" ]]; then
    rm -rf "${tmp_dir}"
  fi
}
```

这样避免 source 后反复覆盖同名函数，也更符合公共 shell 工具的写法。

### 仍未解决

截至当前 fix 版本，以下问题仍然存在：

- 公共默认值复制问题仍在。
- `COAGENTIC_EXTRA_ARGS` 组合方式仍有隐患。
- task 脚本仍然偏厚，注释和实验记录边界还没有完全分离。

## 关于公共默认值与 `COAGENTIC_EXTRA_ARGS` 的深度讨论

这两个问题本质上是同一类问题：**实验配置没有一个明确、结构化、可组合的配置层**。

当前 task 脚本同时承担了三种职责：

- 声明实验差异，例如 `SAMPLE_BUILDER_REQUEST_BATCH=3`。
- 声明同类实验公共默认值，例如 `GPU_MEMORY_UTILIZATION=0.55`、`TOP_N=50`、`ENABLE_ASYNC_LABELING=1`。
- 拼接最终 Hydra override，例如 `COAGENTIC_EXTRA_ARGS=...`。

这会导致两个坏结果：

- 公共默认值在多个 task 脚本中复制，容易漂移。
- 用户想追加一个 override 时，可能不小心覆盖整段 `COAGENTIC_EXTRA_ARGS`，导致默认加速配置丢失。

### 方案 A：抽 shell common helper

这是改动最小、最务实的方案。

新增：

```bash
tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh
```

里面放：

```bash
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export AGENT_WORKERS="${AGENT_WORKERS:-4}"
export TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-4}"
export TOP_N="${TOP_N:-50}"
export TOP_M="${TOP_M:-5}"
export ENABLE_ASYNC_LABELING="${ENABLE_ASYNC_LABELING:-1}"
export ASYNC_LABELING_YAML="${ASYNC_LABELING_YAML:-.../async_labeling_deepseek_flash.yaml}"
```

再提供 helper：

```bash
car_set_default_coagentic_extra_args() { ... }
car_append_coagentic_extra_args() { ... }
car_run_training() { ... }
```

`mix_signal_fix.sh` 变成：

```bash
source "${ROOT}/tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh"

export EXP_NAME="${EXP_NAME:-CAR_async_labeling_ds_flash_mix_signal_b3_v1}"
export SAMPLE_BUILDER_REQUEST_BATCH="${SAMPLE_BUILDER_REQUEST_BATCH:-3}"

car_append_coagentic_extra_args \
  "ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"

car_run_training "$@"
```

优点：

- 改动小。
- 兼容当前 launcher。
- 很快消除 task 脚本重复。

缺点：

- 仍然是 shell 变量和字符串拼接。
- 配置结构不够强，无法像 YAML 一样自然表达层级。

适合当前阶段作为第一步。

### 方案 B：公共默认值用 YAML 管理，task 脚本只选 YAML 和少量变量

可以把同类实验公共默认值沉淀为 YAML，例如：

```bash
tasks/train_tasks/configs/car_async_labeling_deepseek_flash_base.yaml
```

内容示意：

```yaml
env:
  GPU_MEMORY_UTILIZATION: 0.55
  MAX_NUM_SEQS: 16
  AGENT_WORKERS: 4
  TOOL_MAX_CONCURRENT_PER_WORKER: 4
  LOG_PROB_MICRO_BATCH_SIZE_PER_GPU: 8
  ACTOR_MICRO_BATCH_SIZE_PER_GPU: 4
  TOP_N: 50
  TOP_M: 5
  ENABLE_ASYNC_LABELING: 1
  AUTO_START_LLM_JUDGE: 1
  AUTO_STOP_LLM_JUDGE: 1
  LLM_JUDGE_GPU_IDS: "6,7"

hydra_overrides:
  actor_rollout_ref.rollout.max_num_batched_tokens: 32768
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls: 2
  actor_rollout_ref.actor.fsdp_config.param_offload: false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload: false
  actor_rollout_ref.ref.fsdp_config.param_offload: false
  "++data.apply_chat_template_kwargs.enable_thinking": false
```

mix signal 再有一个小 YAML：

```yaml
env:
  EXP_NAME: CAR_async_labeling_ds_flash_mix_signal_b3_v1
  SAMPLE_BUILDER_REQUEST_BATCH: 3
  WAIT_FOR_GPU_RELEASE: 1

hydra_overrides:
  ranker_training.async_labeling.sample_builder_request_batch: 3
```

启动脚本可以接收：

```bash
bash train_car_task.sh \
  --task-yaml car_async_labeling_deepseek_flash_base.yaml \
  --task-yaml car_async_labeling_ds_flash_mix_signal_b3.yaml
```

优点：

- 公共默认值结构化。
- 多个 task 可以组合多个 YAML，减少复制。
- 更接近 Hydra 的配置思路。

缺点：

- 需要写一个 YAML loader，把 `env` 转成 export，把 `hydra_overrides` 转成 dotlist。
- 需要明确优先级：base YAML < experiment YAML < command line。
- shell/YAML 双系统会增加一层复杂度。

这个方案适合中期，但不建议一步到位直接替换全部脚本。

### 方案 C：全部改成 Hydra YAML 管理

理论上最干净的是：把所有训练参数都交给 Hydra，而不是 shell env。

例如：

```yaml
task:
  exp_name: CAR_async_labeling_ds_flash_mix_signal_b3_v1
  gpu:
    agent: "0,1,2,3"
    rank: 4
    recall: 5
    judge: "6,7"
  wait_for_gpu_release: true

trainer:
  ...
ranker_training:
  async_labeling:
    sample_builder_request_batch: 3
```

然后 Python/Hydra 主入口负责：

- 解析 task config。
- 启动/等待服务。
- 设置 visible devices。
- 组装 VERL 参数。
- 运行训练。

优点：

- 配置一致性最好。
- 优先级和 override 可以完全交给 Hydra。
- 更适合长期工程化。

缺点：

- 当前训练入口大量依赖 shell env、bash service launcher、VERL shell 参数拼接。
- 一步迁移成本高，容易影响已有实验。
- GPU wait、service start、日志路径、checkpoint 后处理都要重新适配。

这个方向可以作为长期目标，不适合作为当前立即修复。

### 方案 D：把 task 脚本改成 `--<name> <value>` CLI 形式

用户提出的 `--<name> <value>` 形式可以改善可读性，但它解决的是“调用接口”的问题，不是全部配置治理问题。

例如：

```bash
bash train_car_async_labeling.sh \
  --exp-name CAR_async_labeling_ds_flash_mix_signal_b3_v1 \
  --sample-builder-request-batch 3 \
  --agent-gpus 0,1,2,3 \
  --rank-gpu 4 \
  --recall-gpu 5 \
  --judge-gpus 6,7 \
  --wait-for-gpus
```

优点：

- 比 `export A=... export B=...` 更像正式命令行工具。
- 用户看启动命令就知道改了什么。
- 可以集中做参数校验和 help。

缺点：

- 需要写参数 parser。
- 参数最终仍要转成 env 或 Hydra dotlist，因为底层 launcher 现在吃的是 env 和 Hydra overrides。
- 如果参数很多，CLI 会变长；公共默认值仍然需要有来源。

因此 CLI 形式适合作为统一 wrapper 的外部接口，但不应该替代结构化配置。比较好的组合是：

```text
YAML 管默认值和实验配方
CLI 管临时覆盖
shell helper 管兼容当前 launcher
```

优先级建议：

```text
base YAML
< experiment YAML
< task script hard requirement
< CLI override
```

但对于“实验强定义变量”，例如 `mix_signal_b3` 的 `sample_builder_request_batch=3`，是否允许 CLI 覆盖要明确。如果允许覆盖，它就不再是严格的 b3 实验，而是 b3 默认值实验。

### 推荐的渐进式方案

当前最优雅且风险最低的路径不是一次性全改 YAML，也不是马上写完整 CLI，而是分三步：

1. **先抽 shell common helper。**

   目的：立刻解决公共默认值复制，并修复 `COAGENTIC_EXTRA_ARGS` 拼接语义。

2. **再把 common helper 背后的默认值迁到 YAML。**

   helper 仍保留，但它从 YAML 读默认值。这样 task 脚本不需要知道默认值细节。

3. **最后提供一个统一 CLI wrapper。**

   CLI wrapper 读取 YAML，接收 `--<name> <value>` 临时覆盖，再调用底层训练 launcher。

最终形态可以是：

```bash
bash tasks/train_tasks/run_car_task.sh \
  --recipe tasks/train_tasks/recipes/async_labeling_ds_flash_mix_signal_b3.yaml \
  --run-stamp 260618-xxxx \
  --total-steps 10
```

其中 recipe YAML 负责声明实验配方：

```yaml
extends:
  - tasks/train_tasks/recipes/async_labeling_ds_flash_base.yaml

env:
  EXP_NAME: CAR_async_labeling_ds_flash_mix_signal_b3_v1
  SAMPLE_BUILDER_REQUEST_BATCH: 3
  WAIT_FOR_GPU_RELEASE: 1

hydra_overrides:
  ranker_training.async_labeling.sample_builder_request_batch: 3
```

这个方向可以同时解决：

- 公共默认值复制。
- `COAGENTIC_EXTRA_ARGS` 字符串拼接混乱。
- task 脚本过厚。
- 实验记录和实际配置难以对齐。

### 当前下一步建议

下一步仍建议先做最小工程化修复：

```bash
tasks/train_tasks/lib/car_async_labeling_deepseek_flash_common.sh
```

在这个 helper 中先解决：

- 公共 async-labeling 默认值集中。
- `COAGENTIC_EXTRA_ARGS` 分层拼接。
- `run_car_training "$@"` 统一入口。

这样不改变底层训练入口，也不会打断当前实验节奏。等这个稳定后，再讨论 YAML recipe 和 CLI wrapper。

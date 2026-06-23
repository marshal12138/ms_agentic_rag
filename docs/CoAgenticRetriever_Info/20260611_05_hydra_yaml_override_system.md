# CoAgenticRetriever 训练脚本 YAML 配置管理系统

本文说明 CoAgenticRetriever 训练脚本如何通过外部 YAML 文件覆盖 Hydra 配置，以及如何在其它项目中复用这一套机制。

## 1. 背景

CoAgenticRetriever 的主训练入口使用 Hydra 配置：

```text
CoAgenticRetriever/main_coagentic_retriever.py
  @hydra.main(config_path="config", config_name="coagentic_retriever_trainer")
```

默认训练配置中会加载：

```yaml
defaults:
  - ranker_contrastive
```

因此默认 ranker contrastive 配置来自：

```text
CoSearch_derevitives/CoAgenticRetriever/config/ranker_contrastive.yaml
```

原先如果想临时改某个训练策略，只能在命令行里写 Hydra dotlist：

```bash
COAGENTIC_EXTRA_ARGS="++ranker_training.trajectory_selector.type=best_and_worst_f1 ++ranker_training.trajectory_selector.top_k=1 ++ranker_training.trajectory_selector.bottom_n=2" \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

这种方式可用，但不适合沉淀实验策略，因为：

- 长命令不利于复用。
- 多个参数属于同一策略时，不容易整体管理。
- 不方便在不同训练脚本或项目之间迁移。

因此新增了通用 YAML override 系统：将 partial YAML 文件转换成 Hydra dotlist，并插入训练脚本的 Hydra 参数列表。

## 2. 核心文件

通用实现位于：

```text
CoSearch_derevitives/src/hydra_overrides/
```

包含：

```text
yaml_to_dotlist.py
hydra_overrides.sh
README.md
```

### 2.1 yaml_to_dotlist.py

`yaml_to_dotlist.py` 负责把 partial YAML 展平成 Hydra dotlist。

示例输入：

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
    min_final_reward: 0.0
```

输出：

```text
++ranker_training.trajectory_selector.type=best_and_worst_f1
++ranker_training.trajectory_selector.top_k=1
++ranker_training.trajectory_selector.bottom_n=2
++ranker_training.trajectory_selector.min_final_reward=0.0
```

说明：

- 默认前缀是 `++`。
- `++key=value` 在 Hydra 中表示“存在则覆盖，不存在则创建”。
- 这比 `key=value` 更适合 partial strategy YAML，因为有些字段可能不在默认 YAML 中，例如 `top_k` 和 `bottom_n`。

命令行使用：

```bash
/data04/envs/ms/ms_cosearch_official/bin/python \
  CoSearch_derevitives/src/hydra_overrides/yaml_to_dotlist.py \
  CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml
```

### 2.2 hydra_overrides.sh

`hydra_overrides.sh` 提供 shell 函数，供训练脚本引用。

核心函数：

```bash
hydra_collect_yaml_override_files output_array "${HYDRA_OVERRIDE_YAMLS:-}" "${RANKER_STRATEGY_YAML:-}"
hydra_yaml_overrides_to_array output_array "${PY}" "${yaml_files[@]}"
```

它们的职责是：

- 收集一个或多个 YAML 文件路径。
- 调用 `yaml_to_dotlist.py`。
- 把输出安全地读入 bash array。
- 避免带空格的值被 shell 错误拆分。

## 3. 当前 CoAgenticRetriever 接入方式

当前已接入两个脚本：

```text
CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
CoSearch_derevitives/scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
```

### 3.1 外层训练脚本

外层脚本 source 通用 helper：

```bash
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"
```

它会在 `.env` 中记录：

```text
HYDRA_OVERRIDE_YAMLS=...
RANKER_STRATEGY_YAML=...
```

这样训练结束后，可以从 run env 文件确认本次训练使用了哪些外部 YAML。

外层脚本还把默认 Hydra 参数和用户参数拆开：

```bash
USER_COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-}"
DEFAULT_COAGENTIC_EXTRA_ARGS="..."

export COAGENTIC_DEFAULT_EXTRA_ARGS="${DEFAULT_COAGENTIC_EXTRA_ARGS}"
export COAGENTIC_EXTRA_ARGS="${USER_COAGENTIC_EXTRA_ARGS}"
```

这样 asset 脚本可以在“脚本默认参数”和“用户手写参数”之间插入 YAML override。

### 3.2 Hydra 启动脚本

实际执行 Hydra main 的脚本是：

```text
scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
```

它在进入 `exec "${PY}" "${COAGENTIC_MAIN}"` 前执行：

```bash
hydra_collect_yaml_override_files hydra_yaml_files \
  "${HYDRA_OVERRIDE_YAMLS:-}" \
  "${RANKER_STRATEGY_YAML:-}"

hydra_yaml_overrides_to_array hydra_yaml_args "${PY}" "${hydra_yaml_files[@]}"
```

然后在 Hydra 参数列表末尾按顺序插入：

```bash
"${coagentic_default_args[@]}" \
"${hydra_yaml_args[@]}" \
"${coagentic_extra_args[@]}" \
"$@"
```

### 3.3 CoSearch local 训练脚本

CoSearch local 训练也已接入同一套通用工具。

外层入口：

```text
CoSearch_derevitives/scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

最终执行 Hydra main 的 base 脚本：

```text
CoSearch_derevitives/scripts/cosearch_local/train_cosearch_verl_base.sh
```

外层脚本会在 run env 中记录并向 base 脚本导出：

```text
HYDRA_OVERRIDE_YAMLS=...
COSEARCH_STRATEGY_YAML=...
COSEARCH_EXTRA_ARGS=...
```

base 脚本在执行 `main_co_search_ppo.py` 前执行：

```bash
hydra_collect_yaml_override_files hydra_yaml_files \
  "${HYDRA_OVERRIDE_YAMLS}" \
  "${COSEARCH_STRATEGY_YAML}"

hydra_yaml_overrides_to_array hydra_yaml_args "${PY}" "${hydra_yaml_files[@]}"
```

然后在固定 Hydra 参数之后插入：

```bash
"${hydra_yaml_args[@]}" \
"${cosearch_extra_args[@]}" \
"$@"
```

## 4. 参数优先级

CoAgenticRetriever 训练入口的优先级从低到高为：

```text
Hydra 默认配置
< asset 脚本内固定参数
< COAGENTIC_DEFAULT_EXTRA_ARGS
< HYDRA_OVERRIDE_YAMLS / RANKER_STRATEGY_YAML
< 用户 COAGENTIC_EXTRA_ARGS
< 用户直接传给训练脚本的 "$@"
```

这个顺序很重要。

CoSearch local 训练入口的优先级从低到高为：

```text
Hydra 默认配置
< train_cosearch_verl_base.sh 内固定参数
< HYDRA_OVERRIDE_YAMLS / COSEARCH_STRATEGY_YAML
< 用户 COSEARCH_EXTRA_ARGS
< 用户直接传给训练脚本的 "$@"
```

例如 strategy YAML 写：

```yaml
ranker_training:
  trajectory_selector:
    bottom_n: 2
```

但启动命令写：

```bash
COAGENTIC_EXTRA_ARGS="++ranker_training.trajectory_selector.bottom_n=3"
```

最终生效的是：

```text
bottom_n=3
```

也就是说，YAML 文件用于沉淀实验策略；用户命令行仍然保留最高优先级，便于临时调参。

## 5. 使用方式

### 5.1 使用单个 ranker strategy YAML

当前示例 YAML：

```text
CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml
```

内容：

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
    min_final_reward: 0.0
```

训练命令：

```bash
EXP_NAME=best_worst_sampling_v1 \
RANKER_STRATEGY_YAML=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

### 5.2 使用通用 YAML override 列表

如果不想使用 `RANKER_STRATEGY_YAML` 这个业务别名，可以用通用变量：

```bash
HYDRA_OVERRIDE_YAMLS="/path/to/strategy_a.yaml /path/to/strategy_b.yaml" \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

多个 YAML 会按书写顺序展开并传给 Hydra。

如果两个 YAML 设置同一个字段，后面的 YAML 优先级更高。

### 5.3 YAML + 手写 override 混用

```bash
EXP_NAME=best_worst_sampling_v2 \
RANKER_STRATEGY_YAML=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml \
COAGENTIC_EXTRA_ARGS="++ranker_training.trajectory_selector.bottom_n=3" \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

最终：

```text
type=best_and_worst_f1
top_k=1
bottom_n=3
min_final_reward=0.0
```

### 5.4 CoSearch local 训练脚本使用 YAML override

CoSearch local 训练脚本支持通用变量 `HYDRA_OVERRIDE_YAMLS`，也支持业务别名 `COSEARCH_STRATEGY_YAML`。

示例 YAML：

```yaml
trainer:
  total_training_steps: 20
actor_rollout_ref:
  rollout:
    n: 4
```

启动命令：

```bash
EXP_NAME=cosearch_yaml_override_v1 \
COSEARCH_STRATEGY_YAML=/path/to/cosearch_strategy.yaml \
bash CoSearch_derevitives/scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

如果临时需要覆盖 YAML 中的字段：

```bash
EXP_NAME=cosearch_yaml_override_v2 \
COSEARCH_STRATEGY_YAML=/path/to/cosearch_strategy.yaml \
COSEARCH_EXTRA_ARGS="trainer.total_training_steps=30" \
bash CoSearch_derevitives/scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

如果还在命令末尾直接传入 Hydra 参数，它的优先级最高：

```bash
EXP_NAME=cosearch_yaml_override_v3 \
COSEARCH_STRATEGY_YAML=/path/to/cosearch_strategy.yaml \
COSEARCH_EXTRA_ARGS="trainer.total_training_steps=30" \
bash CoSearch_derevitives/scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh \
  actor_rollout_ref.rollout.n=8
```

## 6. 验证方式

### 6.1 只验证 YAML 展开

```bash
/data04/envs/ms/ms_cosearch_official/bin/python \
  CoSearch_derevitives/src/hydra_overrides/yaml_to_dotlist.py \
  CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml
```

期望输出：

```text
++ranker_training.trajectory_selector.type=best_and_worst_f1
++ranker_training.trajectory_selector.top_k=1
++ranker_training.trajectory_selector.bottom_n=2
++ranker_training.trajectory_selector.min_final_reward=0.0
```

### 6.2 外层脚本 dry-run

```bash
EXP_NAME=test_yaml_override \
DRY_RUN=1 \
RANKER_STRATEGY_YAML=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

检查生成的 env：

```bash
grep -n "HYDRA_OVERRIDE_YAMLS\|RANKER_STRATEGY_YAML" \
  CoSearch_derevitives/log/train_logs/coAgenticRetriever/<run_name>/<run_name>.env
```

### 6.3 Hydra 最终配置验证

可以在不启动训练的情况下使用 `--cfg job` 检查最终 Hydra 配置。

示例：

```bash
PY=/data04/envs/ms/ms_cosearch_official/bin/python \
COAGENTIC_PROJECT_ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever \
MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
TRAIN_DATA=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet \
VAL_DATA=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet \
EXP_NAME=test_cfg_yaml \
OUT_DIR=/tmp/cfg_yaml_out \
LOG_DIR=/tmp/cfg_yaml_log \
GPU_IDS=0,1 \
N_GPUS_PER_NODE=2 \
RETRIEVAL_PREFLIGHT=0 \
RANKER_STRATEGY_YAML=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh --cfg job
```

检查输出中的片段：

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
```

### 6.4 训练后验证

训练启动后，Hydra 会写：

```text
CoAgenticRetriever/outputs/<date>/<time>/.hydra/overrides.yaml
CoAgenticRetriever/outputs/<date>/<time>/.hydra/config.yaml
```

可以检查：

```bash
grep -RIn "best_and_worst_f1\|bottom_n\|top_k" \
  CoSearch_derevitives/CoAgenticRetriever/outputs/<date>/<time>/.hydra
```

### 6.5 本次实现已执行的验证

本次接入时已经执行过以下验证：

```bash
bash -n \
  CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh \
  CoSearch_derevitives/scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh \
  CoSearch_derevitives/src/hydra_overrides/hydra_overrides.sh
```

```bash
/data04/envs/ms/ms_cosearch_official/bin/python -m compileall \
  CoSearch_derevitives/src/hydra_overrides \
  CoSearch_derevitives/CoAgenticRetriever/ranker_strategies
```

```bash
EXP_NAME=test_yaml_override \
DRY_RUN=1 \
RANKER_STRATEGY_YAML=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml \
bash CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

并用 `--cfg job` 验证过：

```text
YAML 文件可以覆盖脚本默认 selector。
用户 COAGENTIC_EXTRA_ARGS 可以继续覆盖 YAML 中的字段。
```

CoSearch local 接入后额外验证过：

```bash
bash -n \
  CoSearch_derevitives/scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh \
  CoSearch_derevitives/scripts/cosearch_local/train_cosearch_verl_base.sh
```

```bash
DRY_RUN=1 \
EXP_NAME=codex_cosearch_yaml_dryrun \
HYDRA_OVERRIDE_YAMLS=/tmp/cosearch_a.yaml \
COSEARCH_STRATEGY_YAML=/tmp/cosearch_b.yaml \
COSEARCH_EXTRA_ARGS="trainer.total_training_steps=3" \
bash CoSearch_derevitives/scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

并用 `train_cosearch_verl_base.sh --cfg job` 验证过：

```text
HYDRA_OVERRIDE_YAMLS 可以覆盖 base 脚本固定参数。
COSEARCH_EXTRA_ARGS 可以继续覆盖 YAML 中的字段。
直接传给训练脚本的 Hydra 参数可以继续覆盖 COSEARCH_EXTRA_ARGS。
```

## 7. 限制与注意事项

### 7.1 只支持 partial value override

`yaml_to_dotlist.py` 不支持 Hydra config composition。

也就是说，YAML 文件中不能包含：

```yaml
defaults:
  - xxx
```

如果检测到 top-level `defaults:`，工具会直接报错。

原因是这套工具的定位是：

```text
把一个局部策略 YAML 转成命令行 override
```

不是替代 Hydra 原生 config group 机制。

### 7.2 RUN_MODE=ranker-only 暂不支持

`RUN_MODE=ranker-only` 分支调用的是独立 smoke 脚本：

```text
scripts/coagenticRetriever_local/assets/01_ranker_contrastive_smoke.py
```

它不走 Hydra main，因此 `HYDRA_OVERRIDE_YAMLS` 和 `RANKER_STRATEGY_YAML` 不会自然生效。

当前外层脚本在 ranker-only 模式下检测到这些变量会报错：

```text
ERROR: HYDRA_OVERRIDE_YAMLS/RANKER_STRATEGY_YAML are only supported in RUN_MODE=full.
```

如果未来需要支持 ranker-only，需要单独改 smoke 脚本参数或让 smoke 脚本也读取 YAML。

### 7.3 字符串中包含空格

Python converter 会输出：

```text
++a.b=hello world
```

shell helper 会用 bash array 保存整行，因此在训练脚本中不会被拆成两个参数。

不建议手写 `COAGENTIC_EXTRA_ARGS` 时直接写包含空格的复杂值；复杂结构优先放进 YAML。

### 7.4 list/dict 的处理

list 会保留成 JSON-like 字符串，例如：

```yaml
a:
  b:
    - x
    - 2
```

转换为：

```text
++a.b=["x",2]
```

嵌套 dict 会递归展平成 dotted key。

## 8. 迁移到其它项目

要在其它 Hydra 训练项目中复用这套机制，可以按以下步骤执行。

### 8.1 复制通用工具

复制：

```text
src/hydra_overrides/yaml_to_dotlist.py
src/hydra_overrides/hydra_overrides.sh
```

到目标项目的 `src/hydra_overrides/`。

目标 Python 环境需要安装：

```text
omegaconf
```

通常使用 Hydra 的项目已经包含它。

### 8.2 在训练脚本中 source helper

```bash
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"
```

### 8.3 收集 YAML override 文件

建议统一支持通用环境变量：

```bash
HYDRA_OVERRIDE_YAMLS="/path/a.yaml /path/b.yaml"
```

如果有业务专用策略，也可以加别名：

```bash
RANKER_STRATEGY_YAML=/path/ranker_strategy.yaml
POLICY_STRATEGY_YAML=/path/policy_strategy.yaml
```

脚本中：

```bash
hydra_collect_yaml_override_files hydra_yaml_files \
  "${HYDRA_OVERRIDE_YAMLS:-}" \
  "${RANKER_STRATEGY_YAML:-}" \
  "${POLICY_STRATEGY_YAML:-}"
```

### 8.4 转成 Hydra 参数数组

```bash
hydra_yaml_overrides_to_array hydra_yaml_args "${PY}" "${hydra_yaml_files[@]}"
```

### 8.5 插入 Hydra 命令行

推荐顺序：

```bash
exec "${PY}" "${MAIN}" \
  ...固定参数... \
  "${script_default_args[@]}" \
  "${hydra_yaml_args[@]}" \
  "${user_extra_args[@]}" \
  "$@"
```

这样可以保证：

```text
脚本默认值 < YAML 策略文件 < 用户临时参数
```

## 9. 推荐目录组织

建议每个训练入口维护自己的策略 YAML 目录，例如：

```text
scripts/coagenticRetriever_local/strategies_yaml/
  ranker_contrastive_new_sampling.yaml
  ranker_contrastive_long_doc.yaml
  ranker_contrastive_no_replay.yaml
```

策略 YAML 只写和该策略相关的局部配置，不复制完整 base config。

推荐：

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
```

不推荐：

```yaml
trainer:
  ...
recall_retriever:
  ...
ranker:
  ...
ranker_training:
  ...
```

原因是完整复制 base config 容易和上游默认配置漂移，后续维护成本高。

## 10. 当前示例：best_and_worst_f1

当前新增的 ranker trajectory selector 是：

```text
CoAgenticRetriever/ranker_strategies/trajectory_selector/best_and_worst_f1.py
```

对应策略 YAML：

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
    min_final_reward: 0.0
```

它会选择：

```text
score 最高的 top_k 条轨迹
score 最低的 bottom_n 条轨迹
```

并将这些轨迹中的 search tool contexts 交给后续：

```text
signal_builder -> sample_builder -> replay_buffer -> ranker contrastive update
```

如果想让 worst 包含负 reward 轨迹，需要降低：

```yaml
min_final_reward: -1.0
```

当前示例使用 `0.0`，因此只在非负 reward 的合法轨迹中选择 best/worst。

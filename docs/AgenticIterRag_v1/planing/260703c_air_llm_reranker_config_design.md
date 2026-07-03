# AIR LLM Reranker GRPO 配置管理详细设计

更新日期：2026-07-03

## 1. 目标

这篇文档专门拆解 LLM reranker GRPO 训练的配置管理。

简单说，这一层要解决的问题是：

- 训练入口应该怎么声明一次实验。
- 新增哪些 YAML 配置组。
- pipeline 里新增哪些 stage。
- resource YAML 怎么描述 reranker 训练需要的服务和 GPU。
- compiler 要校验什么。
- dry-run 要落哪些审计文件。

AIR 现有设计有一个很重要的原则：业务参数必须来自 YAML 或 CLI dotlist，不能散落在 shell 里。这个原则在 LLM reranker 训练里继续保持。

## 2. 非目标

本配置设计不做这些事：

- 不实现真正的 GRPO trainer 逻辑。
- 不实现 continuation rollout。
- 不实现 service bundle 生成。
- 不规定某台机器必须用哪几张 GPU。
- 不允许 shell 入口私自维护模型路径、数据路径、topN/topM、端口或 batch size。

这些都由后续模块设计或实际运行 overlay 决定。

## 3. 配置文件清单

建议新增或调整这些文件：

```text
tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh
tasks/train_tasks/agenticIterRag/configs/llm_reranker_training_overlay.yaml

AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml
AgenticIterRag/config/main_run/agentic_iter_rag_main.yaml
AgenticIterRag/config/pipeline/offline_two_stage.yaml
AgenticIterRag/config/resource/local_8gpu_0_7.yaml

scripts/agenticIterRag_v1/assets/compile_config.py
scripts/agenticIterRag_v1/assets/run_pipeline.py
```

实现时所有新增 YAML 字段都要补充中文注释，注释风格参考：

- `AgenticIterRag/config/pipeline/offline_two_stage.yaml`
- `AgenticIterRag/config/reranker_training/llm_reranker_base.yaml`
- `tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh`

新增 Python 和 shell 代码也要补充足够中文注释。注释重点不是解释语法，而是说明配置来源、字段职责、stage 边界、为什么不允许 shell-only 业务配置。

## 4. 输入和输出

配置管理模块的输入是几类配置事实：

- task shell 选择的配置组。
- `agentic_iter_rag_main.yaml` 里的默认配置组和运行目录。
- `llm_reranker_grpo_branch.yaml` 里的 reranker 训练基础配置。
- `llm_reranker_training_overlay.yaml` 里的本次实验差异。
- pipeline YAML 里的 stage 编排。
- resource YAML 里的 stage 资源计划。
- CLI dotlist 里的少量临时覆盖。

配置管理模块的输出是 runner 和各 stage 可以直接消费的最终配置：

- `pipeline.final_config.yaml`
- `pipeline.final_config.json`
- `execution_plan.yaml`
- `pipeline.env`
- `pipeline.args.txt`
- stage 级 dry-run manifest

这里的关键点是：shell 只负责选配置组，最终业务参数必须能在 final config 里被审计到。

这里还要明确一个主次关系：最核心的 YAML 仍然是 `AgenticIterRag/config/main_run/agentic_iter_rag_main.yaml`。新增的 `llm_reranker_grpo_branch.yaml` 是 `reranker_training` 配置组里的一个新成员，必须挂到 main YAML 的 `config_groups.reranker_training` 体系中，而不是绕开 main YAML 单独运行。

## 5. 训练入口配置

新增训练入口：

```text
tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh
```

这个入口只做配置组选型：

```bash
#!/usr/bin/env bash
set -euo pipefail

# 本任务是 AIR v1 的 LLM reranker GRPO 训练入口。
# 它只声明本次实验选择哪些配置组和 overlay。
# 模型路径、数据路径、topN/topM、端口、batch size 和训练超参必须写在 YAML 中。

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

cd "${ROOT}"

bash "${ROOT}/scripts/agenticIterRag_v1/01_pipeline_launcher.sh" \
  --main-run-config agentic_iter_rag_main \
  --DATA_CONFIG=co_search_ablation \
  --PIPELINE_CONFIG=offline_two_stage \
  --RESOURCE_CONFIG=local_8gpu_0_7 \
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm \
  --INFER_BUDGET_CONFIG=air_aligned_budget \
  --RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=air_async_qwen3_4b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/llm_reranker_training_overlay.yaml \
  "$@"
```

这个 shell 里不要出现：

- `DATA_PATH=...`
- `MODEL_PATH=...`
- `BATCH_SIZE=...`
- `TOP_N=...`
- `PORT=...`
- `REWARD_STRATEGY=...`

需要临时覆盖时，用 CLI dotlist：

```bash
bash tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh \
  --reranker_training.branch_dataset.step_policy=first_point \
  --reranker_training.trainer.train_batch_size=8
```

## 6. Main Run YAML 挂载设计

最核心的顶层配置文件是：

```text
AgenticIterRag/config/main_run/agentic_iter_rag_main.yaml
```

当前 main YAML 里已经有：

```yaml
config_groups:
  # LLM reranker 训练配置组。
  reranker_training: llm_reranker_base
```

实现本方案时要把新的 reranker 训练配置纳入这个配置组体系。推荐方式是：

```yaml
config_groups:
  # LLM reranker 训练配置组。
  # 默认基础配置仍可保留 llm_reranker_base；训练入口通过 --RERANKER_TRAINING_CONFIG 覆盖为 llm_reranker_grpo_branch。
  reranker_training: llm_reranker_base
```

训练入口选择：

```bash
--RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch
```

compiler 的职责是把这个 CLI 选择写回 final config：

```yaml
main_run:
  config_groups:
    reranker_training: llm_reranker_grpo_branch
```

如果后续希望这条训练任务成为 main YAML 的默认 reranker 训练链路，也可以直接把 main YAML 默认值改成：

```yaml
config_groups:
  reranker_training: llm_reranker_grpo_branch
```

但第一版更推荐通过训练入口覆盖。这样不会影响现有静态 reranker dataset 相关任务。

实现时要补中文注释，说明 `llm_reranker_grpo_branch` 是挂在 `main_run.config_groups.reranker_training` 下的配置组，不是绕过 main run 的独立配置。

## 7. Reranker Training YAML 草案

新增：

```text
AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml
```

完整草案如下。实际落地时要保留中文注释，不要只写裸字段。

```yaml
# AIR v1 LLM reranker GRPO 训练配置。
# 这份配置只描述 reranker 训练本身，不描述 agent 训练，也不描述普通静态 reranker dataset 构造。

# 配置名称；task 通过 --RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch 选择它。
name: llm_reranker_grpo_branch

# LLM reranker 的基座模型路径。第一版默认使用 Qwen3-4B。
base_model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B

# 输入数据配置。
input:
  # 增强轨迹 manifest 路径。为空时可由上游 generate_traces stage 输出填充。
  enhanced_trajectory_manifest: null

  # 已经构造好的 branch dataset manifest。非空时可跳过 build_reranker_branch_dataset。
  branch_dataset_manifest: null

# branch dataset 构造配置。
branch_dataset:
  # 是否启用 branch dataset 构造。
  enabled: true

  # branch dataset 版本名；为空时 runtime 根据增强轨迹版本、step policy 和 prompt 版本自动生成。
  version: null

  # 输出目录已存在时是否允许覆盖。默认 false，避免误删长期数据。
  overwrite: false

  # search step 选择策略：first_point=第一步，end_point=最后一步，random_point=固定 seed 随机一步。
  # 配置值使用 snake_case，不使用带空格字符串，方便 CLI dotlist、manifest 和代码枚举校验。
  # 第一版默认 first_point，优先保证训练链路稳定、可解释。
  step_policy: first_point

  # random_point 策略的随机种子。必须写入 manifest，保证数据可复现。
  random_seed: 20260703

  # 候选文档数量。第一版要求完整 top50 排序。
  candidate_top_n: 50

  # agent 实际可见文档数量。无论 reranker 输出多少，agent observation 只取 top5。
  visible_top_m: 5

  # reranker prompt 模板版本。full50 表示要求模型输出 50 个候选的完整排序。
  prompt_template_version: air_rerank_tags_v1_full50

  # 训练数据 formatter；verl_chat 表示输出 VERL chat message 格式。
  formatter: verl_chat

  # 每篇候选文档写入 reranker prompt 的最大字符数。
  max_doc_chars: 2000

  # 没有 search step 的轨迹是否允许跳过。第一版默认 false，便于尽早发现数据异常。
  allow_no_search: false

# continuation rollout 配置。
continuation:
  # frozen search agent 模型路径；为空时可从 infer_runtime.models.trained_agent_model 继承。
  agent_model: null

  # reranker 训练阶段固定 search agent，不更新 agent 参数。
  use_frozen_agent: true

  # continuation 后续 search 的工具模式。第一版必须是 retriever_only。
  search_tool_mode: retriever_only

  # continuation 最大 assistant turn 数。
  max_assistant_turns: 6

  # continuation 最大 tool/user observation turn 数。
  max_user_turns: 6

  # frozen agent prompt 最大 token 长度。
  max_prompt_length: 11264

  # frozen agent 单次 response 最大 token 长度。
  max_response_length: 1024

  # tool observation 最大 token 长度，超过后按 AIR 当前工具格式截断。
  max_tool_response_length: 4096

  # frozen agent 推理温度。
  temperature: 0.0

  # frozen agent 推理 top-p。
  top_p: 1.0

# reranker reward 配置。
reward:
  # reward 策略：answer_reward=直接使用新答案分数；delta_answer_reward=新分数减 baseline。
  strategy: answer_reward

  # reranker 输出格式错误时的直接惩罚。格式错时不触发 frozen agent continuation。
  format_penalty: -0.5

  # agent answer reward 函数。默认复用 search agent 训练时的 QA F1 + format penalty 系列。
  answer_reward_function:
    path: AgenticIterRag/rewards/search_qa_f1_with_format_penalty.py
    name: search_qa_f1_penalty_compute_score

  # 使用 delta_answer_reward 时是否强制要求 baseline_reward 存在。
  require_baseline_reward_for_delta: true

# GRPO trainer 超参。
trainer:
  # 训练方法。第一版只支持 grpo。
  method: grpo

  # 训练 epoch 数。
  total_epochs: 1

  # 全局训练 batch size。
  train_batch_size: 16

  # 每卡 micro batch size。
  micro_batch_size_per_gpu: 1

  # 每个 prompt 采样多少个 reranker 输出，用于 GRPO 组内比较。
  n_samples_per_prompt: 4

  # 学习率。
  learning_rate: 2.0e-5

  # reranker prompt 最大 token 长度。
  max_prompt_length: 12000

  # reranker response 最大 token 长度。
  max_response_length: 2048

  # checkpoint 保存间隔。
  save_freq: 100

  # 日志后端。
  logger:
    - console
    - file

# runtime 字段由 compiler 或 runner 填充，不应在 task shell 中维护。
runtime:
  # stage manifest 路径。
  manifest_path: null

  # 训练输出目录。
  output_dir: null

  # 训练完成后的 service bundle 目录。
  service_bundle_dir: null

  # 是否 dry-run。
  dry_run: false
```

## 8. Training Overlay 草案

新增：

```text
tasks/train_tasks/agenticIterRag/configs/llm_reranker_training_overlay.yaml
```

职责是选择本次实验的执行范围和默认输入。

```yaml
# AIR v1 LLM reranker GRPO 训练任务 overlay。
# 本 overlay 只表达本次实验与基础配置的差异。

main_run:
  project:
    # 本次 reranker 训练任务的默认实验名。
    experiment_name: agentic_iter_rag_v1_llm_reranker_training_260703a

pipeline:
  # 训练入口默认从 branch dataset 构造开始。
  resume_from_stage: build_reranker_branch_dataset

  # 默认执行到 service bundle 产出。
  stop_after_stage: build_service_bundle

  # 默认不跳过所选范围内的 stage。
  skip_stages: []

  stage_configs:
    build_reranker_branch_dataset:
      enabled: true

    train_llm_reranker:
      enabled: true

    build_service_bundle:
      enabled: true

reranker_training:
  input:
    # 第一版要求显式指向增强轨迹 manifest，或由上游 stage 填充。
    enhanced_trajectory_manifest: null

  branch_dataset:
    # 默认替换第一步 search，先保证训练链路稳定可解释。
    step_policy: first_point
    random_seed: 20260703

  reward:
    # 默认不扣 baseline，先让训练信号更稳定。
    strategy: answer_reward
```

## 9. Pipeline 配置扩展

`AgenticIterRag/config/pipeline/offline_two_stage.yaml` 增加三个 stage：

```yaml
stages:
  - train_agent
  - generate_traces
  - build_reranker_dataset
  - build_reranker_branch_dataset
  - train_llm_reranker
  - build_service_bundle
  - infer_matrix
```

新增 stage configs：

```yaml
stage_configs:
  build_reranker_branch_dataset:
    # 是否启用 branch dataset 构造。
    enabled: true

    # 数据构造默认使用 CPU/local 资源。
    resource_key: local_cpu

    # stage 输入字段。
    inputs:
      # 增强轨迹 manifest，默认来自 generate_traces 或手动配置。
      enhanced_trajectory_manifest: null

    # stage 输出字段。
    outputs:
      # branch dataset manifest 路径。
      branch_dataset_manifest: null

      # stage manifest 路径。
      manifest: null

  train_llm_reranker:
    # 是否启用 LLM reranker GRPO 训练。
    enabled: true

    # stage 使用的资源配置键。
    resource_key: train_llm_reranker

    inputs:
      # branch dataset manifest，默认来自 build_reranker_branch_dataset。
      branch_dataset_manifest: null

    outputs:
      # 训练后的 reranker 模型路径。
      reranker_model: null

      # 训练 stage manifest。
      manifest: null

  build_service_bundle:
    # 是否启用服务配置组装。
    enabled: true

    # service bundle 构造只需要本地 CPU。
    resource_key: local_cpu

    inputs:
      # 训练后的 reranker 模型路径。
      reranker_model: null

    outputs:
      # service bundle 目录。
      service_bundle_dir: null

      # service bundle manifest。
      manifest: null
```

## 10. Resource 配置扩展

`AgenticIterRag/config/resource/local_8gpu_0_7.yaml` 需要补 stage-level placement。

示例：

```yaml
stage_resources:
  build_reranker_branch_dataset:
    # branch dataset 构造不启动模型服务，只使用本地 CPU。
    local_cpu:
      workers: 1

  train_llm_reranker:
    services:
      # 训练中的 reranker actor/rollout 模型。
      reranker_actor:
        gpu_ids: [0, 1, 2, 3]
        tensor_parallel_size: 4
        port: 8240

      # frozen search agent，用于 continuation rollout。
      # 首选方案：agent 使用 3 张卡，recall 独占剩余 1 张卡，避免资源重叠。
      frozen_agent_vllm:
        gpu_ids: [4, 5, 6]
        tensor_parallel_size: 3
        port: 8140

      # retriever 服务。后续 search 只能走 retriever。
      # 默认不与 frozen_agent_vllm 共享 GPU。
      recall:
        gpu_ids: [7]
        port: 8130
        retrieval_service_url: http://127.0.0.1:8130/retrieve

  build_service_bundle:
    # service bundle 只写配置文件和 manifest。
    local_cpu:
      workers: 1
```

默认不允许 `frozen_agent_vllm` 和 `recall` 共享 GPU。resource validator 要把 GPU overlap 当成错误。

如果 frozen agent 后端不支持 3 卡 TP，使用备选资源规划：

```yaml
stage_resources:
  train_llm_reranker:
    services:
      reranker_actor:
        # reranker actor 仍使用 0-3。
        gpu_ids: [0, 1, 2, 3]
        tensor_parallel_size: 4
        port: 8240

      frozen_agent_vllm:
        # 备选方案：agent 使用 4-5 两张卡。
        gpu_ids: [4, 5]
        tensor_parallel_size: 2
        port: 8140

      recall:
        # retriever 使用 6-7 两张卡，仍然不与 agent overlap。
        gpu_ids: [6, 7]
        port: 8130
        retrieval_service_url: http://127.0.0.1:8130/retrieve
```

也就是说，默认设计宁可降低 frozen agent TP，或给 retriever 两张卡，也不要让 retriever 和 agent 挤在同一张 GPU 上。

## 11. 配置或接口边界

配置接口分三层。

第一层是 task shell 参数：

- `--RERANKER_TRAINING_CONFIG`
- `--main-run-config`
- `--PIPELINE_CONFIG`
- `--RESOURCE_CONFIG`
- `--OVERLAY_YAML`
- CLI dotlist 覆盖

第二层是 final config 中的稳定字段：

- `reranker_training.input`
- `reranker_training.branch_dataset`
- `reranker_training.continuation`
- `reranker_training.reward`
- `reranker_training.trainer`
- `pipeline.stage_configs`
- `resource.stage_resources`

第三层是 stage manifest：

- `build_reranker_branch_dataset/manifest.json`
- `train_llm_reranker/manifest.json`
- `build_service_bundle/manifest.json`

实现时要把三层边界写清楚。shell 不能越过 final config 直接把业务参数塞给 Python；stage Python 也不能自己猜测上游输出路径。

## 12. 执行流程

配置管理的执行流程是：

1. task shell 选择配置组和 overlay。
2. launcher 合并 base config、group config、overlay 和 CLI dotlist。
3. compiler 做必填字段、非法值、资源计划校验。
4. runner 根据 `resume_from_stage` 和 `stop_after_stage` 生成 stage 列表。
5. runner 写 `execution_plan.yaml` 和 final config。
6. dry-run 模式只写审计文件，不启动模型服务，也不构造长期数据。
7. 非 dry-run 模式按 stage 顺序把 final config 传给具体 stage。

代码实现计划里要补中文注释，尤其说明“为什么 shell 不能维护业务参数”和“为什么 dry-run 不应该产生真实训练副作用”。

## 13. Compiler 校验

`compile_config.py` 需要新增校验。

必填字段：

- `reranker_training.base_model`
- `reranker_training.branch_dataset.step_policy`
- `reranker_training.branch_dataset.candidate_top_n`
- `reranker_training.branch_dataset.visible_top_m`
- `reranker_training.continuation.search_tool_mode`
- `reranker_training.reward.strategy`
- `reranker_training.reward.format_penalty`
- `reranker_training.trainer.method`
- `main_run.config_groups.reranker_training`
- `resource.stage_resources.build_reranker_branch_dataset`
- `resource.stage_resources.train_llm_reranker`
- `resource.stage_resources.build_service_bundle`

禁止字段或非法值：

- `reranker_training.continuation.use_frozen_agent=false`
- `reranker_training.continuation.search_tool_mode != retriever_only`
- `reranker_training.branch_dataset.step_policy=all_steps`
- `reranker_training.branch_dataset.step_policy` 不属于 `first_point/end_point/random_point/all_steps`
- `reranker_training.branch_dataset.visible_top_m > candidate_top_n`
- `reranker_training.reward.strategy=delta_answer_reward` 但 baseline 不可用。
- `resource.stage_resources.train_llm_reranker.services.frozen_agent_vllm.gpu_ids` 和 `resource.stage_resources.train_llm_reranker.services.recall.gpu_ids` 有交集。

错误信息要直接说明配置路径，例如：

```text
unsupported reranker_training.branch_dataset.step_policy=all_steps in AIR v1 single-step training
```

## 14. 错误处理

配置阶段遇到以下情况直接失败：

- 选择了不存在的 `RERANKER_TRAINING_CONFIG`。
- `RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch` 但没有写入 `main_run.config_groups.reranker_training`。
- overlay 文件不存在或 YAML 解析失败。
- 必填路径缺失。
- `visible_top_m > candidate_top_n`。
- `continuation.search_tool_mode` 不是 `retriever_only`。
- `branch_dataset.step_policy=all_steps`。
- stage resource 缺少 `train_llm_reranker`。
- frozen agent 和 recall GPU 资源重叠。
- CLI dotlist 覆盖了不允许覆盖的 shell-only 业务字段。

错误信息要包含配置路径，方便定位，例如 `reranker_training.branch_dataset.visible_top_m`。

## 15. Dry-run 审计

dry-run 后必须能看到：

```text
pipeline.final_config.yaml
pipeline.final_config.json
pipeline.env
pipeline.args.txt
pipeline.manifest.json
execution_plan.yaml
stages/build_reranker_branch_dataset/manifest.json
stages/train_llm_reranker/manifest.json
stages/build_service_bundle/manifest.json
```

`execution_plan.yaml` 要包含：

- selected stages
- stage manifests
- stage resource plan
- reranker training config 摘要
- `main_run.config_groups.reranker_training`
- branch dataset 输出预期路径
- service bundle 输出预期路径

dry-run 不启动模型服务，不写长期数据集，只写审计 manifest。

## 16. 实现计划

代码实现时按这个顺序：

1. 在 `agentic_iter_rag_main.yaml` 的 `config_groups.reranker_training` 体系下挂载 `llm_reranker_grpo_branch`。
2. 新增 `llm_reranker_grpo_branch.yaml`，字段先完整写齐并补中文注释。
3. 新增 `llm_reranker_training_overlay.yaml`，只写实验差异。
4. 新增训练 task shell，保持只选配置组。
5. 扩展 pipeline stage 列表和 stage configs。
6. 扩展 resource stage_resources，默认避免 agent 和 recall GPU overlap。
7. 扩展 compiler required path、非法值校验和 GPU overlap 校验。
8. 扩展 runner dry-run manifest。

中文注释要求：

- YAML 中每个新增字段都要有中文注释。
- main YAML 挂载处要写中文注释，说明 reranker training config group 的默认值和训练入口覆盖关系。
- shell 入口要说明“只做配置组选型，不承载业务参数”。
- compiler 校验代码要在复杂校验前用中文注释说明校验目的。
- runner 写 manifest 的地方要注释 stage 输出和下游依赖关系。

## 17. 测试计划

### 17.1 配置编译测试

运行：

```bash
bash tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh --dry-run
```

期望：

- final config 存在。
- execution plan 存在。
- `main_run.config_groups.reranker_training=llm_reranker_grpo_branch`。
- selected stages 为 `build_reranker_branch_dataset -> train_llm_reranker -> build_service_bundle`。
- stage_resource_plan 包含三个 stage。

### 17.2 CLI dotlist 覆盖测试

运行：

```bash
bash tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh \
  --dry-run \
  --reranker_training.branch_dataset.step_policy=end_point
```

期望 final config 中 `step_policy=end_point`。

### 17.3 非法配置测试

覆盖：

```bash
--reranker_training.branch_dataset.step_policy=all_steps
```

期望 compiler 失败，并提示该模式第一版不支持。

### 17.4 GPU overlap 负向测试

把 `frozen_agent_vllm.gpu_ids` 配成 `[4, 5, 6, 7]`，同时 `recall.gpu_ids` 配成 `[7]`。

期望 compiler 失败，并提示 frozen agent 和 recall 不能共享 GPU。

### 17.5 Shell-only 配置测试

运行：

```bash
DATA_PATH=/tmp/x bash tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh --dry-run
```

期望 compiler 拒绝 shell-only 业务配置。

### 17.6 dry-run 小样本 smoke

运行 dry-run，并给一个小样本增强轨迹 manifest。

期望：

- `execution_plan.yaml` 能看到三个 stage。
- 每个 stage 的 manifest 写出预期输入输出。
- 不创建真实 branch dataset。
- 不启动 frozen agent 或 reranker actor。

### 17.7 注释验收

人工检查新增 YAML 和 shell：

- 字段旁边有中文注释。
- 注释解释业务含义，而不是只复述字段名。
- 注释风格与现有 AIR 配置一致。

# SPAD-RAG Implementation Status Prework

记录时间：2026-07-09

本文件用于记录 AIR 中 SPAD-RAG 训练方法的当前实现现状。它不是新的算法设计，也不替代已有 planning 文档；它的作用是把已经落到代码、配置、运行目录和验证记录里的状态整理清楚，方便后续恢复工作、继续消融或交接实现。

## 1. 文档目的

SPAD-RAG 当前已经从方案设计推进到 AIR `train_agent` 内部实现阶段。为了避免后续执行时混淆“设计目标”和“已经实现的工程状态”，这里统一记录：

1. SPAD-RAG 的核心思路。
2. 它依赖的三篇 planning 文档。
3. 当前 AIR 中的配置、代码、资源和运行目录结构。
4. 三个 sub-stage 的实现状态和边界。
5. 已经验证过的 dry-run、Stage 1 训练和单元测试情况。
6. 仍需后续补齐的工作。

本文档只描述当前状态，不修改三篇 planning 文档中的设计结论。

## 2. 设计来源和依赖文档

SPAD-RAG 的实现必须以以下三篇 planning 文档为基准：

```text
docs/AgenticIterRag_v1/planing/260708e_spad_rag_search_policy_rl_answer_distillation_draft.md
docs/AgenticIterRag_v1/planing/260708f_spad_rag_train_agent_detailed_design_draft.md
docs/AgenticIterRag_v1/planing/260709a_spad_rag_qwen17_glm47_execution_plan.md
```

三篇文档的分工如下：

1. `260708e_spad_rag_search_policy_rl_answer_distillation_draft.md`
   - 算法思路基准。
   - 定义 SPAD-RAG 的三阶段训练逻辑。
   - 明确 Stage 1 teacher answerer 是 reward function 的一部分。
   - 明确 Stage 2 需要用训练后的 actor 重新 rollout，生产 on-policy answer context。
   - 明确 Stage 3 用 answer distillation 把 teacher answer 能力训回 actor。

2. `260708f_spad_rag_train_agent_detailed_design_draft.md`
   - 工程详细设计基准。
   - 定义 SPAD-RAG 作为 AIR `train_agent` stage 的一种 implementation 接入。
   - 明确不新增顶层 `spad_rag_three_stage` pipeline。
   - 定义配置层级、resource 层级、prompt/reward/service/data/checkpoint 的拆分方式。

3. `260709a_spad_rag_qwen17_glm47_execution_plan.md`
   - 第一版实现和执行计划基准。
   - 第一版 actor 是 Qwen3-1.7B。
   - teacher answerer 是本地 GLM-4.7-Flash vLLM 服务。
   - 先跑通、再消融效率、再消融 Stage 1 训练效果，最后进入正式训练。

如果执行中发现三篇文档和当前实现不一致，优先以 `260708f` 的工程设计为准；如果是执行策略细节，则以 `260709a` 为准。当前本文档只记录差异和实现状态，不直接改动原 planning 文档。

## 3. SPAD-RAG 方法简述

SPAD-RAG 全称：

```text
Search-Policy RL with Answer Distillation for Agentic RAG
```

核心思想是把 agentic RAG 的两个能力拆开训练：

```text
Search policy:
  搜什么、搜几轮、什么时候停止

Answer ability:
  基于已经搜到的证据生成最终短答案
```

当前 AIR 实现采用三个 sub-stage：

```text
Stage 1: search_policy_rl
  用 RL 训练 actor 的搜索/停止策略。
  actor 只决定 search 或 stop，不在 Stage 1 训练 answer body。
  teacher answerer 在 reward function 内部基于 search evidence 生成答案和 evidence status。

Stage 2: answer_refresh_data
  冻结 Stage 1 actor，重新对训练数据 rollout。
  不再在 <answer> 处停止，让 actor 生成完整 answer，作为 rejected。
  teacher 基于同一条 search evidence 生成 chosen。
  输出 answer distillation pair 数据。

Stage 3: answer_distillation
  用 Stage 2 的 chosen/rejected pair 训练 actor 的 answer ability。
  SFT phase 默认关闭，保留接口。
  DPO phase 默认开启，当前可执行后端是 local_dpo。
```

这个拆分解决的是 credit assignment 问题：小 actor 的 answer ability 不强时，不能直接用 actor 自己的最终答案 F1 来判断 search policy 好坏。Stage 1 用强 teacher 只评估“搜索证据是否足够”，Stage 3 再把 teacher answer 能力训回 actor。

## 4. AIR 中的实现形态

SPAD-RAG 当前不是一个新的 AIR 顶层 pipeline，而是 `train_agent` 的一种实现。

当前接入关系：

```text
offline_two_stage.yaml
  train_agent
    impl: spad_rag
    impl_config_ref: agent_training

agent_training/spad_rag_base.yaml
  SPAD-RAG 内部 sub_stage_order
  Stage 1/2/3 配置
  reward 配置
  teacher profile
  prompt / rollout / distillation 参数

resource/local_8gpu_0_7.yaml
  resource.stage_resources.train_agent.impls.spad_rag.sub_stages.*
  按 SPAD sub-stage 管理 actor、teacher、recall、DPO trainer 资源
```

当前代码入口：

```text
AgenticIterRag/agentic_iter_rag/agent_training/train_agent_entry.py
AgenticIterRag/agentic_iter_rag/agent_training/registry.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/orchestrator.py
```

`train_agent_entry.py` 从 compiled config 中读取：

```text
pipeline.stage_configs.train_agent.impl
pipeline.stage_configs.train_agent.impl_config_ref
```

当 `impl=spad_rag` 时，通过 registry 分发到：

```text
agentic_iter_rag.agent_training.spad.orchestrator.run_spad_rag
```

SPAD orchestrator 负责根据：

```text
sub_stage_order
resume_from_sub_stage
stop_after_sub_stage
skip_sub_stages
force_rerun_sub_stages
```

选择并依次执行：

```text
search_policy_rl
answer_refresh_data
answer_distillation
```

最终对外仍只暴露 AIR 下游需要的统一字段：

```text
train_agent.outputs.agent_checkpoint
train_agent.outputs.agent_training_manifest
```

## 5. 三个 sub-stage 的实现现状

### 5.1 Stage 1: search_policy_rl

当前状态：已接入 VERL，可真实训练。

主要代码：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/search_policy_rl.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py
AgenticIterRag/verl/verl/experimental/agent_loop/spad_search_policy_agent_loop.py
```

当前实现要点：

1. 支持 `backend=smoke` 和 `backend=verl`。
2. `backend=verl` 会生成 SPAD 专用 VERL command plan 和 launch script。
3. actor 使用 Qwen3-1.7B，正式 overlay 选择：

```text
/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B
```

4. teacher answerer 使用 GLM-4.7-Flash 本地 vLLM service。
5. recall service 默认使用 NPU/GPU accelerator backend，不默认用 CPU。
6. Stage 1 rollout stop sequences 当前为：

```text
</tool_call>
<answer>
```

7. `</tool_call>` 用作 action boundary，避免完整 tool call 后继续空跑。
8. `<answer>` 表示 actor 选择 stop；Stage 1 在 opening `<answer>` 停止，不训练 answer body。
9. 单个 assistant turn 的生成上限通过 `COSEARCH_TURN_MAX_TOKENS` 控制，当前默认 `512`。
10. 整条 trajectory response budget 当前已经与 CoSearch/CAR 对齐：

```text
max_response_length = 4096
max_assistant_turns = 6
max_user_turns = 6
max_tool_response_length = 4096
rollout_max_model_len = 16096
max_search_turns = 5
```

Stage 1 产物包括：

```text
outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/
outputs/stages/train_agent/spad_rag/search_policy_rl/validation_data/
outputs/stages/train_agent/spad_rag/search_policy_rl/manifest.json
runtime_logs/stages/train_agent/spad_rag/search_policy_rl/verl_train.log
checkpoints/AIR/<RUN>/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/
```

### 5.2 Stage 2: answer_refresh_data

当前状态：已实现 smoke 和 rollout 两个后端。

主要代码：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/refresh_rollout.py
```

当前实现要点：

1. `backend=smoke` 用于验证 Stage 3 数据契约。
2. `backend=rollout` 会使用 Stage 1 actor checkpoint 重新 rollout。
3. 如果 Stage 1 checkpoint 是 VERL FSDP shard，会先 merge 成 HF checkpoint，供 actor vLLM 使用。
4. Stage 2 不在 `<answer>` 处 stop，actor 会继续生成 answer。
5. actor answer 写为 DPO `rejected`。
6. teacher answerer 只看 search evidence，不看 actor answer。
7. Stage 2 teacher 也使用带 `<status>` 的 teacher prompt；写入 DPO chosen 时会剔除 `<status>` block，避免 teacher-only evidence status 进入 actor answer 训练目标。
8. 支持过滤条件：

```text
require_teacher_format_valid
require_evidence_sufficient
min_teacher_f1
```

Stage 2 数据输出 schema 目前为：

```text
spad_answer_distill_pair_v1
```

每条样本核心字段：

```text
prompt
messages_before_final_answer
chosen
rejected
metadata.question
metadata.gold_answers
metadata.actor_answer
metadata.teacher_answer
metadata.teacher_f1
metadata.teacher_evidence_status
metadata.search_count
metadata.sub_queries
metadata.visible_top5_docs
```

Stage 2 产物包括：

```text
outputs/stages/train_agent/spad_rag/answer_refresh_data/answer_distill_pairs.jsonl
outputs/stages/train_agent/spad_rag/answer_refresh_data/answer_distill_dataset_manifest.json
outputs/stages/train_agent/spad_rag/answer_refresh_data/manifest.json
runtime_logs/stages/train_agent/spad_rag/answer_refresh_data/
checkpoints/AIR/<RUN>/stages/train_agent/spad_rag/answer_refresh_data/actor_model_hf/
```

### 5.3 Stage 3: answer_distillation

当前状态：已有 smoke、local_dpo、VERL dry-run plan。

主要代码：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/answer_distillation.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/local_dpo.py
```

当前实现要点：

1. Stage 3 内部包含两个 phase：

```text
sft
dpo
```

2. SFT 默认关闭，仅保留接口。
3. DPO 默认开启。
4. 当前可执行 DPO 后端是：

```text
backend = local_dpo
```

5. `local_dpo` 是 AIR 内部的轻量本地 DPO trainer，用于工程验证和小规模消融。
6. `local_dpo` 使用 Stage 2 的 `chosen/rejected` pair，训练 pairwise DPO loss，同时保留 chosen-answer SFT auxiliary loss。
7. Stage 3 也支持：

```text
backend = verl
```

但当前只生成 dry-run plan，不启动真实 VERL DPO trainer。

8. `backend=verl` dry-run 会写出：

```text
runtime_logs/stages/train_agent/spad_rag/answer_distillation/<phase>/verl_command_plan.json
```

9. `backend=verl` 非 dry-run 会明确报错，避免误以为真实 VERL DPO 已经接好。

当前 Stage 3 边界必须明确：

```text
Stage3 local_dpo: 当前可执行
Stage3 verl: 当前只支持 dry-run plan
Stage3 SFT: 默认关闭，只保留配置和规划接口
```

## 6. Reward、Prompt 和 Teacher 现状

### 6.1 Actor prompt

当前 actor prompt 语义沿用 CoSearch / Search-R1 风格：

```text
每个 assistant turn 输出：
1. <reason>...</reason>
2. <tool_call>...</tool_call> 或 <answer>...</answer>
```

Stage 1 训练的是 search/stop policy，不训练 answer body。因此 actor 在生成到 `<answer>` 后停止，reward 由 teacher answerer 基于 search evidence 计算。

### 6.2 Teacher prompt

当前 teacher prompt 是 evidence-grounded answer-only QA prompt，不是 search agent prompt。

代码位置：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/prompts.py
```

Stage 1 teacher 当前输出三段：

```text
<reason>...</reason>
<status>supported_answer | insufficient_evidence | ambiguous_evidence</status>
<answer>...</answer>
```

约束：

1. teacher 只能使用 search evidence。
2. teacher 不允许用参数知识补全。
3. evidence 不足或模糊时，`<answer>` 必须输出固定拒答：

```text
证据不足无法作答
```

4. `<reason>` 必须说明证据支持答案，或说明缺什么证据以及当前证据为什么不足。
5. 原生 thinking 通过 chat template / inference 参数关闭，不在 prompt 中写“关闭 thinking”。

### 6.3 Reward 现状

当前 Stage 1 reward 主要在：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/reward.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py
```

当前规则：

1. actor 格式错误、非法 action、no finish、no search evidence 时，不调用 teacher。
2. teacher 只在 trajectory 格式有效且 actor 合法停止时调用。
3. teacher answer F1 是核心 reward。
4. search cost 已改为 delayed search cost：

```text
effective_search_cost = search_cost * max(0, search_count - free_search_count)
```

当前默认：

```text
search_cost = 0.02
free_search_count = 1
```

5. teacher 输出格式失败不丢弃轨迹，而是给 `teacher_format_error_penalty`，并统计失败率。
6. bad stop 已实现：
   - teacher status 为 `insufficient_evidence` 或 `ambiguous_evidence`。
   - actor 已经停止。
   - search_count 还没达到 `max_search_turns`。
   - 则判定为过早停止，给 bad stop penalty。
7. 当前默认：

```text
bad_stop.penalty = -0.35
bad_stop.max_budget_failed_penalty = -0.15
bad_stop.teacher_format_error_penalty = -0.1
max_search_turns = 5
```

8. reward metrics 会统计：

```text
teacher_called
teacher_format_error_count
teacher_format_error/rate_over_teacher_called
supported_answer_count
insufficient_evidence_count
ambiguous_evidence_count
bad_stop_count
bad_stop/rate_over_teacher_called
search_count
paid_search_count
effective_search_cost
duplicate_query_count
```

### 6.4 Search-R1 original baseline

当前还保留 `search_r1_original` 配置作为 baseline：

```text
AgenticIterRag/config/agent_training/search_r1_original.yaml
```

它仍通过 `train_agent.impl=spad_rag` 的工程路径接入，但 reward 类型切到 Search-R1 原始答案 EM 风格，不调用 teacher answerer。

## 7. 配置、资源和运行目录现状

### 7.1 配置文件

SPAD 主配置：

```text
AgenticIterRag/config/agent_training/spad_rag_base.yaml
```

当前关键默认：

```text
default_backend: smoke
Stage1 backend 默认: smoke
Stage2 backend 默认: smoke
Stage3 DPO backend 默认: smoke
Stage3 SFT 默认: disabled
```

正式实验 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_formal_overlay.yaml
```

正式脚本：

```text
tasks/train_tasks/agenticIterRag/run_260709f_AIR_spad_qwen3_1_7b_glm47_formal.sh
```

当前默认 search budget 已与 CoSearch/CAR 对齐：

```text
max_response_length: 4096
max_assistant_turns: 6
max_user_turns: 6
max_tool_response_length: 4096
rollout_max_model_len: 16096
max_search_turns: 5
```

### 7.2 资源配置

资源配置入口：

```text
AgenticIterRag/config/resource/local_8gpu_0_7.yaml
```

SPAD 资源按 sub-stage 管理：

```text
resource.stage_resources.train_agent.impls.spad_rag.sub_stages.search_policy_rl
resource.stage_resources.train_agent.impls.spad_rag.sub_stages.answer_refresh_data
resource.stage_resources.train_agent.impls.spad_rag.sub_stages.answer_distillation
```

第一版本地资源策略：

```text
Stage 1:
  actor train / rollout: [0, 1, 2, 3]
  GLM4.7 teacher: [4, 5]
  recall: [6, 7]

Stage 2:
  actor vLLM: [0]
  GLM4.7 teacher: [4, 5]
  recall: [6, 7]

Stage 3:
  local DPO trainer: [0]
```

资源生命周期：

1. 每个 sub-stage 可自动启动所需服务。
2. 每个 sub-stage 完成后自动释放本 stage 启动的服务。
3. 外部已有服务不会被强行停止。
4. recall 支持 proxy + 多 backend 实例，当前 Stage 1/2 默认走 NPU/GPU accelerator backend。

### 7.3 运行目录

新 run 目录已经收敛为：

```text
log/agenticIterRag/<RUN_NAME>/
  runtime_logs/
  outputs/
```

其中：

```text
runtime_logs/
  pipeline/
  stages/
    train_agent/
      spad_rag/
        search_policy_rl/
        answer_refresh_data/
        answer_distillation/
          dpo/
```

`outputs/` 存放 manifest、rollout data、dataset 等产物：

```text
outputs/pipeline.manifest.json
outputs/execution_plan.yaml
outputs/stages/train_agent/manifest.json
outputs/stages/train_agent/spad_rag/spad_manifest.json
outputs/stages/train_agent/spad_rag/<sub_stage>/manifest.json
```

checkpoint 已从 run-local outputs 中拆到：

```text
checkpoints/AIR/<RUN_NAME>/stages/train_agent/spad_rag/
```

例如 Stage 1 actor checkpoint：

```text
checkpoints/AIR/<RUN_NAME>/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/
```

## 8. 已验证内容

### 8.1 Stage 1 训练验证

已进行多轮 Stage 1 效率消融和 1-step/3-step 训练验证。代表性成功 run：

```text
log/agenticIterRag/260709-174103-338566-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_stage1_b64_micro2_3step
```

该 run 中 1 step 成功完成，关键信息包括：

```text
train_batch_size = 64
rollout.n = 8
rollout trajectories = 512 / 512
timing_s/step ~= 254s
timing_s/gen ~= 152s
teacher_called = 327 / 512
teacher_format_error_count = 1
bad_stop_count = 162
search_count/mean ~= 1.24
response_length/max = 1514
```

后续已将默认 search budget 从 `1536 / 3 turns` 调整到 `4096 / 6 turns`，用于支持更长多跳搜索轨迹。

### 8.2 VERL Stage3 dry-run 验证

Stage 3 `backend=verl` 已实现 dry-run plan，但非 dry-run 不启动真实训练。

已验证 dry-run run：

```text
log/agenticIterRag/260709-213127-222968-pipeline-agentic_iter_rag_v1_spad_stage3_verl_dryrun
log/agenticIterRag/260709-213224-655642-pipeline-agentic_iter_rag_v1_spad_stage2_stage3_verl_dryrun
```

验证结果：

1. Stage3 DPO `backend=verl` 能写出 `verl_command_plan.json`。
2. Stage2 dry-run 会输出 planned `answer_distill_dataset_manifest.json`。
3. Stage3 dry-run 能消费 Stage2 planned dataset manifest。
4. plan 中明确标记：

```text
implementation_status = dry_run_only
```

### 8.3 4096 / 6 turns dry-run 验证

已执行 SPAD formal dry-run，确认最终 VERL overrides 已包含：

```text
data.max_response_length=4096
actor_rollout_ref.rollout.response_length=4096
actor_rollout_ref.rollout.max_model_len=16096
actor_rollout_ref.rollout.multi_turn.max_user_turns=6
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096
custom_reward_function.reward_kwargs.reward_cfg.max_search_turns=5
```

对应 dry-run：

```text
log/agenticIterRag/260709-220557-807353-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_formal_default_4096_6_dryrun
```

### 8.4 单元测试

当前 SPAD 相关测试：

```text
AgenticIterRag/tests/agent_training/spad/test_search_policy_reward.py
AgenticIterRag/tests/agent_training/spad/test_answer_distillation_verl.py
```

已验证：

1. 首搜免费、第二次 search 起扣 cost。
2. `free_search_count` 可配置。
3. insufficient evidence 的 bad stop 逻辑。
4. teacher format error penalty。
5. invalid/no_finish 不调用 teacher。
6. Stage2 teacher `<status>` block 剥离。
7. Stage3 `backend=verl` dry-run plan 写出。
8. Stage3 `backend=verl` 非 dry-run 报错。
9. Stage2 dry-run 输出 dataset manifest。

已运行过：

```text
PYTHONPATH=AgenticIterRag python -m unittest \
  AgenticIterRag.tests.agent_training.spad.test_answer_distillation_verl \
  AgenticIterRag.tests.agent_training.spad.test_search_policy_reward
```

结果：

```text
Ran 22 tests
OK
```

## 9. 当前限制和后续工作

当前实现还不是最终长期形态，主要限制如下。

### 9.1 Stage3 VERL DPO 尚未真实接入

当前 `backend=verl` 只生成 dry-run plan。原因是 vendored VERL tree 中没有可直接消费 SPAD Stage2 chosen/rejected pair 的离线 DPO Hydra entry。

要真实接入 VERL DPO，需要补：

1. Stage2 JSONL pair 到 VERL preference parquet 的 converter。
2. SPAD offline DPO trainer entry。
3. policy/ref model 的多卡、长上下文、checkpoint、resume、mixed precision 支持。
4. 和当前 AIR checkpoint/log/runtime 目录约定对齐。

### 9.2 local_dpo 只适合工程验证

`local_dpo` 当前是 AIR 内部轻量 DPO trainer。它适合验证 Stage2 数据契约、loss wiring 和小步消融，但不应该长期作为唯一正式 DPO 后端。

正式 Stage3 仍建议后续接 VERL 或更完整 trainer。

### 9.3 Stage2 rollout 仍需正式大规模验证

Stage2 `rollout` 后端已实现，但正式全量 refresh 仍需关注：

1. actor vLLM 长上下文稳定性。
2. teacher 格式失败率。
3. `require_evidence_sufficient` 过滤后的样本保留率。
4. chosen/rejected pair 的 answer 质量。
5. `messages_before_final_answer` 与 Stage3 prompt 的一致性。

### 9.4 Reward 仍是保守版本

当前 Stage1 reward 只实现了必要改造：

```text
teacher F1
bad stop
delayed search cost
duplicate query penalty
missing reason penalty
format/no-finish penalties
```

暂未引入更复杂的：

```text
evidence gain reward
query rewrite quality reward
doc novelty reward
raw-query-repeat penalty
```

这些可以后续作为二期 reward shaping 消融，不应混入当前已验证 baseline。

### 9.5 正式训练仍需按执行计划推进

当前已有多次 Stage1 消融和 dry-run 验证，但完整三阶段正式训练仍需按 `260709a` 执行计划推进：

1. 先确认 Stage1 正式配置第一个 step 正常。
2. 再进入 Stage2 full refresh。
3. 再用 Stage2 数据跑 Stage3 local_dpo 或后续 VERL DPO。
4. 每个 sub-stage 结束后检查服务释放、manifest、checkpoint、runtime logs。

如果出现 OOM 或低利用率，优先调整：

```text
max_num_seqs
max_num_batched_tokens
actor/logprob micro batch
agent_loop_num_workers
teacher_batch_workers
trajectory_timeout_seconds
```

不应把 `train_batch_size` 降到 8，也不应把 `rollout.n` 降到 2；当前底线仍是：

```text
train_batch_size >= 32
rollout.n = 8
优先配置 train_batch_size = 64
```


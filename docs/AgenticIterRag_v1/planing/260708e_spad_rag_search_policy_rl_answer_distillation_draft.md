# SPAD-RAG 三阶段训练方案 Draft

日期：2026-07-08

状态：Draft

## 1. 方案名称

方案名：

```text
SPAD-RAG
```

全称：

```text
Search-Policy RL with Answer Distillation for Agentic RAG
```

中文名：

```text
搜索策略强化 + 答案能力蒸馏的 Agentic RAG 训练方案
```

核心目标是把 agentic RAG 的两个能力拆开：

```text
Search policy: 搜什么、搜几轮、什么时候停止
Answer ability: 基于搜索证据生成最终答案
```

阶段 1 只训练 search policy。阶段 2 重新生产和整理 on-policy answer 数据。阶段 3 再把 teacher answerer
的答案能力蒸馏回 actor。

## 2. 背景和动机

当前端到端 agent 训练容易把两个问题混在一起：

1. actor 是否搜到了足够证据。
2. actor 是否有能力把证据转成正确短答案。

小 actor 的 answer ability 较弱时，会出现两类错误 credit：

1. 搜索路径很好，但小 actor 答错，search policy 被误罚。
2. 搜索路径一般，但小 actor 靠参数记忆或偶然答对，search policy 被误奖。

SPAD-RAG 的核心做法是：

1. 阶段 1 用更强的 teacher answerer 在 reward function 内部评估搜索证据质量。
2. 阶段 2 用训练完成后的 actor 重新 rollout，获得最终 policy 分布下的 answer context。
3. 阶段 3 用 on-policy context 做 answer SFT/DPO，把 teacher answer 能力训回 actor。

最终部署时只使用 actor，不依赖 teacher answerer。

## 3. 总体流程

完整流程为：

```text
Stage 1: Search-Policy RL
  actor rollout stops at <answer>
  teacher answerer is called inside reward function
  update actor search/stop policy

Stage 2: Full-Answer Refresh Data Preparation
  freeze trained actor
  full rollout without <answer> stop
  actor answer becomes rejected
  teacher evidence-based answer becomes chosen
  build answer distillation dataset

Stage 3.1: Optional Answer SFT
  prompt -> teacher response
  default disabled

Stage 3.2: Answer DPO + auxiliary SFT loss
  prompt -> chosen teacher response vs rejected actor response
  default enabled
```

## 4. Stage 1: Search-Policy RL

### 4.1 目标

阶段 1 只训练 actor 的搜索策略和停止策略，不训练答案正文。

Actor 可以使用 Qwen3-1.7B 或 Qwen3-4B。训练时关闭原生 thinking，让模型按 CoSearch/AIR 风格显式输出
`<reason>`。

### 4.2 Actor 动作格式

搜索动作：

```text
<reason>...</reason>
<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
```

停止动作：

```text
<reason>...</reason>
<answer>
```

`<answer>` 是 rollout sampling 的 stop sequence，不是 prompt 中预先写入的文本。生成到 `<answer>` 后停止
decode，并保留 `<answer>` 到 actor 输出中，让 RL 可以把 reward credit 分给“选择停止”这个动作。

阶段 1 不让 actor 继续生成 answer body。因此 teacher answer tokens 不进入 actor rollout response，也不进入
stage 1 loss。

### 4.3 Teacher Answerer 是 Reward Function 的一部分

阶段 1 中 teacher answerer 不是单独数据阶段，而是 reward function 的一部分。

当 actor 触发停止动作后，reward function 内部调用 teacher answerer。Teacher answerer 只看搜索证据：

```text
original question
每轮 sub_query
每轮 actor 实际看到的 visible top5 docs
```

Teacher 不看 actor 的 answer body，因为阶段 1 本来不生成 answer body。

### 4.4 Teacher Answerer Prompt 边界

Teacher answerer 的 prompt 和 actor prompt 不同。

Actor prompt 是 tool-augmented search agent prompt，要求模型决定 search 或 answer。Teacher prompt 是
answer-only prompt，要求模型基于给定 evidence 生成答案。

但是 teacher answerer 的输出协议必须和 actor 最终 answer 协议一致：

```text
<reason>...</reason>
<answer>...</answer>
```

Teacher answerer 也必须使用和 actor 一致的外显推理模式：

1. 关闭原生 `<think>` / thinking mode。
2. 不使用模型内部 reasoning wrapper。
3. 显式输出 `<reason>` 标签。
4. 最终短答案放在 `<answer>` 标签中。

### 4.5 Evidence-grounded Teacher 约束

Teacher answerer 必须 evidence-grounded：

1. 只能使用给定 search evidence。
2. 不允许使用模型自身知识补全证据缺口。
3. 如果 evidence 不足以支持唯一答案，必须拒答。
4. `<reason>` 中要说明需要什么证据，以及当前证据不足的原因。
5. `<answer>` 中只输出短答案，或固定拒答字符串。

证据不足时推荐输出：

```text
<reason>需要能够直接支持问题答案的证据，例如明确给出实体、日期、地点或关系的 passage；当前检索结果只包含相关背景或间接线索，无法唯一确定答案。</reason>
<answer>证据不足无法作答</answer>
```

这个 `<reason>` 不能写成泛泛的“给定证据不足以支持唯一答案”，而应明确：

1. 缺少什么类型的证据。
2. 当前 evidence 为什么不足。

### 4.6 Stage 1 Reward

基础 reward：

```text
R = teacher_f1
    - search_cost
    - invalid_format_penalty
    - duplicate_query_penalty
    - no_finish_penalty
```

如果 teacher answerer 输出：

```text
<answer>证据不足无法作答</answer>
```

则 `teacher_f1 = 0`。可以进一步加 evidence-insufficient penalty，表示 search policy 没有搜到足够证据。

阶段 1 reward 评价的是：

```text
actor 的搜索路径 + 停止时机是否提供了足够证据
```

不是评价 actor 自己的 answer ability。

### 4.7 Stage 1 产物

阶段 1 产物包括：

```text
search-policy actor checkpoint
训练过程 trajectories
teacher_f1 / evidence_sufficient / reward breakdown 等诊断字段
```

训练过程 trajectories 可以用于分析和少量辅助数据，但不作为 Stage 3 的主训练数据。原因是阶段 1 早期
policy 的 token 分布、query 风格、搜索轮数和停止时机都可能与最终 actor 不一致。

## 5. Stage 2: Full-Answer Refresh Data Preparation

### 5.1 目标

阶段 2 是数据准备阶段，不更新 actor。

冻结 Stage 1 训练完成后的 actor，对全量训练集重新 rollout，获得最终 search policy 分布下的 on-policy
answer context。

### 5.2 Full-answer Refresh Rollout

阶段 2 不再在 `<answer>` 处 stop，而是让 actor 自然完整回答：

```text
search -> search -> ... -> final answer
```

需要保存真实 rollout 中的：

```text
messages_before_final_answer
actor_final_response
actor_answer
sub_queries
visible_top5_docs
search_count
format_status
```

其中 `messages_before_final_answer` 是 Stage 3 context 对齐的核心字段。它必须来自真实 refresh rollout，不应
用 teacher prompt 或后处理文本重建。

### 5.3 Teacher Chosen 生成

阶段 2 中，teacher answerer 基于同一条轨迹里的 search evidence 生成 chosen answer。

Teacher 输入仍然只包含：

```text
original question
每轮 sub_query
每轮 visible top5 docs
```

Teacher 不看 actor answer，避免 chosen 被 rejected 污染。

阶段 2 中的 teacher_f1、evidence_sufficient 等字段只用于：

1. 数据过滤。
2. 样本加权。
3. 训练前后诊断。

它们不再作为 RL reward 更新 actor。

### 5.4 Stage 2 输出数据

Stage 2 输出 answer distillation dataset：

```text
prompt   = messages_before_final_answer
chosen   = teacher final response
rejected = actor final response
metadata = teacher_f1, actor_f1, evidence_sufficient, search_count, format_status
```

其中 teacher final response 格式必须是：

```text
<reason>...</reason>
<answer>...</answer>
```

actor final response 作为 rejected，可以是格式正确但答案错误的 response，也可以是格式错误 response。格式错误样本
需要在 metadata 中显式标注。

### 5.5 Stage 2 数据过滤

第一版默认不把 evidence-insufficient 样本放入 Stage 3 answer distillation 数据，除非后续明确要训练
abstain 能力。

推荐过滤条件：

```text
evidence_sufficient == true
teacher_answer != "证据不足无法作答"
teacher output format valid
teacher_f1 >= threshold
search_count within budget
```

如果要保留一小部分拒答样本，需要单独设定比例，避免 actor 在 benchmark 上过度学会拒答而伤害 F1。

## 6. Stage 3: Answer Distillation

### 6.1 目标

阶段 3 把 teacher 的答案能力训回 actor，使最终部署不依赖 teacher answerer。

Stage 3 只使用 Stage 2 产出的 on-policy answer distillation dataset。

### 6.2 Stage 3.1: Answer SFT，默认关闭

Stage 3.1 是预留接口，默认关闭。

训练形式：

```text
prompt = messages_before_final_answer
target = teacher final response
```

target 格式：

```text
<reason>...</reason>
<answer>{teacher_answer}</answer>
```

如果 Stage 2 中 actor 的 answer 格式大量错误、空答案或不输出 `<answer>`，可以打开 3.1 做短程 SFT warmup。

### 6.3 Stage 3.2: Answer DPO，默认开启

Stage 3.2 是默认主训练阶段。

训练样本：

```text
prompt   = messages_before_final_answer
chosen   = teacher final response
rejected = actor final response
```

训练 loss 包含两部分：

```text
L_total = L_pairwise_dpo + lambda * L_sft_chosen
```

含义：

```text
L_pairwise_dpo: 让 actor 更偏好 teacher answer，而不是自己原始 answer
L_sft_chosen: 对 teacher answer 做 NLL/SFT 辅助监督，稳定格式和答案学习
```

因此不必强制先单独 SFT 再 DPO。`lambda` 控制 SFT 辅助强度。actor answer 初始质量差时可以取大一些；
actor 格式稳定时可以取小一些。

## 7. 最终推理

最终部署只使用 actor，不再使用 teacher answerer：

```text
question
-> actor search
-> tool returns visible docs
-> actor search or answer
-> final answer
```

输出协议：

```text
<reason>...</reason>
<answer>...</answer>
```

最终 actor 同时具备：

```text
Stage 1 学到的 search/stop policy
Stage 3 学到的 answer ability
```

## 8. 关键原则

1. Stage 1 的 teacher answerer 是 reward function 的一部分。
2. Stage 2 是数据准备阶段，不更新 actor。
3. Stage 3 的 prompt 必须使用 `messages_before_final_answer`，不能使用 teacher prompt。
4. Teacher prompt 和 actor prompt 不同，但 teacher 输出协议必须和 actor final answer 协议一致。
5. Teacher 必须 evidence-grounded，证据不足时拒答。
6. Teacher 证据不足时，`<reason>` 必须说明缺少什么证据以及当前证据为什么不足。
7. Teacher 和 actor 都关闭原生 thinking，统一使用 `<reason>` 作为外显推理标签。
8. Stage 1 训练日志不是 Stage 3 的主数据；Stage 3 主数据来自 Stage 2 refresh rollout。
9. 最终部署只保留 actor。

## 9. 与 AIR 现有 Pipeline 的关系

SPAD-RAG 的工程设计可以参考现有 AIR pipeline 中的三个机制：

1. `generate_traces` 的 enhanced trajectory 保存方式。
2. `train_llm_reranker` 中从中间状态拼 continuation context 的数据构造思路。
3. stage2 reranker reward 中把 action 接回上下文并用 answer reward 评价的思路。

SPAD-RAG 中对应关系为：

```text
AIR generate_traces
  -> SPAD Stage 1/2 enhanced trajectory logging

AIR train_llm_reranker continuation reward
  -> SPAD Stage 1 teacher_answer_reward

AIR branch dataset context reconstruction
  -> SPAD Stage 2 answer distillation dataset construction
```

区别是 SPAD-RAG 的训练对象从 reranker action 换成了 search actor 的 search/stop action 和最终 answer ability。

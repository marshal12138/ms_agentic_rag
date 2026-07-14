# SPAD 完整答案组级奖励与第三阶段 GRPO 重训计划

初稿：2026-07-10；更新：2026-07-11

## 1. 目标

本计划用于在清洗后的新数据上，分别完成 512-train/350-eval 和 5100-train/350-eval 两种规模的
Search-R1、SPAD 训练与评测，并重新验证 SPAD 三阶段训练。

范围修订（2026-07-11）：本文后续的 `Search-R1` 专指在本计划新数据上从同一 Base 独立训练的
Search-R1；此前口头使用的“结构化 Search-R1”也是这个实验，不是额外的第四类模型。旧
Search-R1、旧 SPAD、旧训练数据、旧评测轨迹和旧 Stage2 pair 全部排除，不进入正式结果表、差值、
显著性检验或效果结论。本轮只比较同一批新数据下的 Base、新数据 Search-R1、SPAD Stage1 与
SPAD Stage3；Stage2 作为数据刷新阶段报告质量统计，不作为模型 checkpoint 评测。

本轮同时包含两项确定的算法变更：

1. Stage1 不再在生成到 `<answer>` opening 时停止，而是生成完整的
   `<answer>...</answer>`，使 actor 最终答案可以直接按 Search-R1 相同的 EM 逻辑评分。
2. Stage3 默认训练策略从 DPO 改为 GRPO，最终答案 reward 使用与 Gold answer 的 token-level F1。

本轮不删除 DPO 的配置、数据、trainer 或代码。DPO 继续作为 Stage3 的另一种可选策略，
但不再是正式配置的默认策略。

计划必须在当前 AIR `train_agent.impl=spad_rag`、三个 SPAD sub-stage、现有 VERL 后端和现有
Hydra/YAML 配置体系中完成。不得建立第二套 SPAD pipeline、第二套配置解析器或独立训练框架。

## 2. 数据与实验前提

正式实验固定为两条相互对齐的规模轨道：

| 轨道 | Train | Eval | 用途 |
| --- | ---: | ---: | --- |
| 小规模 | 512 | 350 | 快速验证奖励和三阶段训练是否有效 |
| 完整规模 | 5100 | 350 | 验证规模扩展后的正式效果 |

两条轨道都必须分别训练 Search-R1 和 SPAD，不能用 512 版本 checkpoint 在 5100 数据上续训后
冒充独立的 5100 版本。默认均从同一个 Base Qwen3-1.7B 初始化。

固定数据与模型：

- 512 训练集：从新 5100 中确定性抽取并落盘到
  `data/global_train_eval_data/512t/co_search_ablation.train.parquet`
- 5100 训练集：`data/global_train_eval_data/5100t/co_search_ablation.train.parquet`
- 平衡评测集：`data/global_train_eval_data/350e/co_search_ablation.eval.parquet`
- 扩展评测集：`data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Base actor：`models/llm/Qwen3-1.7B`
- Teacher：`models/llm/GLM-4.7-Flash`
- 数据种子：42
- 每个 prompt 生成 8 条 rollout
- prompt batch：64
- PPO/GRPO mini-batch：64
- 512 轨道：8 个 step，每个算法实际使用 512 个唯一问题
- 5100 轨道：79 个完整 step；按当前 `drop_last=True` 实际使用 5056 个唯一问题

512 集必须是 5100 集的严格子集，并保持四个训练数据源的近似比例。按现有 5100 配额建议固定为：

| 数据源 | 512 配额 |
| --- | ---: |
| NQ | 205 |
| HotpotQA | 142 |
| MuSiQue | 90 |
| 2WikiMultiHopQA | 75 |

抽样继续使用当前 stable SHA-256 rank 规则，并生成独立 manifest 和 SHA-256。

除本计划明确要求的算法逻辑、数据路径和训练 step 外，任何训练与评估参数都必须保持当前正式配置
不变。尤其不得为了对齐数据量调整 batch size 或 rollout 数。正式配置固定：

```yaml
train_batch_size: 64
ppo_mini_batch_size: 64
n_samples_per_prompt: 8
```

参数不变约束包括但不限于：

- actor/ref micro-batch、log-prob micro-batch、学习率、KL、temperature、top-p 和 seed。
- prompt/response/model context 长度、最大 assistant/user turn 和 tool response 长度。
- agent loop worker、teacher worker、并发数、检索 top-N/top-M 和资源放置。
- Stage2 scheduler、teacher 请求参数和 Stage3 中与当前正式训练对齐的通用 GRPO 参数。
- 评估 batch size 96、六个 agent replica、两个 recall replica、温度 0、top-p 1、最大轮数和检索参数。

允许变化的只有：

- 本计划明确要求的 Stage1 完整 answer、组级 reward 和 Stage3 GRPO/F1 策略。
- 512/5100 对应的数据文件、样本上限、训练 step、实验名和输出路径。
- 为完整日志增加的记录字段和开关；日志改动不得改变训练数值路径。

所有 GRPO 训练路径必须满足以下不可覆盖的硬约束：

```yaml
train_batch_size: 64
ppo_mini_batch_size: 64
n_samples_per_prompt: 8
tensor_parallel_size: 1
```

适用范围包括：

- Search-R1 Stage1 GRPO。
- SPAD Stage1 GRPO。
- SPAD Stage3 answer GRPO。

Qwen3-1.7B 不使用 tensor parallel、pipeline parallel 或其他模型参数并行；对应 parallel size
全部固定为 1。多张 NPU 只用于 data parallel，rollout vLLM 的
`tensor_model_parallel_size` 也必须为 1。任何 dry-run/final config 中出现 TP/PP 大于 1 都应立即
fail fast，不能启动正式训练。

该约束只针对 Qwen3-1.7B actor/ref/rollout。GLM-4.7-Flash teacher 继续保持现有正式资源配置，
包括其现有 `tensor_parallel_size: 2`，不得因 actor 的 TP=1 约束而改动 teacher 并行参数。

5100 不能被 batch 64 整除。当前 VERL train dataloader 固定 `drop_last=True`，因此在不改变现有
batch/data-loader 语义的前提下，5100 数据池每 epoch 产生 79 个完整 step，实际用于更新的是
`79 * 64 = 5056` 个问题，尾部 44 条不进入该 epoch。本报告必须写“5100 数据池、5056 实际训练样本”，
不能声称精确覆盖了 5100 条。若未来要求 5100 条全部参与更新，需要另立数据尾批方案，不能在本轮
静默改变 batch size 或 rollout-n。

5100 条新训练数据均满足单答案契约。评测集中的多答案只允许表示同一对象的合理别名 OR。
训练和评测开始前必须再次记录 parquet SHA-256、代码 commit、模型路径、teacher prompt version、
reward version、检索索引版本和最终配置快照。

## 3. 新训练流程

SPAD 仍然保持当前三个 sub-stage，不改变顶层 AIR pipeline：

```text
Stage1 search_policy_rl
  完整搜索轨迹 + 完整 actor answer
  GRPO：组级 EM / teacher status 回退奖励
             |
             v
Stage2 answer_refresh_data
  使用 Stage1 checkpoint 重新 rollout
  teacher labeling 生成现有 chosen/rejected pair
             |
             v
Stage3 answer_distillation
  默认：GRPO + Gold answer F1
  可选：现有 DPO chosen/rejected 训练
```

Stage2 的两阶段调度、四个 teacher shard、pair schema 和 DPO 数据生产逻辑保持不变。

## 4. Stage1 完整答案生成

### 4.1 终止协议

当前 Stage1 的 stop sequence 是：

```yaml
stop_sequences: ["</tool_call>", "<answer>"]
```

新默认值改为：

```yaml
stop_sequences: ["</tool_call>", "</answer>"]
include_stop_str_in_output: true
```

含义如下：

- 搜索 action 仍在 `</tool_call>` 处结束当前 assistant turn，由 agent loop 执行检索。
- 最终回答必须生成完整的 `<answer>...</answer>`，并将 closing tag 保留在训练 response 中。
- prompt 中原有“Inside `<answer>`, output only the final short answer string”约束继续保留。
- Stage1 不再调用 opening-stop parser，不再以“出现 `<answer>` 但没有 closing tag”作为合法停止。

Stage1 的完整 answer 抽取和 EM 归一化必须与现有 Search-R1 reward 使用同一实现，不能复制一套
近似但行为不同的正则和 normalization。

### 4.2 单条 rollout 的基础量

对同一 prompt 的第 `i` 条 rollout，先计算两个基础量。

Actor EM：

```text
e_i = EM(extract_last_complete_answer(actor_output_i), gold_answers)
```

其中：

- 正确且格式完整时 `e_i = 1`。
- 答案错误、缺少完整 answer、answer 为空或解析失败时 `e_i = 0`。
- Gold 为合法别名 OR 时，命中任意一个合法别名均为 1。

Teacher status reward：

```text
t_i = 0, teacher status == insufficient_evidence
t_i = 1, teacher status == supported_answer
t_i = 1, teacher status == ambiguous_evidence
t_i = 0, teacher 请求失败、超时、XML 解析失败或没有合法 status
```

`ambiguous_evidence` 得 1 是本轮明确规则：只有 `insufficient_evidence` 为 0，其余合法 status 为 1。
格式错误不能被解释成“不是 insufficient，所以为 1”；没有成功解析出合法 status 时必须为 0。

Teacher 只判断 actor 已实际看到的检索证据是否达到非 insufficient 状态，不使用 actor answer，
也不使用 Gold answer。

### 4.3 GRPO 组级最终奖励

令同一 prompt 的 8 条 rollout 构成组 `G`，最终 reward 定义为：

```text
if max(e_j for j in G) == 1:
    r_i = e_i
else:
    r_i = 0.1 * t_i
```

等价地：

```text
r_i = e_i                                  if 组内存在任意 EM=1
r_i = 0.1 * teacher_status_reward_i         if 组内 EM 全为 0
```

示例：

| 组内 EM | 组内 teacher reward | 最终 reward | 说明 |
| --- | --- | --- | --- |
| `[0, 1, 0, ...]` | 任意 | `[0, 1, 0, ...]` | 组内已有最终答案监督，完全使用 EM |
| `[0, 0, 0, ...]` | `[1, 0, 1, ...]` | `[0.1, 0, 0.1, ...]` | 只有全零组启用证据状态 partial reward |
| `[0, 0, 0, ...]` | `[1, 1, 1, ...]` | `[0.1, 0.1, 0.1, ...]` | 仍为恒定组，GRPO advantage 为 0 |
| `[0, 0, 0, ...]` | `[0, 0, 0, ...]` | `[0, 0, 0, ...]` | 无有效训练信号 |

新 reward 不叠加旧 `teacher_f1`、search cost、bad-stop penalty、duplicate penalty 或
missing-reason penalty。否则就不再是指定的 EM/teacher 组级回退公式。

旧 `spad_teacher_f1` reward 类型及其全部配置和代码继续保留，供历史复现或独立消融使用。

### 4.4 Teacher 调用优化

Teacher 调用按组分两阶段执行：

1. 先对整批 rollout 完成 answer 解析和 EM 计算。
2. 按 VERL 原始 prompt UID 恢复 8-rollout group。
3. 组内只要存在 EM=1，整组直接使用 EM，不调用 teacher。
4. 仅对 EM 全零组调用 teacher。
5. 没有任何已执行搜索证据的 rollout，teacher reward 直接记 0，不发送空证据请求。

这样既严格满足 reward 定义，也避免对已有 EM 正信号的组浪费 teacher 推理资源。

### 4.5 组标识与批量 reward manager

组级判断不能依赖 batch 中相邻的 8 行，也不能只依赖 question 文本。VERL 在 rollout 前为每个
原始 prompt 生成 UID，并在 `n=8` repeat 后保留相同 UID；该 UID 是唯一允许的正式分组键。

实现时使用 VERL 已存在的 `BatchRewardManager` 和 SPAD 已存在的 batch reward 入口：

```yaml
reward_manager: batch
use_reward_loop: false
custom_reward_function.name: compute_spad_search_policy_reward_batch
```

`BatchRewardManager` 需要把当前 batch 的 `uid` 和 `tool_extra_fields` 合并进每条
`extra_info`，再交给 SPAD batch reward。这个改动应保持通用 batch reward API 不变，不能为
SPAD 复制一个新的 VERL trainer。

由于 reward 从 rollout 内逐条异步计算改成 rollout 后整批计算，必须验证：

- `tool_call_details`、检索文档和 search count 没有在 batch 路径丢失。
- batch balance/reorder 后 UID 分组仍然正确。
- reward 返回顺序与输入 batch 顺序严格一致。
- 每组 rollout 数量等于配置中的 `n_samples_per_prompt=8`；不满足时 fail fast。

### 4.6 Stage1 reward 日志字段

每条 rollout 至少落盘以下字段：

```text
actor_answer
actor_answer_parse_status
em_reward
teacher_called
teacher_evidence_status
teacher_status_reward
teacher_parse_status
teacher_format_error
group_uid
group_size
group_all_em_zero
partial_reward_applied
final_reward
reward_type
```

每 step 至少汇总：

- EM=1 rollout 数和比例。
- EM 全零 group 数和比例。
- 实际调用 teacher 的 rollout/group 数。
- partial reward 产生非恒定信号的 group 数。
- partial reward 仍为恒定的 group 数。
- teacher status 分布和解析失败率。
- 最终非恒定 GRPO group 数。

### 4.7 Stage1 全量 rollout 日志硬性契约

512 和 5100 两条 SPAD Stage1 训练都必须保存全部 rollout，不能只保留抽样日志、reward 汇总或
teacher cache。日志是正式 checkpoint 的组成部分；日志不完整时，该 checkpoint 不得进入 Stage2
或正式评测。

每一条 rollout 必须保存以下原始或结构化内容：

1. 身份与版本
   - `run_id`、`global_step`、`group_uid`、组内 rollout index。
   - 数据源、原始样本 ID、question、Gold answers。
   - actor checkpoint、数据 SHA-256、reward type/version、teacher prompt version/hash。
2. Actor 完整交互
   - actor 实际收到的初始 prompt。
   - 每个 assistant turn 的原始生成文本。
   - 每个 attempted query 和实际 executed query。
   - 每个 tool call 的结果、错误、耗时和是否被重复 query guard 拒绝。
   - 最终完整 actor response 和抽取出的 answer。
3. Observation
   - 每轮写回 actor 上下文的完整 tool observation，保持与模型实际可见文本逐字符一致。
   - observation 的截断状态和截断前后长度。
   - 不允许只记录 document ID 或 passage 摘要来替代模型实际看到的 observation。
4. Evidence
   - 每轮 `sub_query`。
   - teacher 可见的完整 top-5 evidence：rank、title、passage/contents、文档 ID 和 retrieval score。
   - 若运行时还保留 recall top-50，也应原样落盘到独立字段，不能覆盖 teacher-visible top-5。
   - evidence 的顺序、轮次和字符截断必须与实际 teacher builder 输入一致。
5. Teacher 输入与输出
   - 对每条有 evidence 的 rollout，都物化并保存完整 `teacher_messages`，即实际或本应提交给
     teacher 的 system/user messages。
   - 保存 teacher model、endpoint 标识、temperature、top-p、max tokens、thinking 开关和请求 hash。
   - teacher 实际调用时保存原始 XML response、解析结果、status、answer、耗时和错误。
   - 因组内已有 EM=1 而跳过 teacher 时，仍保存已物化的 `teacher_messages`，并记录
     `teacher_called=false`、`teacher_skip_reason=group_has_positive_em`。
6. Reward
   - Actor EM、teacher binary reward、是否全零组、是否启用 backoff、最终 reward。
   - answer/status parse 状态及全部错误码。

建议每 step 保持一个 append-only JSONL shard，并增加 run-level manifest：

```text
rollout_data/{step}.jsonl
rollout_data/manifest.json
```

manifest 至少记录：

- 预期和实际 step 数、prompt 数、group 数、rollout 数。
- 每个 shard 的记录数、字节数和 SHA-256。
- observation/evidence/teacher messages 字段的非空计数。
- teacher called/skipped/error 计数。
- 首尾 sample UID 和写入完成标记。

512 轨道的预期 Stage1 rollout 数为 `512 * 8 = 4096`。5100 数据池在当前 batch 64、
`drop_last=True` 下实际训练 5056 个问题，因此预期 rollout 数为 `5056 * 8 = 40448`。
训练完成后必须逐 shard 校验总数、UID group size 和必需字段，不能仅根据
trainer 正常退出判断日志完整。

当前通用 rollout dumper 只保证 input/output/Gold/score 和 reward extras。实现时应在现有
`_log_rollout_data`/`_dump_generations` 路径中补充保存 batch 已有的 `tool_extra_fields`、
`__num_turns__`、request ID、UID 和 observation；teacher messages/raw response 由 SPAD batch reward
作为 reward extras 返回。不得另起一个与 trainer 脱节的旁路 logger。

日志写入要求：

- 任何序列化失败必须带 UID 和 step 报错，不能静默跳过。
- shard 采用临时文件写完后原子 rename，并在成功后更新 manifest。
- 不对 observation、evidence 或 teacher messages 做日志层二次截断。
- 可以在 run 完成后无损压缩归档，但必须保留可流式读取和 hash 校验能力。
- checkpoint 清理前必须确认其对应 rollout manifest 已完成并通过校验。

## 5. Stage1 配置融合

在现有 `spad_rag_base.yaml` 的 `reward` 节增加新的显式 reward type，建议配置形状为：

```yaml
reward:
  type: spad_em_teacher_backoff

  spad_em_teacher_backoff:
    em_score: 1.0
    format_score: 0.0
    teacher_backoff_weight: 0.1
    insufficient_status: insufficient_evidence
    non_insufficient_score: 1.0
    teacher_error_score: 0.0
    require_complete_answer: true
    group_by: uid

  # 保留旧策略，非默认。
  spad_teacher_f1:
    teacher_f1_weight: 1.0
    search_cost: 0.02
    free_search_count: 1
    duplicate_query_penalty: -0.1
    bad_stop:
      enabled: true
```

实际字段名可以服从现有 resolver 的命名风格，但必须满足以下原则：

- 新旧 reward type 可由同一 `reward.type` 选择。
- 正式 overlay 显式选择新类型，不能依赖隐式默认值。
- 历史 `search_r1_original` 和 `spad_teacher_f1` 类型继续可运行。
- 最终配置快照能够完整显示 backoff 权重和分组方式。

正式 overlay 同时覆盖：

```yaml
agent_training:
  reward:
    type: spad_em_teacher_backoff

  sub_stages:
    search_policy_rl:
      trainer:
        reward_manager: batch
        train_batch_size: 64
        ppo_mini_batch_size: 64
        n_samples_per_prompt: 8
        preserve_full_rollout_logs: true
      rollout:
        stop_sequences: ["</tool_call>", "</answer>"]
```

实现阶段允许修改现有 base/formal 配置，也允许新增 512/5100 的 scale overlay。两条规模配置
最终解析后除样本路径、`train_max_samples`、`answer_refresh_data.inputs.max_samples`、总 step 和
实验名外，reward、batch、seed、rollout 数、模型、检索和 teacher 配置必须保持一致。

## 6. Stage2 保持不变

Stage2 继续执行：

1. `trajectory_rollout`：使用 Stage1 actor 生成完整搜索轨迹和 actor answer。
2. `teacher_labeling`：使用 GLM-4.7-Flash 生成 evidence-grounded chosen answer。
3. 输出当前 `spad_answer_distill_pair_v1`：
   `messages_before_final_answer`、`gold_answers`、`chosen`、`rejected` 和 evidence metadata。

保留当前过滤条件：teacher 格式合法、证据充分、pair schema 合法。Stage3 GRPO 默认只使用
Stage2 已保留的 evidence-sufficient 上下文，避免在证据不足上下文上训练答案生成。

四个 `offline_vllm_batch` shard 和 TP=2 资源配置继续使用，只做新轨迹上的 smoke，不重新进行
HTTP 并发和 offline shard 全量资源消融。

两条规模轨道分别处理自己的 Stage1 checkpoint 和训练样本：

- 512 轨道：`answer_refresh_data.inputs.max_samples=512`，不得复用 5100 轨道数据。
- 5100 轨道：`answer_refresh_data.inputs.max_samples=5100`，不得只刷新前 512 条。

Stage3 对应使用各自 Stage2 产生的全部 kept pairs。512 和 5100 轨道的 pair 数由 teacher filter
实际结果决定，不强行配成相同数量。

## 7. Stage3 默认 GRPO

### 7.1 输入数据

Stage3 GRPO 复用 Stage2 现有 pair JSONL，不新造 Stage2 数据格式。每条 kept pair 转换为标准
VERL RL row：

```text
prompt = messages_before_final_answer
reward_model.ground_truth.target = gold_answers
data_source = 原始数据源或 spad_answer_distillation
extra_info.question = question
extra_info.stage2_index = index
```

`chosen` 和 `rejected` 在 GRPO 策略中不参与 reward，但继续保留在原始 JSONL，供 DPO 策略读取。

转换产物写在当前 Stage3 phase 目录下，并在 manifest 中记录：

- 源 pair dataset 路径和 SHA-256。
- 转换后的 train/val parquet 路径和 SHA-256。
- 输入、保留、跳过样本数及原因。
- Gold answer 为空、prompt 为空和超长样本计数。

### 7.2 生成协议

Stage3 只训练“已有检索上下文后的最终回答”，不再调用 search tool，也不启动 recall/teacher 服务。

生成要求：

```text
<reason>...</reason>
<answer>short answer</answer>
```

Stage3 rollout 使用单轮生成，stop sequence 为 `</answer>`，保留 closing tag。初始模型必须使用
Stage1 训练完成钩子自动生成并在 manifest 中登记的 HF checkpoint；不得等到 Stage2 或评估前再
人工转换。

### 7.3 F1 reward

对第 `i` 条 Stage3 rollout：

```text
r_i = max(token_f1(extracted_answer_i, gold) for gold in gold_answers)
```

规则：

- 使用现有 `spad.reward.compute_f1` normalization 和 token F1。
- 多个合法别名时取最大 F1。
- 缺少完整 `<answer>...</answer>`、answer 为空或解析失败时 reward 为 0。
- `<reason>` 不参与 F1。
- 不调用 teacher，不混入 teacher F1、EM、search cost 或 DPO chosen/rejected 分数。

Stage3 每个 prompt 默认采样 8 条 answer，用 GRPO 的组内相对 F1 更新模型。

### 7.4 Stage3 配置方式

继续使用当前 `answer_distillation.phase_order` 和 `phases` 分发机制，不建立新的 Stage3 orchestrator。

默认配置改为：

```yaml
answer_distillation:
  phase_order: [grpo]

  phases:
    grpo:
      enabled: true
      backend: verl
      reward_type: gold_answer_f1
      train_batch_size: 64
      ppo_mini_batch_size: 64
      n_samples_per_prompt: 8
      learning_rate: 1.0e-6
      total_epochs: 1
      total_training_steps: -1
      max_prompt_length: 12000
      max_response_length: 1024
      rollout_max_model_len: 13024
      stop_sequences: ["</answer>"]
      apply_chat_template_kwargs:
        enable_thinking: false

    dpo:
      enabled: true
      backend: verl
      train_batch_size: 64
      micro_batch_size_per_gpu: 4
      learning_rate: 1.0e-6
      total_epochs: 1
      total_training_steps: -1
      max_samples: -1
      max_length: 4096
      beta: 0.1
      pairwise_loss_weight: 1.0
      chosen_sft_loss_weight: 0.2
      clip_grad_norm: 1.0
```

`dpo.enabled=true` 不代表默认同时运行；runner 只执行 `phase_order` 中列出的 phase。DPO 消融时通过
现有配置覆盖将 `phase_order` 改为 `[dpo]`。不得删除 DPO block、`local_dpo`、VERL offline DPO
recipe、测试或 resource 配置。

如果未来需要先 SFT 再 GRPO，仍通过现有 phase 顺序表达，例如 `[sft, grpo]`，本轮不启用。

Stage3 GRPO 对应的 Qwen3-1.7B 资源必须在现有 resource tree 中新增同级 `grpo.trainer`，并固定：

```yaml
resource:
  stage_resources:
    train_agent:
      impls:
        spad_rag:
          sub_stages:
            answer_distillation:
              phases:
                grpo:
                  trainer:
                    n_gpus_per_node: 8
                    tensor_parallel_size: 1
```

这里的 8 张卡仅组成 data parallel world；不得把 `tensor_parallel_size` 改为 8。

### 7.5 Stage3 后端复用

Stage3 GRPO 使用项目内现有 `python -m verl.trainer.main_ppo`、GRPO advantage、actor/ref worker 和
checkpoint 机制。应将 Stage1 已有的 VERL GRPO command builder 中通用部分复用到 Stage3，
而不是增加独立的 `train_spad_stage3_grpo.py` trainer 或新 pipeline。

Stage3 与 Stage1 的差异只通过现有配置表达：

- Stage3 使用转换后的 answer-context parquet。
- Stage3 关闭 multi-turn 和 tool config。
- Stage3 不启动 recall/teacher。
- Stage3 自定义 reward 为 Gold answer F1。
- Stage3 response budget 较短。

### 7.6 训练完成后自动转换 HF 模型

自动 HF 转换是训练 phase 的完成条件，不是人工后处理。以下每个真实训练 phase 在 VERL 返回 0 后，
必须自动把最终 actor checkpoint 合并/导出为 Hugging Face 可加载目录：

1. Search-R1-512 与 Search-R1-5100 的 Stage1 最终 step。
2. SPAD-512 与 SPAD-5100 的 Stage1 最终 step。
3. SPAD 两条轨道的默认 Stage3 GRPO 最终 step。
4. 作为备选策略运行 DPO 时，其最终模型也必须通过同一 HF 可加载性验收；已经直接保存为 HF 的
   backend 可跳过合并，但不能跳过校验和 manifest 登记。

实现应复用当前 `refresh_rollout.py` 已有的 FSDP actor merge 能力，将它提升为训练 runner 可调用的
公共 checkpoint finalizer。不得继续把“进入 Stage2 时临时转换”作为 SPAD Stage1 成功的唯一途径，
也不得为 Search-R1 和 SPAD 复制两套转换命令。

每个 phase 的收尾顺序固定为：

```text
VERL return code = 0
  -> 定位预期 final global_step
  -> 自动转换到临时 HF 目录
  -> 校验 config/tokenizer/权重文件与 Transformers 可读性
  -> 原子 rename 为正式 HF 目录
  -> 写入 manifest 的 raw_actor_checkpoint、hf_actor_checkpoint、转换日志与文件 hash
  -> phase status = completed
```

任一步转换或校验失败，都必须使训练入口非零退出，phase 不得写成 `completed`，SPAD 不得进入
Stage2/Stage3，Search-R1 不得进入评估。训练 shell 返回 0 但没有 `hf_actor_checkpoint`，或 manifest
中的路径不是有效 HF 模型，均定义为代码错误，必须修正后重跑收尾流程；禁止用手工执行 merger
后补路径的方式把该 run 认定为正式成功。

自动转换不得改变训练参数，也不得删除原始 VERL checkpoint。转换支持幂等恢复：正式 HF 目录已
存在时先完整校验，校验通过才复用；不完整目录必须报错或从原始 checkpoint 重新原子生成。

## 8. 预计修改位置

### 8.1 Stage1 reward 与 runner

- `AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py`
  - 复用完整 answer EM 解析。
  - 在现有 batch reward 中增加 UID 分组和全零组 teacher backoff。
  - 增加新 reward type 的明细字段和汇总字段。
- `AgenticIterRag/agentic_iter_rag/agent_training/spad/search_policy_rl.py`
  - 新 reward type 默认使用 batch reward manager。
  - 完整透传 reward type、batch manager 和 stop sequence。
  - VERL 成功后调用统一 checkpoint finalizer，自动生成并返回 `hf_actor_checkpoint`。
  - 保留旧 reward 类型执行路径。
- `AgenticIterRag/verl/verl/workers/reward_manager/batch.py`
  - 将 UID 和 `tool_extra_fields` 安全合并到逐条 extra info。
  - 不改变通用 batch `compute_score` 函数签名。
- `AgenticIterRag/verl/verl/trainer/ppo/ray_trainer.py`
  - 扩展现有 rollout dump，保存完整 observation、tool extras 和交互元数据。
  - 保持现有 step shard 写入位置，不增加旁路日志体系。

### 8.2 Stage3 GRPO

- `AgenticIterRag/agentic_iter_rag/agent_training/spad/answer_distillation.py`
  - 在现有 phase dispatch 中支持 `grpo`。
  - 将 Stage2 pair 转换为标准 RL parquet。
  - 构建和执行复用现有 VERL GRPO 后端的计划。
  - GRPO/DPO phase 成功后自动完成 HF 导出或 HF 可加载性校验。
  - DPO 分支保持不变。
- `AgenticIterRag/agentic_iter_rag/agent_training/spad/reward.py`
  - 增加完整 answer 的 Gold F1 reward 入口，复用现有 `compute_f1`。
- `AgenticIterRag/agentic_iter_rag/agent_training/spad/orchestrator.py`
  - Stage1 之后强制读取 `hf_actor_checkpoint`，缺失时立即失败，不再依赖 Stage2 才转换。
  - 继续通过相同的 `run_answer_distillation` 接口接收 Stage3 最终 HF checkpoint。
  - 不新增顶层 stage。
- 当前 `refresh_rollout.py` 的 `_ensure_hf_actor_checkpoint` 与 manifest writer
  - 将已有 FSDP merge 实现抽成 Search-R1/SPAD 共用的 checkpoint finalizer。
  - Stage2 保留兼容性检查，但正式路径直接消费 Stage1 manifest 中的 HF checkpoint。
  - manifest 同时记录原始 VERL 路径、HF 路径、转换状态、日志和 hash。

### 8.3 配置与资源

- `AgenticIterRag/config/agent_training/spad_rag_base.yaml`
  - Stage1 默认完整 answer 和新组级 reward。
  - Stage3 默认 `phase_order: [grpo]`。
  - 完整保留 DPO 配置。
- `tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_formal_overlay.yaml`
  - 正式配置显式选择新 Stage1 reward、完整日志和 Stage3 GRPO。
  - 512/5100 轨道保持 batch 64 和 rollout-n 8，分别执行 8/79 个完整 step。
  - 在现有 Stage3 resource phases 下增加 `grpo.trainer`，保留 `dpo.trainer`。
- `tasks/train_tasks/agenticIterRag/configs/` 下新增四个 scale overlay
  - Search-R1-512、Search-R1-5100、SPAD-512、SPAD-5100 各一个。
  - scale overlay 只表达该正式轨道的路径、样本数、step、实验名和必要的策略选择。
  - 由现有 task shell 的 `--OVERLAY_YAML` 透传加载，不修改 task shell 文件。
- 当前 resource resolver 和 manifest writer
  - 只补充识别 `grpo` phase 所需字段，不建立新资源配置组。

## 9. 单元测试与集成测试

### 9.1 Stage1 完整答案测试

在现有 `test_search_policy_reward.py` 中覆盖：

1. `<answer>Paris</answer>` 对 Gold Paris 的 EM 为 1。
2. 只有 `<answer>` opening 时 EM 为 0。
3. 多个 answer block 使用最后一个完整 answer，与 Search-R1 一致。
4. 合法别名 OR 命中任意一个时 EM 为 1。
5. normalization 与 Search-R1 对大小写、标点和冠词的行为完全一致。

### 9.2 Stage1 组级 reward 测试

至少覆盖：

1. 组内存在一个 EM=1 时，全部最终 reward 等于各自 EM，teacher 不调用。
2. 全零组中 S/A/I 映射为 `0.1/0.1/0`。
3. teacher 超时、异常和 XML 解析错误映射为 0。
4. 全零且 teacher reward 全 1 时，最终组保持常量 0.1。
5. batch 输入被重排后，仍按 UID 正确分组并恢复原顺序。
6. 同 question 不同 UID 不得合并。
7. UID 缺失、组大小不是 8 时 fail fast。
8. `tool_extra_fields` 中的证据和 search count 在 batch 路径可见。
9. 旧 `search_r1_original` 和 `spad_teacher_f1` 测试继续通过。

### 9.3 Stage1 全量日志测试

至少覆盖：

1. rollout shard 保存完整 actor turns、tool observation 和 evidence steps。
2. teacher called 与 skipped 两类记录都包含完整 `teacher_messages`。
3. teacher raw XML、status 和错误信息可逐条对齐。
4. observation 与 agent 实际收到的 tool response 逐字符一致。
5. shard 写入失败时不生成 completed marker。
6. manifest 中记录数、UID group 数和 shard SHA-256 校验通过。
7. 8 条 smoke、512 轨道和 5100 轨道的预期记录数可由配置推导。
8. 日志序列化不改变通用 reward tensor 或训练 batch 顺序。

### 9.4 Stage3 GRPO 测试

扩展现有 `test_answer_distillation_verl.py`：

1. Stage2 pair 可转换为标准 RL parquet。
2. prompt 保持 `messages_before_final_answer` 的角色和内容顺序。
3. Gold answers 正确进入 `reward_model.ground_truth.target`。
4. Stage3 dry-run 生成 `main_ppo + grpo` 计划。
5. Stage3 计划关闭 multi-turn 和 tool service。
6. 完整 answer 的 F1 为预期值；空 answer 和缺失 close tag 为 0。
7. `phase_order: [grpo]` 只运行 GRPO。
8. `phase_order: [dpo]` 仍生成当前 offline DPO 命令。
9. GRPO 和 DPO manifest 都能被顶层 SPAD manifest 正确引用。

### 9.5 配置编译测试

正式 overlay dry-run 后检查最终配置：

- Stage1 stop sequence 包含 `</answer>`，不包含单独的 `<answer>`。
- Stage1 reward type 是新组级 reward。
- Stage1 reward manager 是 batch。
- 所有 GRPO 的 train/mini batch 是 64，rollout-n 是 8，TP/PP 是 1。
- 512 轨道为 8 step；5100 数据池按现有 `drop_last` 语义为 79 step、5056 个实际训练问题。
- Stage1 full rollout logging 开关已启用。
- Stage3 phase order 是 `[grpo]`。
- Stage3 GRPO reward 是 Gold F1。
- DPO block 和 DPO resource plan 仍存在。
- 顶层 pipeline 仍只执行现有 `train_agent -> spad_rag`。

### 9.6 自动 HF 转换测试

至少覆盖：

1. Search-R1 与 SPAD Stage1 成功训练后都会调用同一个 checkpoint finalizer。
2. FSDP shard 成功合并后，manifest 同时包含 raw 与 HF checkpoint 路径。
3. 已经是合法 HF 目录时只做校验，不重复破坏性转换。
4. merger 返回非零、缺少权重、缺少 config/tokenizer 或模型不可读时，phase 非零失败且不得标记完成。
5. 临时目录转换成功后才原子 rename；中断不会留下可被误认成正式产物的 HF 目录。
6. SPAD Stage2 直接读取 Stage1 的 `hf_actor_checkpoint`，不触发延迟人工转换。
7. Stage3 GRPO 和保留的 DPO 路径都返回经过校验的 HF checkpoint。
8. dry-run 只写转换计划和目标路径，不实际调用 merger。

## 10. 分阶段运行计划

### 阶段 A：纯函数和配置验证

1. 完成上述代码与配置融合。
2. 运行 SPAD 单元测试。
3. 运行正式 overlay dry-run。
4. 检查 command plan、resource plan、manifest 和配置快照。

通过条件：全部测试通过，dry-run 不启动服务，且新旧 reward、GRPO/DPO 两条策略均可解析。

### 阶段 B：8 条无参数更新 smoke

使用 8 个 prompt，每个 prompt 8 条 rollout：

- 确认 actor 生成完整 answer。
- 确认 batch reward 得到 8 个 UID group。
- 人工检查至少一个“组内有 EM=1”和一个“组内 EM 全零”的例子。
- 确认 teacher 只调用全零组。
- 确认 Stage3 GRPO 数据转换、reward 前向和 command dry-run 可执行。

本阶段只执行 rollout、reward、日志和配置链路检查，不启动 optimizer、不做参数更新，也不产生训练
checkpoint。通过条件：reward 逐条手算一致，teacher 调用数一致，rollout/manifest 可读取。

### 阶段 C：64 条单步实验

使用一个正式完整 batch：64 prompt × 8 rollout = 512 trajectories。

这是第一个允许发生参数更新的 smoke，训练 batch 仍为 64、rollout-n 仍为 8，Qwen3-1.7B TP/PP
仍为 1；只把 `total_training_steps` 临时设为 1，不修改其他训练参数。

检查：

- 全零 group 比例。
- EM 非恒定 group 比例。
- teacher partial reward 新增的非恒定 group 数。
- teacher 请求错误和 XML parse rate。
- Stage1 更新后的 loss、KL、entropy 和梯度是否有限。
- 完整 answer 率是否明显低于 Search-R1；若低于，先修协议，不进入正式训练。
- rollout shard 是否恰好包含 512 条完整日志，observation/evidence/teacher messages 是否齐全。

### 阶段 D：512 轨道正式训练

先从同一 Base 分别训练两个独立模型：

- Search-R1-512：512 prompt，EM reward，不调用 teacher。
- SPAD-512：相同 512 prompt，新组级 EM/teacher backoff reward。
- seed 42。
- 每 prompt 8 rollout。
- 64 prompt/batch。
- 1 epoch，共 8 个训练 step。
- 学习率 `1e-6`。

两者使用完全相同的 512 样本顺序、batch 切分、rollout 参数和检索配置。SPAD 每 step 落盘
完整 rollout、observation、evidence、teacher messages、EM、teacher reward、组级 final reward 和
action 统计。正式 checkpoint 固定使用 step8，不根据新 350 选择最优 step。

Search-R1-512 与 SPAD-512 Stage1 在 step8 完成后必须由训练入口自动产出 HF 模型并写入各自
manifest；缺失时本阶段失败，不得继续。SPAD-512 Stage2 直接消费该 HF 路径。

SPAD-512 随后继续执行 Stage2-512 和 Stage3-GRPO-512。Stage2 最多处理 512 条，Stage3 使用
该轨道的全部 kept pairs。

### 阶段 E：5100 轨道正式训练

512 轨道通过实现和稳定性验收后，再从同一 Base 分别启动：

- Search-R1-5100：5100 prompt，EM reward。
- SPAD-5100：相同 5100 prompt，新组级 EM/teacher backoff reward。
- seed 42。
- 每 prompt 8 rollout。
- 64 prompt/batch。
- 1 epoch，共 79 个完整训练 step。
- 学习率 `1e-6`。

两者不得从各自的 512 checkpoint 续训，避免把训练顺序差异混入规模比较。SPAD-5100 的
Stage1 按现有 `drop_last=True` 预期生成 40448 条 rollout，所有 step shard 必须完整保留并通过
manifest 校验。正式 checkpoint 固定使用 step79。

Search-R1-5100 与 SPAD-5100 Stage1 在 step79 完成后同样必须自动产出并校验 HF 模型；不能把
手工 merger 作为正式流程的一部分。

SPAD-5100 随后继续执行 Stage2-5100 和 Stage3-GRPO-5100。Stage2 必须覆盖 5100 条输入，
不能沿用旧 overlay 中的 512 上限。

### 阶段 F：Stage2 刷新

两个规模分别从自己的 Stage1 最终 checkpoint 重新生成 Stage2 数据，不复用旧数据或彼此数据：

- 512 轨道 Phase A：512 条 trajectory rollout。
- 5100 轨道 Phase A：5100 条 trajectory rollout。
- Phase B：四 shard offline teacher labeling。
- 输出新的 kept/skipped 统计和 pair dataset hash。

### 阶段 G：Stage3 GRPO

每条规模轨道独立执行：

1. 将本轨道 Stage2 kept pairs 转换为 GRPO parquet。
2. 使用本轨道 Stage1 HF checkpoint 初始化。
3. 每 prompt 采样 8 个完整 answer。
4. 使用 Gold token F1 进行 1 epoch GRPO。
5. 保存本轨道 Stage3 final raw checkpoint、自动转换后的 HF checkpoint、rollout、F1 分布和
   非恒定 group 比例。

Stage3 的 step 数按 `floor(kept_pairs / 64)` 计算，当前 `drop_last=True` 下的尾部不足 64 条数据不进入
该 epoch，报告中必须同时记录 kept pair 总数和实际参与训练的数量。如果 Stage2 kept pair 少于 64，
该轨道的正式 Stage3 训练判定为阻塞：不得调小 batch、rollout-n 或通过临时累积策略改变现有训练
参数；应先增加合规 pair 数或停止该轨道并报告。

## 11. 对照与评测

在同一份新 350 条平衡评测集上，对以下固定 checkpoint 各做三次相同配置评测：

1. Base Qwen3-1.7B。
2. Search-R1-512 step8。
3. SPAD-512 Stage1 step8。
4. SPAD-512 Stage3 GRPO final。
5. Search-R1-5100 step79。
6. SPAD-5100 Stage1 step79。
7. SPAD-5100 Stage3 GRPO final。

必须分别报告两类比较：

- 同规模算法比较：Search-R1-512 vs SPAD-512，Search-R1-5100 vs SPAD-5100。
- 同算法规模比较：Search-R1-512 vs Search-R1-5100，SPAD-512 vs SPAD-5100。

不能只报告“最好的一个 checkpoint”，也不能把 512 与 5100 的结果合并平均。

主指标：

- EM、F1。
- 有效完整 answer 比例。
- 首轮搜索执行率。
- 平均搜索次数和搜索次数桶。
- 重复 query、唯一 query、`max_turns`。
- 七个数据源逐源 EM/F1。
- 单跳与多跳数据集宏平均。

统计要求：

- 三次运行分别报告，并报告均值和波动范围。
- 不能把同一 350 条的三次运行当成 1050 个独立样本。
- 模型间使用同问题 paired bootstrap 置信区间。
- checkpoint 在评测前固定，禁止按 350 结果挑 step。
- 所有版本使用同一个 350 eval hash、同一检索服务、同一最大轮数和同一生成参数。

350 是 3500 的子集。若模型差异小于约 2 EM 或逐数据集方向冲突，在 `3500 - 350` 的独立
3150 条补集上做一次确认，并报告逐数据集宏平均，避免 Bamboogle 数量较少造成 micro-average 偏置。

## 12. 新 reward 的重点审计

Stage1 正式训练后必须回答：

1. 新完整 answer 协议是否使 SPAD 的 EM reward 密度接近 Search-R1。
2. 多少 group 已有 EM 信号，因此完全没有使用 teacher。
3. teacher backoff 在多少全零 group 中真正制造了组内方差。
4. 有多少全零 group 的 teacher reward 仍然恒定，因而对 GRPO 没有作用。
5. teacher 的 false-S/false-I 是否会改变同组轨迹排序。
6. partial reward 是否改善后续 step 的 EM，而不只是提高证据非 insufficient 比例。
7. Stage3 F1 GRPO 是否提高 F1 但损害 EM、格式或搜索行为。

另外从全零 group 抽取 240 条 rollout 做盲审，区分：

- 证据充分且 actor 语义正确。
- 证据充分但 actor 部分正确。
- 证据充分但 actor 错误或没有完整 answer。
- 证据不足。
- 真歧义。

人工审查结果应同时与 `e_i`、`t_i` 和最终 `r_i` 对齐，验证 0.1 backoff 是否把正确方向的轨迹排在前面。

## 13. 验收条件

### 13.1 实现验收

- Stage1 生成完整 `<answer>...</answer>`。
- Stage1 EM 与 Search-R1 对相同字符串逐例完全一致。
- 组级 reward 与定义公式逐例完全一致。
- teacher 只在 EM 全零组调用。
- teacher 合法 A 状态为 1，I 状态为 0，解析错误为 0。
- Stage3 默认执行 GRPO，reward 为 Gold answer F1。
- DPO 配置、代码、测试和可执行路径完整保留。
- 全流程仍由当前 SPAD orchestrator、manifest 和 resource resolver 管理。

### 13.2 训练稳定性验收

- Stage1 和 Stage3 均无 NaN/Inf loss、reward 或 advantage。
- Search-R1/SPAD 的 512 轨道各覆盖相同的 512 个唯一训练问题。
- Search-R1/SPAD 的 5100 数据池各配置为同一 5100 条；当前单 epoch 实际训练问题数均为 5056。
- Stage1 teacher 请求无未处理异常，解析失败有明确统计。
- Stage1 每组 UID 和 rollout 数完整。
- SPAD-512 Stage1 恰好保留 4096 条 rollout 日志。
- SPAD-5100 Stage1 按当前 79 个完整 step 恰好保留 40448 条 rollout 日志。
- 每条 SPAD Stage1 日志包含模型可见 observation、teacher-visible evidence 和完整 teacher messages。
- rollout manifest 的记录数、group size 和 shard hash 全部通过。
- Stage3 F1 非恒定 group 比例可报告且不为全零。
- 所有正式产物带数据、配置、prompt、reward 和代码版本 hash。
- Search-R1/SPAD 各训练 phase 均自动产出有效 HF 模型，manifest 中 raw/HF 路径和转换日志齐全。
- 任何缺少自动 HF 转换产物的训练 run 均不得进入评估或最终报告。

### 13.3 效果验收

- SPAD Stage1 总体 EM/F1 不低于同配置 Base。
- SPAD Stage1 多跳宏平均不低于 Base。
- Stage3 GRPO final 的 F1 不低于 Stage1，且 EM 不出现无法解释的显著下降。
- 有效 answer 比例不低于 Base。
- 重复 query 和 `max_turns` 必须单独报告，不能用平均搜索次数掩盖失败尾部。
- 512 与 5100 两个规模都必须完成同规模 Search-R1/SPAD 比较。
- 最终同时报告相对 Base、同规模新数据 Search-R1 和同规模 SPAD Stage1/Stage3 的差值；不报告
  任何旧 Search-R1 或旧 SPAD 历史结果的差值。

## 14. 不需要重做的内容

- 不重新进行完整 teacher HTTP 并发消融。
- 不重新比较已经失败的全部 teacher prompt 变体。
- 不删除或重写 DPO recipe。
- 不改变 AIR 顶层 pipeline DAG。
- 不创建新的 SPAD 项目目录、训练框架或独立配置系统。
- 不复用旧 5100、旧 350、旧 checkpoint 或旧 Stage2 pair 作为新正式结果。
- 不把 5100 数据池的 run 错报为单 epoch 精确训练了 5100 条；当前实际值是 5056。
- 不以 reward 汇总、teacher cache 或抽样 trajectory 代替 SPAD Stage1 全量 rollout 日志。

## 15. 最终产物

本轮完成后应得到：

1. 融入当前框架的 Stage1 完整答案与组级 reward 实现。
2. 融入现有 Stage3 phase 的 GRPO 策略。
3. 继续可运行的 Stage3 DPO 策略。
4. Search-R1-512、SPAD-512 Stage1、SPAD-512 Stage3 的原始与自动转换 HF checkpoint。
5. Search-R1-5100、SPAD-5100 Stage1、SPAD-5100 Stage3 的原始与自动转换 HF checkpoint。
6. 上述六个 checkpoint 与 Base 在同一 350 上的三次评测结果。
7. SPAD-512 的 4096 条和 SPAD-5100 的 40448 条完整 Stage1 rollout 日志及校验 manifest。
8. 两个规模的 Stage1 全量组级 reward 审计和 240 条人工审查。
9. 一份新的中文工作报告，明确区分：
   - 数据清洗带来的变化。
   - Stage1 完整答案带来的变化。
   - teacher backoff 带来的变化。
   - Stage3 从 DPO 改为 GRPO 带来的变化。
   - 训练规模从 512 扩展到 5100 数据池带来的变化。

只有这些变化被分开报告后，才能重新判断“SPAD 是否优于 Base/Search-R1”以及 teacher partial
reward 是否真正改善了 GRPO 的有效训练信号。

## 16. 训练与评估脚本提醒

本节记录实现完成后的实际执行模板，仅用于提醒和复核。允许按本计划修改 Python 训练代码、
base/formal YAML，并新增四个 scale overlay；以下三个 shell 入口文件保持不变。

### 16.1 固定入口与完整性检查

固定入口：

```text
Search-R1 训练：tasks/train_tasks/agenticIterRag/run_260709g_AIR_search_r1_original_qwen3_1_7b_formal.sh
SPAD 训练：tasks/train_tasks/agenticIterRag/run_260709f_AIR_spad_qwen3_1_7b_glm47_formal.sh
统一 350 评估：tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh
```

2026-07-11 记录的入口文件 SHA-256：

```text
f45d11f1200eb8fbccb9329fb1f4132855e56df35966545c6c0560f62a7c655e  tasks/train_tasks/agenticIterRag/run_260709f_AIR_spad_qwen3_1_7b_glm47_formal.sh
9fb730e4f9d029831b9e7a4e2e1f8f8e232c47d48187690cb4fc1e11ca0ee11f  tasks/train_tasks/agenticIterRag/run_260709g_AIR_search_r1_original_qwen3_1_7b_formal.sh
7c5961dd1c1486c464f98d463c8e75a20d876170b3fa6e8dd25ff8799b194999  tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh
```

代码和配置实现前后均执行：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives

sha256sum -c <<'EOF'
f45d11f1200eb8fbccb9329fb1f4132855e56df35966545c6c0560f62a7c655e  tasks/train_tasks/agenticIterRag/run_260709f_AIR_spad_qwen3_1_7b_glm47_formal.sh
9fb730e4f9d029831b9e7a4e2e1f8f8e232c47d48187690cb4fc1e11ca0ee11f  tasks/train_tasks/agenticIterRag/run_260709g_AIR_search_r1_original_qwen3_1_7b_formal.sh
7c5961dd1c1486c464f98d463c8e75a20d876170b3fa6e8dd25ff8799b194999  tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh
EOF
```

允许新增并由入口透传的 scale overlay：

```text
tasks/train_tasks/agenticIterRag/configs/search_r1_qwen3_1_7b_512_scale_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/search_r1_qwen3_1_7b_5100_scale_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_512_scale_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_scale_overlay.yaml
```

现有两个训练入口会在自己的 formal overlay 之后继续透传 `"$@"`，因此追加的 scale overlay
最后合并并覆盖规模字段，入口脚本本身无需修改。

### 16.2 训练前检查脚本

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
PY=/data05/conda/envs/ms/ms_agt_rag/bin/python
TRAIN_512="${ROOT}/data/global_train_eval_data/512t/co_search_ablation.train.parquet"
TRAIN_5100="${ROOT}/data/global_train_eval_data/5100t/co_search_ablation.train.parquet"
EVAL_350="${ROOT}/data/global_train_eval_data/350e/co_search_ablation.eval.parquet"

cd "${ROOT}"
test -s "${TRAIN_512}"
test -s "${TRAIN_5100}"
test -s "${EVAL_350}"

"${PY}" - "${TRAIN_512}" "${TRAIN_5100}" "${EVAL_350}" <<'PY'
import sys
import pandas as pd

train512 = pd.read_parquet(sys.argv[1])
train5100 = pd.read_parquet(sys.argv[2])
eval350 = pd.read_parquet(sys.argv[3])
assert len(train512) == 512, len(train512)
assert len(train5100) == 5100, len(train5100)
assert len(eval350) == 350, len(eval350)

def question(row):
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("question"):
        return " ".join(str(extra["question"]).lower().split())
    raise AssertionError("missing extra_info.question")

q512 = {question(row) for row in train512.to_dict("records")}
q5100 = {question(row) for row in train5100.to_dict("records")}
assert len(q512) == 512, len(q512)
assert len(q5100) == 5100, len(q5100)
assert q512 <= q5100, "512 train must be a strict subset of 5100 train"
print("data preflight passed: train512=512 train5100=5100 eval350=350")
PY

sha256sum "${TRAIN_512}" "${TRAIN_5100}" "${EVAL_350}"
```

### 16.3 Search-R1 两种规模训练脚本

以下模板始终调用同一个现有 Search-R1 task shell。先给两个规模分别做 dry-run，检查最终配置中
batch、step、train path、reward 和输出目录，再删除 `--dry-run` 启动正式训练。

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
ENTRY="${ROOT}/tasks/train_tasks/agenticIterRag/run_260709g_AIR_search_r1_original_qwen3_1_7b_formal.sh"
CFG_DIR="${ROOT}/tasks/train_tasks/agenticIterRag/configs"

cd "${ROOT}"

# 配置编译检查，不启动训练。
bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/search_r1_qwen3_1_7b_512_scale_overlay.yaml" \
  --dry-run

bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/search_r1_qwen3_1_7b_5100_scale_overlay.yaml" \
  --dry-run

# 正式 Search-R1-512：必须从 Base 初始化，512 条、batch 64、rollout-n 8、8 step。
bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/search_r1_qwen3_1_7b_512_scale_overlay.yaml"

# 正式 Search-R1-5100：再次从 Base 初始化，5100 数据池、batch 64、rollout-n 8、79 step。
bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/search_r1_qwen3_1_7b_5100_scale_overlay.yaml"
```

上述正式命令的成功退出契约包含自动 HF 转换：命令不得在 finalizer 完成前返回 0。Search-R1
manifest 必须同时给出最终 VERL/FSDP checkpoint 和 `hf_actor_checkpoint`；缺少后者即为训练失败，
不得在命令后手工运行 merger 补救并继续评估。

两个 Search-R1 scale overlay 的最终配置必须明确包含：

```yaml
agent_training:
  reward:
    type: search_r1_original
  sub_stages:
    search_policy_rl:
      trainer:
        train_batch_size: 64
        ppo_mini_batch_size: 64
        n_samples_per_prompt: 8
        data_seed: 42
      rollout:
        stop_sequences: ["</tool_call>", "</answer>"]

resource:
  stage_resources:
    train_agent:
      impls:
        spad_rag:
          sub_stages:
            search_policy_rl:
              trainer:
                tensor_parallel_size: 1
```

resolved VERL/vLLM 配置中的 Qwen rollout 还必须满足
`actor_rollout_ref.rollout.tensor_model_parallel_size=1`；该值由现有 resource resolver 从上述
Qwen trainer 资源约束解析，不得被 scale overlay 改写为大于 1。

512 overlay 指向 512 parquet、`train_max_samples=512`；5100 overlay 指向 5100 parquet、
`train_max_samples=5100`。两者均使用同一新 350 作为轻量 val 输入，但正式 350 结果只由独立评估
脚本生成。

### 16.4 SPAD 两种规模训练脚本

以下模板始终调用同一个现有 SPAD formal task shell。SPAD 入口执行 Stage1、Stage2 和默认
Stage3 GRPO；DPO 保留在配置中，但不在这两个正式命令的 phase order 中。

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
ENTRY="${ROOT}/tasks/train_tasks/agenticIterRag/run_260709f_AIR_spad_qwen3_1_7b_glm47_formal.sh"
CFG_DIR="${ROOT}/tasks/train_tasks/agenticIterRag/configs"

cd "${ROOT}"

# 配置和资源 dry-run；必须同时检查 GRPO 主路径与保留的 DPO 配置。
bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/spad_qwen3_1_7b_glm47_512_scale_overlay.yaml" \
  --dry-run

bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/spad_qwen3_1_7b_glm47_5100_scale_overlay.yaml" \
  --dry-run

# 正式 SPAD-512：Stage1 512 -> Stage2 512 -> Stage3 GRPO。
bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/spad_qwen3_1_7b_glm47_512_scale_overlay.yaml"

# 正式 SPAD-5100：重新从 Base 开始，Stage1 使用 5100 数据池 -> Stage2 5100 -> Stage3 GRPO。
bash "${ENTRY}" \
  --OVERLAY_YAML="${CFG_DIR}/spad_qwen3_1_7b_glm47_5100_scale_overlay.yaml"
```

SPAD 正式命令同样必须把自动 HF 转换包含在入口生命周期内：Stage1 转换成功后才能进入 Stage2，
Stage3 GRPO 转换成功后整个 SPAD 命令才能返回 0。Stage1 与 Stage3 manifest 都必须记录
`hf_actor_checkpoint`；不能依赖运行者另开 shell 手工转换。

两个 SPAD scale overlay 的最终配置必须明确包含：

```yaml
agent_training:
  reward:
    type: spad_em_teacher_backoff
  sub_stages:
    search_policy_rl:
      trainer:
        reward_manager: batch
        train_batch_size: 64
        ppo_mini_batch_size: 64
        n_samples_per_prompt: 8
        data_seed: 42
        preserve_full_rollout_logs: true
      rollout:
        stop_sequences: ["</tool_call>", "</answer>"]
    answer_distillation:
      phase_order: [grpo]
      phases:
        grpo:
          enabled: true
          reward_type: gold_answer_f1
          train_batch_size: 64
          ppo_mini_batch_size: 64
          n_samples_per_prompt: 8
        dpo:
          enabled: true

resource:
  stage_resources:
    train_agent:
      impls:
        spad_rag:
          sub_stages:
            search_policy_rl:
              trainer:
                tensor_parallel_size: 1
            answer_distillation:
              phases:
                grpo:
                  trainer:
                    tensor_parallel_size: 1
```

两个 Qwen GRPO phase 的 resolved VERL/vLLM 配置均必须满足
`actor_rollout_ref.rollout.tensor_model_parallel_size=1`。GLM teacher 的现有 TP=2 保持不变。

512 overlay 还必须设置 `train_max_samples=512` 和
`answer_refresh_data.inputs.max_samples=512`；5100 overlay 对应设置为 5100。正式 SPAD 命令完成后，
先校验 Stage1 rollout manifest，再允许入口继续认定 Stage2/Stage3 产物为正式有效结果。

### 16.5 350 三次评估脚本

评估统一调用现有 `eval_spad_agent_search_350.sh`。该入口虽然命名包含 SPAD，但接受任意 HF agent
checkpoint，因此同样用于 Base 和 Search-R1。所有模型顺序执行，避免争用同一组 8 张卡和固定端口。

运行前只能从各训练 run 的 manifest/phase manifest 中取得由训练入口自动登记的
`hf_actor_checkpoint`，再显式设置以下变量；不能填写人工转换出来但未登记的目录：

```bash
export SEARCH_R1_512_HF=/absolute/path/to/search_r1_512/global_step_8_hf
export SPAD_512_STAGE1_HF=/absolute/path/to/spad_512/stage1/global_step_8_hf
export SPAD_512_STAGE3_HF=/absolute/path/to/spad_512/stage3_grpo_final_hf
export SEARCH_R1_5100_HF=/absolute/path/to/search_r1_5100/global_step_79_hf
export SPAD_5100_STAGE1_HF=/absolute/path/to/spad_5100/stage1/global_step_79_hf
export SPAD_5100_STAGE3_HF=/absolute/path/to/spad_5100/stage3_grpo_final_hf
```

正式评估模板：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
PY=/data05/conda/envs/ms/ms_agt_rag/bin/python
ENTRY="${ROOT}/tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh"
DATA="${ROOT}/data/global_train_eval_data/350e/co_search_ablation.eval.parquet"
BASE_MODEL="${ROOT}/models/llm/Qwen3-1.7B"
DATE_TAG="$(date +%y%m%d)"

: "${SEARCH_R1_512_HF:?set SEARCH_R1_512_HF from the training manifest}"
: "${SPAD_512_STAGE1_HF:?set SPAD_512_STAGE1_HF from the training manifest}"
: "${SPAD_512_STAGE3_HF:?set SPAD_512_STAGE3_HF from the training manifest}"
: "${SEARCH_R1_5100_HF:?set SEARCH_R1_5100_HF from the training manifest}"
: "${SPAD_5100_STAGE1_HF:?set SPAD_5100_STAGE1_HF from the training manifest}"
: "${SPAD_5100_STAGE3_HF:?set SPAD_5100_STAGE3_HF from the training manifest}"

cd "${ROOT}"
test -s "${DATA}"

NAMES=(
  base_qwen3_1_7b
  search_r1_512
  spad_512_stage1
  spad_512_stage3_grpo
  search_r1_5100
  spad_5100_stage1
  spad_5100_stage3_grpo
)

MODELS=(
  "${BASE_MODEL}"
  "${SEARCH_R1_512_HF}"
  "${SPAD_512_STAGE1_HF}"
  "${SPAD_512_STAGE3_HF}"
  "${SEARCH_R1_5100_HF}"
  "${SPAD_5100_STAGE1_HF}"
  "${SPAD_5100_STAGE3_HF}"
)

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  model="${MODELS[$idx]}"
  test -e "${model}"

  for replica in 1 2 3; do
    task="${DATE_TAG}-${name}-run${replica}-eval350"

    bash "${ENTRY}" \
      --agent-model "${model}" \
      --data-path "${DATA}" \
      --max-samples 350 \
      --task-name "${task}" \
      --agent-gpu-ids 0,1,2,3,4,5 \
      --agent-instance-count 6 \
      --recall-gpu-ids 6,7 \
      --infer-batch-size 96 \
      --agent-port 8240 \
      --proxy-port 8230

    summary="${ROOT}/log/eval/agenticIterRag/${task}/trace/summary.json"
    test -s "${summary}"
    "${PY}" - "${summary}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert int(summary["success_count"]) == 350, summary
assert int(summary["failure_count"]) == 0, summary
assert int(summary["micro"]["n"]) == 350, summary
assert len(summary["by_data_source"]) == 7, summary["by_data_source"].keys()
print(
    f"eval passed: n={summary['micro']['n']:.0f} "
    f"em={summary['micro']['em']:.6f} f1={summary['micro']['f1']:.6f}"
)
PY
  done
done
```

评估入口已经固定 `TEMPERATURE=0`、`TOP_P=1`、top-50 recall、top-5 visible evidence、最多 6 个
assistant turn 和 full trace。三次运行只更改 task name，不更改 checkpoint、数据或推理参数。

### 16.6 运行后核对顺序

每个正式训练/评估命令完成后按以下顺序检查：

1. 入口脚本 SHA-256 仍与 16.1 一致。
2. `pipeline.final_config.yaml` 中的数据路径、样本数、batch、reward、phase order 正确。
3. 512/5100 训练 manifest 中实际唯一问题数分别为 512/5056；5100 输入池仍为 5100。
4. SPAD Stage1 rollout manifest 分别为 4096/40448 条，必需日志字段完整。
5. Search-R1 Stage1、SPAD Stage1 和 SPAD Stage3 的 manifest 均含自动生成的
   `hf_actor_checkpoint`、原始 checkpoint、转换日志和 hash；HF 目录能由 Hugging Face/vLLM 加载。
6. 每次 eval 的 `success_count=350`、`failure_count=0`，七个数据源各有结果。
7. 只有上述检查全部通过，才将 run 纳入最终对比报告。

评估前可用以下脚本检查六个训练 phase manifest。路径必须来自实际训练输出，不能指向人工补建目录：

```bash
#!/usr/bin/env bash
set -euo pipefail

PY=/data05/conda/envs/ms/ms_agt_rag/bin/python
: "${SEARCH_R1_512_MANIFEST:?set Search-R1-512 Stage1 manifest}"
: "${SEARCH_R1_5100_MANIFEST:?set Search-R1-5100 Stage1 manifest}"
: "${SPAD_512_STAGE1_MANIFEST:?set SPAD-512 Stage1 manifest}"
: "${SPAD_512_STAGE3_MANIFEST:?set SPAD-512 Stage3 GRPO manifest}"
: "${SPAD_5100_STAGE1_MANIFEST:?set SPAD-5100 Stage1 manifest}"
: "${SPAD_5100_STAGE3_MANIFEST:?set SPAD-5100 Stage3 GRPO manifest}"

"${PY}" - \
  "${SEARCH_R1_512_MANIFEST}" \
  "${SEARCH_R1_5100_MANIFEST}" \
  "${SPAD_512_STAGE1_MANIFEST}" \
  "${SPAD_512_STAGE3_MANIFEST}" \
  "${SPAD_5100_STAGE1_MANIFEST}" \
  "${SPAD_5100_STAGE3_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

for manifest_arg in sys.argv[1:]:
    manifest_path = Path(manifest_arg)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload.get("outputs", payload)
    assert record.get("status") == "completed", (manifest_path, record.get("status"))
    raw_path = record.get("raw_actor_checkpoint") or record.get("actor_checkpoint")
    hf_value = record.get("hf_actor_checkpoint")
    assert raw_path, f"missing raw checkpoint in {manifest_path}"
    assert hf_value, f"missing automatic hf_actor_checkpoint in {manifest_path}"
    hf_path = Path(hf_value)
    assert hf_path.is_dir(), hf_path
    assert (hf_path / "config.json").is_file(), hf_path
    assert (hf_path / "tokenizer_config.json").is_file(), hf_path
    has_weights = any(
        (hf_path / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    assert has_weights, f"missing HF weights in {hf_path}"
    AutoConfig.from_pretrained(hf_path, trust_remote_code=True, local_files_only=True)
    AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, local_files_only=True)
    conversion = record.get("hf_conversion") or {}
    assert conversion.get("status") in {"converted", "validated_existing"}, (manifest_path, conversion)
    assert conversion.get("log_path"), f"missing conversion log in {manifest_path}"
    print(f"automatic HF checkpoint passed: {manifest_path} -> {hf_path}")
PY
```

该静态检查之后，正式评估首次启动 vLLM 即完成实际权重加载验收。任一 manifest/HF 检查失败都应
回到训练 runner 修复自动 finalizer，而不是在评估脚本外增加临时转换步骤。

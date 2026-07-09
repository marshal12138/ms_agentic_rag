# AIR LLM Reranker 两阶段 Reward 重构计划

## 1. 背景和目标

现在这版 LLM reranker 训练流程已经能跑起来，但暴露出一个很核心的问题：我们把太多事情都压在一个 reward 函数里了。

当前真实训练里，模型输出排序之后，会先经过格式解析，然后如果格式对，就把排序后的 top5 文档塞回 frozen agent 的历史上下文里，让 frozen agent 继续 rollout，最后用 agent 的 answer F1 给 reranker 打分。这个流程从想法上是对的，但训练上太硬了：

- 模型一开始连 `<reason>/<rerank>` 格式都不稳定。
- 格式错就直接 `-0.5`，不触发后续 rollout。
- 如果一个 prompt 的 `n=16` 个采样全都格式错，GRPO 的组内 advantage 基本就是 0。
- 这会导致训练很慢，而且早期几乎没有有效学习信号。

所以这次要重构 reward 和训练流程。新的思路更分层：

1. 先让模型学会输出合法的 reranker 动作格式。
2. 再让模型在合法动作基础上，通过 agentic RAG rollout 得到最终任务 reward。

新的训练仍然挂在原来的顶层 `train_llm_reranker` stage 里，不新增 pipeline 顶层 stage。

## 2. 最重要的边界

这次改造只允许影响 LLM reranker 训练 stage 内部逻辑，以及 service bundle 读取最终 reranker checkpoint 的必要逻辑。

明确不改：

- 不改 `generate_traces`。
- 不改 `build_reranker_dataset`。
- 不改 `build_reranker_branch_dataset`。
- 不改 `train_agent`。
- 不改 `infer_matrix`。
- 不改 data produce launcher。
- 不改 trace generation backend。
- 不改 dataset builder 行为。
- 不改 branch dataset builder 行为。
- 不改 branch dataset schema。
- 不改 pipeline 顶层 stage 顺序。
- 不改 `run_260702a_AIR_v1_dataproduce.sh`。

如果实现时发现必须碰这些区域，需要先停下来重新评估，不能顺手改。

这点很关键。因为用户已经明确要求：LLM reranker stage 的改动不能影响其他任何顶层 stage，尤其不能影响数据生产相关 stage。

默认的 `run_260703b` 行为也保持：

```bash
selected_stages:
  - train_llm_reranker
  - build_service_bundle
```

默认不能重新进入：

```bash
generate_traces
build_reranker_dataset
build_reranker_branch_dataset
```

如果以后确实需要重跑数据生产，必须显式传：

```bash
--pipeline.resume_from_stage=generate_traces
```

## 3. 新 Reward 架构

新的 reward 架构里有两个同级别 reward function：

1. `reranker_format_reward`
2. `agentic_rag_rollout_reward`

旧的 `answer_reward` 和 `delta_answer_reward` 不再作为顶层 reward function 使用，而是变成 `agentic_rag_rollout_reward` 里的子策略。

这版是对旧 LLM reranker 训练流程的覆盖，不做旧 reward 方案兼容。

### 3.1 `reranker_format_reward`

`reranker_format_reward` 主要管格式，同时加一个很轻的长度约束。

它不关心文档排序质量，也不关心 answer 是否正确。它只关心两件事：

1. 输出是不是一个可执行的 reranker action。
2. 在格式正确的前提下，输出是不是足够短。

它不会做这些事情：

- 不启动 frozen agent。
- 不调用 retriever。
- 不构造新的 tool observation。
- 不跑 continuation。
- 不抽取 answer。
- 不计算 F1。
- 不读取 baseline reward。

它只做：

```text
reranker output -> parser -> response length check -> score
```

分数定义：

```text
格式合法，且输出长度 <= 512 token：1.0
格式合法，但输出长度 > 512 token：0.5
格式不合法：-0.5
```

这里的“长度”默认按 response token 数计算。优先使用 VERL/rollout 已经统计出来的 response token length；如果训练框架拿不到这个值，再用 reranker tokenizer 对完整输出重新 tokenize。

这个设计有两个目的：

- `max_response_length=1024` 仍然是训练硬上限，避免模型因为 512 token 不够而被强行截断。
- `length_threshold_tokens=512` 是 reward 偏好，鼓励模型在格式正确的前提下把 reasoning 和 rerank 输出写得更紧凑。

格式要求和 `agentic_rag_rollout_reward` 里使用的格式要求完全一致，必须复用同一个 parser。

合法输出仍然是：

```text
<reason>
这里写排序理由
</reason>
<rerank>
[1] > [2] > [3] > ... > [50]
</rerank>
```

parser 需要检查：

- 必须有 `<reason>` 和 `</reason>`。
- 必须有 `<rerank>` 和 `</rerank>`。
- `<reason>` 必须在 `<rerank>` 前面。
- reason 内容不能为空。
- rerank 内容不能为空。
- 必须正好有 50 个 index。
- index 必须在 `[1, 50]` 范围内。
- index 不能重复。
- `<rerank>` 里只能出现 `[数字]`、`>` 和空白。
- 逗号、JSON、doc_id、自然语言混入都算格式错误。

这个 reward 的目的很直接：先让模型知道什么叫“可执行的 reranker action”，同时不要养成无意义写很长 reason 的习惯。

### 3.2 `agentic_rag_rollout_reward`

`agentic_rag_rollout_reward` 是完整的 agentic reward。

它的名字强调的是 rollout，而不是 delta。因为这个 reward 真正做的是：

```text
把 reranker 排序动作放进 agentic RAG 后续 rollout 里，再根据 rollout 结果打分。
```

完整流程：

```text
reranker 输出
-> parser 检查格式
-> 把 index 排序映射回 doc_id/doc 内容
-> 取 reranker 排序后的 top5 docs
-> 构造新的 tool observation
-> 拼接 messages_before_tool_response + new_tool_message
-> frozen search agent 继续 rollout
-> 后续 search 只走 retriever
-> frozen agent 产出最终 answer
-> answer 与 gold answer 算分
-> 返回 reranker reward
```

格式错误时：

```text
score = format_invalid_score
默认 -0.5
不触发 continuation
不调用 frozen agent
不调用 retriever
```

格式正确后，进入子策略。

#### 子策略 1：`answer_reward`

```text
score = F1(new_answer, gold_answers)
```

这是 stage2 的默认子策略。

#### 子策略 2：`delta_answer_reward`

```text
score = F1(new_answer, gold_answers) - baseline_reward
```

如果选择 `delta_answer_reward`，但样本的 `extra_info` 里没有 `baseline_reward`，必须直接报错。

不能 silent fallback 到 `answer_reward`，否则训练语义会不可信。

## 4. 两阶段训练流程

顶层 pipeline 不新增 stage。

现有顶层 stage 仍然是：

```text
train_llm_reranker
```

但 `train_llm_reranker` 内部改成两个 phase：

```text
stage1_format
-> stage2_agentic
```

只执行 `enabled: true` 的 phase。

默认行为：

```text
执行 stage1_format
跳过 stage2_agentic
```

### 4.1 `stage1_format`

stage1 用来训练格式和长度偏好。

默认：

- 开启。
- 从 `reranker_training.base_model` 初始化。
- 使用 `reranker_format_reward`。
- 默认全量 1 epoch。
- 默认读取全部 branch samples。
- 使用 8 张 NPU 训练，优先把 stage1 的格式训练吞吐打满。
- 训练参数底线是 `train_batch_size=64` 和 `max_response_length=1024`，这两个值不是消融项。
- reward 规则是：格式正确且输出长度不超过 512 token 给 `1.0`，格式正确但超过 512 token 给 `0.5`，格式错误给 `-0.5`。
- 不启动 frozen agent。
- 不启动 retriever。
- 不启动 recall proxy。
- 不注入 continuation 环境变量。

stage1 的输出目录建议：

```text
stages/train_llm_reranker/reranker_model_verl/stage1_format
```

stage1 训练完成后，需要产出 checkpoint。这个 checkpoint 是 stage2 的默认初始化模型。

### 4.2 `stage2_agentic`

stage2 用来训练 agentic RAG rollout reward。

默认：

- 关闭。
- 如果开启，从 stage1 checkpoint 初始化。
- 使用 `agentic_rag_rollout_reward`。
- 默认子策略是 `answer_reward`。
- 默认全量 1 epoch。
- 默认读取全部 branch samples。
- 训练参数同样以 `train_batch_size=64` 和 `max_response_length=1024` 为底线。
- 其他吞吐相关参数可以消融，但 stage2 默认关闭，避免误启动超长 agentic rollout 训练。

stage2 开启后才会启动：

- frozen agent vLLM。
- retriever backend。
- recall proxy。

stage2 才需要注入 continuation 环境变量，例如：

```text
AIR_CONTINUATION_AGENT_MODEL
AIR_CONTINUATION_TOKENIZER_PATH
AIR_CONTINUATION_AGENT_BASE_URL
AIR_CONTINUATION_AGENT_SERVED_MODEL
AIR_CONTINUATION_RETRIEVAL_URL
AIR_CONTINUATION_CANDIDATE_TOP_N
AIR_CONTINUATION_VISIBLE_TOP_M
AIR_CONTINUATION_MAX_ASSISTANT_TURNS
AIR_CONTINUATION_MAX_USER_TURNS
AIR_CONTINUATION_MAX_PROMPT_LENGTH
AIR_CONTINUATION_MAX_RESPONSE_LENGTH
AIR_CONTINUATION_MAX_TOOL_RESPONSE_LENGTH
AIR_CONTINUATION_TEMPERATURE
AIR_CONTINUATION_TOP_P
AIR_CONTINUATION_REQUEST_TIMEOUT
AIR_CONTINUATION_ENABLE_THINKING
```

stage2 的输出目录建议：

```text
stages/train_llm_reranker/reranker_model_verl/stage2_agentic
```

最终 checkpoint 选择规则：

```text
如果 stage2 开启并成功完成，用 stage2 checkpoint。
否则，用 stage1 checkpoint。
```

## 5. 配置设计

旧的单 reward 配置不再兼容：

```yaml
reranker_training:
  reward:
    strategy: answer_reward
    format_penalty: -0.5
```

新的配置结构放在：

```yaml
reranker_training.training_phases
```

所有新增 YAML 配置项都必须写中文注释。不能新增无注释配置项。

注释风格参考现有 AIR 配置文件，尤其是：

- `AgenticIterRag/config/pipeline/offline_two_stage.yaml`
- `AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml`
- `tasks/train_tasks/agenticIterRag/configs/from_dataprod_to_reranker_training_overlay.yaml`

### 5.1 YAML 草案

```yaml
reranker_training:
  # LLM reranker 的基座模型路径；stage1_format 默认从这里初始化。
  base_model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B

  # 两阶段训练配置；注意这是 train_llm_reranker stage 内部 phase，不是 pipeline 顶层 stage。
  training_phases:
    stage1_format:
      # 是否启用格式训练阶段；默认启用，先让模型学会输出合法 reranker action。
      enabled: true

      # 当前 phase 使用的 reward 函数名称；stage1 检查格式，并对超过 512 token 的合法输出降分。
      reward_name: reranker_format_reward

      # 格式错误时的分数；stage1 格式错误也给负分，直接告诉模型这个 action 不可执行。
      format_invalid_score: -0.5

      # 格式正确且输出长度不超过阈值时的分数；这是 stage1 的最高奖励。
      short_valid_score: 1.0

      # 格式正确但输出长度超过阈值时的分数；保留正分，但弱于短输出，鼓励模型更简洁。
      long_valid_score: 0.5

      # stage1 的长度奖励阈值，按 response token 数计算；max_response_length 仍保持 1024 作为硬上限。
      length_threshold_tokens: 512

      # 当前 phase 的初始化模型；base_model 表示使用 reranker_training.base_model。
      init_model: base_model

      # 默认训练完整 1 个 epoch；如果做 smoke，可用 CLI 临时覆盖 total_training_steps=1。
      total_epochs: 1

      # null 表示交给 VERL 按数据集长度和 total_epochs 自动计算总 step。
      total_training_steps: null

      # -1 表示读取全部 branch samples。
      train_max_samples: -1

      # 训练 batch size 底线；stage1 默认每步读取 64 个 prompt。
      train_batch_size: 64

      # PPO mini batch size；默认和 train_batch_size 对齐。
      ppo_mini_batch_size: 64

      # 每卡 actor update micro batch；先保持 1，避免长 prompt + 1024 response 时显存爆掉。
      micro_batch_size_per_gpu: 1

      # old log prob/ref log prob 每卡 micro batch；先保持 1，优先保证稳定。
      log_prob_micro_batch_size_per_gpu: 1

      # 每个 prompt 的 rollout 数；保留 16 个采样用于 GRPO 组内对比。
      n_samples_per_prompt: 16

      # 极大值表示只保存最终 checkpoint，避免频繁保存拖慢训练。
      save_freq: 1000000000

      # reranker response 最大 token 数；这是用户指定的训练底线，不参与消融。
      max_response_length: 1024

      # reranker prompt 最大 token 数；和 full50 prompt 保持一致。
      max_prompt_length: 12000

      # rollout engine 最大模型长度，默认等于 max_prompt_length + max_response_length。
      rollout_max_model_len: 13024

      # 单批最大 token 数；默认和 rollout_max_model_len 对齐，减少 vLLM-Ascend 额外变量。
      max_num_batched_tokens: 13024

      # rollout 最大并发序列数；默认 16，后续在 8/16/32 上做效率优先消融。
      max_num_seqs: 16

      # rollout vLLM/NPU 显存利用率；比旧值略提高，但不直接拉满，避免 1024 response OOM。
      rollout_gpu_memory_utilization: 0.50

    stage2_agentic:
      # 是否启用 agentic rollout reward 阶段；默认关闭，避免误启动超长训练。
      enabled: false

      # 当前 phase 使用的 reward 函数名称；stage2 会执行 agentic RAG continuation rollout。
      reward_name: agentic_rag_rollout_reward

      # 当前 phase 的初始化模型；stage1_checkpoint 表示接 stage1 的最终 checkpoint。
      init_model: stage1_checkpoint

      # agentic rollout reward 的内部打分策略；默认直接使用新 answer F1。
      sub_strategy: answer_reward

      # reranker 输出格式错误时的分数；格式错误不触发 continuation。
      format_invalid_score: -0.5

      # 默认训练完整 1 个 epoch；只有显式开启 stage2 时才生效。
      total_epochs: 1

      # null 表示交给 VERL 自动计算总 step。
      total_training_steps: null

      # -1 表示读取全部 branch samples。
      train_max_samples: -1

      # 训练 batch size 底线；stage2 默认也按 64 个 prompt 组织 batch，但 stage2 默认关闭。
      train_batch_size: 64

      # PPO mini batch size；默认和 train_batch_size 对齐。
      ppo_mini_batch_size: 64

      # 每卡 actor update micro batch；stage2 也先保持 1，避免 continuation 场景下额外显存风险。
      micro_batch_size_per_gpu: 1

      # old log prob/ref log prob 每卡 micro batch。
      log_prob_micro_batch_size_per_gpu: 1

      # 每个 prompt 的 rollout 数。
      n_samples_per_prompt: 16

      # 极大值表示只保存最终 checkpoint。
      save_freq: 1000000000

      # reranker response 最大 token 数；这是用户指定的训练底线，不参与消融。
      max_response_length: 1024

      # reranker prompt 最大 token 数。
      max_prompt_length: 12000

      # rollout engine 最大模型长度。
      rollout_max_model_len: 13024

      # 单批最大 token 数。
      max_num_batched_tokens: 13024

      # rollout 最大并发序列数；默认 16，可按 8/16/32 消融。
      max_num_seqs: 16

      # rollout vLLM/NPU 显存利用率。
      rollout_gpu_memory_utilization: 0.50

  trainer:
    # 训练后端；真实训练使用 VERL。
    backend: verl

    # 公共默认 rollout 数；phase 内显式配置时以 phase 为准。
    n_samples_per_prompt: 16

    # 公共默认 batch size 底线；phase 内显式配置时以 phase 为准。
    train_batch_size: 64

    # PPO mini batch size。
    ppo_mini_batch_size: 64

    # actor 学习率。
    learning_rate: 2.0e-5
```

### 5.2 配置合并规则

读取顺序固定：

```text
公共 trainer 配置
-> 当前 phase 配置覆盖
-> CLI dotlist 覆盖
```

例子：

```bash
--reranker_training.training_phases.stage1_format.total_training_steps=1
--reranker_training.training_phases.stage1_format.train_max_samples=64
```

这只影响 stage1，不影响 stage2。

开启 stage2：

```bash
--reranker_training.training_phases.stage2_agentic.enabled=true
```

切换 stage2 子策略：

```bash
--reranker_training.training_phases.stage2_agentic.sub_strategy=delta_answer_reward
```

### 5.3 训练参数底线和消融策略

这次训练参数调整有两个硬底线：

```text
train_batch_size = 64
max_response_length = 1024
```

这两个值不作为消融项。原因很直接：

- `train_batch_size=64` 用来把 5100 条 branch samples 的每 epoch step 数从 5100 降到大约 80。
- `max_response_length=1024` 是用户明确指定的 reranker 输出长度上限。

其他和吞吐相关的参数可以消融，目标是训练效率最高，同时不能牺牲基本稳定性。

默认建议：

```text
n_samples_per_prompt = 16
ppo_mini_batch_size = 64
micro_batch_size_per_gpu = 1
log_prob_micro_batch_size_per_gpu = 1
max_prompt_length = 12000
rollout_max_model_len = 13024
max_num_batched_tokens = 13024
max_num_seqs = 16
rollout_gpu_memory_utilization = 0.50
```

这里要特别说明一下历史依据：旧训练配置是 `train_batch_size=1`、`n=16`、`max_response_length=512`、`max_num_seqs=16`，历史日志里单 step 平均大约 385 秒，其中 rollout generation 占绝大多数时间，而且 NPU reserved memory 已经接近 61GB/64GB。因此现在把 `max_response_length` 提到 1024 后，不能默认直接把 `max_num_seqs` 拉到 64。

消融只围绕这些参数做：

```text
max_num_seqs: 8 / 16 / 32
rollout_gpu_memory_utilization: 0.50 / 0.60
max_num_batched_tokens: 13024 / 26048
```

消融底线：

- `max_num_seqs` 不低于 8。
- 不消融 `train_batch_size=64`。
- 不消融 `max_response_length=1024`。
- 优先在 stage1 做消融，因为 stage1 不启动 frozen agent/retriever，变量更少。

消融选择规则：

- 如果 `max_num_seqs=32` 稳定且吞吐最高，就用 32。
- 如果 32 OOM 或 step time 反而更慢，就用 16。
- 如果 16 也 OOM，就用 8。
- 不使用低于 8 的并发。
- 如果 `rollout_gpu_memory_utilization=0.60` 稳定且显著提速，就替换默认 0.50。
- 如果 `max_num_batched_tokens=26048` 稳定且显著提速，再替换默认 13024。

每组消融至少记录：

```text
是否 OOM
是否完成 1 step
timing_s/step
timing_s/gen
timing_s/update_actor
timing_s/old_log_prob
timing_s/ref
response_length/clip_ratio
perf/throughput
perf/max_memory_allocated_gb
perf/max_memory_reserved_gb
```

## 6. 代码改造计划

### 6.1 Reward 代码结构

建议新增目录：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/rewards/
  __init__.py
  common.py
  reranker_format_reward.py
  agentic_rag_rollout_reward.py
```

`common.py` 放公共逻辑：

- 调用 parser。
- index 到 doc_id 的映射。
- 从 rerank index 构造 ranked docs。
- top5 tool message 渲染。
- normalized F1。
- answer 抽取。
- tool call 解析。
- retriever-only search。

`reranker_format_reward.py` 提供：

```python
compute_reranker_format_reward(...)
compute_reranker_format_reward_details(...)
```

VERL 训练入口只用 float 版本：

```python
compute_reranker_format_reward(...)
```

它只返回：

```text
1.0 / 0.5 / -0.5
```

details 版本可返回：

```text
score
format_valid
format_error_code
format_error_message
response_length_tokens
length_threshold_tokens
length_penalty_applied
parse
```

`agentic_rag_rollout_reward.py` 提供：

```python
compute_agentic_rag_rollout_reward(...)
compute_agentic_rag_rollout_reward_details(...)
```

VERL 训练入口只用 float 版本，避免 non_tensor_batch 尺寸问题。

details 版本可返回：

```text
score
format_valid
format_error_code
continuation_status
answer
answer_reward
baseline_reward
sub_strategy
visible_doc_ids
assistant_turns
user_turns
search_count
elapsed_s
```

所有新增 Python 代码要补充必要中文注释，尤其是：

- 为什么 stage1 不启动 continuation。
- 为什么 stage1 格式正确短输出给 1.0、格式正确长输出给 0.5、格式错误给 -0.5。
- 为什么 `max_response_length=1024` 是硬上限，但 `length_threshold_tokens=512` 是 reward 偏好。
- 为什么 stage2 格式错误不继续 rollout。
- 为什么后续 search 禁用 reranker。
- 为什么 VERL 入口只返回 float，不返回调试 dict。

### 6.2 `trainer_entry.py`

`trainer_entry.py` 从“单次训练 runner”改成“phase runner”。

核心流程：

```text
读取 branch dataset
读取 training_phases
按 stage1_format -> stage2_agentic 顺序遍历
跳过 enabled=false 的 phase
解析 phase config
解析 phase init model
构造 VERL command
根据 phase 判断是否启动服务
启动 reporter
执行 VERL
停止 reporter
停止服务
写 phase manifest
最后写 train_llm_reranker 顶层 manifest
```

stage1 特殊点：

- `init_model=base_model`。
- reward function 是 `compute_reranker_format_reward`。
- 使用 8 张 NPU 训练，tensor parallel 和训练资源都按 8 卡规划。
- 不启动 `TrainingServiceManager.start_recall()`。
- 不启动 `TrainingServiceManager.start_frozen_agent()`。
- 不调用 `build_verl_env_vars()` 注入 continuation 环境变量。

stage2 特殊点：

- `init_model=stage1_checkpoint`。
- reward function 是 `compute_agentic_rag_rollout_reward`。
- 需要启动 recall 和 frozen agent。
- 需要注入 continuation 环境变量。

### 6.3 训练资源设计

stage1 的目标是把格式训练尽量跑快，所以默认使用 8 张卡：

```yaml
resource:
  stage_resources:
    train_llm_reranker:
      services:
        stage1_format_actor:
          # stage1 只训练 reranker，不启动 frozen agent/retriever，因此 8 张卡都给 reranker。
          gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]

          # stage1 默认 8 卡 tensor parallel，优先提升长 prompt + 1024 response 的吞吐。
          tensor_parallel_size: 8
```

stage2 默认关闭。开启 stage2 时，需要同时放置 reranker actor、frozen agent 和 retriever，所以不能直接照搬 stage1 的 8 卡独占模式。stage2 的资源建议继续用当前安全拆分：

```yaml
resource:
  stage_resources:
    train_llm_reranker:
      services:
        stage2_agentic_actor:
          # stage2 的 reranker actor 使用 0-3。
          gpu_ids: [0, 1, 2, 3]
          tensor_parallel_size: 4

        frozen_agent_vllm:
          # frozen agent 使用 4-5。
          gpu_ids: [4, 5]
          tensor_parallel_size: 2

        recall:
          # retriever 使用 6-7。
          gpu_ids: [6, 7]
```

实现上可以有两种方式：

1. 在同一个 `train_llm_reranker.services` 下新增 phase 级 actor 配置，例如 `stage1_format_actor` 和 `stage2_agentic_actor`。
2. 或者保留 `reranker_actor` 作为默认，但允许 phase 覆盖 `gpu_ids/tensor_parallel_size`。

推荐第一种。原因是它更清楚，不会让 stage1 的 8 卡配置误伤 stage2 的 frozen agent/retriever 放置。

资源隔离要求：

- stage1 不监听 `8130/8131/8132/8140`。
- stage1 不启动 retriever。
- stage1 不启动 frozen agent。
- stage2 默认关闭。
- stage2 开启后才允许占用 `8140` 和 `8130/8131/8132`。
- stage1 和 stage2 的 resource plan 必须写入各自 phase manifest，方便审计。

### 6.4 VERL command 生成

当前不能再硬编码：

```text
compute_air_branch_continuation_reward
```

而是要由 phase 决定：

stage1：

```text
custom_reward_function.path = .../rewards/reranker_format_reward.py
custom_reward_function.name = compute_reranker_format_reward
actor_rollout_ref.rollout.tensor_model_parallel_size = 8
trainer.n_gpus_per_node = 8
```

stage2：

```text
custom_reward_function.path = .../rewards/agentic_rag_rollout_reward.py
custom_reward_function.name = compute_agentic_rag_rollout_reward
actor_rollout_ref.rollout.tensor_model_parallel_size = 4
trainer.n_gpus_per_node = 4
```

stage1 reward kwargs：

```text
expected_n = 50
format_invalid_score = -0.5
short_valid_score = 1.0
long_valid_score = 0.5
length_threshold_tokens = 512
```

stage2 reward kwargs：

```text
expected_n = 50
visible_top_m = 5
sub_strategy = answer_reward
format_invalid_score = -0.5
```

### 6.5 Manifest

每个 phase 写独立 manifest。

建议路径：

```text
stages/train_llm_reranker/reranker_model_verl/stage1_format/phase_manifest.json
stages/train_llm_reranker/reranker_model_verl/stage2_agentic/phase_manifest.json
```

phase manifest 至少包含：

```text
phase_name
reward_name
enabled
status
init_model
output_dir
checkpoint
branch_dataset_manifest
sample_count
n_samples_per_prompt
total_epochs
total_training_steps
train_max_samples
verl_command_plan
runtime_dir
verl_log
training_report_paths
service_outputs
config_hash
```

顶层 `train_llm_reranker/manifest.json` 汇总：

```text
status
backend
completed_phases
skipped_phases
failed_phase
stage1_checkpoint
stage2_checkpoint
final_reranker_checkpoint
reranker_model
reranker_checkpoint
phase_manifests
branch_dataset_manifest
sample_count
n_samples_per_prompt
config_hash
```

为了让 service bundle 简单稳定，顶层 manifest 必须继续提供：

```text
reranker_model
reranker_checkpoint
```

但它们的值来自：

```text
stage2 完成时 -> stage2 checkpoint
否则 -> stage1 checkpoint
```

### 6.6 Service Bundle

`build_service_bundle` 不需要理解 phase 细节。

它只需要读取 `train_llm_reranker` 顶层 manifest 里的：

```text
reranker_model
reranker_checkpoint
```

这样 service bundle 不会因为训练内部 phase 变化而复杂化。

## 7. 日志和报告

报告目录按 phase 分开。

建议：

```text
stages/train_llm_reranker/training_reports/stage1_format/
stages/train_llm_reranker/training_reports/stage2_agentic/
```

stage1 曲线重点：

- format reward
- response length
- response clip ratio
- loss
- throughput
- step time

stage2 曲线重点：

- agentic rollout reward
- answer reward
- delta reward
- continuation status
- response length
- response clip ratio
- loss
- throughput
- step time

曲线文件名必须带 phase，避免互相覆盖：

```text
air_llm_reranker.stage1_format.metrics.latest_reranker_rewards.png
air_llm_reranker.stage1_format.metrics.latest_reranker_losses.png
air_llm_reranker.stage1_format.metrics.latest_reranker_lengths.png
air_llm_reranker.stage1_format.metrics.latest_reranker_performance.png
```

stage2 同理。

reporter 失败不能打断训练，但必须写入当前 phase 的 report manifest。

## 8. 默认运行行为

`run_260703b` 当前默认从 `train_llm_reranker` 开始，这个保持不变。

默认运行行为变成：

```text
读取已有 branch dataset
执行 stage1_format
跳过 stage2_agentic
训练完成后 build_service_bundle
```

默认不会重新执行：

```text
generate_traces
build_reranker_dataset
build_reranker_branch_dataset
```

如果要做 1-step smoke，可以用：

```bash
RUN_260703B_SMOKE_REAL=1 \
bash tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh
```

后续实现时，`RUN_260703B_SMOKE_REAL=1` 应覆盖 stage1 的：

```text
train_max_samples=64
total_training_steps=1
save_freq=1
```

默认不要覆盖 stage2，因为 stage2 默认关闭。

## 9. 测试计划

这版计划不能只停在 dry-run 或 mock。实现完成后必须跑通使用 NPU 的真实训练，至少包括 stage1 的真实 VERL 训练 1 step。否则只能算代码路径写完，不能算训练链路验收通过。

### 9.1 单元测试

`reranker_format_reward`：

- 合法格式且 response token length <= 512 返回 `1.0`。
- 合法格式但 response token length > 512 返回 `0.5`。
- 缺 `<reason>` 返回 `-0.5`。
- 缺 `<rerank>` 返回 `-0.5`。
- 标签顺序错误返回 `-0.5`。
- reason 为空返回 `-0.5`。
- rerank 为空返回 `-0.5`。
- index 少于 50 返回 `-0.5`。
- index 多于 50 返回 `-0.5`。
- index 重复返回 `-0.5`。
- index 越界返回 `-0.5`。
- rerank 中出现逗号、JSON、doc_id 返回 `-0.5`。
- response token length 缺失时，能用 tokenizer fallback 计算长度。
- response token length 和 tokenizer fallback 都不可用时，必须 fail-fast 报错，不能默认当成短输出给 1.0。

`agentic_rag_rollout_reward`：

- 和 `reranker_format_reward` 使用同一个 parser。
- 格式错误时返回 `format_invalid_score`。
- 格式错误时不调用 continuation。
- 格式正确时会构造 top5 tool message。
- `answer_reward` 子策略计算正确。
- `delta_answer_reward` 子策略计算正确。
- `delta_answer_reward` 缺 baseline 时失败。

### 9.2 dry-run 测试

默认：

```bash
bash tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh --dry-run
```

必须满足：

```text
selected_stages:
  - train_llm_reranker
  - build_service_bundle
```

不能出现：

```text
generate_traces
build_reranker_dataset
build_reranker_branch_dataset
```

训练 phase 必须是：

```text
stage1_format enabled
stage2_agentic skipped
```

stage1 VERL command 必须使用：

```text
compute_reranker_format_reward
```

stage1 command 不能包含 continuation 环境变量。

stage1 参数必须能在 final config 和 VERL command plan 里看到：

```text
train_batch_size = 64
ppo_mini_batch_size = 64
n_samples_per_prompt = 16
max_response_length = 1024
length_threshold_tokens = 512
format_invalid_score = -0.5
short_valid_score = 1.0
long_valid_score = 0.5
max_prompt_length = 12000
rollout_max_model_len = 13024
max_num_batched_tokens = 13024
max_num_seqs = 16
```

stage1 资源必须能在 final config 和 VERL command plan 里看到：

```text
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
tensor_parallel_size = 8
trainer.n_gpus_per_node = 8
```

开启 stage2：

```bash
--reranker_training.training_phases.stage2_agentic.enabled=true
```

必须满足：

- stage2 command 使用 `compute_agentic_rag_rollout_reward`。
- stage2 init model 指向 stage1 checkpoint。
- stage2 command 包含 continuation 环境变量。
- stage2 资源使用安全切分，默认 reranker actor 不和 frozen agent、retriever 抢同一张卡。

### 9.3 数据生产隔离测试

必须跑：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh --dry-run
```

验收：

- selected stages 不变。
- 不因为新增 `training_phases` 影响数据生产。
- `generate_traces` dry-run manifest schema 不变。
- `build_reranker_dataset` dry-run manifest schema 不变。
- `build_reranker_branch_dataset` 不被 `run_260702a` 误触发。

还要检查：

```bash
bash tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh --dry-run
```

验收：

- 默认不重新数据生产。
- 默认 branch dataset manifest 能读取。
- 默认 service bundle 使用 stage1 checkpoint。

### 9.4 小样本真实测试

stage1 NPU 真实训练 smoke：

- 必须使用真实 NPU 跑 VERL，不允许用 mock backend 代替。
- 至少 64 samples，保证 `train_batch_size=64` 真的生效。
- 1 step。
- 使用 8 张 NPU。
- 保持 `train_batch_size=64`。
- 保持 `max_response_length=1024`。
- 保持 `length_threshold_tokens=512`。
- smoke 只允许覆盖 `train_max_samples=64`、`total_training_steps=1` 和 `save_freq=1`。
- 不启动 `8130 / 8131 / 8132 / 8140`。
- 生成 stage1 checkpoint。
- 生成 phase manifest。
- 生成 metrics。
- 生成曲线。
- VERL 日志里能看到真实 step 完成。
- reward 分布里至少能看到 `reranker_format_reward` 指标。
- 训练结束后无残留 VERL/reporter 进程。

stage2 smoke：

- 显式开启 stage2。
- 必须使用真实 NPU 跑 VERL，不允许用 mock backend 代替。
- 至少 64 samples，保证 `train_batch_size=64` 真的生效。
- 1 step。
- 从 stage1 checkpoint 初始化。
- 启动 frozen agent/retriever/proxy。
- 训练结束后停止 frozen agent/retriever/proxy。
- 端口 `8130 / 8131 / 8132 / 8140` 无残留。

### 9.5 训练效率消融测试

这一组测试只针对 stage1。原因很简单：stage1 默认开启，而且不需要 frozen agent 和 retriever，最适合先把 LLM reranker 本身的训练吞吐打满。

固定不动的底线参数：

```text
train_batch_size = 64
max_response_length = 1024
n_samples_per_prompt = 16
```

优先消融这些吞吐参数：

```text
max_num_seqs = 8 / 16 / 32
rollout_gpu_memory_utilization = 0.50 / 0.60
max_num_batched_tokens = 13024 / 26048
```

每组消融用小样本跑：

```text
train_max_samples = 64
total_training_steps = 1
save_freq = 1
```

每组必须记录：

- step 总耗时。
- generation 耗时。
- actor update 耗时。
- response clip ratio。
- 平均 prompt length。
- 平均 response length。
- NPU 显存 reserved / allocated。
- 是否 OOM。
- 是否出现大量截断。

选择规则：

- 如果某组 OOM，直接淘汰。
- 如果 response clip ratio 明显升高，说明 `max_response_length=1024` 仍可能不够或采样太发散，要单独记录，但不能把 `max_response_length` 降回 512。
- 如果 `max_num_seqs=32` 比 16 快且不 OOM，就用 32。
- 如果 32 OOM 或吞吐没有明显收益，就保持 16。
- `max_num_seqs` 不建议低于 8；低于 8 基本是在浪费 8 卡训练资源。

## 10. 实施顺序

建议按这个顺序实现，降低风险：

1. 新增 reward 目录和两个 reward function，先做单元测试。
2. 改配置结构，写足中文注释。
3. 改 `trainer_entry.py` 支持 phase runner。
4. 改 VERL command 生成逻辑，让 reward function 和 init model 都由 phase 决定。
5. 改报告路径，按 phase 隔离。
6. 改顶层 training manifest 汇总 phase 信息。
7. 确认 service bundle 只读顶层 `reranker_model/reranker_checkpoint`。
8. 跑 dry-run。
9. 跑 stage1 NPU 真实训练 1-step smoke，必须使用 8 张 NPU 和 `train_batch_size=64`。
10. 跑 stage1 训练效率消融，把 `max_num_seqs`、`rollout_gpu_memory_utilization`、`max_num_batched_tokens` 调到当前资源下的高吞吐组合。
11. 显式开启 stage2 跑 NPU 真实训练 1-step smoke。
12. 跑 dataproduce dry-run 回归，确认数据生产链路不受影响。

## 11. 验收标准

必须全部满足：

- 默认 `run_260703b --dry-run` 只包含 `train_llm_reranker` 和 `build_service_bundle`。
- 默认只执行 `stage1_format`。
- 默认不启动 frozen agent 和 retriever。
- 默认不进入数据生产相关 stage。
- stage1 reward 是 `reranker_format_reward`。
- stage1 reward 规则必须是：格式正确且输出长度 <= 512 token 给 `1.0`，格式正确但输出长度 > 512 token 给 `0.5`，格式错误给 `-0.5`。
- stage1 默认使用 8 张 NPU。
- stage1 默认 `train_batch_size=64`。
- stage1 默认 `max_response_length=1024`。
- stage1 的效率消融不能降低 `train_batch_size` 和 `max_response_length` 这两个底线。
- stage1 必须跑通使用 NPU 的真实 VERL 训练，至少完成 1 个真实训练 step。
- stage2 reward 是 `agentic_rag_rollout_reward`。
- stage2 默认关闭。
- stage2 开启后从 stage1 checkpoint 初始化。
- stage2 如果开启，也必须跑通使用 NPU 的真实 VERL 训练 smoke。
- stage2 开启后也必须保持 `train_batch_size=64` 和 `max_response_length=1024` 的底线，除非后续另开文档明确说明资源不满足。
- service bundle 使用最终 checkpoint。
- 所有新增 YAML 字段都有中文注释。
- 所有新增关键 Python 分支都有中文注释。
- `run_260702a_AIR_v1_dataproduce.sh --dry-run` 行为不变。
- 旧的 `compute_air_branch_continuation_reward` 不再作为训练 reward 入口使用。

## 12. Assumptions

- 这版是对旧 LLM reranker 训练流程的覆盖，不做旧方案兼容。
- 顶层 pipeline stage 不新增、不重排。
- 改动只允许影响 `train_llm_reranker` 和 service bundle 的 checkpoint 读取。
- stage1 默认全量 1 epoch。
- stage1 默认使用 8 张 NPU。
- `train_batch_size=64` 和 `max_response_length=1024` 是训练参数底线，不作为消融项。
- `length_threshold_tokens=512` 是 stage1 reward 偏好，不是生成硬截断；生成硬上限仍是 `max_response_length=1024`。
- stage1 格式错误默认给 `-0.5`，而不是 0 分。
- 其他吞吐参数按效率优先消融，包括 `max_num_seqs`、`rollout_gpu_memory_utilization`、`max_num_batched_tokens`。
- `max_num_seqs` 原则上不低于 8，避免 8 卡 stage1 训练资源利用率过低。
- stage2 默认关闭。
- stage2 默认从 stage1 checkpoint 继续训练。
- stage2 默认子策略是 `answer_reward`。
- `reranker_format_reward` 和 `agentic_rag_rollout_reward` 必须复用同一个 parser。
- 所有新增配置项必须写中文注释。
- 所有新增代码的关键逻辑必须写中文注释。

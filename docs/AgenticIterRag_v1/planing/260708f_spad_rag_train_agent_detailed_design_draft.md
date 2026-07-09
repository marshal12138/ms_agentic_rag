# SPAD-RAG Train-Agent 详细设计 Draft

日期：2026-07-08

状态：Draft

本文档是 SPAD-RAG 的工程详细设计草案。基础算法思路见：

```text
docs/AgenticIterRag_v1/planing/260708e_spad_rag_search_policy_rl_answer_distillation_draft.md
```

这里重点讨论配置结构、代码拆分、prompt 设计、服务资源管理和 AIR pipeline 接入方式。

## 1. 核心结论

SPAD-RAG 三阶段不做成新的顶层 pipeline，也不把完整三阶段配置全部塞进 `offline_two_stage.yaml`。

正确关系是：

```text
offline_two_stage.yaml
  train_agent
    impl: spad_rag
    impl_config_ref: agent_training

agent_training/spad_rag_base.yaml
  SPAD-RAG 内部 sub-stage DAG
  SPAD-RAG 算法配置
  SPAD-RAG teacher service profile

resource/local_8gpu_0_7.yaml
  train_agent.impls.spad_rag.sub_stages.*
    每个 sub-stage 的 GPU/NPU、端口、服务启动和服务生命周期
```

也就是说：

1. `offline_two_stage.yaml` 继续管理 AIR 的完整离线流程。
2. `train_agent` 仍然是 AIR 顶层 stage。
3. SPAD-RAG 是 `train_agent` 的一种实现。
4. SPAD-RAG 的内部 sub-stage 由 `agent_training/spad_rag_base.yaml` 管理。
5. SPAD-RAG 的资源占用由 `resource.stage_resources.train_agent.impls.spad_rag.sub_stages.*` 管理。
6. Teacher profile 只描述服务形态和默认启动参数，不绑定 GPU/NPU 卡号。
7. `train_agent.outputs.agent_checkpoint` 仍然是下游 `generate_traces`、reranker dataset、infer matrix 消费的唯一 agent checkpoint 入口。

## 2. 配置分层

配置分四层。

第一层是外层 pipeline：

```text
AgenticIterRag/config/pipeline/offline_two_stage.yaml
```

它只表达 AIR 顶层 stage 顺序，以及 `train_agent` 选择哪个实现。

第二层是 agent training 实现配置：

```text
AgenticIterRag/config/agent_training/spad_rag_base.yaml
```

它表达 SPAD-RAG 内部 sub-stage、prompt、reward、dataset、teacher profile、SFT/DPO loss 等训练细节。

第三层是资源配置：

```text
AgenticIterRag/config/resource/local_8gpu_0_7.yaml
```

它表达每个 SPAD sub-stage 的 actor、teacher、retriever、trainer 具体占用哪些卡，监听哪些端口，是否由 AIR 自动启动和停止。

第四层是实验 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/offline_two_stage_overlay.yaml
```

或者新增一个更专门的实验 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/spad_offline_two_stage_overlay.yaml
```

即使新增实验 overlay，也仍然选择：

```yaml
main_run:
  config_groups:
    pipeline: offline_two_stage
```

不新增顶层：

```text
AgenticIterRag/config/pipeline/spad_rag_three_stage.yaml
```

## 3. offline_two_stage.yaml 设计

### 3.1 顶层 stages 不变

`offline_two_stage.yaml` 的顶层 stages 不变：

```yaml
stages:
  - train_agent
  - generate_traces
  - build_reranker_dataset
  - build_reranker_branch_dataset
  - filter_reranker_branch_dataset
  - train_llm_reranker
  - build_service_bundle
  - infer_matrix
```

SPAD-RAG 不应该出现在这里。它是 `train_agent` 的内部实现，不是和 `generate_traces`、`build_reranker_dataset` 平级的 AIR stage。

### 3.2 train_agent 轻量化配置

`train_agent` 只需要声明：

1. 是否启用。
2. 使用哪个 entry。
3. 使用哪个训练实现。
4. 训练实现配置从哪个配置组读取。
5. 顶层输入输出。

建议结构：

```yaml
stage_configs:
  # search-tool agent 训练阶段。
  train_agent:
    # 是否启用该 stage。
    enabled: true

    # stage 使用的资源配置键；实际资源细节在 resource YAML 中维护。
    resource_key: agent

    # agent 训练入口。入口内部根据 impl 分发到具体训练实现。
    entry: AgenticIterRag/main_train_agent.py

    # agent 训练实现。SPAD-RAG 第一版使用 spad_rag。
    impl: spad_rag

    # 训练实现配置引用；spad_rag 的完整配置来自 config.agent_training。
    impl_config_ref: agent_training

    # stage 输入字段。
    inputs:
      # agent 训练文件来自 data.train_files。
      train_files: data.train_files

      # agent 验证文件来自 data.val_files。
      val_files: data.val_files

      # 初始 actor 模型默认来自 model 配置组。
      init_actor_model: model.path

    # stage 输出字段。
    outputs:
      # 最终可部署 agent checkpoint。下游 generate_traces 只依赖这个字段。
      agent_checkpoint: null

      # agent 训练内部 manifest，记录 impl、sub-stage 产物和最终 checkpoint 选择。
      agent_training_manifest: null

      # stage manifest 由 pipeline runner 写入。
      manifest: null
```

这里不要展开 SPAD-RAG 的三个阶段。否则 `offline_two_stage.yaml` 会承担过多训练实现细节。

### 3.3 agent_checkpoint 的语义

`train_agent.outputs.agent_checkpoint` 应该总是指向当前 `train_agent` 产出的最终可部署 actor。

对于 SPAD-RAG：

```text
如果 Stage 3 DPO 执行成功:
  agent_checkpoint = Stage 3 DPO checkpoint

否则如果 Stage 3 SFT 执行成功:
  agent_checkpoint = Stage 3 SFT checkpoint

否则如果 Stage 1 Search-Policy RL 执行成功:
  agent_checkpoint = Stage 1 actor checkpoint
```

下游 stage 不关心 SPAD-RAG 内部哪个 sub-stage 产出了这个 checkpoint。下游只读：

```text
pipeline.stage_configs.train_agent.outputs.agent_checkpoint
```

## 4. agent_training/spad_rag_base.yaml 设计

新增配置组：

```text
AgenticIterRag/config/agent_training/spad_rag_base.yaml
```

配置名不绑定模型大小。不要叫 `spad_rag_qwen3_4b`。具体 actor 模型继续由现有 `model` 配置组控制。

### 4.1 顶层结构

`spad_rag_base.yaml` 是 `train_agent.impl = spad_rag` 的完整内部配置。

建议结构：

```yaml
# SPAD-RAG agent training strategy config.

name: spad_rag_base

# 本配置对应的 train_agent implementation。
impl: spad_rag

# SPAD-RAG 内部子阶段顺序；这些不是 AIR 顶层 pipeline stage。
sub_stage_order:
  - search_policy_rl
  - answer_refresh_data
  - answer_distillation

# SPAD-RAG 内部执行控制；语义对齐 pipeline 顶层控制字段。
resume_from_sub_stage: null
stop_after_sub_stage: null
skip_sub_stages: []
force_rerun_sub_stages: []

# SPAD-RAG 实现级输出；由 train_agent runtime 写入。
outputs:
  search_policy_actor_checkpoint: null
  search_policy_rollout_manifest: null
  search_policy_reward_manifest: null
  answer_refresh_manifest: null
  answer_distill_dataset_manifest: null
  answer_distilled_actor_checkpoint: null
  spad_manifest: null
```

### 4.2 基础引用配置

SPAD-RAG 不硬编码数据路径、actor 模型路径和基础 rollout 配置，而是引用现有配置组。

```yaml
refs:
  # 训练数据来自 data 配置组。
  train_files: data.train_files
  val_files: data.val_files
  train_max_samples: data.train_max_samples
  val_max_samples: data.val_max_samples

  # actor 模型来自 model 配置组。
  init_actor_model: model.path

  # 检索预算来自 infer_runtime。
  candidate_top_n: infer_runtime.retrieval.final_top_n
  visible_top_m: infer_runtime.retrieval.visible_top_m

  # rollout 基础事实来自 rollout 配置组。
  rollout_config: rollout
```

这里的原则是：

1. `data/*.yaml` 管真实数据路径。
2. `model/*.yaml` 管 actor 模型路径。
3. `infer_runtime/*.yaml` 管 retrieval 的 top-n/top-m 事实。
4. `agent_training/spad_rag_base.yaml` 管 SPAD-RAG 阶段如何使用这些事实。

### 4.3 Evidence Budget

Teacher reward 和 answer refresh 必须只使用 actor 实际可见的 evidence。

推荐配置：

```yaml
evidence:
  # recall 候选数量，默认引用 infer_runtime.retrieval.final_top_n。
  candidate_top_n: infer_runtime.retrieval.final_top_n

  # actor 和 teacher 实际可见文档数量，默认引用 infer_runtime.retrieval.visible_top_m。
  visible_top_m: infer_runtime.retrieval.visible_top_m

  # 每篇文档注入 actor/teacher prompt 的最大字符数；沿用 AIR/CAR 现有实践。
  max_doc_chars: 2000

  # 是否包含 sub query。
  include_sub_query: true

  # 是否包含 doc id，便于 debug 和 evidence hit 诊断。
  include_doc_id: true

  # 是否包含 title。
  include_doc_title: true

  # 是否包含正文。
  include_doc_text: true
```

关键约束：

```text
teacher_answerer 只能看每轮 visible_top_m docs，不能看 top50 全量候选。
```

否则 Stage 1 reward 会评价 actor 没有看到的信息，破坏 search policy credit assignment。

### 4.4 Actor Prompt

Actor prompt 可以沿用 CoAgenticRetriever 中 `SEARCH_R1_PROMPT` 的语义：

1. 每个 assistant turn 必须输出两个 tag block。
2. 先输出 `<reason>`。
3. 再输出 `<tool_call>` 或 `<answer>`。
4. 第一轮必须 search。
5. 没有 tool result 之前不能 answer。
6. 不允许输出 `<tool_response>`。
7. `<answer>` 内只输出短答案。

实现上不要跨框架 import。需要把 prompt 文本复制到 AIR 内部：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/prompts.py
```

配置：

```yaml
actor:
  prompt_template_version: coagentic_retriever_search_r1_v1
  output_protocol: reason_tool_or_answer
  reason_tag: "<reason>"
  tool_call_tag: "<tool_call>"
  answer_tag: "<answer>"
```

Stage 1 是否在 `<answer>` 停止不由 prompt 控制，而由 rollout stop sequence 控制。prompt 不需要写空 `<answer>`。

### 4.5 Teacher Answerer 配置

Teacher 一定是本地 OpenAI-compatible vLLM 服务，不设计线上 API 服务，也不在训练进程内直接加载模型。

配置分两部分：

1. `agent_training.teacher_answerer`：描述 teacher client、prompt、服务 profile 模板。
2. `resource.stage_resources.train_agent.impls.spad_rag.sub_stages.*.services.teacher_answerer`：描述当前 sub-stage 具体用哪些卡、端口、实例、是否 auto-start。

Teacher profile 不绑定 GPU/NPU 卡号。卡号属于 resource 配置。

```yaml
teacher_answerer:
  # 当前默认 teacher profile 名。实际 sub-stage 可以在 resource 中覆盖 profile。
  default_service_profile: glm47_single_vllm

  # Teacher 是 answer-only prompt，不使用 Actor 的 tool-use prompt。
  prompt_template_version: spad_teacher_evidence_answer_v1

  # 证据不足时的固定 answer。
  evidence_insufficient_answer: 证据不足无法作答

  # teacher client 调用参数。
  client:
    api_type: openai_chat
    endpoint: null
    model: null
    temperature: 0.0
    top_p: 1.0
    max_tokens: 1024
    request_timeout: 600
    max_retries: 3
    retry_delay: 2.0
    retry_backoff: 2.0
    http_force_close: true

  # chat template / thinking 控制。不要在 prompt 里写“关闭原生 thinking”。
  apply_chat_template_kwargs:
    enable_thinking: false
    thinking: false
    reasoning_effort: none

  # 服务 profile 只描述服务形态、模型路径和默认 vLLM 参数，不描述卡号和端口占用。
  service_profiles:
    glm47_single_vllm:
      backend_type: vllm_single
      model_path: /data01/ms_wksp/agent_up_to_date/models/llm/GLM-4.7-Flash
      served_model_name: GLM-4.7-Flash
      trust_remote_code: true
      default_vllm_args:
        tensor_parallel_size: 2
        max_model_len: 32000
        gpu_memory_utilization: 0.92
        enforce_eager: false

    qwen32b_3x_proxy:
      backend_type: multi_instance_proxy
      model_path: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-32B
      served_model_name: Qwen3-32B
      trust_remote_code: true
      default_instance_args:
        tensor_parallel_size: 1
        max_model_len: 32000
        gpu_memory_utilization: 0.90
        enforce_eager: false
```

Teacher client 解析规则：

```text
1. 当前 sub-stage resource 里有 services.teacher_answerer.endpoint:
     使用该 endpoint。
2. endpoint 为空但有 port:
     拼 http://127.0.0.1:{port}/v1/chat/completions。
3. model 为空:
     使用当前 profile 的 served_model_name，或 resource 里的 served_model_name 覆盖值。
```

### 4.6 Teacher Prompt

Teacher prompt 和 Actor prompt 不同。Teacher 是 answer-only evidence-grounded QA prompt，不包含 tool-call 规则。

Teacher prompt 版本名：

```text
spad_teacher_evidence_answer_v1
```

推荐系统说明：

```text
You are an evidence-grounded QA teacher for an agentic RAG system.

You will be given:
- The original question.
- The search queries issued by the actor.
- The top evidence passages that were visible to the actor at each search step.

Your task:
Answer the original question using ONLY the provided evidence.

Rules:
- Do not use your own parametric knowledge.
- Do not infer facts that are not supported by the evidence.
- If the evidence is sufficient, output the shortest correct answer span.
- If the evidence is insufficient, output exactly: 证据不足无法作答
- Do not output <think> tags.
- Do not output any text outside the required XML-style blocks.

Output format:
<reason>...</reason>
<answer>...</answer>

The <reason> block must:
- Briefly identify which evidence supports the answer; or
- If evidence is insufficient, explain what evidence would be needed and why the current evidence is insufficient.

The <answer> block must:
- Contain only the final short answer string; or
- Contain exactly 证据不足无法作答 if the evidence is insufficient.
```

不建议在 prompt 里明文写：

```text
关闭原生 thinking
```

thinking 应该通过 chat template / inference 参数控制。

Teacher 输入渲染格式：

```text
Original question:
{question}

Evidence visible to the actor:
[Search step 1]
Sub-query: {sub_query_1}

[Doc 1]
Doc id: {doc_id}
Title: {title}
Text: {text_truncated_to_max_doc_chars}

[Doc 2]
...

[Search step 2]
Sub-query: {sub_query_2}
...
```

Teacher 输入只能包含：

```text
original question
每轮 sub_query
每轮 actor visible_top_m docs
```

Teacher 不能看 actor answer，避免 chosen 被 rejected 污染。

证据不足时，Teacher 必须：

1. 在 `<reason>` 中说明需要什么证据。
2. 在 `<reason>` 中说明当前 evidence 为什么不足。
3. 在 `<answer>` 中输出固定字符串。

示例：

```text
<reason>Need evidence that directly states the missing entity, date, location, relation, or event required by the question. The current evidence only provides related background or indirect clues, so it cannot uniquely support a final answer.</reason>
<answer>证据不足无法作答</answer>
```

## 5. SPAD Sub-stage 配置

### 5.1 Stage 1: search_policy_rl

Stage 1 使用 CAR search actor 训练参数作为默认口径。AIR LLM-reranker stage2 的参数只参考服务编排，不作为 search actor RL 默认训练参数。

```yaml
sub_stages:
  search_policy_rl:
    enabled: true
    resource_key: null
    version: null
    overwrite: false

    inputs:
      train_files: data.train_files
      val_files: data.val_files
      init_actor_model: model.path

    trainer:
      backend: verl
      method: grpo

      # CAR search actor 默认 batch 口径。
      train_batch_size: 64
      ppo_mini_batch_size: 64
      n_samples_per_prompt: 8

      train_max_samples: data.train_max_samples
      val_max_samples: data.val_max_samples
      val_batch_size: data.val_batch_size

      total_epochs: 1
      total_training_steps: null
      val_before_train: false
      val_only: false
      test_freq: -1
      save_freq: 10

      actor_micro_batch_size_per_gpu: 2
      log_prob_micro_batch_size_per_gpu: 4

      use_kl_loss: true
      kl_loss_coef: 0.001
      kl_loss_type: low_var_kl
      entropy_coeff: 0.0

      actor_activation_offload: false
      actor_optimizer_offload: false
      actor_param_offload: false
      ref_param_offload: true

      use_dynamic_bsz: true
      use_remove_padding: true

    rollout:
      mode: async
      temperature: 1.0
      top_k: -1
      top_p: 1.0

      max_prompt_length: data.max_prompt_length
      max_response_length: data.max_response_length
      max_model_len: 16096
      max_num_batched_tokens: 16096
      max_num_seqs: 16

      gpu_memory_utilization: 0.55
      enable_chunked_prefill: true
      enable_prefix_caching: true
      enforce_eager: false
      free_cache_engine: true

      calculate_log_probs: false

      multi_turn:
        enable: true
        max_assistant_turns: 6
        max_user_turns: 6
        max_parallel_calls: 2
        format: search_r1

      agent:
        num_workers: 4
        inject_tool_schema: true

      sampling_stop:
        - "<answer>"
      include_stop_str_in_output: true

    teacher_reward:
      enabled: true
      metric: token_f1
      call_policy: valid_finish_only
      skip_on_invalid_format: true
      skip_on_invalid_action: true
      skip_on_no_finish: true

    reward:
      teacher_f1_weight: 1.0
      search_cost: 0.02
      invalid_format_penalty: 1.0
      invalid_action_penalty: 1.0
      duplicate_query_penalty: 0.1
      no_finish_penalty: 1.0
      evidence_insufficient_penalty: 0.0

    logging:
      save_teacher_called: true
      save_teacher_skip_reason: true
      save_format_status: true
      save_action_status: true
      save_stop_status: true
      save_reward_breakdown: true

    outputs:
      actor_checkpoint: null
      rollout_manifest: null
      reward_manifest: null
      example_json: null
      manifest: null
```

### 5.2 Stage 2: answer_refresh_data

Stage 2 不训练，只做 refresh rollout 和 chosen/rejected 数据构造。

```yaml
sub_stages:
  answer_refresh_data:
    enabled: true
    resource_key: null
    version: null
    overwrite: false

    inputs:
      # null 表示默认使用 search_policy_rl.outputs.actor_checkpoint。
      actor_checkpoint: null
      data_files: data.train_files
      max_samples: data.train_max_samples

    rollout:
      mode: async
      temperature: 1.0
      top_k: -1
      top_p: 1.0

      max_prompt_length: data.max_prompt_length
      max_response_length: data.max_response_length
      max_model_len: 16096
      max_num_batched_tokens: 16096
      max_num_seqs: 16

      multi_turn:
        enable: true
        max_assistant_turns: 6
        max_user_turns: 6
        max_parallel_calls: 2
        format: search_r1

      # Stage 2 要让 actor 自然完整回答，不能在 <answer> stop。
      sampling_stop: []
      include_stop_str_in_output: false

    dataset:
      save_messages_before_final_answer: true
      save_actor_answer_as_rejected: true
      save_teacher_answer_as_chosen: true
      teacher_sees_actor_answer: false

    filter:
      require_teacher_format_valid: true
      require_evidence_sufficient: true
      min_teacher_f1: 0.0
      keep_evidence_insufficient_ratio: 0.0

    outputs:
      refresh_jsonl: null
      refresh_parquet: null
      refresh_manifest: null
      dpo_dataset_jsonl: null
      dpo_dataset_parquet: null
      dpo_dataset_manifest: null
      example_json: null
      manifest: null
```

### 5.3 Stage 3: answer_distillation

Stage 3 不启动 teacher/retriever 服务，只读取 Stage 2 数据集训练 actor answer ability。

```yaml
sub_stages:
  answer_distillation:
    enabled: true
    resource_key: null

    phase_order:
      - sft
      - dpo

    resume_from_phase: null
    stop_after_phase: null
    skip_phases: []
    force_rerun_phases: []

    inputs:
      # null 表示默认使用 search_policy_rl.outputs.actor_checkpoint。
      init_actor_checkpoint: null

      # null 表示默认使用 answer_refresh_data.outputs.dpo_dataset_manifest。
      dataset_manifest: null

    outputs:
      final_actor_checkpoint: null
      sft_checkpoint: null
      dpo_checkpoint: null
      answer_distillation_manifest: null
      manifest: null

    phases:
      sft:
        enabled: false
        resource_key: null
        version: null
        overwrite: false
        train_batch_size: 64
        micro_batch_size_per_gpu: 2
        learning_rate: 1.0e-6
        total_epochs: 1
        max_prompt_length: data.max_prompt_length
        max_response_length: data.max_response_length
        use_remove_padding: true
        loss_weight: 1.0
        outputs:
          checkpoint: null
          train_log: null
          manifest: null

      dpo:
        enabled: true
        resource_key: null
        version: null
        overwrite: false
        train_batch_size: 64
        micro_batch_size_per_gpu: 2
        learning_rate: 1.0e-6
        total_epochs: 1
        beta: 0.1
        pairwise_loss_weight: 1.0
        chosen_sft_loss_weight: 0.2
        max_prompt_length: data.max_prompt_length
        max_response_length: data.max_response_length
        use_remove_padding: true
        inputs:
          init_actor_checkpoint: null
          dataset_manifest: null
        outputs:
          checkpoint: null
          train_log: null
          manifest: null
```

## 6. Resource 配置设计

资源配置必须按 SPAD sub-stage 控制，而不是固定在 teacher profile 里。

也就是说：

```text
teacher_answerer.service_profiles
  只描述服务形态和模型启动模板。

resource.stage_resources.train_agent.impls.spad_rag.sub_stages.*
  决定当前 sub-stage 里 actor、teacher、recall、trainer 分别占哪些卡。
```

### 6.1 GLM4.7 Teacher 资源方案

GLM4.7 占两张卡，actor 占四张卡，recall 使用剩余两张卡。

```yaml
resource:
  stage_resources:
    train_agent:
      impls:
        spad_rag:
          sub_stages:
            search_policy_rl:
              trainer:
                gpu_ids: [0, 1, 2, 3]
                n_gpus_per_node: 4

              services:
                teacher_answerer:
                  profile: glm47_single_vllm
                  gpu_ids: [4, 5]
                  tensor_parallel_size: 2
                  port: 8067
                  endpoint: http://127.0.0.1:8067/v1/chat/completions
                  served_model_name: GLM-4.7-Flash
                  auto_start: true
                  auto_stop: true
                  preflight: true
                  wait_seconds: 600

                recall:
                  backend_type: npu
                  instance_count: 2
                  port: 8130
                  backend_base_port: 8131
                  retrieval_service_url: http://127.0.0.1:8130/retrieve
                  auto_start: true
                  auto_stop: true
                  wait_seconds: 360
                  proxy:
                    strategy: least_inflight
                    timeout: 180
                    failure_cooldown_seconds: 10
                    latency_ewma_alpha: 0.2
                    max_retries_per_request: 2
                  accelerator_backend:
                    gpu_ids: [6, 7]
                    query_batch_size: 32
                    doc_dtype: float16

            answer_refresh_data:
              services:
                actor_vllm:
                  gpu_ids: [0, 1, 2, 3]
                  tensor_parallel_size: 4
                  port: 8340
                  served_model_name: spad-refresh-actor
                  auto_start: true
                  auto_stop: true

                teacher_answerer:
                  profile: glm47_single_vllm
                  gpu_ids: [4, 5]
                  tensor_parallel_size: 2
                  port: 8067
                  endpoint: http://127.0.0.1:8067/v1/chat/completions
                  served_model_name: GLM-4.7-Flash
                  auto_start: true
                  auto_stop: true
                  preflight: true
                  wait_seconds: 600

                recall:
                  backend_type: npu
                  instance_count: 2
                  port: 8130
                  backend_base_port: 8131
                  retrieval_service_url: http://127.0.0.1:8130/retrieve
                  auto_start: true
                  auto_stop: true
                  accelerator_backend:
                    gpu_ids: [6, 7]
                    query_batch_size: 32
                    doc_dtype: float16

            answer_distillation:
              phases:
                sft:
                  trainer:
                    gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
                    n_gpus_per_node: 8

                dpo:
                  trainer:
                    gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
                    n_gpus_per_node: 8
```

### 6.2 Qwen32B 3 实例 Teacher 资源方案

Qwen32B teacher 使用 3 个单卡 vLLM 实例，通过一个 proxy endpoint 对外服务。actor 占四张卡，recall 使用最后一张卡。

```yaml
resource:
  stage_resources:
    train_agent:
      impls:
        spad_rag:
          sub_stages:
            search_policy_rl:
              trainer:
                gpu_ids: [0, 1, 2, 3]
                n_gpus_per_node: 4

              services:
                teacher_answerer:
                  profile: qwen32b_3x_proxy
                  backend_type: multi_instance_proxy
                  endpoint: http://127.0.0.1:8067/v1/chat/completions
                  served_model_name: Qwen3-32B
                  auto_start: true
                  auto_stop: true
                  preflight: true
                  wait_seconds: 900
                  proxy:
                    host: 127.0.0.1
                    port: 8067
                    strategy: least_inflight
                    timeout: 600
                    failure_cooldown_seconds: 10
                    latency_ewma_alpha: 0.2
                    max_retries_per_request: 3
                  instances:
                    - name: qwen32b_teacher_0
                      gpu_ids: [4]
                      tensor_parallel_size: 1
                      port: 8068
                    - name: qwen32b_teacher_1
                      gpu_ids: [5]
                      tensor_parallel_size: 1
                      port: 8069
                    - name: qwen32b_teacher_2
                      gpu_ids: [6]
                      tensor_parallel_size: 1
                      port: 8070

                recall:
                  backend_type: npu
                  instance_count: 1
                  port: 8130
                  backend_base_port: 8131
                  retrieval_service_url: http://127.0.0.1:8130/retrieve
                  auto_start: true
                  auto_stop: true
                  accelerator_backend:
                    gpu_ids: [7]
                    query_batch_size: 32
                    doc_dtype: float16

            answer_refresh_data:
              services:
                actor_vllm:
                  gpu_ids: [0, 1, 2, 3]
                  tensor_parallel_size: 4
                  port: 8340
                  served_model_name: spad-refresh-actor
                  auto_start: true
                  auto_stop: true

                teacher_answerer:
                  profile: qwen32b_3x_proxy
                  backend_type: multi_instance_proxy
                  endpoint: http://127.0.0.1:8067/v1/chat/completions
                  served_model_name: Qwen3-32B
                  auto_start: true
                  auto_stop: true
                  proxy:
                    host: 127.0.0.1
                    port: 8067
                    strategy: least_inflight
                  instances:
                    - name: qwen32b_teacher_0
                      gpu_ids: [4]
                      tensor_parallel_size: 1
                      port: 8068
                    - name: qwen32b_teacher_1
                      gpu_ids: [5]
                      tensor_parallel_size: 1
                      port: 8069
                    - name: qwen32b_teacher_2
                      gpu_ids: [6]
                      tensor_parallel_size: 1
                      port: 8070

                recall:
                  backend_type: npu
                  instance_count: 1
                  port: 8130
                  backend_base_port: 8131
                  retrieval_service_url: http://127.0.0.1:8130/retrieve
                  auto_start: true
                  auto_stop: true
                  accelerator_backend:
                    gpu_ids: [7]
                    query_batch_size: 32
                    doc_dtype: float16

            answer_distillation:
              phases:
                sft:
                  trainer:
                    gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
                    n_gpus_per_node: 8

                dpo:
                  trainer:
                    gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
                    n_gpus_per_node: 8
```

### 6.3 Resource 校验

SPAD-RAG 的 sub-stage 是顺序执行的，所以 resource validator 不能把 `search_policy_rl`、`answer_refresh_data`、`answer_distillation` 三个 sub-stage 的 GPU 占用放在同一时间片里做冲突检查。

正确校验规则：

```text
1. train_agent 顶层只校验 impls.spad_rag 存在。
2. 对每个 selected SPAD sub-stage 单独校验 GPU/port 冲突。
3. 同一个 sub-stage 内默认不允许 GPU 重叠，除非显式 allow_gpu_overlap。
4. 不同 sub-stage 之间允许复用 GPU。
5. answer_distillation.sft 和 answer_distillation.dpo 是 phase，通常顺序执行，也允许复用全 8 卡。
```

## 7. Overlay 设计

SPAD-RAG 实验可以继续使用现有：

```text
tasks/train_tasks/agenticIterRag/configs/offline_two_stage_overlay.yaml
```

也可以新增一个更专门的实验 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/spad_offline_two_stage_overlay.yaml
```

无论哪种方式，overlay 都不切换顶层 pipeline，只覆盖：

```yaml
main_run:
  config_groups:
    pipeline: offline_two_stage
    agent_training: spad_rag_base
```

示例：

```yaml
main_run:
  project:
    experiment_name: agentic_iter_rag_v1_spad_rag_glm47
  config_groups:
    pipeline: offline_two_stage
    agent_training: spad_rag_base

pipeline:
  stage_configs:
    train_agent:
      enabled: true
      impl: spad_rag
      impl_config_ref: agent_training

agent_training:
  teacher_answerer:
    default_service_profile: glm47_single_vllm

  resume_from_sub_stage: null
  stop_after_sub_stage: null
  skip_sub_stages: []
  force_rerun_sub_stages: []

  sub_stages:
    search_policy_rl:
      enabled: true

    answer_refresh_data:
      enabled: true

    answer_distillation:
      enabled: true
      phases:
        sft:
          enabled: false
        dpo:
          enabled: true
```

实验 overlay 可以覆盖 SPAD-RAG 的内部开关和少量实验参数，但默认策略留在 `spad_rag_base.yaml`，避免 overlay 过长。

## 8. main_run 和 compile_config 调整

为了支持新的 `agent_training` 配置组，需要调整：

```text
AgenticIterRag/config/main_run/agentic_iter_rag_main.yaml
scripts/agenticIterRag_v1/assets/compile_config.py
```

`main_run.config_groups` 增加：

```yaml
agent_training: spad_rag_base
```

`compile_config.py` 增加：

```python
GROUP_DIRS = {
    ...
    "agent_training": "agent_training",
}

GROUP_ARG_DESTS = {
    ...
    "agent_training_config": "agent_training",
}
```

CLI 增加：

```text
--AGENT_TRAINING_CONFIG
--agent-training-config
```

配置校验需要新增条件逻辑：

```text
如果 pipeline.stage_configs.train_agent.impl == "spad_rag":
  必须存在 agent_training.impl == "spad_rag"
  必须存在 agent_training.sub_stage_order
  必须存在 agent_training.sub_stages.search_policy_rl
  必须存在 agent_training.sub_stages.answer_refresh_data
  必须存在 agent_training.sub_stages.answer_distillation
  必须存在 resource.stage_resources.train_agent.impls.spad_rag
```

现有 offline reranker 任务即使不读取 `agent_training`，也可以接受这个配置组被编译进最终 YAML。这样不会破坏既有 pipeline。

## 9. 代码设计

### 9.1 总体结构

`main_train_agent.py` 不直接写 SPAD-RAG 的全部逻辑。它只做四件事：

1. 读取最终编译后的 AIR config。
2. 读取 `pipeline.stage_configs.train_agent.impl`。
3. 根据 `impl` 找到对应 trainer。
4. 调用 trainer，并把最终 checkpoint 写回 `train_agent.outputs.agent_checkpoint`。

建议代码结构：

```text
AgenticIterRag/main_train_agent.py

AgenticIterRag/agentic_iter_rag/agent_training/
  __init__.py
  registry.py
  train_agent_entry.py
  spad/
    __init__.py
    orchestrator.py
    config.py
    resource.py
    service_manager.py
    prompts.py
    parsers.py
    reward.py
    teacher_answerer.py
    search_policy_rl.py
    refresh_rollout.py
    dataset_builder.py
    answer_distillation.py
    manifest.py
```

`registry.py` 负责：

```text
impl name -> trainer class/function
```

第一版只有：

```text
spad_rag -> SpadRagTrainer
```

### 9.2 main_train_agent.py

`main_train_agent.py` 建议保持很薄：

```text
parse args
load final config
stage_cfg = config["pipeline"]["stage_configs"]["train_agent"]
impl = stage_cfg["impl"]
impl_cfg = resolve_impl_config(config, stage_cfg["impl_config_ref"])
trainer = registry.get(impl)
result = trainer.run(config, stage_cfg, impl_cfg)
update stage outputs
write manifest
```

这样 `main_train_agent.py` 不关心 SPAD-RAG 有几个 sub-stage，也不关心 teacher answerer 怎么调用。

### 9.3 SPAD Orchestrator

`spad/orchestrator.py` 是 SPAD-RAG 的内部 runner。

职责：

1. 读取 `agent_training.sub_stage_order`。
2. 应用 `resume_from_sub_stage`、`stop_after_sub_stage`、`skip_sub_stages`、`force_rerun_sub_stages`。
3. 顺序执行 enabled 的 sub-stage。
4. 管理 sub-stage 的输入输出依赖。
5. 为当前 sub-stage 解析 resource plan。
6. 启动和停止当前 sub-stage 需要的服务。
7. 汇总 `spad_manifest`。
8. 选择最终 `agent_checkpoint`。

内部逻辑类似现有 pipeline runner，但作用域只限于 `train_agent.impl = spad_rag`。

伪流程：

```text
run_spad_rag(config, train_agent_stage_cfg, spad_cfg):
  selected = select_sub_stages(spad_cfg)

  for sub_stage in selected:
    resource_plan = resolve_spad_sub_stage_resource(config, sub_stage)
    validate_spad_sub_stage_resource(resource_plan)
    with managed_services(resource_plan):
      run_sub_stage(sub_stage, config, spad_cfg, resource_plan)

  final_checkpoint = choose_final_checkpoint(...)
  write spad_manifest
  return TrainAgentResult(agent_checkpoint=final_checkpoint, manifest=...)
```

### 9.4 Resource Resolver

`spad/resource.py` 负责从当前 sub-stage 取资源，而不是从 teacher profile 取资源。

输入：

```text
config
sub_stage_name
phase_name optional
```

输出：

```text
当前 sub-stage 的 trainer/services resource plan
```

解析路径：

```text
resource.stage_resources.train_agent.impls.spad_rag.sub_stages.{sub_stage}
```

如果是 Stage 3 phase：

```text
resource.stage_resources.train_agent.impls.spad_rag.sub_stages.answer_distillation.phases.{phase}
```

### 9.5 Service Manager

`spad/service_manager.py` 负责启动、检查、停止服务。

需要支持：

1. `vllm_single` teacher。
2. `multi_instance_proxy` teacher。
3. `npu` recall。
4. `actor_vllm`。

Teacher 启动时需要合并两部分配置：

```text
agent_training.teacher_answerer.service_profiles.{profile}
resource 当前 sub-stage services.teacher_answerer
```

例如：

```text
profile 提供 model_path、served_model_name、默认 vLLM 参数
resource 提供 gpu_ids、tensor_parallel_size、port、endpoint、instances、auto_start
```

Teacher 调用时，`teacher_answerer.py` 从 service manager 获取 active endpoint：

```text
endpoint = active_service.endpoint
model = active_service.served_model_name
```

不要让 teacher client 自己猜 GPU 或启动服务。

### 9.6 SPAD Config Resolver

`spad/config.py` 负责把配置引用解析成真实值。

需要支持：

```text
data.train_files
data.val_files
model.path
infer_runtime.retrieval.visible_top_m
pipeline.stage_configs.train_agent.outputs.agent_checkpoint
```

这里不要用 ad hoc string replace。可以复用 AIR 现有 config 访问工具；如果没有合适工具，新增一个局部 `deep_get(config, dotted_path)`。

### 9.7 Stage 1: Search-Policy RL

主要模块：

```text
spad/search_policy_rl.py
spad/reward.py
spad/teacher_answerer.py
spad/parsers.py
```

职责：

1. 启动 actor training。
2. rollout 时在 `<answer>` stop。
3. 解析 actor action。
4. 识别 search / finish / invalid。
5. 先做 trajectory 格式和 action 合法性校验。
6. 只有 trajectory 格式有效且 actor 合法停止时，才调用 teacher answerer。
7. teacher 只看 actor 可见 evidence，不看 actor answer。
8. 计算 teacher_f1 和 reward breakdown。
9. 写出 Stage 1 actor checkpoint、rollout manifest 和 reward manifest。

Teacher reward 必须短路无效轨迹：

```text
valid_finish:
  调 teacher，计算 teacher_f1

invalid_format:
  不调 teacher，teacher_called=false，给 invalid_format_penalty

invalid_action:
  不调 teacher，teacher_called=false，给 invalid_action_penalty

no_finish:
  不调 teacher，teacher_called=false，给 no_finish_penalty

retrieval_failure:
  默认标记 service_error，不把它当成 actor fault

valid_search_but_insufficient_evidence:
  调 teacher，teacher 输出证据不足，teacher_f1=0
```

Reward 计算顺序：

```text
1. 校验每个 assistant turn 的 tag 格式。
2. 校验 tool_call JSON。
3. 校验 action 约束：
   - 第一轮必须 search。
   - tool result 前不能 answer。
   - 不能同时 tool_call 和 answer。
   - 不能输出 <tool_response>。
4. 校验是否合法停止到 <answer>。
5. 只有 1-4 全部通过，才调用 teacher answerer。
```

需要保存的诊断字段：

```text
teacher_called
teacher_skip_reason
format_status
action_status
stop_status
teacher_f1
final_reward
reward_breakdown
```

Stage 1 的产物路径写回：

```text
agent_training.outputs.search_policy_actor_checkpoint
agent_training.sub_stages.search_policy_rl.outputs.actor_checkpoint
```

### 9.8 Stage 2: Answer Refresh Data

主要模块：

```text
spad/refresh_rollout.py
spad/dataset_builder.py
spad/teacher_answerer.py
spad/parsers.py
```

职责：

1. 冻结 Stage 1 actor。
2. 对训练集重新 full rollout。
3. 不在 `<answer>` stop。
4. 保存 `messages_before_final_answer`。
5. 保存 actor final response 作为 rejected。
6. 调用 teacher 基于同一条轨迹 evidence 生成 chosen。
7. 输出 DPO/SFT 数据集 manifest。

核心样本 schema：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": "<reason>...</reason>\n<answer>...</answer>",
  "rejected": "<reason>...</reason>\n<answer>...</answer>",
  "metadata": {
    "question": "...",
    "gold_answers": ["..."],
    "sub_queries": ["..."],
    "teacher_f1": 1.0,
    "actor_f1": 0.0,
    "evidence_sufficient": true,
    "search_count": 2,
    "format_status": "valid"
  }
}
```

### 9.9 Stage 3: Answer Distillation

主要模块：

```text
spad/answer_distillation.py
```

职责：

1. 读取 Stage 2 dataset。
2. 按 `phase_order` 执行 SFT/DPO。
3. SFT 默认关闭。
4. DPO 默认开启。
5. DPO 同时包含 pairwise loss 和 chosen SFT auxiliary loss。

Loss：

```text
L_total = pairwise_loss_weight * L_pairwise_dpo
        + chosen_sft_loss_weight * L_sft_chosen
```

`answer_distillation.py` 内部可以有：

```text
run_sft_phase(...)
run_dpo_phase(...)
choose_answer_distilled_checkpoint(...)
```

第一版可以先把 SFT/DPO 入口和数据格式打通，再继续细化 VERL trainer 参数。

### 9.10 Manifest 设计

`spad/manifest.py` 负责统一写 manifest。

建议每层都有 manifest：

```text
train_agent stage manifest
spad_manifest
sub-stage manifest
phase manifest
```

`spad_manifest` 至少包含：

```json
{
  "type": "spad_rag_train_agent_manifest",
  "impl": "spad_rag",
  "selected_sub_stages": [],
  "sub_stage_outputs": {},
  "service_profiles": {},
  "resource_summary": {},
  "final_agent_checkpoint": "...",
  "created_at": "..."
}
```

这样下游只读 `agent_checkpoint`，调试时可以沿 manifest 追踪 SPAD-RAG 内部产物和服务资源。

## 10. run_pipeline 接入

当前 `run_pipeline.py` 对 `train_agent` 只是写 placeholder manifest。接入 SPAD-RAG 后，需要让它真实调用：

```text
AgenticIterRag/main_train_agent.py
```

建议保持和其它 stage 一样：

1. `run_pipeline.py` 负责 stage 级调度、资源计划和 manifest。
2. `main_train_agent.py` 负责 train_agent 内部实现分发。
3. `spad/orchestrator.py` 负责 SPAD-RAG 内部 sub-stage。

不要让 `run_pipeline.py` 直接理解 `search_policy_rl`、`answer_refresh_data`、`answer_distillation`。否则 SPAD-RAG 内部复杂度又会泄漏到 AIR 顶层 pipeline runner。

## 11. 注释和风格要求

所有新增代码和配置都要按 AIR 当前风格做好注释：

1. YAML 中每个非显然字段都写清楚用途。
2. Python 顶层入口写 docstring，说明输入 config path、产物和 manifest。
3. 核心 schema 写字段注释。
4. prompt version 写清楚来源和适配边界。
5. 不把实验参数硬编码在 shell 里。

## 12. 第一版实施顺序

建议实施顺序：

1. 新增 `agent_training/spad_rag_base.yaml`。
2. 扩展 `main_run` 和 `compile_config.py` 支持 `agent_training` 配置组。
3. 轻量扩展 `offline_two_stage.yaml` 的 `train_agent`：增加 `impl`、`impl_config_ref`、`val_files`、`init_actor_model`、`agent_training_manifest`。
4. 在 `resource/local_8gpu_0_7.yaml` 增加 `train_agent.impls.spad_rag.sub_stages` 资源 skeleton。
5. 实现 `main_train_agent.py` 的 impl registry 分发。
6. 实现 SPAD config resolver、resource resolver、service manager、manifest、prompt/parser。
7. 先实现 Stage 2 answer refresh dataset，因为它最容易离线验证 context 对齐。
8. 再实现 Stage 1 teacher reward。
9. 最后接 Stage 3 SFT/DPO trainer。

这个顺序能优先验证最关键的事情：

```text
messages_before_final_answer
-> chosen teacher response
-> rejected actor response
```

只要这条链路稳定，Stage 1 reward 和 Stage 3 trainer 都更容易独立调试。

## 13. 当前结论

1. 不新增顶层 `spad_rag_three_stage.yaml` pipeline。
2. `offline_two_stage.yaml` 继续管理完整 AIR 离线流程。
3. `offline_two_stage.yaml` 中的 `train_agent` 只保留轻量 impl 引用。
4. SPAD-RAG 的内部 sub-stage、开关和算法配置放在 `agent_training/spad_rag_base.yaml`。
5. SPAD-RAG 的 GPU/NPU/端口/服务生命周期放在 `resource.stage_resources.train_agent.impls.spad_rag.sub_stages.*`。
6. Teacher profile 不绑定卡号，只描述服务形态和模型启动模板。
7. Stage 1 训练参数以 CAR search actor RL 配置经验为默认。
8. Recall 默认使用 NPU，不默认 CPU。
9. GLM4.7 teacher profile 通常在 sub-stage resource 中占 2 张卡。
10. Qwen32B teacher profile 通常在 sub-stage resource 中占 3 个单卡实例，并通过 proxy 暴露一个 endpoint。
11. Actor prompt 可沿用 CoAgenticRetriever Search-R1 语义，但复制到 AIR 内部。
12. Teacher prompt 是 answer-only evidence-grounded prompt，不写“关闭原生 thinking”，thinking 由推理参数控制。
13. Stage 3.1 SFT 默认关闭，Stage 3.2 DPO 默认开启，并带 chosen SFT auxiliary loss。

# SPAD-RAG Qwen3-1.7B + GLM4.7 执行计划 Draft

日期：2026-07-09

状态：Draft

本文档是 SPAD-RAG 第一版工程实现、消融和正式训练的执行计划。它不是新的算法方案，也不是替代详细设计的文档。执行时必须以以下两份文档为基准：

```text
docs/AgenticIterRag_v1/planing/260708e_spad_rag_search_policy_rl_answer_distillation_draft.md
docs/AgenticIterRag_v1/planing/260708f_spad_rag_train_agent_detailed_design_draft.md
```

其中：

1. `260708e...draft.md` 是算法思路基准，定义 SPAD-RAG 的三阶段训练逻辑、Stage 1 reward 中 teacher answerer 的角色、Stage 2 on-policy answer 数据准备，以及 Stage 3 answer distillation 的目标。
2. `260708f...detailed_design_draft.md` 是工程设计基准，定义 SPAD-RAG 如何作为 AIR `train_agent` stage 的实现接入、配置层级如何组织、资源如何按 sub-stage 管理、prompt/reward/service 如何拆分。

如果执行中发现本文档和上述两份基准文档冲突，默认以上述两份基准文档为准；如果确实需要改变设计，应先更新详细设计文档，再改执行计划和代码。如果第一份260708e文档和第二份260708f文档在细节上有冲突，则以第二份260708f文档为准。

## 1. 第一版目标

第一版实验组合：

```text
actor / agent model: Qwen3-1.7B
teacher answerer: GLM-4.7-Flash local vLLM service
```

目标不是一次性做完所有泛化实验，而是按下面顺序推进：

1. 先把 SPAD-RAG 三个 sub-stage 在 AIR `train_agent` 中实际跑通。
2. 再用少量 step 消融每个 sub-stage 的训练或数据生产效率。
3. 再用 5 step 单独消融 Stage 1 `search_policy_rl` 的训练效果。
4. 最后开启正式训练。

当前已确认 Qwen3-1.7B 本地路径为：

```text
/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B
```

执行时必须使用该路径新增 AIR model config。不能静默降级到 Qwen3-4B 或 Qwen3-0.6B，除非后续明确修改本执行计划。

## 2. 实现基准和边界

实现必须遵守 `260708e` 中的核心思路：

1. Stage 1 只训练 search policy：搜什么、搜几轮、什么时候停止。
2. Stage 1 的 teacher answerer 是 reward function 的一部分，只基于搜索证据回答并计算 F1。
3. Stage 1 中 actor 到 `<answer>` 后停止，answer body 不参与 search trajectory 质量判断。
4. actor 中间格式错误、非法 action、未合法停止时，不调用 teacher answerer，直接走格式或停止相关惩罚。
5. Stage 2 使用 Stage 1 训练后的 actor 重新 rollout，生成 on-policy answer context。
6. Stage 2 不再在 `<answer>` 处 stop；actor answer 作为 DPO rejected，teacher answer 作为 DPO chosen。
7. Stage 3 把 teacher answer 能力蒸馏回 actor，SFT 默认关闭，DPO 默认开启，并在 DPO 中同时保留 chosen-answer SFT loss 接口。

实现必须遵守 `260708f` 中的工程设计：

1. 不新增顶层 `spad_rag_three_stage` pipeline。
2. SPAD-RAG 是 `offline_two_stage.yaml -> train_agent` 的一种实现。
3. `offline_two_stage.yaml` 只保留轻量入口配置：`impl: spad_rag`、`impl_config_ref: agent_training`、输入输出契约。
4. SPAD-RAG 的内部 sub-stage、prompt、reward、dataset、teacher profile 和 loss 配置放在 `AgenticIterRag/config/agent_training/spad_rag_base.yaml`。
5. 资源占用放在 `resource.stage_resources.train_agent.impls.spad_rag.sub_stages.*`，按 sub-stage 管理，不写死在 teacher profile 里。
6. 所有可从 CAR/AIR 复用的代码，要复制到 AIR 内部再适配，不能让 AIR 新代码依赖 AIR 之外的项目代码；`src` 中已有公共代码除外。
7. 所有新增代码和配置都要按 AIR 当前风格写必要注释，尤其是配置字段、prompt version、reward breakdown、manifest schema 和资源字段。

## 3. 需要落地的工程改动

配置侧：

1. 新增 Qwen3-1.7B model config，路径由执行前确认。
2. 扩展 config compiler，使其支持 `agent_training` 配置组。
3. 扩展 main run 默认配置或实验 overlay，使本实验选择 `agent_training: spad_rag_base`。
4. 在 `offline_two_stage.yaml` 的 `train_agent` 下加入 `impl` 和 `impl_config_ref`，但不展开 SPAD-RAG 三阶段细节。
5. 新增或扩展 SPAD 专用 overlay，第一版选择 Qwen3-1.7B + GLM4.7 teacher。
6. 扩展 resource config，在 `train_agent.impls.spad_rag.sub_stages` 下配置 Stage 1、Stage 2、Stage 3 的资源。

代码侧：

1. 将 `main_train_agent.py` 从 placeholder 改为真实入口，根据 `pipeline.stage_configs.train_agent.impl` 分发到 `spad_rag`。
2. 修改 AIR pipeline runner，使 `train_agent` 调用真实入口，并把 `agent_checkpoint`、`agent_training_manifest` 写回 final config。
3. 新增 AIR 内部 SPAD-RAG 模块，至少包含：
   - config resolver
   - sub-stage runner
   - service manager
   - actor/search prompt builder
   - teacher answer prompt builder
   - trajectory parser
   - reward calculator
   - teacher answer client
   - Stage 2 refresh dataset builder
   - Stage 3 SFT/DPO dataset and trainer wrapper
   - manifest/report writer
4. service manager 第一版必须支持 GLM-4.7-Flash local vLLM 单实例服务；teacher profile 只描述服务形态，实际卡号来自 sub-stage resource。
5. reward calculator 必须实现 teacher 短路逻辑：格式错误、非法 action、no finish 不调用 teacher。
6. Stage 2 dataset builder 必须保存 `messages_before_final_answer`，并保证 Stage 3 prompt 与 actor rollout 到回答时刻的上下文一致。

## 4. 第一版资源默认

第一版按 8 卡本地资源设计：

```text
actor train / rollout: [0, 1, 2, 3]
GLM4.7 teacher vLLM: [4, 5]
recall service: [6, 7]
```

注意：

1. GLM4.7 必须占两张卡。
2. actor llm 默认占四张卡，即使第一版 actor 是 Qwen3-1.7B，也先保持和当前 AIR/CAR 训练资源结构一致。
3. recall 不默认用 CPU；Stage 1 和 Stage 2 默认使用 NPU/CUDA accelerator backend。
4. Stage 3 不需要 teacher 和 recall 服务，只需要 DPO/SFT 训练资源。
5. 资源校验必须按当前 active sub-stage 做，不能把顺序执行的 sub-stage 资源误判为同一时刻冲突。

## 5. 执行顺序

### 5.1 前置校验

先做不训练的校验：

1. 确认 Qwen3-1.7B 模型路径存在。
2. 确认 GLM-4.7-Flash 模型路径存在。
3. dry-run 编译配置，确认 `agent_training`、`pipeline`、`resource`、`model`、`rollout` 能合并出合法 final config。
4. dry-run 生成 execution plan，确认 selected stage 是 `train_agent`，且 active impl 是 `spad_rag`。
5. 启动/预检 GLM teacher service 和 recall service 的最小 health check。

如果 Qwen3-1.7B 路径不存在，停止执行，不自动换模型。当前前置校验结果是路径存在。

### 5.2 完整链路 smoke

先跑 tiny 规模完整链路：

```text
search_policy_rl: 1 training step
answer_refresh_data: 少量样本
answer_distillation: DPO 1 training step，SFT 关闭
```

验收标准：

1. 三个 sub-stage 都能正常结束。
2. 每个 sub-stage 都有 manifest。
3. Stage 1 有 reward breakdown，并能看到 teacher_called / teacher_skip_reason。
4. Stage 2 能生成 chosen/rejected 数据。
5. Stage 3 能读取 Stage 2 数据并写出 checkpoint。
6. `train_agent.outputs.agent_checkpoint` 能写回 final config。

### 5.3 sub-stage 效率消融

每个 sub-stage 单独做少量 step 的效率消融。

默认设置：

```text
search_policy_rl: 2 training steps
answer_refresh_data: 小样本数据生产，不做训练
answer_distillation: DPO 2 training steps，SFT 关闭
```

记录指标：

1. Stage 1：step time、rollout latency、teacher QPS、recall latency、valid trajectory rate、teacher skip rate、平均 search 次数。
2. Stage 2：samples/min、actor answer latency、teacher answer latency、recall latency、chosen/rejected 构造成功率。
3. Stage 3：step time、tokens/sec、显存占用、loss 是否正常下降或至少数值稳定。

消融任务监控策略：

1. 启动阶段每 30 秒探查一次状态。
2. 进入训练或稳定数据生产后，每 5 分钟探查一次状态。
3. 探查内容包括：主进程是否存活、服务 health、GPU/NPU 占用、最新日志错误、最新 metrics step、checkpoint 或 dataset 是否持续写入。

### 5.4 Stage 1 5-step 训练效果消融

只消融 Stage 1 `search_policy_rl` 的训练效果，不消融 Stage 3 的训练效果。

默认设置：

```text
enabled sub-stage: search_policy_rl only
training steps: 5
teacher: GLM-4.7-Flash
actor init: Qwen3-1.7B
```

对比指标：

1. format valid rate
2. legal stop rate
3. teacher F1 / total reward
4. 平均 search 次数
5. duplicate query rate
6. no finish rate
7. evidence insufficient rate

如果 5 step 后出现明显格式崩坏、legal stop rate 大幅下降、teacher 调用大量被 short-circuit，则暂停正式训练，先修 prompt/reward/stop 逻辑。

### 5.5 正式训练

正式训练默认三阶段全跑：

```text
Stage 1: search_policy_rl full training
Stage 2: answer_refresh_data full refresh
Stage 3.1: SFT disabled
Stage 3.2: DPO enabled
```

正式训练监控策略：

1. 启动后按 30 秒间隔确认服务和训练进程进入正常状态。
2. 确认第一个 training step 正常完成后，就不再持续人工跟踪。
3. “第一个 training step 正常”至少包括：loss/reward 有记录、metrics JSONL 有 global step、没有服务错误、没有 OOM、checkpoint/output 路径可写。

## 6. Prompt 和 reward 执行要求

Actor prompt：

1. 沿用 CoAgenticRetriever / Search-R1 的搜索 agent 语义，但复制到 AIR 内部实现。
2. 每个 assistant turn 输出 `<reason>`，然后输出 `<tool_call>` 或 `<answer>`。
3. Stage 1 在 `<answer>` stop，不让 answer body 影响 search policy reward。
4. Stage 2 不在 `<answer>` stop，要生成 actor answer 作为 rejected。

Teacher prompt：

1. teacher 是 evidence-grounded answer-only QA，不是 search agent。
2. teacher 不看 actor answer，只看原始问题、每轮 sub query、每轮 visible top5 docs。
3. teacher 只能使用 evidence，不允许用参数知识补全。
4. teacher 认为证据不足时，`<reason>` 写缺什么证据、当前证据为什么不足，`<answer>` 输出 `证据不足无法作答`。
5. teacher 输出格式固定为 `<reason>...</reason><answer>...</answer>`。
6. 关闭原生 thinking 应通过 chat template / inference 参数控制，不在 prompt 中写“关闭原生 thinking”这类容易污染任务语义的句子。

Reward：

1. teacher answer 是 Stage 1 reward function 的一部分。
2. teacher 只在 trajectory 格式有效且 actor 合法停止时调用。
3. actor answer body 不参与 Stage 1 search trajectory 质量判断。
4. reward manifest 必须保存 total reward、teacher F1、search cost、format penalty、duplicate query penalty、no finish penalty、teacher skip reason。

## 7. 产物和验收

每次运行至少产出：

1. final config YAML
2. execution plan YAML
3. pipeline manifest
4. train_agent manifest
5. SPAD sub-stage manifests
6. Stage 1 rollout/reward sample
7. Stage 2 chosen/rejected dataset manifest
8. Stage 3 checkpoint 或 smoke checkpoint
9. efficiency metrics report
10. Stage 1 5-step effect report

正式训练前必须确认：

1. smoke 完整通过。
2. sub-stage 效率消融没有 OOM、服务阻塞、显存泄露或明显吞吐异常。
3. Stage 1 5-step 效果消融没有暴露格式崩坏或 reward 短路过高的问题。

## 8. 默认假设

1. Qwen3-1.7B 模型路径使用 `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B`。
2. GLM-4.7-Flash 使用本地 vLLM 服务，不使用线上 API。
3. 第一版 teacher 只要求 GLM4.7 单实例服务跑通；Qwen32B 三实例 proxy 保留在详细设计中，不作为第一版执行阻塞项。
4. Stage 3 SFT 默认关闭，但代码和配置保留接口。
5. Stage 3 DPO 默认开启，loss 包含 pairwise DPO loss 和 chosen-answer SFT loss。
6. 正式训练启动后只人工确认第一个 training step 正常，不持续值守整个训练。

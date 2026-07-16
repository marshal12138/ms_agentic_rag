# SPAD Teacher PE 消融指令汇总

本文档汇总用户在本轮 SPAD teacher prompt engineering 工作中给出的目标、约束和执行要求。
它记录“要怎么做”，不替代 `ABLATION_HISTORY.md` 中的实验结果，也不把 Codex 的推断写成用户指令。

## 0. 2026-07-15 新数据消融覆盖条款

以下要求晚于本文其余章节，发生冲突时以本节为准：

1. 从最近一次 5100-step 新数据训练实验的 rollout 中，按四个 20-step 层抽取 512 个 group，每个 group 只取一条轨迹，并完成人工 S/I/A 标注。
2. 新消融同时关注 I 标签 precision/recall 与 teacher answer 对 gold answer 的命中程度，两部分同等重要。
3. 优先沿用历史消融中已成功的 question-tail evidence-only 路径，再做有依据的正交变化。
4. 资源加载阶段每 2 分钟探查一次；未到时间不查询、不报告。
5. 进入推理后每 1-5 分钟探查一次，按预计任务时长选择间隔，禁止用 30 秒或 1 分钟轮询伪装等待，也禁止为了维持命令而反复调用 wait/write_stdin。
6. 反复消融期间持续复用同一组 vLLM 服务，不重启；只有用户明确要求停止消融时才关闭服务。
7. 本轮抽样、标注、代码、推理结果和工作日志仍全部落在本目录。

## 1. 工作目录与落盘

1. 所有 prompt engineering 代码、运行产物和结论统一放在：
   `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/teacher_PE`。
2. 先把 237 条人工判断保存成表格，再开始实现和消融。
3. 每次推理都要落盘输入、输出和指标，不能只在终端保留临时结果。
4. 新生产的数据必须从 agent LLM 开始完整生成并落盘，不能只构造 teacher 输入。
5. 文档、计划、代码说明和最终结论均应保存在上述目录，并由 README 解释用途。

## 2. 人工判断与 S/I/A 定义

1. 237 条判断来自三个历史 teacher bucket 的人工审查；evidence 是否存在必须人工理解，不能用 alias
   匹配等代码规则代替人工判断。
2. S/I/A 始终针对 **Original question 与当前累计 Search evidence**，不是针对 actor 生成的 sub-query。
3. `S / supported_answer`：当前证据支持可回答 Original question 的完整答案。
4. `I / insufficient_evidence`：当前证据缺少回答 Original question 所必需的事实或关系。
5. `A / ambiguous_evidence`：针对 Original question，当前证据支持多个同样满足约束但互不兼容的答案。
6. A 不必然表示数据集 gold 有多个；它可能来自原问题固有歧义，也可能来自检索/指代范围过宽。
7. S 与 A 混淆可以暂时接受，但 I 不应与 S/A 混淆。

## 3. Prompt 版本工程要求

1. YAML 中的 `teacher_answerer.prompt_version` 必须真实透传到 `build_teacher_messages`，不能只有假配置。
2. prompt version 必须有 registry、未知版本提前报错、输出中记录实际版本，并可回退。
3. 历史 500-run prompt 与当前 prompt 必须能用不同版本明确区分。
4. 当前 teacher user prompt 的层级格式为：

```text
Original question:
   {question}

Search evidence:
   Round {round_index}:
      sub_query: {retrieval_query}
      retrieved contents:
         [1] {title_1}
             {contents_1}
         ...
         [5] {title_5}
             {contents_5}
```

5. 修改 prompt 后必须说明实际发给 teacher 的结果，不能只改文档示例。

## 4. 两条核心消融思路

### 4.1 Instruction-only

1. 第一条思路保持输入内容与当前 prompt 一致，只改变 instruction 部分。
2. 目标是让 GLM-4.7-Flash 的 S/I/A 判断与 237 条人工看法对齐。
3. 应系统尝试不同判定规则，而不是只做一次 prompt 修改。

### 4.2 Gold-aware

1. 第二条思路在输入中增加 gold answer。
2. 让 GLM-4.7 判断当前 evidence 是否支持该答案。
3. Gold 不能被当成 evidence；必须检查 Original question 到 gold 的完整证据关系。
4. Gold-aware 与无 gold 方案分开报告，不能混成同一种生产设置。

## 5. 指标与选择规则

1. S/A 都尽量与人工对齐，但主目标是 I。
2. 若只选一个目标，则要求 I precision 与 I recall 尽量接近 1.0。
3. 理想停止线为 I precision 和 recall 都超过 0.98。
4. 双 0.98 是上限，不是必须交付门槛；若没有达到，使用实际效果最好的方案。
5. S 与 A 互相混淆可作为妥协，但任何涉及 I 的错误都必须单独统计。
6. 不能只看一次运行的最高分；应考虑评估集、多次运行稳定性、格式成功率、时耗和调用成本。

## 6. 数据切分与数据泄漏

1. 237 条数据在概念上拆分为 observation（类似训练集）与 evaluation/holdout。
2. 同一 Original question 的重复样本不得跨 split，避免问题级泄漏。
3. Prompt 选择主要使用 observation，冻结后再看 evaluation。
4. Few-shot 坚决不能包含现有 237 条数据：不能放入其问题、答案、evidence、人工标签，也不能放入
   可识别改写或从中派生的实体/数值案例。
5. 违反该规则的实验必须标记为数据泄漏并从所有比较、推荐和生产选择中排除。
6. 合法 few-shot 只能使用与现有数据无关的独立虚构示例。

## 7. 输出格式与长度

1. 不在 `</status>` 处提前停止生成；保留完整 `<reason><status><answer>`。
2. `<answer>` 应尽量像 gold answer，保持简短，只输出必要答案 span，不输出解释或冗余前缀。
3. Reason 可以存在，但应尽量简短，不能让长 reasoning 挤掉最终 status/answer。
4. Thinking 可以后续测试；优先穷尽 no-thinking 方法。
5. 当前后端若把 thinking 直接写入 content，必须考虑耗时、token 和截断，不能只看准确率。

## 8. 需要尝试的方法

1. 先穷尽 no-thinking 的单 prompt 方法。
2. 尝试 few-shot，但必须遵守无现有数据泄漏约束。
3. 尝试多个 prompt 投票。
4. 尝试多个 prompt 组成串行工作流，例如首判、critic、arbiter。
5. 尝试 gold-aware 判断。
6. Thinking 模式放在 no-thinking 方法之后，并重点记录时耗。
7. 多调用策略必须报告额外调用数、token 和延迟；若没有明显收益，不作为最终方案。

## 9. vLLM 与硬件执行

1. 使用 8 张卡启动 GLM-4.7 teacher 实例。
2. 已采用 4 个 replica、每个 TP=2 的拓扑；prompt 改变后无需重启 vLLM 服务。
3. 首次确认新 prompt 全量推理成功后，普通 237 条 no-thinking 批次约几十秒，可约 30 秒探查一次。
4. 若 thinking 或其他配置明显变慢，应根据实际耗时延长探查间隔，避免频繁轮询。
5. 离线消融中，能并行的策略应并行，不能拖慢消融时间。
6. 并行不能导致 OOM；需要控制每个 replica 的合计并发并检查服务状态和 KV cache。
7. “真实资源稀缺”用于评价策略成本，不表示离线实验必须串行；策略之间仍应安全并行。
8. 有效消融和独立重复禁止复用 runner response cache；运行时使用 `--disable-cache`，并要求
   `run.json.cache_hits=0`。vLLM prefix KV cache 只复用前缀计算、不复放历史答案，应与 response cache
   分开记录，不能把 cache replay 当成新推理。

## 10. 运行节奏与记录

1. 在未达到双 0.98 时持续进行有依据的消融；原定本轮运行到北京时间 18:00 左右。
2. 每尝试 10 个有效消融方案，就把方案、效果和反思写入历史文档。
3. 数据泄漏实验不计入有效十次，但必须保留审计记录。
4. 消融时间结束前，先记录当前维持策略和结论，再列持续消融计划。
5. 同时生成代码结构介绍，并在 README 中说明每个文档的作用。
6. 最终汇报必须分别回答“目前指标最佳的策略”和“综合准确率、稳定性与时间成本后的最佳策略”，
   并区分单次最高观测与多次重复运行的可复现结论。

## 11. 达标后的新 200 条验证

1. 若 I precision/recall 达到双 0.98，冻结 prompt。
2. 从 agent LLM 开始重新生产 200 条新数据并完整落盘。
3. 在新数据上重新评估，确认分数位置，不能只报告对 237 条数据的拟合效果。
4. 即使未达到上限而选择最佳方案，后续生产集成前仍应按 `持续消融计划.md` 做独立新数据验证。

## 12. 当前文档关系

- 本文：用户给出的 PE 目标和约束。
- `PLAN.md`：开始消融前制定的工程执行计划。
- `ABLATION_HISTORY.md`：实际尝试、每十次复盘和淘汰原因。
- `REPLICA_STABILITY.md`：多次独立运行的均值与范围。
- `持续消融计划.md`：当前维持策略和下一阶段动作。
- `代码结构介绍.md`：代码、数据和结果目录结构。

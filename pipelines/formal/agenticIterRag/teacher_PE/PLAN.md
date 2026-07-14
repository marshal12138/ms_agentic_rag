# GLM-4.7 Teacher Prompt Engineering 消融计划

日期：2026-07-10

## 1. 实验目标

在固定的 237 条人工审计样本上，使 GLM-4.7-Flash 对 Original question 的证据状态判断
尽量与人工 `S/I/A` 对齐：

- `S / supported_answer`：当前累计 evidence 支持一个完整、可回答 Original question 的
  唯一答案；问题本身要求集合时，完整答案集合视为一个答案。
- `I / insufficient_evidence`：没有任何候选具备回答 Original question 所需的完整事实链。
- `A / ambiguous_evidence`：至少两个完整候选同时满足 Original question 的实体、谓词和
  范围约束，但当前问题与 evidence 无法选择唯一候选。

S/I/A 的判定对象始终是 Original question。actor 生成的 `sub_query` 只解释 evidence 的
来源，不是 teacher 要回答或分类的问题。

## 2. 优先指标

完整报告保留三分类 accuracy、macro-F1、每类 precision/recall 和混淆矩阵，但选择 prompt
时采用以下优先级：

1. `I precision`：人工非 I 的样本不得被错误判成 I，目标 1.0。
2. `I recall`：人工 I 的样本不得漏判，目标 1.0。
3. `I binary accuracy`：将 S/A 合并为 non-I 后的二分类准确率。
4. XML parse rate：必须为 1.0。
5. 三分类 macro-F1 与 accuracy。

S 与 A 之间的混淆单独统计为 `tolerated_SA_confusion`。它仍是三分类错误，但在 I 指标达到
目标前，不作为淘汰 prompt 的首要原因。所有涉及 I 的错误分为：

- `false_I`：人工 S/A，模型 I，降低 I precision。
- `missed_I_as_S`：人工 I，模型 S。
- `missed_I_as_A`：人工 I，模型 A；后两者降低 I recall。

## 3. 数据冻结与防泄漏

1. `manual_judgments_237.tsv` 是唯一人工标签源。
2. 从 TSV 引用的原始 rollout 行恢复 Original question、每轮 sub-query 和可见 top-5
   passage，生成 `benchmark_237.jsonl`。
3. 每条 case 保存 rollout `uid`、源文件/行号、evidence 内容 SHA256、manual label 和 gold；
   runner 不再依赖运行中的 retriever。
4. 按规范化 Original question 分组切分 development/holdout；同一问题在不同 teacher
   bucket 中出现时必须位于同一侧。
5. baseline 可以在全量 237 条上运行；prompt 选择只看 development。holdout 在候选冻结后
   运行，避免反复调 prompt 后把训练集分数当泛化结果。
6. Gold-aware 实验显式使用标签信息，只作为可实现上界和 teacher 设计对照，不与无 gold
   的生产 prompt 直接比较为同一设置。

## 4. 固定推理条件

- Model：`GLM-4.7-Flash`
- 设备：8 张 Ascend 910B3
- 服务拓扑：4 个 vLLM replica，每个 `TP=2`
- 卡分配：`[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`
- 端口：`8067`、`8068`、`8069`、`8070`
- `temperature=0.0`
- `top_p=1.0`
- `max_tokens=512`
- `enable_thinking=false`
- 完整生成 `<reason><status><answer>`，不在 `</status>` 截止。
- `<answer>` 必须采用最短 evidence span，避免解释、前缀、完整句和非必要候选列表，尽量
  接近 gold-answer 风格。
- 每个 replica 有独立请求队列；237 条按 least-inflight 并行分发。
- 每个响应、错误、耗时、endpoint、prompt hash 和原始 XML 全量落盘。

## 5. 实验线 A：Instruction-only

该实验线严格保持当前 v2 user 输入内容与顺序不变：

```text
Original question
Search evidence
  Round / sub_query / top-5 retrieved contents
```

不加入 gold、不改 evidence、不改文档截断、不增加额外 user 字段，只替换 system 指令。

### A0 `baseline_current_v2`

使用当前生产 `TEACHER_STATUS_SYSTEM_PROMPT`，获得层级化 user layout 下的 live baseline。
历史 78.5% 来自旧 layout 和历史输出，只作为参考，不代替 A0。

### A1 `candidate_count`

要求模型先在 reason 中完成以下判定，再输出状态：

1. 固定 Original question 的目标实体、谓词、时间/版本范围。
2. 只把能够完成整条关系链的答案计为 `complete candidate`。
3. `0` 个 complete candidate -> I。
4. `1` 个 complete candidate -> S。
5. `>=2` 个 complete candidate -> A。

不同谓词、只命中相关实体、缺少 bridge、外部知识补全和未出现别名均不能形成 complete
candidate。

### A2 `candidate_count_i_guard`

在 A1 上加强 I 边界：

- 判 I 前必须明确指出缺失的最小事实或 bridge。
- passage 内直接读取、一步算术、日期区间端点、显式缩写展开和简单所有权组合不得判 I。
- 判 S/A 前，每个 complete candidate 必须引用 passage 编号并覆盖所问谓词。
- 同一事实的兼容粒度、上下位表述和不同谓词不得制造 A。

### A3 `candidate_count_i_guard_compact`

与 A2 决策规则相同，但压缩措辞和 reason 模板，检查收益是否来自规则本身而非更长上下文。

Instruction-only 候选先在 development 上运行。保留 Pareto 最优候选：优先 I precision/recall，
其次三分类 macro-F1，再比较延迟和输出长度。

## 6. 实验线 B：Gold-aware

在 A 线最佳 system 指令基础上，user 输入增加：

```text
Reference gold answer:
   {gold_answers}
```

Gold 只作为“待验证候选”，不能作为证据。模型必须判断当前 evidence 是否建立了
Original question 到 gold 的完整关系链。

### B1 `gold_support_check`

- evidence 完整支持 gold，且没有同范围竞争候选 -> S。
- evidence 完整支持 gold，同时存在同范围竞争候选 -> A。
- evidence 不支持 gold，但支持唯一其他完整答案 -> S。
- evidence 支持多个其他完整答案 -> A。
- evidence 既不支持 gold，也没有其他完整答案 -> I。

### B2 `gold_i_guard`

在 B1 上要求显式输出：gold 是否被支持、支持 gold 的 passage、其他 complete candidates、
以及缺失 bridge。目标是减少 teacher 因只看到相关实体就误判 S，也避免 evidence 已含简单
答案时误判 I。

Gold-aware 结果必须单独报告。若它显著提高 I 指标，说明 gold 能帮助绑定问题隐藏意图；
这不能直接部署到无 gold 的在线 Stage1，但可以用于离线数据过滤或 teacher labeling。

## 7. 评测与选择

每个 variant 生成：

- `predictions.jsonl`：逐样本 messages、raw response、parsed status、answer、reason、耗时。
- `metrics.json`：总体与 dev/holdout 指标、三分类混淆矩阵、I 二分类指标。
- `errors.tsv`：所有不一致样本及人工原因。
- `report.md`：指标表、I 错误清单、S/A 容忍混淆和相对 baseline 翻转。

候选选择采用词典序约束：

1. 最大化 `min(I precision, I recall)`。
2. 最大化 `I F1`。
3. 最小化涉及 I 的总错误数。
4. 最大化三分类 macro-F1。
5. 最小化平均延迟和输出 token。

不能为了在 development 上达到 1.0 而把人工标签或完整示例直接写进 prompt。few-shot 若后续
启用，只能来自 development，且作为第三条实验线单独报告。

## 8. 生产集成门槛

无 gold 的 instruction-only 候选只有同时满足以下条件才进入生产 registry：

- holdout I precision 和 recall 均不低于 live baseline。
- 涉及 I 的错误总数下降。
- XML parse rate 为 1.0。
- 三分类 macro-F1 不出现明显退化。
- prompt 版本可配置、可回退，Stage1、Stage2 HTTP 和 offline batch 使用同一版本。

Gold-aware 候选不直接替换在线 Stage1 prompt，只用于离线 teacher、数据清洗或作为可达上界。

## 9. 当前执行顺序

1. 修正文档中 A 的定义。
2. 构建并校验 237 条 benchmark 与 group split。
3. 实现 prompt variant registry、并行 runner、缓存和 scorer。
4. 启动 4 个 `TP=2` teacher replica，占用全部 8 张卡。
5. 运行 A0 baseline。
6. 在 development 上运行 A1-A3，选出 instruction-only 最优候选。
7. 冻结 A 线候选后运行 holdout。
8. 运行 B1-B2 gold-aware 对照。
9. 写出最终消融报告，再决定是否注册新生产 prompt。

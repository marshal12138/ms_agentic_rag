# AgenticIterRag 中间数据管理与字段设计草案

更新日期：2026-07-02

## 1. 目标

本文记录 AgenticIterRag v1 中间数据的目录、版本和字段设计。

`dataproduce` 只是一种 pipeline 流程，不是一类数据集。该流程当前会生产两类可复用数据集：

- `trajectory`：已有 agent 对指定数据推理后产生的搜索轨迹数据。
- `llm_reranker_train_set`：由 `trajectory` 构造出的 LLM reranker 数据集合。

`llm_reranker_train_set` 内部再分两层：

- `input_dataset`：结构化主数据，只描述 query、候选文档和来源，不绑定 prompt 模板。
- `train_dataset`：经过 prompt 拼装后的可训练数据。它必须在 `llm_reranker_train_set` 版本内部继续区分 prompt/template 版本。

所有长期需要复用和追溯的中间数据统一沉淀到：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/
```

`outputs/agenticIterRag/` 只保留单次 pipeline run 的审计产物、stage manifest 和 execution plan，不作为长期数据集目录。

## 2. 目录结构

### 2.1 Trajectory 数据集

```text
data/AgenticIterRag/
  trajectory/
    <trajectory_version>/
      example.json
      trajectory.jsonl
      raw_traces.jsonl
      metrics.jsonl
      summary.json
      manifest.json
      final_config.yaml
```

文件说明：

- `example.json`：从 `trajectory.jsonl` 中抽取的一条样例记录，用于人工快速理解字段。
- `trajectory.jsonl`：标准化后的 canonical trajectory 主数据。
- `raw_traces.jsonl`：推理后端直接输出的原始 trace，保留用于排查和重新抽取。
- `metrics.jsonl`：每个源样本的推理指标。
- `summary.json`：本版本轨迹数据的聚合统计。
- `manifest.json`：该 trajectory 数据集版本的来源、配置和产物索引。
- `final_config.yaml`：生成该数据集时使用的最终配置快照。

### 2.2 LLM Reranker Train Set 数据集

```text
data/AgenticIterRag/
  llm_reranker_train_set/
    <reranker_dataset_version>/
      example.json
      input_dataset/
        example.json
        dataset.jsonl
        dataset.parquet
        manifest.json
      train_dataset/
        <train_dataset_version>/
          example.json
          dataset.jsonl
          dataset.parquet
          manifest.json
      manifest.json
      source_trajectory.manifest.json
      final_config.yaml
```

文件说明：

- 顶层 `example.json`：默认指向或复制 `input_dataset/example.json`，用于快速理解该版本数据。
- `input_dataset/example.json`：从 `input_dataset/dataset.jsonl` 中抽取的一条样例记录。
- `input_dataset/dataset.jsonl`：LLM reranker 结构化主数据，不包含固定 prompt 拼装结果。
- `input_dataset/dataset.parquet`：结构化主数据的 Parquet 镜像；写入失败时不影响 JSONL 主产物。
- `input_dataset/manifest.json`：结构化主数据的样本数、字段版本和来源索引。
- `train_dataset/<train_dataset_version>/example.json`：从对应 prompt 版本训练数据中抽取的一条样例记录。
- `train_dataset/<train_dataset_version>/dataset.jsonl`：经过 prompt 拼装后的可训练数据。
- `train_dataset/<train_dataset_version>/dataset.parquet`：可训练数据的 Parquet 镜像。
- `train_dataset/<train_dataset_version>/manifest.json`：prompt/template 版本、formatter、输入数据版本和输出样本数。
- 顶层 `manifest.json`：该 reranker train set 版本的来源、配置和产物索引。
- `source_trajectory.manifest.json`：生成本数据集所依赖的 trajectory manifest 快照。
- `final_config.yaml`：生成该数据集时使用的最终配置快照。

## 3. `build_reranker_dataset` Stage 设计

`build_reranker_dataset` 是 pipeline 内部的一个父 stage，不应该拆成两个外部 task 或两个顶层 launcher。它的职责是把已有 trajectory 转换成 LLM reranker 可消费的数据集合。

但这个父 stage 内部必须拆成两个子 stage：

1. `build_input_dataset`：从 `trajectory` 生产结构化 `input_dataset`。
2. `build_train_dataset`：从 `input_dataset` 生产带 prompt 的 `train_dataset`。

这样拆分的原因是：`input_dataset` 和 `train_dataset` 的稳定性不同。`input_dataset` 绑定的是数据来源、候选文档抽取和标签策略；`train_dataset` 绑定的是训练范式、prompt 模板、formatter 和输出格式。后续我们会频繁调整 prompt/template 或训练格式，如果不拆开，就会导致每次改 prompt 都要重新解释为“重新从 trajectory 建数据”，不利于版本管理和复用。

### 3.1 父 Stage 职责

`build_reranker_dataset` 父 stage 只做三件事：

- 统一接收上游 trajectory 输入，或接收一个已经存在的 `input_dataset` manifest。
- 按配置决定是否执行两个内部子 stage。
- 汇总两个子 stage 的 manifest，写出 `llm_reranker_train_set/<reranker_dataset_version>/manifest.json` 和 pipeline stage manifest。

父 stage 不直接写具体样本字段，也不直接承载 label policy、prompt template、top-N/top-M 等细节参数。这些参数必须下沉到两个子 stage 的配置里。

### 3.2 子 Stage 一：`build_input_dataset`

`build_input_dataset` 的输入是 canonical trajectory，输出是结构化主数据：

```text
llm_reranker_train_set/<reranker_dataset_version>/input_dataset/
  example.json
  dataset.jsonl
  dataset.parquet
  manifest.json
```

它负责的事情包括：

- 从每条 trajectory record 中抽取 `question`、`sub_query` 和候选文档。
- 根据 `candidate_source` 选择候选来源，例如 `recall_topn_docs` 或 `ranked_docs`。
- 根据 `candidate_top_n` 截断候选数量。
- 根据 `dedupe_policy` 对候选文档按 `doc_id` 去重。
- 写入结构化字段，例如 `query_id`、`candidate_docs`、`source_trace_id`、`metadata`。
- 记录标签策略字段，但第一版默认不生成真实排序监督信号。

这个子 stage 的开关和配置应该独立存在：

```yaml
pipeline:
  stage_configs:
    build_reranker_dataset:
      sub_stages:
        build_input_dataset:
          enabled: true
          version: null
          overwrite: false
          derive_version_from_trajectory: true
          schema_version: v1
          builder_policy: recall_topn_to_candidates
          candidate_source: recall_topn_docs
          candidate_top_n: 50
          dedupe_policy: keep_first
          label_policy: null
          positive_policy: null
          target_ranking_policy: none
```

字段解释：

- `enabled`：是否执行从 trajectory 到 input_dataset 的转换。
- `version`：手动指定 `llm_reranker_train_set/<reranker_dataset_version>`；为空时自动生成。
- `overwrite`：版本目录已存在时是否允许覆盖。
- `derive_version_from_trajectory`：是否优先从 trajectory 版本名派生 input_dataset 版本。
- `schema_version`：input_dataset 字段契约版本。
- `builder_policy`：构造策略名称，用于 manifest 和版本 hash。
- `candidate_source`：候选文档来源字段。
- `candidate_top_n`：每个 query 保留的候选数量。
- `dedupe_policy`：候选文档去重策略。
- `label_policy`：标签策略。当前默认 `null`，表示不构造监督标签。
- `positive_policy`：正例策略。当前默认 `null`。
- `target_ranking_policy`：目标排序策略。当前默认 `none`，明确不把 no-ranker trace 顺序当强监督排序。

### 3.3 子 Stage 二：`build_train_dataset`

`build_train_dataset` 的输入是 `input_dataset/manifest.json` 或 `input_dataset/dataset.jsonl`，输出是某个 prompt/template 版本的训练数据：

```text
llm_reranker_train_set/<reranker_dataset_version>/train_dataset/<train_dataset_version>/
  example.json
  dataset.jsonl
  dataset.parquet
  manifest.json
```

它负责的事情包括：

- 读取结构化 `input_dataset`。
- 根据训练范式生成训练样本，例如第一版 `format=grpo`。
- 根据 prompt 模板版本渲染 prompt。
- 根据 formatter 输出目标训练框架需要的字段，例如 `verl_chat`。
- 写入 `reward_model`、`extra_info`、`candidate_index_to_doc_id` 等训练和审计字段。
- 写入 train_dataset 自己的 manifest，记录它来自哪一个 input_dataset 版本。

这个子 stage 的开关和配置也必须独立存在：

```yaml
pipeline:
  stage_configs:
    build_reranker_dataset:
      sub_stages:
        build_train_dataset:
          enabled: true
          version: null
          overwrite: false
          format: grpo
          prompt_template_version: cosearch_tags_v1
          formatter: verl_chat
          ground_truth_policy: empty
          reward_policy: none
          output_schema: cosearch_rerank_tags
          reranker_top_m: null
          max_doc_chars: 2000
```

字段解释：

- `enabled`：是否执行从 input_dataset 到 train_dataset 的转换。
- `version`：手动指定 `train_dataset/<train_dataset_version>`；为空时自动生成。
- `overwrite`：该 train_dataset 版本已存在时是否允许覆盖。
- `format`：训练范式。第一版优先实现 `grpo`。
- `prompt_template_version`：prompt 模板版本。第一版使用 `cosearch_tags_v1`。
- `formatter`：训练数据格式化器。第一版使用 `verl_chat`。
- `ground_truth_policy`：ground-truth 写入策略。第一版使用 `empty`，即显式为空。
- `reward_policy`：reward 策略。第一版使用 `none`，后续再接真实排序 reward。
- `output_schema`：模型输出格式。第一版是 CoSearch 的 `<reason>` + `<rerank>` 标签格式。
- `reranker_top_m`：要求 reranker 输出多少个候选。为空时可从 `infer_runtime.retrieval.visible_top_m` 派生。
- `max_doc_chars`：每篇候选文档写入 prompt 的最大字符数。

### 3.4 两个开关的组合语义

两个子 stage 的 `enabled` 组合需要有明确语义：

| build_input_dataset | build_train_dataset | 含义 |
| --- | --- | --- |
| true | true | 默认全流程：从 trajectory 生成 input_dataset，再生成 train_dataset。 |
| true | false | 只沉淀结构化 input_dataset，暂不渲染 prompt 训练数据。适合先检查候选文档和字段质量。 |
| false | true | 复用已有 input_dataset，只重新生成某个 prompt/template 版本的 train_dataset。必须显式提供 `input_dataset_manifest`。 |
| false | false | 父 stage 无实际数据产出，应视为配置错误，除非 runner 明确支持 no-op dry-run。 |

### 3.5 推荐的父 Stage 配置结构

完整配置建议如下。后续真正实现时，应把这些字段写入 `AgenticIterRag/config/pipeline/offline_two_stage.yaml`，并在 task overlay 中只覆盖实验需要变化的少数字段。

```yaml
pipeline:
  stage_configs:
    build_reranker_dataset:
      enabled: true
      resource_key: local_cpu
      sub_stage_order:
        - build_input_dataset
        - build_train_dataset
      inputs:
        canonical_trace_jsonl: null
        trajectory_manifest: null
        input_dataset_manifest: null
      outputs:
        input_dataset_manifest: null
        train_dataset_manifest: null
        reranker_train_set_manifest: null
        manifest: null
      sub_stages:
        build_input_dataset:
          enabled: true
          version: null
          overwrite: false
          derive_version_from_trajectory: true
          schema_version: v1
          builder_policy: recall_topn_to_candidates
          candidate_source: recall_topn_docs
          candidate_top_n: 50
          dedupe_policy: keep_first
          label_policy: null
          positive_policy: null
          target_ranking_policy: none
          outputs:
            dataset_jsonl: null
            dataset_parquet: null
            example_json: null
            manifest: null
        build_train_dataset:
          enabled: true
          version: null
          overwrite: false
          format: grpo
          prompt_template_version: cosearch_tags_v1
          formatter: verl_chat
          ground_truth_policy: empty
          reward_policy: none
          output_schema: cosearch_rerank_tags
          reranker_top_m: null
          max_doc_chars: 2000
          outputs:
            dataset_jsonl: null
            dataset_parquet: null
            example_json: null
            manifest: null
```

### 3.6 断点与重跑边界

pipeline 顶层仍然只暴露 `build_reranker_dataset` 一个 stage，不新增 shell launcher，也不新增 task。外部调度仍通过：

```text
pipeline.resume_from_stage=build_reranker_dataset
pipeline.stop_after_stage=build_reranker_dataset
pipeline.force_rerun_stages=[build_reranker_dataset]
```

控制父 stage。

父 stage 内部需要额外支持子 stage 级别的重跑控制。建议后续新增：

```yaml
pipeline:
  stage_configs:
    build_reranker_dataset:
      resume_from_sub_stage: null
      stop_after_sub_stage: null
      force_rerun_sub_stages: []
```

但第一版可以先不实现复杂子 stage 断点，只通过两个 `enabled` 开关满足最关键需求：

- 只构造 `input_dataset`：`build_input_dataset.enabled=true`，`build_train_dataset.enabled=false`。
- 只重新渲染 `train_dataset`：`build_input_dataset.enabled=false`，`build_train_dataset.enabled=true`，同时提供已有 `input_dataset_manifest`。

### 3.7 与配置职责的关系

`build_input_dataset` 和 `build_train_dataset` 的配置应放在 `pipeline.stage_configs.build_reranker_dataset.sub_stages` 下，而不是放在 `reranker_training` 下。

原因：

- `candidate_top_n`、`label_policy`、`positive_policy` 属于数据生产策略，不属于 reranker 模型训练超参。
- `prompt_template_version`、`formatter`、`output_schema` 属于 train_dataset 生产策略，也不属于训练器本身。
- `reranker_training` 应只保留训练阶段消费数据所需的字段，例如 `dataset_manifest`、`train_file`、`base_model`、batch size、learning rate、max length、checkpoint 策略。

这样可以保证同一份 `input_dataset` 能派生多个 `train_dataset`，而不需要复制训练配置或改动 task shell。

## 4. 版本命名

版本管理分三层理解：

1. `trajectory/<trajectory_version>` 是 agent 推理轨迹版本。
2. `llm_reranker_train_set/<reranker_dataset_version>` 是结构化 reranker 输入数据版本，也就是 `input_dataset` 的外部版本。
3. `train_dataset/<train_dataset_version>` 是从同一个 `input_dataset` 经过某个 prompt/template 拼装后得到的训练数据版本。

不要再给 `input_dataset/` 单独加一层版本目录。否则目录会变成三层版本，后续引用和 manifest 都会变复杂。当前约定是：

```text
llm_reranker_train_set/<reranker_dataset_version>/input_dataset/
```

其中 `<reranker_dataset_version>` 就代表这一版 `input_dataset`。

新增统一配置字段建议如下：

```yaml
data_artifacts:
  root: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag

  trajectory:
    version: null
    overwrite: false

  llm_reranker_train_set:
    version: null
    overwrite: false
    derive_version_from_trajectory: true
    input_dataset:
      schema_version: v1
      builder_policy: recall_topn_to_candidates
      candidate_source: recall_topn_docs
      candidate_top_n: 50
      label_policy: null
      positive_policy: null
      target_ranking_policy: none
    train_dataset:
      version: null
      overwrite: false
      format: grpo
      prompt_template_version: cosearch_tags_v1
      formatter: verl_chat
      ground_truth_policy: empty
      reward_policy: none
      output_schema: cosearch_rerank_tags
```

### 4.1 Trajectory 版本规则

- 如果 `trajectory.version` 非空，轨迹数据集使用该版本名。
- 如果 `trajectory.version` 为空，自动生成：

```text
yymmdd-HHMMSS_AIR_v1_traj_<data_slug>_<agent_slug>_<config8>
```

示例：

```text
260702-153012_AIR_v1_traj_cosearch_gs79_abcd1234
```

### 4.2 Input Dataset 外部版本规则

`llm_reranker_train_set.version` 表示结构化主数据 `input_dataset` 的版本。它不是 prompt 拼装后的训练数据版本。

- 如果 `llm_reranker_train_set.version` 非空，直接使用该版本名。
- 如果 `llm_reranker_train_set.version` 为空且 `derive_version_from_trajectory=true`，自动从 trajectory 版本派生：

```text
<trajectory_version>__input_<builder_policy_slug>_<config8>
```

当前默认策略下，建议展开为：

```text
<trajectory_version>__input_recall50_emptylabel_<config8>
```

含义：

- `<trajectory_version>`：说明这版 input dataset 来自哪版 agent 轨迹。
- `input`：说明这是结构化 reranker 输入数据。
- `recall50`：候选文档来自 recall top-50。
- `emptylabel`：当前不构造正例和排序监督标签。
- `<config8>`：input dataset 构造配置哈希。

如果 `derive_version_from_trajectory=false`，自动生成：

```text
yymmdd-HHMMSS_AIR_v1_input_<trajectory_slug>_<builder_policy_slug>_<config8>
```

### 4.3 Train Dataset 内部版本规则

已确认第一版优先实现 GRPO 格式，并且 ground-truth 暂时为空。因此 train dataset 自动版本名建议为：

```text
grpo_<prompt_template_version>_<formatter>_gt-empty_reward-none_<config8>
```

示例：

```text
grpo_cosearch-tags-v1_verl-chat_gt-empty_reward-none_a1b2c3d4
```

含义：

- `grpo`：训练范式。
- `cosearch-tags-v1`：复用 CoSearch 的 reranker prompt，要求输出 `<reason>` 和 `<rerank>` 两个标签块。
- `verl-chat`：面向 VERL chat prompt 格式。
- `gt-empty`：ground-truth 暂时为空。
- `reward-none`：当前还没有真实排序 reward 信号。
- `<config8>`：prompt 拼装配置哈希。

具体规则：

- 如果 `train_dataset.version` 非空，使用该版本名。
- 如果 `train_dataset.version` 为空，根据 `format`、`prompt_template_version`、`formatter`、`ground_truth_policy`、`reward_policy` 和配置哈希自动生成。
- 如果 `prompt_template_version` 或 `formatter` 为空，当前阶段不强制生成 `train_dataset`，只生成 `input_dataset`。

### 4.4 覆盖规则

- `overwrite: true`：允许清空并重写对应版本目录。
- `overwrite: false`：
  - 手动版本目录已存在时直接失败。
  - 自动版本目录冲突时追加 `_r01`、`_r02`、`_r03`。

## 5. Trajectory 字段草案

`trajectory.jsonl` 每行表示一次有效搜索 query 对应的训练候选上下文。一个源样本中如果发生多次搜索，会产生多条 trajectory record。

必填字段：

- `trace_id`：全局唯一轨迹记录 ID，建议格式为 `<sample_id>:search:<turn_idx>`。
- `sample_id`：源数据样本 ID。
- `question`：源问题文本。
- `gold_answers`：标准答案列表。
- `sub_query`：agent 实际发起的搜索 query。
- `recall_topn_docs`：recall retriever 返回的候选文档列表，通常 top-50。
- `ranked_docs`：当前排序后的候选文档列表；no-ranker 模式下等同 recall 顺序。
- `visible_docs`：实际暴露给 agent 的文档列表，通常 top-5。
- `source`：来源信息，包括原始 trace 路径、行号、搜索 turn 等。

建议字段：

- `origin_query`：初始问题；如果和 `question` 一致可以冗余保留，便于兼容 CAR 风格 trace。
- `turn_idx`：搜索发生在第几个 tool turn。
- `tool_call_id`：原始 tool call ID。
- `raw_trace_ref`：回溯到 `raw_traces.jsonl` 的引用，至少包含相对路径和行号。
- `final_answer`：agent 最终答案。
- `reward`：该样本最终 reward 或 answer score。
- `metrics`：该样本指标，例如 EM、F1、hit、latency、token 数。

已确认决策：

- `trajectory.jsonl` 不保留完整 `raw_agent_messages`。
- 完整 agent 消息只保留在 `raw_traces.jsonl`。
- `trajectory.jsonl` 通过 `raw_trace_ref`、`turn_idx`、`tool_call_id` 回溯原始消息。
- 后续如需人工排查，可从 canonical record 定位到 raw trace。

文档字段建议：

每个 doc 至少包含：

- `doc_id`
- `title`
- `text`
- `rank`

可选保留：

- `score`
- `retrieval_score`
- `rerank_score`
- `source`
- `metadata`

## 6. LLM Reranker Train Set 字段草案

`llm_reranker_train_set` 分为结构化主数据和 prompt 拼装后的训练数据。

### 6.1 `input_dataset/dataset.jsonl`

`input_dataset/dataset.jsonl` 每行表示一个 LLM reranker 候选训练样本，由一条 trajectory record 转换而来。它是后续不同 prompt/template 版本的共同输入，不直接绑定 SFT chat 格式。

必填字段：

- `query_id`：训练样本 ID，默认等于源 `trace_id`。
- `question`：源问题文本。
- `sub_query`：需要 reranker 排序的搜索 query。
- `candidate_docs`：候选文档列表，来自 `ranked_docs` 或 `recall_topn_docs`，按 `doc_id` 去重。
- `label_policy`：标签构造策略。当前默认值为空，表示暂不构造排序监督信号。
- `source_trace_id`：来源 trajectory record ID。

可选监督字段：

- `target_ranking`：目标 doc_id 排序列表。当前阶段不从 no-ranker trace 生成强监督排序。
- `positive_doc_ids`：正例 doc_id 列表。来源由策略配置决定，当前默认空。
- `weak_reference_order`：候选文档的原始参考顺序，例如 recall 顺序；不能当作强监督目标。

建议字段：

- `query_text`：实际给 reranker 的 query 文本；默认等于 `sub_query`。
- `metadata`：来源和审计信息，包括 `sample_id`、`reward`、`metrics`、`trajectory_version`、`candidate_top_n`、`positive_policy`。

候选文档字段建议和 trajectory doc 保持一致，至少包含：

- `doc_id`
- `title`
- `text`
- `rank`

LLM reranker 训练侧可额外增加：

- `input_rank`
- `is_positive`
- `label_score`

已确认决策：

- `input_dataset` 只保留结构化字段，不直接写 prompt 拼装结果。
- `positive_doc_ids` 是策略配置项，当前默认为空。
- 不把 no-ranker trace 中的 `target_ranking` 当强监督排序。
- 排序训练数据的信号构造后续作为独立重点策略设计。

### 6.2 `train_dataset/<train_dataset_version>/dataset.jsonl`

`train_dataset` 是从 `input_dataset` 经过 prompt/template 拼装后得到的可训练数据。

已确认第一版首先支持 GRPO 格式。GRPO 数据集不需要 `target_text` 作为监督答案；模型会基于 prompt rollout 生成排序结果，后续由 reward 策略打分。当前 ground-truth 暂时为空，reward 策略也先记录为 `none`。

第一版 `train_dataset` 的 reranker prompt 明确复用 CoSearch 里同等链路的 prompt，而不是新设计 JSON 排序 prompt。对应源代码位置如下：

- prompt 模板定义：`CoSearch/verl/verl/tools/utils/prompts.py` 中的 `RERANK_PROMPT_WITH_INITIAL_QUERY`。
- 在线工具调用：`CoSearch/verl/verl/tools/co_search_tool.py` 的 `_build_reranker_prompt()` 会用该模板构造 reranker 输入。
- 轨迹训练数据构造：`CoSearch/verl/verl/experimental/trajectory_store/trajectory_dataset.py` 的 `_attach_prompt()` 也用同一个模板构造 `raw_prompt`。
- 候选文档格式化：`CoSearch/verl/verl/tools/utils/search.py` 的 `format_tool_response_with_docid_map()` 会把候选文档格式化为 `[1] ...`、`[2] ...` 这种 1-based index，并返回 `docid_map`。

这意味着 AIR v1 第一版 `train_dataset` 要遵守 CoSearch 的两个关键约定：

- prompt 只包含一个 user message：`[{"role": "user", "content": prompt_text}]`，其中 `prompt_text` 由 `RERANK_PROMPT_WITH_INITIAL_QUERY.format(...)` 得到。
- 模型输出不是 JSON，也不是直接输出真实 `doc_id`；模型输出 `<rerank>[27] > [233] > ...</rerank>` 这种候选序号排序，后处理再通过 `docid_map` 映射回真实文档。

这里用候选序号而不是直接用 `doc_id`，主要是为了让 prompt 更短、更稳定。真实 `doc_id` 可能很长，也可能包含模型不容易稳定复制的字符；候选序号固定是 `[1]` 到 `[N]`，更适合做格式约束和输出校验。因此 `train_dataset` 必须同时保存候选序号到真实文档的映射，否则模型输出无法还原成可评估的文档排序。

CoSearch 同款 prompt 的语义如下：

```text
You are a professional document reranker specialized in multi-step search and reasoning tasks.

You will be given:
- An Initial Query: the user's ultimate question and final goal.
- A Current Sub-Query: a focused query generated to retrieve information for the current step.
- A list of {N} candidate passages.

Your goal is:
Rank EXACTLY {M} passages that are MOST USEFUL at this step.

Primary principle:
Ranking is based on the Current Sub-Query,
but the Sub-Query MUST be interpreted and constrained by the Initial Query.

In particular:
- Prefer passages that can directly help answer the Initial Query.
- If none can directly answer it, prefer passages that best match the Sub-Query
  WHILE staying strictly within the scope and intent of the Initial Query.

# === STRICT OUTPUT FORMAT (must match EXACTLY) ===
<reason> ... </reason>
<rerank> ... </rerank>

Anything outside these two tags or in a different order is invalid.

# === BLOCK 1: <reason> ... </reason>
Explain your ranking decisions clearly and concretely.

Follow these steps:
1. Identify what the Initial Query is ultimately asking.
2. Identify what specific information the Current Sub-Query is seeking.
3. Explain how the selected passages either directly help answer the Initial Query,
   or provide the most relevant information for the Sub-Query without drifting away
   from the Initial Query.
4. If a passage matches the Sub-Query but is off-topic or irrelevant to the Initial Query,
   explain why it is ranked lower.
5. When multiple passages are similar, break ties using factuality, entity specificity,
   and usefulness for later steps.

Write 5-8 short sentences.
Do NOT include passage indices here.

# === BLOCK 2: <rerank> ... </rerank>
Output EXACTLY {M} distinct indices from [1] to [{N}], chained with " > ".
The first passage is the most useful, and usefulness decreases left to right.

Example for M=5:
<rerank>[27] > [233] > [105] > [729] > [688]</rerank>

# === INPUT BEGINS ===
Initial Query:
{initial_query}

Current Sub-Query:
{sub_query}

Passages ({N} total):
{passages_block}
# === INPUT ENDS ===
```

上面为了文档可读性做了轻微压缩；实现时必须直接引用或等价复制 CoSearch 源码中的 `RERANK_PROMPT_WITH_INITIAL_QUERY`，不能根据这段说明重新手写一个语义相近但格式不同的模板。

推荐字段：

- `sample_id`：训练样本 ID，默认等于 `query_id`。
- `source_query_id`：来源 input dataset 的 `query_id`。
- `data_source`：固定建议为 `agentic_iter_rag.llm_reranker.grpo`。
- `ability`：固定建议为 `llm_reranker`。
- `prompt`：已拼装的 VERL chat prompt。
- `reward_model`：保留 VERL 兼容结构，当前 `ground_truth.target` 为 `null`。
- `prompt_template_version`：prompt 模板版本。
- `formatter`：格式化器名称。
- `target_text`：GRPO 第一版不使用；如为了 schema 统一保留，必须显式为 `null`。
- `extra_info`：包含 `input_dataset_version`、`source_trace_id`、`label_policy`、`candidate_index_to_doc_id`、`candidate_doc_ids` 等来源信息。

推荐单条样例形态：

```json
{
  "data_source": "agentic_iter_rag.llm_reranker.grpo",
  "ability": "llm_reranker",
  "prompt": [
    {
      "role": "user",
      "content": "You are a professional document reranker specialized in multi-step search and reasoning tasks.\n\nInitial Query:\n...\n\nCurrent Sub-Query:\n...\n\nPassages (50 total):\n[1] Title: ...\n...\n\n# === INPUT ENDS ==="
    }
  ],
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "target": null
    }
  },
  "extra_info": {
    "source_query_id": "...",
    "source_trace_id": "...",
    "input_dataset_version": "...",
    "trajectory_version": "...",
    "candidate_doc_ids": ["doc1", "doc2"],
    "candidate_index_to_doc_id": {
      "1": "doc1",
      "2": "doc2"
    },
    "ground_truth_policy": "empty",
    "reward_policy": "none",
    "output_schema": "cosearch_rerank_tags"
  }
}
```

prompt 输出约束建议：

- 必须先输出 `<reason> ... </reason>`，再输出 `<rerank> ... </rerank>`。
- `<reason>` 中解释排序依据，但不能包含 passage index。
- `<rerank>` 中必须输出 exactly `M` 个不同候选序号，序号范围是 `[1]` 到 `[N]`。
- `<rerank>` 中序号用 ` > ` 连接，不输出逗号、分数或额外文本。
- 模型输出的候选序号必须通过 `candidate_index_to_doc_id` 或 `docid_map` 映射回真实 `doc_id`。
- 不允许编造候选序号，不允许重复候选序号。

当前不做的事：

- 不写真实 `target_text`。
- 不写非空 ground-truth。
- 不把 no-ranker recall 顺序当成训练标签。
- 不在这个阶段实现真实排序 reward。
- 不在第一版 train_dataset 中引入 JSON 输出格式；如后续需要 JSON 排序 prompt，应作为新的 `prompt_template_version` 单独新增。

后续待设计：

- 真实排序 reward 的策略配置、生成链路和质量校验。
- 是否需要在 GRPO 之外同时支持 SFT、DPO、listwise rerank 等多种训练格式。
- 同一个 `input_dataset` 下多个 `train_dataset` 版本之间的命名、复用和淘汰规则。

## 7. Manifest 字段草案

### 7.1 `trajectory/manifest.json`

建议记录：

- `dataset_type`: `trajectory`
- `version`
- `version_dir`
- `source_data_files`
- `agent_checkpoint`
- `infer_runtime_config`
- `infer_budget_config`
- `model_config`
- `rollout_config`
- `raw_traces_jsonl`
- `trajectory_jsonl`
- `example_json`
- `metrics_jsonl`
- `summary_json`
- `sample_count`
- `trace_record_count`
- `pipeline_run_id`
- `final_config_yaml`
- `created_at`
- `config_hash`

### 7.2 `llm_reranker_train_set/manifest.json`

建议记录：

- `dataset_type`: `llm_reranker_train_set`
- `version`
- `version_dir`
- `source_trajectory_version`
- `source_trajectory_manifest`
- `label_policy`
- `input_dataset_schema_version`
- `input_dataset_builder_policy`
- `input_dataset_build_hash`
- `candidate_top_n`
- `positive_policy`
- `input_dataset_jsonl`
- `input_dataset_parquet`
- `input_dataset_example_json`
- `train_dataset_versions`
- `example_json`
- `sample_count`
- `dedupe_policy`
- `pipeline_run_id`
- `final_config_yaml`
- `created_at`
- `config_hash`

### 7.3 `train_dataset/<train_dataset_version>/manifest.json`

建议记录：

- `dataset_type`: `llm_reranker_train_dataset`
- `version`
- `version_dir`
- `source_input_dataset_manifest`
- `prompt_template_version`
- `formatter`
- `format`
- `ground_truth_policy`
- `reward_policy`
- `output_schema`
- `dataset_jsonl`
- `dataset_parquet`
- `example_json`
- `sample_count`
- `created_at`
- `config_hash`

## 8. 已确认决策与待讨论问题

已确认：

- `trajectory.jsonl` 采用 raw 引用方案，不内嵌完整 `raw_agent_messages`。
- `llm_reranker_train_set` 内部拆分为 `input_dataset` 和 `train_dataset`。
- `input_dataset` 是结构化主数据。
- `train_dataset` 是 prompt 拼装后的训练数据，并且需要内部版本区分。
- `positive_doc_ids` 是策略配置项，当前默认空。
- no-ranker trace 不提供强监督排序目标。

后续实现前需要确认：

- 排序监督信号的策略配置、生成链路和质量校验。
- 是否需要在同一 `input_dataset` 下同时保留多种训练范式的数据，例如 SFT、DPO、listwise。

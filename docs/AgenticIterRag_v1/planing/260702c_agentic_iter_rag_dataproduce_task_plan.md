# AgenticIterRag v1 Data Produce Task 实施计划

更新日期：2026-07-02

## 1. 目标

本文只讨论 `data produce task` 的落地计划和验收方式，不再重复展开中间数据目录、版本命名和字段设计。

数据管理与字段设计以当前 design 文档为准：

- [260702b_agentic_iter_rag_intermediate_data_design.md](./260702b_agentic_iter_rag_intermediate_data_design.md)

本 task 的原始目的很明确：

1. 使用一个已经存在的 agent checkpoint。
2. 对指定数据进行推理，默认使用 AIR source 下的 ablation 训练集 parquet 稳定路径。
3. 产出 agent 搜索轨迹 `trajectory`。
4. 从 `trajectory` 产出 LLM reranker 的候选训练数据集合 `llm_reranker_train_set`。
5. 暂时跳过 agent 训练、LLM reranker 训练和最终 infer matrix。

正式 task 入口为：

```text
tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh
```

## 2. 非目标

第一版 data produce task 不做以下事情：

- 不训练 agent。
- 不训练 LLM reranker。
- 不跑完整 infer matrix。
- 不设计真实排序 reward。
- 不把 no-ranker trace 的原始顺序当作强监督排序目标。
- 不在 shell 中维护业务配置，例如数据路径、top-N/top-M、batch size、模型路径、端口和 prompt 参数。

## 3. 目标流程

`run_260702a_AIR_v1_dataproduce.sh` 仍调用统一 pipeline launcher：

```text
scripts/agenticIterRag_v1/01_pipeline_launcher.sh
```

但它选择一个 data produce 专用 overlay，使 pipeline 实际只运行：

```text
generate_traces
build_reranker_dataset
```

其中 `build_reranker_dataset` 在设计上是父 stage，内部包含两个子 stage：

```text
build_input_dataset
build_train_dataset
```

这两个子 stage 的配置、开关和产物规则见 design 文档的 `build_reranker_dataset Stage 设计` 章节。

## 4. 配置改造计划

### 4.1 新增 data produce overlay

新增：

```text
tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml
```

职责：

- 设置实验名，例如 `agentic_iter_rag_v1_dataproduce_260702a`。
- 跳过 `train_agent`、`train_llm_reranker`、`infer_matrix`。
- 启用 `generate_traces`。
- 启用 `build_reranker_dataset`。
- 在 `build_reranker_dataset` 内默认同时启用：
  - `build_input_dataset`
  - `build_train_dataset`
- 设置 data produce 默认数据为 AIR source 下的 ablation 训练集 parquet 稳定路径。
- 设置 agent checkpoint 字段为可覆盖字段，默认使用 task 当前指定的已训 agent。

### 4.2 调整 data produce task

`run_260702a_AIR_v1_dataproduce.sh` 保持现有 task 风格：

- 显式列出 `DATA_CONFIG`、`PIPELINE_CONFIG`、`RESOURCE_CONFIG`、`INFER_RUNTIME_CONFIG`、`INFER_BUDGET_CONFIG`、`RERANKER_TRAINING_CONFIG`、`MODEL_CONFIG`、`ROLLOUT_CONFIG`。
- 只选择 YAML 配置组和 overlay。
- 允许少量 CLI dotlist 用于临时 dry-run、小样本 smoke、断点或 checkpoint 覆盖。
- 不在 shell 中新增业务参数。

data produce task 应使用：

```bash
--OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml
```

而不是复用 full offline two-stage overlay。

### 4.3 配置职责调整

后续实现时，`build_reranker_dataset` 的数据生产参数应从 `reranker_training` 移到：

```text
pipeline.stage_configs.build_reranker_dataset.sub_stages
```

原因：

- `candidate_top_n`、`candidate_source`、`label_policy` 是 input_dataset 构造参数。
- `prompt_template_version`、`formatter`、`ground_truth_policy`、`reward_policy` 是 train_dataset 构造参数。
- `reranker_training` 应只描述后续训练器如何消费数据和训练模型。

## 5. 代码实现计划

### 5.1 Pipeline 配置

更新：

```text
AgenticIterRag/config/pipeline/offline_two_stage.yaml
```

新增或调整内容：

- `build_reranker_dataset.sub_stage_order`
- `build_reranker_dataset.sub_stages.build_input_dataset`
- `build_reranker_dataset.sub_stages.build_train_dataset`
- 两个子 stage 独立的 `enabled`、输入、输出和配置字段。

### 5.2 Pipeline runner

更新：

```text
scripts/agenticIterRag_v1/assets/run_pipeline.py
```

需要支持：

- 父 stage `build_reranker_dataset` 仍是 pipeline 顶层 stage。
- dry-run 时写出父 stage manifest，并在 manifest 中展示两个子 stage 的启用状态、配置摘要和预期产物路径。
- 真实执行时按顺序运行 `build_input_dataset` 和 `build_train_dataset`。
- 当 `build_input_dataset.enabled=false` 且 `build_train_dataset.enabled=true` 时，必须要求已有 `input_dataset_manifest`。
- 当两个子 stage 都 disabled 时，默认报错。

### 5.3 数据构造代码

更新或拆分：

```text
AgenticIterRag/agentic_iter_rag/reranker_dataset/build_dataset.py
```

建议拆出两个内部函数：

- `build_input_dataset(...)`
- `build_train_dataset(...)`

第一版可仍由同一个 CLI 入口调用，但 manifest 和日志必须区分两个子 stage。

### 5.4 Prompt 渲染

第一版 `train_dataset` 的 LLM-as-ranker prompt 使用 AIR 内置模板：

```text
AgenticIterRag/agentic_iter_rag/llm_reranker/format.py::AIR_RERANK_PROMPT_WITH_INITIAL_QUERY
```

实现时必须保持输出约束：

- `<reason> ... </reason>`
- `<rerank> ... </rerank>`
- `<rerank>` 中输出候选序号 `[1] > [2]`，再通过 `candidate_index_to_doc_id` 映射到真实 `doc_id`。

## 6. 验收计划

### 6.1 Dry-run 验收

命令：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh --dry-run
```

必须满足：

- 生成 `pipeline.final_config.yaml`。
- 生成 `pipeline.final_config.json`。
- 生成 `pipeline.env`。
- 生成 `pipeline.args.txt`。
- 生成 `pipeline.manifest.json`。
- 生成 `execution_plan.yaml`。
- `execution_plan.yaml` 中只选择 data produce 需要的 stage：
  - `generate_traces`
  - `build_reranker_dataset`
- `train_agent` 不执行。
- `train_llm_reranker` 不执行。
- `infer_matrix` 不执行。
- `build_reranker_dataset` manifest 中能看到两个子 stage 的配置：
  - `build_input_dataset`
  - `build_train_dataset`

### 6.2 配置治理验收

必须验证 shell-only 业务配置会失败。

负向命令示例：

```bash
DATA_PATH=/tmp/x bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh --dry-run
```

预期：

- compiler 拒绝运行。
- 错误信息能说明存在未登记的 shell-only 业务配置。

同时检查：

- agent checkpoint 不是 shell 变量独立维护，而是通过 YAML 字段或 CLI dotlist 覆盖进入 final config。
- 数据路径、top-N/top-M、prompt 模板、batch size 不在 task shell 中单独定义。

### 6.3 小样本真实生产验收

用 5 到 10 条样本做 smoke。

命令形式：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh \
  data.trace_max_samples=10
```

通过条件：

- 成功生成 `trajectory` 数据集目录。
- 成功生成 `llm_reranker_train_set` 数据集目录。
- `trajectory/manifest.json` 存在。
- `trajectory/example.json` 存在。
- `trajectory/trajectory.jsonl` 存在且非空。
- `llm_reranker_train_set/<version>/input_dataset/example.json` 存在。
- `llm_reranker_train_set/<version>/input_dataset/dataset.jsonl` 存在且非空。
- `llm_reranker_train_set/<version>/train_dataset/<train_dataset_version>/example.json` 存在。
- `llm_reranker_train_set/<version>/train_dataset/<train_dataset_version>/dataset.jsonl` 存在且非空。

如果小样本中 agent 没有产生任何 search query，默认视为失败。除非显式设置调试字段，例如 `allow_empty_trace=true`，否则 data produce task 的目标没有达成。

### 6.4 Trajectory 字段验收

抽查 `trajectory/example.json` 和 `trajectory/trajectory.jsonl` 前几行。

每条 trajectory record 必须包含：

- `trace_id`
- `sample_id`
- `question`
- `gold_answers`
- `sub_query`
- `recall_topn_docs`
- `ranked_docs`
- `visible_docs`
- `raw_trace_ref`

必须满足：

- `sub_query` 非空。
- `recall_topn_docs` 非空。
- 文档字段至少能解析出 `doc_id` 和 `text`。
- `raw_trace_ref` 能定位到 `raw_traces.jsonl`。

### 6.5 Input Dataset 字段验收

抽查 `input_dataset/example.json` 和 `input_dataset/dataset.jsonl` 前几行。

每条 input sample 必须包含：

- `query_id`
- `question`
- `sub_query`
- `candidate_docs`
- `label_policy`
- `source_trace_id`
- `metadata`

必须满足：

- `candidate_docs` 非空。
- 单条样本内 `doc_id` 不重复。
- `candidate_docs` 数量不超过 `candidate_top_n`。
- 当前第一版 `label_policy` 可以为空。
- 当前第一版 `positive_doc_ids` 可以为空。
- 当前第一版 `target_ranking` 不从 no-ranker trace 构造强监督排序。

### 6.6 Train Dataset 字段验收

抽查 `train_dataset/<train_dataset_version>/example.json` 和 `dataset.jsonl` 前几行。

每条 train sample 必须包含：

- `data_source`
- `ability`
- `prompt`
- `reward_model`
- `extra_info`

必须满足：

- `prompt` 是 VERL chat message 格式。
- 第一版 prompt 使用 AgenticIterRag 的 `RERANK_PROMPT_WITH_INITIAL_QUERY` 语义。
- prompt 中包含 `Initial Query`。
- prompt 中包含 `Current Sub-Query`。
- prompt 中包含 `<reason>` 和 `<rerank>` 输出格式要求。
- prompt 中候选文档使用 `[1]`、`[2]` 这种候选序号。
- `reward_model.ground_truth.target` 显式为 `null`。
- `extra_info.candidate_index_to_doc_id` 存在。
- `target_text` 不写，或显式为 `null`。

### 6.7 Manifest 验收

必须能从 manifest 完整追溯：

- 使用的 agent checkpoint。
- 使用的源数据文件。
- trajectory 版本。
- input_dataset 版本。
- train_dataset 版本。
- top-N/top-M。
- prompt 模板版本。
- formatter。
- ground-truth 策略。
- reward 策略。
- final config 路径。
- 生成时间。
- 样本数。
- 配置 hash。

父 stage manifest 必须汇总两个子 stage manifest：

- `build_input_dataset.manifest.json`
- `build_train_dataset.manifest.json`

### 6.8 复用与重跑验收

必须验证两个典型场景：

1. 只生产 `input_dataset`：

```text
build_input_dataset.enabled=true
build_train_dataset.enabled=false
```

预期：

- 生成 `input_dataset`。
- 不生成新的 `train_dataset`。
- manifest 中明确记录 `build_train_dataset` 被跳过。

2. 复用已有 `input_dataset` 只生产新 `train_dataset`：

```text
build_input_dataset.enabled=false
build_train_dataset.enabled=true
input_dataset_manifest=<existing_manifest>
```

预期：

- 不重新读取 trajectory。
- 不重写 `input_dataset`。
- 在同一个 `llm_reranker_train_set` 下新增一个 `train_dataset/<train_dataset_version>`。
- 新 train_dataset manifest 能指向原 input_dataset manifest。

### 6.9 版本冲突验收

必须验证：

- 手动指定版本且目录已存在、`overwrite=false` 时失败。
- 手动指定版本且目录已存在、`overwrite=true` 时允许重写，并在 manifest 记录覆盖行为。
- 自动版本冲突时追加 `_r01`、`_r02` 或生成新的唯一版本名。

### 6.10 最终通过标准

data produce task 第一版验收通过需要同时满足：

- dry-run 通过。
- shell-only 配置负向测试通过。
- 5 到 10 条小样本真实生产通过。
- trajectory、input_dataset、train_dataset 三类 example 文件存在且字段可读。
- 所有 manifest 可追溯到源 agent checkpoint 和源数据。
- 两个子 stage 的开关组合至少验证 `全开`、`只 input`、`只 train` 三种。
- task shell 中没有新增业务参数。

## 7. 交付物清单

计划实现完成后，应交付：

- `tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh`
- `tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml`
- 更新后的 `AgenticIterRag/config/pipeline/offline_two_stage.yaml`
- 更新后的 pipeline runner
- 更新后的 reranker dataset builder
- smoke run 的 `pipeline.manifest.json`
- smoke run 的 trajectory 数据集目录
- smoke run 的 llm reranker train set 数据集目录
- 简短验收记录，记录命令、输出路径和是否通过

## 8. 当前实现验收记录

实现完成后需要至少记录以下命令的结果：

```bash
python3 -m compileall AgenticIterRag/agentic_iter_rag scripts/agenticIterRag_v1/assets
```

通过标准：新增 Python 文件无语法错误。

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh --dry-run
```

通过标准：

- `execution_plan.yaml` 只包含 `generate_traces` 和 `build_reranker_dataset`。
- `generate_traces/manifest.json` 中能看到 trajectory 预期路径、AIR infer command 和 runtime env 摘要。
- `build_reranker_dataset/manifest.json` 中能看到 `build_input_dataset` 与 `build_train_dataset` 两个子阶段 manifest 路径。

```bash
DATA_PATH=/tmp/x bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh --dry-run
```

通过标准：compiler 必须失败，并指出 `DATA_PATH` 属于禁止的 shell-only 业务配置。

可选的本地转换 smoke 可以通过已有 raw trace 触发，不启动 vLLM：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh \
  pipeline.stage_configs.generate_traces.inputs.existing_raw_trace_jsonl=<raw_traces.jsonl> \
  main_run.data_artifacts.trajectory.version=<smoke_trajectory_version> \
  main_run.data_artifacts.trajectory.overwrite=true \
  pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.version=<smoke_train_set_version> \
  pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.overwrite=true \
  pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.version=<smoke_train_dataset_version> \
  pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.overwrite=true
```

通过标准：

- `data/AgenticIterRag/trajectory/<version>/example.json` 存在。
- `data/AgenticIterRag/llm_reranker_train_set/<version>/input_dataset/example.json` 存在。
- `data/AgenticIterRag/llm_reranker_train_set/<version>/train_dataset/<train_dataset_version>/example.json` 存在。
- train dataset 的 prompt 包含 `Initial Query`、`Current Sub-Query`、`<reason>` 和 `<rerank>`。
- 当前环境如果缺少 pandas/pyarrow/fastparquet，Parquet 可以不生成，但 manifest 必须把 `dataset_parquet` 记录为 `null`。

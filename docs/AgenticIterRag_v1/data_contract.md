# AgenticIterRag v1 数据契约

## 源数据

AgenticIterRag v1 当前通过 AIR source 稳定路径读取 ablation parquet 数据文件：

- 训练集：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/source/co_search_ablation.train.parquet`
- 推理集：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/source/co_search_ablation.infer.parquet`

`data/AgenticIterRag/source/` 下的文件可以是软链，用于把历史磁盘数据目录名和 AIR 当前运行配置隔离开。

## Trace Record

标准 trace record 包含：

- `trace_id`
- `sample_id`
- `question`
- `gold_answers`
- `sub_query`
- `recall_topn_docs`
- `ranked_docs`
- `visible_docs`
- `final_answer`
- `reward`
- `metrics`
- `source`
- `raw_trace_ref`

`trajectory/example.json` 必须存在，内容是 `trajectory.jsonl` 中的一条样例记录。

## Reranker Sample

### Input Dataset

标准 reranker sample 包含：

- `query_id`
- `question`
- `sub_query`
- `candidate_docs`
- `label_policy`
- `target_ranking`
- `positive_doc_ids`
- `source_trace_id`
- `metadata`

第一版 `label_policy`、`positive_doc_ids` 和 `target_ranking` 可以为空，因为排序监督信号后续单独建设。

### Train Dataset

第一版 train dataset 使用 GRPO 格式，包含：

- `sample_id`
- `source_query_id`
- `data_source`
- `ability`
- `prompt`
- `reward_model`
- `prompt_template_version`
- `formatter`
- `target_text`
- `extra_info`

`prompt` 使用 VERL chat message 格式，并复用 AgenticIterRag 的 `<reason> ... </reason>` 与 `<rerank> ... </rerank>` reranker prompt 语义。

`reward_model.ground_truth.target` 第一版显式为 `null`。

dataset builder 会优先写 JSONL；当 pandas/pyarrow/fastparquet 等本地依赖可用时，也会同时写 Parquet。依赖不可用时，manifest 中的 `dataset_parquet` 为 `null`。

`input_dataset/example.json` 和 `train_dataset/<version>/example.json` 必须存在。

## Manifest

pipeline 顶层 manifest 记录本次 pipeline 运行结果。

execution plan 记录本次实际选择的 stage、断点参数和每个 stage 的 manifest 路径。

每个 stage manifest 记录该 stage 的输入、输出和执行状态。

`build_reranker_dataset` 父 stage manifest 会汇总两个内部子 stage manifest：

- `stages/build_reranker_dataset/build_input_dataset/manifest.json`
- `stages/build_reranker_dataset/build_train_dataset/manifest.json`

reranker train set 顶层 manifest 记录：

- 源 trajectory manifest
- input_dataset manifest
- train_dataset manifest
- 输出 JSONL 路径
- 输出 Parquet 路径或 `null`
- label policy
- 候选 top-N
- prompt 模板版本
- formatter
- ground-truth 策略
- reward 策略
- 样本数量

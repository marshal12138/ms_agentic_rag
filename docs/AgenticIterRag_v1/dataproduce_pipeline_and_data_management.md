# AgenticIterRag v1 Data Produce Pipeline 与数据格式管理

更新日期：2026-07-03

本文是当前实现的操作规范，覆盖 AgenticIterRag v1 的数据生产 pipeline、长期数据目录、数据格式、版本命名、README/manifest 管理和清理策略。

历史 planning 文档仍可作为设计背景参考，但不作为当前执行规范：

- `docs/planning/260702c_agentic_iter_rag_dataproduce_task_plan.md`
- `docs/planning/260702b_agentic_iter_rag_intermediate_data_design.md`

## 1. Pipeline 定位

当前 data produce 入口是：

```text
tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh
```

它调用统一 launcher：

```text
scripts/agenticIterRag_v1/01_pipeline_launcher.sh
```

当前 data produce 只执行两个 stage：

```text
generate_traces
build_reranker_dataset
```

其中 `build_reranker_dataset` 是父 stage，内部包含两个子 stage：

```text
build_input_dataset
build_train_dataset
```

本流程生产两类可复用长期数据：

- `trajectory`：已有 agent checkpoint 对指定源数据推理后的搜索轨迹。
- `llm_reranker_train_set`：由 `trajectory` 构造出的 LLM reranker 候选训练数据集合。

本流程不做以下事情：

- 不训练 agent。
- 不训练 LLM reranker。
- 不跑 infer matrix 评测。
- 不构造真实 ranking reward 训练闭环。
- 不把 no-ranker trace 的原始顺序当作强监督排序目标。

## 2. 入口脚本与配置来源

当前入口脚本选择的配置组是：

```text
main_run_config        agentic_iter_rag_main
DATA_CONFIG            co_search_ablation
PIPELINE_CONFIG        offline_two_stage
RESOURCE_CONFIG        local_8gpu_0_7
INFER_RUNTIME_CONFIG   agentic_iter_rag_vllm
INFER_BUDGET_CONFIG    air_aligned_budget
RERANKER_TRAINING_CONFIG llm_reranker_base
MODEL_CONFIG           qwen3_4b
ROLLOUT_CONFIG         air_async_qwen3_4b
OVERLAY_YAML           tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml
```

当前脚本内的关键 CLI override 是：

```text
data.trace_max_samples=-1
infer_budget.infer_batch_size=96
infer_budget.vllm.gpu_memory_utilization=0.8
```

配置合并优先级是：

```text
base YAML < selected config group < overlay YAML < CLI override
```

`dataproduce_overlay.yaml` 负责 data produce 任务级差异，包括：

- 实验名。
- `generate_traces -> build_reranker_dataset` 的执行范围。
- 已训练 agent checkpoint。
- `trajectory` 和 `llm_reranker_train_set` 的版本/覆盖策略。
- `build_input_dataset` 和 `build_train_dataset` 的生产策略。
- agent/retriever HTTP retry、fail-on-error 和 keep-alive 策略。

注意：保留下来的当前全量样例是历史运行产物，它的 `final_config.yaml` 中 `infer_budget.infer_batch_size=64`、`infer_budget.vllm.gpu_memory_utilization=0.6`。因此样例数据只能说明当次生产事实，不能反推当前入口脚本默认值。

## 3. 资源与服务拓扑

资源配置文件：

```text
AgenticIterRag/config/resource/local_8gpu_0_7.yaml
```

该文件采用 stage-level resource placement，而不是全局 `agent`/`recall` 角色绑卡。原因是 AgenticIterRag 是多阶段 pipeline，不同 stage 需要不同资源布局。

当前 `generate_traces` 的默认资源意图是：

```text
agent_vllm:
  gpu_ids: [0, 1, 2, 3]
  tensor_parallel_size: 4
  port: 8140

recall:
  gpu_ids: [6, 7]
  port: 8130
  backend_base_port: 8131
  retrieval_service_url: http://127.0.0.1:8130/retrieve
```

这里的 `recall` 可以启动多个 retriever backend，并通过统一 proxy 地址暴露给 AIR infer backend。preflight 只需要验证统一 proxy 地址；如果 preflight 失败，则整个数据生产失败。

这种设计避免了以下问题：

- 单一全局 GPU 配置无法表达多阶段 pipeline 的资源差异。
- `generate_traces`、`train_llm_reranker`、`infer_matrix` 的服务组合不同，不能共享同一套语义。
- 同一个 GPU id 在不同 stage 可以合法复用，但在同一 stage 内必须按 service 明确分配。

## 4. 长期数据目录

长期可复用数据统一写入：

```text
data/AgenticIterRag/
```

主要子目录和文件：

```text
data/AgenticIterRag/
  source/
  trajectory/
  llm_reranker_train_set/
  .trash/
  cleanup_keep_full_dataset.sh
  README.md
```

目录职责：

- `source/`：源数据或稳定软链，不由清理脚本删除。
- `trajectory/`：AIR 推理轨迹数据版本。
- `llm_reranker_train_set/`：由 trajectory 构造出的 reranker input/train 数据版本。
- `.trash/`：清理脚本移动旧数据的位置，不做永久删除。
- `cleanup_keep_full_dataset.sh`：保留全量有效数据、清理空目录和非全量产物的工具。
- `README.md`：数据根目录的人类可读说明。

当前保留下来的全量样例：

```text
trajectory/260703f_AIR_v1_traj_co_search_ablation.train_global_step_79/
llm_reranker_train_set/260703f_AIR_v1_traj_co_search_ablation.train_global_step_79__input_recall50_emptylabel/
```

## 5. 命名与覆盖规则

自动命名模式下，数据版本名不再包含 `hhmmss` 时间戳，也不再包含末尾短 hash。

日期前缀支持自动递增版本：

```text
260703_xxx
260703b_xxx
260703c_xxx
```

覆盖语义：

- `overwrite=false`：如果目标名已存在，自动递增日期前缀版本，例如从 `260703_xxx` 变成 `260703b_xxx`。
- `overwrite=true`：如果目标名已存在，完整覆盖原目录。

该规则适用于三类数据：

- `trajectory`
- `llm_reranker_train_set/input_dataset`
- `llm_reranker_train_set/train_dataset`

`trajectory` 的版本主要由日期、任务名、数据配置和 checkpoint 标识组成。`llm_reranker_train_set` 的外层版本默认从 source trajectory 派生，并追加 input dataset 策略摘要。`train_dataset` 在外层 reranker set 内继续区分 prompt/template/formatter/reward 版本。

## 6. Trajectory 数据格式

一个 trajectory 版本目录通常包含：

```text
raw_traces.jsonl
trajectory.jsonl
metrics.jsonl
summary.json
manifest.json
final_config.yaml
example.json
README.md
```

文件职责：

- `raw_traces.jsonl`：agent 原始推理轨迹，每行对应一个源样本的 AIR infer 输出。
- `trajectory.jsonl`：标准化后的 canonical trajectory 主数据，通常一条源样本会拆出多条 search-query 级记录。
- `metrics.jsonl`：每个源样本的执行状态、轮数、耗时、EM/F1 和检索统计。
- `summary.json`：聚合统计。
- `manifest.json`：数据集类型、版本、来源、计数、配置 hash 和产物索引。
- `final_config.yaml`：本次生产的最终合并配置快照。
- `example.json`：一条样例记录。
- `README.md`：人类可读的数据说明、字段概览和统计。

当前 canonical `trajectory.jsonl` 的核心字段包括：

```text
trace_id
sample_id
question
sub_query
gold_answers
final_answer
recall_topn_docs
ranked_docs
visible_docs
reward
metrics
source
raw_trace_ref
```

`metrics.jsonl` 中的状态用于判断本次数据生产是否有失败样本。`failed` 和 `error` 是不可接受的最终错误状态；`max_turns`、`no_valid_answer` 等状态需要按任务语义单独解释，不等同于传输或服务失败。

## 7. Reranker 数据格式

一个 `llm_reranker_train_set` 外层目录通常包含：

```text
manifest.json
source_trajectory.manifest.json
final_config.yaml
example.json
README.md
input_dataset/
train_dataset/<train_dataset_version>/
```

`input_dataset/` 通常包含：

```text
dataset.jsonl
dataset.parquet
manifest.json
example.json
README.md
```

`train_dataset/<train_dataset_version>/` 通常包含：

```text
dataset.jsonl
dataset.parquet
manifest.json
example.json
README.md
```

`input_dataset` 是结构化候选数据，不绑定具体 prompt。核心字段包括：

```text
query_id
question
sub_query
candidate_docs
label_policy
target_ranking
positive_doc_ids
source_trace_id
metadata
```

当前 input dataset 生产策略：

```text
schema_version: v1
builder_policy: recall_topn_to_candidates
candidate_source: recall_topn_docs
candidate_top_n: 50
dedupe_policy: keep_first
label_policy: null
positive_policy: null
target_ranking_policy: none
```

`train_dataset` 是已渲染 prompt 的训练数据。核心字段包括：

```text
sample_id
source_query_id
data_source
ability
prompt
reward_model
prompt_template_version
formatter
target_text
extra_info
```

当前 train dataset 生产策略：

```text
format: grpo
prompt_template_version: air_rerank_tags_v1
formatter: verl_chat
ground_truth_policy: empty
reward_policy: none
output_schema: air_rerank_tags
reranker_top_m: 5
max_doc_chars: 2000
```

JSONL 是主产物，Parquet 是辅助镜像。Parquet 写入失败时，不应影响 JSONL 主产物保留，但 warning 必须能在运行日志或 README 中追溯。

## 8. README 与 Manifest 管理

每个长期数据目录都应该有 `README.md` 和 `manifest.json`。

二者职责不同：

- `manifest.json` 给程序读取，用于数据追溯、清理判断和下游消费。
- `README.md` 给人读取，用于快速确认这个目录里实际有多少数据、有哪些字段、来自哪里、是否有异常。

README 至少应包含：

- 数据集名称和类型。
- 版本名。
- 生产时间。
- 来源数据或上游 manifest。
- 关键配置摘要。
- 文件清单。
- 总行数。
- 字段列表。
- 状态分布或数据源分布。
- 示例文件位置。
- 已知 warning 或异常。

manifest 至少应覆盖：

- `dataset_type`
- `version`
- `created_at`
- source manifest 或 source data files
- sample/record counts
- production config summary 或 config hash
- artifact paths

`final_config.yaml` 是生产现场的完整配置快照。后续排查“为什么这批数据这样生成”时，应优先查看该文件，而不是只看当前代码默认值。

## 9. 清理与保留策略

清理脚本位置：

```text
data/AgenticIterRag/cleanup_keep_full_dataset.sh
```

基本原则：

- `source/` 永远不清理。
- 空目录直接移动到 `.trash/`。
- 已知数据目录优先依据 `manifest.json` 和 `final_config.yaml` 判断。
- 不再只依赖固定数量，例如 `5100`。
- 非空目录如果缺少关键元信息，默认保守保留。
- 清理动作移动到 `.trash/cleanup_YYYYMMDD_HHMMSS/`，不做永久删除。

trajectory 被判定为有效全量生产数据，需要同时满足：

- `trace_max_samples` 为 `null` 或 `<= 0`，表示请求源数据全量覆盖。
- 预期源数据条数可以从 `final_config.yaml` 的 `data.train_max_samples` 或显式 fallback 推断。
- `raw_traces.jsonl` 行数不少于预期源数据条数。
- `metrics.jsonl` 行数不少于预期源数据条数。
- `metrics.jsonl` 中没有 `status="failed"` 或 `status="error"`。

reranker 数据被保留，需要满足：

- 它的 manifest 或 source 信息指向一个已保留的全量 trajectory。
- `input_dataset/dataset.jsonl` 有记录。
- 至少一个 `train_dataset/<version>/dataset.jsonl` 有记录。

常用命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag

bash cleanup_keep_full_dataset.sh --dry-run
bash cleanup_keep_full_dataset.sh
```

当旧数据缺少 `data.train_max_samples`，但确实需要数值兜底时，可以显式传入：

```bash
bash cleanup_keep_full_dataset.sh --fallback-full-sample-count 5100
```

这个 `5100` 只是当前 `co_search_ablation` 全量样例的源数据规模，不是通用规则。

## 10. 当前保留数据快照

当前保留的 trajectory 版本：

```text
260703f_AIR_v1_traj_co_search_ablation.train_global_step_79
```

当前保留的 reranker 版本：

```text
260703f_AIR_v1_traj_co_search_ablation.train_global_step_79__input_recall50_emptylabel
```

当前样例数据规模：

```text
raw_traces.jsonl:             5100
metrics.jsonl:                5100
trajectory.jsonl:             12390
input_dataset/dataset.jsonl:  12390
train_dataset/dataset.jsonl:  12390
```

当前 trajectory 状态分布：

```text
answered:        5098
max_turns:       1
no_valid_answer: 1
```

当前数据源分布：

```text
nq:              2006
hotpotqa:        1462
musique:         905
2wikimultihopqa: 727
```

这些数字只是当前 `co_search_ablation` 全量产物的事实记录，不应写入清理逻辑或未来数据集判断逻辑。

## 11. 运行后检查流程

数据生产完成后，按以下顺序检查：

1. 查看 pipeline run output 目录，确认 `generate_traces` 和 `build_reranker_dataset` 均完成。
2. 查看 `data/AgenticIterRag/trajectory/<version>/README.md`。
3. 查看 `data/AgenticIterRag/trajectory/<version>/manifest.json`。
4. 查看 `data/AgenticIterRag/trajectory/<version>/final_config.yaml`。
5. 检查 `metrics.jsonl` 中是否存在 `failed` 或 `error`。
6. 查看 `data/AgenticIterRag/llm_reranker_train_set/<version>/README.md`。
7. 查看 `input_dataset/README.md` 和 `input_dataset/manifest.json`。
8. 查看 `train_dataset/<version>/README.md` 和 `train_dataset/<version>/manifest.json`。
9. 检查 JSONL 主产物是否存在且行数符合预期。
10. 检查 Parquet 镜像是否存在；如果缺失，确认日志或 README 中有 warning。
11. 对照 `final_config.yaml` 的 `data.trace_max_samples` 和 `data.train_max_samples` 判断是否为全量生产。

可直接使用的检查命令示例：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag

wc -l trajectory/<trajectory_version>/raw_traces.jsonl
wc -l trajectory/<trajectory_version>/metrics.jsonl
wc -l trajectory/<trajectory_version>/trajectory.jsonl

grep -E '"status"[[:space:]]*:[[:space:]]*"(failed|error)"' \
  trajectory/<trajectory_version>/metrics.jsonl

wc -l llm_reranker_train_set/<reranker_version>/input_dataset/dataset.jsonl
find llm_reranker_train_set/<reranker_version>/train_dataset \
  -mindepth 2 -maxdepth 2 -name dataset.jsonl -exec wc -l {} \;
```

如果 `grep` 没有输出，表示没有 `failed` 或 `error` 状态。

## 12. 旧 planning 文档与当前实现的差异

旧 planning 文档提出过“不在 shell 中维护 batch size”等方向。当前实现保留了少量 CLI override，主要用于全量/烟测切换和临时性能参数覆盖：

```text
data.trace_max_samples
infer_budget.infer_batch_size
infer_budget.vllm.gpu_memory_utilization
```

因此当前规范是：

- 业务语义、数据路径、checkpoint、stage 开关和数据构造策略应写在 YAML。
- 少量运行态参数可以在 task shell 中作为默认 CLI override。
- 用户仍可通过 `"$@"` 追加更高优先级 CLI override 做 smoke、dry-run 或临时调参。

旧中间数据设计文档中的目录结构和字段设计已基本落地，但当前实现额外补充了：

- 每层数据目录的 `README.md`。
- config-driven 清理脚本。
- stage-level resource placement。
- agent/retriever retry 与 `fail_on_error` 策略。
- 多 retriever backend 通过统一 proxy 暴露服务地址。

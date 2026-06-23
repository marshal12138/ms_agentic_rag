# Train Report System 1.0

本文档记录当前训练报告系统的设计、调用链、schema 约定和产出规则。这里讨论的是 **train report system**，不覆盖后续可能新增的 `eval report system`、`data_process report system` 等报告系统。

## 目标

训练报告系统要解决三个问题：

1. 不同训练子项目共用同一套报告生成引擎，公共代码集中在 `src/logs/report_system`。
2. 不同训练子项目维护自己的指标 schema，schema 放在各自 `scripts/<project>/assets/report_schema.py`。
3. 报告产物只保存最新版本。周期性 snapshot 覆盖同一组 `latest` 文件；训练结束后 final/all report 再覆盖同一组 `latest` 文件。

## 代码边界

公共层：

```text
src/logs/report_system/
  logging_reports.sh
  train_timing_report.py
  train_metrics_report.py
  train_metrics_plots.py
  train_max_metric_step.py
  report_io.py
  report_schema.py
```

项目 schema：

```text
scripts/cosearch_local/assets/report_schema.py
scripts/coagenticRetriever_local/assets/report_schema.py
scripts/iterRag_scripts/assets/report_schema.py
```

兼容 wrapper：

```text
scripts/cosearch_local/generate_timing_report.py
scripts/cosearch_local/generate_training_metrics_report.py
scripts/cosearch_local/generate_detailed_metrics_report.py
scripts/cosearch_local/plot_training_metrics.py
```

这些 wrapper 只用于兼容旧手动调用，内部转发到 `src/logs/report_system/train_*`。新训练主链路不应通过某个项目目录去调用另一个项目的报告脚本。

## 调用链

以 CoAgenticRetriever 训练为例：

```text
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
  source src/logs/report_system/logging_reports.sh
  REPORT_SCHEMA_PATH=scripts/coagenticRetriever_local/assets/report_schema.py
  coagentic_start_training_reporter()
    -> cosearch_start_training_reporter()
      -> cosearch_generate_training_reports()
        -> train_report_system_generate_reports(snapshot)
          -> train_timing_report.py
          -> train_metrics_report.py
          -> train_metrics_plots.py
  coagentic_generate_final_training_reports()
    -> train_report_system_generate_reports(final)
```

CoSearch、CoAgenticRetriever、AgenticIterRag 都通过 `src/logs/report_system/logging_reports.sh` 进入公共训练报告系统；区别只在 `REPORT_SCHEMA_PATH`。

## Snapshot 与 Final 覆盖策略

报告系统只写固定的 latest 产物：

```text
<RUN_NAME>.timing_report.latest.md
<RUN_NAME>.training_metrics_report.latest.md
<RUN_NAME>.detailed_metrics_report.latest.md
<RUN_NAME>.metrics.latest_<plot_group>.png
```

周期性 snapshot：

- 由 `cosearch_start_training_reporter "${ROOT}"` 启动后台 reporter。
- 默认 `TRAIN_REPORT_SNAPSHOT_MODE=latest`，每次读取 metrics JSONL 当前最大 step，并生成 `steps <= <max_step>` 的 latest 报告。
- 每次 snapshot 都覆盖同一组 latest 文件，不产生 `step10`、`step20` 等历史报告文件。

训练结束 final/all：

- 训练脚本调用 `cosearch_generate_final_training_reports "${ROOT}"`。
- final 模式不传 `--step-limit`，报告覆盖到 metrics JSONL 中所有已完成 step。
- final 仍然写同一组 latest 文件，因此会覆盖训练过程中的 snapshot 报告。

兼容旧式定点 snapshot：

- 如果显式设置 `TRAIN_REPORT_SNAPSHOT_MODE=scheduled`，则 `REPORT_STEPS` 被解释为 snapshot step 列表。
- 报告系统会选择 `REPORT_STEPS` 中不超过当前最大 step 的最后一个 step 作为 `--step-limit`。
- 即使使用 scheduled 模式，也仍然覆盖 latest 文件，不保留历史 step 文件。

## Schema 约定

每个训练子项目通过一个 Python schema 文件描述自己的指标字段。公共引擎只读取 schema，不硬编码任何项目指标名。

核心字段：

```python
PROJECT_NAME = "CoAgenticRetriever"

METRIC_GROUPS = {
    "agent_scores": [
        "main_agent/score_mean",
        "main_agent/f1_mean",
        "main_agent/valid_rate",
    ],
    "ranker_losses": [
        "ranker/loss",
    ],
}

PLOT_GROUPS = {
    "agent_scores": {
        "agent": [
            "main_agent/score_mean",
            "main_agent/f1_mean",
        ],
    },
}

DETAILED_METRIC_KEYS = [
    "agent_rollout_num",
    "main_agent/score_mean",
    "ranker/loss",
]

ROLLOUT_ROLE_DIRS = ["main_agent", "main"]

GPU_GROUPS = {
    "main_agent": "MAIN_GPU_IDS",
    "ranker": "RANKER_GPU_IDS",
}
```

可选 hook：

```python
def normalize_metric_row(row): ...
def build_extra_markdown_sections(context): ...
def build_extra_plot_specs(context): ...
def discover_rollout_dirs(rollout_data_dir): ...
```

这些 hook 用于新增项目的特殊指标规范化、额外 markdown 区块、额外图表、特殊 rollout 目录发现。

## 当前项目 Schema

CoSearch：

- 保留旧 key：`main/*`、`reranker/*`、`main_actor/*`、`reranker_actor/*`。
- rollout 目录默认识别 `rollout_data/main`、`rollout_data/reranker`。
- GPU 分组为 `main_actor` 与 `reranker`。

CoAgenticRetriever：

- agent LLM 指标使用 `main_agent/*`、`main_agent_actor/*`、`main_agent_response_length/*`。
- ranker 对比学习指标使用 `ranker/*`，例如 `ranker/loss`、`ranker/acc@1`、`ranker/mrr`、`ranker/score_margin`。
- rollout 目录默认识别 `rollout_data/main_agent`，兼容 fallback `rollout_data/main`。
- GPU 分组为 `main_agent` 与 `ranker`。

AgenticIterRag：

- 当前训练日志仍使用历史 CoSearch key，因此 schema 独立放在 `scripts/iterRag_scripts/assets/report_schema.py`，但字段与 CoSearch 兼容。

## 产出内容

Timing report：

- train step 平均、p50、p90、max 耗时。
- `timing_s/*` 下各 action 的 count、avg、p50、p90、max。
- search timing JSONL 中 HTTP retrieve 成功调用耗时。
- nvidia-smi CSV 的 GPU 利用率、显存、功耗分组统计。

Training metrics report：

- rollout dump 目录、step 文件数、每 step dump 行数统计。
- 按 schema 的 `METRIC_GROUPS` 输出 count、first、last、avg、min、p50、p90、max。
- 列出当前 metrics JSONL 中所有 numeric key。

Detailed metrics report：

- 按 schema 的 `DETAILED_METRIC_KEYS` 输出 per-step 表格。
- `agent_rollout_num` 是派生字段：从 rollout dump 的 `output` 文本估算每条轨迹 assistant action cycles。
- rollout 目录自动按 schema 的 `ROLLOUT_ROLE_DIRS` 识别，不再写死 `main`。

Metrics plots：

- 按 schema 的 `PLOT_GROUPS` 输出 PNG。
- CoAgenticRetriever 默认输出 agent scores、agent losses、agent lengths、ranker quality、ranker losses 五类图。

## 新增项目接入

新增训练子项目时：

1. 在项目脚本目录中新增 `assets/report_schema.py`。
2. 训练脚本 source `src/logs/report_system/logging_reports.sh`。
3. 设置 `REPORT_SCHEMA_PATH="${ASSETS_DIR}/report_schema.py"`。
4. 启动周期性报告：`cosearch_start_training_reporter "${ROOT}"`，或定义项目自己的薄 wrapper。
5. 训练结束生成 final/all：`cosearch_generate_final_training_reports "${ROOT}"`。
6. 不要从一个项目脚本目录调用另一个项目目录下的报告 Python。

## 当前验证

已用以下历史 run 离线验证新 CoAgenticRetriever schema：

```text
log/train_logs/coAgenticRetriever/260611-000819-fullpara_naiveConSample
```

验证结果：

- latest timing report 覆盖到 `completed_train_steps=79`、`max_step_in_report=79`。
- latest training metrics report 正确读取 `main_agent/*` 和 `ranker/*`。
- latest detailed metrics report 正确识别 `rollout_data/main_agent`，不再全是 `N/A`。
- latest plots 生成：
  - `metrics.latest_agent_scores.png`
  - `metrics.latest_agent_losses.png`
  - `metrics.latest_agent_lengths.png`
  - `metrics.latest_ranker_quality.png`
  - `metrics.latest_ranker_losses.png`


# AIR LLM Reranker 训练日志结构

本文档描述 AIR LLM reranker 训练流程当前使用的日志目录结构、命名规则、文件内容和排查顺序。

核心原则是：pipeline 每次运行都会生成一个动态 `RUN_NAME`，所有日志和产物都必须挂到这个动态 run 下面，避免不同训练任务互相覆盖。

## 1. Run 级日志目录

每次启动 pipeline，compiler 会动态生成一个 `RUN_NAME`。

目录规则：

```text
log/agenticIterRag/<RUN_NAME>/runtime_logs
```

`RUN_NAME` 规则：

```text
<日期时间微秒>-pipeline-<experiment_name>
```

示例：

```text
260705-143953-503341-pipeline-agentic_iter_rag_v1_dataprod_to_llm_reranker_training_260703b
```

历史 run 示例目录：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/260705-143953-503341-pipeline-agentic_iter_rag_v1_dataprod_to_llm_reranker_training_260703b/runtime_logs
```

主要文件：

```text
runtime_logs/
  pipeline.terminal.log
  pipeline.final_config.yaml
  pipeline.final_config.json
  pipeline.args.txt
  pipeline.env
  pipeline.runtime_env.sh
```

文件含义：

```text
pipeline.terminal.log
```

pipeline launcher 的 terminal 输出，同时写屏幕和磁盘。

```text
pipeline.final_config.yaml
pipeline.final_config.json
```

本次 run 的最终配置快照，包含 base config、overlay、CLI override 和动态 `runtime_compiled` 字段。

```text
pipeline.args.txt
```

本次 launcher 收到的命令行参数。

```text
pipeline.env
pipeline.runtime_env.sh
```

本次 run 的动态环境变量，包含 `LOG_DIR`、`ARTIFACT_ROOT`、`RUN_NAME`、`FINAL_CONFIG_YAML` 等。
其中 `ARTIFACT_ROOT` 是兼容变量名，实际指向本次 run 内部的 `outputs/` 目录。

## 2. LLM Reranker Phase 日志目录

LLM reranker 训练日志挂在 run 级 `runtime_logs` 下面。

目录规则：

```text
runtime_logs/train_llm_reranker/<phase>/
```

当前 phase：

```text
stage1_format
stage2_agentic
```

默认结构：

```text
runtime_logs/
  train_llm_reranker/
    stage1_format/
      rollout_data/
      validation_data/
    stage2_agentic/
      rollout_data/
      validation_data/
```

`stage1_format` 是格式奖励训练阶段，reward 是 `reranker_format_reward`，主要检查 `<reason>/<rerank>` 格式和长度。

`stage2_agentic` 是 agentic rollout reward 阶段，reward 是 `agentic_rag_rollout_reward`。当前默认关闭。开启后会把 reranker 输出接回 frozen search agent 做 continuation rollout。

## 3. Rollout Data 目录

目录规则：

```text
runtime_logs/train_llm_reranker/<phase>/rollout_data/
```

VERL 会按 step 写 JSONL。

文件名通常是：

```text
<global_step>.jsonl
```

示例：

```text
rollout_data/
  1.jsonl
  2.jsonl
  3.jsonl
```

每行是一条 rollout 样本，核心字段来自 VERL generation dump：

```json
{
  "input": "...decoded prompt...",
  "output": "...decoded model response...",
  "gts": "...ground truth/reward_model...",
  "score": -0.5,
  "step": 1
}
```

可能还会带 reward extra info，例如：

```json
{
  "format_valid": false,
  "format_error_code": "missing_rerank_tag",
  "response_length_tokens": 1024
}
```

实际字段取决于当前 reward function 返回的 extra info。

最关键字段：

```text
input
```

LLM reranker 的真实输入 prompt。

```text
output
```

LLM reranker 的真实输出文本。

```text
score
```

该 rollout 的 reward 分数。

```text
step
```

训练 step。

## 4. Validation Data 目录

目录规则：

```text
runtime_logs/train_llm_reranker/<phase>/validation_data/
```

内容结构和 `rollout_data` 类似，也是按 step 写 JSONL。

当前默认配置中：

```text
test_freq: -1
val_before_train: false
```

所以默认不会周期 validation。这个目录主要是预留，保证 `stage1_format` 和 `stage2_agentic` 的日志配置方式一致。

## 5. Train Stage Artifact 目录

除 `runtime_logs` 外，训练 stage 还有 artifact 目录。

目录规则：

```text
log/agenticIterRag/<RUN_NAME>/outputs/stages/train_llm_reranker/
```

结构示例：

```text
stages/train_llm_reranker/
  manifest.json
  runtime_services/
    stage1_format/
      run_verl_reranker_grpo.sh
      verl_command.argv
      verl_command_plan.json
      verl_train.log
      air_reranker_reporter.log
      air_reranker_reporter.stop
  training_reports/
    stage1_format/
      air_llm_reranker.metrics.jsonl
      air_llm_reranker.training_metrics_report.latest.md
      air_llm_reranker.detailed_metrics_report.latest.md
      air_llm_reranker.report_manifest.json
      air_llm_reranker.metrics.latest_reranker_rewards.png
      air_llm_reranker.metrics.latest_reranker_losses.png
      air_llm_reranker.metrics.latest_reranker_lengths.png
      air_llm_reranker.metrics.latest_reranker_performance.png
  reranker_model_verl/
    stage1_format/
      global_step_10/
      global_step_20/
      latest_checkpointed_iteration.txt
```

重点文件：

```text
runtime_services/<phase>/run_verl_reranker_grpo.sh
```

实际启动 VERL 的 shell 脚本。

```text
runtime_services/<phase>/verl_command.argv
```

最终传给 `python -m verl.trainer.main_ppo` 的参数，一行一个 override。这里可以确认 batch size、rollout n、response length、rollout data dir 等真实生效配置。

```text
runtime_services/<phase>/verl_command_plan.json
```

结构化命令计划，包含资源、模型路径、dataset、reward 和日志路径。

```text
runtime_services/<phase>/verl_train.log
```

VERL stdout/stderr 训练日志。周期 reporter 会解析这个文件来生成 metrics 和曲线。

```text
training_reports/<phase>/air_llm_reranker.metrics.jsonl
```

从训练日志解析出来的指标序列。

```text
training_reports/<phase>/*.png
```

周期刷新的训练曲线图。

```text
reranker_model_verl/<phase>/global_step_<N>/
```

VERL checkpoint 目录。

## 6. 全局 Metrics JSONL

AIR LLM reranker 训练还会写一个公共 metrics 文件。

示例路径：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/agentic_iter_rag_reranker/agentic_iter_rag_v1_dataprod_to_llm_reranker_training_260703b.jsonl
```

内容是每个 step 的聚合 metrics，例如：

```json
{
  "step": 42,
  "data": {
    "critic/score/mean": -0.5,
    "response_length/mean": 1024.0,
    "response_length/clip_ratio": 1.0,
    "actor/pg_loss": 0.0,
    "actor/grad_norm": 0.0,
    "timing_s/step": 1370.53
  }
}
```

这个文件适合看整体训练状态，但不保存逐条 input/output。

逐条 input/output 应该看：

```text
runtime_logs/train_llm_reranker/<phase>/rollout_data/<step>.jsonl
```

## 7. 默认日志配置规则

`reranker_training.trainer.rollout_data_dir` 默认是 `null`。

为空时自动解析为：

```text
${runtime_compiled.LOG_DIR}/train_llm_reranker/<phase>/rollout_data
```

`reranker_training.trainer.validation_data_dir` 默认是 `null`。

为空时自动解析为：

```text
${runtime_compiled.LOG_DIR}/train_llm_reranker/<phase>/validation_data
```

如果用户显式配置相对路径，则相对以下目录解析：

```text
${runtime_compiled.LOG_DIR}/train_llm_reranker
```

如果用户显式配置绝对路径，则以该绝对路径为 dump 根目录。

无论配置为空、相对路径还是绝对路径，最终都会自动追加：

```text
<phase>/rollout_data
<phase>/validation_data
```

这样可以保证 `stage1_format` 和 `stage2_agentic` 使用同一套日志配置方法，但训练样本输出不会混在同一个目录里。

## 8. 排查顺序

建议按下面顺序看日志：

1. 看最终配置：

```text
runtime_logs/pipeline.final_config.yaml
```

2. 看 VERL 实际命令：

```text
outputs/stages/train_llm_reranker/runtime_services/<phase>/verl_command.argv
```

3. 看训练过程：

```text
outputs/stages/train_llm_reranker/runtime_services/<phase>/verl_train.log
```

4. 看曲线和指标：

```text
outputs/stages/train_llm_reranker/training_reports/<phase>/
```

5. 看 LLM reranker 真实输入输出：

```text
runtime_logs/train_llm_reranker/<phase>/rollout_data/<step>.jsonl
```

6. 看 checkpoint：

```text
outputs/stages/train_llm_reranker/reranker_model_verl/<phase>/global_step_<N>/
```

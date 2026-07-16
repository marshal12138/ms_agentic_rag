# SPAD Teacher Prompt Engineering

## 2026-07-15 New-Data PE

The current PE cycle uses 512 manually audited trajectories sampled from the new 5100-step training run. The frozen ablation benchmark is `benchmark_newdata_512_ablation.jsonl`: 384 dev cases are used for prompt selection and 128 cases were initially sealed for holdout. Each of the four 20-step strata contributes 96 dev and 32 holdout cases. The holdout was first unsealed for the frozen R5 single-prompt evaluation and was later consumed by combination-strategy diagnostics; it must not be represented as untouched for another selection cycle.

The new equal-weight objective is `0.5 * I_F1 + 0.5 * gold_token_F1_coverage_on_manual_S`. A manual-S case predicted as non-S contributes zero answer coverage. The runner also reports gold EM, conditional answer quality, manual-answer agreement, the actual `teacher_called=true` operational slice, controls, and every step layer.

The frozen single-prompt candidate is `gold_support_evidence_only_v3`: Original question plus reference gold plus full title/passage evidence, with sub-query strings hidden and no repeated question tail. On three cache-free dev runs its teacher-called equal objective is `0.7525 [0.7516, 0.7535]`; on three frozen holdout runs it is `0.8149 [0.7993, 0.8417]`.

The current combination leader is `hard_gate_r5_literal_canonical_v2`. A production no-gold prompt makes a binding I decision; R5 is called only for Stage-A S/A, supported answers are selected across stages, and a reference answer is substituted only when its normalized literal occurs in Search evidence. Three dev runs give teacher-called I F1 `0.8924`, gold coverage `0.6825`, and equal objective `0.7874` at `1.3558x` mean elapsed. Its reused-holdout diagnostic is `0.8812/0.9000/0.8906` at `1.2307x`. Because the combination cycle already consumed that holdout, a new untouched rollout sample is required for an unbiased final estimate. Gold-aware and no-gold results remain separate production regimes.

The source training run used `spad_teacher_evidence_status_answer_v2`, which is byte-equivalent to PE variant `baseline_current_v2`. Three fresh dev repeats of that production prompt give teacher-called I F1 `0.8924`, gold coverage `0.3180`, and equal objective `0.6052`. Hard-gate v2 therefore preserves the production I boundary exactly while improving the non-I answer path.

New artifacts are documented chronologically in `NEW_DATA_PE_WORKLOG.md`. Sampling, annotation, and benchmark construction are implemented by `sample_newdata_rollouts_512.py`, `build_manual_judgments_newdata_512.py`, and `build_newdata_ablation_benchmark.py`.
Completed new-data runs are indexed by `build_newdata_results_index.py` in `NEW_DATA_RESULTS_INDEX.md`.
Candidate repeats are summarized by `build_newdata_stability_report.py` in `NEW_DATA_STABILITY.md`.
Combination variants and execution are implemented by `composite_prompt_variants.py` and `run_composite_ablation.py`; deterministic policies derived from persisted independent stages use `derive_composite_policy.py` and are marked as derived in result metadata.
Detailed production/R5/Hard-gate behavior, metrics, costs, and selection boundaries are documented in `NEW_DATA_STRATEGY_COMPARISON.md`.

本目录用于对 GLM-4.7-Flash teacher 做可复现的 prompt/layout 消融。当前完成 50 个有效方案，并对
Top 5 各做 3 次无缓存新推理；领先的
单调用策略是 `question-tail evidence-only`：隐藏 sub_query、保留完整 title/passage，并在证据后重申
Original question。最新三次 holdout 均值为 I precision `0.8556`、recall `0.8667`、F1 `0.8606`。
它同时也是当前综合时间后的最佳策略：单样本一次 teacher 调用，parse rate `1.0`，平均 completion
约 `83.1 tokens`，平均请求 `7.71s`。最高单次 `0.920/0.920` 仅作为观测上限，不作为稳定效果。

## 文档导航

| 文档 | 作用 |
| --- | --- |
| `readme.md` | 目录入口、当前状态、文档与代码导航 |
| `PLAN.md` | 消融开始前冻结的原始目标、指标、数据切分、硬件和 A/B 实验计划 |
| `PE消融指令.md` | 汇总用户在本轮 PE 工作中给出的全部目标、约束、执行和停止规则 |
| `ABLATION_HISTORY.md` | 完整消融历史；每 10 个有效方案记录效果、失败模式和反思 |
| `RESULTS_INDEX.md` | 由脚本生成的所有结果指标索引，包含 invalid 标记 |
| `REPLICA_STABILITY.md` | Top 5 严格三重复的准确率/耗时排名，以及此前累计稳定性 |
| `持续消融计划.md` | 当前维持策略、生产 v3 集成、新 200 条数据验证和后续停止规则 |
| `代码结构介绍.md` | 数据、prompt registry、runner、评分、服务脚本、测试和结果目录说明 |
| `results/_invalid_data_leakage/INVALID.md` | 解释哪些实验存在 few-shot 数据泄漏以及为何永久禁用 |
| `results/_invalid_cached_replay/INVALID.md` | 解释共享 response cache 复放为何不算独立推理 |

`PLAN.md` 是实验前计划，不代表当前完成状态；当前结论以 `ABLATION_HISTORY.md`、
`REPLICA_STABILITY.md` 和 `持续消融计划.md` 为准。

## 数据文件

| 文件 | 作用 |
| --- | --- |
| `manual_judgments_237.tsv` | 固定的 237 条人工 S/I/A 判断表 |
| `benchmark_237.jsonl` | 从原 rollout 恢复的冻结 question/evidence benchmark |
| `benchmark_237.manifest.json` | benchmark 数量、split、标签统计与 SHA256 |

## 主要代码

| 文件 | 作用 |
| --- | --- |
| `build_manual_judgments_237.py` | 从历史 rollout 重建人工判断表并校验统计 |
| `build_benchmark.py` | 构建 grouped dev/holdout benchmark |
| `prompt_variants.py` | system prompt 和 user layout registry |
| `run_ablation.py` | 完整 XML 推理、可显式禁用的响应缓存、解析、评分和结果落盘 |
| `run_binary_gate.py` | one-token I/non-I 概率 gate 研究 runner |
| `evaluate_ensemble.py` | 已落盘多 prompt 投票与成本评估 |
| `build_results_index.py` | 生成 `RESULTS_INDEX.md` |
| `build_stability_report.py` | 生成 `REPLICA_STABILITY.md` |
| `test_teacher_pe.py` | benchmark、prompt、parser 和 metrics 测试 |

## 服务脚本

```bash
./launch_teacher_replicas.sh
./status_teacher_replicas.sh
./stop_teacher_replicas.sh
```

启动拓扑为 4 个 TP=2 replica，端口 `8067-8070`。离线消融可让策略之间并行，但每个 replica
所有进程的总并发应不超过 16，避免 OOM。

截至 2026-07-10 18:23 CST，四个 replica 均保持运行且 ready，供后续任务继续使用。

## Search-R1 奖励离线重放

2026-07-10 18 时使用当前最佳 `baseline_question_tail_evidence_only_v2` 对最新 Search-R1
训练的全部 4096 条 rollout 完成一次 cache-free fresh replay。4096 次请求全部首次成功，
墙钟 512.99 秒；XML parse rate 为 4092/4096，即 99.90%。

面向 SPAD Stage1 的 judge reward 将非恒定 GRPO group 从原始 138/512 提升至 210/512；
包含 actor-answer F1 的完整轨迹诊断分数提升至 324/512。240 条人工 evidence 标签上的
accuracy 为 77.9%，I precision/recall/F1 为 81.2%/91.9%/86.2%。信号密度明显增加，
但 false-I、4 条格式错误和逐样本排序重叠使其尚未达到启动训练的条件。

结果目录为 `results/R1_reward_replay_qtail_evidence_260710_18a`，执行脚本为
`run_search_r1_reward_replay.py`；完整中文结论见
`docs/AgenticIterRag_v1/work_report/260710-18a_Search-R1最佳LLM-Judge奖励离线重放实验.md`。

## 常用命令

```bash
python -m unittest -v test_teacher_pe.py

python run_ablation.py \
  --variant baseline_question_tail_evidence_only_v2 \
  --split all \
  --disable-cache \
  --output-dir results/<new-experiment-id>

python run_ablation.py \
  --benchmark benchmark_newdata_512_ablation.jsonl \
  --variant baseline_question_tail_evidence_only_v2 \
  --split dev \
  --disable-cache \
  --output-dir results_newdata/<new-experiment-id>

python build_results_index.py
python build_stability_report.py
```

有效消融必须使用 `--disable-cache` 且确认 `run.json.cache_hits=0`。不要覆盖已有 result 目录，不要把
237 条数据或其改写放入 few-shot，不要从 invalid 目录选择候选。

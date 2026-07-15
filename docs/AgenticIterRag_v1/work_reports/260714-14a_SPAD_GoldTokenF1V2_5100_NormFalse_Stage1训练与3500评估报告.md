# SPAD Gold Token-F1 V2：5100、Norm=False Stage1 训练与 3500 评估报告

日期：2026-07-14，02pm（北京时间）

> 状态：正式 Stage1 训练、HF checkpoint 导出、3500 单次确定性评估均已完成。训练与评估进程退出码均为 0。本报告不包含 Stage2 或 Stage3。

## 1. 任务与结论摘要

本轮使用 `spad_em_teacher_backoff_gold_token_f1_bonus` 的 V2 bonus eligibility，在 5100 规模训练配置上关闭 GRPO 组内标准差归一化：

```yaml
norm_adv_by_std_in_grpo: false
```

正式训练完成 79 steps，导出 `global_step_79`。随后在固定 3500e 数据集上进行一次 temperature=0 的 no-ranker 评估，3500 条全部成功。

| 评估对象 | EM | F1 | 完整答案率 | 首轮搜索率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gold Token-F1 V2，5100，norm=false | 0.1831 | 0.2673 | 0.7906 | 1.0000 | 1.8889 | 0.2154 | 0.1863 |

核心观察：

1. 当前 checkpoint 的 F1 为 `0.2673`，与历史 SPAD stable 5100 的 `0.2700` 很接近，但 EM 低 `0.0092`。本报告未对这两个独立训练 checkpoint 做新的 paired bootstrap，因此不宣称差异显著或不显著。
2. 相对历史 Gold Token-F1 V1 5100，本次 EM 基本持平（`-0.0006`），F1 高 `0.0097`，完整答案率高 `0.1572`，平均搜索数少 `1.1182`，Max-turn 率低 `0.1726`。行为上的改善很明显。
3. 但本次相对历史 V1 同时改变了 `bonus eligibility V1 -> V2` 和 `norm_adv_by_std_in_grpo true -> false`，而且是新的随机训练 run。不能把改善单独归因于其中任一项。
4. 训练后 3 steps 的平均 reward 为 `0.3336`，其中 base reward 为 `0.3279`、extra bonus 仅为 `0.0056`。V2 bonus 在后期仍提供排序信号，但对平均 reward 的直接数值贡献较小。
5. 配置使用 `train_max_samples=5100`，但 79 steps、每 step 64 个 prompt，实际完整消费 `5056` 个 prompt，而非 5100；每题 8 条 rollout，共落盘 `40448` 条训练轨迹。

## 2. 正式训练身份与启动方式

### 2.1 Run 与入口

正式 run：

```text
260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1
```

规范启动入口：

```bash
bash tasks/train_tasks/agenticIterRag/run_260713_AIR_spad_qwen3_1_7b_glm47_5100_gold_token_f1_v2_normfalse_stage1.sh
```

该入口按顺序叠加三个 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_formal_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_scale_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_gold_token_f1_v2_normfalse_stage1_overlay.yaml
```

第三个 overlay 显式设置：

```yaml
reward.type: spad_em_teacher_backoff_gold_token_f1_bonus
partial_reward: 0.1
gold_token_f1_bonus: 0.1
total_training_steps: 79
train_max_samples: 5100
save_freq: 79
data_seed: 42
norm_adv_by_std_in_grpo: false
stream_group_max_inflight: 2
stop_after_sub_stage: search_policy_rl
```

`answer_refresh_data` 与 `answer_distillation` 均禁用，因此本轮只运行 SPAD Stage1。

### 2.2 关键控制变量

| 项目 | 实际设置 |
|---|---|
| 基座模型 | Qwen3-1.7B |
| 训练数据 | `data/global_train_eval_data/5100t/co_search_ablation.train.parquet` |
| `train_max_samples` | 5100 |
| 实际 prompt 数 | 5056（79 × 64） |
| 每题 rollout 数 | 8 |
| 实际 rollout 数 | 40448 |
| 训练 batch | 64 prompts |
| 总 step | 79 |
| 数据 shuffle / seed | `true` / `42` |
| Actor 学习率 | `1e-6` |
| Actor rollout | temperature 1.0，top_p 1.0 |
| 最大 assistant/user turns | 6 / 6 |
| `norm_adv_by_std_in_grpo` | `false` |
| `stream_group_max_inflight` | 2 |
| 保存频率 | 79，只保存最终 step |
| 执行阶段 | 仅 Stage1 |

关闭标准差归一化后，GRPO 仍在题内中心化 reward，但不再除以题内 reward 标准差：

```text
advantage_i = reward_i - mean(reward_group)
```

因此 `0.1` backoff 与 `1.0` EM 的绝对量级差异可以保留到 advantage 中。本轮日志中的 advantage 最大值为 `0.875`，与 8-rollout 组内只有一条 reward=1 时的未标准差归一化结果一致，说明开关确实生效。

### 2.3 资源与 Teacher 参数

| 组件 | 资源/参数 |
|---|---|
| Trainer / Actor | NPU 0、1、2、3，共 4 卡 |
| Teacher | GLM-4.7-Flash，TP=2，NPU 4、5 |
| Recall | NPU 6、7，两个 backend |
| Teacher temperature | 0 |
| Teacher top_p | 1 |
| Teacher max_tokens | 512 |
| Teacher timeout | 180 秒 |
| Teacher batch workers | 16 |
| Teacher prompt | `spad_teacher_evidence_status_answer_v2` |

## 3. Reward 原理与 V2 的关键区别

实现模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/
search_policy_teacher_reward_gold_match_bonus.py
```

正式 reward 名称保持为：

```text
spad_em_teacher_backoff_gold_token_f1_bonus
```

当前代码的 bonus eligibility 版本为：

```text
actor_answer_closed_teacher_supported_v2
```

### 3.1 Stable base reward

新模块组合已有 `spad_em_teacher_backoff`，不修改 stable reward 的原实现：

1. 每个问题采样 8 条 Actor rollout，并按问题分组。
2. 若组内至少一条 Actor answer 对任一 gold alias 的 EM 命中，每条轨迹只按自己的 Actor EM 得 `1/0`。
3. 若整组 Actor EM 全为 0，调用 GLM-4.7-Flash 审查检索证据。
4. Teacher 判断证据支持答案或证据存在歧义时，base backoff 为 `0.1`；证据不足、格式错误等情况为 `0`。

### 3.2 Gold Token-F1 extra bonus

V2 在 stable base reward 之外计算：

```text
teacher_gold_token_f1 = max(token_f1(teacher_answer, gold_alias_i))
extra_bonus = 0.1 * teacher_gold_token_f1
final_reward = stable_base_reward + extra_bonus
```

extra bonus 只有在以下条件全部满足时才发放：

- 本题 8 条 Actor rollout 的 EM 全为 0；
- Teacher 实际被调用；
- Teacher 输出成功解析、没有格式错误，且 Teacher answer 非空；
- Actor 自己输出了合法、闭合的最终答案；
- Teacher evidence status 为 `supported_answer` 或 `ambiguous_evidence`。

V2 与历史 V1 的核心差别是：V1 主要约束 Teacher 输出合法；V2 进一步要求 Actor 已合法闭合答案，并要求 Teacher evidence status 属于被认可的两类。Actor 没有闭合答案时，符合条件的轨迹仍可获得原 stable `0.1` backoff，但不能获得额外 token-F1 bonus。

## 4. 训练过程与统计

### 4.1 时间线与完成状态

| 事件 | 时间 |
|---|---|
| Pipeline 后台进程启动 | 2026-07-14 09:10:18 |
| VERL command plan 生成、训练服务进入启动 | 2026-07-14 09:16:05 |
| Step 1 rollout 落盘 | 2026-07-14 09:24:20 |
| Step 79 rollout 落盘 | 2026-07-14 15:06:06 |
| VERL checkpoint 写入 | 2026-07-14 15:06:12 起 |
| HF `model.safetensors` 完成 | 2026-07-14 15:07:32 |
| SPAD manifest 写入、流水线完成 | 2026-07-14 15:07:58 |
| 外层 runner 退出 | 2026-07-14 15:08:00，exit 0 |

时间统计：

- VERL 训练/服务阶段墙钟：`21023.2s`，即约 5 小时 50 分 23 秒；
- 进度条 79 steps：5 小时 46 分 51 秒；
- 从外层启动到最终退出：约 5 小时 57 分 42 秒；
- 79 steps 的 `timing_s/step` 均值：`262.7s`，即约 4 分 23 秒/step；
- 最后 3 steps 均值：`254.3s`，即约 4 分 14 秒/step。

训练 manifest 状态为 `completed`，`return_code=0`，未超时。79 个 rollout shard 全部标记完成，实际/期望 step、prompt 与 rollout 数完全一致。

### 4.2 Reward 与行为走势

下表是训练日志按 step 的算术平均；“全程”对 79 个 step 等权平均，每个 step 都含 512 条 rollout，因此也等价于按 rollout 平均。

| 指标 | Steps 1-3 | 全程 79 steps | Steps 77-79 |
|---|---:|---:|---:|
| Final reward | 0.1439 | 0.3091 | 0.3336 |
| Stable base reward | 0.1332 | 0.3024 | 0.3279 |
| Extra bonus | 0.0107 | 0.0066 | 0.0056 |
| Actor legacy EM | 0.1035 | 0.2805 | 0.3079 |
| Actor legacy F1 | 0.1863 | 0.3511 | 0.3701 |
| 全零 EM 组率 | 0.7135 | 0.5742 | 0.5469 |
| Teacher 调用率 | 0.7057 | 0.5726 | 0.5469 |
| Bonus eligible 率 | 0.2259 | 0.1978 | 0.1855 |
| 正 bonus 覆盖率 | 0.1393 | 0.0996 | 0.0807 |
| 平均搜索数 | 1.9225 | 1.3406 | 1.4531 |
| 平均重复查询数 | 0.5775 | 0.1551 | 0.1888 |
| 单 step 耗时 | 299.8s | 262.7s | 254.3s |

训练后 3 steps 明细：

| Step | Final reward | Base reward | Bonus | Actor EM | Actor F1 | Teacher 调用率 | 正 bonus 覆盖率 | 平均搜索数 | Step 耗时 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 77 | 0.3474 | 0.3422 | 0.0052 | 0.3223 | 0.3719 | 0.5312 | 0.0625 | 1.4609 | 239.8s |
| 78 | 0.2904 | 0.2848 | 0.0057 | 0.2637 | 0.3319 | 0.5781 | 0.1074 | 1.4609 | 264.1s |
| 79 | 0.3628 | 0.3568 | 0.0060 | 0.3379 | 0.4065 | 0.5312 | 0.0723 | 1.4375 | 259.0s |

从前 3 steps 到后 3 steps，Actor EM 从 `0.1035` 升至 `0.3079`，全零 EM 组率从 `0.7135` 降至 `0.5469`。由于 bonus 只在全零 EM 组内触发，extra bonus 均值从 `0.0107` 自然下降到 `0.0056`。全程 extra bonus 均值只占 final reward 均值约 2.1%，但在具体全零组内仍可能改变 rollout 的相对排序。

### 4.3 Teacher 调用审计

| 项目 | 数量 | 比例 |
|---|---:|---:|
| 总 rollout | 40448 | 100% |
| Teacher called | 23159 | 57.26% |
| Teacher skipped | 17289 | 42.74% |
| Teacher error | 9 | 占 called 0.039% |

`teacher_called + teacher_skipped = 40448`。9 个 Teacher error 是样本级失败，没有导致 step、checkpoint 或流水线失败。

## 5. Checkpoint 与训练产物

最终 HF checkpoint：

```text
checkpoints/AIR/
260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79
```

模型身份：

```text
model.safetensors SHA256:
14cd1b39bad7ce85672d3e243b37fe6650ea30bfcfe6bd5afdbe689909f39f4b

checkpoint fingerprint:
5b45880d83ebc94be953c2277a45c0beca03cef46b6b0890815277e387adf0c4
```

主要产物大小：

| 产物 | 大小 |
|---|---:|
| HF checkpoint | 3.8G |
| VERL checkpoint | 21G |
| 79-step rollout data | 7.6G |

训练日志与 manifest：

```text
log/agenticIterRag/260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1/
runtime_logs/stages/train_agent/spad_rag/search_policy_rl/verl_train.log

log/agenticIterRag/260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1/
outputs/stages/train_agent/spad_rag/spad_manifest.json

log/agenticIterRag/260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1/
outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/manifest.json
```

## 6. 3500 评估协议与过程

### 6.1 评估身份

评估 task：

```text
260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1
```

评估数据：

```text
data/global_train_eval_data/3500e/co_search_ablation.eval.parquet
SHA256 bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283
```

数据源分布：2WikiMultiHopQA 563、Bamboogle 125、HotpotQA 562、MuSiQue 562、NQ 562、PopQA 563、TriviaQA 563。

### 6.2 推理参数

| 项目 | 设置 |
|---|---|
| 模式 | no-ranker |
| Actor vLLM | NPU 0-5，6 个 DP replica，TP=1 |
| Recall | NPU 6-7，2 个 backend |
| Infer batch | 384 |
| 每 Actor `max_num_seqs` | 64 |
| Flush every N | 500 |
| temperature / top_p | 0 / 1 |
| Recall Top N | 50 |
| 模型可见 Top M | 5 |
| 最大 assistant turns | 6 |
| 单轮 response 上限 | 1024 tokens |
| Trace | full |
| Repeat ID | 1 |

评估外层进程于 15:15:40 启动，于 15:30:52 正常退出，总墙钟约 15 分 12 秒，包含 Recall/6 个 Actor 服务加载、推理、full trace 写盘与服务清理。报告记录的纯推理墙钟为 `534.897s`，即 8 分 55 秒。

评估 manifest 明确记录 `output_reuse=false`，本次没有复用旧输出；模型指纹与上述 checkpoint 一致。

## 7. 3500 评估结果

### 7.1 总体效果与行为指标

| N | 成功 | 失败 | EM | F1 | 完整答案率 | 首轮搜索率 | 平均搜索数 | 唯一查询数 | 重复查询率 | Max-turn 率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3500 | 3500 | 0 | 0.1831 | 0.2673 | 0.7906 | 1.0000 | 1.8889 | 1.2474 | 0.2154 | 0.1863 |

状态计数：

| 状态 | 数量 | 比例 |
|---|---:|---:|
| `answered` | 2767 | 0.7906 |
| `no_valid_answer` | 81 | 0.0231 |
| `max_turns` | 652 | 0.1863 |

搜索次数分布：

| 搜索次数 | 数量 | 比例 |
|---|---:|---:|
| 1 | 2397 | 0.6849 |
| 2 | 404 | 0.1154 |
| 3 | 42 | 0.0120 |
| 4 | 5 | 0.0014 |
| 5+ | 652 | 0.1863 |

该模型所有样本都至少搜索一次；68.5% 的样本只搜索一次。`5+` 桶与 `max_turns` 数量相同，说明主要失败模式是持续搜索至轮次上限，而不是完全不搜索。

### 7.2 分数据源结果

| 数据源 | N | EM | F1 | 平均搜索数 |
|---|---:|---:|---:|---:|
| 2WikiMultiHopQA | 563 | 0.1279 | 0.1792 | 2.0533 |
| Bamboogle | 125 | 0.0880 | 0.1732 | 1.6320 |
| HotpotQA | 562 | 0.1886 | 0.2839 | 1.6512 |
| MuSiQue | 562 | 0.0320 | 0.0825 | 2.6281 |
| NQ | 562 | 0.3025 | 0.3958 | 1.4004 |
| PopQA | 563 | 0.3055 | 0.3527 | 2.1030 |
| TriviaQA | 563 | 0.1634 | 0.3305 | 1.5542 |

Macro-average（7 个数据源等权）EM/F1 为 `0.1726/0.2568`。MuSiQue 仍是最弱数据源，且平均搜索数最高；NQ 的 EM/F1 最高。

## 8. 与历史同协议结果的描述性比较

以下历史结果均使用同一 3500e 数据、no-ranker、Top N=50、Top M=5、temperature=0 的协议，但来自不同训练 checkpoint。

| 模型 | 关键训练差别 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|
| SPAD stable 5100 | stable reward，norm=true，inflight=2 | 0.1923 | 0.2700 | 0.7220 | 2.6557 | 0.5906 | 0.2443 |
| Gold Token-F1 V1 5100 | V1 eligibility，norm=true，inflight=2 | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |
| Gold Token-F1 V2 5100，本次 | V2 eligibility，norm=false，inflight=2 | 0.1831 | 0.2673 | 0.7906 | 1.8889 | 0.2154 | 0.1863 |

本次减去历史对照：

| 对照 | ΔEM | ΔF1 | Δ完整答案率 | Δ平均搜索数 | Δ重复查询率 | ΔMax-turn 率 |
|---|---:|---:|---:|---:|---:|---:|
| 相对 SPAD stable 5100 | -0.0092 | -0.0027 | +0.0686 | -0.7668 | -0.3752 | -0.0580 |
| 相对 Gold Token-F1 V1 5100 | -0.0006 | +0.0097 | +0.1572 | -1.1182 | -0.3609 | -0.1726 |

另与 512 Gold Token-F1 V2、norm=false 的三次训练均值比较：512 均值 EM/F1 为 `0.1206/0.2029`，完整答案率 `0.6470`。本次 5100 单 run 分别高 `0.0625/0.0644/0.1436`，表明扩大训练规模后的单点结果明显更强；但一边是三次 checkpoint 均值、一边是单次 checkpoint，不能据此估计训练方差或做严格显著性结论。

本次历史比较是描述性的。它不能回答“改善来自 V2 eligibility 还是 norm=false”，原因包括：

1. V1 到本次同时改变两个训练语义；
2. 各模型是独立随机训练，异步 rollout 与服务调度无法由 `data_seed=42` 完全固定；
3. 本次只训练一次、只评估一次，没有覆盖训练方差；
4. stable 对照使用不同 reward，本身不是单变量消融。

若要分离因果，应固定同一代码时间点、训练数据、seed、inflight 和其余参数，至少形成以下 2×2 对照，并对每格做多次独立训练：

```text
V1 + norm=true
V1 + norm=false
V2 + norm=true
V2 + norm=false
```

## 9. 评估产物

单次评估报告：

```text
reports/eval/agenticIterRag/
260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1.report.md
```

完整 trace、逐题 metrics、summary、run config 与评估 manifest：

```text
log/eval/agenticIterRag/
260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1/
```

主要文件：

```text
trace/traces.jsonl
trace/metrics.jsonl
trace/summary.json
trace/run_config.json
runtime_logs/eval_run_manifest.json
```

Trace 目录约 3.5G。评估 manifest 中记录：

```text
data SHA256: bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283
model fingerprint: 5b45880d83ebc94be953c2277a45c0beca03cef46b6b0890815277e387adf0c4
repeat_id: 1
output_reuse: false
```

## 10. 最终判断

本轮 Gold Token-F1 V2 5100、norm=false 的正式训练和 3500 评估均完整成功。当前 checkpoint 的主要价值不在于超过历史 stable 的精度，而是在 F1 接近 stable 的同时，完整答案率更高、搜索更短、重复查询和 Max-turn 明显更少。

不过，这次结果仍是单次训练 checkpoint，且相对历史 V1 同时改变了 bonus eligibility 与 advantage 标准化。可以把它作为后续 V1/V2 × norm 开关消融的候选基线和正式 checkpoint，但不能把当前优势直接解释为某一个参数的独立贡献。

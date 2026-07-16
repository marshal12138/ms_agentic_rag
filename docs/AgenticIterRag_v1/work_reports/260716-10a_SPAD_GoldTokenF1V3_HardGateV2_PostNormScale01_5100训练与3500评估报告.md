# SPAD Gold Token-F1 V3：Hard-Gate v2、Post-Norm 0.1、5100 Stage1 训练与 3500 评估报告

生成时间：2026-07-16 10am（北京时间）

> 状态：Hard-Gate v2 已作为独立 Teacher 策略和独立 reward 接入正式训练 runtime；5100
> Stage1 的 79 steps、半严格 rollout 审计、HF checkpoint 收尾、3500e 单次评估和相对原
> V3 postnorm01 的逐题配对比较均已完成。训练中只执行 Stage1，没有执行 Stage2/Stage3；
> smoke 训练产生的临时日志和 checkpoint 已删除。

## 1. 结论

本轮保持 V3 postnorm01 的 raw reward、GRPO 组内归一化和 Teacher 组 post-norm scale=0.1，
只把单 Teacher prompt 替换为冻结的 Hard-Gate v2 两阶段策略。

在同一 3500e 数据和同一 no-ranker 协议上，Hard-Gate v2 模型的 F1、完整答案率和搜索行为
同时改善：

| 模型 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|
| Search-R1 512 | 0.1180 | 0.1965 | 0.6271 | 2.3489 | 0.3640 | 0.2569 |
| Search-R1 5100 | 0.1800 | 0.2509 | 0.7317 | 1.7291 | 0.1786 | 0.1549 |
| SPAD stable 5100 | 0.1923 | 0.2700 | 0.7220 | 2.6557 | 0.5906 | 0.2443 |
| Gold F1 V1 5100 | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |
| Gold F1 V2 5100 | 0.1831 | 0.2673 | 0.7906 | 1.8889 | 0.2154 | 0.1863 |
| V3 postnorm01 5100 | 0.1994 | 0.2787 | 0.8340 | 1.6969 | 0.1571 | 0.1369 |
| V3 postnorm03 5100 | 0.1929 | 0.2734 | 0.7100 | 2.6883 | 0.5649 | 0.2714 |
| **V3 postnorm01 + Hard-Gate v2 5100** | **0.2069** | **0.2911** | **0.8611** | **1.5934** | **0.1331** | **0.1071** |
| Hard-Gate v2 - 原 postnorm01 | +0.0074 | **+0.0124** | **+0.0271** | **-0.1034** | **-0.0240** | **-0.0297** |

前七组直接沿用 `260715-09a` 报告中已审计的同协议 3500e 横向结果，本报告不重新读取或
重算这些历史 trace；最后一组为本次新评估。Hard-Gate v2 在八组中取得最高 EM、F1 和完整
答案率，同时取得最低平均搜索数、重复查询率和 Max-turn 率。

逐题 paired bootstrap（10,000 次，seed=42）显示：

- F1：`+0.01241`，95% CI `[+0.00247, +0.02225]`，区间不跨 0。
- 完整答案率：`+0.02714`，95% CI `[+0.01429, +0.04000]`，区间不跨 0。
- EM：`+0.00743`，95% CI `[-0.00200, +0.01686]`，区间跨 0。

因此可以确认本次 run 的 F1 和完整答案率有正向配对证据；EM 的点估计提高约 3.72%，但当前
单次训练不足以确认 EM 提升。相对原 postnorm01，F1 相对提高约 4.45%，平均搜索数减少约
6.10%，Max-turn 率相对下降约 21.71%。

## 2. 实验身份与冻结配置

### 2.1 策略与 reward ID

| 项目 | 值 |
|---|---|
| Teacher strategy ID | `spad_teacher_hard_gate_r5_literal_canonical_v2` |
| Reward type | `spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2` |
| Stage-A prompt | `spad_teacher_evidence_status_answer_v2` |
| Stage-B prompt | `gold_support_evidence_only_v3` |
| Teacher fallback partial reward | 0.1 |
| Gold token-F1 bonus weight | 0.1 |
| Teacher group post-norm scale | 0.1 |
| GRPO std normalization | 开启 |
| seed / batch / steps | 42 / 64 / 79 |
| 每题 rollout 数 | 8 |

正式训练 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1_overlay.yaml
```

正式训练入口：

```text
tasks/train_tasks/agenticIterRag/run_260716_AIR_spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1.sh
```

### 2.2 Hard-Gate v2 语义

Hard-Gate v2 来自 2026-07-15 的 Teacher PE 冻结结果。Stage A 使用原 Production prompt
判断 S/I/A；仅当 Stage A 为可解析的 S/A 时调用读取 gold 的 Stage B：

```text
Actor EM 组 -> 不调用 Teacher，保持 Actor EM reward

Teacher fallback 组
  -> Stage A Production prompt
     -> I：直接保持 I，不调用 Stage B
     -> S/A：调用 Stage B R5（question + gold + evidence）
             -> Stage B 非 I 且有效：参与 supported answer 择优
             -> Stage B 为 I/格式失败：回退 Stage A
             -> gold 字面值必须真实出现在 evidence 中才能规范化
```

合并规则保证 `Final is I <=> Stage A is I`。Stage B 只能改进非 I 路径的答案内容，不能用
gold 字符串推翻 Production prompt 的证据不足边界。

### 2.3 V3 reward 与 post-norm 保持不变

每个 UID 的 8 条 rollout 仍按原 V3 规则分组：

1. 组内至少一条 Actor EM 命中时，各轨迹使用自身 Actor EM，Teacher 不调用。
2. 整组 Actor EM 全为 0 时，Teacher 的 S/A 状态给 0.1 base。
3. Actor answer 闭合、Teacher 可解析且状态合格时，再加
   `0.1 * token_F1(teacher_answer, gold)`。
4. raw reward 不缩放；GRPO 组内标准化后，Teacher fallback 组 advantage 乘 0.1，Actor EM
   组保持 1.0。

## 3. Reward 独立性与实现边界

Hard-Gate v2 使用独立 reward 模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2.py
```

现有 `spad_em_teacher_backoff`、Gold Token-F1 V1/V2/V3 reward 文件没有被改写。新模块显式
复用原 stable Stage-A batch reward 和原 V3 bonus/post-norm 函数，只在自己的入口内执行
Stage B 与选择逻辑；配置类型不匹配、prompt version 不匹配或 strategy ID 不匹配时立即失败。

新增策略注册与冻结校验位于：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/teacher_strategies.py
```

共享训练路由只新增一个 reward type 分支。半严格 rollout 审计属于 checkpoint finalization
策略，不参与任何 reward 数值计算；它允许核心字段无效轨迹率不超过 0.5%，同时继续严格校验
JSON、记录数、step 数、group 数、每组 rollout 数、shard hash 和 Teacher audit 对齐。

### 3.1 实现验证

本轮重跑以下三组测试：

```text
AgenticIterRag.tests.agent_training.spad.test_rollout_manifest
AgenticIterRag.tests.agent_training.spad.test_teacher_hard_gate_strategy
AgenticIterRag.tests.agent_training.spad.test_teacher_prompt_version_propagation
```

结果为 `24/24` 通过，覆盖 Stage-A I 不可推翻、Stage-B 失败回退、双 supported answer
按 gold token-F1 择优、evidence-literal guard、V3 bonus/post-norm 保持、冻结 strategy 配置、
prompt/strategy 传播，以及 0.5% 边界通过和超界失败。新增 spec、评估 summary、训练 audit 和
eval manifest 均通过 JSON 解析，`git diff --check` 通过。

## 4. 训练执行与全量审计

### 4.1 Run 与时间线

正式训练 run：

```text
260716-005244-008472-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1
```

| 事件 | 北京时间 |
|---|---:|
| Pipeline 启动 | 2026-07-16 00:52:44 |
| VERL 日志创建 | 00:58:10 |
| Step 1 rollout 落盘 | 01:06:48 |
| Step 79 rollout 落盘 | 07:12:25 |
| Raw checkpoint marker=79 | 07:12:43 |
| HF checkpoint 导出完成 | 09:17:21 |
| 半严格 rollout manifest 收尾 | 09:20:51 |
| Pipeline completed manifest | 09:22:42 |

从 Step 1 到 Step 79 rollout 落盘为 6 小时 5 分 37 秒；从 Pipeline 启动到 raw checkpoint
为 6 小时 19 分 59 秒。后续严格审计发现 step 25 的 1 条空 output，按用户要求改为不超过
0.5% 的半严格策略后单独完成审计和 checkpoint finalization，没有重训 79 steps。

### 4.2 训练规模与审计结果

| 审计项 | 结果 |
|---|---:|
| step / rollout 行数 | 79 / 40,448 |
| question group 数 | 5,056 |
| 每组 rollout 数 | 全部为 8 |
| malformed JSON line | 0 |
| 无效轨迹 | 1（0.002472%） |
| 半严格上限 | 0.5% |
| 出现的 post-norm scale | 0.1、1.0 |
| 组内混合 scale | 0 |
| I 边界保持 | 40,448 / 40,448 |

唯一无效轨迹是 step 25 的一条空 output。它被明确计为 invalid trajectory，没有伪造答案，
实际无效率约为上限的 1/202。

### 4.3 训练统计

| 指标 | 全程 | 前 10 step | 后 10 step |
|---|---:|---:|---:|
| raw/final reward | 0.324030 | 0.234554 | 0.330426 |
| Actor EM | 0.289310 | 0.191992 | 0.295117 |
| rollout token-F1 | 0.360548 | 0.273801 | 0.368305 |
| Teacher fallback 组率 | 0.562500 | 0.643750 | 0.553125 |
| mean post-norm scale | 0.493750 | 0.420625 | 0.502187 |
| 平均搜索数 | 1.295837 | 1.360352 | 1.433398 |
| 平均重复查询次数 | 0.131774 | 0.208984 | 0.190234 |

Hard-Gate v2 运行时统计：

| 项目 | 结果 |
|---|---:|
| Teacher fallback rollout | 22,752（2,844 groups） |
| 实际 Teacher-called rollout | 22,711 |
| 因无搜索证据跳过 Teacher | 41 |
| Stage-B 调用 | 8,572（占 Teacher-called 37.7438%） |
| Stage-B 最终采用 | 5,676（占 Stage-B 调用 66.2156%） |
| evidence-literal gold 规范化 | 1,272 |
| 平均 Teacher 调用预算 / Teacher fallback | 1.3774x |
| Stage-A 格式错误 | 14 |
| Stage-B 格式错误 | 2 |
| Teacher 调用异常 | 0 |

正式训练的 `1.3774x` 调用预算与 PE dev 的约 `1.3558x` 接近，低于预设 2x 上限。
`Teacher fallback` 是组级 reward 路径口径；`Teacher-called` 是轨迹实际发出 Stage-A 请求的
口径。两者相差的 41 条轨迹没有 search evidence，沿用 stable reward 的既有规则跳过 Teacher。

### 4.4 与原 postnorm01 的训练曲线对照

| 指标 | 原 postnorm01 | Hard-Gate v2 | 差值 |
|---|---:|---:|---:|
| raw reward | 0.309348 | 0.324030 | +0.014682 |
| Actor EM | 0.280508 | 0.289310 | +0.008802 |
| rollout token-F1 | 0.352092 | 0.360548 | +0.008456 |
| Teacher fallback 组率 | 0.575554 | 0.562500 | -0.013054 |
| mean post-norm scale | 0.482002 | 0.493750 | +0.011748 |
| 平均搜索数 | 1.307185 | 1.295837 | -0.011348 |

这两次是不同训练 run，不能把训练曲线差值当作严格 paired 因果估计；它们用于确认 reward
量级、fallback 比例和搜索行为没有出现异常漂移。

## 5. Checkpoint

最终 HF 模型：

```text
checkpoints/AIR/260716-005244-008472-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79
```

`model.safetensors` 大小为 4,063,515,640 bytes，SHA256：

```text
cc19a3ec9568e243b4c30eb070df30c0f39effbcb40c2d6e38a5bfce7c1f3772
```

Raw VERL checkpoint 保留在同一 run 的
`actor_model_verl/global_step_79`，HF 目录通过 model type、config hash 和 weight hash 校验。

## 6. 3500e 评估

评估任务：

```text
260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1
```

正式调用使用通用评估入口，关键参数如下：

```bash
bash tasks/eval_tasks/agenticIterRag/eval_agent_search.sh \
  --agent-model checkpoints/AIR/260716-005244-008472-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79 \
  --data-path data/global_train_eval_data/3500e/co_search_ablation.eval.parquet \
  --max-samples 3500 \
  --task-name 260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1 \
  --repeat-id 1 \
  --agent-gpu-ids 0,1,2,3,4,5 \
  --agent-instance-count 6 \
  --agent-max-num-seqs 64 \
  --recall-gpu-ids 6,7 \
  --infer-batch-size 384 \
  --flush-every-n 500
```

协议与原 postnorm01 完全一致：同一 3500e parquet、no-ranker、Actor NPU0-5 六副本、Recall
NPU6-7、batch=384、每副本 max_num_seqs=64、temperature=0、top_p=1、topN=50、topM=5、
最多 6 个 assistant turns、完整 trace、每 500 条刷新一次中间产物。

| 项目 | 结果 |
|---|---:|
| 任务启动 | 2026-07-16 09:32:09 |
| 推理 trace 开始写入 | 09:38:09 |
| 自动报告写入 | 09:46:00 |
| 全部服务清理完成 | 09:46:41 |
| 资源加载约用时 | 6 分钟 |
| 纯推理 wall time | 472.0412 秒（7分52秒） |
| 总 wall time | 约 14分32秒 |
| 成功 / 失败 | 3,500 / 0 |
| EM / F1 | 0.2069 / 0.2911 |
| 完整答案率 | 0.8611 |
| 首轮搜索率 | 1.0000 |
| 平均搜索数 | 1.5934 |
| 重复查询率 | 0.1331 |
| Max-turn 率 | 0.1071 |

状态计数：`answered=3014`、`no_valid_answer=111`、`max_turns=375`。评估完成后 Actor、
Recall 服务全部退出，8 张 NPU 无残留进程。

评估身份审计：

| 项目 | SHA256 / fingerprint |
|---|---|
| 3500e parquet | `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283` |
| Model fingerprint | `cbbe6ccf4c5d8f95fd8092fa5be66e078d14162906307a0f020117bdebbb37d3` |
| `model.safetensors` | `cc19a3ec9568e243b4c30eb070df30c0f39effbcb40c2d6e38a5bfce7c1f3772` |
| `metrics.jsonl` | `27eb7dcd2e1a5bec9cabd84d66f498735fab0ef137ddca59c1e9375929c44753` |
| `traces.jsonl` | `68744d3fb7555a2b8b20c92eb16ac6a1cbd01c62ca09a1a89a08b9885e9ac396` |

### 6.1 分数据集结果

| 数据集 | N | EM | F1 | 平均搜索数 |
|---|---:|---:|---:|---:|
| 2Wiki | 563 | 0.1350 | 0.1814 | 1.6252 |
| Bamboogle | 125 | 0.1440 | 0.2244 | 1.2640 |
| HotpotQA | 562 | 0.1993 | 0.2895 | 1.4235 |
| MuSiQue | 562 | 0.0409 | 0.0948 | 2.1263 |
| NQ | 562 | 0.3114 | 0.3987 | 1.2829 |
| PopQA | 563 | 0.3908 | 0.4382 | 1.8792 |
| TriviaQA | 563 | 0.1776 | 0.3584 | 1.2966 |

## 7. 八组横向与相对原 V3 postnorm01 的效果对比

### 7.1 八组横向比较

| 模型组 | Reward / 训练语义 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Search-R1 512 | Actor EM；norm=true；512 | 0.1180 | 0.1965 | 0.6271 | 2.3489 | 0.3640 | 0.2569 |
| Search-R1 5100 | Actor EM；norm=true；5100 | 0.1800 | 0.2509 | 0.7317 | 1.7291 | 0.1786 | 0.1549 |
| SPAD stable 5100 | stable reward；norm=true | 0.1923 | 0.2700 | 0.7220 | 2.6557 | 0.5906 | 0.2443 |
| Gold F1 V1 5100 | V1 eligibility；norm=true | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |
| Gold F1 V2 5100 | V2 eligibility；norm=false | 0.1831 | 0.2673 | 0.7906 | 1.8889 | 0.2154 | 0.1863 |
| V3 postnorm01 5100 | V2 eligibility；norm=true；Teacher x0.1 | 0.1994 | 0.2787 | 0.8340 | 1.6969 | 0.1571 | 0.1369 |
| V3 postnorm03 5100 | V2 eligibility；norm=true；Teacher x0.3 | 0.1929 | 0.2734 | 0.7100 | 2.6883 | 0.5649 | 0.2714 |
| **V3 postnorm01 + Hard-Gate v2 5100** | **V3 postnorm01；Hard-Gate v2 Teacher** | **0.2069** | **0.2911** | **0.8611** | **1.5934** | **0.1331** | **0.1071** |

七个历史对照来自参考报告
`260715-09a_SPAD_GoldTokenF1V3_PostNormScale03_5100训练与3500评估报告.md`；所有组使用同一
3500e 数据和评估协议，每个模型只有一个 checkpoint run。

### 7.2 相对原 postnorm01 的行为指标

| 指标 | 原 postnorm01 | Hard-Gate v2 | 差值 |
|---|---:|---:|---:|
| EM | 0.1994 | 0.2069 | +0.0074 |
| F1 | 0.2787 | 0.2911 | +0.0124 |
| 完整答案率 | 0.8340 | 0.8611 | +0.0271 |
| 首轮搜索率 | 0.9971 | 1.0000 | +0.0029 |
| 平均搜索数 | 1.6969 | 1.5934 | -0.1034 |
| 重复查询率 | 0.1571 | 0.1331 | -0.0240 |
| Max-turn 率 | 0.1369 | 0.1071 | -0.0297 |

原 postnorm01 状态为 answered 2917、no_valid_answer 94、max_turns 479、
multiple_tool_calls 8、direct_answer_before_search 2。本次 answered 增加 97，max_turns 减少
104；no_valid_answer 增加 17，但总体完整答案仍净增 95 条。

### 7.3 相对原 postnorm01 的分数据集差异

| 数据集 | EM 差值 | F1 差值 |
|---|---:|---:|
| 2Wiki | -0.0195 | -0.0134 |
| Bamboogle | +0.0640 | +0.0523 |
| HotpotQA | -0.0196 | -0.0166 |
| MuSiQue | +0.0053 | +0.0141 |
| NQ | +0.0053 | +0.0118 |
| PopQA | +0.0551 | +0.0660 |
| TriviaQA | +0.0053 | +0.0037 |

总体提升主要由 Bamboogle、PopQA 和其余若干数据集的小幅增益贡献；2Wiki 与 HotpotQA
下降，说明 Hard-Gate v2 并非对每个多跳数据集都一致改善。

### 7.4 相对原 postnorm01 的 Paired bootstrap

| 指标 | Hard-Gate v2 - 原 postnorm01 | 95% CI |
|---|---:|---:|
| EM | +0.0074 | [-0.0020, +0.0169] |
| F1 | **+0.0124** | **[+0.0025, +0.0222]** |
| 完整答案率 | **+0.0271** | **[+0.0143, +0.0400]** |

每个模型只有一个训练 checkpoint 和一次 temperature=0 评估。bootstrap 是在相同 3,500 个
问题上配对重采样，不把重复推理当作独立样本，也不能替代多 seed 训练复现。

## 8. 解释、限制与决策建议

Hard-Gate v2 在训练中把约 37.7% 的 Teacher-called 轨迹送入 Stage B，并在约 66.2% 的
Stage-B 调用中采用 Stage-B 结果；同时全部 40,448 条轨迹保持 Stage-A I 边界。训练 raw
reward、Actor EM 和 rollout F1 均高于原 postnorm01，最终 Actor 的 F1、完整答案率、重复
查询和 Max-turn 指标也同向改善，符合“增强非 I 答案信号、不过度鼓励继续搜索”的设计目标。

但应保留三项边界：

1. 当前只有一个训练 seed 和一次 3500e 推理；EM CI 跨 0。
2. 2Wiki 和 HotpotQA 的 EM/F1 下降，需要在多跳切片继续审计。
3. Teacher PE 使用的 holdout 曾参与组合策略诊断；最终泛化结论应以本次未调用 Teacher 的
   3500e Actor 评估为准，而不能把 PE holdout 当作新的未触碰估计。

当前结果支持把 Hard-Gate v2 作为优于原 V3 postnorm01 的候选策略，尤其是 F1、完整答案率
和搜索效率；在晋升为唯一默认配置前，建议至少补一个独立训练 seed，并把 2Wiki/HotpotQA
设为单独门槛。

## 9. 产物索引

- 历史七组参考报告：
  `docs/AgenticIterRag_v1/work_reports/260715-09a_SPAD_GoldTokenF1V3_PostNormScale03_5100训练与3500评估报告.md`
- 历史七组汇总：
  `reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm-scale-ablation-aggregate/summary.json`
- 正式训练 overlay：
  `tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1_overlay.yaml`
- 正式训练入口：
  `tasks/train_tasks/agenticIterRag/run_260716_AIR_spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1.sh`
- 自动评估报告：
  `reports/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1.report.md`
- 评估 trace：
  `log/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1/trace`
- 两模型对比 spec：
  `tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260716_hardgatev2_vs_postnorm01.json`
- 两模型 paired 汇总：
  `reports/eval/agenticIterRag/260716-newdata3500-spad-5100-hardgatev2-vs-postnorm01-aggregate/summary.json`
- 两模型自动对比报告：
  `reports/eval/agenticIterRag/260716-newdata3500-spad-5100-hardgatev2-vs-postnorm01-aggregate/report.md`
- 训练全量审计：
  `reports/eval/agenticIterRag/260716-newdata3500-spad-5100-hardgatev2-vs-postnorm01-aggregate/training_audit.json`
- 训练曲线：
  `reports/eval/agenticIterRag/260716-newdata3500-spad-5100-hardgatev2-vs-postnorm01-aggregate/training_curve.csv`

# SPAD Gold Token-F1 V3：Teacher 组 Post-Norm 0.3、5100 Stage1 训练与 3500 评估报告

日期：2026-07-15，09am（北京时间）

> 状态：V3 post-norm scale=0.3 的独立 5100 Stage1 训练、最终 checkpoint 导出、3500e
> 评估、历史组横向比较和训练 rollout 审计均已完成。64 条 smoke 训练产生的临时日志与
> checkpoint 已删除。本实验只执行 Stage1，未执行 Stage2 或 Stage3。

## 1. 结论

本轮只在 `spad_em_teacher_backoff_gold_token_f1_bonus_v3` 的基础上，将
`teacher_group_postnorm_scale` 从 0.1 改为 0.3；raw reward、数据、seed、rollout 数和
GRPO 归一化配置保持一致。

在同一 3500e 数据上，postnorm03 的 EM/F1 与 postnorm01 没有显著差异，但完整答案率和
搜索行为明显恶化：

| 模型 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|
| V3 postnorm01 | **0.1994** | **0.2787** | **0.8340** | **1.6969** | **0.1571** | **0.1369** |
| V3 postnorm03 | 0.1929 | 0.2734 | 0.7100 | 2.6883 | 0.5649 | 0.2714 |
| postnorm03 - postnorm01 | -0.0066 | -0.0053 | **-0.1240** | **+0.9914** | **+0.4078** | **+0.1345** |

postnorm03 因为把 Teacher fallback 组的梯度权重提高到 postnorm01 的 3 倍，训练中的 raw
reward 曲线并没有显著变化，但推理策略转向更长、更重复的搜索。当前不建议把 0.3 晋升为默认
策略，postnorm01 仍是本轮更好的工作点。

## 2. Reward 与实现语义

### 2.1 Raw reward

每题 8 条 rollout 按 UID 分组：

1. 组内至少一条 Actor EM 命中 gold 时，每条轨迹使用自身 Actor EM（1 或 0）。
2. 若整组 Actor EM 全为 0，则调用 Teacher。Teacher 状态为 supported 或 ambiguous 时，
   给 0.1 base，否则为 0。
3. 合法 `<answer>...</answer>`、Teacher 可解析且状态合格时，增加
   `0.1 * token_F1(teacher_answer, gold)` bonus。

V3 不缩放 raw reward，缩放只作用于 GRPO 的组内标准化之后：

```text
z_i = (r_i - group_mean) / (group_std + 1e-6)
A_i = z_i       , Actor EM 组
A_i = 0.3 * z_i , Teacher fallback 组
```

同一 UID 的 8 条 rollout 共享一个 scale，Actor EM 组为 1.0，Teacher fallback 组为 0.3。
因此 0.3 是困难组的梯度权重，不是把 Teacher 的 token-F1 reward 直接乘 0.3。

### 2.2 代码与配置

- Reward：`AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus_v3.py`
- GRPO post-norm：`AgenticIterRag/verl/verl/trainer/ppo/core_algos.py`
- 训练路由：`AgenticIterRag/verl/verl/trainer/ppo/ray_trainer.py`、
  `AgenticIterRag/agentic_iter_rag/agent_training/spad/search_policy_rl.py`
- 正式 overlay：
  `tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm03_stage1_overlay.yaml`
- 正式入口：
  `tasks/train_tasks/agenticIterRag/run_260715_AIR_spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm03_stage1.sh`

关键配置：

```yaml
reward.type: spad_em_teacher_backoff_gold_token_f1_bonus_v3
teacher_group_postnorm_scale: 0.3
algorithm.norm_adv_by_std_in_grpo: true
algorithm.group_postnorm_advantage_scale_key: advantage_postnorm_scale
```

## 3. 训练执行与审计

### 3.1 Run 与时间

正式 run：

```text
260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1
```

| 事件 | 北京时间 |
|---|---:|
| 训练命令启动 | 2026-07-15 00:59:06 |
| 第一个 step 开始计算 | 约 01:08 |
| Step 1 rollout 文件落盘 | 01:13:02 |
| Step 79 rollout 文件落盘 | 07:21:37 |
| 最终 checkpoint marker=79 | 07:21:52 |
| HF safetensors 导出完成 | 08:59:51 |

训练从命令启动到最终 checkpoint 共 6 小时 22 分 46 秒。`verl_train.log` 在 step 22 后
停止刷新的原因是日志捕获不完整，不代表训练停止；79 个 rollout 文件和最终 checkpoint
marker 连续存在，训练实际正常完成。

### 3.2 配置与规模

| 项目 | 值 |
|---|---|
| 初始模型 | Qwen3-1.7B Base |
| 训练数据 | `data/global_train_eval_data/5100t/co_search_ablation.train.parquet` |
| max samples / 实际 prompt slots | 5100 / 79 x 64 = 5056 |
| 每题 rollout / 总 rollout | 8 / 40448 |
| seed / batch / steps | 42 / 64 / 79 |
| Teacher fallback post-norm scale | 0.3 |
| Stage2 / Stage3 | 关闭 / 关闭 |

### 3.3 Rollout 全量审计

从 79 个 rollout JSONL 文件重建训练曲线，得到：

| 审计项 | 结果 |
|---|---:|
| step / rollout 行数 | 79 / 40448 |
| UID 组数 | 5056 |
| 每组 rollout 数 | 全部为 8 |
| Teacher fallback rollout | 23312（0.576345） |
| Actor EM rollout | 17136 |
| 出现的 scale | 0.3、1.0 |
| 组内混合 scale | 0 |
| malformed JSON line | 0 |

全量训练统计：

| 指标 | 全程 | 前 10 step | 后 10 step |
|---|---:|---:|---:|
| raw reward | 0.312153 | 0.217767 | 0.328721 |
| Actor EM | 0.281646 | 0.180469 | 0.297266 |
| rollout token-F1 | 0.352392 | 0.266631 | 0.370605 |
| Teacher fallback rate | 0.576345 | 0.659375 | 0.573438 |
| mean post-norm scale | 0.596559 | 0.538438 | 0.598594 |
| 平均搜索次数 | 1.360537 | 1.388672 | 1.562109 |
| 平均重复查询次数 | 0.138425 | 未记录 | 0.229492 |

`mean(postnorm_scale) = 1 - 0.7 * fallback_rate`，与每组 scale 审计一致。按来源拆分，
Teacher fallback 组占 post-scale absolute advantage 约 21.5%，Actor EM 组约 78.5%；相比
postnorm01，Teacher 组得到约 3 倍的相对梯度权重。

训练曲线文件：

```text
reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm-scale-ablation-aggregate/training_curve.csv
```

### 3.4 与 postnorm01 的训练曲线对照

| 指标 | postnorm01 | postnorm03 | 差值 |
|---|---:|---:|---:|
| raw reward | 0.309348 | 0.312153 | +0.002805 |
| Actor EM | 0.280508 | 0.281646 | +0.001138 |
| rollout token-F1 | 0.352092 | 0.352392 | +0.000300 |
| Teacher fallback rate | 0.575554 | 0.576345 | +0.000791 |
| mean post-norm scale | 0.482002 | 0.596559 | +0.114557 |
| 平均搜索次数 | 1.307185 | 1.360537 | +0.053352 |

这说明本次消融的主要变化确实是 advantage 权重，而不是 raw reward 定义或 Teacher fallback
比例；最终推理行为变化不能用 raw reward 上升解释。

## 4. Checkpoint 导出

训练只生成 VERL FSDP checkpoint，因此为评估额外导出 HF safetensors。最终模型：

```text
checkpoints/AIR/260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/global_step_79/hf_safetensors/actor
```

`model.safetensors` 大小为 4,063,515,640 bytes，SHA256：

```text
94d9b6b7b24e43080216e9a8a881b5e5929a860025e247a4a37694758251fe19
```

导出第一次使用基础 Python 环境时因缺少 `tensordict` 失败，随后使用仓库兼容 Python overlay
重试成功。该异常增加了约 1 小时 37 分钟的训练完成后等待，不改变训练结果。

## 5. 3500e 评估

后补的单模型复现入口（原始评估未保存独立 wrapper）：

```text
tasks/eval_tasks/agenticIterRag/run_260715_spad_5100_gold_token_f1_v3_postnorm03_3500eval.sh
```

评估任务：

```text
260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1
```

协议与上一份 postnorm01 报告一致：同一 3500e parquet、no-ranker、Actor NPU0-5 六副本、
Recall NPU6-7、temperature=0、top_p=1、topN=50、topM=5、最多 6 个 assistant turns、
完整 trace。

| 项目 | 结果 |
|---|---:|
| 评估启动 manifest | 09:03:06 |
| 评估报告写入 | 09:22:26 |
| 成功 / 失败 | 3500 / 0 |
| 推理 wall time | 798.4587 秒（13分18.5秒） |
| 总运行 wall time（含加载和清理） | 约 19分20秒 |
| EM / F1 | 0.1929 / 0.2734 |
| 完整答案率 | 0.7100 |
| 首轮搜索率 | 0.9926 |
| 平均搜索数 | 2.6883 |
| 重复查询率 | 0.5649 |
| Max-turn 率 | 0.2714 |

状态计数：answered 2483、no_valid_answer 41、max_turns 950、multiple_tool_calls 24、
direct_answer_before_search 2。

分数据集结果：

| 数据集 | N | EM | F1 | 平均搜索数 |
|---|---:|---:|---:|---:|
| 2Wiki | 563 | 0.1119 | 0.1618 | 2.7815 |
| Bamboogle | 125 | 0.2160 | 0.3031 | 2.4400 |
| HotpotQA | 562 | 0.2242 | 0.3089 | 2.6032 |
| MuSiQue | 562 | 0.0569 | 0.1124 | 3.1779 |
| NQ | 562 | 0.2954 | 0.3825 | 2.3541 |
| PopQA | 563 | 0.2860 | 0.3202 | 2.7957 |
| TriviaQA | 563 | 0.1776 | 0.3478 | 2.4725 |

评估完成后 Actor、Recall、Teacher 服务均已退出，8 张 NPU 释放。

## 6. 七组横向比较

所有组均使用同一 3500e 数据和同一评估协议；每个模型只有一个 checkpoint run，因此表中不
把重复推理当作独立样本。

| 模型组 | Reward/训练语义 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Search-R1 512 | Actor EM；norm=true；512 | 0.1180 | 0.1965 | 0.6271 | 2.3489 | 0.3640 | 0.2569 |
| Search-R1 5100 | Actor EM；norm=true；5100 | 0.1800 | 0.2509 | 0.7317 | 1.7291 | 0.1786 | 0.1549 |
| stable 5100 | stable reward；norm=true | 0.1923 | 0.2700 | 0.7220 | 2.6557 | 0.5906 | 0.2443 |
| Gold F1 V1 5100 | V1 eligibility；norm=true | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |
| Gold F1 V2 5100 | V2 eligibility；norm=false | 0.1831 | 0.2673 | 0.7906 | 1.8889 | 0.2154 | 0.1863 |
| **V3 postnorm01 5100** | **V2 eligibility；norm=true；Teacher x0.1** | **0.1994** | **0.2787** | **0.8340** | **1.6969** | **0.1571** | **0.1369** |
| **V3 postnorm03 5100** | **V2 eligibility；norm=true；Teacher x0.3** | **0.1929** | **0.2734** | **0.7100** | **2.6883** | **0.5649** | **0.2714** |

相对 postnorm01 的 paired bootstrap（10000 次，seed=42）：

| 指标 | postnorm03 - postnorm01 | 95% CI |
|---|---:|---:|
| EM | -0.0066 | [-0.0171, 0.0043] |
| F1 | -0.0053 | [-0.0167, 0.0062] |
| 完整答案率 | **-0.1240** | **[-0.1403, -0.1080]** |

EM 和 F1 区间跨 0，完整答案率显著下降。相对其他组的 paired 结果保存在：

```text
reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm-scale-ablation-aggregate/summary.json
```

## 7. 行为解释与后续建议

postnorm03 的 raw reward、Actor EM 和 token-F1 训练曲线基本不变，说明 0.3 没有改变 reward
定义本身。变化发生在优化权重：Teacher fallback 组从 0.1 提到 0.3 后，困难组获得更强的
更新，最终策略更倾向继续搜索；这与平均搜索数、重复查询率和 Max-turn 率同时上升相符。

但这只是单个训练 seed 的行为证据，不能把因果结论写成跨 seed 定律。当前决策建议：

1. 默认继续使用 V3 postnorm01，不采用 postnorm03 作为生产训练配置。
2. 若继续做消融，优先测试 0.15、0.2 或按 fallback 难度动态裁剪 scale，并把完整答案率、
   Max-turn 率、重复查询率和按数据集 EM/F1 设为与 EM/F1 同等重要的门槛。
3. 下一轮至少复现两个 seed，并按 `advantage_source` 记录组级 mean(abs(advantage))、
   policy-gradient norm 和每类问题的行为变化，避免只比较 raw reward。

## 8. 产物索引

- 后补的单模型复现入口：
  `tasks/eval_tasks/agenticIterRag/run_260715_spad_5100_gold_token_f1_v3_postnorm03_3500eval.sh`
- 自动评估报告：
  `reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1.report.md`
- 七组汇总报告：
  `reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm-scale-ablation-aggregate/report.md`
- 训练曲线：
  `reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm-scale-ablation-aggregate/training_curve.csv`
- 评估 trace：
  `log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1/trace`
- 评估 run spec：
  `tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260715_gold_token_f1_v3_postnorm_scale_ablation.json`

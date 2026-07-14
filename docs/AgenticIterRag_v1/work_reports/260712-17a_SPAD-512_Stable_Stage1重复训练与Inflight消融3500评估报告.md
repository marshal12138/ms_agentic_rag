# SPAD-512 Stable Stage1 重复训练与 Inflight 消融 3500 评估报告

日期：2026-07-12

## 1. 结论

本轮完成了两次 SPAD-512 Stage1 训练和各一次 3500 条评估：先按历史 stable 配置重复
`stream_group_max_inflight=1`，再只把该参数改为 `2` 做消融。两次训练均完成 8/8 step，
两次评估均为 3500/3500 成功、失败 0。

| 实验 | Inflight | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 |
|---|---:|---:|---:|---:|---:|---:|
| 历史 stable | 1 | 0.1360 | 0.2265 | 0.6989 | 2.3391 | 0.3340 |
| 本次重复 | 1 | 0.1051 | 0.1737 | 0.5431 | 2.5257 | 0.4326 |
| 本次消融 | 2 | 0.1054 | 0.1798 | 0.5900 | 2.3566 | 0.3466 |

`inflight=2` 相对本次 `inflight=1`：

- EM `+0.0003`，95% CI `[-0.0066, 0.0071]`，不能确认有提升。
- F1 `+0.0061`，95% CI `[-0.0007, 0.0131]`，不能确认有提升。
- 完整答案率 `+0.0469`，95% CI `[0.0334, 0.0603]`，在本次两个固定 checkpoint 的
  逐题比较中显著提高。
- 8 step 的 `timing_s/step` 总和由 2992.39 秒降至 2406.68 秒，缩短 19.6%，等价于
  训练阶段约 1.24 倍吞吐。

最重要的解释边界是：训练数据和每一步的问题集合确实一致，也固定了数据 seed；但 GRPO actor
rollout 使用随机采样，异步执行也没有做到逐请求固定随机种子。因此本轮不是位级确定性复现。
`inflight=2` 的明确收益是训练加速；一次对一次实验不足以把 EM/F1 或完整答案率差异严格归因于
inflight 参数。

## 2. 实验、checkpoint 与关键差别

| 实验 | 训练 run | 关键参数 | checkpoint |
|---|---|---|---|
| 历史 stable | `260711-103304-616277-...newdata_512` | `inflight=1` | `checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` |
| 本次 stable 重复 | `260712-131305-696244-...stable_stage1_repeat` | `inflight=1` | `checkpoints/AIR/260712-131305-696244-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_repeat/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` |
| 本次 inflight 消融 | `260712-143738-025140-...inflight2_ablation` | **仅将 `inflight` 从 1 改为 2** | `checkpoints/AIR/260712-143738-025140-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_inflight2_ablation/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` |

本次两组的共同训练条件：Qwen3-1.7B Base、512 条训练数据、8 step、每 step 64 个问题、每题
8 条 rollout、GRPO、reward=`spad_em_teacher_backoff`、data seed 42。消融 overlay 的配置差异只有
`stream_group_max_inflight: 1 -> 2`。

历史 stable 是较早且效果更好的 `260711-103304-616277`，不是后来的
`...em_teacher_backoff_dev` 训练。历史 run 与本次 run 相隔一天，无法证明使用了字节级完全一致的
代码工作区；因此它适合作为历史参照，不应当作严格可复现的同配置 repeat。

作为背景，SPAD-5100 Stage1 的 checkpoint 是：

```text
checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79
```

它也使用 `inflight=2`，3500 评估为 EM 0.1923、F1 0.2700、完整答案率 0.7220；但它有
5100 条数据和 79 step，不能作为 inflight 的严格消融证据。

## 3. Teacher 推理参数

历史 stable、本次 `inflight=1`、本次 `inflight=2` 与 SPAD-5100 的 Teacher 关键参数一致：

| 参数 | 值 |
|---|---|
| Teacher | `GLM-4.7-Flash` |
| 设备 / TP | NPU 4-5 / TP=2 |
| dtype | BF16 |
| vLLM max model length | 32000 |
| temperature / top_p | 0 / 1 |
| max_tokens / timeout | 512 / 180 秒 |
| thinking | false |
| batch workers | 16 |
| reward | `spad_em_teacher_backoff` |

关键消融参数只有 `stream_group_max_inflight`：历史/本次 512 stable 为 1，SPAD-5100 和本次
inflight 消融为 2。它控制每个 agent-loop worker 同时向 Teacher 推进的 UID group 数量，会改变
请求排队和流式回传调度，不改变单次 Teacher 的采样参数或 reward 公式。

## 4. 训练数据与随机性审计

三次 512 训练均使用：

```text
data/global_train_eval_data/512t/co_search_ablation.train.parquet
SHA256: 2f9eb86fb40fbb69fab2aca7f6a4e4a05d6879e6dbbcd0fbe1d73e1a1a010558
```

已直接解析三次训练的 24 份 rollout JSONL：每个 run 都是 8 step，每 step 512 条 rollout，
对应 64 个唯一问题且每题恰好重复 8 次。逐 step 比较后，历史 stable、本次 `inflight=1` 和
本次 `inflight=2` 的问题 multiset 在 8/8 step 全部相等。这排除了训练样本或 batch 划分不同。

已限制的随机性：

- 数据 shuffle 开启，`data_seed=42`，所以每一步取到的问题集合一致。
- Teacher 使用 `temperature=0`、`top_p=1`。
- 3500 评估使用同一数据、`temperature=0`、同一检索协议。

未完全限制的随机性：

- actor rollout 使用 `temperature=1`、`top_p=1`，每个问题会采样 8 条轨迹。
- Ray/vLLM 异步 worker 的请求到达和完成顺序不同。三次训练的逐 step 问题集合相同，但 JSONL
  完成顺序并不相同。
- 服务 seed 不能替代“每个样本绑定固定 seed”；调度变化会改变随机数消费顺序。
- 每个 checkpoint 只做了一次 3500 推理，本报告的 bootstrap CI 是对 3500 个问题重采样，
  不包含重新训练产生的方差。

所以准确说法是：数据、batch 和主要 seed 已控制，但训练轨迹没有完全确定。若要对 inflight 做
因果结论，应至少为每个设置做多个训练 seed；更强方案是为每个 `question + rollout_index + step`
绑定固定生成 seed，并固定软件版本和服务调度条件。

## 5. 训练 reward 与耗时

`Reward` 为日志中的 `critic/score/mean`，`Actor EM` 为 `reward_extra/em_reward/mean`。

| 实验 | 平均 Reward | 平均 Actor EM | 最后 3 步 Reward | 最后 3 步 Actor EM | 单 step 平均 | 8 step 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 历史 stable, inflight=1 | 0.1545 | 0.1257 | 0.1779 | 0.1504 | 373.85 秒 | 2990.81 秒 |
| 本次重复, inflight=1 | 0.1558 | 0.1289 | 0.1810 | 0.1556 | 374.05 秒 | 2992.39 秒 |
| 本次消融, inflight=2 | 0.1590 | 0.1311 | 0.1884 | 0.1628 | 300.83 秒 | 2406.68 秒 |

| Step | Repeat-1 Reward | Repeat-1 EM | Repeat-1 秒 | Inflight-2 Reward | Inflight-2 EM | Inflight-2 秒 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1318 | 0.1074 | 372.34 | 0.1256 | 0.0996 | 290.24 |
| 2 | 0.0963 | 0.0723 | 403.09 | 0.1047 | 0.0801 | 313.84 |
| 3 | 0.1043 | 0.0703 | 340.36 | 0.1473 | 0.1172 | 278.01 |
| 4 | 0.1861 | 0.1621 | 361.87 | 0.1645 | 0.1328 | 291.33 |
| 5 | 0.1846 | 0.1523 | 374.38 | 0.1650 | 0.1309 | 358.74 |
| 6 | 0.1789 | 0.1543 | 387.60 | 0.2014 | 0.1758 | 293.96 |
| 7 | 0.1703 | 0.1426 | 401.27 | 0.1688 | 0.1426 | 282.02 |
| 8 | 0.1937 | 0.1699 | 351.49 | 0.1951 | 0.1699 | 298.54 |

三次训练的平均训练 reward 很接近，但历史 checkpoint 的 3500 F1 明显更高。这说明 512 条训练上
的 rollout reward 不能可靠预测最终 3500 泛化结果，且单次训练噪声不可忽略。

## 6. 3500 评估与配对比较

评估数据：

```text
data/global_train_eval_data/3500e/co_search_ablation.eval.parquet
SHA256: bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283
```

统一协议为 no-ranker、Recall Top N=50、模型可见 Top M=5、最多 6 轮 assistant、
temperature=0、top_p=1。使用 10000 次 paired bootstrap，seed 42。

| 比较（右减左） | EM 差值及 95% CI | F1 差值及 95% CI | 完整答案率差值及 95% CI |
|---|---:|---:|---:|
| 历史 stable -> 本次 inflight=1 | -0.0309 [-0.0403, -0.0217] | -0.0528 [-0.0626, -0.0431] | -0.1557 [-0.1734, -0.1383] |
| 本次 inflight=1 -> 本次 inflight=2 | +0.0003 [-0.0066, 0.0071] | +0.0061 [-0.0007, 0.0131] | +0.0469 [0.0334, 0.0603] |
| 历史 stable -> 本次 inflight=2 | -0.0306 [-0.0394, -0.0217] | -0.0467 [-0.0559, -0.0372] | -0.1089 [-0.1263, -0.0914] |

历史 stable 与本次同为 `inflight=1`，但效果差异显著。这里的“显著”只表示两个已训练模型在
这 3500 个问题上的差异稳定，不表示同配置训练的总体期望不同；后一个问题需要多个独立训练 run。

分数据源 F1：

| 数据源 | 历史 stable | 本次 inflight=1 | 本次 inflight=2 |
|---|---:|---:|---:|
| 2WikiMultiHopQA | 0.1107 | 0.0997 | 0.0905 |
| Bamboogle | 0.2485 | 0.1967 | 0.1742 |
| HotpotQA | 0.2443 | 0.1793 | 0.1879 |
| MuSiQue | 0.0967 | 0.0668 | 0.0656 |
| NQ | 0.2966 | 0.2137 | 0.2179 |
| PopQA | 0.2831 | 0.2407 | 0.2587 |
| TriviaQA | 0.3227 | 0.2370 | 0.2593 |

## 7. 产物位置

本次训练入口和 overlay：

```text
tasks/train_tasks/agenticIterRag/run_260712_AIR_spad_qwen3_1_7b_glm47_512_stable_stage1_repeat.sh
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_512_stable_stage1_repeat_overlay.yaml
tasks/train_tasks/agenticIterRag/run_260712_AIR_spad_qwen3_1_7b_glm47_512_stable_stage1_inflight2_ablation.sh
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_512_stable_stage1_inflight2_ablation_overlay.yaml
```

单次评估报告：

```text
reports/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-run1.report.md
reports/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-repeat-inflight1-run1.report.md
reports/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-inflight2-ablation-run1.report.md
```

消融聚合：

```text
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260712_spad512_inflight_ablation.json
reports/eval/agenticIterRag/260712-spad512-inflight-ablation-aggregate/report.md
reports/eval/agenticIterRag/260712-spad512-inflight-ablation-aggregate/summary.json
```

## 8. 后续消融必须保留的提醒

原 512 stable 和本次 stable repeat 的 `stream_group_max_inflight=1`；SPAD-5100 和本次消融为
`stream_group_max_inflight=2`。除此之外 Teacher 模型、TP、采样参数、长度、超时、batch workers
和 reward 公式一致。后续消融应只改变这一项，同时固定训练 parquet、batch 划分、初始 checkpoint、
代码版本和其余配置，并用多个独立训练 seed 或逐样本固定生成 seed 报告均值和方差。

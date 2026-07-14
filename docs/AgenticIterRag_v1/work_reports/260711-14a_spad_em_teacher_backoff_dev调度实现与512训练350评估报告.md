# spad_em_teacher_backoff_dev调度实现与512训练350评估报告

日期：2026-07-11

## 1. 结论

本轮完成了独立reward类型`spad_em_teacher_backoff_dev`的实现、512条数据的SPAD
Stage1训练、4096条rollout审计，以及一次350条正式评估。

结论分成两部分：

1. 调度提速成立。训练墙钟从稳定版`50:08`降到`34:17`，节省`15:51`
   （`31.6%`）。8步平均`timing_s/gen`从`266.1s`降到`144.3s`（降低`45.8%`），
   平均`timing_s/step`从`373.9s`降到`255.0s`（降低`31.8%`）。
2. reward语义验收通过，但这次训练出的模型效果没有复现稳定版水平。dev单次评估
   `EM=0.1200`、`F1=0.1882`、完整答案率`0.6457`；稳定版三次均值分别为
   `0.1314/0.2251/0.7010`。因此当前dev实现可以保留为实验分支，但不能仅凭本轮结果
   替换稳定版正式配置。

这里的“不一致”不是已经发现reward公式错误。历史稳定版4096条rollout重放时，dev与
稳定版关键reward字段0差异；本轮dev的4096条正式rollout审计也为PASS、公式错误0。
更直接的差异是训练采样轨迹：稳定版和dev虽然都得到515条`EM=1` rollout，但完整答案
分别为3012和2943条。Stage1训练使用`temperature=1.0`，异步调度改变请求完成顺序后，
不能期待采样轨迹和最终checkpoint逐位一致。

## 2. 实现边界

新增独立模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_dev.py
```

调度流程为：

```text
单条完整rollout结束
  -> 在per-rollout reward loop中异步预取teacher结果
  -> 同UID的8条rollout齐备
  -> batch reward按原EM/backoff规则决定是否采用预取结果
```

具体规则保持为：同UID内任一rollout的actor答案EM为1时，整组不使用teacher reward，
每条最终reward等于自身EM；整组EM全零时，有证据的rollout使用
`0.1 * teacher_status_reward`，无证据为0。

为避免重复请求，完全相同的teacher request hash在同一reward worker内合并执行。teacher
prompt不包含actor答案，只由question、evidence和请求参数构成，因此合并不改变reward
输入语义。组裁决后仍保留实际采用的teacher审计字段。

稳定版`search_policy_teacher_reward.py`没有被本轮实现修改。共享框架只增加了dev门控能力：

- reward loop可单独选择`reward_loop_manager=naive`，最终组reward仍使用`batch`。
- 单条预取结果带`spad_dev_prefetch_only`标记，不会提前写成最终reward。
- batch manager透传`spad_dev_prefetched_teacher_detail`。
- dev路由使用独立reward模块、入口函数和overlay；稳定版路由维持原值。

## 3. 验证

- 仓库`unittest discover`：`68/68`通过。
- dev路由dry-run：确认只运行Stage1、8 step、`batch + naive reward loop`、闭合
  `</answer>`停止条件和stream group reward。
- 稳定版4096条历史rollout重放：score、actor answer/parse、EM、teacher调用与状态、
  all-zero组、partial应用等关键字段差异均为0。
- dev正式rollout审计：PASS，4096条、512组、公式错误0、shard哈希一致。

dev正式rollout摘要：

| Rollouts | Groups | EM=1 | Complete answer | Evidence | Teacher calls | All-zero groups | Teacher format errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 512 | 515 | 2943 | 4070 | 2967 | 373 | 3 |

稳定版对应值为4096、512、515、3012、4068、2995、377、4。两轮训练的8步平均
reward分别为`0.1545`和`0.1528`，非常接近，但rollout并非同一批采样结果。

审计产物：

```text
reports/eval/agenticIterRag/260711-spad512-stage1-em-teacher-backoff-dev-rollout-audit/
```

## 4. 训练速度

稳定版run：

```text
260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512
```

dev run：

```text
260711-220950-337984-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_em_teacher_backoff_dev
```

| Step | Stable gen s | Dev gen s | Gen降低 | Stable step s | Dev step s | Step降低 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 277.1 | 215.2 | 22.3% | 379.7 | 318.7 | 16.1% |
| 2 | 273.4 | 139.7 | 48.9% | 381.6 | 255.3 | 33.1% |
| 3 | 245.9 | 158.8 | 35.4% | 349.2 | 270.1 | 22.7% |
| 4 | 259.6 | 121.3 | 53.3% | 369.7 | 228.0 | 38.3% |
| 5 | 278.9 | 137.5 | 50.7% | 388.6 | 244.2 | 37.2% |
| 6 | 268.3 | 117.2 | 56.3% | 374.8 | 233.0 | 37.8% |
| 7 | 286.8 | 124.6 | 56.5% | 398.7 | 236.5 | 40.7% |
| 8 | 239.0 | 139.8 | 41.5% | 348.5 | 254.0 | 27.1% |
| 平均 | 266.1 | 144.3 | 45.8% | 373.9 | 255.0 | 31.8% |

训练step累计时间从`2990.8s`降到`2039.9s`，节省`951.0s`；与进度条墙钟节省
`951s`一致。最终HF checkpoint：

```text
checkpoints/AIR/260711-220950-337984-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_em_teacher_backoff_dev/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

## 5. 350评估

dev正式评估：

```text
260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1
```

评估使用与稳定版相同的350e数据、no-ranker、Recall Top N=50、模型可见Top M=5、
最多6轮assistant、`temperature=0.0`和`top_p=1.0`。350/350成功，失败0。

| 模型/口径 | EM | F1 | 完整答案率 | 平均工具调用 | no_valid_answer率 |
|---|---:|---:|---:|---:|---:|
| Stable run1 | 0.1286 | 0.2201 | 0.7057 | 2.4314 | 0.0314 |
| Stable 3-repeat均值 | 0.1314 | 0.2251 | 0.7010 | 2.4505 | 0.0286 |
| Dev run1 | 0.1200 | 0.1882 | 0.6457 | 2.3343 | 0.1371 |

dev与stable run1逐样本配对结果：

- EM差`-0.0086`，95% bootstrap CI `[-0.0400, 0.0229]`；dev独赢15题，
  stable独赢18题，27题都对。
- F1差`-0.0319`，95% bootstrap CI `[-0.0647, 0.0009]`；dev较高47题，
  stable较高57题。
- 两个区间都包含0，单次评估不足以统计确认退化；但F1、完整答案率和
  `no_valid_answer`三项同向变差，工程上不能判定“效果一致”。

分数据集上，dev相对stable三次均值在2WikiMultiHopQA和HotpotQA更高，在Bamboogle、
MuSiQue、NQ、PopQA和TriviaQA更低；下降不是单一数据集噪声。

评估报告：

```text
reports/eval/agenticIterRag/260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1.report.md
```

## 6. 使用判断

当前可以确认：

1. 快速调度实现有效，且没有改动稳定版reward策略文件。
2. dev最终reward公式和稳定版一致，历史重放与正式审计均通过。
3. 不同调度下的`temperature=1.0`训练并不产生同一组采样轨迹；reward语义一致不等于
   单个checkpoint效果必然一致。
4. 本轮dev checkpoint的350评估低于稳定版，因此不应直接把dev设为正式默认值。

若后续要决定是否晋升dev，最低限度应再做独立训练seed重复，而不是只重复评估同一个
checkpoint。若目标是严格对照调度本身，还需要给每个UID/rollout固定独立采样seed，避免
请求调度顺序改变随机数分配；这属于新的训练可复现性改造，不应混入本轮已验收的reward
调度代码。

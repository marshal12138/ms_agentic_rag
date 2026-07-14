# SPAD / Search-R1 Stage1 Reward、训练与评估交接 Prework

记录时间：2026-07-13，北京时间

工作目录：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives`

本文用于在对话、终端或人员中断后恢复当前 AgenticIterRag v1 的 Search-R1 / SPAD 工作。它重点记录
2026-07-11 至 2026-07-13 已经完成的训练、评估、reward 消融、推理加速和最新代码状态，并明确下一位
接手者应如何继续。本文不是新的结果宣称；涉及具体统计结论时，应以文中链接的正式 work report、
aggregate `summary.json` 和原始 trace 为准。

## 1. 接手前必须先读的结论

1. 当前最可靠的 Stage1 主线仍是 `spad_em_teacher_backoff`。在 3500 条同口径评估上，历史
   SPAD-512 stable 和 SPAD-5100 stable 均显著高于同规模 Search-R1 original 的 EM/F1。
2. 新 reward `spad_em_teacher_backoff_gold_token_f1_bonus` 已经训练过 512 和 5100，但这些 checkpoint
   使用的是初版 V1 eligibility。V1 在一次较弱的 512 同调度复训对照上提高，但在 5100 上相对 stable
   F1 和完整答案率下降，不能替换 stable。
3. 同名 reward 的当前代码已经更新为 V2 eligibility：只有 Actor 合法闭合答案，且 Teacher status 为
   `supported_answer` 或 `ambiguous_evidence` 时才发 extra bonus。V2 代码已经通过定向测试，但尚未进行
   任何正式训练或评估。
4. `spad_em_teacher_backoff` 的 `partial_reward=0.1` 在当前 GRPO 设置下并没有实现“梯度只有 EM 的
   十分之一”。所有既有 Stage1 实验都使用 `norm_adv_by_std_in_grpo=true`，组内标准化会基本消除
   `0.1` 与 `1.0` 的整体尺度差异。这是下一阶段最重要的 advantage 消融点。
5. `stream_group_max_inflight=2` 相对 `1` 的明确收益是训练提速约 19.6%；一次训练对一次训练没有证明
   它提高或降低最终准确率。后续效果消融必须把 inflight 固定，不应再与 reward/advantage 同时变化。
6. SPAD 默认配置已经改为只执行 Stage1：`stop_after_sub_stage: search_policy_rl`。除非实验明确要求，
   不要默认启动 Stage2/Stage3。
7. 记录本文时训练和评估任务均已退出，8 张 NPU 已释放。重新开始前仍应重新执行资源检查，不要假定
   该状态永久不变。
8. 当前工作树包含大量未提交和未跟踪的已有实现、报告、配置及实验产物。不要使用 `git reset --hard`
   或批量 checkout；续作必须在现状上增量修改，并先检查目标文件的已有 diff。

## 2. 名词和解释边界

### 2.1 训练规模

- `512`：8 step，每 step 64 个问题，每题 8 条 rollout；实际 512 个 prompt group、4096 条 rollout。
- `5100`：配置 `train_max_samples=5100`，但全局 batch 为 64，实际执行 79 个完整 step，即 5056 个
  prompt group、40448 条 rollout。最后不足一个 batch 的 44 条没有进入优化。
- 因此文档中的“5100 模型”是实验配置名。精确描述实际训练量时必须写 5056 个 prompt group。

### 2.2 评估 repeat 和训练 seed

- 350 数据上的三次 repeat 是同一个 checkpoint 的三次独立推理，不是三次独立训练。
- 3500 数据上的正式比较通常每个 checkpoint 只推理一次；推理使用 `temperature=0`。
- paired bootstrap 对问题做重采样，只表示两个既定 checkpoint 在评估问题上的不确定性，不包含训练
  随机性。不能根据置信区间不跨 0 就宣称某 reward 在所有训练 seed 上必然更好。

### 2.3 “完整答案率”

完整答案率表示模型产生了可解析、非空的 final answer，不表示该答案正确。它必须与 EM/F1、
`max_turns`、平均搜索数一起解释。

### 2.4 Gold Token-F1 Reward 的 V1 / V2

正式 reward 名称一直是：

```text
spad_em_teacher_backoff_gold_token_f1_bonus
```

文件名仍保留早期草案名称 `search_policy_teacher_reward_gold_match_bonus.py`，不要因为文件名包含
`gold_match` 就误认为当前使用 EM。实际实现使用归一化 token-F1。

- V1：已经训练和评估；只要求 Teacher 被调用、格式合法、answer 可解析且非空。
- V2：当前代码；额外要求 Actor answer 完整闭合，并要求 Teacher evidence status 属于
  `supported_answer` / `ambiguous_evidence`。
- V2 审计版本：`actor_answer_closed_teacher_supported_v2`。
- V1 checkpoint 不会因为代码升级而自动变成 V2 checkpoint，禁止混报。

## 3. 数据资产和哈希

| 用途 | 路径 | 规模 | SHA-256 |
|---|---|---:|---|
| 512 训练 | `data/global_train_eval_data/512t/co_search_ablation.train.parquet` | 512 | `2f9eb86fb40fbb69fab2aca7f6a4e4a05d6879e6dbbcd0fbe1d73e1a1a010558` |
| 5100 训练 | `data/global_train_eval_data/5100t/co_search_ablation.train.parquet` | 5100 配置，实际训练 5056 | `6e9307a8b3a866ecd045170bc0e92048e7e00fba0a0098b4ced5dd227ba9b09c` |
| 小评估 | `data/global_train_eval_data/350e/co_search_ablation.eval.parquet` | 350，7 源各 50 | `ddd7297f5f77253392ccfca331639280bdef672e0c85210ad1267a711601b660` |
| 主评估 | `data/global_train_eval_data/3500e/co_search_ablation.eval.parquet` | 3500 | `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283` |

3500e 的数据源分布：2WikiMultiHopQA 563、Bamboogle 125、HotpotQA 562、MuSiQue 562、
Natural Questions 562、PopQA 563、TriviaQA 563。

## 4. 被训练和评估对象的实现方法

### 4.1 Qwen3-1.7B Base

- 初始模型，不做 RL 训练。
- 用于 350 小评估的下界参照。
- 350 三次均值：EM `0.0810`、F1 `0.1567`、完整答案率 `0.5905`。
- 本轮 3500 四模型主比较没有重复跑 Base；重点是 Search-R1 与 SPAD 同规模比较。

### 4.2 Search-R1 original

配置入口：

```text
AgenticIterRag/config/agent_training/search_r1_original.yaml
```

关键公式：

```text
Actor 输出完整非空 <answer>...</answer> 且命中任一 gold alias 的规范化 EM：reward=1
否则：reward=0
Teacher：不调用
```

实现仍复用 SPAD Stage1 agent loop、检索工具、完整答案停止协议和 checkpoint finalizer，但 reward type
为 `search_r1_original`。这使 Search-R1 与 SPAD 可以在同一训练/评估基础设施下对比，主要差异集中在
reward，而不是 agent 推理协议。

正式配置共同点：GRPO、每题 8 rollout、Actor `temperature=1/top_p=1`、每 step 64 个问题、
`learning_rate=1e-6`、最多 6 轮 assistant、Recall Top N=50、Actor 可见 Top M=5。

### 4.3 SPAD stable Stage1：`spad_em_teacher_backoff`

稳定实现：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py
compute_spad_em_teacher_backoff_batch
```

对同一 UID 的 8 条 rollout 分组。令第 `i` 条 Actor 完整答案的 EM 为 `em_i`，默认
`partial_reward=0.1`：

```text
如果组内存在任意 em_i=1：
    reward_i = em_i
    整组不调用 Teacher

如果整组 em_i 全为 0：
    当前轨迹有检索证据，且 Teacher XML 合法，status 为
    supported_answer 或 ambiguous_evidence：reward_i=0.1
    否则：reward_i=0
```

设计目的不是让 Teacher 代替 Actor 答题，而是在整组 EM 全零、原始 GRPO 没有区分信号时，对“虽然
Actor 没答对，但检索证据有价值”的轨迹提供 backoff credit。Actor 答错但证据好而得 0.1 是设计
本意，不是实现错误。

一个重要边界：stable base backoff 不要求 Actor 已合法闭合答案。只要整组 EM 全零且当前轨迹有证据，
Teacher 就可能被调用。后续 V2 只限制 extra bonus，不修改 stable base 的这个语义。

### 4.4 SPAD 调度实验分支：`spad_em_teacher_backoff_dev`

独立模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_dev.py
```

它没有改变 stable 最终公式，而是尝试在 per-rollout reward loop 中提前异步预取 Teacher 结果，等同 UID
的 8 条 rollout 完成后，再由 batch reward 做全零 EM 裁决。历史 rollout 重放显示关键 reward 字段与
stable 一致，但调度改变了 Actor 随机采样请求顺序，因此最终 checkpoint 没有复现历史 stable 效果。

512 dev 训练：8 step，训练墙钟从 stable 的约 `50:08` 降至 `34:17`；一次 350 评估为
EM `0.1200`、F1 `0.1882`、完整答案率 `0.6457`。该分支可保留作调度研究，不能作为正式 stable
替代品。

### 4.5 SPAD Stage2 / Stage3

Stage2 不产生独立模型。它从 Stage1 轨迹生成 Actor rejected answer、Teacher refreshed answer，筛掉
证据不足等样本，产生 answer-distillation pair。

Stage3 当前正式实现为 answer-only Gold-answer token-F1 GRPO：

- 从 Stage1 checkpoint 初始化；
- 使用 Stage2 保留的 pair；
- 单轮 answer rollout，不在线调用 search/recall/Teacher；
- 输出 `<reason>...</reason><answer>...</answer>`；
- 按最终 Actor answer 对 gold 的 token-F1 训练。

Stage3 的风险已经被实际观察到：5100 Stage3 长训练后搜索行为大幅收缩，搜索率约从 Stage1 的
`0.9248` 降到 `0.4638`，EM 也显著下降。因此当前默认不再自动执行 Stage2/Stage3。

### 4.6 Gold Token-F1 Bonus V1：已训练版本

独立实现：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus.py
compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch
```

V1 先完整调用 stable `compute_spad_em_teacher_backoff_batch`，然后对 stable 结果副本追加：

```text
teacher_gold_token_f1 = max(token_f1(teacher_answer, gold_alias_j))
extra_bonus = 0.1 * teacher_gold_token_f1
final_reward = stable_base_reward + extra_bonus
```

V1 extra bonus eligibility：

```text
group_all_em_zero=true
Teacher 确实被调用
Teacher 没有 format error
Teacher parse status=parsed
Teacher answer 非空
```

V1 没有要求 Actor answer 合法闭合，也没有显式要求 Teacher evidence status 为 supported/ambiguous。
历史训练中正 bonus 的实际状态如下：

| 规模 | 正 bonus | Actor parsed | Actor `missing_answer_close` | supported | ambiguous | insufficient |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 735 | 618 | 117 | 695 | 40 | 0 |
| 5100 | 4679 | 4299 | 380 | 4402 | 272 | 5 |

因此 V1 的主要 bonus 仍来自 Teacher supported/ambiguous，但确实有一部分发给了 Actor 未闭合轨迹，
5100 中另有极少数 insufficient status 获得 bonus。

### 4.7 Gold Token-F1 Bonus V2：当前代码、尚未训练

正式 reward 名仍为：

```text
spad_em_teacher_backoff_gold_token_f1_bonus
```

V2 在 V1 条件上新增：

```text
actor_answer_parse_status == parsed
teacher_evidence_status in {supported_answer, ambiguous_evidence}
```

完整公式：

```text
base_reward = 原 spad_em_teacher_backoff 结果

eligible = 全零 EM 组
           且 Teacher 调用/解析成功
           且 Actor 输出完整非空 <answer>...</answer>
           且 Teacher status 为 supported_answer 或 ambiguous_evidence

extra_bonus = 0.1 * max_token_f1(Teacher answer, gold aliases) if eligible else 0
final_reward = base_reward + extra_bonus
```

V2 只收紧 extra bonus。Actor 未闭合但证据好的轨迹仍可获得 stable base 的 0.1，不会获得新增 bonus。

新增审计字段：

```text
teacher_gold_token_f1_bonus_eligible
teacher_gold_token_f1_bonus_eligibility_version=actor_answer_closed_teacher_supported_v2
```

定向验证：reward 单元测试 11 项、路由/prompt 传播测试 8 项，共 19 项通过；Python 编译和
`git diff --check` 通过。该数字对应当前 V2；历史 V1 报告中的 17 项是当时版本的测试数。

## 5. GRPO Advantage 标准化问题

### 5.1 当前真实配置

SPAD Stage1 强制：

```text
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=false
critic.enable=false
```

`algorithm.norm_adv_by_std_in_grpo` 没有被 Stage1 override，继承 VERL 默认值 `true`。实际实现：

```text
score_i = token_level_rewards.sum()
mean_g = 同 UID 8 条 rollout 的 score 均值
std_g = 同 UID 8 条 rollout 的 score 标准差
A_i = (score_i - mean_g) / (std_g + 1e-6)
```

代码位置：

```text
AgenticIterRag/verl/verl/trainer/ppo/core_algos.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/search_policy_rl.py
AgenticIterRag/config/ppo_trainer.yaml
```

### 5.2 为什么 `partial_reward=0.1` 基本没有实现弱梯度

对同一组：

```text
[0, 0, 0.1, 0, ...]
[0, 0, 1.0, 0, ...]
```

第二组只是第一组整体乘 10；减均值并除以组内 std 后，advantage 几乎完全相同，只有 epsilon 造成的
可忽略差异。因此 stable 设计中“Teacher backoff 最高 0.1，Actor EM 为 1，所以 Teacher 梯度弱十倍”
在当前 GRPO 中不成立。

同理，如果某个 all-zero 组的 stable base 都相同，而 token-F1 extra bonus 只产生很小的组内差异，
该差异仍可能被除以很小的 std 放大为单位级 advantage。bonus 的绝对数值不能直接代表其梯度强度。

### 5.3 推荐修改方向，但当前尚未修改

首选最小消融：只对 SPAD Stage1 设置：

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false
```

对应 advantage：

```text
A_i = score_i - mean_g
```

这样 0/0.1 backoff 的组内差异才真正是 0/1 EM 组差异的十分之一，微小 token-F1 bonus 也不会被
组内 std 放大。VERL 已支持该开关，不需要先修改 core algorithm。

备选方案是在标准化后的 advantage 上显式乘 reward-source group weight，但这需要改 Trainer，而且仍需
解决微小组内差异被放大的问题。当前建议先做 `norm_adv_by_std_in_grpo=false` 的受控实验。

重要：截至本文记录时，这个配置尚未写入默认配置或正式 overlay；所有历史 checkpoint 均为 `true`。

## 6. 共同训练基础设施

### 6.1 Actor 与资源

- Actor 初始模型：Qwen3-1.7B Base。
- Stage1 Actor：通常 NPU 0-3，4 个训练/rollout worker 对应的数据并行布局。
- Teacher：GLM-4.7-Flash，NPU 4-5，TP=2，BF16。
- Recall：NPU 6-7。
- Actor rollout：`temperature=1`、`top_p=1`、每题 8 条。
- Teacher：`temperature=0`、`top_p=1`、`max_tokens=512`、thinking=false、timeout=180 秒。
- Teacher batch workers：16。
- `stream_group_reward=true`：同 UID 8 条完成后即可提交组 reward，不等待整个 batch。

### 6.2 Stage1 核心参数

| 参数 | 值 |
|---|---:|
| train batch / actor batch | 64 / 64 |
| PPO mini batch | 64 |
| Actor micro batch per GPU | 2 |
| log-prob micro batch per GPU | 4 |
| rollout per prompt | 8 |
| learning rate | 1e-6 |
| max prompt / response | 12000 / 4096 |
| rollout max model len | 16096 |
| max assistant turns | 6 |
| visible retrieval docs | 5 |
| data shuffle / seed | true / 42 |

Stage1 stop sequences：

```text
</tool_call>
</answer>
```

### 6.3 Checkpoint finalizer

Stage1 和 Stage3 训练结束后，finalizer 会：

1. 合并 4 个 FSDP Actor shard；
2. 生成独立 HF checkpoint；
3. 使用 Transformers 做本地加载校验；
4. 原子落盘，避免留下半成品 checkpoint。

主要实现：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/checkpoint_finalizer.py
```

## 7. 已完成训练和 checkpoint

### 7.1 Search-R1 original

| 规模 | Run | Step / rollout | HF checkpoint | model.safetensors SHA-256 |
|---|---|---|---|---|
| 512 | `260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512` | 8 / 4096 | `checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `5cc4e18701b8184140e56875405b9c816c33a0b91c59407509ba0211b0b1facf` |
| 5100 | `260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100` | 79 / 40448 | `checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` | `67f0f161369452b05a0b00b36a36ec1474cc1f0d1168c0404e1e2d20d2bea376` |

Search-R1-5100 训练阶段耗时 `14408.9s`，约 4 小时；79/79 step 完成，返回码 0。

### 7.2 SPAD stable Stage1

| 对象 | Run / 关键点 | HF checkpoint | SHA-256 |
|---|---|---|---|
| 512 历史最佳，inflight=1 | `260711-103304-616277-...newdata_512` | `checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `df6385a0fc31811413d159ca7e1c4502c8fd055f9e1a136a43cd2387ac87c4ea` |
| 512 stable repeat，inflight=1 | `260712-131305-696244-...stable_stage1_repeat` | `checkpoints/AIR/260712-131305-696244-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_repeat/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `b6db99bd1b7df6eb63d8b54bf37a24cd418e0b2d0d083af376c691b2db5ede0f` |
| 512 inflight=2 消融 | `260712-143738-025140-...inflight2_ablation` | `checkpoints/AIR/260712-143738-025140-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_inflight2_ablation/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `d9ca546bec8a95958d67f5d5be7ccd45832e411894646ddccbc505dfba55af93` |
| 5100 stable，inflight=2 | `260711-235953-727858-...newdata_5100` | `checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` | `41f869b0d1e22b4e0306fab2d02ce56347fa5177918b959f16f4342792fc25d0` |

历史 stable 512 完成 8/8 step、4096 rollout、512 UID group；3012 条有完整答案、515 条 EM=1、
377 个全零组，Teacher 调用 2995 次，格式错误 4 次。

SPAD-5100 完整三阶段流水线耗时：Stage1 墙钟约 6 小时 26 分，Stage2 约 35 分 36 秒，Stage3
约 56 分 40 秒。Stage2 从 5100 条轨迹最终保留 2468 对；Stage3 训练 38 step。

### 7.3 SPAD Stage3 checkpoint

| 规模 | HF checkpoint | SHA-256 |
|---|---|---|
| 512 | `checkpoints/AIR/260711-115144-826023-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stage3_resume/stages/train_agent/spad_rag/answer_distillation/grpo/grpo_checkpoint_verl/actor_model_hf/global_step_3` | `5e19a1f7304f2294e1a5e4cd6289bb208a6f13bec9652485cd00b0b50da9b1b1` |
| 5100 | `checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/answer_distillation/grpo/grpo_checkpoint_verl/actor_model_hf/global_step_38` | `79a566f7e0797a84c7188463107975ae8e627fc84d0cd318ecf3bbe8fc2ab045` |

### 7.4 Gold Token-F1 Bonus V1 checkpoint

| 规模 | Run | Step / rollout | 耗时 | HF checkpoint | SHA-256 |
|---|---|---|---|---|---|
| 512 | `260713-011350-061908-...newdata_512_gold_token_f1_bonus_stage1` | 8 / 4096 | step 合计 40分44秒 | `checkpoints/AIR/260713-011350-061908-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `0a8027322624139667726da8a6cfa1c2d3d9d2f5d8551379002754de5d0361a3` |
| 5100 | `260713-022724-631051-...newdata_5100_gold_token_f1_bonus_stage1` | 79 / 40448 | step 合计 6小时17分03秒 | `checkpoints/AIR/260713-022724-631051-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` | `e33af413dd7492a8fcfea0048df32584eb5d4d1d69bc6aec38254a7cf7a86565` |

两个 run 均明确只选择 `search_policy_rl`，没有执行 Stage2/Stage3。

V1 训练 reward 审计：

| 规模 | 平均 base | 平均 bonus | 平均 final | Actor EM | Actor F1 | 正 bonus / rollout | Teacher 调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.1489 | 0.0131 | 0.1620 | 0.1189 | 0.2129 | 735/4096 | 3014/4096 |
| 5100 | 0.2957 | 0.0079 | 0.3035 | 0.2722 | 0.3460 | 4679/40448 | 23347/40448 |

5100 最后 3 步平均 bonus 仅 `0.0058`。这说明触发条件随 Actor EM 提高而变稀疏；但由于当前
GRPO 组内标准化，不能仅凭 bonus 均值小就判断其梯度影响小。

## 8. 350 条三次重复评估

统一协议：no-ranker、Recall Top N=50、模型可见 Top M=5、最多 6 轮 assistant、temperature=0、
top_p=1。表中是三次推理均值，不是训练 seed 均值。

| 模型 | EM | F1 | 完整答案率 | 平均搜索数 | 重复 query 率 | Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B Base | 0.0810 | 0.1567 | 0.5905 | 2.3943 | 0.3676 | 0.2248 |
| Search-R1-512 | 0.0981 | 0.1818 | 0.6352 | 2.4333 | 0.3838 | 0.2619 |
| Search-R1-5100 | 0.1619 | 0.2346 | 0.7171 | 1.7914 | 0.1990 | 0.1686 |
| SPAD-512 Stage1 historical stable | 0.1314 | 0.2251 | 0.7010 | 2.4505 | 0.3524 | 0.2590 |
| SPAD-512 Stage3 | 0.1362 | 0.2273 | 0.6610 | 2.5286 | 0.3610 | 0.2867 |
| SPAD-5100 Stage1 | 0.1705 | 0.2414 | 0.6914 | 2.7676 | 0.5848 | 0.2848 |
| SPAD-5100 Stage3 | 0.1371 | 0.2116 | 0.7143 | 0.5686 | 0.0419 | 0.0048 |

关键统计结论：

1. 512：SPAD Stage1 相对 Search-R1 的 EM/F1 为 `+0.0333/+0.0434`，95% CI 均不跨 0。
2. 512：Stage3 相对 Stage1 的 EM/F1 差值 CI 均跨 0，完整答案率显著下降 `-0.0400`。
3. Search-R1：5100 相对 512 的 EM/F1/完整答案率均显著提升。
4. 5100：SPAD Stage1 与 Search-R1-5100 的均值接近，350 题 CI 跨 0，不能在小评估上确认谁更好。
5. 5100：Stage3 相对 Stage1 的 EM 显著下降 `-0.0333`，搜索策略发生强烈收缩。

正式报告：

```text
docs/AgenticIterRag_v1/work_reports/260711-13a_新数据512_Base_Search-R1_original_SPAD训练与评估报告.md
docs/AgenticIterRag_v1/work_reports/260711-14a_Search-R1-5100训练与三次350评估报告.md
docs/AgenticIterRag_v1/work_reports/260712-15a_SPAD-5100三次350评估报告.md
```

## 9. 3500 条 Stage1 单次评估

### 9.1 Search-R1 与 stable SPAD 主比较

| 模型 | EM | F1 | 完整答案率 | 搜索率 | 平均搜索数 | 重复 query 率 | Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Search-R1-512 | 0.1180 | 0.1965 | 0.6271 | 0.9831 | 2.3489 | 0.3640 | 0.2569 |
| Search-R1-5100 | 0.1800 | 0.2509 | 0.7317 | 1.0000 | 1.7291 | 0.1786 | 0.1549 |
| SPAD-512 historical stable | 0.1360 | 0.2265 | 0.6989 | 0.9689 | 2.3391 | 0.3340 | 0.2426 |
| SPAD-5100 stable | 0.1923 | 0.2700 | 0.7220 | 0.9397 | 2.6557 | 0.5906 | 0.2443 |

10000 次、seed 42 的按题 paired bootstrap：

| 比较（右减左） | EM delta [95% CI] | F1 delta [95% CI] | 完整答案率 delta [95% CI] |
|---|---:|---:|---:|
| Search-R1-512 -> Search-R1-5100 | +0.0620 [0.0503, 0.0737] | +0.0544 [0.0423, 0.0666] | +0.1046 [0.0849, 0.1249] |
| SPAD-512 -> SPAD-5100 | +0.0563 [0.0457, 0.0674] | +0.0435 [0.0330, 0.0547] | +0.0231 [0.0069, 0.0394] |
| Search-R1-512 -> SPAD-512 | +0.0180 [0.0089, 0.0271] | +0.0300 [0.0202, 0.0397] | +0.0717 [0.0540, 0.0900] |
| Search-R1-5100 -> SPAD-5100 | +0.0123 [0.0009, 0.0240] | +0.0191 [0.0070, 0.0318] | -0.0097 [-0.0280, 0.0086] |

3500 题下，两个规模的 stable SPAD Stage1 EM/F1 都显著高于同规模 Search-R1 original；
SPAD-5100 的完整答案率没有显著高于 Search-R1-5100，并且搜索更多、重复 query 更多。

产物：

```text
docs/AgenticIterRag_v1/work_reports/260712-16a_四模型3500单次评估与推理加速报告.md
reports/eval/agenticIterRag/260712-newdata3500-stage1-formal-aggregate/report.md
reports/eval/agenticIterRag/260712-newdata3500-stage1-formal-aggregate/summary.json
```

### 9.2 Stable 512 重复训练与 inflight 消融

| 模型 | Inflight | EM | F1 | 完整答案率 | 平均搜索数 | 重复 query 率 |
|---|---:|---:|---:|---:|---:|---:|
| 历史 stable | 1 | 0.1360 | 0.2265 | 0.6989 | 2.3391 | 0.3340 |
| stable repeat | 1 | 0.1051 | 0.1737 | 0.5431 | 2.5257 | 0.4326 |
| inflight 消融 | 2 | 0.1054 | 0.1798 | 0.5900 | 2.3566 | 0.3466 |

`inflight=2` 相对本次 `inflight=1`：EM `+0.0003`、F1 `+0.0061`，CI 均跨 0；完整答案率
`+0.0469 [0.0334, 0.0603]`。8 step 时间从 2992.39 秒降到 2406.68 秒，缩短 19.6%。

历史 stable 与同配置 repeat 的泛化差距很大，而训练 reward 接近，说明 512 单次训练方差不可忽略。

产物：

```text
docs/AgenticIterRag_v1/work_reports/260712-17a_SPAD-512_Stable_Stage1重复训练与Inflight消融3500评估报告.md
reports/eval/agenticIterRag/260712-spad512-inflight-ablation-aggregate/report.md
reports/eval/agenticIterRag/260712-spad512-inflight-ablation-aggregate/summary.json
```

### 9.3 Gold Token-F1 Bonus V1

| 模型 | EM | F1 | 完整答案率 | 搜索率 | 平均搜索数 | 重复 query 率 | Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 stable historical, inflight=1 | 0.1360 | 0.2265 | 0.6989 | 0.9689 | 2.3391 | 0.3340 | 0.2426 |
| 512 stable repeat, inflight=2 | 0.1054 | 0.1798 | 0.5900 | 0.9711 | 2.3566 | 0.3466 | 0.2363 |
| 512 Gold Token-F1 V1, inflight=2 | 0.1231 | 0.2046 | 0.6220 | 0.9720 | 2.4654 | 0.3820 | 0.2511 |
| 5100 stable, inflight=2 | 0.1923 | 0.2700 | 0.7220 | 0.9397 | 2.6557 | 0.5906 | 0.2443 |
| 5100 Gold Token-F1 V1, inflight=2 | 0.1837 | 0.2576 | 0.6334 | 0.9971 | 3.0071 | 0.5763 | 0.3589 |

关键 paired 结果：

| 比较（右减左） | EM delta [95% CI] | F1 delta [95% CI] | 完整答案率 delta [95% CI] |
|---|---:|---:|---:|
| 512 stable inflight=2 -> V1 inflight=2 | +0.0177 [0.0097, 0.0257] | +0.0248 [0.0163, 0.0330] | +0.0320 [0.0166, 0.0477] |
| 512 historical stable inflight=1 -> V1 inflight=2 | -0.0129 [-0.0203, -0.0051] | -0.0220 [-0.0295, -0.0142] | -0.0769 [-0.0917, -0.0617] |
| 5100 stable inflight=2 -> V1 inflight=2 | -0.0086 [-0.0177, 0.0009] | -0.0125 [-0.0224, -0.0025] | -0.0886 [-0.1046, -0.0726] |
| V1 512 -> V1 5100 | +0.0606 [0.0497, 0.0720] | +0.0530 [0.0420, 0.0644] | +0.0114 [-0.0071, 0.0303] |

解释：V1 相对一次较弱的 512 同 inflight checkpoint 有提升，但低于历史最佳；5100 相对同规模 stable
F1 显著下降，完整答案率大幅下降。5100 V1 的搜索数和 Max-turn 同时上升，说明策略更偏向继续搜索，
没有更好地完成最终答案。该现象可能来自 reward proxy 与停止/完整作答的平衡，也可能叠加训练方差；
现有单 seed 不能给出最终因果结论。

产物：

```text
docs/AgenticIterRag_v1/work_reports/260713-18a_SPAD_GoldTokenF1Bonus新Reward_512与5100训练及3500评估报告.md
reports/eval/agenticIterRag/260713-gold-token-f1-bonus-3500-aggregate/report.md
reports/eval/agenticIterRag/260713-gold-token-f1-bonus-3500-aggregate/summary.json
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260713_gold_token_f1_bonus.json
```

## 10. 评估实现和推理加速

### 10.1 正式评估入口

```text
tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh
scripts/agenticIterRag_v1/assets/infer_backend/02_air_infer_launcher.sh
scripts/agenticIterRag_v1/assets/infer_backend/infer_air_vllm.py
```

虽然 wrapper 名称保留 `_350`，它支持 `--data-path` 和 `--max-samples`，已用于 3500 评估。

SPAD 评估 wrapper 当前默认：

```text
Actor NPU: 0,1,2,3,4,5
Actor replica: 6
Actor TP: 1
Actor max_num_seqs: 64
Recall NPU: 6,7
Recall replica: 2
infer_batch_size: 384
flush_every_n: 500
proxy strategy: least_inflight
run mode: no-ranker
Recall Top N: 50
Actor visible Top M: 5
temperature/top_p: 0/1
```

注意：更底层的通用 `02_air_infer_launcher.sh` 仍有自己的保守默认值；本轮“默认加速参数”指
`eval_spad_agent_search_350.sh` 这个正式 SPAD 评估入口，不要混淆。

### 10.2 I/O 瓶颈和 flush 修正

原逻辑每完成 10 条就重写“截至当前的全部 metrics 和 full trace”。3500 条时形成二次方累计写入：
完成 710/3500 条时主进程累计写盘已达 30.86 GB，而当时两份有效 trace 合计仅约 0.92 GB；按原方式
外推，单模型累计写入约 750 GB。

现在由参数 `flush_every_n` 控制中间重写粒度，3500 正式评估使用 500。任务完成时仍会写最终完整
metrics、trace、summary 和 report，不会丢失最终产物。

### 10.3 为什么选择 6 Actor + 2 Recall

- 7 Actor + 1 Recall 实测出现约 1.95 秒 Recall 排队，Actor 负载不均。
- 每个 Recall 索引约占 31 GB，每个 Actor 约占 43.7 GB，无法在一张 65.5 GB NPU 上共卡。
- 6+2 是当前 8 NPU 机器上更平衡的配置。

### 10.4 完整性保护

正式 wrapper 会：

1. 为每次推理写 eval run manifest，记录 task、repeat、数据 SHA、模型指纹、解码参数；
2. 拒绝复用非空 trace/runtime 目录；
3. 拒绝覆盖已有 report；
4. 每个 run 保存 `metrics.jsonl`、`traces.jsonl`、`summary.json`、`run_config.json`；
5. aggregate 校验样本数、唯一 index、模型指纹、数据 SHA 和 repeat ID；
6. paired bootstrap 默认 10000 次、seed 42。

每个新模型和每次 repeat 必须使用新的 `--task-name`。失败任务目录可以保留审计，但不能作为正式
结果复用。

## 11. 训练和评估产物目录结构

### 11.1 训练

```text
checkpoints/AIR/<pipeline-run>/stages/train_agent/spad_rag/search_policy_rl/
  actor_model_verl/global_step_<N>/
  actor_model_hf/global_step_<N>/

log/agenticIterRag/<pipeline-run>/
  runtime_logs/
  outputs/pipeline.manifest.json
  outputs/stages/train_agent/manifest.json
  outputs/stages/train_agent/spad_rag/spad_manifest.json
  outputs/stages/train_agent/spad_rag/search_policy_rl/manifest.json
  outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/<step>.jsonl
```

正式验收至少检查：pipeline/stage/sub-stage manifest 状态、实际 step、rollout 行数、每 UID 恰好 8 条、
reward type、Teacher 错误、HF `model.safetensors` 存在且可加载、模型 SHA。

### 11.2 评估

```text
reports/eval/agenticIterRag/<task>.report.md

log/eval/agenticIterRag/<task>/
  trace/metrics.jsonl
  trace/traces.jsonl
  trace/summary.json
  trace/run_config.json
  trace/validation_data/
  runtime_logs/eval_run_manifest.json
  runtime_logs/agent_timing.jsonl
  runtime_logs/search_timing.jsonl
```

3500 正式 run 必须确认 metrics/traces 各 3500 行、success=3500、failure=0、index 0-3499 完整。

## 12. 随机性与可复现性

已经固定：

- 训练 parquet 和 SHA；
- `data_shuffle=true`、`data_seed=42`；
- 512 stable 历史/repeat/inflight 消融的逐 step 问题 multiset 已核验一致；
- Teacher `temperature=0/top_p=1`；
- 正式评估 `temperature=0/top_p=1`；
- 评估数据、检索口径和模型指纹有 manifest。

尚未完全固定：

- Actor 训练 rollout 使用 `temperature=1`；
- 没有为每个 `question + rollout_index + step` 绑定独立生成 seed；
- Ray/vLLM 异步请求顺序会改变随机数消费顺序；
- 不同调度/inflight 即使输入问题相同，也不保证产生相同轨迹；
- 大多数 reward 结论只有一个训练 seed。

因此后续要判断 reward 或 advantage 的总体效果，必须增加独立训练 seed。重复评估同一 checkpoint 不能
替代重复训练。

## 13. 当前代码状态和历史结果的对应关系

| 项目 | 当前状态 | 是否已有训练/评估 |
|---|---|---|
| `search_r1_original` | 已实现、可运行 | 512/5100 已训练；350 三次、3500 一次 |
| `spad_em_teacher_backoff` | stable 实现保留 | 512/5100 已训练和评估；含 repeat/inflight 消融 |
| `spad_em_teacher_backoff_dev` | 独立调度实验分支 | 512 一次训练、350 一次评估 |
| Gold Token-F1 Bonus V1 | 代码已被 V2 条件覆盖，但历史 rollout/checkpoint 保留 | 512/5100 已训练、3500 一次 |
| Gold Token-F1 Bonus V2 | 当前同名 reward 实现 | 测试完成；尚未训练、尚未评估 |
| `norm_adv_by_std_in_grpo=false` | 仅讨论方案，尚未写入正式配置 | 尚无训练、尚无评估 |
| Stage1-only 默认 | `spad_rag_base.yaml` 已设 stop at Stage1 | 新 Gold V1 两个 run 已验证只跑 Stage1 |

不要用当前 V2 代码离线重算后，把分数误标到 V1 已训练 checkpoint 上；训练时策略看到的 reward 已经
确定，后处理不能改变 checkpoint 的训练语义。

## 14. 推荐的下一步实验顺序

### 14.1 第一优先级：解决 advantage 尺度问题

先对 stable `spad_em_teacher_backoff` 做受控消融，暂时不要同时加入 Gold V2：

| 组 | Reward | `norm_adv_by_std_in_grpo` | Inflight | 数据 | 目的 |
|---|---|---:|---:|---:|---|
| A | stable | true | 2 | 512 | 当前算法基线，建议重新跑多 seed，而非只引用历史单点 |
| B | stable | false | 2 | 512 | 检验 0.1 backoff 是否真正成为弱梯度 |

建议每组至少 3 个独立训练 seed。其余参数、代码版本、parquet、batch、Teacher、初始模型完全固定。
如果还没有逐样本 seed 支持，至少为每个 run 记录顶层 seed、启动时间、软件版本和完整 command plan。

先检查：

- 训练期 EM group / all-zero group 占比；
- Teacher status 分布；
- advantage 均值、std、按 reward source 的绝对值；
- Actor 合法闭合率；
- 搜索数、重复 query、Max-turn；
- 3500 EM/F1/完整答案率。

只有 B 在多个训练 seed 上表现稳定，才考虑把 `norm=false` 晋升为 SPAD Stage1 默认值。

### 14.2 第二优先级：评估 Gold Token-F1 Bonus V2

在 advantage 语义确定后，再做同 advantage 设置下的 reward 消融：

| 组 | Reward | Eligibility | Advantage 设置 | Inflight |
|---|---|---|---|---:|
| Stable | `spad_em_teacher_backoff` | 原 stable | 与 V2 完全一致 | 2 |
| V2 | `spad_em_teacher_backoff_gold_token_f1_bonus` | Actor closed + Teacher supported/ambiguous | 与 stable 完全一致 | 2 |

V2 首轮仍建议 512 多 seed；不要直接消耗 6 小时以上跑 5100。只有 512 的训练 seed 均值和行为指标
显示没有明显停止退化，再启动 5100。

### 14.3 可选的完整 2 x 2 设计

如果资源允许，最干净的设计是：

```text
reward in {stable, Gold V2}
norm_adv_by_std_in_grpo in {true, false}
```

四个 cell 都固定 inflight=2，每个 cell 至少 3 个训练 seed。这样才能分离 reward eligibility 和
advantage normalization 的主效应/交互。资源不足时按 14.1 -> 14.2 分阶段执行。

### 14.4 评估策略

- 训练中只按约定的 2-10 分钟间隔检查，不用训练 reward 提前选 winner。
- 每个训练 seed 的 final checkpoint 在 3500e 上做一次确定性评估即可。
- 对训练 seed 先报告均值/方差；paired bootstrap 仍用于每个 checkpoint 的逐题差异，但不要把多个
  seed 或推理 repeat 当成独立题目简单池化。
- 主指标：EM、F1、完整答案率。
- 必报行为指标：搜索率、平均搜索数、重复 query 率、Max-turn 率、搜索次数桶。
- 必报训练审计：base reward、extra bonus、eligible/applied count、Actor parse status、Teacher status。

## 15. 运行监控约束

用户已明确要求后续长任务遵守以下频率：

- 资源加载：每 1 分钟检查一次；
- 训练：按任务阶段每 2-10 分钟检查一次；
- 评估：按任务阶段每 2-5 分钟检查一次；
- 未到时间不查、不报，除非用户主动询问；
- 禁止口头声称等待，实际每 30 秒或 1 分钟轮询训练/评估；
- 禁止为了维持一个等待命令而反复调用 `wait` / `write_stdin`。

检查频率应根据单 step 或 batch 的历史耗时选择。例如 Stage1 单 step 约 5 分钟时，可在预计完成附近
检查，而不是固定高频查询。

## 16. 常见错误和避免方式

1. 不要把 Gold V1 checkpoint 写成当前 V2 reward 的结果。
2. 不要把 5100 配置写成实际优化了 5100 个 prompt；实际是 5056。
3. 不要把 Teacher `partial_reward=0.1` 直接解释为当前梯度只有 EM 的 0.1；组内 std 标准化会消尺度。
4. 不要只看训练 `critic/score/mean` 判断泛化。历史 stable/repeat 的训练 reward 接近，3500 F1 差距大。
5. 不要把 `stream_group_max_inflight` 当成 Teacher 采样参数；它只改变每 worker 同时推进的 UID group 数。
6. 不要把 inflight=1 历史最佳与 inflight=2 新 reward 当成严格单变量 reward 消融。
7. 不要把同 checkpoint 三次推理当作三个训练 seed。
8. 不要把 Stage3 推理更快解释为系统吞吐提高；5100 Stage3 主要是因为大量不搜索。
9. 不要复用非空 eval task 目录或 report 名称。
10. 不要把底层通用 infer launcher 的默认值与 SPAD 正式评估 wrapper 的默认值混写。
11. 不要默认开启 Stage2/Stage3；当前默认只到 Stage1。
12. 不要清理或回退当前 dirty worktree 中不属于当前任务的改动。

## 17. 关键代码、配置和报告索引

### 17.1 Reward 和训练路由

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_dev.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus.py
AgenticIterRag/agentic_iter_rag/agent_training/spad/search_policy_rl.py
AgenticIterRag/config/agent_training/spad_rag_base.yaml
AgenticIterRag/config/agent_training/search_r1_original.yaml
```

### 17.2 V1 Gold Bonus 训练入口

```text
tasks/train_tasks/agenticIterRag/run_260713_AIR_spad_qwen3_1_7b_glm47_gold_match_bonus_512_stage1.sh
tasks/train_tasks/agenticIterRag/run_260713_AIR_spad_qwen3_1_7b_glm47_gold_match_bonus_5100_stage1.sh
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_gold_match_bonus_512_stage1_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_gold_match_bonus_5100_stage1_overlay.yaml
```

这些 launcher/overlay 选择的是同名 reward；如果现在重新启动，它们会使用当前 V2 代码，而不是历史 V1。
若需要复现实验，必须先冻结具体代码 revision，不能只凭脚本名判断公式。

### 17.3 评估与聚合

```text
tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh
tasks/eval_tasks/agenticIterRag/run_260712_newdata3500_stage1_formal_4runs.sh
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260712_spad512_inflight_ablation.json
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260713_gold_token_f1_bonus.json
scripts/agenticIterRag_v1/assets/infer_backend/infer_air_vllm.py
```

### 17.4 详细 work reports

```text
docs/AgenticIterRag_v1/work_reports/260711-11a_当前reward_spad_em_teacher_backoff梳理和使用.md
docs/AgenticIterRag_v1/work_reports/260711-13a_新数据512_Base_Search-R1_original_SPAD训练与评估报告.md
docs/AgenticIterRag_v1/work_reports/260711-14a_Search-R1-5100训练与三次350评估报告.md
docs/AgenticIterRag_v1/work_reports/260711-14a_spad_em_teacher_backoff_dev调度实现与512训练350评估报告.md
docs/AgenticIterRag_v1/work_reports/260712-15a_SPAD-5100三次350评估报告.md
docs/AgenticIterRag_v1/work_reports/260712-16a_四模型3500单次评估与推理加速报告.md
docs/AgenticIterRag_v1/work_reports/260712-17a_SPAD-512_Stable_Stage1重复训练与Inflight消融3500评估报告.md
docs/AgenticIterRag_v1/work_reports/260713-18a_SPAD_GoldTokenF1Bonus新Reward_512与5100训练及3500评估报告.md
```

## 18. 恢复工作时的最短检查清单

1. 阅读本文第 1、2、5、13、14 节。
2. 执行 `git status --short`，确认不要覆盖现有改动。
3. 检查 8 张 NPU、残留 vLLM/Recall/Ray 进程和端口。
4. 确认目标实验使用 stable、Gold V2 还是历史 V1；V1 当前不能仅靠同名脚本复现。
5. 确认 `norm_adv_by_std_in_grpo` 的实际 resolved value，并写入 command plan/manifest。
6. 固定 `stream_group_max_inflight=2`，除非本轮专门消融该参数。
7. 只执行 Stage1，除非用户明确要求 Stage2/Stage3。
8. 使用唯一 run name 和 eval task name。
9. 训练结束后核对 step、rollout、UID group、reward audit、HF checkpoint 和 SHA。
10. 评估结束后核对 3500 metrics、3500 traces、0 failure、模型/数据 manifest。
11. 生成按题 paired bootstrap，同时单独报告训练 seed 方差。
12. 将新实验的 reward revision、advantage 设置、checkpoint 路径和关键差别追加到新的 work report，
    不要覆盖历史结果。

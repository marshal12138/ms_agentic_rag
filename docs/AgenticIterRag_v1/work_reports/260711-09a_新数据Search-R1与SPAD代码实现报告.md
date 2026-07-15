# 新数据 Search-R1 与 SPAD 代码实现报告

日期：2026-07-11

## 1. 范围

本轮实现只服务于同一批新数据上的以下实验：

1. Base Qwen3-1.7B。
2. 新数据 Search-R1-512 与 Search-R1-5100。
3. 新数据 SPAD-512、SPAD-5100 的 Stage1 与 Stage3 GRPO。
4. SPAD Stage2 数据刷新质量统计。

本轮 Search-R1 统一使用 `search_r1_original` 名称和 reward。当前 `single-or-v2` 数据已清理为
单一答案或同义 alias OR，因此不再引入另一种 Search-R1 模型表述。旧 Search-R1、旧 SPAD、
旧数据、旧 checkpoint、旧 Stage2 pair 和旧评测轨迹均不进入正式比较。

## 2. 固定数据

| 数据 | 行数 | SHA-256 |
| --- | ---: | --- |
| `512t/co_search_ablation.train.parquet` | 512 | `2f9eb86fb40fbb69fab2aca7f6a4e4a05d6879e6dbbcd0fbe1d73e1a1a010558` |
| `5100t/co_search_ablation.train.parquet` | 5100 | `6e9307a8b3a866ecd045170bc0e92048e7e00fba0a0098b4ced5dd227ba9b09c` |
| `350e/co_search_ablation.eval.parquet` | 350 | `ddd7297f5f77253392ccfca331639280bdef672e0c85210ad1267a711601b660` |

512t 是 5100t 的严格子集，包含 NQ 205、HotpotQA 142、MuSiQue 90、2WikiMultiHopQA 75。
350e 包含七个数据源各 50 条。

## 3. Stage1 组级奖励

SPAD Stage1 现在生成完整 `<answer>...</answer>`，按 VERL 原始 prompt UID 将 8 条 rollout 分组。
对第 `i` 条 rollout 先计算完整答案 Search-R1 EM `e_i`：

```text
若组内存在任意 e_j=1：r_i=e_i，整组不调用 teacher
若组内 EM 全为 0：r_i=0.1*t_i
```

`t_i` 只判断 actor 实际看到的检索证据：`supported_answer` 和 `ambiguous_evidence` 为 1，
`insufficient_evidence`、请求错误或 XML 解析错误为 0。新 reward 不叠加 search cost、bad-stop、
duplicate query penalty 或旧 teacher F1。

关键实现：

- `search_policy_teacher_reward.py`：完整答案 EM、UID 分组、全零组 teacher 回退和审计字段。
- `verl/workers/reward_manager/batch.py`：把 agent loop 展平后的 `tool_call_details`、messages、UID 和
  turn 信息逐条传入 batch reward。
- `search_policy_rl.py`：选择 BatchRewardManager、关闭 reward loop、固定 64/64/8 和 TP=1。
- agent loop manager 按完整 UID 组切分 worker；同 UID 的 8 条 rollout 一完成即在该 worker 内提交
  batch reward，与其他 UID 的生成重叠。
- 每个 rollout worker 同时处理一个 UID reward，组内最多 8 个 teacher 请求；完成后的 `rm_scores`
  和全部审计 extras 直接回传，训练器不再建立中央整批 teacher 屏障。
- agent loop worker 只在真正启用旧异步 reward loop/resource-pool router 时创建
  `RewardManagerWorker`。

0710 teacher-answer-F1 reward 已独立命名为 `spad_teacher_f1_0710`，冻结在
`search_policy_teacher_reward_0710.py`。选择该名称会自动切换到 naive reward loop、关闭 UID 流式
组奖励并恢复 `<answer>` opening-stop；详细梳理见
`260711-10a_旧reward_spad_teacher_f1_0710梳理和使用.md`。

## 4. 全量 rollout 契约

agent loop 和 trainer 现在记录：

- 原始 prompt、每个 assistant turn 原始文本与实际执行文本。
- attempted/executed query、tool error、耗时。
- 实际可见 observation、截断前后文本与长度。
- recall/rank/teacher-visible evidence。
- teacher messages、请求 hash、生成参数、原始 XML、解析状态和错误。
- EM、teacher binary reward、组级状态与最终 reward。

每一步写入临时 JSONL 后原子替换为 `{step}.jsonl`。`manifest.json` 增量记录 shard SHA、字节数、
记录数、UID 组数、字段非空计数、teacher called/skipped/error 计数和 completed 标记。训练入口在
manifest 的 step、prompt、group、rollout 数量和 shard hash 全部复验前，不允许进入 HF finalizer
或 Stage2。

新增 `scripts/cosearch_local/audit_spad_stage1_rollouts.py`，逐 UID 重算奖励公式并汇总 EM 全零组、
teacher 调用组、backoff 非恒定组和最终非恒定组。

## 5. Stage3 GRPO

Stage3 默认从 DPO 改为 Gold-answer F1 GRPO，同时保留 DPO 配置、代码和可执行路径：

- Stage2 pair 转为 VERL prompt/reward parquet。
- 使用 Stage1 HF tokenizer 和真实 chat template 统计 prompt token；超过 12000 token 的 pair 记录为
  `prompt_too_long` 并跳过，避免 dataloader 后置失败。
- 单轮 answer rollout，不启动 search、recall 或 teacher。
- 输出协议为 `<reason>...</reason><answer>...</answer>`。
- 每 prompt 8 条，reward 为抽取答案对多个 gold 的最大 token F1。
- train batch 64、mini-batch 64、TP=1，多卡只做 data parallel。
- `save_freq=1000000`：当前数据规模只因 `is_last_step` 保存最终 checkpoint。

## 6. 自动 HF checkpoint

新增共享 `checkpoint_finalizer.py`，供 Search-R1 Stage1、SPAD Stage1 和 SPAD Stage3 共用：

1. 定位最终 VERL `global_step_N`。
2. FSDP shard 合并到临时 HF 目录。
3. 校验 config、tokenizer、safetensors 和 Transformers 可读性。
4. 原子 rename。
5. 在 manifest 登记 raw/HF 路径、转换日志及 config/权重 SHA。

转换或验证失败会使 phase 非零退出，不能进入下一阶段或评估。

## 7. 评估缓存隔离

`eval_agent_search.sh` 现在拒绝复用非空 trace/runtime 目录或已有报告；每个模型、每次 repeat
必须使用唯一 task name。每次评估在启动服务前记录：

- eval parquet 内容 SHA。
- 模型 config/tokenizer/全部 safetensors 内容 SHA 与综合指纹。
- repeat ID、温度、top-p 和样本数。

vLLM prefix cache 只缓存 KV，不复用已生成答案。正式三次重复评估分别执行新推理，聚合器仅接受
带上述隔离 manifest 的三个 run。

新增 `aggregate_newdata_model_eval.py`：每个问题先对三个 repeat 取均值，再对 350 个问题做 paired
bootstrap，避免把三次运行错误当成 1050 个独立样本。

## 8. 配置与验证

配置中保留两个规模的 overlay，但本轮只执行512规模，完成训练、评估和报告后暂停：

- Search-R1-512：8 step。
- SPAD-512：Stage1 8 step，Stage2 最多 512 条，Stage3 使用本轨道 kept pairs。
- Search-R1-5100 与 SPAD-5100 仅保留配置，不在本轮启动。

训练运行时完整环境中的单元测试结果：`65/65` 通过。SPAD-512 dry-run 确认：

- Stage1：batch reward、reward loop 关闭、64/64/8、TP=1、8 step、stop `</answer>`。
- Stage3：单轮 Gold-F1 GRPO、64/64/8、TP=1、final-only checkpoint 保存。

0710 reward overlay dry-run 另确认：独立模块、naive manager、reward loop 开启、流式组奖励关闭、
stop `['</tool_call>', '<answer>']` 五项同时生效。

真实训练和模型评估结果单独写入
`260711-13a_新数据512_Base_Search-R1_original_SPAD训练与评估报告.md`，不在本文引用旧实验数字。

## 9. 512真实执行验证

以上代码路径已在完整512实验上验证：

- Search-R1 original、SPAD Stage1和SPAD Stage3均产生可由Transformers本地加载的HF checkpoint。
- 四个模型各完成三次独立350条评估，共12个run、4200条trace。
- 聚合器实际复验data SHA、repeat ID、模型指纹、输出隔离标记和每个run的350条完整性。
- 10000次paired bootstrap（seed 42）成功产生五组模型差值及95%置信区间。
- 正式聚合产物位于`reports/eval/agenticIterRag/260711-newdata512-formal-aggregate/`。

# Search-R1 结构化多 Gold 模型训练与评估报告

> **已废止（2026-07-11）**：本文包含旧 Search-R1 与旧评测轨迹的比较，不符合后续明确的
> “只比较同一批新数据上的 Base、新数据 Search-R1 和 SPAD 各阶段”口径。本文所有训练/评估
> 数字均不得进入当前结果表、差值、显著性检验或结论；当前结果以新的训练评估报告为准。

生成时间：2026-07-11 08:10 CST

## 1. 结论

结构化 Search-R1 独立训练成功完成，产出 Qwen3-1.7B global step 8 checkpoint。评估只比较同一份
新结构化 350 数据上的不同模型，不使用旧数据集指标作为结论。

三次复评结果：

| 模型 | Runs | Legacy EM | Structured EM | Group F1 | Group Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B Base | 3 | `0.1038±0.0092` | `0.1062±0.0094` | `0.1794±0.0047` | `0.1072±0.0094` |
| 旧 Search-R1 模型 | 3 | `0.1362±0.0044` | `0.1345±0.0029` | `0.1956±0.0021` | `0.1377±0.0029` |
| 结构化 Search-R1 模型 | 3 | `0.1390±0.0044` | `0.1413±0.0045` | `0.2042±0.0045` | `0.1445±0.0045` |

在相同新数据口径下，结构化模型相对旧 Search-R1 模型：

- Structured EM：`+0.0068`，约 `+5.1%` 相对提升。
- Group F1：`+0.0086`，约 `+4.4%` 相对提升。
- Group Recall：`+0.0068`，约 `+5.0%` 相对提升。
- Legacy EM：`+0.0029`，没有以牺牲旧口径为代价。

提升方向在四项指标上一致，但样本仅 350，且三轮之间存在可见波动。当前证据支持“结构化 reward
模型优于 Base，且均值高于旧 Search-R1”，不应扩张为统计显著性结论。

## 2. 实验范围

唯一评估数据：

`data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/search_r1_structured.eval.parquet`

- 行数：350，每个数据源 50 条。
- SHA256：`ce01777aabcbfae4e48343b09fc76bb6f043f500177a8f51df039beea47453db`。
- Structured eligible：342；8 条评估样本因歧义从结构化分母排除。
- Legacy 指标分母：350。

旧 Base 和旧 Search-R1 的已有三轮轨迹只复用模型输出，不复用旧评分。复评脚本已逐条验证轨迹
index、数据源和问题文本与新 parquet 一致，再用当前共享指标重新计算，因此表中所有分数都属于
同一份新数据口径。

## 3. 独立训练

### 3.1 配置

| 项目 | 值 |
| --- | --- |
| 初始模型 | `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B` |
| 训练数据 | 512 prompts，SHA256 `631b6023...c8237d` |
| Reward | `search_r1_structured`, `and_of_or_v1` |
| Epoch / steps | 1 / 8 |
| Train batch | 64 prompts/step |
| Rollout | 8 samples/prompt，512 rollouts/step |
| 训练总 rollout | 4096 |
| Actor NPU | 0,1,2,3，TP=1 |
| Recall NPU | 6,7 |
| Max response | 4096 tokens |
| Rollout prefix cache | enabled，仅 KV 计算缓存 |

训练任务：

`260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512`

返回码 0，无超时，耗时 `1670.1s`（27.8 分钟）。8 个 rollout shard 均为 512 条，共 4096 条。

### 3.2 Step 级观测

| Step | Structured reward mean | Response length mean | Search count mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.0742 | 1815.3 | 1.9512 |
| 2 | 0.0605 | 1753.8 | 1.8340 |
| 3 | 0.1074 | 1670.8 | 1.6895 |
| 4 | 0.0977 | 1576.4 | 1.4980 |
| 5 | 0.2051 | 1430.7 | 1.2695 |
| 6 | 0.1660 | 1301.9 | 1.1367 |
| 7 | 0.2324 | 1333.5 | 1.1367 |
| 8 | 0.1934 | 1422.8 | 1.2188 |

后四步 reward mean 高于前四步，同时平均响应长度和搜索次数下降。但每步是不同训练 batch，不能把
该表当作独立验证集学习曲线；最终结论以 350 条端到端复评为准。

训练期间按要求在实际训练进入后至少间隔十分钟检查，未使用高频轮询。训练结束后将 VERL
checkpoint 合并为 HF checkpoint，并验证 Qwen3 config 和 tokenizer 可加载。

### 3.3 Checkpoint

- VERL：`checkpoints/AIR/260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/global_step_8`
- HF：`checkpoints/AIR/260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- `model.safetensors`：约 3.8 GiB。

## 4. 端到端评估协议

所有模型使用相同的新 350 数据和以下推理参数：

- no-ranker 搜索路径。
- Recall Top N=50，agent 可见 Top M=5。
- `temperature=0.0`，`top_p=1.0`，thinking disabled。
- 6 个 Qwen3-1.7B vLLM 数据并行副本，least-inflight 路由。
- 2 个检索副本。
- 最多 6 assistant turns。
- 每个模型族三轮，均以三轮均值和样本标准差报告。

结构化模型三轮实际结果：

| Run | Success | Legacy EM | Legacy F1 | Structured EM | Group F1 | Group Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 349/350 | 0.1343 | 0.2021 | 0.1374 | 0.2009 | 0.1406 |
| Run 2 | 350/350 | 0.1429 | 0.2114 | 0.1462 | 0.2093 | 0.1494 |
| Run 3 | 349/350 | 0.1400 | 0.2067 | 0.1404 | 0.2025 | 0.1435 |

每次实际评估启动后均避免短间隔轮询；服务冷启动与模型推理分开识别，结果完成后再核查 350 条
metrics 和 350 条 traces。

## 5. 缓存影响审计

重复评估需要区分三类缓存：

| 缓存 | 本实验状态 | 是否复放答案 | 影响 |
| --- | --- | --- | --- |
| Response/trajectory cache | 未实现、未启用 | 否 | 不影响独立重复 |
| vLLM prefix KV cache | 每轮内部启用 | 否 | 只减少相同前缀 prefill 计算 |
| HF/ModelScope 文件 cache | 启用 | 否 | 只避免重复下载/加载模型文件 |

证据：

1. AIR infer 代码没有 response cache 读取或写入路径。
2. 启动器检测到 agent 端口已有模型时直接报错，禁止服务复用。
3. 每个任务使用独立 task name、trace 目录和新 vLLM PID；每轮结束后服务均停止。
4. vLLM 日志明确记录 `enable_prefix_caching=True` 和约 25%-35% prefix hit。该缓存保存已计算的
   prompt KV block，不保存 sampled token 或完整响应。
5. 三轮输出和指标并不相同，也排除了整条答案缓存复放的可能。

因此 prefix cache 不使三次评估变成同一批答案的重复计数。仍然存在的运行波动来自 6 副本动态路由、
批处理形状和 NPU 数值非确定性；历史和当前实验在 `temperature=0` 下均观察到这种波动，所以采用
三重复均值而不是挑选最好单轮。

## 6. 同一新数据上的模型比较

相对 Base，旧 Search-R1 已带来明显提升；结构化 Search-R1 在其上进一步提高三轮均值：

| 对比 | Structured EM delta | Group F1 delta | Group Recall delta |
| --- | ---: | ---: | ---: |
| 旧 Search-R1 - Base | +0.0283 | +0.0162 | +0.0305 |
| 结构化 Search-R1 - Base | +0.0351 | +0.0248 | +0.0373 |
| 结构化 Search-R1 - 旧 Search-R1 | +0.0068 | +0.0086 | +0.0068 |

按数据源观察，结构化模型的变化并非所有子集一致；350 条数据中每个数据源仅 50 条，子集波动较大，
不据此做过强归因。整体 Group F1 的提升与新 reward 关注“必答 group 覆盖且答案简短”的目标一致。

`required_set` 在评估集中仅 3 条、`multi_slot` 仅 4 条，严格 Structured EM 的粒度很粗。Group F1
能显示部分覆盖改善，但还需要更大规模集合/槽位评估集才能稳定量化专项收益。

## 7. 产物

### 7.1 评估报告

- Run 1：`reports/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_eval350.report.md`
- Run 2：`reports/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350.report.md`
- Run 3：`reports/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350.report.md`
- 跨模型统一复评：`reports/eval/agenticIterRag/260711-search_r1-structured-comparison/report.md`
- 机器可读汇总：`reports/eval/agenticIterRag/260711-search_r1-structured-comparison/summary.json`

### 7.2 训练日志

- Pipeline manifest：`log/agenticIterRag/260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512/outputs/pipeline.manifest.json`
- 训练日志：同任务目录下 `runtime_logs/stages/train_agent/spad_rag/search_policy_rl/verl_train.log`。
- Rollout：同任务目录下 `outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/`。

## 8. 限制与后续判定标准

1. 当前只有一个训练 seed；三次评估重复不能替代多个训练 seed。
2. 两轮各有 1 条请求失败，失败按空答案计 0；三轮成功率均不低于 99.7%。
3. 结构化模型相对旧 Search-R1 的均值提升约为 1.5 个旧模型标准差到 4 个标准差，方向可信但样本
   规模不足以直接宣称显著。
4. 后续若继续投入训练，应优先增加至少 3 个训练 seed 和集合/槽位专项评估，而不是只增加同一
   checkpoint 的推理重复次数。

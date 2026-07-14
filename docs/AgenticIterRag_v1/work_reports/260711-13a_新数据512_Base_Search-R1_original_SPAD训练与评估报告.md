# 新数据512 Base、Search-R1 original与SPAD训练评估报告

日期：2026-07-11

## 1. 结论

本轮已完成512条训练数据规模下的全部训练、12次独立评估和配对统计。在同一
350e数据、同一检索和解码协议下，三次repeat均值为：

| 模型 | EM | F1 | 完整答案率 |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B Base | 0.0810 | 0.1567 | 0.5905 |
| Search-R1 original-512 | 0.0981 | 0.1818 | 0.6352 |
| SPAD-512 Stage1 | 0.1314 | 0.2251 | **0.7010** |
| SPAD-512 Stage3 | **0.1362** | **0.2273** | 0.6610 |

主要结论：

1. Search-R1相对Base的F1提升为`+0.0251`，95% CI `[0.0013, 0.0492]`；EM提升
   `+0.0171`，但区间`[-0.0057, 0.0410]`跨0。
2. SPAD Stage1相对Base的EM/F1提升为`+0.0505/+0.0685`，置信区间均不跨0。
3. SPAD Stage1相对Search-R1的EM/F1提升为`+0.0333/+0.0434`，置信区间均不跨0。
4. Stage3相对Stage1的EM/F1只增加`+0.0048/+0.0021`，95% CI均跨0，不能宣称
   Stage3在答案准确率上有可确证提升。
5. Stage3完整答案率相对Stage1下降`-0.0400`，95% CI `[-0.0648, -0.0152]`，
   这是本轮Stage3最明确的退化信号。

因此，512规模下最稳健的收益来自SPAD Stage1。Stage3平均EM/F1最高，但没有显著超过
Stage1，且完整答案率明显下降。

## 2. 范围与固定协议

本轮只包含四个模型：Base、Search-R1 original-512、SPAD-512 Stage1和SPAD-512 Stage3。
SPAD Stage2只产生刷新数据，不产生独立模型。0710 reward消融和5100规模实验均未启动。

| 用途 | 路径 | 行数 | SHA-256 |
| --- | --- | ---: | --- |
| 训练 | `data/global_train_eval_data/512t/co_search_ablation.train.parquet` | 512 | `2f9eb86fb40fbb69fab2aca7f6a4e4a05d6879e6dbbcd0fbe1d73e1a1a010558` |
| 评估 | `data/global_train_eval_data/350e/co_search_ablation.eval.parquet` | 350 | `ddd7297f5f77253392ccfca331639280bdef672e0c85210ad1267a711601b660` |

评估统一使用no-ranker、Recall Top N=50、模型可见Top M=5、最多6轮assistant、
`temperature=0.0`、`top_p=1.0`。每个模型做三次独立推理，每次使用新服务进程、
唯一task name和独立trace/runtime目录。

## 3. 训练与数据刷新

### 3.1 SPAD Stage1

正式run：

```text
260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512
```

- 完成8/8 step、4096 rollout、512 UID group。
- 3012条rollout有完整答案，515条EM=1，4068条有检索证据。
- 377个组全零EM，调用teacher 2995次；teacher格式错误4次，reward公式错误0次。
- 全量rollout审计PASS。

HF checkpoint：

```text
checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

### 3.2 SPAD Stage2

- 输入512条trajectory，actor合法完成484条，未完成23条，无检索证据5条。
- teacher eligible/completed为484/484，无teacher请求、超时或格式错误。
- 证据不足过滤233条，最终保留251个pair，schema错误0，chosen=rejected为0。

最终pair：

```text
log/agenticIterRag/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/outputs/stages/train_agent/spad_rag/answer_refresh_data/answer_distill_pairs.jsonl
```

### 3.3 SPAD Stage3

恢复run：

```text
260711-115144-826023-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stage3_resume
```

- 251个Stage2 pair全部转为GRPO parquet，无超长、空prompt、空gold或非法行。
- 完成3/3 step、1536 rollout，返回码0。
- 三个训练batch的Gold-F1 mean为0.1265、0.1295、0.2183；它们不是验证集学习曲线。

HF checkpoint：

```text
checkpoints/AIR/260711-115144-826023-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stage3_resume/stages/train_agent/spad_rag/answer_distillation/grpo/grpo_checkpoint_verl/actor_model_hf/global_step_3
```

权重SHA-256：`5e19a1f7304f2294e1a5e4cd6289bb208a6f13bec9652485cd00b0b50da9b1b1`。

### 3.4 Search-R1 original-512

正式run：

```text
260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512
```

- `reward.type=search_r1_original`，teacher调用0次。
- 完成8/8 step、4096 rollout、512 UID group，8个shard全部有SHA且manifest `completed=true`。
- 训练后完整FSDP shard已通过共享finalizer合并、Transformers本地校验并原子落盘，
  不需要重跑训练。

HF checkpoint：

```text
checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

权重SHA-256：`5cc4e18701b8184140e56875405b9c816c33a0b91c59407509ba0211b0b1facf`。

## 4. 正式评估完整性

| 模型 | 模型指纹 | 正式run | 完整性 |
| --- | --- | --- | --- |
| Base | `93cbc1b5e618...c5ce220` | `260711-newdata512-base-retry1-run{1,2,3}` | 3 x 350 |
| Search-R1-512 | `d4329ecb6e79...866a6ffa` | `260711-newdata512-search-r1-run{1,2,3}` | 3 x 350 |
| SPAD Stage1 | `d3d0d47a0d27...1f88130b` | `260711-newdata512-spad-stage1-run{1,2,3}` | 3 x 350 |
| SPAD Stage3 | `a40aab7c06f3...15b63423` | `260711-newdata512-spad-stage3-retry1-run{1,2,3}` | 3 x 350 |

聚合器成功复验data SHA、repeat ID、模型指纹、`output_reuse=false`和每个run的
350个唯一索引trace及350条metrics。同模型三次指纹一致，不同模型指纹不同。

两次资源加载中断不进入正式结果：首个Base run因Search-R1训练遗留recall服务导致
NPU OOM；首个Stage3 run因前台终端连接中断而停止。两者都为0 trace/0 metrics，失败目录保留审计，
正式run spec改用全新task name。

## 5. 总体评估结果

`structured_em`与EM、`answer_group_f1`与F1、`answer_group_recall`与EM在本批
single-or-v2数据上数值相同，是单一答案或同义alias OR口径的预期结果。

| 模型 | EM | F1 | Structured EM | Group F1 | Group recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.0810 +/- 0.0087 | 0.1567 +/- 0.0092 | 0.0810 +/- 0.0087 | 0.1567 +/- 0.0092 | 0.0810 +/- 0.0087 |
| Search-R1-512 | 0.0981 +/- 0.0059 | 0.1818 +/- 0.0050 | 0.0981 +/- 0.0059 | 0.1818 +/- 0.0050 | 0.0981 +/- 0.0059 |
| SPAD Stage1 | 0.1314 +/- 0.0029 | 0.2251 +/- 0.0064 | 0.1314 +/- 0.0029 | 0.2251 +/- 0.0064 | 0.1314 +/- 0.0029 |
| SPAD Stage3 | 0.1362 +/- 0.0033 | 0.2273 +/- 0.0024 | 0.1362 +/- 0.0033 | 0.2273 +/- 0.0024 | 0.1362 +/- 0.0033 |

| 模型 | 完整答案率 | 首次搜索率 | 平均搜索数 | 重复query率 | 最大轮数率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.5905 | 0.9886 | 2.3943 | 0.3676 | 0.2248 |
| Search-R1-512 | 0.6352 | 0.9886 | 2.4333 | 0.3838 | 0.2619 |
| SPAD Stage1 | **0.7010** | 0.9790 | 2.4505 | **0.3524** | 0.2590 |
| SPAD Stage3 | 0.6610 | 0.9829 | 2.5286 | 0.3610 | 0.2867 |

## 6. 配对Bootstrap

对每个问题先取三次repeat均值，再对350个问题做10000次有放回配对抽样，seed 42。

| 比较 | EM差值 [95% CI] | F1差值 [95% CI] | 完整答案率差值 [95% CI] |
| --- | ---: | ---: | ---: |
| Base -> Search-R1 | +0.0171 [-0.0057, 0.0410] | +0.0251 [0.0013, 0.0492] | +0.0448 [-0.0029, 0.0914] |
| Base -> SPAD Stage1 | +0.0505 [0.0229, 0.0800] | +0.0685 [0.0414, 0.0970] | +0.1105 [0.0638, 0.1552] |
| Base -> SPAD Stage3 | +0.0552 [0.0267, 0.0857] | +0.0706 [0.0423, 0.0993] | +0.0705 [0.0238, 0.1152] |
| Search-R1 -> SPAD Stage1 | +0.0333 [0.0057, 0.0610] | +0.0434 [0.0159, 0.0718] | +0.0657 [0.0171, 0.1133] |
| SPAD Stage1 -> Stage3 | +0.0048 [-0.0124, 0.0219] | +0.0021 [-0.0122, 0.0164] | -0.0400 [-0.0648, -0.0152] |

## 7. 数据源差异

| 数据源 | Base F1 | Search-R1 F1 | Stage1 F1 | Stage3 F1 |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | 0.1096 | 0.1350 | 0.1325 | **0.1671** |
| Bamboogle | 0.1462 | 0.1814 | **0.2345** | 0.2226 |
| HotpotQA | 0.2002 | 0.1888 | **0.2148** | 0.2000 |
| MuSiQue | 0.0662 | 0.1139 | **0.1577** | 0.1376 |
| NQ | 0.2070 | 0.2045 | 0.3039 | **0.3210** |
| PopQA | 0.1610 | 0.2164 | 0.2596 | **0.2844** |
| TriviaQA | 0.2065 | 0.2324 | **0.2729** | 0.2582 |

Stage3的收益主要出现在2WikiMultiHopQA、NQ和PopQA；在Bamboogle、HotpotQA、MuSiQue和
TriviaQA上低于Stage1。这种混合结果与Stage1 -> Stage3总体置信区间跨0一致。

## 8. 产物索引

- SPAD Stage1审计：`reports/eval/agenticIterRag/260711-spad512-stage1-rollout-audit/`
- 12个单run报告：`reports/eval/agenticIterRag/260711-newdata512-*.report.md`
- 聚合报告：`reports/eval/agenticIterRag/260711-newdata512-formal-aggregate/report.md`
- 机器可读聚合：`reports/eval/agenticIterRag/260711-newdata512-formal-aggregate/summary.json`
- 实际run spec：`tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260711_512_formal.json`
- 顺序执行脚本：`tasks/eval_tasks/agenticIterRag/run_260711_newdata512_formal_12runs.sh`

## 9. 验收与停止

- 完整单元测试`65/65`通过。
- SPAD Stage1/2/3和Search-R1 original-512训练产物验收完成。
- 四模型各三次正式评估与五组配对统计完成。
- 本轮到此暂停，不启动0710 reward消融，不启动5100训练或评估。

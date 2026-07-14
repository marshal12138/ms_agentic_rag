# Search-R1 最佳 LLM-Judge 奖励离线重放实验

生成时间：2026-07-10 18 时

命名说明：`260710-18a` 表示 2026 年 7 月 10 日 18 点生成的第 1 篇报告。

## 结论

本实验使用 teacher_PE 当前最佳单调用策略 `question-tail evidence-only`，对最新
Search-R1 训练的全部 4096 条 rollout 做了一次 cache-free fresh inference。实验没有
重启 vLLM，也没有修改生产 prompt、生产 reward 或默认配置。

实验结果证明：LLM judge 确实能增加训练信号，但当前版本尚未达到直接启动 Stage1
训练的条件。

| 检查项 | 结果 | 判断 |
| --- | ---: | --- |
| 原始 0/1 reward 非恒定 group | 138/512，27.0% | 基线 |
| Stage1 judge reward 非恒定 group | 210/512，41.0% | 信号明显增加 |
| 完整轨迹 reward 非恒定 group | 324/512，63.3% | 信号明显增加，但不等同于 Stage1 reward |
| 4096 条请求错误 | 0 | 通过 |
| XML parse rate | 4092/4096，99.90% | 未达到 teacher_PE 的 1.0 硬门槛 |
| 240 条人工 evidence accuracy | 77.9% | 不足 |
| 人工 `I` precision / recall / F1 | 81.2% / 91.9% / 86.2% | 偏保守，false-I 较多 |
| 人工排序均值 | 符合预设层级 | 通过 |
| 人工逐样本排序 | 分离度不足 | 未通过 |

当前 judge reward 将非恒定 group 净增加 72 个，但它不是在原有 138 个 group 上简单
追加信号：139 个原本恒定的 group 获得方差，同时有 67 个原本具有 0/1 方差的 group
在新 reward 下变成恒定。它改变了轨迹排序，而不只是把 0/1 reward 平滑化。

最主要的阻塞是 judge 的证据判定仍有明显噪声。240 条人工审查中，人工 `S` 的 95 条
有 25 条被 judge 判成 `I`、9 条判成 `A`；人工 `A` 的 9 条只有 1 条被正确识别。虽然
`I` recall 达到 91.9%，但 `I` precision 只有 81.2%，低于该 prompt 在原 237 条 benchmark
上的稳定均值 85.6%。因此，本实验不建议立刻启动正式训练。

## 实验范围

Search-R1 rollout：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/{1..8}.jsonl`

共 8 个 step，每个 step 512 条轨迹，由 64 个 prompt、每个 prompt 8 条 rollout 组成；
总计 4096 条轨迹、512 个 GRPO group。

人工验证集：

[`260710-17a_Search-R1零奖励人工审查240样本明细.tsv`](./260710-17a_Search-R1零奖励人工审查240样本明细.tsv)

实验 runner：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/teacher_PE/run_search_r1_reward_replay.py`

结果目录：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/teacher_PE/results/R1_reward_replay_qtail_evidence_260710_18a`

主要结果文件：

- `predictions.jsonl`：4096 条逐轨迹 judge 原始输出、解析结果和候选 reward。
- `manual_240_predictions.jsonl`：240 条人工样本与 judge 结果的逐条对齐。
- `metrics.json`：分数分布、逐 step group 方差和人工验证指标。
- `run.json`：端点、参数、耗时、cache 和错误计数。
- `system_prompt.txt`、`variant.json`：实际使用的 prompt 快照与 hash。

## Judge 策略

使用的 registry 变体为：

`baseline_question_tail_evidence_only_v2`

完整 variant hash 为：

`d27ed640aa7b94e8931763fa5a39053846922973f8c09a9436ea976e8573d311`

该 hash 与 teacher_PE 已验证的最佳 T1 结果完全一致。请求保持：

- 隐藏 actor 的 sub-query。
- 保留每轮完整 top-5 title 和 passage。
- 在全部 evidence 后重申 Original question。
- 单次 GLM-4.7-Flash 调用。
- `temperature=0`、`top_p=1`、`max_tokens=512`。
- 不使用 thinking、gold、few-shot、critic 或投票。
- 不读取或写入 response cache，`cache_hits=0`。

对于达到最大轮数后残留最后一次 `<tool_call>`、但没有对应 `<tool_response>` 的轨迹，
只将 actor 实际已经看到的 response 作为 evidence。由于该布局不展示 sub-query，无需把未执行
query 补入 teacher prompt。

## 候选 reward 定义

### Stage1 搜索策略分数

SPAD Stage1 在 `<answer>` opening 停止，actor 的答案正文不属于搜索策略 credit。因此本实验
首先定义面向 Stage1 的 judge reward：

```text
judge parse error:
  reward = -0.1

judge status = insufficient_evidence / ambiguous_evidence:
  reward = 0

judge status = supported_answer:
  reward = 0.25 + 0.75 * F1(teacher_answer, gold)
```

该映射将“judge 认为证据充分，但 teacher 短答案与 gold 完全不一致”设为 0.25，而不是
0.5；teacher answer 完全正确时为 1.0。它不对 `I/A` 施加强 bad-stop 负分。

### 完整 Search-R1 轨迹分数

为了同时检查 actor 最终答案，另计算诊断用完整轨迹分数：

```text
合法 actor answer:
  reward = 0.75 * F1(actor_answer, gold)
         + 0.25 * evidence_answer_score

非法或无 actor answer:
  reward = 0.10 * evidence_answer_score

evidence_answer_score:
  judge status = S 时为 F1(teacher_answer, gold)，否则为 0
```

该分数不能直接替代 SPAD Stage1 reward，因为 Stage1 当前不生成完整 answer body。本报告将
它作为完整 Search-R1 轨迹排序的诊断项。

另行计算了每次重复 query 扣 0.1 的版本，但没有将其作为 judge 信号改善的主结果。重复
惩罚会独立制造组内方差，不能据此宣称 evidence judge 更准确。

## 运行结果

| 指标 | 数值 |
| --- | ---: |
| 总轨迹 | 4096 |
| 墙钟耗时 | 512.99 秒，约 8.55 分钟 |
| 请求错误 | 0 |
| 首次请求成功 | 4096/4096 |
| 平均请求耗时 | 7.96 秒 |
| 请求耗时中位数 | 7.57 秒 |
| 请求耗时 P95 | 13.26 秒 |
| 平均 prompt tokens | 1449.5 |
| 平均 completion tokens | 85.1 |
| XML parse 成功 | 4092/4096，99.90% |

四个 replica 分别处理 1029、1027、1024 和 1016 条请求，负载基本均衡。推理结束后
`8067-8070` 四个服务均保持 ready。

Judge 状态分布：

| Judge 标签 | 条数 | 占比 |
| --- | ---: | ---: |
| `S` | 1654 | 40.38% |
| `I` | 2214 | 54.05% |
| `A` | 224 | 5.47% |
| 格式错误 `E` | 4 | 0.10% |

4 条格式错误中，3 条输出了 `<reason>` opening 后没有 `</reason>`，随后直接输出
`<status>`；另 1 条 completion 达到 512 token 后被截断。没有 HTTP 请求错误。

## GRPO 组内信号

| 分数 | 非恒定 group | 占 512 group | 恒定 group |
| --- | ---: | ---: | ---: |
| 原始 Search-R1 0/1 reward | 138 | 27.0% | 374 |
| Actor answer F1 | 287 | 56.1% | 225 |
| Teacher answer F1 | 168 | 32.8% | 344 |
| Stage1 judge reward | 210 | 41.0% | 302 |
| 完整轨迹 reward | 324 | 63.3% | 188 |
| Stage1 judge reward + duplicate penalty | 372 | 72.7% | 140 |
| 完整轨迹 reward + duplicate penalty | 428 | 83.6% | 84 |

逐 step 非恒定 group 数：

| Step | 原始 0/1 | Stage1 judge reward | 完整轨迹 reward |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 31 | 47 |
| 2 | 13 | 34 | 43 |
| 3 | 18 | 32 | 42 |
| 4 | 11 | 20 | 39 |
| 5 | 23 | 21 | 42 |
| 6 | 17 | 27 | 38 |
| 7 | 17 | 21 | 35 |
| 8 | 23 | 24 | 38 |
| 合计 | 138 | 210 | 324 |

Stage1 judge reward 并非每个 step 都增加方差，第 5 步从 23 降到 21。全量 group 转换为：

| 原始 reward | Stage1 judge reward | Group 数 |
| --- | --- | ---: |
| 恒定 | 恒定 | 235 |
| 恒定 | 非恒定 | 139 |
| 非恒定 | 恒定 | 67 |
| 非恒定 | 非恒定 | 71 |

因此，净增加 72 个非恒定 group 的同时，judge 也重新排列并抹平了部分原 0/1 差异。

## 原始 0/1 与 Judge 的关系

| 原始 reward | 轨迹数 | Judge `S` | Judge `I` | Judge `A` | 格式错误 | Stage1 judge reward 均值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3457 | 1179 | 2083 | 191 | 4 | 0.2455 |
| 1 | 639 | 475 | 131 | 33 | 0 | 0.7239 |

Judge 能把原始 reward 1 的轨迹整体排在 reward 0 之上，但两者明显重叠。原始 reward 0
中有 34.1% 被 judge 判为 `S`；这包含严格 EM 误伤、证据充分但 actor 答错，以及 judge
false-S。原始 reward 1 中也有 25.7% 被判为 `I/A`；这既可能是模型依靠参数记忆答对，
也可能是 judge false-I，不能只依据原始 EM 判断哪一方正确。

## 240 条人工 Evidence 验证

| 人工标签 \\ Judge | `S` | `I` | `A` | 格式错误 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S` | 61 | 25 | 9 | 0 | 95 |
| `I` | 10 | 125 | 1 | 0 | 136 |
| `A` | 4 | 4 | 1 | 0 | 9 |
| 合计 | 75 | 154 | 11 | 0 | 240 |

指标：

| 指标 | 数值 |
| --- | ---: |
| Evidence 三分类 accuracy | 77.9% |
| `I` precision | 81.2% |
| `I` recall | 91.9% |
| `I` F1 | 86.2% |
| XML parse rate | 100% |

该 judge 的特点是高 `I` recall、较低 `I` precision，即倾向于把已有充分证据判成不足。
如果 `I` 只得到 0 而不接受强负分，false-I 的主要影响是漏掉正向轨迹；如果重新启用
`bad_stop` 强惩罚，同一错误会直接变成反向训练信号，风险明显更大。

`A` 只有 9 条，且只判对 1 条，再次说明 `ambiguous` 不适合作为独立的强 reward 分支。

## 人工排序验证

按人工 evidence 和 actor answer 组合分组：

| 人工类别 | 样本 | Stage1 judge reward 均值 | 完整轨迹 reward 均值 |
| --- | ---: | ---: | ---: |
| `S+C`：证据充分、actor 语义正确 | 43 | 0.6427 | 0.5400 |
| `S+P`：证据充分、actor 部分正确 | 7 | 0.5374 | 0.4224 |
| `S+W`：证据充分、actor 回答错误 | 20 | 0.4982 | 0.2542 |
| `S+N`：证据充分、没有合法 answer | 25 | 0.3027 | 0.0270 |
| `I`：证据不足 | 136 | 0.0476 | 0.0556 |
| `A`：真实歧义 | 9 | 0.1944 | 0.0222 |

均值满足主要预设顺序：

`S+C > S+P > S+W > S+N > I`

但逐样本分离并不可靠：

- Stage1 reward 中，随机一条 `S+C` 高于随机一条 `S+W` 的成对概率只有 56.2%。
- `S+W` 高于 `I` 的成对概率为 79.6%，方向较好但仍有约两成反序或持平。
- 完整轨迹 reward 中，`C` 高于 `P` 的成对概率只有 55.7%。
- 完整轨迹 reward 中，`P` 高于 `W` 的成对概率为 88.1%。

原因是最佳 question-tail evidence-only prompt 可靠地输出 evidence status 和 teacher 短答案，
但不直接判断 actor answer 的语义正确性。完整轨迹分数仍依赖 token F1；人工标记为 `C`
的 58 条轨迹，其 actor F1 均值只有 0.468，中位数 0.4。因此本实验没有解决“语义正确
但 F1 较低”的全部问题。

## 是否进入训练

本轮不进入正式 Stage1 训练，理由如下：

1. XML parse rate 为 99.90%，没有达到 teacher_PE 约定的 1.0。
2. 新 240 条上的 `I` precision 为 81.2%，低于原 holdout 稳定均值 85.6%，false-I 较多。
3. 非恒定 group 从 27.0% 提升到 41.0%，证明方向有效，但仍有 59.0% group 没有组内
   Stage1 judge reward 方差。
4. 预设层级只在均值上成立，逐样本排序没有达到可直接训练的可靠程度。
5. 完整轨迹 reward 的 63.3% 非恒定 group 不能用于证明 Stage1 已通过，因为它包含当前
   Stage1 不生成的 actor answer body 信号。

## 下一步建议

下一步仍应保持单变量、离线验证：

1. 保持 question-tail evidence-only 布局和单 teacher 调用，不回退到 sub-query prompt。
2. 增加 teacher XML 格式重试或受约束输出，使 4096 规模 parse rate 达到 1.0。
3. `I/A` 继续只给 0，不恢复 `bad_stop` 强负分；在当前 81.2% 的 `I` precision 下，强惩罚
   会把 false-I 放大成反向梯度。
4. evidence bonus 从 0.25 继续向下做一次纯离线敏感性分析，例如 0.10、0.20、0.25；任何
   正 bonus 基本都能保留 status 带来的组内方差，较小 bonus 可降低 false-S 的伤害。
5. 如果目标是完整 Search-R1 轨迹，而不是 SPAD Stage1 搜索策略，则需单独开发并人工验证
   actor semantic-answer judge；不能声称当前 evidence judge 已经把语义正确答案稳定映射到 1。
6. 完成上述校准后，仍先固定相同 512 条数据和 seed 进行小规模 Stage1 单变量对照，不直接
   进入 Stage2、Stage3 或全流程训练。

## 数据解析说明

4096 条中有 35 条没有可见 `<tool_response>`；另有 10 条 response 在第一个 passage header
前包含异常文本。runner 对后者忽略 header 前异常文本并保留随后实际可见的 title/passage。
全量 gold 字面命中统计重新校验为 reward 0 中 1544 条命中、1913 条未命中，与
`260710-17a` 报告完全一致。

这些解析 warning 不等同于 LLM 请求错误；所有 4096 次 API 请求均在第一次尝试成功。

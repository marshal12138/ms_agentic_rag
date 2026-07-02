# Mix Signal 配置重复与漂移隐患记录

本文现在只保留还需要继续判断的配置点，以及已经落地、会影响风险判断的关键事实。已经收敛、已经补校验、或者不再作为当前风险跟踪的内容，只在必要处简要说明。

本文的判断口径仍然是：

- overlay 覆盖 base 是正常机制，不算重复风险。
- 多个基础配置同时写同一个语义，才算 base/base 重复风险。
- 多个普通 overlay 同时改同一个实验语义，才算 overlay/overlay 重复风险。
- launcher 生成的 runtime override / run_mode override 是编译产物，不当作人工维护的第三份配置。

## 本次已执行的代码改动

ranker training sample builder 的审计已经落地：审计文件现在会输出 ranker training 当前到底走哪条 sample builder 路径。

新增 `.env` 字段：

- `RANKER_TRAINING_SIGNAL_SOURCE`
- `RANKER_TRAINING_RANKER_TRAINABLE`
- `RANKER_TRAINING_UPDATE_MODE`
- `RANKER_TRAINING_ASYNC_ENABLE`
- `RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH`
- `RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_TYPE`
- `RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_JSON`
- `RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON`

这个审计值来自最终 Hydra config，不从单个 overlay 猜。当前 no-ranker 下会明确写成 disabled；full async 模式下才会摘出实际启用的 async sample builder。

top-N/top-M/top-K 相关日志和报告也已经同步：

- canonical v2 dry-run 会直接打印 recall final top-N、ranker final top-K、searchTool final top-M。
- canonical `.env` 会记录 Hydra、runtime tool config、静态 tool config 三边的最终 top 值。
- 训练 metrics report 会增加 `Retrieval Cutoffs` 小节。
- 新 v2 eval 的 shell report、`.env`、Python eval report、`run_config.json` 会直接展示 `RECALL_FINAL_TOP_N`、`SEARCH_TOOL_FINAL_TOP_M`、`RANKER_FINAL_TOP_K` 和 `LLM_IO_MAX_RECORDS`。

`format_penalty` 的 canonical 训练侧重复也已经收敛：

- 最终 answer 格式惩罚的事实源只剩 Hydra `custom_reward_function.reward_kwargs.format_penalty`。
- 静态 CoAgenticRetriever tool config 不再写 `format_penalty`。
- `CoAgenticRetrieverTool` 不再读取 `format_penalty`。
- v2 launcher 会拒绝静态 tool config 里的 `config.format_penalty`。
- canonical `.env` 只记录 `REWARD_FORMAT_PENALTY` 和 `REWARD_FORMAT_PENALTY_SOURCE`。

## 当前入口与真实运行形态

当前入口脚本是：

- `tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh`

它选择了 ranker / async ranker training 相关 base 和 overlay，但末尾又传了：

- `--run_mode=no-ranker`

所以当前真实运行形态是 recall-only / no-ranker。full mix-signal 的说明仍然有意义，因为同一条 canonical 链路切回 `run_mode=full` 后会重新启用 ranker 和 async ranker training。

## 风险分级

- P0: 值不一致时可能导致训练行为错误、服务连接失败或 silent fallback。
- P1: 值不一致时可能导致实验语义漂移、指标不可比、资源占用错误或调参结论误判。
- P2: 当前可运行，但维护成本、理解成本或审计成本偏高。

## P1 隐患

### 7. top-N / top-M / top-K / max-docs 名字相似但不是一回事

这一项已经做了一轮代码收敛，但还要继续跟踪 rank50 judge 的一致性校验。

先说当前结论：

- 训练侧不保留旧 top 字段兼容。旧字段如果出现在训练配置里，会直接报错。
- v2 eval 侧已经迁到 canonical runtime config，入口只接受结构化配置和明确支持的 override。
- 正确的新 Hydra overlay 不会失效。`recall_retriever.recall_final_top_n` 和 `ranker.top_k` 会在最终 Hydra compose 后写进本次 run 的 runtime tool config。
- 错误的旧 overlay 不会静默失效。`recall_retriever.top_k`、`ranker.final_top_k`、顶层 `default_top_n/default_top_m/searchTool_final_top_m` 都会被 canonical launcher 拒绝。

这次收敛的方向是：不要再让静态 tool config、Hydra recall 配置、Hydra ranker 配置同时写很像的 `top_k` 字段。现在按语义拆成三层：

| 语义 | 当前事实源 | 当前值 | 谁消费 |
| --- | --- | ---: | --- |
| recall 候选池大小 | Hydra `recall_retriever.recall_final_top_n` | 50 | launcher 生成 runtime tool config；trainer enrichment 截取 recall docs |
| ranker 排序后保留多少篇 | Hydra `ranker.top_k` | 50 | launcher 生成 runtime tool config；trainer enrichment 写 rank-final docs |
| agent 最终看多少篇 | tool config `searchTool_final_top_m` | 5 | `CoAgenticRetrieverTool.execute()` 格式化 tool response |
| judge 每次评估多少篇 | async ranker training stage `max_docs_per_request` | 50 | async judge request/prompt/schema |
| judge 排名前几篇算 positive | async sample builder `positive_top_k` | 5 | `llm_judge_topk` signal builder |

#### 本次删除和改名的字段

静态 tool config 以前是这样：

```yaml
# CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
default_top_n: 50
default_top_m: 5
ranker:
  top_k: 50
```

现在变成：

```yaml
# CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
searchTool_final_top_m: 5
ranker:
  backend: ray_actor
  required: true
```

也就是说：

- `default_top_n` 已删除。静态 tool config 不再决定 recall 候选池大小。
- `default_top_m` 已改名为 `searchTool_final_top_m`。这个名字直接说明它是 search tool 最终给 agent 看的文档数。
- tool config 里的 `ranker.top_k` 已删除。tool config 不再写第二份 ranker top-K。

Hydra ranker base 以前是这样：

```yaml
# CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml
recall_retriever:
  top_k: 50

ranker:
  top_k: 5
```

现在变成：

```yaml
# CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml
recall_retriever:
  recall_final_top_n: 50

ranker:
  top_k: 50
```

这里的两个 50 不是重复写同一个字段，而是两个连续阶段：

- `recall_final_top_n=50`：先从 recall retriever 拿 50 篇候选。
- `ranker.top_k=50`：ranker 对这 50 篇排序后，保留 50 篇作为 rank-final 结果，供 judge 和训练信号使用。

#### 为什么之前 `ranker.top_k` 是 5

之前默认值是 5，是因为旧实现把 Hydra `ranker.top_k` 用成了“训练侧 top5 trace 截断”：

```python
detail["rank_top5_docs"] = rank_top50[: int(_cfg_require(self.config, "ranker.top_k"))]
```

这和我们现在对齐的语义不一致。我们现在要的 `ranker.top_k` 不是“给 agent 看 5 篇”，也不是“训练日志里只保留 top5”，而是“ranker 排完 recall 候选后保留多少篇”。在 rank50 judge 设计下，它应该是 50。

代码已经按这个语义改掉：

```python
rank_final_top_docs = rank_top50[: int(_cfg_require(self.config, "ranker.top_k"))]
detail["rank_top50_docs"] = rank_final_top_docs
detail["rank_top5_docs"] = rank_final_top_docs[:5]
detail["rank_final_top_docs"] = rank_final_top_docs
```

所以现在：

- `ranker.top_k=50` 控制 ranker 排序后保留 50 篇。
- `rank_top50_docs` 仍然保留，兼容当前 rank50 judge/request builder。
- `rank_top5_docs` 只作为兼容旧消费者的 top5 视图，固定取前 5 篇，不再由 `ranker.top_k` 决定。

#### runtime tool config 如何知道这些值

静态 tool config 删除 `default_top_n` 和 `ranker.top_k` 后，tool 运行时仍然需要知道 recall top-N 和 ranker final top-K。现在由 canonical launcher 在 compose 最终 Hydra config 后生成 runtime tool config。

runtime tool config 会包含类似：

```yaml
recall_final_top_n: 50
searchTool_final_top_m: 5
ranker:
  backend: ray_actor
  required: true
  actor_name: coagentic_shared_dense_ranker
  actor_namespace: null
  final_top_k: 50
  max_query_length: 256
  max_doc_length: 512
```

对应关系是：

- `recall_final_top_n` 来自最终 Hydra `recall_retriever.recall_final_top_n`。
- `ranker.final_top_k` 来自最终 Hydra `ranker.top_k`。
- `searchTool_final_top_m` 来自静态 tool config。
- `actor_name/actor_namespace/max_query_length/max_doc_length` 也由 launcher 从最终 Hydra 配置生成。

审计文件也会记录：

```text
HYDRA_RECALL_FINAL_TOP_N=50
HYDRA_RANKER_FINAL_TOP_K=50
RUNTIME_TOOL_RECALL_FINAL_TOP_N=50
RUNTIME_TOOL_RANKER_FINAL_TOP_K=50
SEARCH_TOOL_FINAL_TOP_M=5
RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M=5
TOP_M=5
```

这样读者可以一眼看出：recall/ranker top 值来自 Hydra，agent-visible top-M 来自 tool config。

#### 这次实际改了哪些训练脚本

训练侧改动的边界是：当前 canonical 训练链路必须只认新事实源，不再从旧字段兜底。

实际涉及这些文件：

```text
scripts/coagenticRetriever_v2/01_train_launcher.sh
scripts/coagenticRetriever_v2/assets/trainer_launcher/compile_config.py
scripts/coagenticRetriever_v2/assets/trainer_launcher/runtime_env.py
scripts/coagenticRetriever_v2/assets/trainer_launcher/tool_config.py
scripts/coagenticRetriever_v2/assets/trainer_launcher/audit_files.py
scripts/coagenticRetriever_v2/assets/report_schema.py
scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh
CoAgenticRetriever/scripts/train_coagentic_retriever_grpo.sh
```

具体行为是：

- `01_train_launcher.sh` 仍然是 canonical 训练入口，本身不再拼 topK/topM 训练语义；它调用 compiler 生成最终 env、Hydra args 和 runtime tool config。
- `compile_config.py` 会先 compose 最终 Hydra config，再读取 `recall_retriever.recall_final_top_n` 和 `ranker.top_k`，写入审计 env，并调用 runtime tool config generator。
- `runtime_env.py` 只从静态 tool config 读取 `searchTool_final_top_m`，不再从 tool config 读取 recall top-N，也不再接受静态 tool config 缺 top-M 后悄悄用默认值。
- `tool_config.py` 现在会拒绝训练侧旧字段：静态 `default_top_n/default_top_m`、静态 `ranker.top_k/final_top_k`，以及静态 ranker 里的 model/device/token length 等运行字段。静态 `ranker` 段只允许 `backend` 和 `required`。
- `audit_files.py` 增加了 `HYDRA_RECALL_FINAL_TOP_N`、`HYDRA_RANKER_FINAL_TOP_K`、`RUNTIME_TOOL_RECALL_FINAL_TOP_N`、`RUNTIME_TOOL_RANKER_FINAL_TOP_K`、`SEARCH_TOOL_FINAL_TOP_M`、`RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M`，便于比较 Hydra、runtime tool config、静态 tool config 的最终值。
- `01_train_launcher.sh` 的 dry-run 日志会直接打印 recall final top-N、ranker final top-K、searchTool final top-M；`report_schema.py` 也会在训练指标报告里单独生成 `Retrieval Cutoffs` 小节。这样读报告时不需要反推 `TOP_N/TOP_M` 到底是什么意思。
- `assets/00_run_agentic_iter_rag_verl.sh` 是旧 asset runner，不是当前 canonical 主链路。它也被收紧了：只读新字段，缺 `recall_final_top_n/searchTool_final_top_m` 就失败，不再 fallback 到 `default_top_n/default_top_m` 或 `TOP_N=10`。
- `CoAgenticRetriever/scripts/train_coagentic_retriever_grpo.sh` 已改成退役 stub。这个旧入口原来从静态 tool config 读 ranker model/device/top-k，不再符合当前事实源划分，所以不允许继续作为训练入口。

#### Eval 链路当前状态

v2 eval 已经迁到 `scripts/coagenticRetriever_v2/02_infer_launcher.sh`、`scripts/coagenticRetriever_v2/assets/eval_launcher/compile_config.py`、`scripts/coagenticRetriever_v2/evaluate_coagentic_vllm.py` 这条链路。它只从 eval runtime、eval budget、resource、task overlay 和明确支持的 override 取值；runtime tool config 由 compiler 生成。

v2 目录里容易误读的 eval 入口、aligned evaluator 和 eval budget YAML 已移除，只保留一个 eval launcher 和一个 evaluator。推理报告、`.env`、`run_config.json` 和 dry-run 输出都使用 `EVAL_TASK_NAME`、`RECALL_FINAL_TOP_N`、`SEARCH_TOOL_FINAL_TOP_M`、`RANKER_FINAL_TOP_K`、`LLM_IO_MAX_RECORDS` 等规范字段。

#### overlay 会不会失效

目前分两种情况。

第一种是正确的新 Hydra 字段：

```yaml
recall_retriever:
  recall_final_top_n: 40

ranker:
  top_k: 40
```

这不会失效。canonical launcher 的顺序是：

1. 先把 base、overlay、run mode override、runtime override、CLI override compose 成最终 Hydra config。
2. 从最终 Hydra config 读取 `recall_retriever.recall_final_top_n` 和 `ranker.top_k`。
3. 生成本次 run 的 runtime tool config。
4. 再把 `actor_rollout_ref.rollout.multi_turn.tool_config_path` 指到这份 runtime tool config。

runtime override 只写 service URL、device、tool_config_path 这类运行态字段，不写 `recall_final_top_n` 或 `ranker.top_k`。所以 overlay 里正确设置的新字段不会被 runtime override 覆盖。

第二种是旧字段或写错层级：

```yaml
recall_retriever:
  top_k: 40

ranker:
  final_top_k: 40

default_top_n: 40
default_top_m: 5
searchTool_final_top_m: 5
```

这些不会再静默失效，而是直接失败：

- `recall_retriever.top_k`：拒绝，要求改成 `recall_retriever.recall_final_top_n`。
- `ranker.final_top_k`：拒绝，要求训练 Hydra 里写 `ranker.top_k`；`final_top_k` 只允许出现在 runtime tool config。
- 顶层 `default_top_n/default_top_m/searchTool_final_top_m`：拒绝，因为它们不是 Hydra 训练 override。现在 `searchTool_final_top_m` 的事实源仍然是静态 tool config。

因此当前实现不会导致“overlay 看起来生效、实际没生效”的 silent failure。更准确地说：正确 overlay 会生效，错误 overlay 会 fail-fast。

#### agent top-M 不是 ranker top-K

tool 真正返回给 agent 的文档数现在只看 `searchTool_final_top_m` 和实际 ranked docs 数量：

```python
agent_top_k = min(top_m, len(ranked_docs))
final_docs = ranked_docs[:agent_top_k]
```

当前 `searchTool_final_top_m=5`，所以 agent 仍然只看 5 篇。把 `ranker.top_k` 改成 50 不会让 agent 看 50 篇；它只是保证 ranker/judge/training 侧有 50 篇 rank-final 候选。

#### judge max-docs 仍然是单独语义

async ranker training 的 LLM judge stage 仍然写：

```yaml
# CoAgenticRetriever/config/experimental/async_ranker_training_base/async_ranker_training.yaml
stages:
  - type: llm_as_judge
    score_schema: ranked_ids_top50
    max_docs_per_request: 50
```

这表示每个 judge 请求评估 50 篇 ranked chunks。当前 async ranker training schema 里还有硬约束：`ranked_chunk_list` 必须正好有 50 篇。

所以现在应该守住这组关系：

```text
recall_retriever.recall_final_top_n = 50
ranker.top_k = 50
judge max_docs_per_request = 50
ranked_chunk_list length = 50
```

如果只把其中一个改成 20 或 100，full 模式就可能出现配置说法和 judge/schema 约束不一致。

#### positive_top_k 是标签规则

async sample builder 里还有：

```yaml
sample_builder:
  strategy_kwargs:
    signal_builder_type: llm_judge_topk
    positive_top_k: 5
```

这个 5 的意思是：LLM judge 排名前 5 的 doc 被当作 positive，其余 judged docs 可以作为 negative。它不是 agent top-M，也不是 ranker final top-K。它和 `searchTool_final_top_m=5` 数值一样，但语义不同。

#### 后续真正要补的校验

这项现在不再是“字段到处重复”的问题，已经收敛成“rank50 链路要一致”的问题。后续应补 full-mode 静态校验：

- `recall_retriever.recall_final_top_n` 必须能提供至少 50 篇给 rank50 judge。
- `ranker.top_k` 必须等于 rank50 judge 需要的文档数，当前就是 50。
- `max_docs_per_request` 如果不是 50，要么拒绝启动，要么要求显式切换到新的非 rank50 schema。
- `searchTool_final_top_m <= recall_retriever.recall_final_top_n` 必须成立。
- reward preflight 的 top-M 限制要和最终 `searchTool_final_top_m` 一致。

## 已收敛项

### 9. format_penalty 重复问题已收敛

这一项在 canonical v2 训练链路里已经解决，不再作为当前 P2 风险跟踪。

#### 当前事实源

最终 answer 格式惩罚只看 Hydra：

```yaml
# CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
custom_reward_function:
  reward_kwargs:
    format_penalty: -0.2
```

这个值会传给：

```text
CoAgenticRetriever/rewards/search_qa_f1_with_format_penalty.py
search_qa_f1_penalty_compute_score(..., format_penalty=-0.2)
```

reward 函数的语义仍然是：answer 格式正确且满足 search 约束时给 F1 分；否则给 `format_penalty`。所以 `custom_reward_function.reward_kwargs.format_penalty` 是最终 answer reward 的事实源。

#### 静态 tool config 现在不再写 format_penalty

当前静态 CoAgenticRetriever tool config 里没有 `format_penalty`：

```yaml
# CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
searchTool_final_top_m: 5
ranker_enabled: true
ranker:
  backend: ray_actor
  required: true
```

no-ranker 版本也是同样的事实源划分：

```yaml
# CoAgenticRetriever/config/coagentic_retriever_tool_config_no_ranker.yaml
searchTool_final_top_m: 5
ranker_enabled: false
ranker:
  backend: ray_actor
  required: false
```

`CoAgenticRetrieverTool` 现在也不再读取 `format_penalty`。tool reward 仍然由 hit/NDCG 这类 tool-side 指标计算；最终 answer 格式惩罚由 Hydra reward 函数负责。

#### canonical launcher 如何防止重复回来

v2 canonical launcher 读静态 tool config 时会拒绝这些旧字段：

```text
config.default_top_n
config.default_top_m
config.format_penalty
config.ranker.*  # backend/required 以外的 ranker 字段
```

也就是说，如果有人把 `format_penalty` 又写回 `CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`，canonical 训练启动前会直接失败，不会再出现“Hydra 一个值、tool config 一个值”的双事实源。

#### 审计现在怎么写

canonical `.env` 只记录 reward 侧事实源：

```text
REWARD_FORMAT_PENALTY=-0.2
REWARD_FORMAT_PENALTY_SOURCE=custom_reward_function.reward_kwargs.format_penalty
```

不再记录含义模糊的 `FORMAT_PENALTY`，也不再记录 `TOOL_CONFIG_FORMAT_PENALTY`。因为当前 tool config 已经不再维护这个字段。

#### 非 canonical 入口边界

本文不继续维护非 canonical 训练入口的配置语义。新训练任务应使用 `scripts/coagenticRetriever_v2/01_train_launcher.sh`，新评估任务应使用 `scripts/coagenticRetriever_v2/02_infer_launcher.sh`。

## 建议后续处理顺序

1. 先补 top/rank50 的 full-mode 一致性校验。
   - 字段删除、改名、日志/report 和 runtime 审计已经完成。
   - 剩下要校验的是 recall final top-N、ranker final top-K、judge max-docs、ranked chunk 数量必须一起满足 rank50 语义。
2. 保持 task 文档和任务入口只指向 v2 canonical launcher。
   - 当前 canonical 事实源已经在 v2 launcher 中收敛。
   - 非 canonical 入口不再作为新任务的配置事实源。

## 暂不处理事项

- 本文不改变当前 Hydra 覆盖优先级。
- 本文不判断应该把所有 top 字段统一成一个名字；它们不是同一个语义，不能简单合并。

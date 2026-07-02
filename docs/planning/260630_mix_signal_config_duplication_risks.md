# Mix Signal 配置重复与漂移隐患记录

本文现在只保留还需要继续判断的配置点。已经收敛、已经补校验、或者不再作为当前风险跟踪的内容，不在正文里展开。

本文的判断口径仍然是：

- overlay 覆盖 base 是正常机制，不算重复风险。
- 多个基础配置同时写同一个语义，才算 base/base 重复风险。
- 多个普通 overlay 同时改同一个实验语义，才算 overlay/overlay 重复风险。
- launcher 生成的 runtime override / run_mode override 是编译产物，不当作人工维护的第三份配置。

## 本次已执行的代码改动

第 484 行那条建议已经落地：审计文件现在会输出 ranker training 当前到底走哪条 sample builder 路径。

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

### 6. ranker 路径、设备和 token 长度分散在多处

这个点先保留，但不是本文这次的重点。

full 模式下有两个 ranker 角色：

- 训练侧 ranker：被 contrastive update 更新。
- 推理侧 shared ranker：rollout/search tool 调用它来重排 recall 候选。

这两边最好使用同一套 encoder 语义。现在 ranker 的模型路径、encoder 路径、device、query/doc 最大长度，既能在 Hydra ranker 配置里看到，也能在 tool config 的 `ranker` 段里看到。runtime override 会注入 Hydra 的 `ranker.device`，但不会自动改写静态 tool 模板里的所有 ranker 字段。

后续重点不是“看到重复就删”，而是先确认哪些字段还有运行意义：

- `ranker.max_query_length/max_doc_length`：训练侧和 shared ranker 推理侧应保持一致。
- tool config 里的 `ranker.max_query_length/max_doc_length`：full 模式下 tool 会把它们传给 ranker actor 的调用路径，不能随便当成纯注释。
- tool config 里的 `ranker.model_path/encoder_path/device`：ray actor 模式下主要是模板残留还是仍被某些调试路径消费，需要单独确认。

建议后续补一个 full-mode 静态校验：最终 Hydra 的 ranker token length 要和运行时 tool config 的 ranker token length 一致。

### 7. top-N / top-M / top-K / max-docs 名字相似但不是一回事

这是本文现在最需要讲清楚的点之一。这里的问题不是“有很多 top 字段所以一定错”，而是这些字段名字太像，但处在不同阶段，改错一个数字会悄悄改变实验语义。

先把一条 search tool call 的链路按顺序拆开：

1. agent 生成一个 search query。
2. tool 用 `top_n` 去 recall 服务拿候选文档。
3. 如果 no-ranker，就保持 recall 原顺序；如果 full ranker，就对这批候选做 dense rerank。
4. tool 只把前 `top_m` 篇左右的文档格式化进 tool response，让 agent 继续生成。
5. full async ranker training 还会把 rank 后的 50 篇文档交给 LLM judge，让 judge 排序并产出 ranker 训练信号。

这里至少有六组“看起来像 top-k”的配置。

| 名字 | 当前值 | 真实含义 | 主要消费者 |
| --- | ---: | --- | --- |
| tool `default_top_n` | 50 | 每次 recall 请求返回多少篇候选 | `CoAgenticRetrieverTool.execute()` 调 recall API |
| tool `default_top_m` | 5 | 最终塞进 tool response、agent 真正看见多少篇 | `final_docs = ranked_docs[:agent_top_k]` |
| tool `ranker.top_k` | 50 | tool 侧最终可见文档的额外上限之一 | `min(top_m, ranker.top_k, len(ranked_docs))` |
| Hydra `recall_retriever.top_k` | 50 | 训练侧补 ranker trace 时最多取多少篇 recall docs | ranker trainer enrichment |
| Hydra `ranker.top_k` | 5 | 训练侧写 `rank_top5_docs` 时保留多少篇 | ranker trainer enrichment |
| judge `max_docs_per_request` | 50 | 每个 LLM judge 请求应该评估多少篇文档 | async ranker training request/prompt |
| sample builder `positive_top_k` | 5 | judge 排名前几篇当作 positive | `llm_judge_topk` signal builder |

#### recall top-N 是候选池大小

tool config 里的 `default_top_n: 50` 是第一层候选池大小。tool 执行时会这样取值：

- 如果 `create_kwargs` 显式传了 `top_n`，就用它。
- 否则用 tool config 的 `default_top_n`。

然后 tool 调 recall 服务，并把结果记录成类似：

- `recall_top50_docs = recall_docs[:top_n]`

所以这里的 50 不是 agent 看到的文档数，而是后面 ranker、judge、训练信号能使用的候选池大小。

#### agent top-M 是 agent 可见文档数

tool config 里的 `default_top_m: 5` 控制最终进入 tool response 的文档数。当前代码里最终可见文档数是：

```text
agent_top_k = min(top_m, tool_ranker_top_k, len(ranked_docs))
final_docs = ranked_docs[:agent_top_k]
```

当前配置下：

- `top_m = 5`
- tool `ranker.top_k = 50`
- `ranked_docs` 通常最多 50 篇

所以 agent 实际看到的是 5 篇，不是 50 篇。这个 5 会影响 agent 后续回答时能引用多少上下文，也会影响 reward preflight 对“可见文档”的判断。

#### tool `ranker.top_k` 现在不是 rerank 池大小

这是最容易误读的地方。

tool config 里有：

- `ranker.top_k: 50`

看名字像“ranker 重排 top-K”，但当前 tool 代码会对 recall top-N 全量重排。local 路径显式传的是 `top_k=len(recall_docs)`，ray actor 路径也是围绕同一批 recall docs 做排序。tool `ranker.top_k` 最后只参与 agent 可见文档的 cap：

```text
min(default_top_m, ranker.top_k, 文档数)
```

因此，当前 `ranker.top_k=50` 不代表“agent 看 50 篇”，也不代表“judge 看 50 篇”。它只是保证不会比 50 更多；真正让 agent 只看 5 篇的是 `default_top_m=5`。

#### Hydra `ranker.top_k` 是训练侧 trace 的 top5 语义

Hydra ranker base 里也有：

- `ranker.top_k: 5`

这和 tool config 的 `ranker.top_k: 50` 名字一样，但语义不同。训练侧在补 ranker trace 时会：

1. 从 tool detail 里拿 recall docs。
2. 按 `recall_retriever.top_k` 截到 50。
3. 调训练侧 ranker 排序全部 50 篇。
4. 写：
   - `rank_top50_docs = rank_top50`
   - `rank_top5_docs = rank_top50[:ranker.top_k]`

所以 Hydra `ranker.top_k=5` 更接近“训练日志和 contrastive 样本里保留的 top5 视图”。它不应该被拿去解释 recall 候选池，也不应该被拿去解释 LLM judge 一次看多少篇。

#### judge max-docs 不是 agent top-M

async ranker training 的 LLM judge stage 里有：

- `max_docs_per_request: 50`

这表示每个 judge 请求评估 50 篇 ranked chunks。更关键的是，当前 async ranker training schema 里有硬约束：

```text
ranked_chunk_list must contain exactly 50 chunks
```

也就是说，当前这条链路不是“最多 50 篇也行，少一点也行”。它实际上要求 rank50。prompt、parser、request validation、`rank_top50_docs` 命名都是按 rank50 设计的。

所以如果有人把 recall top-N 从 50 改成 20，但不改 async judge 链路，full 模式会直接不满足 rank50 请求语义；如果有人把 judge `max_docs_per_request` 改成 20，但 schema 仍然要求 50，也会出现配置说法和代码约束不一致。

#### positive_top_k 是标签规则，不是可见文档数

async sample builder 里还有：

- `strategy_kwargs.positive_top_k: 5`

这个 5 的意思是：LLM judge 排名前 5 的 doc 被当作 positive，其余 judged docs 可以作为 negative。它不是 agent top-M，也不是 `rank_top5_docs` 的唯一来源。

当前它和 agent top-M 都是 5，只是数值碰巧一样。语义上它属于“训练标签规则”，不是“tool response 展示规则”。

#### 当前应该守住的约束

如果继续保持 rank50 judge 设计，至少要守住这些关系：

- `tool default_top_n` 应该是 50。
- Hydra `recall_retriever.top_k` 应该是 50。
- `rank_top50_docs` 应该真的有 50 篇。
- judge `max_docs_per_request` 应该是 50，或者相关 schema/prompt/parser 一起改。
- agent top-M 可以是 5，但不要把它误认为 judge 输入大小。
- Hydra `ranker.top_k=5` 可以继续表达训练侧 top5 trace，但不要和 tool `ranker.top_k=50` 混用。
- `positive_top_k=5` 是 judge 标签策略；改它会改变 ranker 训练正样本定义。

#### 改这些数字时应该怎么改

如果目标是“agent 多看几篇文档”，优先改：

- tool `default_top_m`
- reward/preflight 对可见文档数的限制
- rollout token budget，确保更多 tool response 放得下

不要只改 judge `max_docs_per_request`，因为 judge 看多少篇不决定 agent 看多少篇。

如果目标是“judge 不再评估 50 篇，而是评估 20 篇”，需要成组修改：

- stage `max_docs_per_request`
- request schema 的 rank50 校验
- prompt 里对输出数量的要求
- parser/validator 对返回 doc_id 数量的要求
- `rank_top50_docs` 这类字段名或兼容逻辑

不要只改一个 YAML 字段。

如果目标是“recall 候选池扩大到 100”，要先决定 judge 是否也升级到 rank100：

- 如果 judge 仍是 rank50，就需要明确从 100 篇里选哪 50 篇给 judge。
- 如果 judge 也变 rank100，就要同步改 schema、prompt、parser、日志字段名和显存/上下文预算。

#### 建议的后续治理

短期建议先增强审计，而不是马上改数字：

- `.env` 或 final summary 里把 `RECALL_TOP_K/TOP_N/TOP_M`、Hydra `recall_retriever.top_k`、Hydra `ranker.top_k`、tool `ranker.top_k`、judge `max_docs_per_request`、`positive_top_k` 分开列。
- 字段展示时不要只写 `top_k=5`，要写成 `agent_visible_top_m=5`、`judge_docs_per_request=50`、`label_positive_top_k=5` 这种带语义的名字。

中期建议补 full-mode 静态校验：

- rank50 judge 启用时，最终 `rank_top50_docs` 来源必须能提供 50 篇。
- `max_docs_per_request` 如果不是 50，要么拒绝启动，要么要求显式切换到新的非 rank50 schema。
- `default_top_m <= default_top_n` 必须成立。
- reward preflight 的 top-M 限制要和最终 agent-visible 文档数一致。

## P2 隐患

### 9. format_penalty 在 reward 配置和 tool config 里重复

这是本文另一个重点。它看起来只是一个同名数字，但真正的问题是：同名字段在两个地方出现，读者会自然以为它们都控制同一件事；而当前代码里，它们的运行语义并不等价。

#### 先说结论

当前最终 answer 格式惩罚的事实源是 Hydra：

- `custom_reward_function.reward_kwargs.format_penalty`

当前值是：

- `-0.2`

tool config 里也有：

- `tools[0].config.format_penalty`

当前值也是：

- `-0.2`

因为两个值现在相同，所以不会立刻暴露问题。但这不是一个好的长期状态：以后如果只改其中一个，训练实际行为和审计展示就可能分叉。

#### Hydra 里的 format_penalty 控制最终 answer reward

Hydra trainer config 里配置了自定义 reward：

- 文件：`CoAgenticRetriever/rewards/search_qa_f1_with_format_penalty.py`
- 函数：`search_qa_f1_penalty_compute_score`
- 参数：`custom_reward_function.reward_kwargs.format_penalty`

这个 reward 函数会检查最终 answer 格式，以及是否满足至少一次 search 等约束。当前逻辑大致是：

```text
如果 answer 格式正确，并且满足 search 约束：
    score = F1 answer score
否则：
    score = format_penalty
```

所以 Hydra 的 `format_penalty=-0.2` 是最终 answer reward 的实际惩罚值。它会影响训练优化目标。

#### tool config 里的 format_penalty 当前不是最终 answer penalty

tool config 也写了 `format_penalty: -0.2`。tool 类初始化时确实会读取它：

```text
self.format_penalty = float(_require_config(config, "format_penalty"))
```

但当前 `CoAgenticRetrieverTool._compute_tool_reward()` 只根据 hit/NDCG 计算 tool reward，没有使用 `self.format_penalty`。也就是说，在当前代码路径里，tool config 的 `format_penalty` 主要有两个作用：

- 满足 tool config schema/初始化要求，因为 tool 类现在要求这个字段存在。
- 被 launcher 读出来写进 env 审计里的 `FORMAT_PENALTY`。

它不是最终 answer reward 的事实源。

#### 为什么这会误导审计

假设以后有人只改了 Hydra：

```text
custom_reward_function.reward_kwargs.format_penalty = -0.5
tools[0].config.format_penalty = -0.2
```

训练真正使用的是 `-0.5`，但 `.env` 里当前的 `FORMAT_PENALTY` 来自 tool config，就可能显示 `-0.2`。读日志的人会误以为格式错误只扣 `-0.2`。

反过来，如果有人只改 tool config：

```text
custom_reward_function.reward_kwargs.format_penalty = -0.2
tools[0].config.format_penalty = -0.5
```

训练最终 answer reward 仍然按 `-0.2` 扣分，但审计 env 可能显示 `-0.5`。这会让实验复盘和指标解释都变得不可信。

#### 为什么不能现在直接删 tool config 字段

不能只从 YAML 里删掉 `tools[0].config.format_penalty`，原因很简单：tool 类当前用 `_require_config(config, "format_penalty")` 强制读取这个字段。直接删 YAML 会导致 tool 初始化失败。

所以处理这个重复字段有两条安全路线。

#### 路线 A：保留字段，但让 Hydra 生成它

这是最稳的短期方案。

做法是继续沿用“静态 tool config 只做模板、canonical launcher 生成 runtime tool config”的模式：静态 tool config 不再手写 `format_penalty`，canonical launcher 在生成 runtime tool config 时，从最终 Hydra 里读取：

```text
custom_reward_function.reward_kwargs.format_penalty
```

然后写进本次 run 的 runtime tool config。

这样有几个好处：

- tool 类仍然能读到 `format_penalty`，不会破坏初始化。
- 静态 tool config 不再是第二事实源。
- `.env` 可以同时写清楚：
  - `REWARD_FORMAT_PENALTY=-0.2`
  - `TOOL_CONFIG_FORMAT_PENALTY=-0.2`
  - `FORMAT_PENALTY_SOURCE=hydra_reward_kwargs`
- 如果将来 tool reward 真的也要用这个惩罚，两边仍然来自同一个 Hydra 值。

这条路线的缺点是：tool config 里仍然会出现字段，只是它变成 launcher 生成结果，不再是人工维护的事实源。

#### 路线 B：确认 tool 不再需要后，删除 tool 字段

这是更干净但更需要确认的方案。

需要先改代码：

- `CoAgenticRetrieverTool` 不再 `_require_config(config, "format_penalty")`。
- launcher 不再从 tool config 读 `format_penalty` 作为 `FORMAT_PENALTY`。
- 推理脚本里如果只是打印 `STATIC_FORMAT_PENALTY`，也要同步删除或改名。
- 审计文件只记录 Hydra reward kwargs 里的 `format_penalty`。

这条路线的优点是事实源最清楚：格式惩罚就是 reward 配置。缺点是要确认没有旧工具、旧推理脚本或外部流程还依赖 tool config 的这个字段。

#### 当前建议

短期建议走路线 A：让 Hydra reward kwargs 成为唯一事实源，同时由 launcher 生成 runtime tool config 里的 `format_penalty`。这能最快消除“人工维护两份同名字段”的问题，又不会破坏 tool 初始化。

中期再决定是否走路线 B：如果确认 tool reward 永远不需要单独的 format penalty，就把 tool schema 和静态模板里的字段彻底删掉。

#### 审计上应该怎么写

不要再只写一个模糊的：

```text
FORMAT_PENALTY=-0.2
```

更清楚的审计应该是：

```text
REWARD_FORMAT_PENALTY=-0.2
TOOL_CONFIG_FORMAT_PENALTY=-0.2
FORMAT_PENALTY_SOURCE=hydra_reward_kwargs
```

这样读者一眼能看出：最终 answer reward 看 Hydra；tool config 里的值只是为了 runtime tool schema 保持一致。

## 建议后续处理顺序

1. 先处理 `format_penalty` 的事实源。
   - 推荐短期做法：runtime tool config 的 `format_penalty` 由 Hydra reward kwargs 生成。
   - 同时把审计字段拆成 reward/tool/source 三个名字，避免 `FORMAT_PENALTY` 继续含糊。
2. 再处理 top 字段审计。
   - 不急着改数值，先把每个 top 字段按语义输出清楚。
   - full-mode 启动前补 rank50 judge 相关一致性校验。
3. 最后处理 ranker token length。
   - 校验训练侧 ranker 和 tool/shared ranker 的 query/doc token length 是否一致。

## 暂不处理事项

- 本文不改变当前 Hydra 覆盖优先级。
- 本文不判断应该把所有 top 字段统一成一个名字；它们不是同一个语义，不能简单合并。

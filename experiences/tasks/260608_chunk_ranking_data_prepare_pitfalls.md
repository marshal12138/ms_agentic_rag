# 260608 Chunk Ranking 数据制备任务踩坑记录

## 任务目标

利用 `02_infer_qwen3_4b_ablation_val_only.sh` 中的 agent rollout 和 retriever 调用，生成用于 LLM-as-judge 的 chunk ranking 数据：

```json
{"origin_query": "...", "sub_query": "...", "passage_list_top50": [...]}
```

字段来源：

- `origin_query`: train data 中的原始问题，在 validation dump 里对应 `initial_query`。
- `sub_query`: agent rollout 中调用 search tool 时产生的 query。
- `passage_list_top50`: retriever 对 `sub_query` 返回的 top50 排序结果。

## 1. checkpoint 路径要用 global_step 父目录，不是 actor 子目录

正确路径：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79
```

不要传：

```text
.../global_step_79/actor
```

VERL resume 逻辑会自己进入 `actor/` 查找分片。

## 2. 该 checkpoint 没有 optimizer 分片，val_only 要只加载 model

症状：

```text
FileNotFoundError: optim_world_size_4_rank_*.pt
```

原因：

- `global_step_79/actor/` 下只有：
  - `model_world_size_4_rank_*.pt`
  - `extra_state_world_size_4_rank_*.pt`
  - `fsdp_config.json`
- 没有 `optim_world_size_4_rank_*.pt`。

正确处理：

```text
actor_rollout_ref.actor.checkpoint.load_contents=['model']
```

本任务是 infer / val_only，不需要恢复 optimizer。

## 3. “不使用 LLM reranker”要同时关闭工具 reranker 和双 agent reranker server

只设置 tool config 里的 `use_reranker: false` 还不够。

必须同时满足：

```yaml
use_reranker: false
save_top_n_documents: true
```

以及 Hydra 覆盖：

```text
+trainer.disable_reranker_rollout=true
```

否则即使 tool 不调用 reranker，双 agent 框架仍可能初始化固定 reranker vLLM server，导致显存额外占用和启动失败。

确认方式：

- 日志里应出现：`Reranker rollout disabled`
- 不应出现：`Starting reranker server initialization`
- CoSearchTool preflight metrics 里 `reranker_success` 应为 `false`

反思：

- `use_reranker: false` 是 tool-level retrieval-only 开关，不是完整的 no-llm-reranker 训练 / 推理模式。
- `reranker_actor_rollout_ref.trainable=false` 只表示固定 reranker / 不训练 reranker，也不是“不使用 reranker”。
- no-llm-reranker 应是一等 ablation 配置，而不是临时 override。主入口、resource pool、role mapping、worker init、agent loop、metrics 都要有一致分支。

## 3.1 不要按报错点逐个补 no-reranker 分支

错误做法：

- 先加 `disable_reranker_rollout`。
- 跑到 role assertion 报错，再补 assertion。
- 跑到 worker 创建报错，再补 worker。
- 跑到 agent loop handles 报错，再补 handles。

问题：

- 这是逐报错补洞，不是完整设计。
- 容易遗漏后续路径，例如 metrics、reward trace、checkpoint、resource allocation。

正确处理：

第一次实现 no-reranker 模式时，应完整梳理并一次性处理：

1. CLI / Hydra 配置项，例如 `trainer.disable_reranker_rollout=true`。
2. `main_co_search_ppo.py` 是否注册 reranker role。
3. resource pool 是否还拆分 reranker pool。
4. trainer constructor 是否仍要求 reranker role。
5. `init_workers()` 是否仍创建 reranker worker group。
6. `CoSearchAgentLoopManager` 是否允许 `reranker_worker_group=None`。
7. tool config 是否为 `use_reranker:false`。
8. validation / reward metrics 是否会对非数值 trace 求均值。

## 3.2 补丁边界：不要误改 CoSearch 源项目

本任务涉及三个容易混淆的代码区：

```text
CoSearch_derevitives/CoSearch/                    # 外部源项目 / 共享项目
CoAgenticRtriver/CoSearch/                  # 本项目原始兼容副本，只作参考和回退
CoAgenticRtriver/CoAgenticRtriver/          # 本项目当前实际使用的核心副本
```

正确处理：

- 默认只改 `CoAgenticRtriver/CoAgenticRtriver/` 和本任务 pipeline。
- 不要修改 `CoSearch_derevitives/CoSearch/`，除非用户明确要求改源项目。
- 不要修改 `CoAgenticRtriver/CoSearch/`，除非确认当前脚本实际使用该兼容副本。
- 每次补丁后，用嵌套 git 仓库分别检查：

```bash
git -C /data01/ms_wksp/agent_up_to_date/CoAgenticRtriver/CoAgenticRtriver status --short
git -C /data01/ms_wksp/agent_up_to_date/CoAgenticRtriver/CoSearch status --short
git -C /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoSearch status --short
```

注意：

- 这些仓库可能本来就是 dirty。不要把已有 dirty 状态误判为本次改动。
- 判断自己是否影响源项目，要以本次实际写操作、`apply_patch` 目标路径和新增文件路径为准。

## 4. top50 passage 不要重新写 retriever 逻辑，直接打开 CoSearchTool trace

`CoSearchTool` 已支持保存 top-n documents。任务需要的配置是：

```yaml
default_top_n: 50
default_top_m: 5
save_top_n_documents: true
use_reranker: false
```

agent loop 会把 search tool 的结果记录进：

```text
tool_call_details[].top_50_documents
```

抽取时从这里取 `passage_list_top50`。

## 5. validation dump 里的 trace 字段在顶层，不一定在 reward_extra_info 中

本次 dump 文件：

```text
.../infer_out/validation_data/79.jsonl
```

每行关键字段实际在顶层：

```text
input
output
gts
score
step
reward
valid
f1
tool_call_details
initial_query
answers
```

抽取脚本应优先读：

```text
row["initial_query"]
row["tool_call_details"]
```

不要只从 `row["reward_extra_info"]` 里找。

## 6. 非数值 trace 会污染 validation metrics 汇总

症状：

```text
TypeError: unsupported operand type(s) for /: 'dict' and 'int'
```

原因：

- wrapper reward 把 `tool_call_details` 这类 list/dict trace 放进 reward extra info。
- validation metrics 后处理会对 extra info 里的变量求 `np.mean`，遇到 dict/list 就报错。

影响：

- 报错发生在 validation generation 和 dump 写出之后。
- 本次 `79.jsonl` 已完整写出 180 行，因此仍可直接抽取数据。

正确处理：

- 如果只需要数据集，可在 dump 写出后直接运行抽取脚本。
- 如果需要完整无报错结束，需要过滤 metrics 汇总里的非数值字段，或只把 trace 放到不会参与 metrics mean 的行级 dump 字段中。

## 7. 抽取规则要严格保证可用样本

每条样本必须满足：

- `origin_query` 非空。
- `sub_query` 非空。
- `passage_list_top50` 是 list。
- `len(passage_list_top50) >= 50`。
- 按 `(origin_query, sub_query)` 去重。

最终只保留：

```text
origin_query
sub_query
passage_list_top50
```

## 8. 每 10 条先落盘，最终只保留合并文件

抽取阶段按 10 条写临时分片：

```text
part_000010.jsonl
part_000020.jsonl
...
```

达到目标条数后合并为：

```text
chunk_ranking_judge_examples_100.jsonl
```

然后删除 `part_*.jsonl`，输出目录最终只保留合并后的数据集。

## 9. 本任务已跑通的稳定参数

在当前机器已有显存占用的情况下，稳定跑通参数为：

```bash
VAL_MAX_SAMPLES=180
SUBSET_MAX_SAMPLES=240
VAL_BATCH_SIZE=4
TRAIN_BATCH_SIZE=4
TRAIN_MAX_SAMPLES=4
ACTOR_BATCH_SIZE=4
AGENT_WORKERS=4
TOOL_MAX_CONCURRENT_PER_WORKER=4
MAX_NUM_SEQS=4
RETRIEVER_INSTANCES=1
GPU_IDS=0,1,2,3
GPU_MEMORY_UTILIZATION=0.30
```

原计划并行 5 在当前显存状态下不稳，实际用 4 并发完成。

## 复用检查清单

下次重复 chunk ranking 数据制备前先确认：

1. `RESUME_FROM_PATH` 指向 `global_step_*` 父目录。
2. infer 覆盖了 `actor_rollout_ref.actor.checkpoint.load_contents=['model']`。
3. tool config 里 `use_reranker: false` 且 `save_top_n_documents: true`。
4. Hydra 覆盖了 `+trainer.disable_reranker_rollout=true`。
5. no-reranker 是否完整覆盖 role mapping、resource pool、worker init、agent loop、metrics。
6. 禁用 reranker 后是否先检查 checkpoint world size，再决定是否使用 8 卡。
7. retriever/proxy 是否已经可复用，避免每次重启。
8. extractor 从顶层 `initial_query` / `tool_call_details` 读字段。
9. 最终校验 `wc -l == 100` 且所有 `passage_list_top50` 长度为 50。

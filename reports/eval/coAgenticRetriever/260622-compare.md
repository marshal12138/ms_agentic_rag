结论：这两次评估里没有看到 agent/ranker 模型加载异常。`full` 确实加载了训练后的 ranker；效果没提升主要不是 prefix 或 checkpoint 没加载，而是 dense ranker 的重排改变太激进，并且训练目标和 agentic RAG 最终收益不完全一致。

**加载行为**

- `full` 报告中：
  - Agent model: `global_step_79/hf_safetensors/actor`
  - Ranker base/tokenizer: `models/retriever/e5-base-v2`
  - Ranker encoder: `global_step_79/ranker/rank_encoder`
- `no-ranker` 报告中：
  - Agent model 是同一个 `global_step_79/hf_safetensors/actor`
  - Ranker: `not used`

`run_config.json` 也一致：`full` 的 `ranker_encoder` 指向训练 checkpoint 下的 `ranker/rank_encoder`，`no-ranker` 是 `None`。vLLM 日志显示两个 run 都加载同一个 actor。trace 里 `full` 的 347 次 ranker stage 全部 `ranker_success=true`，可见 top5 里也都有 `rank_score/rank_rank` 字段，所以 ranker 实际参与了排序。

另一个细节：`.env` 里有 `RANK_GPU_ID=2`，但实际 `RANKER_DEVICE=cuda:4`，这是因为 launcher 从 static tool config 覆盖了 ranker device。由于本次 ranker 全部成功，这不是效果下降原因，但这个配置气味后面应该清理。

**效果差异**

`full` vs `no-ranker`：

- EM: `0.3086` vs `0.3171`，下降 `0.0086`
- F1: `0.3928` vs `0.3974`，下降 `0.0046`

这不是全面崩掉，而是数据集之间抵消：

- 变好：`nq +0.0817 F1`，`popqa +0.0248`，`musique +0.0271`，`2wikimultihopqa +0.0110`
- 变差：`bamboogle -0.0950`，`triviaqa -0.0806`，`hotpotqa -0.0011`

逐样本上：

- F1 变好：40 条
- F1 变差：36 条
- 不变：274 条

所以它不是完全无效，而是收益和损伤互相抵消，最后略亏。

**为什么加 Dense Ranker 没效果**

最核心的问题是 ranker 改动太大。`full` 的最终 top5：

- 347/350 个样本 top5 顺序与 no-ranker 不同
- 344/350 个样本 top5 文档集合发生变化
- 343/350 个样本把 recall rank > 5 的文档拉进最终 top5
- 平均每个样本 top5 里有 `2.74` 个文档来自 recall rank > 5
- 217/350 个样本的最终 top1 不是 recall rank 1

也就是说，ranker 不是轻微校正 recall，而是在大规模替换 agent 可见证据。只要 ranker 没显著强于 recall，这种 top5 替换就会带来大量正负抵消。

训练侧还有目标不一致：训练配置里 `disable_reranker_rollout: true`，ranker 训练信号来自 async LLM judge 的 `rank50` 伪标签，最终样本构造用的是 `llm_judge_top5`。这让 ranker 学的是“LLM judge 认为 top50 中哪些单段更相关”，不是“哪些 5 篇组合起来最能让当前 agent 答对”。多跳/比较题尤其容易受伤，因为 agent 需要互补证据集合，而 dense ranker 更像单文档相关性排序器。

E5 prefix 这次是对齐的：eval ranker 会把 query 格式化成 `query: ...`，doc 格式化成 `passage: ...`；训练 worker 里也是同一套逻辑。证据在 [evaluate_coagentic_vllm.py](/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/evaluate_coagentic_vllm.py:611) 和 [e5_ranker_worker.py](/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/verl/verl/workers/ranker/e5_ranker_worker.py:69)。

建议下一步不要先继续训练更久，而是先做一个更稳的 ablation：`recall_score/rank_score` 融合排序，例如 `final = alpha * rank_score + (1-alpha) * recall_score`，扫 `alpha=0.2/0.4/0.6/0.8`；再对比 `ranker_top_k=10/20/50` 和 `top_m=5`。现在的 ranker 太容易把 recall top5 里的有效证据挤掉。
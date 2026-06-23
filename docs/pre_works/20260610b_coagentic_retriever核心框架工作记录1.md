## USER: 

conagenticRetriever框架术语统一：接下来，为了防止我们造成误会，我们统一一下conagenticRetriever 中的术语： 
1. "retriever"：现在我们明确定义retriever就是retriever服务，也就是recall retriever，在agentic rag流程中它不训练，只做top50 passage(或称doc、chunk)的召回；
2. "ranker": 特指我们在conagenticRetriever 加入的 dense ranker，目前默认是用e5-base-v2模型，它是将recall retriever输出的top50 doc进行精准排序，输出50 doc的新排序；在这之后会将新排序的50 doc截断top5，输出给agent llm；
3. "rerank"或"reranker":这是cosearch路线遗留下来的词，因为我们当前的conagenticRetriever是根据cosearch路线改动出来的，reranker原本是指cosearch中使用的llm reranker；我需要你把相关配置、代码、文档全部剔除；
4. "cosearch"：正如上一条所说，我们当前的conagenticRetriever是根据cosearch路线改动出来的，所以conagenticRetriever核心框架中不可避免地留下很多含有cosearch名称的代码，你将它们全部替换成conagenticRetriever，以免造成误解；
行动：
1. 那么首先，此checkpoint目录的名称存在问题，/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_retriever_contrastive_smoke/global_step_79/retriever, 其目录名不应该叫retriever，而是叫ranker
2. 根据我上面的讲述，将retriever、ranker的配置名称和代码中的指代整理一下；
3. 剔除"rerank"或"reranker"、以及"cosearch"的痕迹；
先别动手，你先从头到尾地梳理一遍，告诉我你的详细修改计划，和我确认后再动手


## Assisstant: 
我还没有动任何文件。梳理后我建议按“严格清理当前 CoAgenticRetriever 核心框架，历史 CoSearch 副本和生成日志不改”的方式做。

**当前发现**
1. checkpoint 确实命名错误：  
`.../global_step_79/retriever/` 里保存的是 dense ranker encoder：`rank_encoder/model.safetensors` 等，应改成 `.../global_step_79/ranker/`。

2. 当前新入口已经有一部分新术语：  
[main_coagentic_retriever.py](/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/main_coagentic_retriever.py:1) 和 [coagentic_ranker_contrastive_ray_trainer.py](/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py:1) 使用了 `ranker`，checkpoint 保存新路径也已经写成 `"ranker"`。

3. 但旧兼容层还很多：  
`retriever_training`、`rank_retriever`、`rerank_top50_docs`、`reranker_*`、`CoSearchRayTrainer`、`co_search_tool.py`、`co_search_agent_loop.py`、`main_co_search_ppo.py` 等仍在当前框架中。

4. 需要特别注意一点：现在 dense ranker 主要是在 rollout 后由 trainer 补写 rank trace：`_enrich_tool_calls_with_ranker()`。如果严格按照你定义的“ranker 排序 top50 后截断 top5 给 agent llm”，还需要把 dense ranker 接入 agent tool 的实时返回链路，而不是只做训练样本构造后的补写。

**修改计划**
1. checkpoint 迁移：
   - 将 `/global_step_79/retriever` 移动为 `/global_step_79/ranker`。
   - 更新 docs/eval/script 中对 `retriever/rank_encoder` 的引用为 `ranker/rank_encoder`。
   - 不保留 `retriever` 兼容目录，避免继续误导。

2. 配置统一：
   - 保留 `recall_retriever`，只表示冻结 recall retriever 服务。
   - 统一使用 `ranker` 和 `ranker_training`。
   - 删除/替换 `retriever_contrastive.yaml`，只保留 `ranker_contrastive.yaml`。
   - 删除 `rank_retriever.*`、`rank_retriever_training.*`、`retriever_training.*` 的 fallback。
   - 将 `rerank_top_k` 改为类似 `agent_top_k: 5`，避免 “rerank” 残留。

3. 代码命名统一：
   - `retriever_strategies/` 真正内容改为 `ranker_strategies/`，类名从 `RetrieverContrastive*` 改为 `RankerContrastive*`。
   - `rerank_top50_docs` 改为 `rank_top50_docs`。
   - `rerank_top5_docs` 改为 `rank_top5_docs`。
   - `rerank_top50_sample` 改为 `rank_top50_sample`。
   - `rerank_top5_sample` 改为 `rank_top5_sample`。
   - 删除 `process_retriever_contrastive_step`、`update_retriever_contrastive`、`E5RetrieverContrastiveWorker`、`LocalRetrieverContrastiveWorker` 等旧 alias。
   - 删除 `verl/workers/retriever/` 这个 dense ranker 兼容 wrapper；真正的 retriever 只保留服务侧代码。

4. 剔除 LLM reranker 路线：
   - 删除或从当前入口断开 `reranker_dataset.py`。
   - 删除 `config/*reranker*`、`actor/reranker_dp_actor.yaml`、`rollout/reranker_rollout.yaml`、`model/reranker_hf_model.yaml`、`ref/reranker_dp_ref.yaml` 等旧配置。
   - 删除/断开 `RERANKER_AGENT_SETUP.md`、`reranker_reward_manager.py`、`reranker_reward_functions.py`、`custom_rewards/reranker_rewards.py`、`experimental/reward/reward_loop/reranker*`。
   - 当前 dense ranker 不再使用 LLM reranker actor/rollout/ref 资源池。

5. 剔除 CoSearch 命名：
   - `co_search_trainer.yaml` 改为 `coagentic_retriever_trainer.yaml`。
   - `co_search_agent_loop_config.yaml` 改为 `coagentic_retriever_agent_loop_config.yaml`。
   - `co_search_agent_loop.py` 改为 `coagentic_retriever_agent_loop.py`。
   - `CoSearchAgentLoop` 改为 `CoAgenticRetrieverAgentLoop`。
   - registry name 从 `co_search_agent` 改为 `coagentic_retriever_agent`。
   - `co_search_tool.py` 改为 `coagentic_retriever_tool.py`，class 从 `CoSearchTool` 改为 `CoAgenticRetrieverTool`。
   - `CoSearchRayTrainer` 在当前路径中改为 `CoAgenticRetrieverRayTrainer`，并清掉旧 alias。

6. 文档和脚本：
   - 更新 CoAgenticRetriever README、当前框架总结、eval 文档、pipeline 中活跃脚本的术语。
   - `co_search_tool_config.yaml` 改为 `coagentic_retriever_tool_config.yaml`。
   - `COSEARCH_*` 环境变量改为 `COAGENTIC_RETRIEVER_*`。
   - 历史运行产物 `outputs/`、`logs/`、早期 CoSearch 复现文档我建议不批量改，否则会破坏实验记录的可追溯性。

7. 验证：
   - grep 当前核心路径，确保活跃代码/config/docs 中没有 `rerank|reranker|co_search|cosearch|CoSearch`。
   - grep `retriever`，只允许出现在 `recall_retriever`、retriever 服务、模型路径 `/models/retriever/e5-base-v2`、以及框架名 `CoAgenticRetriever`。
   - 运行 Python compile/import 检查；GPU 训练不默认跑，只做可执行入口和配置解析检查。

需要你确认三点后我再动手：

1. 是否按“严格清理”执行，不保留旧字段/旧 alias 的兼容？
2. `CoAgenticRetriever/outputs/`、外层 `logs/`、历史 `CoSearch/` 目录和早期复现文档是否排除在本次改名之外？
3. 是否要同时修正行为：让 dense ranker 的 top5 真正进入 agent llm 的 tool response，而不仅仅是 trainer 事后补写 rank trace？

## USER

1. 严格清理; 2. 排除; 3. 修正

## Assisstant:

收到。我会按严格清理执行：当前 CoAgenticRetriever 核心代码/配置/活跃脚本/当前框架文档会改，CoAgenticRetriever/outputs/、外层 logs/、历史 CoSearch/ 目录和早期复现记录不动；同时把 dense ranker 接入工具返回链路，让 agent 看到 ranker top5。

我先从活跃链路下手：main_coagentic_retriever.py -> trainer -> agent_loop -> tool。这里必须先确认工具实例在什么进程里拿到配置，否则把 ranker 接进 top5 返回可能会出现 GPU/配置不一致。


Explored 4 files
我会先把改动范围固定成：CoAgenticRetriever 下的活跃入口、config、verl/verl/tools、verl/verl/experimental/agent_loop、verl/verl/trainer/ppo、verl/verl/workers/ranker、ranker_strategies，再同步外层当前评测/训练脚本中直接引用这些入口的部分。历史输出和旧复现记录不改。


Explored 2 files, 1 search
我准备把旧双 agent / LLM reranker 路线从当前入口断开，而不是做表面替换。核心做法是让 CoAgenticRankerContrastiveRayTrainer 继承一个只服务 main agent + dense ranker 的 base trainer，并把 tool 改成 recall retriever + dense ranker 排序返回 top5。


Explored 3 files
接下来会开始实际编辑。第一步我会先改入口和基础 trainer 的依赖，把当前训练路径从旧 CoSearch + LLM reranker 基类剥离出来，避免后面工具链改名时还被旧类名拖住。


Explored 1 file
我开始执行重命名和清理。先改文件名/入口名，再改内部符号；这样每一步都能用 grep 检查剩余旧术语，不会靠人工记忆追。

我会优先建立新的 coagentic_retriever_* 基类文件，而不是在旧 search_r1_reranker_* 上做逐词替换。旧文件后面会删除或从活跃链路断开，避免 LLM reranker 逻辑混进 dense ranker 路线。


Explored 1 file, 2 searches
我发现当前 CoAgenticRankerContrastiveRayTrainer 继承的是旧 CoSearchRayTrainer，这会把大量 LLM reranker 资源池、worker、checkpoint、profiling 逻辑带进来。为避免半清理，我会新建一个只含 main agent + recall service + dense ranker 的 CoAgenticRetrieverRayTrainer，然后让 ranker trainer 继承它。


Explored 1 file, 2 searches
我会尽量做成“活跃框架无旧术语”，同时对明显历史/上游 VERL 示例不做大规模清洗。对于会影响 import 的旧 LLM reranker 文件，我会先断开引用，再删除当前框架入口下的旧配置和 wrapper。


Explored 2 files, ran 2 commands
我先确认了一下普通 RayPPOTrainer 已经有单 agent 的 worker 初始化、checkpoint、validation 和 async agent loop 管理逻辑。接下来会复用这些基础能力，只保留我们需要的 rollout stats/logging 和 single-agent PPO step helper，避免继续依赖旧双 agent trainer。


Explored 1 file, 1 search
范围比较大，我会并行做两件事：主线改活跃训练/推理路径，另一路用搜索持续找残留术语。先不碰历史输出和旧 CoSearch/。

我会把 ranker strategy 的真实实现从 retriever_strategies 迁到 ranker_strategies，并把字段从 rerank_top* 改成 rank_top*。retriever_score 这种召回服务分数也会改成 recall_score，保留的是 recall_retriever 这个明确术语。


Explored 4 files
我会先选择更稳的路径：让 ranker trainer 直接继承 VERL 原生 RayPPOTrainer，只补我们需要的 rollout 统计方法。这样不需要复制一整套旧双 agent trainer，也能立刻剥离 Reranker* 角色。


Explored 3 searches
我会避免对 VERL upstream recipe/example 做大面积改名，那些不是当前 CoAgenticRetriever 核心链路。当前链路里会做到无 rerank/reranker/co_search/CoSearch，同时保留 recall_retriever 作为服务术语。

下一步是文件级重命名：co_search_* 活跃文件会改成 coagentic_retriever_*，retriever_strategies 会改成真正的 ranker_strategies 实现目录。已有 ranker_strategies 只是 wrapper，我会先替换它。


Ran 4 commands
文件级重命名已经开始了。mv 的输出里有 pyenv rehash 提示，这是 login shell 的环境噪音，不是项目文件错误；后续我会用非 login shell 来减少这类输出。


Listed files, ran 1 command
Worked for 7m 8s
我会先简化 trainer，使当前 ranker 训练不再依赖任何旧 LLM reranker worker。这个改动是后续清理的关键路径。

<中断在这里了>
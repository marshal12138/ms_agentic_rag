# CoSearch 本地复现说明

本文档记录 `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives` 目录下 CoSearch 的本地复现状态，覆盖数据准备、检索器、训练、checkpoint、评测和指标提取流程。

本次复现遵循论文和官方开源代码的核心流程，但按任务要求将论文中的基模替换为本地 `Qwen3-0.6B`。训练和评测已经在真实 GPU 上跑通，不是 CPU 或 mock 验证。

## 代码与环境

- 官方代码目录：`CoSearch/`
- 官方仓库来源：`git@github.com:snap-research/CoSearch.git`
- 当前官方代码 commit：`763bf8c`
- Search-R1 子模块目录：`CoSearch/Search-R1`
- Search-R1 commit：`598e61b`
- Conda 环境：`/data04/envs/ms/ms_cosearch_official`
- 默认 Python：`/data04/envs/ms/ms_cosearch_official/bin/python`
- 注意：当前训练、评估、retrieval 相关脚本默认都使用 `ms_cosearch_official`；旧的 `/data04/envs/ms/ms_cosearch` 不是本项目当前运行环境。
- 本地替代基模：`models/Qwen3-0.6B`
- 检索模型：`models/e5-base-v2`
- QA 数据来源：`/data01/ms_wksp/agent_up_to_date/Agentic_R_Learn/data/raw/hhjinjiajie__FlashRAG_Dataset`
- 检索语料来源：Search-R1 官方 `wiki-18` corpus，解压/必要时抽取为 `data/retrieval/wiki-18/wiki-18.jsonl`

环境中的关键包版本：

- `torch 2.8.0+cu128`
- `vllm 0.11.0`
- `transformers 4.57.6`
- `flash_attn 2.8.1`
- `trl 0.9.6`
- `verl 0.7.0.dev0`，来自官方项目内 `verl/` 的 editable install

由于服务器无法稳定访问 HuggingFace，模型和数据获取优先使用本地已有缓存或 ModelScope。`models/Qwen3-0.6B` 是从本地 ModelScope/已有模型目录链接得到；`models/e5-base-v2` 是通过 ModelScope 或本地已存在资源得到。

## 本地补丁

官方框架保持为主路径，没有另起一套训练框架。为了让官方 CoSearch/VERL 流程在当前服务器和 Qwen3-0.6B 替代基模下跑通，做了以下局部修复：

- `CoSearch/main_co_search_ppo.py`
  - 在 `ray.init` 前增加 Ray faulthandler 兼容处理。
  - 当前服务器上 Ray 初始化时可能因 faulthandler 分配失败抛出 `OSError: [Errno 12] Cannot allocate memory`，该补丁只跳过 faulthandler 安装，不改变训练逻辑。

- `CoSearch/verl/verl/experimental/agent_loop/co_search_agent_loop.py`
  - 调整 Qwen3 输出解析顺序：当输出里存在合法 `<tool_call>...</tool_call>` 时，优先执行工具调用。
  - 增加截断逻辑，只保留第一个完整 `</tool_call>` 之前的内容，避免 Qwen3 同一轮同时追加 `<answer>` 导致 agent loop 误判。

- `CoSearch/verl/verl/experimental/agent_loop/agent_loop.py`
  - 增加安全的加权平均逻辑。
  - 当 reranker 格式失败导致权重全为 0 时，避免 `ZeroDivisionError` 直接中断训练。

- `CoSearch/verl/verl/tools/co_search_tool.py`
  - reranker 使用 Qwen3 chat template 时设置 `enable_thinking=False`。
  - 目的是减少 `<think>` 风格输出对 CoSearch XML/JSON 工具格式的干扰。

这些改动都是围绕服务器兼容和 Qwen3 格式适配，训练器、reward、agent loop、retrieval tool、reranker 仍然走官方 CoSearch/VERL 路径。

## 数据准备

数据准备脚本：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash scripts/cosearch_local/00_prepare_assets.sh
```

该脚本会执行：

- `scripts/cosearch_local/download_modelscope_assets.py`
  - 准备或链接 `Qwen3-0.6B`
  - 按需准备 `e5-base-v2`

- `scripts/cosearch_local/prepare_cosearch_data.py`
  - 从本地 FlashRAG 数据集中生成 CoSearch/VERL 可读的 parquet 数据。
  - prompt 使用 Search-R1 风格的 `<reason>`、`<tool_call>`、`<answer>` XML 标签。
  - 第一轮 assistant 被明确要求必须先调用 search，不允许还没拿到工具结果就直接输出 answer。

- `scripts/cosearch_local/check_paper_mechanics.py`
  - 做一个确定性的小检查，确认 Hit@K、语义分组、reranker reward 组合逻辑可运行。

已生成的数据文件：

- `data/co_search/local_flashrag/co_search_rl_51k.train.parquet`
- `data/co_search/local_flashrag/co_search_rl_smoke.train.parquet`
- `data/co_search/local_flashrag/co_search_7bench.eval.parquet`
- `data/co_search/local_flashrag/co_search_7bench_smoke.eval.parquet`
- `data/co_search/local_flashrag/eval_by_dataset/*.parquet`
- `data/co_search/local_flashrag/manifest.json`

51,200 条训练数据混合比例按论文设置：

- NQ：20,480
- HotpotQA：14,220
- MuSiQue：9,000
- 2WikiMultiHopQA：7,500

评测集合覆盖 7 个 benchmark：

- NQ
- TriviaQA
- PopQA
- HotpotQA
- 2WikiMultiHopQA
- MuSiQue
- Bamboogle

## Dense Retriever

论文路径使用 Search-R1 dense retriever：`wiki-18.jsonl` Wikipedia passage corpus、`e5_Flat.index`、`intfloat/e5-base-v2`/本地 `models/e5-base-v2`，检索 top-50 后由 generative reranker 保留 top-5。

准备/检查命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash scripts/cosearch_local/01b_download_e5_and_build_dense_retriever.sh
```

论文一致的检索文件：

- `data/retrieval/wiki-18/wiki-18.jsonl`
- `data/retrieval/wiki-18/e5_Flat.index`

其中 `wiki-18.jsonl` 和 `e5_Flat.index` 都应使用 Search-R1 发布的官方资源；`e5_Flat.index` 由官方 index parts 合并得到：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoSearch/Search-R1
/data04/envs/ms/ms_cosearch_official/bin/python scripts/download.py \
  --save_path /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/retrieval/wiki-18
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash scripts/cosearch_local/01b_download_e5_and_build_dense_retriever.sh
```

旧的 `data/retrieval/e5_wiki18_5k/` 和 `data/retrieval/e5_wiki18_20k/` smoke-only 子集已删除，不再用于任何默认 retrieval 流程。

启动 dense retriever 服务：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
PORT=8010 RETRIEVER_GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/cosearch_local/02b_start_dense_retriever_server.sh
```

服务接口：

```text
http://127.0.0.1:8010/retrieve
```

CoSearch tool 会从该接口取 top-50 文档，然后由 generative reranker 保留 top-5。

注意：训练和评测结束后应关闭 retriever 服务，避免残留占用 GPU。

## 训练流程

训练入口：

```text
scripts/cosearch_local/05_train_paper_qwen3_dense_smoke.sh
```

该脚本包装了：

```text
scripts/cosearch_local/train_cosearch_verl_base.sh
```

底层实际调用：

```text
CoSearch/main_co_search_ppo.py
```

训练路径来自官方 `CoSearch/scripts/train_co_search_grpo.sh`，并保留以下官方机制：

- `CoSearchRayTrainer`
- `CoSearchTool`
- `CoSearchAgentLoop`
- Search-R1 XML 工具调用格式
- 多 rollout
- top-N retrieval
- generative reranker top-M
- reranker prompt 使用 Initial Query 和 Current Sub-Query
- token-level F1 语义分组
- reranker score assign
- Hit@1/3/5 与 final-answer F1 组合 reward
- final answer F1 reward
- 格式错误惩罚，当前 smoke 中为 `-0.2`
- GRPO 风格 actor update
- agent 与 reranker 的 LoRA/FSDP 训练路径

## 日志系统与实验命名

当前训练入口 `scripts/cosearch_local/train_cosearch_verl_base.sh` 已接入统一日志默认配置。上层脚本 `09_train_qwen3_4b_dense_5step_probe.sh` 和 `09b_train_qwen3_4b_b32_4retrievers_20step_timing.sh` 都会继承这套规则。

1. 新日志目录默认规则

```text
log/<yymmdd>-<hhmm>-<experiment_name>/
```

如果没有显式指定 `LOG_DIR`，日志默认写到与 `scripts/` 同级的 `log/` 目录。其中 `experiment_name` 默认是 `default`。

2. `EXPERIMENT_NAME` / `LOG_EXPERIMENT_NAME` / `LOG_DIR` 用法

通过 `EXPERIMENT_NAME` 或 `LOG_EXPERIMENT_NAME` 指定实验名：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
EXPERIMENT_NAME=qwen3_4b_b32_probe \
  bash scripts/cosearch_local/09b_train_qwen3_4b_b32_4retrievers_20step_timing.sh
```

如果需要写入固定目录，直接指定 `LOG_DIR`：

```bash
LOG_DIR=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/manual_exp_001 \
  bash scripts/cosearch_local/09b_train_qwen3_4b_b32_4retrievers_20step_timing.sh
```

3. 09/09b 生成的日志产物

- `<RUN_NAME>.env`：本次运行的关键环境变量。
- `<RUN_NAME>.train.log`：训练控制台日志。
- `<RUN_NAME>.metrics.jsonl`：VERL file logger 指标。
- `<RUN_NAME>.search_timing.jsonl`：search 调用耗时。
- `<RUN_NAME>.nvidia_smi.csv`：`nvidia-smi` 级 GPU 采样。
- `<RUN_NAME>.timing_report.step<N>.md`：分 step 的耗时和资源报告。

当前成功 smoke 训练命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
GPU_IDS=0,1,2,3 PORT=8010 RERANKER_TRAINABLE=true \
TOTAL_STEPS=1 TRAIN_MAX_SAMPLES=4 VAL_MAX_SAMPLES=2 \
TRAIN_BATCH_SIZE=2 VAL_BATCH_SIZE=2 N_ROLLOUTS=2 ACTOR_BATCH_SIZE=2 \
MAX_USER_TURNS=1 MAX_ASSISTANT_TURNS=2 TOP_N=50 TOP_M=5 \
MAX_PROMPT_LENGTH=11264 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LEN=12288 \
MAX_TOOL_RESPONSE_LENGTH=2048 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.20 \
AGENT_WORKERS=1 RAY_NUM_CPUS=16 RAY_OBJECT_STORE_MEMORY=2147483648 \
TEMPERATURE=0.7 \
  bash scripts/cosearch_local/05_train_paper_qwen3_dense_smoke.sh \
  > log/paper_qwen3_dense_train_20260602_2242_real_longctx4gpu_patch2.log 2>&1
```

成功输出：

- checkpoint：`checkpoints/paper_qwen3_dense_smoke/global_step_1`
- main rollout：`checkpoints/paper_qwen3_dense_smoke/rollout_data/main/1.jsonl`
- reranker rollout：`checkpoints/paper_qwen3_dense_smoke/rollout_data/reranker/1.jsonl`
- 训练日志：`log/paper_qwen3_dense_train_20260602_2242_real_longctx4gpu_patch2.log`

训练成功的关键现象：

- Ray 正常初始化。
- main vLLM 与 reranker vLLM 正常启动。
- dense retriever 收到 POST 请求。
- main agent 完成 generation、reward、update actor。
- reranker 完成 generation、reward、update actor。
- checkpoint 正常保存到 `global_step_1`。

## 评测流程

评测入口：

```text
scripts/cosearch_local/06_eval_paper_qwen3_dense_smoke.sh
```

该脚本同样走官方 CoSearch/VERL trainer，只是设置：

- `trainer.val_before_train=True`
- `trainer.val_only=True`
- 默认从 `checkpoints/paper_qwen3_dense_smoke/global_step_1` 加载 checkpoint

当前成功 smoke 评测命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
GPU_IDS=0,1,2,3 PORT=8010 RERANKER_TRAINABLE=true \
OUT_DIR=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/paper_qwen3_dense_eval_step1_smoke \
EXP_NAME=paper_qwen3_0_6b_dense_eval_step1_smoke \
TRAIN_MAX_SAMPLES=4 VAL_MAX_SAMPLES=8 \
TRAIN_BATCH_SIZE=2 VAL_BATCH_SIZE=2 N_ROLLOUTS=2 ACTOR_BATCH_SIZE=2 \
MAX_USER_TURNS=1 MAX_ASSISTANT_TURNS=2 TOP_N=50 TOP_M=5 \
MAX_PROMPT_LENGTH=11264 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LEN=12288 \
MAX_TOOL_RESPONSE_LENGTH=2048 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.20 \
AGENT_WORKERS=1 RAY_NUM_CPUS=16 RAY_OBJECT_STORE_MEMORY=2147483648 \
TEMPERATURE=0.7 \
  bash scripts/cosearch_local/06_eval_paper_qwen3_dense_smoke.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/paper_qwen3_dense_smoke/global_step_1 \
  > log/paper_qwen3_dense_eval_20260602_2252_step1_smoke8.log 2>&1
```

成功输出：

- validation dump：`checkpoints/paper_qwen3_dense_eval_step1_smoke/validation_data/1.jsonl`
- 评测日志：`log/paper_qwen3_dense_eval_20260602_2252_step1_smoke8.log`
- 本次 smoke 评测样本数：8

评测成功的关键现象：

- 日志中出现 `Load from checkpoint folder: .../global_step_1`。
- 日志中出现 `Setting global step to 1`。
- actor 与 reranker checkpoint shard 均被加载。
- validation rollout 正常生成并写入 `validation_data/1.jsonl`。
- dense retriever 收到检索请求。

## 指标提取

指标提取脚本：

```text
scripts/cosearch_local/08_extract_metrics_from_log.py
```

训练指标提取命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
/data04/envs/ms/ms_cosearch_official/bin/python \
  scripts/cosearch_local/08_extract_metrics_from_log.py \
  log/paper_qwen3_dense_train_20260602_2242_real_longctx4gpu_patch2.log \
  --pretty
```

本次训练 step 1 的关键指标：

- `training/global_step`: 1
- `main/score_mean`: -0.20000000298023224
- `main/f1_mean`: 0.0
- `main/valid_rate`: 0.0
- `reranker/score_mean`: -0.20000000298023224
- `main_perf/max_memory_allocated_gb`: 24.283271312713623
- `reranker_perf/max_memory_allocated_gb`: 40.26497220993042
- `perf/time_per_step`: 53.0356880328618

评测指标提取命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
/data04/envs/ms/ms_cosearch_official/bin/python \
  scripts/cosearch_local/08_extract_metrics_from_log.py \
  log/paper_qwen3_dense_eval_20260602_2252_step1_smoke8.log \
  --pretty
```

本次评测 smoke 的关键指标：

- `val-core/musique/reward/mean@1`: -0.20000000298023224
- `val-aux/musique/f1/mean@1`: 0.0
- `val-aux/musique/valid/mean@1`: 0.0
- `val-core/nq/reward/mean@1`: -0.20000000298023224
- `val-aux/nq/f1/mean@1`: 0.0
- `val-aux/nq/valid/mean@1`: 0.0
- `val-core/triviaqa/reward/mean@1`: -0.20000000298023224
- `val-aux/triviaqa/f1/mean@1`: 0.0
- `val-aux/triviaqa/valid/mean@1`: 0.0
- `val-core/hotpotqa/reward/mean@1`: -0.20000000298023224
- `val-aux/hotpotqa/f1/mean@1`: 0.0
- `val-aux/hotpotqa/valid/mean@1`: 0.0
- `val-aux/num_turns/mean`: 4.0

## 一键 smoke 脚本

一键流程脚本：

```text
scripts/cosearch_local/07_run_paper_qwen3_dense_smoke.sh
```

它会依次执行：

1. 准备模型和数据。
2. 检查 Search-R1 `wiki-18` corpus/index 是否存在；缺 index 时提示下载官方 parts。
3. 启动 Search-R1 dense retriever 服务。
4. 跑训练 smoke。
5. 跑 validation-only 评测 smoke。
6. 退出时通过 trap 关闭 retriever 服务。

命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
GPU_IDS=0,1,2,3 PORT=8010 RETRIEVER_GPU_ID=7 \
  bash scripts/cosearch_local/07_run_paper_qwen3_dense_smoke.sh
```

注意：该脚本会真实启动 GPU 服务和训练进程，适合服务器执行，不适合普通静态检查。

## 当前结果的含义

当前结果是“官方流程已跑通”的 smoke 复现，不是 paper-scale 指标复现。

低分原因：

- 按要求使用了 `Qwen3-0.6B` 替代论文基模，模型能力更小。
- 当前只跑了 1 个 PPO step。
- 每次只用 4 条训练样本和 8 条评测样本做 smoke。
- Qwen3-0.6B 在 Search-R1 XML/JSON 格式上仍不稳定。
- 部分 rollout 中 reranker 格式失败，触发 fallback。
- 格式错误会拿到 `-0.2` 惩罚，因此 step-1 smoke 指标为负。

这并不表示流程失败。日志已经证明训练、检索、reranker、reward、checkpoint、checkpoint 加载评测全部经过官方代码路径执行完成。

## 后续扩展建议

如果要从 smoke 走向更有意义的分数，可以沿用当前脚本，只扩大参数：

- 增大 `TOTAL_STEPS`
- 增大 `TRAIN_MAX_SAMPLES`
- 增大 `VAL_MAX_SAMPLES`
- 保持 `TOP_N=50`
- 保持 `TOP_M=5`
- 根据显存调整 `GPU_IDS`、`MAX_PROMPT_LENGTH`、`MAX_RESPONSE_LENGTH`、`GPU_MEMORY_UTILIZATION`
- 保持 dense retriever 服务运行在 `PORT=8010`

建议先按如下顺序扩展：

1. `TOTAL_STEPS=10`，`TRAIN_MAX_SAMPLES=64`，确认稳定性。
2. `TOTAL_STEPS=50`，`TRAIN_MAX_SAMPLES=512`，观察格式有效率。
3. 扩大到完整 `co_search_rl_51k.train.parquet`。
4. 将 `VAL_DATA` 切到完整 `co_search_7bench.eval.parquet` 或按数据集逐个评测。

如果 Qwen3-0.6B 仍然持续格式失败，可以先做一个小规模格式 warmup/SFT，再进入 CoSearch PPO。目录下已有 `checkpoints/format_warmup`，但当前主复现结果仍以官方 VERL PPO smoke 为准。

## 常用检查命令

检查脚本语法：

```bash
bash -n scripts/cosearch_local/05_train_paper_qwen3_dense_smoke.sh
bash -n scripts/cosearch_local/06_eval_paper_qwen3_dense_smoke.sh
/data04/envs/ms/ms_cosearch_official/bin/python -m py_compile \
  scripts/cosearch_local/08_extract_metrics_from_log.py
```

检查 checkpoint：

```bash
ls -la checkpoints/paper_qwen3_dense_smoke/global_step_1
```

检查 validation 输出：

```bash
wc -l checkpoints/paper_qwen3_dense_eval_step1_smoke/validation_data/1.jsonl
```

检查 dense retriever 是否残留：

```bash
pgrep -af 'retrieval_server.py|dense_retrieval_server|main_co_search_ppo|ray::|VLLM::|vLLMHttpServer'
```

检查 GPU：

```bash
nvidia-smi
```

## API 配置

用户提供的 DashScope 兼容 API 配置在：

```text
/data01/ms_wksp/agent_up_to_date/cars_info_agent/llama_index_rag_build/conf/common.json
```

当前 CoSearch 本地复现未使用外部模型 API。生成、训练和评测均使用本地 `Qwen3-0.6B`。

## CoAgenticRetriever Async Labeling 讨论版计划

本节记录 2026-06-15 对 CoAgenticRetriever 新样本构造框架的讨论结论，供下一步设计和实现前继续确认。这里不是已落地实现说明。

### 背景

当前 CoAgenticRetriever 通过 ranker contrastive step 优化 dense ranker。现有默认链路是：

```text
fresh trajectories
  -> trajectory_selector
  -> signal_builder
  -> sample_builder
  -> replay_buffer
  -> collator
  -> ranker_wg.update_ranker_contrastive
```

其中默认信号构造仍较原始，例如按当前 ranker 排名 top-k 构造 pseudo positive。下一步希望加入异步样本信号生成能力：在额外 GPU06/GPU07 上启动 LLM-as-judge 服务，例如 DeepSeek-Flash，用它对 `origin_query + sub_query + ranked_chunk_list` 进行打分，然后将打分结果用于构造 ranker contrastive samples。

### 总体原则

异步 ranker training不应阻塞 GRPO / agent LLM 主训练链路。

推荐将系统拆成三条链路：

```text
GRPO / rollout 主链路
  -> 产生 origin_query + sub_query + ranked_chunk_list
  -> async_ranker_training_labeler.submit(...)
  -> 继续 agent GRPO/PPO update
  -> 不等待 judge，不等待 ranker sample

async_ranker_training 链路
  -> request queue
  -> LLM-as-judge stage
  -> 后续可插拔 extra scoring stage
  -> score merger
  -> completed CandidateSignalData buffer

ranker contrastive 链路
  -> 从 completed buffer pop 最新 N 组 CandidateSignalData
  -> sample_builder 构造 ContrastiveSample
  -> ranker contrastive update
```

关键约束：

- `async_ranker_training_labeler.submit()` 必须非阻塞。
- `sample_builder` 可以等待 buffer 中有足够数据。
- `sample_builder` 的等待只能阻塞 ranker contrastive 链路，不能阻塞 GRPO 主链路。
- ranker contrastive step 应从主训练循环中拆出，变成 Ray actor、后台进程或等价的异步 trainer。
- 如果只是 `ranker_step.remote()` 后立即 `ray.get()`，仍然会阻塞主链路，不符合目标。

### async_ranker_training 作为可配置策略

`async_ranker_training` 应作为 ranker 训练下的一种可配置策略。基础版本先使用 LLM-as-judge 直接给分；后续还会在它之后紧接一次新的评分，因此设计上应使用 pipeline/stage，而不是写死单一 judge。

建议配置放在：

```text
CoAgenticRetriever/config/async_ranker_training.yaml
```

本地 DeepSeek-Flash / GPU06-GPU07 相关覆盖配置可放在：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

建议配置形态：

```yaml
ranker_training:
  async_ranker_training:
    enable: true

    # 每个 global_step 最多提交多少个 sub_query/tool call 到 async_ranker_training_labeler。
    max_sub_query: 10

    # 请求允许滞后的最大 global step 数。
    max_glb_step_lag: 3

    request_queue_size: 2048
    completed_buffer_size: 4096
    drop_policy: drop_oldest

    num_workers: 4
    request_timeout_seconds: 60
    max_retries: 2

    # completed judge signal 一旦进入 buffer，就由后台 ranker async trainer
    # 立即交给 signal_builder/sample_builder；sample_builder 本身不等待 signal 池。

    stages:
      - type: llm_as_judge
        endpoint: http://127.0.0.1:8067/v1/chat/completions
        model: deepseek-flash
        prompt_version: llm_judge_relevance_v1
        score_schema: relevance_0_3
        max_docs_per_request: 20
        temperature: 0.0

      - type: extra_scorer
        enable: false
        weight: 0.3
```

`max_sub_query` 的含义：

- 每个 `global_step` 进入 async ranker training labeler 的 sub-query / tool-call 数不能超过该值。
- 限制粒度是 tool call，不是 trajectory。
- 即使 `trajectory_selector` 只选中 3 条轨迹，如果每条轨迹有 8 次 search，也最多只提交 10 个 sub-query，避免一次 step 产生 24 个 judge 请求。

`max_glb_step_lag` 的含义：

- 处理请求时检查 `current_global_step - request.created_global_step`。
- 如果超过该值，说明请求已经过期，直接跳过，不再消耗 LLM judge 资源。
- 该检查应发生两次：worker 取出 request 时检查一次，judge 返回准备写入 completed buffer 前再检查一次。

### async_ranker_training_labeler 输入输出

`async_ranker_training_labeler` 的输入是从 rollout `tool_call_details` 中提取出的候选排序上下文，而不是 `ContrastiveSample`。

建议输入结构：

```text
AsyncLabelRequest:
  request_id
  created_global_step
  trajectory_id
  tool_call_id
  turn_idx

  origin_query
  sub_query
  golden_answers

  ranked_chunk_list:
    [
      {
        doc_id
        title
        text
        recall_rank
        recall_score
        rank_rank
        rank_score
        metadata
      }
    ]

  trajectory_score
  score_type
  trace_metadata
  label_policy
  prompt_version
```

`ranked_chunk_list` 建议优先来自当前 ranker 的 `rank_top50_docs`，没有时 fallback 到 `recall_top50_docs`。

`async_ranker_training_labeler` 的输出是 `CandidateSignalData`。这是 completed candidate signal buffer 里的单位元素。

建议输出结构：

```text
CandidateSignalData:
  signal_id
  request_id
  created_global_step
  completed_global_step
  completed_at

  trajectory_id
  tool_call_id
  turn_idx
  origin_query
  sub_query
  golden_answers

  ranked_chunk_list

  scores:
    [
      {
        doc_id
        judge_score
        extra_score
        final_score
        confidence
        reason_code
      }
    ]

  positives:
    [doc_id, ...]

  negatives:
    [doc_id, ...]

  label_source
  score_version
  prompt_version
  judge_model

  status
  error
  latency_ms
  raw_response_ref
```

`sample_builder` 后续只消费 `CandidateSignalData`，不关心 LLM judge 服务如何部署、如何 prompt、如何重试。

### CandidateSignalData Buffer 语义

训练用 completed buffer 建议采用 destructive queue 语义：

```text
CandidateSignalBuffer.pop_latest(n=N, wait=True)
  -> 返回最新 N 个 CandidateSignalData
  -> 返回的数据从 active buffer 中移除
  -> sample_builder 使用这些数据构造 ContrastiveSample
```

因此，当 buffer 内长度低于 N 时，`sample_builder` 会等待；该等待会导致 ranker contrastive step 等待，但不应导致 GRPO step 等待。

为兼顾可追溯性，建议区分：

```text
completed_signal_queue    # 训练消费队列，pop 后消失
signal_audit_store        # JSONL/SQLite 落盘记录，永久保留，用于复盘和 debug
```

也就是说，训练语义是“消费即删除”，实验审计语义是“所有请求和结果都可追溯”。

### 样本构造建议

第一版不建议让 LLM judge 对 top50 做完整排序。更稳妥的方式是 pointwise relevance scoring：

```text
0: irrelevant
1: related but not useful
2: useful supporting evidence
3: directly sufficient / contains answer evidence
```

候选 docs 不必全量 top50，可从每个 tool call 中挑 12-20 个文档：

- 当前 ranker top5。
- recall top5。
- ranker 排名前但 judge 可能判低的 hard negatives。
- ranker 排名靠后但可能含答案的探索样本。

构造 contrastive 样本时：

- `final_score >= positive_threshold` 作为 positive。
- `final_score <= negative_threshold` 作为 negative。
- 当前 ranker 排名前但 final_score 低的文档优先作为 hard negative。
- 当前 ranker 排名靠后但 final_score 高的文档优先作为 hard positive。

这样可以复用当前 `ContrastiveSample = 1 positive + N negatives` 和 InfoNCE loss，不需要第一版就引入 pairwise/listwise loss。

### ranker contrastive 异步化

当前 `process_ranker_contrastive_step()` 在训练主循环中同步执行。新框架下建议拆成后台 ranker trainer：

```text
main trainer loop:
  main_batch = rollout()
  async_ranker_training_labeler.submit(main_batch.tool_call_details, global_step=...)
  launch/continue main_agent_grpo_update()
  do not wait for ranker samples

RankerAsyncTrainer:
  while training:
    signals = completed_signal_queue.pop_latest(n=N, wait=True)
    samples = sample_builder.build(signals)
    ranker.update(samples)
```

实现形态可以是：

- Ray actor。
- 独立后台进程。
- 本地后台 thread，仅用于第一版验证。

推荐优先 Ray actor，因为当前训练框架已依赖 Ray，后续 metrics、生命周期和 checkpoint 管理更容易接入。

### 滞后风险和负面影响

如果 async ranker training 变慢，可能出现 agent LLM 连续更新多次，而 ranker contrastive step 长时间没有更新。这是异步设计允许出现的情况，但有以下负面影响：

- ranker 滞后于 agent query 分布。agent 已经学出新的 search 行为，但 ranker 仍在旧分布上训练。
- tool 质量改善延迟。ranker 不更新时，agent 继续看到旧 ranker 给出的 top5。
- 候选信号过期。judge 很慢才完成的 label 可能来自几步前的 ranker 排序，当时的 `ranked_chunk_list` 已不代表当前 ranker 行为。
- 训练节奏失衡。agent 更新很多步、ranker 更新很少步，系统会退化成“主要训练 agent，ranker 基本旁路”。

对应缓解策略：

- 使用 `max_glb_step_lag=3` 丢弃过期请求和过期结果。
- 使用 `max_sub_query=10` 控制每个 global step 的请求量。
- 只消费最新 signals，过旧 signals 从 active queue 丢弃。
- 记录 `ranker/agent_step_lag`、`ranker/update_per_agent_step`、`async_ranker_training/labeler_expired_count` 等指标。
- 当 judge 队列积压时，按 high-value policy 选择 tool call，而不是无差别提交所有 sub-query。

### 建议新增指标

```text
async_ranker_training/labeler_submitted_count
async_ranker_training/selected_tool_calls
async_ranker_training/labeler_expired_count
async_ranker_training/labeler_request_queue_size
async_ranker_training/labeler_completed_buffer_size
async_ranker_training/labeler_completed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_avg_lag_steps
async_ranker_training/labeler_max_lag_steps
async_ranker_training/labeler_avg_latency_ms
ranker/async_consumed_signals
ranker/async_wait_seconds
ranker/update_per_agent_step
```

### 第一版落地边界

第一版建议避免过度工程化：

- 不引入 Redis/Kafka，先使用内存 queue + JSONL/SQLite audit store。
- 不同步等待 LLM judge。
- 不 judge 全量 top50。
- 不废弃 pseudo-rank fallback。
- 不改变 ranker InfoNCE loss。
- 不让 GRPO 主链路等待 ranker signal。

第一版目标是验证：

1. GPU06/GPU07 的 DeepSeek-Flash judge 服务能稳定处理请求。
2. async_ranker_training_labeler 能将 rollout tool calls 转成 `CandidateSignalData`。
3. ranker async trainer 能从 completed queue 中 pop 最新 N 组 signal 并更新 ranker。
4. GRPO step 和 ranker contrastive step 可以解耦运行。

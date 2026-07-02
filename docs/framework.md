# 项目能力

`CoSearch_derevitives` 是围绕 Search-R1 / CoSearch 做本地复现、训练改造、检索服务、评估和报告沉淀的项目目录。当前主要承载三条实验主线：

- `CoSearch`：原始 CoSearch 复现和本地训练评估链路。
- `CoAgenticRetriever`：在 agent LLM 训练链路之外增加可训练 rank retriever 的协同检索训练框架。
- `AgenticIterRag`：迭代式 agentic RAG 训练和推理链路。

项目级公共能力包括：

- 统一 dense retriever 启动、GPU/CPU 检索服务和 round-robin proxy。
- 统一训练日志、评估日志、GPU 采样、markdown report 和指标图生成。
- 统一 VERL/FSDP checkpoint 转 HuggingFace checkpoint 的后处理入口。
- 统一 GPU 等待、串行任务编排和 GPU 残留进程释放能力。
- 面向 CoSearch、CoAgenticRetriever、AgenticIterRag 的训练、推理、评估脚本管理。
- 沉淀复现过程、方案设计、任务经验和环境坑点。

# 项目结构介绍

## CoSearch

`CoSearch/` 保留原始 CoSearch 复现和改造代码，包括 Search-R1、VERL、配置、脚本和论文资料。该目录用于追溯原始实现和承载 CoSearch 主线的训练逻辑。

## CoAgenticRetriever

`CoAgenticRetriever/` 是当前重点改造主线。它保留 agent LLM PPO/GRPO 路径，同时增加可训练 E5 rank retriever。核心结构和训练数据流见 `docs/framework_CoAgenticRetriever.md`。

主要职责：

- 维护 CoAgenticRetriever 的 Hydra/YAML 配置。
- 维护 rank retriever 训练策略、样本构造、replay buffer 和 collator。
- 维护 rank retriever worker 与 VERL trainer 集成代码。
- 支持 recall retriever top-50、rank retriever rerank top-5、agent LLM 再回答的协同流程。

## AgenticIterRag

`AgenticIterRag/` 用于 AgenticIterRag 主线的训练、推理和评估。该目录结构与 CoSearch / CoAgenticRetriever 类似，保留 Search-R1、VERL、配置和脚本入口。当前框架说明见 `docs/framework_AgenticIterRag.md`。

## src

`src/` 存放跨主线复用的项目级公共代码。一次性实验入口不放这里；只有被 `scripts/cosearch_local`、`scripts/coagenticRetriever_local`、`scripts/iterRag_scripts` 共同复用的能力才应放入 `src/`。

当前核心模块：

- `src/retrievers/`：dense retriever 服务、GPU retriever、round-robin proxy 和检索资源校验。
- `src/logs/report_system/`：训练/评估日志路径、GPU 采样、训练报告、评估报告和指标图生成。
- `src/checkpoints/`：VERL/FSDP checkpoint 转 HuggingFace checkpoint 的公共转换逻辑。
- `src/runtime/`：运行时辅助能力，包括 GPU 等待和 train/eval 串行任务编排。

详细说明见 `docs/src.md`。

`src/runtime/` 当前主要提供两个公共入口：

- `wait_for_gpus.sh`：等待指定 GPU 空闲。单个训练/评估 task 可以 source 后调用 `wait_for_gpus_if_enabled`。
- `task_sequence.sh`：串行编排多个 task，支持每个子任务前等待 GPU、记录 `log/task_sequences/<stamp>-<TASK_SEQUENCE_NAME>/summary.tsv`、失败续跑、dry-run 和显式 GPU 释放。

使用原则：

- 单个 task 只需要启动前等 GPU 时，用 `wait_for_gpus.sh`。
- 多个 train/eval task 需要串行执行时，在 `tasks/experiments/` 中 source `task_sequence.sh`。
- 如果等待逻辑由编排层负责，子任务命令里应传 `WAIT_FOR_GPU_RELEASE=0`，避免重复等待。
- GPU 释放是显式兜底动作；只有 `TASK_SEQUENCE_RELEASE_GPUS=1` 时，`task_sequence_release_gpus` 才会真正发送信号。

## scripts

`scripts/` 存放本地可执行脚本入口，按实验主线分组：

- `scripts/cosearch_local/`：CoSearch 本地复现、检索服务、训练、推理和评估脚本。
- `scripts/coagenticRetriever_local/`：CoAgenticRetriever 数据处理、训练、推理和评估脚本。
- `scripts/iterRag_scripts/`：AgenticIterRag 训练、推理和检索服务脚本。

约定：

- 稳定、可复用的运行入口放在对应主线的 `scripts/` 子目录。
- 多条主线共享的逻辑不要互相跨目录引用，应上移到 `src/`。
- 训练脚本应优先使用 `src/logs/report_system/` 的统一日志和报告函数。
- 检索服务应优先使用 `src/retrievers/` 的统一入口。
- 主线内部的公共执行逻辑应沉淀在 `scripts/<主线>_local/`，例如服务启动、preflight、Hydra 参数拼接、日志目录、checkpoint 后处理和 report 生成。
- `scripts/` 不应成为一次性实验配方的堆积目录；如果只是某次实验的变量组合，应放入 `tasks/`。

## tasks

`tasks/` 用于存放训练、评估、数据处理等任务脚本，避免 `scripts/` 无限扩张。

当前分组：

- `tasks/train_tasks/`：训练任务封装脚本。
- `tasks/eval_tasks/`：评估任务封装脚本。
- `tasks/experiments/`：跨训练、评估和释放动作的串行实验编排脚本。
- `tasks/data_proc_tasks/`：数据处理任务脚本。
- `tasks/human_task_record.md`：人工任务记录。

其中 `tasks/train_tasks/` 的定位是训练实验配方层，而不是训练框架实现层。每个 train task 脚本应尽量回答“这次实验和其它实验相比改了什么”，例如：

- 实验名和 run identity：`EXP_NAME`、`RUN_STAMP`。
- 实验假设和 ablation 差异：是否启用 async ranker training、使用哪个 judge、每次 ranker update 消费多少条 signal。
- 资源默认值：agent/ranker/recall/judge 使用的 GPU、batch、并发、显存比例。
- 策略配置入口：`ASYNC_RANKER_TRAINING_YAML`、`RANKER_STRATEGY_YAML`、`HYDRA_OVERRIDE_YAMLS`。
- 少量实验特有覆盖：追加到 `COAGENTIC_EXTRA_ARGS` 的 Hydra dotlist 参数。

`tasks/train_tasks/` 调用 `scripts/coagenticRetriever_local/` 下的训练入口是刻意的分层：

- `tasks/train_tasks/` 负责声明“做什么实验”。
- `scripts/coagenticRetriever_local/` 负责把实验安全、完整、可复现地跑起来。
- `scripts/coagenticRetriever_local/strategies_yaml/` 负责保存可组合的策略配置，例如 async ranker training、ranker sampling、judge prompt 和 sample builder 设置。

以 CoAgenticRetriever 当前训练链路为例，`tasks/train_tasks/train_CAR_async_ranker_training_ds_flash_mix_signal.sh` 只声明该实验启用 DeepSeek-V4-Flash async ranker training，并把 `ranker_training.async_ranker_training.sample_builder_request_batch` 覆盖为 3；真正的 recall service、LLM judge preflight、Hydra 覆盖顺序、训练日志、checkpoint 目录和 report 生成，仍由 `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh` 及其 assets 统一处理。

维护原则：

- 新增一次性训练实验时，优先在 `tasks/train_tasks/` 增加薄脚本，并同步更新 `tasks/train_tasks/train_tasks_records.md`。
- 如果多个 train task 复制了同一段复杂启动逻辑，应下沉到 `scripts/<主线>_local/` 或 `src/`。
- 如果某个 task 脚本稳定为长期正式入口，可以保留在 `tasks/` 作为实验入口，同时把通用能力继续沉淀到 `scripts/`。
- `tasks/train_tasks/` 中允许有少量实验特有资源调度逻辑，例如等待指定 GPU 空闲；但服务管理、日志系统、checkpoint 管理和 report 生成不应在每个 task 中重复实现。

`tasks/experiments/` 的定位是“实验编排层”，用于把多个 task 串成一个可重复执行的流程。例如：

- 先运行一个 train task。
- 训练结束后执行 GPU 释放兜底。
- 再运行一个 eval task。

编排脚本应使用 `src/runtime/task_sequence.sh`，而不是自己重复实现等待 GPU、写日志、失败续跑和释放 GPU。任务名只作为编排日志中的标记，不应作为业务硬约束。

当前示例：

- `tasks/experiments/tasks_TrainEval_00_example.sh`：演示一次 CoAgenticRetriever async-ranker-training train + eval 串行编排。

示例 dry-run：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
TASK_SEQUENCE_DRY_RUN=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

真实运行并允许编排层释放 GPU：

```bash
TASK_SEQUENCE_RELEASE_GPUS=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

## pipelines

`pipelines/` 存放阶段性流水线和专题实验流程。

- `pipelines/formal/`：可以作为正式流程保留的 pipeline。
- `pipelines/temp/`：临时实验、压测、数据准备、模型 judge、GPU retriever 探测等流程。

临时 pipeline 如果稳定下来，应迁移为正式脚本或沉淀到 `tasks/` / `scripts/`，并同步补充对应文档。

## data

`data/` 存放项目本地数据和中间数据。

当前主要分组：

- `data/retrieval/`：检索语料、索引或检索相关数据。
- `data/co_search/`：CoSearch 主线数据。
- `data/coAgenticRetriever/`：CoAgenticRetriever 主线数据。
- `data/llm_judge/`：LLM judge 相关数据。

## log 和 reports

`log/` 存放运行日志和 trace，`reports/` 存放整理后的 markdown 报告和图表。

主要约定：

- `log/train_logs/`：训练日志、metrics JSONL、GPU 采样和训练过程报告。
- `log/eval_res/`：评估 trace、runtime log 和中间结果。
- `reports/train/`：训练 summary 或聚合报告。
- `reports/eval/`：评估 markdown 报告。

日志和报告系统的完整设计见 `docs/log_and_report/report_system_1.0.md`。

## checkpoints 和 outputs

`checkpoints/` 用于保存训练得到的模型 checkpoint 或转换后的 checkpoint。

`outputs/` 用于保存训练、推理、探测过程中产生的本地输出。长期需要追溯的正式结果应整理到 `reports/` 或对应文档中，不应只依赖 `outputs/`。

## docs

`docs/` 存放项目说明、框架说明、训练评估说明、方案设计、历史工作总结和 FAQ。

常用入口：

- `docs/readme.md`：`docs/` 文档索引和更新规则。
- `docs/framework.md`：当前文件，说明项目能力、项目结构和标准使用过程。
- `docs/framework_CoAgenticRetriever.md`：CoAgenticRetriever 核心框架说明。
- `docs/src.md`：项目公共代码说明。
- `docs/Experiments/`：训练和评估操作说明。
- `docs/train_and_eval/`：正式训练/评估说明。
- `docs/planning/`：未来方案和未稳定设计。
- `docs/pre_works/`：历史复现、阶段性总结和交接材料。
- `docs/FQA/`：常见问题和机制解释。

## experiences

`experiences/` 用于沉淀可复用经验，不记录一次性流水账。

- `experiences/env/`：环境、依赖、权限、GPU、服务启动等经验。
- `experiences/tasks/`：任务执行、数据构建、评测流程、排查顺序等经验。

经验文件应优先记录可复用的事实、排查方法、执行顺序、坑点和验证方式。

# 标准使用过程

1. 准备模型、检索语料、FAISS index 和数据集，必要时参考 `scripts/cosearch_local/00_prepare_assets.sh` 及 `data/` 下的主线数据。
2. 启动或确认 dense retriever 服务，优先使用 `src/retrievers/start_dense_retriever_server.sh` 或对应主线脚本中的包装入口。
3. 按主线选择训练入口：
   - CoSearch：使用 `scripts/cosearch_local/` 或 `tasks/train_tasks/` 中的 CoSearch 训练脚本。
   - CoAgenticRetriever：使用 `scripts/coagenticRetriever_local/` 或 `tasks/train_tasks/` 中的 CAR 训练脚本。
   - AgenticIterRag：使用 `scripts/iterRag_scripts/` 中的训练脚本。
4. 训练过程中使用 `src/logs/report_system/` 统一生成训练日志、metrics、GPU 采样、latest/final 训练报告和图表。
5. 训练结束后按需要转换 checkpoint，公共转换逻辑放在 `src/checkpoints/`。
6. 使用对应主线的推理或评估脚本生成 trace、runtime log 和评估报告。
7. 如果需要连续执行 train、释放 GPU、eval 等多个任务，优先在 `tasks/experiments/` 中使用 `src/runtime/task_sequence.sh` 编排。
8. 将稳定结论沉淀到 `reports/`、`docs/train_and_eval/`、`docs/pre_works/`、`docs/planning/` 或 `experiences/` 中。

`readings/` 类型目录如果后续出现，仅作为外部调研资料存放，不作为项目运行入口。

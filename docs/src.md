# `src/` 目录说明

`src/` 存放可被多个脚本集合复用的项目级代码。一次性实验入口仍放在 `scripts/`；跨 `cosearch_local`、`coagenticRetriever_local`、`iterRag_scripts` 复用的日志、报告、checkpoint 等基础设施应放在 `src/`。

## `src/checkpoints/`

Checkpoint 处理模块。

当前文件：

- `convert_verl_fsdp_checkpoint.py`：将 VERL/FSDP sharded checkpoint 转换为 HuggingFace safetensors shard，同时保留原始 VERL/FSDP checkpoint，便于继续训练。
- `checkpoint_conversion.sh`：shell 包装函数，供训练脚本在训练结束后调用 Python 转换脚本。
- `__init__.py`：Python package 标记文件。

使用约定：

- 训练脚本需要 checkpoint 后处理时，source `src/checkpoints/checkpoint_conversion.sh`。
- checkpoint 格式转换、checkpoint 清理、checkpoint 寻址等跨脚本复用逻辑优先放在此目录。

## `src/logs/`

日志和报告系统模块。该模块是训练/推理脚本的统一日志与报告基础设施；
`scripts/cosearch_local` 和 `scripts/coagenticRetriever_local` 不应各自维护一套
日志目录、GPU 采样、训练报告生成逻辑。

当前子目录：

- `log_system/`：预留给后续更完整的日志抽象，目前为空。
- `report_system/`：当前承载日志路径和报告路径初始化工具。

### `src/logs/report_system/`

当前文件：

- `logging_reports.sh`：训练日志、评估日志、GPU 采样和 train report 生成的统一 shell 工具。
- `train_timing_report.py`：schema 驱动的训练 timing markdown 报告生成器。
- `train_metrics_report.py`：schema 驱动的训练 metrics markdown 报告生成器；通过 `--detailed` 切换 per-step detailed report。
- `train_metrics_plots.py`：schema 驱动的训练指标图生成器。
- `train_max_metric_step.py`：读取训练 metrics JSONL 的最大 step，用于周期性 snapshot。
- `report_io.py`：metrics JSONL、env、train log、nvidia-smi CSV、rollout dump 的公共读取和统计工具。
- `report_schema.py`：项目 train report schema 加载器。

兼容文件：

- `generate_timing_report.py`、`generate_metrics_report.py`、`plot_training_metrics.py`、`max_metric_step.py`：保留为 `train_*` 入口的兼容 wrapper。新代码应优先使用 `train_*` 命名。

主要函数：

- `setup_cosearch_training_log_defaults`：设置训练日志目录和 `TRAIN_LOG`、`METRICS_JSONL`、`SEARCH_TIMING_JSONL`、`NVIDIA_SMI_CSV`、`REPORT_PREFIX`。
- `setup_cosearch_training_report_defaults`：设置训练 timing/metrics/detailed report 和 plot prefix 的 latest 默认路径。
- `train_report_system_generate_reports`：公共 train report 生成入口。`snapshot` 模式按当前最大 step 生成 latest snapshot；`final` 模式生成 all/final 并覆盖 latest。
- `setup_cosearch_summary_report_defaults`：设置训练 sweep/summary report 默认路径。
- `setup_cosearch_eval_artifact_defaults`：设置评估 trace、runtime log、markdown report 默认路径。
- `cosearch_generate_training_reports`：兼容命名，内部调用 `train_report_system_generate_reports snapshot`。
- `cosearch_generate_final_training_reports`：训练结束后生成 all/final 报告，覆盖训练过程中的 latest snapshot。
- `cosearch_start_training_reporter`：启动后台 train reporter，每隔 `REPORT_INTERVAL_SECONDS` 秒调用一次 snapshot 生成。
- `cosearch_start_nvidia_smi_sampler`：启动后台 `nvidia-smi` CSV 采样，采样间隔由 `NVIDIA_SMI_INTERVAL` 控制，输出到 `NVIDIA_SMI_CSV`。
- `cosearch_stop_background_pid`：停止由训练脚本记录的后台 reporter / GPU sampler 等辅助进程。
- `setup_cosearch_logging_defaults`：兼容旧函数名，内部转发到训练日志默认初始化。

默认路径约定：

- 训练日志：`log/train_logs/`
- 评估 trace 和 runtime log：`log/eval_res/`
- 评估 markdown report：`reports/eval/`
- 训练 sweep/summary report：`reports/train/`

训练日志命名规则：

```text
log/train_logs/<GROUP_SLUG>/<RUN_NAME>/
  <RUN_NAME>.env
  <RUN_NAME>.train.log
  <RUN_NAME>.metrics.jsonl
  <RUN_NAME>.search_timing.jsonl
  <RUN_NAME>.nvidia_smi.csv
  <RUN_NAME>.timing_report.latest.md
  <RUN_NAME>.training_metrics_report.latest.md
  <RUN_NAME>.detailed_metrics_report.latest.md
  <RUN_NAME>.metrics.latest_<plot_group>.png
```

周期性 snapshot 和训练结束 final/all 都写同一组 `latest` 文件。新主链路不再生成 `*.step<N>.*` 历史报告文件；历史 run 中已经存在的 step 报告不会被自动删除。

评估日志命名规则：

```text
log/eval_res/<TASK_NAME>/
  retriever_infer_smoke.jsonl
  runtime_logs/
    <RUN_NAME>.env
    <RUN_NAME>.infer.log
    <RUN_NAME>.recall_retriever_server.log
reports/eval/<TASK_NAME>.report.md
```

关键环境变量：

- `LOG_TIMESTAMP`：训练日志目录时间戳，默认 `date +%y%m%d-%H%M`。
- `RUN_NAME`：训练 run 名称，也作为训练日志文件名前缀。
- `LOG_EXPERIMENT_NAME` / `EXPERIMENT_NAME`：训练日志目录名的实验名来源；未设置时使用调用方传入的 default experiment。
- `REPORT_SCHEMA_PATH`：项目 train report schema 文件路径。不同项目应在自己的 `scripts/<project>/assets/report_schema.py` 中维护 schema。
- `TRAIN_REPORT_SNAPSHOT_MODE`：周期性 snapshot 模式，默认 `latest`。设置为 `scheduled` 时才按 `REPORT_STEPS` 选择 step-limit。
- `REPORT_STEPS`：兼容旧式 scheduled snapshot 的 step 列表，例如 `10` 或 `10 50 100`。
- `REPORT_INTERVAL_SECONDS`：后台 reporter 生成周期，默认由调用脚本设置，建议 `60`。
- `NVIDIA_SMI_INTERVAL`：GPU CSV 采样间隔，默认由调用脚本设置，建议 `10`。
- `MAIN_GPU_IDS`：timing report 中标记 agent/main GPU 的物理 GPU 列表。
- `RERANKER_GPU_IDS`：timing report 中标记 reranker/rank retriever GPU 的物理 GPU 列表。
- `TASK_NAME`：评估任务名；未设置时由 `setup_cosearch_eval_artifact_defaults` 生成 `<YYMMDD-HHMM>-<STRATEGY_SLUG>`。

使用约定：

- 训练、推理、评估入口脚本可 source `src/logs/report_system/logging_reports.sh` 使用统一日志路径；其中本文档描述的 `train_*` Python 入口只服务训练报告。
- 不要在各脚本集合里重复拼接日志和报告默认路径。
- 不同训练子项目不要互相引用 `scripts/<other_project>` 里的报告脚本。项目只维护自己的 `assets/report_schema.py`，公共报告逻辑统一走 `src/logs/report_system`。
- 训练脚本需要完整控制台日志时，应使用 `2>&1 | tee "${TRAIN_LOG}"`。
- 训练脚本需要周期性报告时，应调用 `cosearch_start_training_reporter "${ROOT}"`；训练结束后应调用 `cosearch_generate_final_training_reports "${ROOT}"` 生成 final/all，并覆盖 snapshot 的 latest 产物。
- 训练脚本需要 GPU 使用率采样时，应调用 `cosearch_start_nvidia_smi_sampler`，退出时用 `cosearch_stop_background_pid` 清理后台进程。
- checkpoint 目录不属于日志系统；dry-run、日志初始化和 tool config 写入不应创建 checkpoint 目录。训练脚本应把 rollout/validation trace 默认放在 `LOG_DIR` 下，checkpoint 目录仅保留给实际模型 checkpoint 写入。
- `scripts/cosearch_local/generate_timing_report.py`、`generate_training_metrics_report.py`、`generate_detailed_metrics_report.py`、`plot_training_metrics.py` 只保留为兼容 wrapper，新脚本不应继续依赖它们。

项目 schema：

- `scripts/cosearch_local/assets/report_schema.py`：保留 CoSearch 旧指标 key，例如 `main/*`、`reranker/*`、`main_actor/*`、`reranker_actor/*`。
- `scripts/coagenticRetriever_local/assets/report_schema.py`：使用 CoAgenticRetriever 新指标 key，例如 `main_agent/*`、`main_agent_actor/*`、`ranker/*`。
- `scripts/iterRag_scripts/assets/report_schema.py`：AgenticIterRag 当前仍兼容历史 CoSearch key，但 schema 独立维护。

当前使用方：

- `scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh`：使用该模块初始化训练日志、启动后台 reporter、启动 nvidia-smi 采样、训练结束生成最终报告。
- `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`：与 `10_train...sh` 使用同一套训练日志、报告和 GPU 采样函数。
- `scripts/iterRag_scripts/01_train_qwen3_4b_ablation_1epoch_timing.sh`：使用该模块生成 AgenticIterRag train latest/final 报告。
- `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`：使用该模块的 eval artifact 默认路径，将推理结果落在 `log/eval_res/<TASK_NAME>/`。

完整设计见 `docs/log_and_report/report_system_1.0.md`。

## `src/runtime/`

运行时辅助模块。该目录放置跨训练、评估、实验编排任务复用的 shell 运行能力；`tasks/` 下的任务脚本和 `scripts/` 下的底层 launcher 都可以引用这里的公共函数。

当前文件：

- `wait_for_gpus.sh`：GPU 空闲等待工具。可以被 source 后调用 `wait_for_gpus_if_enabled`，也可以作为 CLI 直接执行。
- `task_sequence.sh`：串行任务编排工具。用于把多个 train/eval/task 脚本按顺序运行，并统一处理 GPU 等待、日志记录、失败策略和 GPU 释放动作。

### `wait_for_gpus.sh`

核心函数：

- `wait_for_gpu_release "<gpu_csv>" "<interval>" "<timeout>" "<label>"`：等待指定 GPU 上没有 compute process。
- `wait_for_gpus_if_enabled`：读取环境变量决定是否等待。

关键环境变量：

- `WAIT_FOR_GPU_RELEASE`：是否启用等待，`1/true/yes/on` 表示启用。
- `WAIT_FOR_GPUS`：要等待的 GPU 列表，例如 `0,1,2,3`。
- `WAIT_FOR_GPU_INTERVAL_SECONDS`：轮询间隔，默认由调用方设置，常用 `30`。
- `WAIT_FOR_GPU_TIMEOUT_SECONDS`：超时时间，`0` 表示不超时。
- `WAIT_FOR_GPU_LABEL`：日志前缀。

示例：

```bash
WAIT_FOR_GPU_RELEASE=1 \
WAIT_FOR_GPUS=0,1,2,3 \
bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix.sh
```

也可以直接作为命令使用：

```bash
bash src/runtime/wait_for_gpus.sh --gpus "0,1,2,3" --interval 30 --timeout 0 --label "train wait"
```

### `task_sequence.sh`

核心函数：

- `task_sequence_run "<label>" "<gpu_csv>" command args...`：等待指定 GPU 空闲后执行命令，并把 stdout/stderr 写入 task sequence 日志。
- `task_sequence_release_gpus "<label>" "<gpu_csv>"`：收集指定 GPU 上的进程，并按配置决定是否发送 `SIGTERM`/`SIGKILL`。

任务名说明：

- `label` 只是编排脚本中的日志标记，不是硬性约束。
- 编排结果会写入 `log/task_sequences/<stamp>-<TASK_SEQUENCE_NAME>/summary.tsv`。

关键环境变量：

- `TASK_SEQUENCE_NAME`：编排任务组名称，默认 `task_sequence`。
- `TASK_SEQUENCE_DRY_RUN`：设置为 `1` 时只展开命令，不执行任务，也不释放 GPU。
- `TASK_SEQUENCE_START_INDEX`：从第几个任务开始执行，用于失败后跳过前面已经完成的任务。
- `TASK_SEQUENCE_CONTINUE_ON_FAIL`：子任务失败后是否继续执行后续任务。
- `TASK_SEQUENCE_WAIT_FOR_GPUS`：是否由编排层等待 GPU，默认 `1`。
- `TASK_SEQUENCE_RELEASE_GPUS`：`task_sequence_release_gpus` 是否真正发送信号。默认 `0`，只列出候选进程；设置为 `1` 才会释放。
- `TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY`：默认 `1`，只释放当前用户进程。
- `TASK_SEQUENCE_RELEASE_GRACE_SECONDS`：发送 `SIGTERM` 后等待多久再发送 `SIGKILL`。

使用约定：

- 多个 train/eval 子任务需要串行运行时，优先在 `tasks/experiments/` 中写一个编排脚本，并 source `src/runtime/task_sequence.sh`。
- 如果等待逻辑放在编排层，子任务命令里应传 `WAIT_FOR_GPU_RELEASE=0`，避免子任务内部重复等待。
- GPU 释放是显式动作；只有 `TASK_SEQUENCE_RELEASE_GPUS=1` 时才会真正 kill 进程。
- `task_sequence_release_gpus` 是按 GPU 上的进程兜底清理，不应替代子任务自身的正常 shutdown 逻辑。

示例见：

- `tasks/experiments/tasks_TrainEval_00_example.sh`
- `tasks/eval_tasks/coAgenticRetriever/tasks_0622a.sh`

## `src/retrievers/`

Retriever 服务模块。

该目录存放跨 `scripts/cosearch_local`、`scripts/coagenticRetriever_local`、`scripts/iterRag_scripts` 复用的检索服务启动和代理逻辑。`scripts/` 下的某个子目录不是公共层；如果多个脚本集合都要启动 dense retriever 或 round-robin proxy，应引用这里的公共入口。

当前文件：

- `start_dense_retriever_server.sh`：dense retriever server 的统一启动脚本。默认 `--mode cpu`，启动 Search-R1 原生 FAISS/CPU retriever；也支持 `--mode gpu`，启动本项目的 GPU-resident torch retriever。支持通过环境变量覆盖 `COSEARCH_PROJECT_ROOT`、`EXTERNAL_MODEL_ROOT`、`EXTERNAL_RETRIEVAL_ROOT`、`RETRIEVAL_DATA_DIR`、`INDEX_FILE`、`CORPUS_FILE`、`RETRIEVER_MODEL`、`SEARCH_R1_RETRIEVAL_SERVER` 等路径。
- `gpu_dense_retriever_server.py`：GPU dense retriever server。该服务从 `e5_Flat.index` 读取全量 doc embeddings，转成 torch tensor 常驻 GPU；query encoder 也在 GPU 上运行；召回计算为 `query_emb @ doc_embeddings.T` 后 `torch.topk`。用于当前环境 faiss-gpu API 不完整、无法可靠走 Search-R1 原生 FAISS GPU 路径的场景。
- `retrieval_round_robin_proxy.py`：多个 dense retriever 实例的 round-robin HTTP proxy，统一暴露 `/retrieve`。
- `verify_official_retrieval_assets.py`：校验 Search-R1 wiki-18 corpus 和 FAISS index 是否可用，也可选校验 `/retrieve` endpoint。

`start_dense_retriever_server.sh` 主要参数：

- `--mode cpu|gpu`：启动模式，默认 `cpu`。
- `--port PORT`：服务端口，默认 `8010`。
- `--gpu-id GPU_ID`：GPU 模式使用的 GPU id，默认 `5`。
- `--doc-dtype float16|float32`：GPU doc embedding dtype，默认 `float16`。
- `--query-batch-size N`：GPU server 内部 query batch size，默认 `32`。

示例：

```bash
# CPU / Search-R1 原生 FAISS 路径
bash src/retrievers/start_dense_retriever_server.sh

# GPU5 + float16 doc embedding
bash src/retrievers/start_dense_retriever_server.sh --mode gpu --gpu-id 5 --doc-dtype float16 --port 8050

# GPU5 + float32 doc embedding，更接近原始 IndexFlatIP 数值行为
bash src/retrievers/start_dense_retriever_server.sh --mode gpu --gpu-id 5 --doc-dtype float32 --port 8050
```

GPU 模式资源和对齐注意事项：

- 当前 wiki-18 全量索引为 `21,015,324 x 768`。
- H20 GPU5 实测：`float16` 服务压测后约占 `42GB` 显存；`float32` batch=1 压测峰值约 `66GB` 显存。
- 启动到 ready 状态约 `40-45s`，其中包括读取 FAISS flat index、将 doc embeddings 拷贝到 GPU、加载 E5 encoder 和 corpus。
- GPU 模式没有调用 FAISS `index.search()`，而是使用 torch GPU matmul/topk；数学形式与 Search-R1 的 `IndexFlatIP` inner product 对齐，但 `float16` 可能因量化导致边界样本 top-k 排序微差。
- 严格论文复现建议优先使用 `--doc-dtype float32`，并抽样对比 CPU FAISS top-k 与 GPU torch top-k overlap。
- GPU 服务启动、GPU 压测和本机 HTTP 访问需要在非沙盒/提权环境执行；沙盒中可能出现 `nvidia-smi` 可见但 PyTorch CUDA 或 socket 不可用。

使用约定：

- 训练和评估脚本启动 dense retriever 时，应调用 `src/retrievers/start_dense_retriever_server.sh`。
- 需要多个 retriever 后端负载均衡时，应调用 `src/retrievers/retrieval_round_robin_proxy.py`。
- 不要从 `scripts/cosearch_local/` 或 `scripts/coagenticRetriever_local/` 跨目录引用 retriever 启动脚本；这些目录只放各自实验入口。

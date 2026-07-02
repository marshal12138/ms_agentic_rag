# Mix-Signal Fix 配置治理刷新计划

日期：2026-06-26

参考旧计划：

```bash
docs/planning/260623_mix_signal_fix_config_governance_plan.md
```

目标 canonical 入口：

```bash
tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

## 1. 最新状态判断

旧计划的核心意图仍然成立：不要把 task shell 写成半个 launcher；基础配置应回到 Hydra config group；实验差异进入 partial overlay YAML；运行环境和 GPU/device override 由 launcher 生成可审计的 runtime override。

但当前代码状态相比 2026-06-23 已经变化：

- 仓库当前工作树干净，可以直接新建治理分支实施。
- `src/hydra_overrides/yaml_to_dotlist.py` 和 `src/hydra_overrides/hydra_overrides.sh` 已经存在，并且明确拒绝 top-level `defaults:`，适合作为 partial overlay 转 dotlist 工具。
- `scripts/coagenticRetriever_v2/01_train_launcher.sh` 已经具备日志、checkpoint、recall service、LLM judge service、dry-run、env snapshot 和 YAML override 转 dotlist 能力。
- 新增了 `scripts/coagenticRetriever_local/`，它基本复制 v2 训练链路，但增加 `compatible_accelerator.sh`、`RUN_MODE=no-ranker`、`coagentic_retriever_tool_config_no_ranker.yaml` 等本地/异构加速路径。
- canonical mix-signal task 当前使用 `scripts/coagenticRetriever_local/strategies_yaml/*.yaml` 作为 overlay，但最终仍调用 `scripts/coagenticRetriever_v2/01_train_launcher.sh`。这是新的不一致点。
- canonical task 仍然通过 `HYDRA_OVERRIDE_YAMLS`、`COAGENTIC_EXTRA_ARGS`、`DEFAULT_COAGENTIC_EXTRA_ARGS` 和 `"$@"` 传递关键 Hydra 参数，旧问题没有解决。
- `CoAgenticRetriever/config/data/`、`model/`、`rollout/` 目前仍只有 `legacy_data.yaml`、`hf_model.yaml`、`rollout.yaml`，旧计划中建议的新 config group 尚未落地。

本轮刷新计划应先修正治理路线，避免同时治理 v2/local 两套 launcher。

## 2. 治理边界

本轮只治理 canonical mix-signal fix 入口及其直接调用链：

- `tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh`
- `scripts/coagenticRetriever_v2/01_train_launcher.sh`
- `scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh`
- `src/hydra_overrides/*.sh|*.py`
- 必要的 `CoAgenticRetriever/config/<group>/` 和 `scripts/tasks` overlay YAML

不在本轮治理：

- `scripts/coagenticRetriever_local/` 的 no-ranker 和异构加速路径。
- 其它 legacy task，例如 `train_CAR_async_ranker_training_ds_flash.sh`、`train_CAR_async_ranker_training_ds_flash_mix_signal.sh`、`train_CAR_async_ranker_training_ds_flash_mix_signal_fix_v1.sh`、`train_CAR_async_ranker_training_ds_flash_mix_signal_fix_exp02.sh`。
- `tasks/train_tasks/coAgenticRetriever/train_0625a_npu_async_ranker_training_ds_flash_mix_signal_fix_exp03.sh`，它当前依赖 `_v1` task 和 CLI 透传，本轮只记录为待迁移入口。

如果后续要把 local/no-ranker 路径纳入治理，应在 canonical v2 路径稳定后单独制定第二阶段计划。

## 3. 目标形态

### 3.1 配置分层

最终形成四层配置，优先级从低到高：

```text
Hydra 主配置 + config groups
< reusable overlay YAML
< task overlay YAML
< runtime env override YAML
```

职责划分：

- Hydra config groups：长期稳定基础配置，例如数据集、Qwen3-4B 模型、CoSearch-aligned rollout budget。
- reusable overlay：可被多个任务复用的策略覆盖，例如 DeepSeek-Flash async ranker training rank50 select-all。
- task overlay：单个实验差异，例如 mix-signal b3 的 `sample_builder_request_batch=3` 和默认资源布局。
- runtime override：launcher 根据 task env 生成的最终运行覆盖，例如 GPU/device、run name、log/checkpoint 路径。

### 3.2 canonical task

canonical task 只声明：

- 任务身份：`EXP_NAME`、`GROUP_NAME`、可选 `RUN_STAMP`。
- 运行资源：`AGENT_GPU_IDS`、`RANK_GPU_ID`、`RECALL_GPU_ID`、`LLM_JUDGE_GPU_IDS`。
- 服务生命周期：recall/judge auto-start、auto-stop、wait/preflight。
- 明确选择 config group 和 overlay YAML。

canonical task 不再设置：

- `COAGENTIC_EXTRA_ARGS`
- `DEFAULT_COAGENTIC_EXTRA_ARGS`
- `COAGENTIC_DEFAULT_EXTRA_ARGS`
- `HYDRA_OVERRIDE_YAMLS`
- `SAMPLE_BUILDER_REQUEST_BATCH`
- 训练 batch、rollout budget、micro batch 等算法参数 env
- `"$@"` 隐式透传

### 3.3 launcher 接口

v2 launcher 增加显式参数接口：

```bash
bash scripts/coagenticRetriever_v2/01_train_launcher.sh \
  --DATA_CONFIG=co_search_ablation \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=cosearch_async_qwen3_4b \
  --OVERLAY_YAML=scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_deepseek_flash_rank50_select_all.yaml \
  --OVERLAY_YAML=tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml \
  --LLM_JUDGE_SERVICE_CONFIG=CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

解析规则：

- `--DATA_CONFIG=name` -> `data@data=name`，并校验 `CoAgenticRetriever/config/data/${name}.yaml`。
- `--MODEL_CONFIG=name` -> `model@actor_rollout_ref.model=name`，并校验 `CoAgenticRetriever/config/model/${name}.yaml`。
- `--ROLLOUT_CONFIG=name` -> `rollout@actor_rollout_ref.rollout=name`，并校验 `CoAgenticRetriever/config/rollout/${name}.yaml`。
- `--OVERLAY_YAML=path` 可重复，按传入顺序转 dotlist，后者覆盖前者。
- `--LLM_JUDGE_SERVICE_CONFIG=path` 只用于 judge 服务启动和 dry-run 校验。

为降低风险，v2 launcher 可以短期保留旧 env 入口兼容其它 task；但 canonical task 不能再使用旧入口。

## 4. 需要新增或调整的配置

### 4.1 Hydra config groups

新增：

```bash
CoAgenticRetriever/config/data/co_search_ablation.yaml
CoAgenticRetriever/config/model/qwen3_4b.yaml
CoAgenticRetriever/config/rollout/cosearch_async_qwen3_4b.yaml
```

建议迁入内容：

- `data/co_search_ablation.yaml`：`train_files`、`val_files`、`train_max_samples`、`val_max_samples`、batch 默认值、`apply_chat_template_kwargs.enable_thinking=false`。
- `model/qwen3_4b.yaml`：Qwen3-4B `path`、`trust_remote_code=true`、`use_remove_padding`、LoRA 默认值、attention override 相关稳定默认。
- `rollout/cosearch_async_qwen3_4b.yaml`：CoSearch-aligned prompt/response budget、vLLM rollout mode、`max_model_len`、`max_num_batched_tokens`、`max_num_seqs`、multi-turn/tool budget、agent worker 默认值。

注意：这些文件可以通过 Hydra `defaults` 继承现有 base group，但不得放 run-name、GPU、service lifecycle。

### 4.2 Reusable overlay

新增或复制归一：

```bash
scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_deepseek_flash_rank50_select_all.yaml
```

内容以当前 `scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml` 为基准，保留 DeepSeek-Flash async ranker training rank50 select-all 策略：

- `ranker_training.signal_source=async_ranker_training`
- `ranker_training.shared_inference_ranker.*`
- `ranker_training.async_ranker_training.*`
- `llm_as_judge` stage endpoint/model/prompt/schema
- sample builder 和 async logging 默认

不放：

- `sample_builder_request_batch=3`
- GPU/device
- run-name/log/checkpoint 路径

### 4.3 Task overlay

新增：

```bash
tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml
```

内容只放 canonical b3 实验差异：

```yaml
ranker_training:
  async_ranker_training:
    sample_builder_request_batch: 3

resources:
  agent_gpu_ids: "0,1,2,3"
  rank_gpu_id: "4"
  recall_gpu_id: "5"
  llm_judge_gpu_ids: "6,7"
```

如需保留 `inject_tool_schema=false` 和 FSDP offload override，应优先判断它们是否属于长期默认：

- 若所有 Qwen3-4B CoSearch-aligned 训练都需要，放入 config group 或 reusable overlay。
- 若仅 canonical b3 需要，放入 task overlay。
- 不再通过 task shell 拼 `COAGENTIC_EXTRA_ARGS`。

### 4.4 Runtime override YAML

v2 launcher 生成：

```bash
${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml
```

至少包含：

```yaml
trainer:
  experiment_name: ${EXP_NAME}
  default_local_dir: ${OUT_DIR}
  n_gpus_per_node: <AGENT_GPU_IDS count>
  rollout_data_dir: ${ROLLOUT_DATA_DIR}
  validation_data_dir: ${VALIDATION_DATA_DIR}

recall_retriever:
  device: cuda:${RECALL_GPU_ID}
  service_url: ${RETRIEVAL_SERVICE_URL}

ranker:
  device: cuda:${RANK_GPU_ID}

ranker_training:
  construction_log_jsonl: ${LOG_DIR}/${RUN_NAME}.contrastive_construction.jsonl

resources:
  agent_gpu_ids: ${AGENT_GPU_IDS}
  rank_gpu_id: ${RANK_GPU_ID}
  recall_gpu_id: ${RECALL_GPU_ID}
  llm_judge_gpu_ids: ${LLM_JUDGE_GPU_IDS}
```

runtime override 必须排在所有 overlay YAML 之后，确保 task env 的 GPU/device 覆盖最后生效。

## 5. Launcher 和 asset runner 调整

### 5.1 v2 launcher

新增功能：

- 解析 `--DATA_CONFIG`、`--MODEL_CONFIG`、`--ROLLOUT_CONFIG`、重复 `--OVERLAY_YAML`、`--LLM_JUDGE_SERVICE_CONFIG`。
- 校验 config group 文件存在。
- 校验 overlay YAML 存在且不含 top-level `defaults:`。
- 读取 task overlay 中的 `resources` 默认值，再用 env GPU 变量覆盖。
- 生成 runtime override YAML，并通过 `yaml_to_dotlist.py` 转为 Hydra args。
- dry-run 写出完整审计文件，不启动训练和服务。

保留但标记 legacy：

- `HYDRA_OVERRIDE_YAMLS`
- `RANKER_STRATEGY_YAML`
- `ASYNC_RANKER_TRAINING_YAML`
- `COAGENTIC_EXTRA_ARGS`
- `COAGENTIC_DEFAULT_EXTRA_ARGS`

这些 legacy 入口只能服务旧 task；canonical task 不再使用。

### 5.2 Asset runner

调整 `scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh`：

- 支持接收 launcher 组装好的 Hydra group selections、overlay dotlist、runtime dotlist。
- 对 canonical path，最终参数顺序固定为：

```text
minimal script defaults
< Hydra group selections
< reusable overlay dotlist
< task overlay dotlist
< runtime env override dotlist
```

- canonical path 不再插入 `COAGENTIC_DEFAULT_EXTRA_ARGS` 和 `COAGENTIC_EXTRA_ARGS`。
- `"$@"` 只允许 legacy path 使用；canonical path 禁止 CLI 透传，避免不可审计覆盖。

如果一次性删除 hardcoded Hydra args 风险过高，第一阶段允许保留 asset runner 的通用 hardcoded defaults，但必须保证 canonical 相关字段可被 group/overlay/runtime override 覆盖，并在审计文件中可见最终顺序。

## 6. 审计输出

dry-run 和正式运行都写入：

```bash
${LOG_DIR}/${RUN_NAME}.env
${LOG_DIR}/${RUN_NAME}.hydra_groups.txt
${LOG_DIR}/${RUN_NAME}.overlay_yamls.txt
${LOG_DIR}/${RUN_NAME}.runtime_env_overrides.yaml
${LOG_DIR}/${RUN_NAME}.hydra_args.txt
```

要求：

- `.env` 保留当前已有环境快照，并增加新接口字段。
- `hydra_groups.txt` 记录 `data@data=...`、`model@actor_rollout_ref.model=...`、`rollout@actor_rollout_ref.rollout=...`。
- `overlay_yamls.txt` 记录 overlay 文件顺序。
- `runtime_env_overrides.yaml` 记录 env 覆盖后的 GPU/device/run path。
- `hydra_args.txt` 记录最终传给 Python 主程序的 dotlist，按真实顺序输出。

canonical dry-run 审计中不得出现：

- `COAGENTIC_EXTRA_ARGS`
- `COAGENTIC_DEFAULT_EXTRA_ARGS`
- task 内部 `HYDRA_OVERRIDE_YAMLS`
- task 末尾 `"$@"`

## 7. 实施顺序

1. 建立保护分支，确认工作树干净。
2. 新增三个 Hydra config group，不接入运行链路。
3. 新增 reusable overlay 和 task overlay，不改变旧 task。
4. 给 v2 launcher 增加新参数解析、文件校验和 overlay 列表组装。
5. 给 v2 launcher 增加 runtime override YAML 生成和审计输出。
6. 调整 asset runner，让 canonical path 使用显式 group/overlay/runtime args。
7. 重写 canonical mix-signal task 为声明式入口，并移除旧 extra-args/env 透传。
8. 运行静态检查、overlay 转换、dry-run 验证。
9. 只在 canonical path 验收通过后，再评估是否把同样机制迁移到 `_v1`、`exp02`、`0625a_npu` 或 `coagenticRetriever_local`。

每一步单独提交，方便回滚。

## 8. 验证计划

静态检查：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash -n tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
bash -n scripts/coagenticRetriever_v2/01_train_launcher.sh
bash -n scripts/coagenticRetriever_v2/assets/00_run_agentic_iter_rag_verl.sh
```

Hydra config group 验证：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever
/data04/envs/ms/ms_cosearch_official/bin/python main_coagentic_retriever.py \
  --cfg job \
  data@data=co_search_ablation \
  model@actor_rollout_ref.model=qwen3_4b \
  rollout@actor_rollout_ref.rollout=cosearch_async_qwen3_4b
```

Overlay 转换验证：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
/data04/envs/ms/ms_cosearch_official/bin/python \
  src/hydra_overrides/yaml_to_dotlist.py \
  scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_deepseek_flash_rank50_select_all.yaml \
  tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml
```

Canonical dry-run：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
DRY_RUN=1 bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

NPU/local 非目标路径回归检查：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash -n tasks/train_tasks/coAgenticRetriever/train_0625a_npu_async_ranker_training_ds_flash_mix_signal_fix_exp03.sh
bash -n scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
bash -n scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
```

说明：本轮不迁移 NPU/local 路径，但这些脚本必须保持语法可用，且实现过程中不得删除它们依赖的 legacy env/CLI 兼容入口。

验收条件：

- dry-run 成功生成所有审计文件。
- `hydra_groups.txt` 包含三个目标 config group selection。
- `overlay_yamls.txt` 中 reusable overlay 排在 task overlay 前。
- `runtime_env_overrides.yaml` 包含 env 覆盖后的 GPU/device。
- `hydra_args.txt` 包含 `ranker_training.async_ranker_training.sample_builder_request_batch=3`。
- `hydra_args.txt` 中 runtime device override 排在 task overlay 之后。
- canonical task 不再包含 `COAGENTIC_EXTRA_ARGS`、`DEFAULT_COAGENTIC_EXTRA_ARGS`、`HYDRA_OVERRIDE_YAMLS`、`SAMPLE_BUILDER_REQUEST_BATCH`、`"$@"`。
- NPU/local 非目标路径的 `bash -n` 回归检查通过。
- 旧 task 未被修改，除非单独进入后续迁移阶段。

## 9. 风险控制

- 不同时治理 `v2` 和 `local` 两套 launcher。canonical 先固定走 v2；local/no-ranker 后续再迁移。
- 不删除 legacy env 入口，只禁止 canonical task 使用，避免破坏其它训练脚本。
- 不删除 `_v1`、`0625a_npu`、`coagenticRetriever_local` 当前依赖的 `"$@"`、`COAGENTIC_EXTRA_ARGS`、`HYDRA_OVERRIDE_YAMLS` 等兼容入口。
- canonical v2 的 runtime device override 可以使用 `cuda:${id}`；该逻辑不得复用到 NPU/local 路径，NPU/local 后续迁移必须经过 `compatible_accelerator.sh` 或等价 device 映射层。
- 不把 `yaml_to_dotlist.py` 扩展成 Hydra composition 工具；config group selection 必须走 Hydra 原生 override。
- 不把 service lifecycle、GPU wait、checkpoint cleanup 放进训练 YAML。
- 不把 task overlay 写成大而全 recipe；它只表达单个实验差异。

建议提交粒度：

```bash
git commit -m "add mix-signal hydra config groups and overlays"
git commit -m "support explicit canonical config selections in v2 launcher"
git commit -m "audit canonical mix-signal hydra args"
git commit -m "rewrite canonical mix-signal task declaration"
```

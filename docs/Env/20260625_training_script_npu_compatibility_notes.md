# 训练脚本 NPU 兼容改造记录

本文记录为了让 CoAgenticRetriever 训练脚本在 Ascend NPU 机器上跑通所做的代码兼容改造、遇到的问题、修复逻辑和验证结果。

入口脚本：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh
```

最终验证模式：

```bash
COSEARCH_ACCELERATOR=npu
RUN_MODE=no-ranker
TOTAL_STEPS=1
```

## 目标和约束

目标：

1. 保持入口训练脚本使用体验尽量不变。
2. 默认仍兼容原 CUDA/H20 环境。
3. 当检测到 NPU 环境时，自动将 GPU/CUDA 相关命令和设备配置切换到 NPU/Ascend。
4. 尽量把兼容逻辑放在共享层和被调用脚本中，而不是把入口 task 脚本改得很重。

关键约束：

- 不直接重写训练主逻辑。
- 不为了 NPU 支持移除 CUDA 运行能力。
- 不强行让 no-ranker 以外的 async-labeling/judge 路径假装已验证。

## 总体调用链

入口 task 脚本调用链大致为：

```text
tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh
  -> src/runtime/wait_for_gpus.sh
  -> scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
     -> scripts/coagenticRetriever_local/00_start_dense_retriever_server.sh
        -> src/retrievers/gpu_dense_retriever_server.py
     -> scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
        -> CoAgenticRetriever/main_coagentic_retriever.py
           -> CoAgenticRetriever/verl/...
```

本轮主要改造集中在：

```text
src/env_manage/compatible_accelerator.sh
src/env_manage/compatible_python.sh
src/runtime/wait_for_gpus.sh
scripts/coagenticRetriever_local/00_start_dense_retriever_server.sh
scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
src/retrievers/gpu_dense_retriever_server.py
src/retrievers/start_dense_retriever_server.sh
CoAgenticRetriever/verl/verl/trainer/constants_ppo.py
CoAgenticRetriever/verl/verl/utils/vllm/utils.py
CoAgenticRetriever/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
```

另有一个临时 site-packages patch：

```text
.venvs/ms_agt_rag_overlay/lib/python3.11/site-packages/vllm/v1/engine/core.py
```

该 patch 不在 git 管理中，重建 overlay venv 后需要重新处理。

## 新增 NPU/GPU 兼容层

新增共享脚本：

```bash
src/env_manage/compatible_accelerator.sh
```

作用：

1. 自动检测当前加速器类型。
2. 封装 CUDA/NPU 可见设备变量。
3. 封装 `cuda`/`npu` 设备前缀。
4. 封装设备数量、设备 id、进程查询。
5. NPU 模式下自动加载 CANN/ATB。
6. NPU 模式下设置 vLLM Ascend 和 HCCL 默认环境变量。

### 加速器检测

检测逻辑：

```bash
co_accel_detect() {
  if [[ -n "${COSEARCH_ACCELERATOR:-}" ]]; then
    printf '%s\n' "${COSEARCH_ACCELERATOR}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'gpu\n'
    return 0
  fi
  if command -v npu-smi >/dev/null 2>&1 || compgen -G "/dev/davinci[0-9]*" >/dev/null; then
    printf 'npu\n'
    return 0
  fi
  printf 'cpu\n'
}
```

默认优先 GPU，这是为了不改变原 H20/CUDA 机器行为。如果一台机器同时有 `nvidia-smi` 和 `npu-smi`，可以显式覆盖：

```bash
export COSEARCH_ACCELERATOR=npu
```

### 设备变量封装

NPU 使用：

```bash
ASCEND_RT_VISIBLE_DEVICES
```

CUDA 使用：

```bash
CUDA_VISIBLE_DEVICES
```

统一封装：

```bash
co_accel_visible_devices_var
co_accel_export_visible_devices
co_accel_env_visible_devices_cmd
```

NPU 下为了兼容部分仍读取 CUDA 变量的第三方代码，`co_accel_export_visible_devices` 同时设置：

```bash
ASCEND_RT_VISIBLE_DEVICES="${ids}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${ids}}"
```

### 设备前缀封装

训练 Hydra 参数、检索服务 device 参数都通过：

```bash
co_accel_device_prefix
```

得到：

```text
gpu/cuda 环境 -> cuda
npu/ascend 环境 -> npu
```

### CANN/ATB 自动加载

NPU 模式下兼容层自动尝试：

```bash
source "${COSEARCH_ASCEND_CANN_SET_ENV:-/usr/local/Ascend/cann/set_env.sh}"
source "${COSEARCH_ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}" \
  "--cxx_abi=${COSEARCH_ASCEND_ATB_CXX_ABI:-1}"
```

为什么需要这个：

- 训练中的 vLLM Ascend/torch_npu ATB 路径需要 `libatb.so`。
- 仅 source CANN 不够，必须 source NNAL/ATB。
- 当前 torch CXX11 ABI 为 True，所以默认 `cxx_abi=1`。

### NPU 默认环境变量

兼容层设置：

```bash
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
```

其中 `VLLM_ASCEND_ENABLE_NZ=0` 是为了解决 vLLM Ascend RL 路径报错：

```text
FRACTAL_NZ mode is enabled...
Please set VLLM_ASCEND_ENABLE_NZ=0.
```

`HCCL_*TIMEOUT` 是通信 watchdog 上限，不表示训练固定等待这么久。

## Python 兼容层

修改：

```bash
src/env_manage/compatible_python.sh
```

主要变化：

1. 保留调用方传入的 `PY`。
2. 默认从历史 Python 环境或 `/data05/conda/envs/ms/ms_agt_rag` 中选择可用 Python。
3. 将 Python `LIBDIR` 前置到 `LD_LIBRARY_PATH`。

目的：

- 避免动态库加载时找不到 Python 自身 libdir。
- 减少每个启动脚本重复写环境路径。

可关闭：

```bash
export COSEARCH_PREPEND_PYTHON_LIBDIR=0
```

## wait_for_gpus 支持 NPU

修改：

```bash
src/runtime/wait_for_gpus.sh
```

原问题：

- 入口脚本会先等待 GPU 释放。
- 原等待逻辑基于 `nvidia-smi`。
- NPU 机器没有 `nvidia-smi` 或不应使用 `nvidia-smi`。

改造内容：

1. 如果 `co_accel_device_ids` 等 helper 不存在，自动 source `compatible_accelerator.sh`。
2. `wait_for_gpu_release` 在 NPU 模式下转发到 `wait_for_npu_release`。
3. `wait_for_npu_release` 使用 `npu-smi` 检查目标 NPU 是否存在、是否有进程占用。

踩坑：

曾经只在 `COSEARCH_ACCELERATOR` 为空时 source 兼容层：

```bash
if [[ -z "${COSEARCH_ACCELERATOR:-}" ]]; then
  source compatible_accelerator.sh
fi
```

但实际调试经常会显式设置：

```bash
COSEARCH_ACCELERATOR=npu
```

这会导致兼容层没有加载，进而出现：

```text
co_accel_device_ids: command not found
```

修复为：

```bash
if ! declare -F co_accel_device_ids >/dev/null 2>&1; then
  source compatible_accelerator.sh
fi
```

## dense retriever 服务 NPU 兼容

涉及文件：

```text
scripts/coagenticRetriever_local/00_start_dense_retriever_server.sh
src/retrievers/start_dense_retriever_server.sh
src/retrievers/gpu_dense_retriever_server.py
```

### 启动脚本改造

原逻辑硬编码 CUDA：

```bash
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU_IDS}"
```

改造后：

```bash
source src/env_manage/compatible_accelerator.sh
DEVICE="${DEVICE:-$(co_accel_device_prefix)}"
env $(co_accel_env_visible_devices_cmd "${RETRIEVER_GPU_IDS}") ...
```

NPU 下实际效果：

```bash
ASCEND_RT_VISIBLE_DEVICES=5
CUDA_VISIBLE_DEVICES=5
DEVICE=npu
```

### Python server 改造

`gpu_dense_retriever_server.py` 原本只接受 cuda device：

```python
if not device.startswith("cuda"):
    raise ValueError("gpu_dense_retriever_server requires a cuda device")
```

改造后支持：

```python
if not (device.startswith("cuda") or device.startswith("npu")):
    raise ValueError("gpu_dense_retriever_server requires a cuda or npu device")
```

同时增加：

```python
ensure_device_backend(device)
synchronize_device(device)
memory_status(device)
```

用于：

- NPU 模式下导入 `torch_npu`。
- 将 `torch.cuda.synchronize` 切换为 `torch.npu.synchronize`。
- 将 CUDA memory status 切换为 NPU memory status。

## VERL 启动脚本 NPU 兼容

修改：

```bash
scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
```

### source 兼容层

新增：

```bash
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_accelerator.sh"
```

### 可见设备

原逻辑：

```bash
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
```

改造后：

```bash
co_accel_export_visible_devices "${GPU_IDS}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
if co_accel_is_npu; then
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
fi
```

### trainer.device

新增 Hydra 参数：

```bash
trainer.device="$(co_accel_device_prefix)"
```

NPU 下为：

```bash
trainer.device=npu
```

### HCCL/NCCL timeout

新增环境变量：

```bash
NCCL_TIMEOUT="${NCCL_TIMEOUT:-${HCCL_TIMEOUT:-}}"
```

NPU 下默认：

```bash
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
```

传入 Hydra：

```bash
actor_rollout_ref.nccl_timeout="${NCCL_TIMEOUT:-600}"
```

说明：

- 这个字段在 verl 中仍叫 `nccl_timeout`，但 NPU 后端会走 HCCL。
- 原默认值为 600 秒。
- 第一次 NPU smoke 在 `compute_log_prob` 的 `ALLGATHER` 阶段触发过 600 秒 watchdog，报错类似：

```text
Watchdog caught collective operation timeout:
WorkHCCL(SeqNum=622, OpType=ALLGATHER, Timeout(ms)=600000)
```

设置为 1800 后，不是让训练固定等 30 分钟，而是把单次 collective 的 watchdog 上限提高。最终 smoke 中完整 step 实际训练耗时约 210 秒。

### NPU 下默认关闭 actor torch compile

新增：

```bash
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-}"
if co_accel_is_npu; then
  ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
fi
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-true}"
```

传入 Hydra：

```bash
actor_rollout_ref.actor.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}"
```

原因：

- verl 官方 NPU 示例中也显式设置 `actor_rollout_ref.actor.use_torch_compile=False`。
- 本轮失败日志显示首轮会进入 torch compile debug trace，然后 HCCL all_gather 等待很久。
- 对当前 smoke 来说，关闭 actor torch compile 后训练完整跑通。

如需实验打开：

```bash
export ACTOR_USE_TORCH_COMPILE=true
```

### TOOL_CONFIG 支持外部覆盖

原逻辑固定：

```bash
TOOL_CONFIG="${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml"
```

改为：

```bash
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml}"
```

这样入口 no-ranker 模式可以传入：

```bash
CoAgenticRetriever/config/coagentic_retriever_tool_config_no_ranker.yaml
```

## Ray runtime env 保留 vLLM Ascend 变量

修改：

```bash
CoAgenticRetriever/verl/verl/trainer/constants_ppo.py
```

问题：

- verl 的 `get_ppo_ray_runtime_env()` 会把 parent env 中已经存在的变量从 runtime_env 里移除。
- `VLLM_ASCEND_ENABLE_NZ=0` 虽然在父进程里设置了，但 Ray worker 中不一定能保留下来。
- vLLM Ascend worker 仍会报 FRACTAL_NZ 相关错误。

修复：

```python
for key in os.environ:
    if key.startswith("VLLM_ASCEND_"):
        runtime_env["env_vars"].setdefault(key, os.environ[key])

for key in list(runtime_env["env_vars"].keys()):
    if os.environ.get(key) is not None and not key.startswith("VLLM_ASCEND_"):
        runtime_env["env_vars"].pop(key, None)
```

验证结果：

```text
ray init kwargs ... 'VLLM_ASCEND_ENABLE_NZ': '0'
```

## vLLM 0.13 API 兼容

当前环境采用：

```text
vllm==0.13.0
vllm-ascend==0.13.0
```

仓库代码原本更偏向另一版 vLLM API，因此做了兼容。

### LoRAModel import 兼容

修改：

```bash
CoAgenticRetriever/verl/verl/utils/vllm/utils.py
```

兼容：

```python
try:
    from vllm.lora.models import LoRAModel
except ModuleNotFoundError:
    from vllm.lora.lora_model import LoRAModel
```

当前 smoke 中 `lora_rank=0`，不是 LoRA 训练，但该 import 仍会被框架路径加载。

### FlexibleArgumentParser/get_tcp_uri import 兼容

修改：

```bash
CoAgenticRetriever/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
```

兼容：

```python
try:
    from vllm.utils import FlexibleArgumentParser, get_tcp_uri
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import get_tcp_uri
```

### AsyncLLM.from_vllm_config 参数兼容

不同 vLLM 版本中参数名不同：

- 一些版本使用 `disable_log_requests`
- 当前版本使用 `enable_log_requests`

改造为通过 `inspect.signature` 判断：

```python
if "disable_log_requests" in inspect.signature(AsyncLLM.from_vllm_config).parameters:
    llm_kwargs["disable_log_requests"] = disable_log_requests
else:
    llm_kwargs["enable_log_requests"] = not disable_log_requests
```

### init_app_state 签名兼容

之前遇到错误：

```text
TypeError: init_app_state() takes 3 positional arguments but 4 were given
```

兼容逻辑：

```python
init_app_state_params = inspect.signature(init_app_state).parameters
if len(init_app_state_params) >= 4:
    init_app_state_result = init_app_state(engine_client, vllm_config, app.state, args)
else:
    init_app_state_result = init_app_state(engine_client, app.state, args)
if inspect.isawaitable(init_app_state_result):
    await init_app_state_result
```

## vLLM EngineCore None future 临时 patch

问题：

vLLM Ascend 在本环境中出现：

```text
AttributeError: 'NoneType' object has no attribute 'result'
```

位置：

```bash
.venvs/ms_agt_rag_overlay/lib/python3.11/site-packages/vllm/v1/engine/core.py
```

原因：

- vLLM V1 `EngineCore.step()` 假设 `execute_model(..., non_block=True)` 返回 future。
- 当前 vLLM Ascend executor 某些情况下会同步返回 `None`。
- 原代码直接调用 `future.result()`，因此崩溃。

临时 patch 逻辑：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with self.log_error_detail(scheduler_output):
    if future is None:
        model_output = None
    elif hasattr(future, "result"):
        model_output = future.result()
    else:
        model_output = future
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
```

注意：

- 这是 site-packages 临时补丁，不是仓库源码。
- 删除或重建 `.venvs/ms_agt_rag_overlay` 后会丢失。
- 后续最好升级到 vLLM Ascend 官方修复版本，或把 patch 作为明确的环境构建步骤固化。

## no-ranker 模式相关

为了在当前双角色 NPU 资源下先验证训练主链路，使用了：

```bash
RUN_MODE=no-ranker
```

该模式下：

```bash
ENABLE_ASYNC_LABELING=0
AUTO_START_LLM_JUDGE=0
AUTO_STOP_LLM_JUDGE=0
LLM_JUDGE_PREFLIGHT=0
```

使用 tool config：

```bash
CoAgenticRetriever/config/coagentic_retriever_tool_config_no_ranker.yaml
```

关键效果：

- `ranker_enabled: false`
- 不启动 LLM judge
- 不启动 async labeling
- 保留 recall retriever 服务和 search tool

最终 smoke 验证的是 no-ranker 训练主链路，不代表 async-labeling + LLM judge 全链路也已完成 NPU 验证。

## 逐层暴露的问题和处理

### 1. faiss 缺失

现象：

```text
ModuleNotFoundError: No module named 'faiss'
```

处理：在 overlay 环境安装 `faiss-cpu==1.13.2`。

### 2. vllm 未安装/版本不匹配

requirements 中写了 `vllm==0.16.0`，但当前可运行组合改为：

```text
vllm==0.13.0
vllm-ascend==0.13.0
```

并对 repo 中 vLLM API 做兼容。

### 3. libatb.so 缺失

现象：

```text
libatb.so: cannot open shared object file
```

处理：安装 Ascend NNAL/ATB，并在兼容层自动 source。

### 4. init_app_state 签名错误

现象：

```text
init_app_state() takes 3 positional arguments but 4 were given
```

处理：用 `inspect.signature` 兼容 3 参数和 4 参数版本。

### 5. FRACTAL_NZ 模式错误

现象：

```text
FRACTAL_NZ mode is enabled...
Please set VLLM_ASCEND_ENABLE_NZ=0.
```

处理：

- 兼容层设置 `VLLM_ASCEND_ENABLE_NZ=0`
- Ray runtime env 保留 `VLLM_ASCEND_*`

### 6. EngineCore None future

现象：

```text
AttributeError: 'NoneType' object has no attribute 'result'
```

处理：临时 patch vLLM site-packages，让 `future is None` 时直接进入 `sample_tokens`。

### 7. HCCL 600 秒 watchdog

现象：

```text
Watchdog caught collective operation timeout:
WorkHCCL(... OpType=ALLGATHER ... Timeout(ms)=600000)
```

处理：

- 设置 `actor_rollout_ref.nccl_timeout=1800`
- 设置 `HCCL_EXEC_TIMEOUT=1800`
- NPU 下默认关闭 `actor.use_torch_compile`

说明：

独立 4 卡 HCCL `all_gather` smoke test 0.4-0.6 秒通过，所以该问题不是基础通信完全不可用，而是训练 workload 首轮长操作超过默认 watchdog。

### 8. wait_for_gpus 加载顺序问题

现象：

```text
co_accel_device_ids: command not found
```

原因：

显式设置 `COSEARCH_ACCELERATOR=npu` 后，旧逻辑没有 source 兼容层。

处理：

```bash
if ! declare -F co_accel_device_ids >/dev/null 2>&1; then
  source compatible_accelerator.sh
fi
```

## 验证命令

### 语法检查

```bash
bash -n \
  src/runtime/wait_for_gpus.sh \
  src/env_manage/compatible_accelerator.sh \
  scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh \
  tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh
```

结果：通过。

### 入口 DRY_RUN

```bash
env \
  PY=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin/python \
  PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin:$PATH \
  COSEARCH_ACCELERATOR=npu \
  RUN_MODE=no-ranker \
  DRY_RUN=1 \
  bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh
```

结果：配置解析和日志路径生成成功。

### 训练 smoke

```bash
timeout 5400s env \
  PY=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin/python \
  PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin:$PATH \
  COSEARCH_ACCELERATOR=npu \
  RUN_MODE=no-ranker \
  TOTAL_STEPS=1 \
  EXP_NAME=codex_npu_hccl_timeout_probe \
  RUN_STAMP=20260625_003200 \
  WAIT_FOR_GPU_TIMEOUT_SECONDS=60 \
  WAIT_FOR_GPU_INTERVAL_SECONDS=5 \
  RECALL_SERVICE_WAIT_SECONDS=900 \
  REPORT_INTERVAL_SECONDS=30 \
  NVIDIA_SMI_INTERVAL=30 \
  bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh
```

日志目录：

```bash
log/train_logs/coAgenticRetriever/20260625_003200-codex_npu_hccl_timeout_probe
```

关键结果：

```text
Recall retrieval semantic preflight passed
Ray runtime env contains VLLM_ASCEND_ENABLE_NZ=0
actor_rollout_ref.nccl_timeout=1800
actor_rollout_ref.actor.use_torch_compile=False
rollout-progress: 512/512
actor-update: micro_batch=32/32
Training Progress: 100%
checkpoint conversion: done
actor HF safetensors validation passed
```

训练 step 指标摘录：

```text
timing_s/gen: 107.70
timing_s/main_agent_old_log_prob: 23.17
timing_s/main_agent_ref_log_prob: 16.84
timing_s/main_agent_update_actor: 44.07
timing_s/step: 210.01
perf/total_num_tokens: 633625
perf/throughput: 754.29
```

这说明当前 no-ranker NPU 训练主链路已经可运行。

## 当前训练模式说明

本轮 smoke 不是 LoRA 训练。配置中：

```text
actor_rollout_ref.model.lora_rank=0
actor_rollout_ref.model.lora_alpha=16
```

`lora_rank=0` 表示 LoRA adapter 关闭，当前是全参 FSDP/GRPO 路径。

NPU 0-3 显存占用在不同阶段会变化：

- 模型尚未完全加载时占用较低。
- rollout/vLLM server 初始化阶段占用会逐步上升。
- actor update 阶段 NPU 0-3 每卡 HBM 曾达到约 `36.5GB/64GB`，训练日志中 `main_agent_perf/max_memory_allocated_gb` 约 `62.3GB`。
- NPU 5 是 recall retriever 服务，占用约 `31.5GB`。

因此低显存截图通常只是阶段性状态，不代表训练没有使用 0-3。

## 当前仍需注意的问题

### Triton import 日志

日志中仍可能出现：

```text
Failed to import Triton kernels. Please make sure your triton version is compatible.
Error: No module named 'triton.language.target_info'
```

当前 smoke 证明它不是阻塞项。后续如果要清理日志，可以单独研究 vLLM/torch/triton 版本匹配。

### vLLM site-packages patch 没有固化到 repo

`EngineCore.step()` 的 None future patch 位于 overlay venv 的 site-packages 中。重建 venv 后会丢失。

建议后续两种处理方式二选一：

1. 升级到 vLLM Ascend 官方已修复版本。
2. 在环境构建脚本中显式 patch 该文件，并记录 patch 校验。

### async labeling / LLM judge 未在本轮完成全链路验证

当前训练 smoke 使用：

```bash
RUN_MODE=no-ranker
```

这会关闭：

```text
ENABLE_ASYNC_LABELING
AUTO_START_LLM_JUDGE
LLM_JUDGE_PREFLIGHT
```

如果要恢复 LLM judge，需要单独验证：

- DeepSeek judge 模型 NPU 加载
- FP4/FP8 量化配置
- vLLM Ascend serving 参数
- judge endpoint preflight
- async labeling buffer 和 sample builder

### `nccl_timeout=1800` 不是性能优化

它只是避免默认 600 秒 watchdog 过早误杀长操作。如果后续稳定后想优化性能，应从以下方向入手：

- 缩短 `MAX_PROMPT_LENGTH`/`MAX_RESPONSE_LENGTH`
- 减少 `N_ROLLOUTS`
- 减小 `TRAIN_BATCH_SIZE`
- 调低 `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`
- 调低 `ACTOR_MICRO_BATCH_SIZE_PER_GPU`
- 考虑重新评估 `ACTOR_USE_TORCH_COMPILE`

## 后续维护建议

1. 将 NPU 专用环境构建步骤单独脚本化，不要复用 H20/CUDA requirements 直接重装。
2. 将 vLLM site-packages patch 固化，否则重建 overlay 后训练会再次遇到 `NoneType.result`。
3. 保留 `COSEARCH_ACCELERATOR=npu` 作为显式开关，避免混合机器误判。
4. 每次改依赖后都跑三类验证：
   - ATB register
   - 4 卡 HCCL all_gather
   - `TOTAL_STEPS=1 RUN_MODE=no-ranker` 训练 smoke
5. 如果要启用 ranker 或 async-labeling，应新增独立 smoke，不要直接复用 no-ranker 结论。

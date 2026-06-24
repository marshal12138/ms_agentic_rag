# NPU 环境依赖补齐记录

本文记录 2026-06-24 到 2026-06-25 在当前机器上为 CoSearch/CoAgenticRetriever 训练补齐 NPU 运行依赖的过程，包括执行过的操作、验证方式、踩过的坑和后续维护建议。

## 目标

入口训练脚本原本主要在 H20/CUDA 服务器上运行：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh
```

当前机器是 Ascend NPU 环境，需要让同一套训练代码在 NPU 上至少能跑通 `RUN_MODE=no-ranker` 的 `TOTAL_STEPS=1` smoke 训练。

本轮依赖补齐遵循两个原则：

1. 尽量不破坏已有 `/data05/conda/envs/ms/ms_agt_rag` 基础环境。
2. 对高风险 Python 依赖优先使用 overlay venv 或明确 pin 版本，避免全量重装 requirements 导致环境崩溃。

## 当前环境基线

关键路径：

```bash
PROJECT_ROOT=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
BASE_ENV=/data05/conda/envs/ms/ms_agt_rag
OVERLAY_ENV=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay
CANN_HOME=/usr/local/Ascend/cann-8.5.2
CANN_LINK=/usr/local/Ascend/cann
ATB_HOME=/usr/local/Ascend/nnal/atb
```

当前 Python 实际使用 overlay venv：

```bash
PY=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin/python
PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin:$PATH
```

关键包版本验证结果：

```text
python: .venvs/ms_agt_rag_overlay/bin/python
torch: 2.8.0+cpu
torch_npu: 2.8.0.post4
vllm: 0.13.0
vllm_ascend: import ok
faiss: 1.13.2
pytrec_eval: 0.5.10
hydra: 1.3.2
```

注意：`torch` 显示 `+cpu` 不代表 NPU 不可用。当前环境依赖 `torch_npu` 提供 NPU 后端，应使用如下方式判断：

```bash
python - <<'PY'
import torch
import torch_npu  # noqa
print("has torch.npu:", hasattr(torch, "npu"))
print("npu available:", torch.npu.is_available())
PY
```

## 安装前做的安全动作

先保存基础环境快照，便于回滚和对比：

```bash
mkdir -p /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/env_snapshots
/data05/conda/envs/ms/ms_agt_rag/bin/python -m pip freeze \
  > /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/env_snapshots/ms_agt_rag_pip_freeze_20260624_213325.txt
```

没有直接在 base env 里全量执行：

```bash
pip install -r env/ms_agt_rag.requirements.txt
```

原因是该 requirements 很大，包含 CUDA 相关包、vLLM、模型服务依赖和大量上层应用依赖。直接重装会高概率改动 torch/torch_npu/vllm 组合，风险太大。

## overlay venv 策略

为了降低风险，实际运行使用了 overlay venv：

```bash
/data05/conda/envs/ms/ms_agt_rag/bin/python -m venv \
  --system-site-packages \
  /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay
```

之后训练和验证通过如下环境进入 overlay：

```bash
export PY=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin/python
export PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin:$PATH
```

这样 overlay 里安装的新包优先于 base env，但仍能读取 base env 已有包。

踩坑点：

- overlay 的好处是降低破坏 base env 的概率。
- overlay 不是完全隔离环境，`--system-site-packages` 会让 base env 包仍可见。
- 如果未来删除 `.venvs/ms_agt_rag_overlay`，需要重新安装 overlay 中补过的包和重新应用本轮对 vLLM site-packages 的临时 patch。

## Python 依赖补齐

训练过程中逐层暴露出缺失包。当前关键补齐项包括：

```bash
faiss-cpu==1.13.2
pytrec-eval-terrier==0.5.10
hydra-core==1.3.2
tensordict==0.10.0
numpy==1.26.4
vllm==0.13.0
vllm-ascend==0.13.0
```

说明：

- `faiss-cpu` 用于加载和校验检索索引。
- `pytrec-eval-terrier` 用于检索/评测相关代码路径。
- `hydra-core` 是训练配置入口必需依赖。
- `tensordict` 是 verl/PPO 训练路径间接依赖。
- `numpy==1.26.4` 是为了兼容当前 FAISS/科学计算栈，避免直接落到 requirements 中较新的 `numpy==2.2.6` 带来 ABI 问题。
- requirements 中有 `vllm==0.16.0`，但当前 NPU 可运行组合采用了 `vllm==0.13.0` 和 `vllm-ascend==0.13.0`。这是有意降级，不是漏装。

当前不建议为了消除 `pip check` 报告而盲目升级 torch/vllm。已知 `pip check` 仍可能报告一些冲突，例如：

- vLLM 与 torch 版本声明不完全一致。
- `flashinfer-python`、`arctic-inference` 等包缺失或版本不满足。

本训练路径已经 smoke 通过，说明这些冲突至少不是当前 no-ranker NPU smoke 的阻塞项。后续如果要升级，应单独建立新环境验证。

## Python 动态库路径问题

部分包在当前环境下需要找到 Python 自身的 `LIBDIR`。为避免脚本手工 source 很多路径，已在：

```bash
src/env_manage/compatible_python.sh
```

中增加逻辑：

```bash
python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
PY
```

解析出 Python libdir 后前置到 `LD_LIBRARY_PATH`。可通过变量关闭：

```bash
export COSEARCH_PREPEND_PYTHON_LIBDIR=0
```

## NNAL/ATB 缺失问题

### 现象

在 vLLM Ascend/torch_npu ATB 路径中遇到：

```text
OSError: libatb.so: cannot open shared object file: No such file or directory
Please check that the nnal package is installed.
Please run 'source set_env.sh' in the NNAL installation path.
```

验证命令：

```bash
ldd /data05/conda/envs/ms/ms_agt_rag/lib/python3.11/site-packages/torch_npu/lib/libop_plugin_atb.so \
  | grep -E 'libatb|not found'
```

当时结果是：

```text
libatb.so => not found
```

即使执行：

```bash
source /usr/local/Ascend/cann-8.5.2/set_env.sh
```

仍然找不到 `libatb.so`。这说明当前机器只有 CANN，缺少 NNAL/ATB 运行库。

### 判断依据

当前系统环境：

```text
CANN: /usr/local/Ascend/cann-8.5.2
torch-npu: 2.8.0.post4
设备: 910B3
vllm-ascend 识别设备类型: A2
torch CXX11 ABI: True
```

因此 ATB 应优先使用 `cxx_abi_1`。

### 下载 NNAL RPM

下载地址：

```text
https://repo.oepkgs.net/ascend/cann/aarch64/Packages/Ascend-cann-nnal-8.5.0-linux.aarch64.rpm
```

本地缓存：

```bash
/data05/cache/ascend_nnal/Ascend-cann-nnal-8.5.0-linux.aarch64.rpm
```

校验：

```bash
sha256sum /data05/cache/ascend_nnal/Ascend-cann-nnal-8.5.0-linux.aarch64.rpm
```

当前结果：

```text
d4592dc25ea9854ea69ed5de0c089c0de9eb3a583f450e87f56ebe8fd89ab6f3
```

### RPM 直接安装踩坑

直接执行 `rpm -ivh` 时 post install 阶段因为没有 `/dev/tty` 失败，包没有正常注册。

因此没有继续强行修 rpm 数据库，而是手工提取 RPM 内部 `.run` 安装器。

### 提取 RPM 并安装 ATB

执行步骤：

```bash
rm -rf /data05/cache/ascend_nnal/extracted
mkdir -p /data05/cache/ascend_nnal/extracted
cd /data05/cache/ascend_nnal/extracted
rpm2cpio ../Ascend-cann-nnal-8.5.0-linux.aarch64.rpm | cpio -idmv

chmod +x /data05/cache/ascend_nnal/extracted/usr/local/Ascend/Ascend-cann-nnal_8.5.0_linux-aarch64.run

/data05/cache/ascend_nnal/extracted/usr/local/Ascend/Ascend-cann-nnal_8.5.0_linux-aarch64.run \
  --install \
  --quiet \
  --install-for-all \
  --install-path=/usr/local/Ascend \
  --whitelist=atb
```

安装完成后出现：

```bash
/usr/local/Ascend/nnal/atb/set_env.sh
/usr/local/Ascend/nnal/atb/8.5.0/atb/set_env.sh
/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib/libatb.so
/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_1/lib/libatb.so
```

### ATB 验证

加载 CANN 和 ATB：

```bash
source /usr/local/Ascend/cann-8.5.2/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
```

验证 `libatb.so` 链接：

```bash
ldd /data05/conda/envs/ms/ms_agt_rag/lib/python3.11/site-packages/torch_npu/lib/libop_plugin_atb.so \
  | grep -E 'libatb|not found'
```

期望结果类似：

```text
libatb.so => /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so
```

Python 层验证：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives

PY=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin/python \
PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin:$PATH \
bash -lc '
source src/env_manage/compatible_python.sh
source /usr/local/Ascend/cann-8.5.2/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
python - <<PY
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
_register_atb_extensions()
print("ATB ok")
PY
'
```

通过后输出：

```text
ATB ok
```

## 将 CANN/ATB 自动加载接入兼容层

为避免每次手工 source，已将 CANN/ATB 自动加载接入：

```bash
src/env_manage/compatible_accelerator.sh
```

NPU 模式下会自动尝试：

```bash
source "${COSEARCH_ASCEND_CANN_SET_ENV:-/usr/local/Ascend/cann/set_env.sh}"
source "${COSEARCH_ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}" \
  "--cxx_abi=${COSEARCH_ASCEND_ATB_CXX_ABI:-1}"
```

可覆盖变量：

```bash
export COSEARCH_ASCEND_CANN_SET_ENV=/usr/local/Ascend/cann-8.5.2/set_env.sh
export COSEARCH_ASCEND_ATB_SET_ENV=/usr/local/Ascend/nnal/atb/set_env.sh
export COSEARCH_ASCEND_ATB_CXX_ABI=1
```

踩坑点：ATB 的 `set_env.sh` 在 `set -u` shell 下会引用未定义变量，例如 `ZSH_VERSION`。兼容层通过临时关闭 nounset 解决：

```bash
case "$-" in
  *u*) had_nounset=1; set +u ;;
esac
source "${path}" "$@"
if [[ "${had_nounset}" == "1" ]]; then
  set -u
fi
```

## vLLM Ascend 相关环境变量

vLLM Ascend 在 RL 路径中遇到过：

```text
ValueError: FRACTAL_NZ mode is enabled...
Please set VLLM_ASCEND_ENABLE_NZ=0.
```

因此 NPU 兼容层默认设置：

```bash
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
```

同时，为 HCCL 设置较稳妥的默认值：

```bash
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
```

注意：这些是 watchdog/通信连接上限，不代表训练会固定等待这么久。

## 不要做的事

### 不建议直接 pip install atb

不要用：

```bash
pip install atb
```

原因是 PyPI 上的 `atb` 名称无法确认是华为 Ascend NNAL/ATB，容易装错包。正确路径是安装与 CANN/torch_npu 匹配的 Ascend NNAL/ATB 系统运行库。

### 不建议盲目升级 torch/torch_npu/vllm

当前能跑通的组合是：

```text
torch 2.8.0
torch_npu 2.8.0.post4
vllm 0.13.0
vllm-ascend 0.13.0
CANN 8.5.2
NNAL/ATB 8.5.0
```

requirements 中的 `vllm==0.16.0` 不等于当前可运行组合。升级到 0.16.0 需要重新验证 vLLM Ascend、torch_npu、CANN、ATB 之间的匹配关系。

## 独立 HCCL 验证

为了确认机器 0-3 号 NPU 基础通信没有问题，跑过独立 4 卡 HCCL `all_gather` smoke test。核心逻辑：

```bash
timeout 240s env \
  PY=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin/python \
  PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/.venvs/ms_agt_rag_overlay/bin:$PATH \
  COSEARCH_ACCELERATOR=npu \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  HCCL_CONNECT_TIMEOUT=1500 \
  HCCL_EXEC_TIMEOUT=1800 \
  HCCL_HOST_SOCKET_PORT_RANGE=60000-60050 \
  HCCL_NPU_SOCKET_PORT_RANGE=61000-61050 \
  bash -lc '
source src/env_manage/compatible_python.sh
source src/env_manage/compatible_accelerator.sh
python - <<PY
import datetime
import os
import socket
import time
from multiprocessing import get_context

def worker(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    import torch
    import torch_npu
    import torch.distributed as dist
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=120))
    x = torch.full((1024,), rank, device=f"npu:{rank}", dtype=torch.float32)
    out = [torch.empty_like(x) for _ in range(world_size)]
    start = time.time()
    dist.all_gather(out, x)
    torch.npu.synchronize()
    print(rank, [float(t[0].cpu()) for t in out], time.time() - start, flush=True)
    dist.barrier()
    dist.destroy_process_group()

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    port = free_port()
    ctx = get_context("fork")
    procs = [ctx.Process(target=worker, args=(r, 4, "127.0.0.1", port)) for r in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print("exit_codes", codes)
    raise SystemExit(0 if all(c == 0 for c in codes) else 1)
PY
'
```

结果：4 个 rank 都成功拿到 `[0.0, 1.0, 2.0, 3.0]`，耗时约 `0.4s - 0.6s`。这说明机器基础 HCCL 通信栈可用，训练中遇到的 600 秒 timeout 不是基础 HCCL 完全不可用导致的。

## 训练 smoke 验证结果

最终训练 smoke 入口：

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

结果：

```text
rollout: 512/512 completed
actor update: 32/32 micro batches completed
Training Progress: 100%
checkpoint conversion: done
actor HF safetensors validation passed
```

日志目录：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/train_logs/coAgenticRetriever/20260625_003200-codex_npu_hccl_timeout_probe
```

## 常见问题和处理建议

### libatb.so not found

优先检查：

```bash
find /usr/local/Ascend -name 'libatb.so*'
ldd /data05/conda/envs/ms/ms_agt_rag/lib/python3.11/site-packages/torch_npu/lib/libop_plugin_atb.so \
  | grep -E 'libatb|not found'
```

如果仍是 not found，先确认 NNAL/ATB 是否安装，不要先动 Python 依赖。

### source ATB set_env.sh 后仍失败

检查是否用了正确 ABI：

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
```

当前 torch CXX11 ABI 为 True，对应 `cxx_abi_1`。

### vllm/vllm-ascend 版本冲突

当前采取的是能跑通优先，而不是严格满足 requirements 中每个版本。不要仅因为 requirements 写了 `vllm==0.16.0` 就升级当前环境。

### Triton kernel import warning

训练日志中仍可能出现：

```text
Failed to import Triton kernels. Error: No module named 'triton.language.target_info'
```

本轮 smoke 中该日志没有导致训练失败。它目前是可选 Triton kernel 路径的 warning/error 日志，不是当前阻塞点。

## 后续维护建议

1. 如果要长期使用这台机器，建议把 overlay venv 的关键包列表单独固化成一个 NPU 专用 requirements，而不是复用 H20/CUDA 的完整 requirements。
2. 如果重建 overlay venv，需要重新验证：
   - `torch_npu` import
   - ATB extension register
   - vLLM Ascend import
   - 4 卡 HCCL all_gather
   - `TOTAL_STEPS=1` smoke
3. 如果升级 CANN、torch_npu、vllm-ascend 任一组件，应把 NNAL/ATB 版本一起重新匹配。
4. 如果以后要恢复 async labeling 或 LLM judge NPU 服务，需要单独验证 DeepSeek judge 模型的量化和 vLLM Ascend 加载路径，不要把训练 no-ranker smoke 通过等同于 judge 服务完全可用。

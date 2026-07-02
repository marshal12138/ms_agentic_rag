#!/usr/bin/env python3
"""Run canonical CoAgenticRetriever training from compiled launcher files.

这个脚本是 `scripts/coagenticRetriever_v2/01_train_launcher.sh` 的训练执行 helper。
它替代 v2 launcher 原来对 `assets/00_run_agentic_iter_rag_verl.sh` 的 canonical 分支调用。

边界非常明确：

- 它只支持 canonical 配置模式。
- 它只读取 `compile_config.py` 已经生成好的 `CANONICAL_HYDRA_ARGS_FILE`。
- 它只设置训练进程真正需要的 runtime 环境变量。
- 它最后用 `os.execvpe` 直接执行 `main_coagentic_retriever.py`。

它不负责：

- 不解析 task CLI。
- 不合并 main_run/resource/overlay。
- 不等待 GPU。
- 不启动 recall retriever 或 LLM judge。
- 不启动日志 reporter / nvidia-smi sampler。
- 不做 checkpoint conversion。
- 不保留 legacy env-to-Hydra 拼接逻辑。

因此 v2 主链路变为：

`01_train_launcher.sh -> compile_config.py -> run_canonical_training.py -> main_coagentic_retriever.py`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import MutableMapping, Sequence


def _truthy_one(value: str | None) -> bool:
    """只兼容旧 shell 分支中的 `== 1` 判断。

    `ALLOW_VLLM_EXPANDABLE_SEGMENTS` 旧逻辑只接受 `1`，不是通用 truthy。这里保持行为
    不变，避免迁移后环境变量解释发生变化。
    """

    return value == "1"


def _is_npu(accelerator: str) -> bool:
    """判断当前 launcher 编译出的 accelerator 是否为 Ascend/NPU。"""

    return accelerator in {"npu", "ascend"}


def _prepend_path(env: MutableMapping[str, str], name: str, entries: Sequence[Path]) -> None:
    """向 PATH/PYTHONPATH 类变量前置目录，并避免重复插入。

    旧 shell runner 会把 `PROJECT_ROOT` 和 `PROJECT_ROOT/verl` 放到 `PYTHONPATH` 前面。
    Python runner 继续保持这个顺序，确保项目代码优先于环境中可能存在的旧安装包。
    """

    existing = [part for part in env.get(name, "").split(":") if part]
    result: list[str] = []
    for entry in entries:
        text = str(entry)
        if text and text not in result:
            result.append(text)
    for part in existing:
        if part not in result:
            result.append(part)
    env[name] = ":".join(result)


def _set_default(env: MutableMapping[str, str], name: str, value: str) -> None:
    """按 shell `${VAR:-default}` 语义设置默认值：变量不存在或为空都使用默认值。"""

    if not env.get(name):
        env[name] = value


def _require_path(path: Path, label: str) -> None:
    """训练启动前检查关键路径存在，错误信息直接指向缺失项。"""

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _read_hydra_args(path: Path) -> list[str]:
    """读取 `hydra_args.txt`。

    `compile_config.py` 已经把最终 Hydra 参数一行一条写好。这里不重新理解 overlay，
    只把非空行转成 argv，保证训练进程看到的参数和审计文件一致。
    """

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _materialize_visible_devices(env: MutableMapping[str, str]) -> None:
    """按 accelerator 类型设置训练主进程可见设备。

    这对应旧 shell runner 中的 `co_accel_export_visible_devices "${GPU_IDS}"`：

    - GPU/CUDA：设置 `CUDA_VISIBLE_DEVICES`。
    - NPU/Ascend：设置 `ASCEND_RT_VISIBLE_DEVICES`，并在 CUDA_VISIBLE_DEVICES 未显式设置时
      同步写入同一组 ids，兼容仍读取 CUDA 变量的下游组件。
    """

    gpu_ids = env.get("GPU_IDS", "")
    if not gpu_ids:
        return
    if _is_npu(env.get("COSEARCH_ACCELERATOR", "")):
        env["ASCEND_RT_VISIBLE_DEVICES"] = gpu_ids
        env.setdefault("CUDA_VISIBLE_DEVICES", gpu_ids)
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids


def _prepare_training_env(env: MutableMapping[str, str]) -> tuple[Path, Path, Path, list[str]]:
    """校验 canonical 输入，并物化训练进程环境。

    返回值分别是：

    - project root
    - CoAgenticRetriever main 文件
    - hydra args 文件
    - hydra argv 列表
    """

    if env.get("CANONICAL_CONFIG_MODE") != "1":
        raise RuntimeError("01_train_launcher.sh only supports canonical config mode")

    project_root_value = env.get("PROJECT_ROOT") or env.get("COAGENTIC_PROJECT_ROOT")
    if not project_root_value:
        raise RuntimeError("PROJECT_ROOT is required for canonical training")
    project_root = Path(project_root_value).resolve()
    coagentic_main = Path(env.get("COAGENTIC_MAIN") or project_root / "main_coagentic_retriever.py").resolve()
    hydra_args_file_value = env.get("CANONICAL_HYDRA_ARGS_FILE", "")
    if not hydra_args_file_value:
        raise RuntimeError("CANONICAL_HYDRA_ARGS_FILE is required for canonical training")
    hydra_args_file = Path(hydra_args_file_value).resolve()

    _require_path(project_root, "PROJECT_ROOT")
    _require_path(project_root / "verl", "VERL root")
    _require_path(coagentic_main, "COAGENTIC_MAIN")
    _require_path(hydra_args_file, "CANONICAL_HYDRA_ARGS_FILE")

    _prepend_path(env, "PYTHONPATH", [project_root, project_root / "verl"])

    # vLLM CuMem memory pool 和 expandable_segments:True 冲突。旧 shell runner 会在未显式
    # 允许时 unset 这个变量，这里保持同样行为。
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in alloc_conf and not _truthy_one(env.get("ALLOW_VLLM_EXPANDABLE_SEGMENTS")):
        print(
            "WARNING: ignoring PYTORCH_CUDA_ALLOC_CONF="
            f"{alloc_conf} because vLLM CuMem memory pool rejects expandable_segments:True",
            file=sys.stderr,
        )
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    _materialize_visible_devices(env)

    env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    if _is_npu(env.get("COSEARCH_ACCELERATOR", "")):
        env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"

    _set_default(env, "TOKENIZERS_PARALLELISM", "false")
    env["VLLM_DISABLE_FLASHINFER"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    _set_default(env, "VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    _set_default(env, "ATTN_IMPLEMENTATION", "flash_attention_2")
    _set_default(env, "GLOO_SOCKET_IFNAME", "lo")
    _set_default(env, "NCCL_SOCKET_IFNAME", "lo")
    env["WANDB_MODE"] = "disabled"

    return project_root, coagentic_main, hydra_args_file, _read_hydra_args(hydra_args_file)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。

    本脚本不接受额外 CLI 参数。所有训练参数都必须已经由 `compile_config.py` 写入
    `hydra_args.txt`，这样训练入口和审计文件保持同源。
    """

    args = list(argv if argv is not None else sys.argv[1:])
    if args:
        print("ERROR: run_canonical_training.py does not accept CLI passthrough: " + " ".join(args), file=sys.stderr)
        return 2

    env = dict(os.environ)
    try:
        project_root, coagentic_main, hydra_args_file, hydra_args = _prepare_training_env(env)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    py = env.get("PY") or sys.executable
    env["COAGENTIC_PROJECT_ROOT"] = str(project_root)
    print(f"Launching canonical CoAgenticRetriever training: {coagentic_main}", flush=True)
    print(f"Hydra args file: {hydra_args_file}", flush=True)
    os.chdir(project_root)
    os.execvpe(py, [py, str(coagentic_main), *hydra_args], env)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

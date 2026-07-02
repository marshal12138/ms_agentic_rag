"""Shared data structures for the launcher config compiler.

这些 dataclass 只表达编译过程中传递的数据，不做文件 IO、不启动子进程、
也不读取 YAML。把它们独立出来可以避免 `compile_config.py` 和各个 helper
模块之间互相定义类型，降低循环 import 的风险。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .cli import LauncherSelection
from .hydra_args import CanonicalHydraSelection
from .main_run_config import MainRunManifest


@dataclass(frozen=True)
class CompilerContext:
    """Bash 传给 Python compiler 的不可变进程上下文。

    这类信息不是实验配置，也不是 runtime overlay，而是 launcher 所在机器/仓库的
    执行上下文。Bash 仍负责 source 仓库 helper、探测 accelerator 和设备变量名；
    Python compiler 只接收已经解析好的结果，因此这里不会再 source shell 代码。
    """

    repo_root: Path
    script_dir: Path
    assets_dir: Path
    project_root: Path
    external_model_root: Path
    external_retrieval_root: Path
    device_prefix: str
    visible_devices_var: str
    accelerator: str

    def device_spec(self, index: str | None = None) -> str:
        """返回训练配置中使用的设备表达式，例如 `cuda:4` 或仅 `cuda`。"""

        return f"{self.device_prefix}:{index}" if index else self.device_prefix


@dataclass
class RunFiles:
    """一次 launcher 调用会生成的全部文件路径。

    这些文件都放在当前 run 的 `LOG_DIR` 下。集中建模的目的是避免不同模块各自拼接
    文件名，导致 `.env`、Bash source 文件和实际写入位置不一致。
    """

    runtime_env_sh: Path
    env_file: Path
    runtime_override_yaml: Path
    run_mode_override_yaml: Path
    runtime_tool_config_yaml: Path
    hydra_args_file: Path
    final_config_yaml: Path
    final_config_json: Path
    trainer_main_hydra_config_file: Path
    hydra_groups_file: Path
    hydra_cli_overrides_file: Path
    overlay_yamls_file: Path
    legacy_cli_args_file: Path


@dataclass
class CompiledConfig:
    """配置编译过程中在内存中传递的完整结果。

    注意这里保存的是“已经合并但尚未全部落盘”的状态。主流程会先完成校验，再把
    这些内容写成审计文件和 Bash source 文件。
    """

    env: dict[str, str] = field(default_factory=dict)
    canonical: bool = False
    selection: LauncherSelection = field(default_factory=LauncherSelection)
    manifest: MainRunManifest = field(default_factory=MainRunManifest)
    hydra_selection: CanonicalHydraSelection | None = None
    files: RunFiles | None = None
    hydra_args: list[str] = field(default_factory=list)

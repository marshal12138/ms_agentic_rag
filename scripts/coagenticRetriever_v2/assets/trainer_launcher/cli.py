"""Launcher CLI parsing for config compilation.

Bash launcher 过去一边解析 CLI、一边合并 YAML、一边写 Hydra 参数，导致优先级很难
检查。这个模块只做第一步：把原始 launcher 参数拆成“launcher 自己理解的选择项”
和“要透传给训练 Hydra 的临时 override”。它不读取 YAML，也不做默认值决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .paths import resolve_repo_path


@dataclass
class LauncherSelection:
    """launcher 参数解析后的中间选择结果。

    这里的字段还没有应用 main_run_config 默认值，也没有检查文件是否都存在。它只是
    忠实记录 task 脚本或用户在命令行中传了什么。
    """

    main_run_config: str = ""
    trainer_main_hydra_config: str = ""
    data_config: str = ""
    model_config: str = ""
    rollout_config: str = ""
    ranker_base_config: str = ""
    async_ranker_training_base_config: str = ""
    run_mode: str = ""
    resource_config: str = ""
    overlay_yamls: list[Path] = field(default_factory=list)
    llm_judge_service_config: Path | None = None
    trainer_cli_overrides: list[str] = field(default_factory=list)

    def has_canonical_signal(self) -> bool:
        """判断本次调用是否进入 canonical 配置模式。

        只要出现 main_run_config、Hydra group、resource 或 overlay 之一，就说明调用
        方希望由 Python compiler 生成 `hydra_args.txt`，而不是走 legacy passthrough。
        """

        return any(
            [
                self.main_run_config,
                self.trainer_main_hydra_config,
                self.data_config,
                self.model_config,
                self.rollout_config,
                self.ranker_base_config,
                self.async_ranker_training_base_config,
                self.run_mode,
                self.resource_config,
                self.overlay_yamls,
            ]
        )


def _require_value(option: str, value: str | None) -> str:
    """统一处理 `--key=` 或缺参场景，避免空字符串被当成有效配置。"""

    if value is None or value == "":
        raise ValueError(f"{option} requires a non-empty value")
    return value


def _consume_value(argv: Sequence[str], index: int, option: str, inline_value: str | None) -> tuple[str, int]:
    """读取 `--foo value` 或 `--foo=value` 两种 launcher 参数写法。"""

    if inline_value is not None:
        return _require_value(option, inline_value), index + 1
    if index + 1 >= len(argv):
        raise ValueError(f"{option} requires a value")
    return _require_value(option, argv[index + 1]), index + 2


def normalize_cli_override(raw: str) -> str:
    """把 task 末尾的临时 override 规整成 Hydra dotlist 语法。

    支持 `key=value`、`--key=value` 和 `~key`。不支持裸参数，因为裸参数无法明确
    表达是 launcher 选项还是 Hydra override。
    """

    value = _require_value("canonical CLI override", raw)
    if value.startswith("--"):
        value = value[2:]
    if value.startswith("--"):
        raise ValueError(f"unsupported canonical CLI override: {raw}; use key=value or --key=value")
    if "=" in value or value.startswith("~"):
        return value
    raise ValueError(f"unsupported canonical CLI override: {raw}; use key=value or --key=value")


def append_unique_cli_override(values: list[str], raw: str) -> None:
    """追加一个去重后的 Hydra override，保留首次出现的顺序。"""

    normalized = normalize_cli_override(raw)
    if normalized not in values:
        values.append(normalized)


def parse_launcher_args(argv: Sequence[str], *, repo_root: Path) -> LauncherSelection:
    """解析 launcher 参数，并把未知参数收集为训练 Hydra override。

    这里故意不读取 YAML，也不填默认值。原因是 main_run_config、resource、overlay 的
    优先级需要在主编译流程中统一处理；CLI 解析阶段只负责分流。
    """

    selection = LauncherSelection()
    i = 0
    passthrough: list[str] = []
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            passthrough.extend(argv[i + 1 :])
            break

        option = arg
        inline_value: str | None = None
        if arg.startswith("--") and "=" in arg:
            option, inline_value = arg.split("=", 1)

        if option in {"--HYDRA_MAIN_CONFIG", "--TRAINER_MAIN_HYDRA_CONFIG"}:
            raise ValueError(
                "--HYDRA_MAIN_CONFIG has been renamed. "
                "Use --trainer_main_hydra_config or --main_run_config."
            )
        if option == "--main_run_config":
            selection.main_run_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--trainer_main_hydra_config":
            selection.trainer_main_hydra_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--DATA_CONFIG":
            selection.data_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--MODEL_CONFIG":
            selection.model_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--ROLLOUT_CONFIG":
            selection.rollout_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--RANKER_BASE_CONFIG":
            selection.ranker_base_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--ASYNC_RANKER_TRAINING_BASE_CONFIG":
            selection.async_ranker_training_base_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--RESOURCE_CONFIG":
            selection.resource_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--OVERLAY_YAML":
            value, i = _consume_value(argv, i, option, inline_value)
            selection.overlay_yamls.append(resolve_repo_path(repo_root, value))
        elif option == "--LLM_JUDGE_SERVICE_CONFIG":
            value, i = _consume_value(argv, i, option, inline_value)
            selection.llm_judge_service_config = resolve_repo_path(repo_root, value)
        else:
            passthrough.append(arg)
            i += 1

    # Unknown CLI args are intentionally kept raw here. In canonical mode the
    # compiler will validate them as Hydra overrides; in legacy mode Bash still
    # passes the original argv to the old asset runner.
    selection.trainer_cli_overrides.extend(passthrough)
    return selection

"""Canonical run-mode resolution for the v2 train launcher.

`run_mode` 是 launcher 级训练形态参数，不是 CoAgenticRetriever trainer 自己消费的
Hydra 字段。它可以出现在：

- `CoAgenticRetriever/config/main_run/coAgenticRetriever_main.yaml`
- 普通 overlay YAML，例如 `tasks/.../train_args_overlay.yaml`
- task 末尾显式 CLI override，例如 `run_mode=no-ranker`

compiler 会先解析最终 run mode，再把它转换成真正的 Hydra override：

- `full`：保留 ranker / async ranker training overlay 的正常训练语义。
- `no-ranker`：关闭 ranker 训练、关闭共享 ranker 推理副本、关闭 async ranker training。

这样使用者可以用一个高层 `run_mode` 表达训练形态，但训练进程最终只看到具体的
Hydra 参数，不需要理解 launcher-only 的 `run_mode` 字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .yaml_utils import load_mapping

RUN_MODE_KEYS = {"run_mode", "RUN_MODE", "train_mode", "TRAIN_MODE"}


@dataclass(frozen=True)
class RunModeResolution:
    """最终 run mode 解析结果。

    `trainer_cli_overrides` 是移除了 launcher-only run_mode override 后的剩余 Hydra CLI
    参数，后续可以安全写入 `hydra_args.txt`。
    """

    run_mode: str
    effective_run_mode: str
    source: str
    trainer_cli_overrides: list[str] = field(default_factory=list)


def normalize_run_mode_value(value: object | None) -> str:
    """把用户/YAML 输入规整成 canonical run mode 名称。"""

    text = str(value or "full").strip().strip("\"'")
    if text in {"full", "co-training", "co_training"}:
        return "full"
    if text in {"no-ranker", "no_ranker"}:
        return "no-ranker"
    raise ValueError(f"unsupported run_mode={text}; use full or no-ranker")


def _scalar_run_mode(data: dict[str, Any], *, source: Path) -> str | None:
    """从一个 YAML mapping 中读取 launcher-only run_mode 字段。"""

    found: list[tuple[str, Any]] = [(key, data[key]) for key in RUN_MODE_KEYS if key in data]
    if not found:
        return None
    if len(found) > 1:
        keys = ", ".join(key for key, _ in found)
        raise ValueError(f"{source} defines multiple run mode keys: {keys}")
    key, value = found[0]
    if isinstance(value, (dict, list)):
        raise TypeError(f"{key} in {source} must be a scalar")
    return normalize_run_mode_value(value)


def _split_run_mode_cli_override(raw: str) -> tuple[str | None, str | None]:
    """识别 task 末尾 CLI override 中的 run_mode 参数。

    支持 `run_mode=no-ranker`、`++run_mode=no-ranker` 和 `--run_mode=no-ranker` 经过
    `cli.py` 归一后的形式。返回 `(run_mode, original_raw)`；不是 run_mode 时返回
    `(None, None)`。
    """

    value = raw[2:] if raw.startswith("--") else raw
    if "=" not in value:
        return None, None
    key, rhs = value.split("=", 1)
    key = key.lstrip("+")
    if key not in RUN_MODE_KEYS:
        return None, None
    return normalize_run_mode_value(rhs), raw


def resolve_run_mode(
    *,
    main_run_mode: str,
    overlay_yamls: list[Path],
    trainer_cli_overrides: list[str],
) -> RunModeResolution:
    """按 main_run < overlay < CLI 的优先级解析最终 run mode。"""

    run_mode = normalize_run_mode_value(main_run_mode or "full")
    source = "main_run_config" if main_run_mode else "default"

    for path in overlay_yamls:
        data = load_mapping(path, label="overlay YAML")
        overlay_run_mode = _scalar_run_mode(data, source=path)
        if overlay_run_mode is not None:
            run_mode = overlay_run_mode
            source = str(path)

    filtered_cli: list[str] = []
    for raw in trainer_cli_overrides:
        cli_run_mode, _ = _split_run_mode_cli_override(raw)
        if cli_run_mode is None:
            filtered_cli.append(raw)
            continue
        run_mode = cli_run_mode
        source = "trainer_cli_override"

    return RunModeResolution(
        run_mode=run_mode,
        effective_run_mode=run_mode,
        source=source,
        trainer_cli_overrides=filtered_cli,
    )


def build_run_mode_hydra_overrides(run_mode: str) -> dict[str, Any]:
    """把 launcher run mode 转换成真正进入 Hydra 的训练语义 override。"""

    mode = normalize_run_mode_value(run_mode)
    if mode == "full":
        # full 模式沿用 ranker_base 和 async-ranker overlay 已经表达的训练语义。
        return {}
    if mode == "no-ranker":
        return {
            "trainer": {
                "ranker_trainable": False,
                "ranker_update_mode": "disabled",
                "disable_reranker_rollout": True,
            },
            "ranker_training": {
                "signal_source": "pseudo_rank",
                "shared_inference_ranker": {
                    "enable": False,
                },
                "async_ranker_training": {
                    "enable": False,
                },
            },
        }
    raise AssertionError(f"unreachable run_mode={mode}")

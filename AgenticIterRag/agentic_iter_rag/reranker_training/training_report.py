"""AIR LLM reranker training report and curve generation.

这个模块做两件事：

1. 把 VERL 控制台日志里的 ``step:1 - metric:value`` 行转成公共 report_system 的
   ``metrics.jsonl`` 格式。
2. 调用公共 ``src/logs/report_system`` 里的报告和画图脚本，生成 AIR LLM reranker 的
   latest markdown 报告与曲线图。

注意：这里不调用 CAR 项目目录下的任何 schema 或 launcher，只复用公共 src 实现。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPORT_PREFIX = "air_llm_reranker"
PLOT_GROUPS = [
    "reranker_rewards",
    "reranker_losses",
    "reranker_lengths",
    "reranker_performance",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_project_root(repo_root: Path) -> Path:
    return repo_root / "AgenticIterRag"


def default_schema_path(repo_root: Path) -> Path:
    return repo_root / "scripts" / "agenticIterRag_v1" / "assets" / "report_schema.py"


def default_report_system_dir(repo_root: Path) -> Path:
    return repo_root / "src" / "logs" / "report_system"


def resolve_compatible_python(repo_root: Path) -> str:
    """解析和 CAR launcher 一致的兼容 Python。

    公共 report_system 依赖 Python 3.10+ 语法和 matplotlib；系统 /usr/bin/python 可能不满足。
    因此 AIR reporter 也通过 ``compatible_python.sh`` 解析 ``$PY``。
    """

    explicit = os.environ.get("PY")
    if explicit:
        return explicit
    compat = repo_root / "src" / "env_manage" / "compatible_python.sh"
    cmd = f"source {str(compat)!r}; printf '%s\\n' \"$PY\""
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode != 0:
        return sys.executable
    resolved = result.stdout.strip()
    return resolved or sys.executable


def resolve_compatible_python_env(repo_root: Path) -> tuple[str, dict[str, str]]:
    """解析公共 report_system 子进程需要的 Python 和环境变量。

    只拿 ``$PY`` 不够：当前仓库的 matplotlib 等报告依赖可能来自 repo-local overlay，
    overlay 是 ``compatible_python.sh`` 通过 PYTHONPATH/PATH 注入的。因此这里显式 source
    该脚本，并把它导出的关键环境变量合并到子进程环境里。
    """

    env = os.environ.copy()
    compat = repo_root / "src" / "env_manage" / "compatible_python.sh"
    if not compat.exists():
        return resolve_compatible_python(repo_root), env

    # 这里不用 ``$PY -c`` 再反查环境，避免 PYTHONPATH 被当前 Python 启动方式污染；
    # 直接读取 bash source 后的 env 更接近 shell launcher 的真实行为。
    cmd = f"source {str(compat)!r}; env -0"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=str(repo_root),
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode != 0:
        return resolve_compatible_python(repo_root), env
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        raw_key, raw_value = item.split(b"=", 1)
        key = raw_key.decode("utf-8", errors="replace")
        value = raw_value.decode("utf-8", errors="replace")
        if value:
            env[key] = value
    return env.get("PY") or resolve_compatible_python(repo_root), env


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def build_pythonpath(repo_root: Path, existing: str | None = None) -> str:
    """构造公共 report_system 可 import 的 PYTHONPATH。

    公共脚本内部用 ``from report_io import ...`` 这种同目录 import，所以必须把
    ``src/logs/report_system`` 显式放到 PYTHONPATH 里。
    """

    parts = [
        str(default_report_system_dir(repo_root)),
        str(default_project_root(repo_root)),
        str(repo_root),
    ]
    if existing is None:
        existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def read_console_metrics(repo_root: Path, verl_log: Path, step_limit: int | None = None) -> list[dict[str, Any]]:
    """复用公共 report_io 解析 VERL 控制台日志。

    这里用延迟 import，是为了保证调用方可以先设置 PYTHONPATH；同时也避免把公共 src 代码复制到 AIR
    包里。
    """

    report_system_dir = default_report_system_dir(repo_root)
    if str(report_system_dir) not in sys.path:
        sys.path.insert(0, str(report_system_dir))
    from report_io import read_console_metrics as _read_console_metrics  # type: ignore

    return _read_console_metrics(verl_log, step_limit)


def max_step(rows: list[dict[str, Any]]) -> int:
    steps = [row.get("step") for row in rows if isinstance(row.get("step"), int)]
    return max(steps) if steps else 0


@dataclass(frozen=True)
class ReportPaths:
    out_dir: Path
    metrics_jsonl: Path
    env_file: Path
    train_log_copy: Path
    training_report: Path
    detailed_training_report: Path
    plot_prefix: Path
    report_manifest: Path

    @property
    def curve_paths(self) -> dict[str, Path]:
        return {
            group: self.plot_prefix.with_name(f"{self.plot_prefix.name}_{group}.png")
            for group in PLOT_GROUPS
        }


def build_report_paths(out_dir: Path) -> ReportPaths:
    out_dir = out_dir.resolve()
    return ReportPaths(
        out_dir=out_dir,
        metrics_jsonl=out_dir / f"{REPORT_PREFIX}.metrics.jsonl",
        env_file=out_dir / f"{REPORT_PREFIX}.env",
        train_log_copy=out_dir / f"{REPORT_PREFIX}.train.log",
        training_report=out_dir / f"{REPORT_PREFIX}.training_metrics_report.latest.md",
        detailed_training_report=out_dir / f"{REPORT_PREFIX}.detailed_metrics_report.latest.md",
        plot_prefix=out_dir / f"{REPORT_PREFIX}.metrics.latest",
        report_manifest=out_dir / f"{REPORT_PREFIX}.report_manifest.json",
    )


def load_config_yaml(path: Path | None) -> dict[str, Any] | None:
    """读取 final config，用于给报告补运行环境字段。

    reporter 的核心功能不依赖配置；配置只用于 markdown 里的审计信息。读取失败时由调用方降级处理。
    """

    if path is None:
        return None
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def write_report_env(
    paths: ReportPaths,
    *,
    config: dict[str, Any] | None = None,
    extra_env: dict[str, Any] | None = None,
) -> None:
    """写公共 report markdown 会读取的 companion env 文件。

    这些字段主要用于报告说明和后续审计，不参与训练逻辑。
    """

    def active_phase_services(cfg: dict[str, Any]) -> dict[str, Any]:
        """读取当前 phase 的资源服务配置；优先使用新的 phase_services 层级。"""

        rt_cfg = cfg.get("reranker_training", {})
        phase_name = str(rt_cfg.get("_active_phase_name") or "stage1_format")
        train_resource = cfg.get("resource", {}).get("stage_resources", {}).get("train_llm_reranker", {})
        phase_services = train_resource.get("phase_services")
        if isinstance(phase_services, dict) and phase_name in phase_services:
            services = phase_services.get(phase_name, {}).get("services", {})
            return services if isinstance(services, dict) else {}
        services = train_resource.get("services", {})
        return services if isinstance(services, dict) else {}

    env: dict[str, Any] = {}
    if config:
        rt_cfg = config.get("reranker_training", {})
        trainer_cfg = rt_cfg.get("trainer", {})
        branch_cfg = rt_cfg.get("branch_dataset", {})
        services = active_phase_services(config)
        env.update(
            {
                "RERANKER_ROLLOUT_N": trainer_cfg.get("n_samples_per_prompt", ""),
                "RERANKER_BRANCH_STEP_POLICY": branch_cfg.get("step_policy", ""),
                "RERANKER_GPU_IDS": ",".join(str(x) for x in services.get("reranker_actor", {}).get("gpu_ids", [])),
                "FROZEN_AGENT_GPU_IDS": ",".join(str(x) for x in services.get("frozen_agent_vllm", {}).get("gpu_ids", [])),
                "RETRIEVER_GPU_IDS": ",".join(str(x) for x in services.get("recall", {}).get("gpu_ids", [])),
            }
        )
    if extra_env:
        env.update(extra_env)
    paths.env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in sorted(env.items())]
    paths.env_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_train_log_for_report(verl_log: Path, paths: ReportPaths) -> None:
    """复制一份 companion train.log，方便公共报告链接到稳定文件名。

    训练中日志可能正在写入；读取失败时直接跳过，下一轮 reporter 会再次尝试。
    """

    if not verl_log.exists():
        return
    try:
        text = verl_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    paths.train_log_copy.parent.mkdir(parents=True, exist_ok=True)
    paths.train_log_copy.write_text(text, encoding="utf-8")


def run_public_report_script(
    repo_root: Path,
    script_name: str,
    args: list[str],
    *,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """调用公共 report_system 脚本并捕获失败原因。"""

    script = default_report_system_dir(repo_root) / script_name
    python_bin, env = resolve_compatible_python_env(repo_root)
    # 先保留 compatible_python.sh 注入的 overlay，再把公共 report_system 放到最前面。
    env["PYTHONPATH"] = build_pythonpath(repo_root, existing=env.get("PYTHONPATH"))
    started = time.time()
    result = subprocess.run(
        [python_bin, str(script), *args],
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    return {
        "script": str(script),
        "return_code": result.returncode,
        "elapsed_s": time.time() - started,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def generate_markdown_reports(
    *,
    repo_root: Path,
    schema_path: Path,
    paths: ReportPaths,
    step_limit: int | None,
) -> list[dict[str, Any]]:
    common_args = [
        "--metrics-jsonl",
        str(paths.metrics_jsonl),
        "--train-log",
        str(paths.train_log_copy),
        "--env-file",
        str(paths.env_file),
        "--report-schema",
        str(schema_path),
    ]
    if step_limit is not None:
        common_args.extend(["--step-limit", str(step_limit)])
    return [
        run_public_report_script(
            repo_root,
            "train_metrics_report.py",
            [*common_args, "--out", str(paths.training_report)],
        ),
        run_public_report_script(
            repo_root,
            "train_metrics_report.py",
            [*common_args, "--detailed", "--out", str(paths.detailed_training_report)],
        ),
    ]


def generate_plots(
    *,
    repo_root: Path,
    schema_path: Path,
    paths: ReportPaths,
    step_limit: int | None,
) -> dict[str, Any]:
    args = [
        "--metrics-jsonl",
        str(paths.metrics_jsonl),
        "--report-schema",
        str(schema_path),
        "--out-prefix",
        str(paths.plot_prefix),
    ]
    if step_limit is not None:
        args.extend(["--step-limit", str(step_limit)])
    return run_public_report_script(repo_root, "train_metrics_plots.py", args)


def build_manifest(
    *,
    status: str,
    mode: str,
    repo_root: Path,
    verl_log: Path,
    paths: ReportPaths,
    rows: list[dict[str, Any]],
    markdown_results: list[dict[str, Any]] | None = None,
    plot_result: dict[str, Any] | None = None,
    error: str | None = None,
    reporter_pid: int | None = None,
) -> dict[str, Any]:
    curve_paths = paths.curve_paths
    existing_curves = {
        name: str(path)
        for name, path in curve_paths.items()
        if path.exists() and path.stat().st_size > 0
    }
    return {
        "type": "air_llm_reranker_training_report",
        "status": status,
        "mode": mode,
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "verl_log": str(verl_log),
        "metrics_jsonl": str(paths.metrics_jsonl),
        "env_file": str(paths.env_file),
        "train_log_copy": str(paths.train_log_copy),
        "training_report": str(paths.training_report),
        "detailed_training_report": str(paths.detailed_training_report),
        "curve_paths": {name: str(path) for name, path in curve_paths.items()},
        "existing_curve_paths": existing_curves,
        "step_count": len(rows),
        "max_step": max_step(rows),
        "markdown_results": markdown_results or [],
        "plot_result": plot_result or {},
        "reporter_pid": reporter_pid,
        "error": error,
    }


def generate_once(
    *,
    verl_log: Path,
    out_dir: Path,
    repo_root: Path | None = None,
    schema_path: Path | None = None,
    step_limit: int | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "snapshot",
) -> dict[str, Any]:
    """生成一次 AIR LLM reranker metrics、markdown 和曲线。

    这个函数会尽量产出可审计文件；即使公共画图脚本失败，也会写 manifest，把错误留给调用方判断。
    """

    resolved_repo_root = (repo_root or default_repo_root()).resolve()
    resolved_schema_path = schema_path or default_schema_path(resolved_repo_root)
    paths = build_report_paths(out_dir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    markdown_results: list[dict[str, Any]] = []
    plot_result: dict[str, Any] | None = None
    status = "completed"
    error: str | None = None

    try:
        rows = read_console_metrics(resolved_repo_root, verl_log, step_limit)
        write_jsonl(paths.metrics_jsonl, rows)
        write_report_env(paths, config=config)
        copy_train_log_for_report(verl_log, paths)
        if not rows:
            status = "no_metrics"
        else:
            limit = step_limit if step_limit is not None else max_step(rows)
            markdown_results = generate_markdown_reports(
                repo_root=resolved_repo_root,
                schema_path=resolved_schema_path,
                paths=paths,
                step_limit=limit,
            )
            plot_result = generate_plots(
                repo_root=resolved_repo_root,
                schema_path=resolved_schema_path,
                paths=paths,
                step_limit=limit,
            )
            failed = [item for item in markdown_results if int(item.get("return_code", 1)) != 0]
            if plot_result and int(plot_result.get("return_code", 1)) != 0:
                failed.append(plot_result)
            if failed:
                status = "partial"
                error = "; ".join(
                    f"{Path(str(item.get('script', 'unknown'))).name}: {item.get('stderr') or item.get('stdout')}"
                    for item in failed
                )
    except Exception as exc:  # noqa: BLE001 - reporter 不能影响主训练，只记录错误。
        status = "failed"
        error = repr(exc)

    manifest = build_manifest(
        status=status,
        mode=mode,
        repo_root=resolved_repo_root,
        verl_log=verl_log,
        paths=paths,
        rows=rows,
        markdown_results=markdown_results,
        plot_result=plot_result,
        error=error,
    )
    write_json(paths.report_manifest, manifest)
    return manifest


def should_refresh(last_step: int, current_step: int, step_interval: int) -> bool:
    if current_step <= 0:
        return False
    if current_step <= last_step:
        return False
    return (current_step - last_step) >= max(step_interval, 1)


def run_periodic_reporter(
    *,
    verl_log: Path,
    out_dir: Path,
    repo_root: Path | None = None,
    schema_path: Path | None = None,
    interval_seconds: int = 60,
    step_interval: int = 1,
    stop_file: Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """训练过程中周期刷新 latest 图。

    循环退出条件是 stop_file 出现。每轮先解析日志最大 step，只有发现新的 completed step 时才重新生成图，
    避免长时间训练时无意义地反复画同一批数据。
    """

    resolved_repo_root = (repo_root or default_repo_root()).resolve()
    resolved_schema_path = schema_path or default_schema_path(resolved_repo_root)
    paths = build_report_paths(out_dir)
    last_rendered_step = 0
    heartbeat_path = paths.out_dir / f"{REPORT_PREFIX}.reporter_heartbeat.json"
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        rows: list[dict[str, Any]] = []
        try:
            rows = read_console_metrics(resolved_repo_root, verl_log, None)
            current_step = max_step(rows)
            write_json(
                heartbeat_path,
                {
                    "status": "running",
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "last_rendered_step": last_rendered_step,
                    "current_step": current_step,
                    "stop_file": str(stop_file) if stop_file else None,
                },
            )
            if should_refresh(last_rendered_step, current_step, step_interval):
                manifest = generate_once(
                    verl_log=verl_log,
                    out_dir=out_dir,
                    repo_root=resolved_repo_root,
                    schema_path=resolved_schema_path,
                    step_limit=current_step,
                    config=config,
                    mode="periodic",
                )
                last_rendered_step = int(manifest.get("max_step") or current_step)
        except Exception as exc:  # noqa: BLE001 - 后台 reporter 失败只落盘，不影响训练。
            write_json(
                paths.report_manifest,
                build_manifest(
                    status="failed",
                    mode="periodic",
                    repo_root=resolved_repo_root,
                    verl_log=verl_log,
                    paths=paths,
                    rows=rows,
                    error=repr(exc),
                    reporter_pid=os.getpid(),
                ),
            )

        if stop_file is not None and stop_file.exists():
            break
        time.sleep(max(int(interval_seconds), 1))

    write_json(
        heartbeat_path,
        {
            "status": "stopped",
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "last_rendered_step": last_rendered_step,
            "stop_file": str(stop_file) if stop_file else None,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AIR LLM reranker training reports and curves.")
    parser.add_argument("--verl-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--schema-path", type=Path)
    parser.add_argument("--config-yaml", type=Path)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--mode", choices=["once", "periodic"], default="once")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--step-interval", type=int, default=1)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    config = load_config_yaml(args.config_yaml)

    if args.mode == "periodic":
        run_periodic_reporter(
            verl_log=args.verl_log,
            out_dir=args.out_dir,
            repo_root=args.repo_root,
            schema_path=args.schema_path,
            interval_seconds=args.interval_seconds,
            step_interval=args.step_interval,
            stop_file=args.stop_file,
            config=config,
        )
        return

    manifest = generate_once(
        verl_log=args.verl_log,
        out_dir=args.out_dir,
        repo_root=args.repo_root,
        schema_path=args.schema_path,
        step_limit=args.step_limit,
        config=config,
        mode="manual",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

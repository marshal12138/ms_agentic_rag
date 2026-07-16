"""Stage 1 Search-Policy RL runner for SPAD-RAG."""

from __future__ import annotations

import os
import json
import hashlib
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.config import init_actor_model, input_train_files
from agentic_iter_rag.agent_training.spad.checkpoint_finalizer import finalize_actor_checkpoint
from agentic_iter_rag.agent_training.spad.data import load_rl_rows, row_gold_answers, row_question
from agentic_iter_rag.agent_training.spad.manifest import (
    SEMI_STRICT_INVALID_ROLLOUT_RATE,
    is_invalid_rollout_record,
    write_records,
    write_sub_stage_manifest,
)
from agentic_iter_rag.agent_training.spad.prompts import (
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
    resolve_teacher_prompt,
)
from agentic_iter_rag.agent_training.spad.reward import compute_search_policy_reward
from agentic_iter_rag.agent_training.spad.teacher_strategies import (
    validate_teacher_strategy_config,
)
from agentic_iter_rag.agent_training.spad.service_manager import SpadServiceManager, as_int_list, csv_ids, project_root, repo_root, tail_text
from agentic_iter_rag.utils.io import write_json, write_yaml


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _scalar_override(key: str, value: Any) -> str:
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={'True' if value else 'False'}"
    if isinstance(value, (int, float)):
        return f"{key}={value}"
    return f"{key}={_quote(value)}"


def _string_override(key: str, value: Any) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{escaped}"'


def _list_override(key: str, values: list[Any]) -> str:
    rendered = ",".join(_quote(item) for item in values)
    return f"{key}=[{rendered}]"


def _dict_field_overrides(prefix: str, value: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for key, item in value.items():
        item_key = f"{prefix}.{key}"
        if isinstance(item, dict):
            overrides.extend(_dict_field_overrides(item_key, item))
        elif isinstance(item, list):
            overrides.append(_list_override(item_key, item))
        else:
            overrides.append(_scalar_override(item_key, item))
    return overrides


def _reward_type(spad_cfg: dict[str, Any]) -> str:
    return str((spad_cfg.get("reward") or {}).get("type") or "spad_teacher_f1")


def _uses_teacher_reward(spad_cfg: dict[str, Any]) -> bool:
    return _reward_type(spad_cfg) not in {"search_r1_original", "search_r1_structured"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_rollout_manifest(
    rollout_data_dir: Path,
    *,
    require_teacher_audit: bool,
) -> dict[str, Any]:
    manifest_path = rollout_data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completed training is missing rollout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("completed")):
        raise ValueError(f"rollout manifest is incomplete: {manifest_path}")
    for actual_key, expected_key in (
        ("actual_step_count", "expected_steps"),
        ("actual_prompt_count", "expected_prompt_count"),
        ("actual_group_count", "expected_group_count"),
        ("actual_rollout_count", "expected_rollout_count"),
    ):
        if int(manifest.get(actual_key, -1)) != int(manifest.get(expected_key, -2)):
            raise ValueError(
                f"rollout manifest count mismatch {actual_key}={manifest.get(actual_key)} "
                f"!= {expected_key}={manifest.get(expected_key)}"
            )
    field_counts: dict[str, int] = defaultdict(int)
    invalid_trajectory_count = 0
    for shard in manifest.get("shards") or []:
        shard_path = Path(str(shard["path"]))
        if not shard_path.is_file():
            raise FileNotFoundError(f"rollout shard is missing: {shard_path}")
        if _sha256_file(shard_path) != str(shard["sha256"]):
            raise ValueError(f"rollout shard hash mismatch: {shard_path}")
        with shard_path.open(encoding="utf-8") as handle:
            line_count = 0
            shard_invalid_count = 0
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                line_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"rollout shard contains invalid JSON: {shard_path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"rollout shard record is not an object: {shard_path}:{line_number}"
                    )
                shard_invalid_count += int(is_invalid_rollout_record(record))
        if line_count != int(shard["record_count"]):
            raise ValueError(f"rollout shard line count mismatch: {shard_path}")
        invalid_trajectory_count += shard_invalid_count
        for key, value in (shard.get("field_nonempty_counts") or {}).items():
            field_counts[str(key)] += int(value)
    expected_rollouts = int(manifest["expected_rollout_count"])
    invalid_rate = invalid_trajectory_count / expected_rollouts if expected_rollouts else 0.0
    if invalid_rate > SEMI_STRICT_INVALID_ROLLOUT_RATE:
        raise ValueError(
            "rollout audit invalid trajectory rate exceeds semi-strict limit: "
            f"invalid={invalid_trajectory_count}/{expected_rollouts} "
            f"({invalid_rate:.6%}) > {SEMI_STRICT_INVALID_ROLLOUT_RATE:.6%}"
        )
    if require_teacher_audit:
        evidence_records = field_counts.get("search_count", 0)
        if field_counts.get("tool_call_details", 0) != evidence_records:
            raise ValueError("tool_call_details are not aligned with non-zero search_count records")
        if field_counts.get("teacher_messages", 0) != evidence_records:
            raise ValueError("teacher_messages are not materialized for every rollout with evidence")
        if int(manifest.get("teacher_called_count", 0)) > field_counts.get("teacher_messages", 0):
            raise ValueError("teacher calls exceed materialized teacher message records")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "summary": {
            **manifest,
            "invalid_trajectory_count": invalid_trajectory_count,
            "invalid_trajectory_rate": invalid_rate,
            "invalid_trajectory_rate_limit": SEMI_STRICT_INVALID_ROLLOUT_RATE,
            "invalid_trajectory_policy": "semi_strict_allow_at_most_0.5_percent",
        },
    }


def _env_export_overrides(env_vars: dict[str, str]) -> dict[str, str]:
    return {f"+ray_kwargs.ray_init.runtime_env.env_vars.{key}": value for key, value in env_vars.items()}


def _find_latest_checkpoint(output_dir: Path) -> Path:
    candidates = [path for path in output_dir.rglob("global_step_*") if path.is_dir()]
    if not candidates:
        return output_dir

    def step_num(path: Path) -> int:
        suffix = path.name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    return sorted(candidates, key=lambda item: (step_num(item), str(item)))[-1]


def _run_shell_script(
    script_path: Path,
    log_path: Path,
    timeout_s: int | None = None,
    monitored_processes: list[Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = f"set -o pipefail; bash {_quote(script_path)} 2>&1 | tee {_quote(log_path)}"
    process = subprocess.Popen(["bash", "-lc", command], cwd=str(repo_root()), start_new_session=True)
    service_error = ""
    timed_out = False
    while True:
        return_code = process.poll()
        if return_code is not None:
            break
        if timeout_s is not None and time.time() - started > float(timeout_s):
            timed_out = True
            service_error = f"VERL command timed out after {timeout_s}s"
            break
        for managed in monitored_processes or []:
            managed_rc = managed.poll()
            if managed_rc is None:
                continue
            service_error = (
                f"monitored service exited during VERL run: {managed.name} "
                f"return_code={managed_rc} log={managed.log_path}\n{tail_text(managed.log_path)}"
            )
            break
        if service_error:
            break
        time.sleep(2.0)

    if process.poll() is None:
        import signal

        os.killpg(process.pid, signal.SIGTERM)
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait(timeout=30)
    return {
        "return_code": return_code,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
        "service_error": service_error,
        "timed_out": timed_out,
    }


def _write_runtime_tool_config(path: Path, *, retrieval_url: str, top_n: int, top_m: int, timeout: int) -> Path:
    payload = {
        "tools": [
            {
                "class_name": "verl.tools.agentic_iter_rag_retriever_tool.AgenticIterRagRetrieverTool",
                "config": {
                    "type": "native",
                    "retrieval_service_url": retrieval_url,
                    "timeout": timeout,
                    "recall_final_top_n": top_n,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "fail_fast_on_recall_error": True,
                    "searchTool_final_top_m": top_m,
                    "hit_cutoffs": [1, 3, 5],
                    "tool_score_metric": "hit",
                    "trivial_answers": ["yes", "no", "true", "false"],
                    "max_concurrent_per_worker": 4,
                    "ranker_enabled": False,
                    "ranker": {"backend": "ray_actor", "required": False},
                },
            }
        ]
    }
    write_yaml(path, payload)
    return path


def _write_verl_script(*, script_path: Path, plan: dict[str, Any], actor_gpu_ids: list[int]) -> Path:
    compat_python = repo_root() / "src" / "env_manage" / "compatible_python.sh"
    air_accel = repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "00_air_accelerator.sh"
    actor_gpu_csv = csv_ids(actor_gpu_ids)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {_quote(repo_root())}",
        f"source {_quote(compat_python)}",
        f"source {_quote(air_accel)}",
        f"export ASCEND_RT_VISIBLE_DEVICES={_quote(actor_gpu_csv)}",
        f"export CUDA_VISIBLE_DEVICES={_quote(actor_gpu_csv)}",
        "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1",
        "export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}",
        "export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}",
        "export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}",
        "export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-lo}",
        "export TOKENIZERS_PARALLELISM=false",
        "export VLLM_DISABLE_FLASHINFER=1",
        "export VLLM_USE_FLASHINFER_SAMPLER=0",
        "export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}",
        "export VLLM_ALLREDUCE_USE_SYMM_MEM=0",
        "export WANDB_MODE=disabled",
        f"export PYTHONPATH={_quote(plan['verl_root'])}:{_quote(project_root())}:${{PYTHONPATH:-}}",
        f"mkdir -p {_quote(plan['output_dir'])}",
        "cmd=(\"$PY\" -m verl.trainer.main_ppo)",
        "cmd+=(",
    ]
    for override in plan["hydra_overrides"]:
        lines.append(f"  {_quote(override)}")
    lines.extend(
        [
            ")",
            f"printf '%s\\n' \"${{cmd[@]}}\" > {_quote(script_path.parent / 'verl_command.argv')}",
            "exec \"${cmd[@]}\"",
        ]
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o750)
    return script_path


def _build_verl_plan(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    resource_plan: dict[str, Any],
    teacher_output: dict[str, Any],
    recall_output: dict[str, Any],
) -> dict[str, Any]:
    sub_cfg = spad_cfg["sub_stages"]["search_policy_rl"]
    trainer_cfg = dict(sub_cfg.get("trainer") or {})
    rollout_cfg = dict(sub_cfg.get("rollout") or {})
    teacher_cfg = dict(spad_cfg.get("teacher_answerer") or {})
    teacher_prompt_version, _ = resolve_teacher_prompt(
        str(teacher_cfg.get("prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION),
        include_status=True,
    )
    actor_resource = resource_plan["trainer"]
    actor_gpu_ids = as_int_list(actor_resource["gpu_ids"])
    actor_tp = int(actor_resource.get("tensor_parallel_size") or actor_resource.get("n_gpus_per_node") or len(actor_gpu_ids))
    output_dir = checkpoint_dir / "actor_model_verl"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_data_dir = stage_dir / "rollout_data"
    validation_data_dir = stage_dir / "validation_data"
    runtime_dir = log_dir
    tool_config_path = _write_runtime_tool_config(
        runtime_dir / "spad_search_tool_config.yaml",
        retrieval_url=str(recall_output["retrieval_url"]),
        top_n=int(config.get("infer_runtime", {}).get("retriever", {}).get("recall_final_top_n") or 50),
        top_m=int(rollout_cfg.get("visible_top_m", 5)),
        timeout=int(rollout_cfg.get("search_timeout", 30)),
    )
    train_files = input_train_files(config, spad_cfg)
    val_files = config.get("data", {}).get("val_files") or train_files
    if isinstance(val_files, str):
        val_files = [val_files]
    max_prompt_length = int(trainer_cfg.get("max_prompt_length") or config.get("data", {}).get("max_prompt_length") or 12000)
    max_response_length = int(trainer_cfg.get("max_response_length") or config.get("data", {}).get("max_response_length") or 4096)
    rollout_n = int(trainer_cfg.get("n_samples_per_prompt", 8))
    train_batch_size = int(trainer_cfg.get("train_batch_size", 64))
    max_model_len = int(trainer_cfg.get("rollout_max_model_len") or (max_prompt_length + max_response_length))
    configured_total_training_steps = trainer_cfg.get("total_training_steps")
    total_training_steps = None
    if configured_total_training_steps is not None:
        total_training_steps = int(configured_total_training_steps)
        # SPAD overlays use -1 to mean "formal/full run". VERL expects null for
        # that behavior; a negative trainer.total_training_steps would skip work.
        if total_training_steps < 0:
            total_training_steps = None
    teacher_request = dict(teacher_cfg.get("request") or {})
    teacher_request["endpoint"] = teacher_output["endpoint"]
    teacher_request["model"] = teacher_output["model"]
    reward_type = str(spad_cfg.get("reward", {}).get("type") or "")
    hard_gate_v3_reward = (
        reward_type
        == "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2"
    )
    configured_strategy_id = str(teacher_cfg.get("strategy_id") or "")
    if configured_strategy_id and not hard_gate_v3_reward:
        raise ValueError(
            "teacher_answerer.strategy_id is only supported by the independent "
            "Hard-Gate v2 reward"
        )
    teacher_strategy = (
        validate_teacher_strategy_config(teacher_cfg) if hard_gate_v3_reward else None
    )
    reward_kwargs = {
        "reward_cfg": spad_cfg.get("reward", {}),
        "teacher_request": teacher_request,
        "teacher_prompt_version": teacher_prompt_version,
        "visible_top_m": int(rollout_cfg.get("visible_top_m", 5)),
        "batch_workers": int(trainer_cfg.get("teacher_batch_workers", 16)),
        "n_samples_per_prompt": rollout_n,
    }
    if teacher_strategy is not None:
        reward_kwargs["teacher_strategy_id"] = teacher_strategy.strategy_id
    legacy_0710_reward = reward_type == "spad_teacher_f1_0710"
    current_group_reward = reward_type == "spad_em_teacher_backoff"
    gold_token_f1_bonus_reward = reward_type == "spad_em_teacher_backoff_gold_token_f1_bonus"
    gold_token_f1_bonus_v3_reward = (
        reward_type == "spad_em_teacher_backoff_gold_token_f1_bonus_v3"
    )
    dev_group_reward = reward_type == "spad_em_teacher_backoff_dev"
    reward_module_name = (
        "search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2.py"
        if hard_gate_v3_reward
        else (
            "search_policy_teacher_reward_0710.py"
            if legacy_0710_reward
            else (
                (
                    "search_policy_teacher_reward_gold_match_bonus_v3.py"
                    if gold_token_f1_bonus_v3_reward
                    else "search_policy_teacher_reward_gold_match_bonus.py"
                )
                if gold_token_f1_bonus_reward or gold_token_f1_bonus_v3_reward
                else (
                    "search_policy_teacher_reward_dev.py"
                    if dev_group_reward
                    else "search_policy_teacher_reward.py"
                )
            )
        )
    )
    reward_path = (
        project_root()
        / "agentic_iter_rag"
        / "agent_training"
        / "spad"
        / "rewards"
        / reward_module_name
    )
    agent_loop_config = project_root() / "config" / "spad_search_policy_agent_loop_config.yaml"
    env_vars = {
        "SPAD_TEACHER_ENDPOINT": str(teacher_output["endpoint"]),
        "SPAD_TEACHER_MODEL": str(teacher_output["model"]),
        "SPAD_TEACHER_BATCH_WORKERS": str(trainer_cfg.get("teacher_batch_workers", 16)),
        "COSEARCH_ROLLOUT_PROGRESS_INTERVAL": str(trainer_cfg.get("rollout_progress_interval", 60)),
        "COSEARCH_ACTOR_PROGRESS_INTERVAL": str(trainer_cfg.get("actor_progress_interval", 8)),
        "COSEARCH_TRAJECTORY_TIMEOUT_SECONDS": str(trainer_cfg.get("trajectory_timeout_seconds", "")),
        "COSEARCH_TURN_MAX_TOKENS": str(rollout_cfg.get("turn_max_tokens", max_response_length)),
    }
    if legacy_0710_reward:
        reward_manager = "naive"
    elif (
        current_group_reward
        or gold_token_f1_bonus_reward
        or gold_token_f1_bonus_v3_reward
        or hard_gate_v3_reward
        or dev_group_reward
    ):
        reward_manager = "batch"
    else:
        reward_manager = str(trainer_cfg.get("reward_manager") or "naive")
    stream_group_reward = (
        False
        if legacy_0710_reward
        else bool(trainer_cfg.get("stream_group_reward", reward_manager == "batch"))
    )
    if hard_gate_v3_reward:
        reward_function_name = (
            "compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_"
            "hard_gate_v2_batch"
        )
        stop_sequences = [
            str(item)
            for item in rollout_cfg.get("stop_sequences", ["</tool_call>", "</answer>"])
        ]
    elif legacy_0710_reward:
        reward_function_name = "compute_spad_teacher_f1_0710_details"
        stop_sequences = ["</tool_call>", "<answer>"]
    elif current_group_reward:
        reward_function_name = "compute_spad_em_teacher_backoff_batch"
        stop_sequences = [
            str(item)
            for item in rollout_cfg.get("stop_sequences", ["</tool_call>", "</answer>"])
        ]
    elif gold_token_f1_bonus_reward:
        reward_function_name = "compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch"
        stop_sequences = [
            str(item)
            for item in rollout_cfg.get("stop_sequences", ["</tool_call>", "</answer>"])
        ]
    elif gold_token_f1_bonus_v3_reward:
        reward_function_name = (
            "compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_batch"
        )
        stop_sequences = [
            str(item)
            for item in rollout_cfg.get("stop_sequences", ["</tool_call>", "</answer>"])
        ]
    elif dev_group_reward:
        reward_function_name = "compute_spad_em_teacher_backoff_dev"
        stop_sequences = [
            str(item)
            for item in rollout_cfg.get("stop_sequences", ["</tool_call>", "</answer>"])
        ]
    else:
        reward_function_name = (
            "compute_spad_search_policy_reward_batch"
            if reward_manager == "batch"
            else "compute_spad_search_policy_reward_details"
        )
        stop_sequences = [
            str(item)
            for item in rollout_cfg.get("stop_sequences", ["</tool_call>", "</answer>"])
        ]
    hydra_overrides: list[str] = [
        _scalar_override("algorithm.adv_estimator", "grpo"),
        _scalar_override(
            "algorithm.norm_adv_by_std_in_grpo",
            bool(trainer_cfg.get("norm_adv_by_std_in_grpo", False)),
        ),
        _scalar_override("algorithm.use_kl_in_reward", False),
        _scalar_override("critic.enable", False),
        _scalar_override("reward_model.enable", False),
        _scalar_override(
            "+reward_model.use_reward_loop",
            dev_group_reward or reward_manager != "batch",
        ),
        _scalar_override(
            "+reward_model.reward_loop_manager",
            "naive" if dev_group_reward else reward_manager,
        ),
        _scalar_override(
            "+reward_model.stream_group_reward",
            stream_group_reward,
        ),
        _scalar_override(
            "+reward_model.stream_group_max_inflight",
            int(trainer_cfg.get("stream_group_max_inflight", 1)),
        ),
        _scalar_override("reward_model.reward_manager", reward_manager),
        _list_override("data.train_files", [str(item) for item in train_files]),
        _list_override("data.val_files", [str(item) for item in val_files]),
        _scalar_override("data.train_batch_size", train_batch_size),
        _scalar_override("data.val_batch_size", int(trainer_cfg.get("val_batch_size", config.get("data", {}).get("val_batch_size", 8)))),
        _scalar_override("data.train_max_samples", int(trainer_cfg.get("train_max_samples", config.get("data", {}).get("train_max_samples", -1)))),
        _scalar_override("data.val_max_samples", int(trainer_cfg.get("val_max_samples", config.get("data", {}).get("val_max_samples", 8)))),
        _scalar_override("data.shuffle", bool(trainer_cfg.get("data_shuffle", True))),
        _scalar_override("data.seed", int(trainer_cfg.get("data_seed", 42))),
        _scalar_override("data.max_prompt_length", max_prompt_length),
        _scalar_override("data.max_response_length", max_response_length),
        _scalar_override("data.truncation", str(config.get("data", {}).get("truncation", "error"))),
        _scalar_override("data.return_raw_chat", True),
        _scalar_override("data.trust_remote_code", True),
        _scalar_override("data.dataloader_num_workers", int(trainer_cfg.get("dataloader_num_workers", 0))),
        _scalar_override("+data.apply_chat_template_kwargs.enable_thinking", False),
        _scalar_override("actor_rollout_ref.model.path", init_actor_model(config, spad_cfg)),
        _scalar_override("actor_rollout_ref.model.trust_remote_code", True),
        _scalar_override("actor_rollout_ref.model.use_shm", bool(config.get("model", {}).get("use_shm", False))),
        _scalar_override("actor_rollout_ref.model.use_remove_padding", bool(trainer_cfg.get("use_remove_padding", True))),
        _scalar_override("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        _scalar_override("actor_rollout_ref.actor.optim.lr", float(trainer_cfg.get("learning_rate", 1.0e-6))),
        _scalar_override("actor_rollout_ref.actor.use_torch_compile", bool(trainer_cfg.get("use_torch_compile", False))),
        _scalar_override("actor_rollout_ref.actor.ppo_mini_batch_size", int(trainer_cfg.get("ppo_mini_batch_size", train_batch_size))),
        _scalar_override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", int(trainer_cfg.get("actor_micro_batch_size_per_gpu", 2))),
        _scalar_override("actor_rollout_ref.actor.use_kl_loss", bool(trainer_cfg.get("use_kl_loss", True))),
        _scalar_override("actor_rollout_ref.actor.kl_loss_coef", float(trainer_cfg.get("kl_loss_coef", 0.001))),
        _scalar_override("actor_rollout_ref.actor.kl_loss_type", str(trainer_cfg.get("kl_loss_type", "low_var_kl"))),
        _scalar_override("actor_rollout_ref.actor.entropy_coeff", float(trainer_cfg.get("entropy_coeff", 0.0))),
        _scalar_override("actor_rollout_ref.actor.fsdp_config.param_offload", bool(trainer_cfg.get("actor_param_offload", False))),
        _scalar_override("actor_rollout_ref.actor.fsdp_config.optimizer_offload", bool(trainer_cfg.get("actor_optimizer_offload", False))),
        _scalar_override("actor_rollout_ref.actor.fsdp_config.use_torch_compile", bool(trainer_cfg.get("use_torch_compile", False))),
        _scalar_override("actor_rollout_ref.ref.use_torch_compile", bool(trainer_cfg.get("use_torch_compile", False))),
        _scalar_override("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", int(trainer_cfg.get("log_prob_micro_batch_size_per_gpu", 4))),
        _scalar_override("actor_rollout_ref.ref.fsdp_config.param_offload", bool(trainer_cfg.get("ref_param_offload", False))),
        _scalar_override("actor_rollout_ref.ref.fsdp_config.use_torch_compile", bool(trainer_cfg.get("use_torch_compile", False))),
        _scalar_override("actor_rollout_ref.rollout.name", "vllm"),
        _scalar_override("actor_rollout_ref.rollout.mode", str(trainer_cfg.get("rollout_mode", "async"))),
        _scalar_override("actor_rollout_ref.rollout.tensor_model_parallel_size", actor_tp),
        _scalar_override("actor_rollout_ref.rollout.n", rollout_n),
        _scalar_override("actor_rollout_ref.rollout.temperature", float(trainer_cfg.get("rollout_temperature", 1.0))),
        _scalar_override("actor_rollout_ref.rollout.top_p", float(trainer_cfg.get("rollout_top_p", 1.0))),
        _scalar_override("actor_rollout_ref.rollout.gpu_memory_utilization", float(trainer_cfg.get("rollout_gpu_memory_utilization", 0.6))),
        _scalar_override("actor_rollout_ref.rollout.max_model_len", max_model_len),
        _scalar_override("actor_rollout_ref.rollout.prompt_length", max_prompt_length),
        _scalar_override("actor_rollout_ref.rollout.response_length", max_response_length),
        _scalar_override("actor_rollout_ref.rollout.max_num_batched_tokens", int(trainer_cfg.get("max_num_batched_tokens", max_model_len))),
        _scalar_override("actor_rollout_ref.rollout.max_num_seqs", int(trainer_cfg.get("max_num_seqs", rollout_n))),
        _scalar_override("actor_rollout_ref.rollout.enforce_eager", bool(trainer_cfg.get("enforce_eager", True))),
        _scalar_override("actor_rollout_ref.rollout.enable_chunked_prefill", bool(trainer_cfg.get("enable_chunked_prefill", False))),
        _scalar_override("actor_rollout_ref.rollout.enable_prefix_caching", bool(trainer_cfg.get("enable_prefix_caching", False))),
        _scalar_override("actor_rollout_ref.rollout.calculate_log_probs", bool(trainer_cfg.get("calculate_rollout_log_probs", False))),
        _scalar_override("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", int(trainer_cfg.get("log_prob_micro_batch_size_per_gpu", 4))),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.enable", True),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.max_user_turns", int(rollout_cfg.get("max_user_turns", 6))),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.max_assistant_turns", int(rollout_cfg.get("max_assistant_turns", 6))),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.max_parallel_calls", 1),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.max_tool_response_length", int(rollout_cfg.get("max_tool_response_length", 4096))),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side", str(rollout_cfg.get("tool_response_truncate_side", "left"))),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.format", "search_r1"),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.tool_config_path", str(tool_config_path)),
        _scalar_override("actor_rollout_ref.rollout.agent.num_workers", int(trainer_cfg.get("agent_loop_num_workers", 8))),
        _scalar_override("actor_rollout_ref.rollout.agent.default_agent_loop", "spad_search_policy_agent"),
        _scalar_override("actor_rollout_ref.rollout.agent.agent_loop_config_path", str(agent_loop_config)),
        _list_override("+actor_rollout_ref.rollout.stop", stop_sequences),
        _scalar_override("+actor_rollout_ref.rollout.include_stop_str_in_output", bool(rollout_cfg.get("include_stop_str_in_output", True))),
        _scalar_override("custom_reward_function.path", str(reward_path)),
        _scalar_override("custom_reward_function.name", reward_function_name),
        _scalar_override("+custom_reward_function.reward_kwargs.visible_top_m", reward_kwargs["visible_top_m"]),
        _scalar_override("+custom_reward_function.reward_kwargs.batch_workers", reward_kwargs["batch_workers"]),
        _scalar_override(
            "+custom_reward_function.reward_kwargs.n_samples_per_prompt",
            reward_kwargs["n_samples_per_prompt"],
        ),
        _scalar_override(
            "+custom_reward_function.reward_kwargs.teacher_prompt_version",
            reward_kwargs["teacher_prompt_version"],
        ),
        _scalar_override("trainer.n_gpus_per_node", len(actor_gpu_ids)),
        _scalar_override("trainer.nnodes", int(trainer_cfg.get("nnodes", 1))),
        _scalar_override("trainer.device", str(trainer_cfg.get("device", "npu"))),
        _list_override("trainer.logger", [str(item) for item in trainer_cfg.get("logger", ["console"])]),
        _scalar_override("trainer.total_epochs", int(trainer_cfg.get("total_epochs", 1))),
        _scalar_override("trainer.total_training_steps", total_training_steps),
        _scalar_override("trainer.project_name", str(trainer_cfg.get("project_name", "spad_rag"))),
        _scalar_override("trainer.experiment_name", str(config["main_run"]["project"]["experiment_name"])),
        _scalar_override("trainer.default_local_dir", str(output_dir)),
        _scalar_override("trainer.val_before_train", bool(trainer_cfg.get("val_before_train", False))),
        _scalar_override("trainer.save_freq", int(trainer_cfg.get("save_freq", 5))),
        _scalar_override("trainer.test_freq", int(trainer_cfg.get("test_freq", -1))),
        _scalar_override("trainer.max_actor_ckpt_to_keep", int(trainer_cfg.get("max_actor_ckpt_to_keep", 1))),
        _scalar_override("trainer.rollout_data_dir", str(rollout_data_dir)),
        _scalar_override("trainer.validation_data_dir", str(validation_data_dir)),
        _scalar_override("+trainer.num_examine", int(trainer_cfg.get("num_examine", 0))),
        _scalar_override("+trainer.val_num_examine", int(trainer_cfg.get("val_num_examine", 0))),
        _scalar_override("+ray_kwargs.ray_init.num_cpus", int(trainer_cfg.get("ray_num_cpus", 24))),
        _scalar_override("+ray_kwargs.ray_init.include_dashboard", False),
        _scalar_override("+ray_kwargs.ray_init.ignore_reinit_error", True),
    ]
    if hard_gate_v3_reward:
        hydra_overrides.append(
            _scalar_override(
                "+custom_reward_function.reward_kwargs.teacher_strategy_id",
                reward_kwargs["teacher_strategy_id"],
            )
        )
    if gold_token_f1_bonus_v3_reward or hard_gate_v3_reward:
        if not bool(trainer_cfg.get("norm_adv_by_std_in_grpo", False)):
            raise ValueError(
                "Gold Token-F1 V3 requires norm_adv_by_std_in_grpo=true"
            )
        hydra_overrides.extend(
            [
                _scalar_override(
                    "+algorithm.group_postnorm_advantage_scale_key",
                    "advantage_postnorm_scale",
                ),
                _scalar_override(
                    "+algorithm.group_postnorm_advantage_scale_version",
                    "teacher_fallback_v1",
                ),
            ]
        )
    hydra_overrides.extend(
        _dict_field_overrides("+custom_reward_function.reward_kwargs.reward_cfg", reward_kwargs["reward_cfg"])
    )
    hydra_overrides.extend(
        _dict_field_overrides("+custom_reward_function.reward_kwargs.teacher_request", reward_kwargs["teacher_request"])
    )
    for key, value in _env_export_overrides(env_vars).items():
        hydra_overrides.append(_string_override(key, value))
    verl_root = Path(str(trainer_cfg.get("verl_root") or project_root() / "verl"))
    return {
        "status": "planned",
        "entry": "python -m verl.trainer.main_ppo",
        "verl_root": str(verl_root),
        "output_dir": str(output_dir),
        "rollout_data_dir": str(rollout_data_dir),
        "validation_data_dir": str(validation_data_dir),
        "runtime_dir": str(runtime_dir),
        "tool_config_path": str(tool_config_path),
        "agent_loop_config_path": str(agent_loop_config),
        "reward_path": str(reward_path),
        "hydra_overrides": hydra_overrides,
        "env_vars": env_vars,
        "actor_gpu_ids": actor_gpu_ids,
    }


def _run_verl_backend(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    resource_plan: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    sub_cfg = spad_cfg["sub_stages"]["search_policy_rl"]
    runtime_dir = log_dir
    trainer_cfg = dict(sub_cfg.get("trainer") or {})
    manager = SpadServiceManager(runtime_dir=runtime_dir / "services", verl_root=Path(str(trainer_cfg.get("verl_root") or project_root() / "verl")))
    service_outputs: dict[str, Any] = {}
    try:
        uses_teacher_reward = _uses_teacher_reward(spad_cfg)
        teacher_resource = resource_plan.get("services", {}).get("teacher_answerer", {})
        recall_resource = resource_plan.get("services", {}).get("recall", {})
        skipped_teacher_output = {"status": "skipped", "reason": f"reward_type={_reward_type(spad_cfg)}", "endpoint": "", "model": ""}
        if dry_run:
            service_outputs = {
                "teacher": (
                    {"status": "planned"}
                    if uses_teacher_reward
                    else {"status": "skipped", "reason": f"reward_type={_reward_type(spad_cfg)}"}
                ),
                "recall": {"status": "planned"},
            }
            teacher_output = (
                {
                    "endpoint": teacher_resource.get("endpoint", "http://127.0.0.1:8067/v1/chat/completions"),
                    "model": teacher_resource.get("served_model_name", "GLM-4.7-Flash"),
                }
                if uses_teacher_reward
                else skipped_teacher_output
            )
            recall_output = {"retrieval_url": recall_resource.get("retrieval_service_url", "http://127.0.0.1:8130/retrieve")}
        else:
            if uses_teacher_reward:
                service_outputs["teacher"] = manager.start_teacher(
                    teacher_cfg=spad_cfg["teacher_answerer"],
                    resource_cfg=teacher_resource,
                )
            else:
                service_outputs["teacher"] = skipped_teacher_output
            service_outputs["recall"] = manager.start_recall(
                recall_cfg=recall_resource,
                final_top_n=int(config.get("infer_runtime", {}).get("retriever", {}).get("recall_final_top_n") or 50),
                recall_model=str(config["infer_runtime"]["models"]["recall_model_path"]),
            )
            teacher_output = service_outputs["teacher"]
            recall_output = service_outputs["recall"]
        plan = _build_verl_plan(
            config=config,
            spad_cfg=spad_cfg,
            stage_dir=stage_dir,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            resource_plan=resource_plan,
            teacher_output=teacher_output,
            recall_output=recall_output,
        )
        write_json(runtime_dir / "verl_command_plan.json", plan)
        script_path = _write_verl_script(
            script_path=runtime_dir / "run_spad_search_policy_grpo.sh",
            plan=plan,
            actor_gpu_ids=plan["actor_gpu_ids"],
        )
        if dry_run:
            checkpoint = str(Path(plan["output_dir"]))
            rollout_manifest = {
                "status": "planned",
                "path": str(Path(plan["rollout_data_dir"]) / "manifest.json"),
            }
            finalizer = finalize_actor_checkpoint(
                checkpoint,
                hf_root=checkpoint_dir / "actor_model_hf",
                log_dir=runtime_dir,
                dry_run=True,
            )
            status = "planned"
            run_info = None
        else:
            log_path = runtime_dir / "verl_train.log"
            run_info = _run_shell_script(
                script_path,
                log_path,
                timeout_s=trainer_cfg.get("timeout_seconds"),
                monitored_processes=manager.processes,
            )
            if int(run_info["return_code"]) != 0:
                service_error = str(run_info.get("service_error") or "")
                service_context = f"\n{service_error}" if service_error else ""
                raise RuntimeError(
                    f"SPAD Stage 1 VERL exited with code {run_info['return_code']}; log={log_path}"
                    f"{service_context}\n{tail_text(log_path)}"
                )
            rollout_manifest = _validate_rollout_manifest(
                Path(plan["rollout_data_dir"]),
                require_teacher_audit=_reward_type(spad_cfg)
                in {
                    "spad_em_teacher_backoff",
                    "spad_em_teacher_backoff_gold_token_f1_bonus",
                    "spad_em_teacher_backoff_gold_token_f1_bonus_v3",
                    "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2",
                },
            )
            checkpoint = str(_find_latest_checkpoint(Path(plan["output_dir"])))
            finalizer = finalize_actor_checkpoint(
                checkpoint,
                hf_root=checkpoint_dir / "actor_model_hf",
                log_dir=runtime_dir,
            )
            status = "completed"
        outputs = {
            "status": status,
            "backend": "verl",
            "init_actor_model": init_actor_model(config, spad_cfg),
            "actor_checkpoint": checkpoint,
            "raw_actor_checkpoint": checkpoint,
            "hf_actor_checkpoint": finalizer["hf_actor_checkpoint"],
            "checkpoint_finalizer": finalizer,
            "verl_command_plan": str(runtime_dir / "verl_command_plan.json"),
            "verl_launch_script": str(script_path),
            "runtime_dir": str(runtime_dir),
            "rollout_jsonl": str(Path(plan["rollout_data_dir"])),
            "reward_jsonl": str(Path(plan["rollout_data_dir"])),
            "rollout_manifest": rollout_manifest,
            "service_outputs": service_outputs,
            "run_info": run_info,
            "resource_plan": resource_plan,
        }
        outputs["manifest"] = str(stage_dir / "manifest.json")
        write_sub_stage_manifest(outputs["manifest"], sub_stage="search_policy_rl", outputs=outputs)
        return outputs
    finally:
        if bool(trainer_cfg.get("auto_stop_services", True)):
            manager.stop_all()


def run_search_policy_rl(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    resource_plan: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Run Stage 1. The first implementation provides a smoke backend."""

    sub_cfg = spad_cfg["sub_stages"]["search_policy_rl"]
    backend = str(sub_cfg.get("backend") or spad_cfg.get("default_backend") or "smoke")
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"actor_checkpoint_{backend}"
    rollout_jsonl = stage_dir / "rollout_smoke.jsonl"
    reward_jsonl = stage_dir / "reward_smoke.jsonl"
    if backend == "verl":
        return _run_verl_backend(
            config=config,
            spad_cfg=spad_cfg,
            stage_dir=stage_dir,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            resource_plan=resource_plan,
            dry_run=dry_run,
        )
    if dry_run:
        outputs = {
            "status": "planned",
            "backend": backend,
            "actor_checkpoint": str(checkpoint_path),
            "rollout_jsonl": str(rollout_jsonl),
            "reward_jsonl": str(reward_jsonl),
            "runtime_dir": str(log_dir),
            "resource_plan": resource_plan,
        }
        outputs["manifest"] = str(stage_dir / "manifest.json")
        write_sub_stage_manifest(outputs["manifest"], sub_stage="search_policy_rl", outputs=outputs)
        return outputs
    if backend != "smoke":
        raise ValueError(f"unsupported SPAD Stage 1 backend: {backend}")

    max_samples = int(sub_cfg.get("smoke", {}).get("max_samples", 8))
    rows = load_rl_rows(input_train_files(config, spad_cfg), max_samples=max_samples)
    rollout_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = row_question(row)
        gold_answers = row_gold_answers(row)
        teacher_answer = gold_answers[0] if gold_answers else ""
        actor_output = f"<reason>Smoke backend stops after evidence placeholder for: {question}</reason>\n<answer>{teacher_answer}</answer>"
        reward = compute_search_policy_reward(
            actor_output=actor_output,
            teacher_answer=teacher_answer,
            gold_answers=gold_answers,
            search_count=1,
            duplicate_query_count=0,
            reward_cfg=spad_cfg.get("reward", {}),
            legal_stop=True,
        )
        rollout_records.append(
            {
                "index": index,
                "question": question,
                "gold_answers": gold_answers,
                "actor_output": actor_output,
                "search_count": 1,
                "teacher_answer": teacher_answer,
            }
        )
        reward_records.append({"index": index, "question": question, **reward})

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    (checkpoint_path / "README.txt").write_text(
        "SPAD-RAG Stage 1 smoke checkpoint placeholder. Real actor training requires backend=verl.\n",
        encoding="utf-8",
    )
    write_records(rollout_jsonl, rollout_records)
    write_records(reward_jsonl, reward_records)
    outputs = {
        "status": "completed",
        "backend": backend,
        "sample_count": len(rows),
        "init_actor_model": init_actor_model(config, spad_cfg),
        "actor_checkpoint": str(checkpoint_path),
        "rollout_jsonl": str(rollout_jsonl),
        "reward_jsonl": str(reward_jsonl),
        "runtime_dir": str(log_dir),
        "resource_plan": resource_plan,
        "warning": "smoke backend does not update actor parameters",
    }
    outputs["manifest"] = str(stage_dir / "manifest.json")
    write_sub_stage_manifest(outputs["manifest"], sub_stage="search_policy_rl", outputs=outputs)
    return outputs

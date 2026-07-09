"""Stage 1 Search-Policy RL runner for SPAD-RAG."""

from __future__ import annotations

import os
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.config import init_actor_model, input_train_files
from agentic_iter_rag.agent_training.spad.data import load_rl_rows, row_gold_answers, row_question
from agentic_iter_rag.agent_training.spad.manifest import write_records, write_sub_stage_manifest
from agentic_iter_rag.agent_training.spad.reward import compute_search_policy_reward
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


def _run_shell_script(script_path: Path, log_path: Path, timeout_s: int | None = None) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = f"set -o pipefail; bash {_quote(script_path)} 2>&1 | tee {_quote(log_path)}"
    process = subprocess.Popen(["bash", "-lc", command], cwd=str(repo_root()), start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        import signal

        os.killpg(process.pid, signal.SIGTERM)
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait(timeout=30)
    return {"return_code": return_code, "elapsed_s": time.time() - started, "log": str(log_path)}


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
    resource_plan: dict[str, Any],
    teacher_output: dict[str, Any],
    recall_output: dict[str, Any],
) -> dict[str, Any]:
    sub_cfg = spad_cfg["sub_stages"]["search_policy_rl"]
    trainer_cfg = dict(sub_cfg.get("trainer") or {})
    rollout_cfg = dict(sub_cfg.get("rollout") or {})
    actor_resource = resource_plan["trainer"]
    actor_gpu_ids = as_int_list(actor_resource["gpu_ids"])
    actor_tp = int(actor_resource.get("tensor_parallel_size") or actor_resource.get("n_gpus_per_node") or len(actor_gpu_ids))
    output_dir = stage_dir / "actor_model_verl"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_data_dir = stage_dir / "rollout_data"
    validation_data_dir = stage_dir / "validation_data"
    runtime_dir = stage_dir / "runtime"
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
    total_training_steps = trainer_cfg.get("total_training_steps")
    if total_training_steps is not None:
        total_training_steps = int(total_training_steps)
    teacher_request = dict(spad_cfg.get("teacher_answerer", {}).get("request") or {})
    teacher_request["endpoint"] = teacher_output["endpoint"]
    teacher_request["model"] = teacher_output["model"]
    reward_kwargs = {
        "reward_cfg": spad_cfg.get("reward", {}),
        "teacher_request": teacher_request,
        "visible_top_m": int(rollout_cfg.get("visible_top_m", 5)),
        "batch_workers": int(trainer_cfg.get("teacher_batch_workers", 16)),
    }
    reward_path = project_root() / "agentic_iter_rag" / "agent_training" / "spad" / "rewards" / "search_policy_teacher_reward.py"
    agent_loop_config = project_root() / "config" / "spad_search_policy_agent_loop_config.yaml"
    env_vars = {
        "SPAD_TEACHER_ENDPOINT": str(teacher_output["endpoint"]),
        "SPAD_TEACHER_MODEL": str(teacher_output["model"]),
        "SPAD_TEACHER_BATCH_WORKERS": str(trainer_cfg.get("teacher_batch_workers", 16)),
        "COSEARCH_ROLLOUT_PROGRESS_INTERVAL": str(trainer_cfg.get("rollout_progress_interval", 60)),
        "COSEARCH_ACTOR_PROGRESS_INTERVAL": str(trainer_cfg.get("actor_progress_interval", 8)),
    }
    hydra_overrides: list[str] = [
        _scalar_override("algorithm.adv_estimator", "grpo"),
        _scalar_override("algorithm.use_kl_in_reward", False),
        _scalar_override("critic.enable", False),
        _scalar_override("reward_model.enable", False),
        _scalar_override("+reward_model.use_reward_loop", True),
        _scalar_override("reward_model.reward_manager", str(trainer_cfg.get("reward_manager") or "naive")),
        _list_override("data.train_files", [str(item) for item in train_files]),
        _list_override("data.val_files", [str(item) for item in val_files]),
        _scalar_override("data.train_batch_size", train_batch_size),
        _scalar_override("data.val_batch_size", int(trainer_cfg.get("val_batch_size", config.get("data", {}).get("val_batch_size", 8)))),
        _scalar_override("data.train_max_samples", int(trainer_cfg.get("train_max_samples", config.get("data", {}).get("train_max_samples", -1)))),
        _scalar_override("data.val_max_samples", int(trainer_cfg.get("val_max_samples", config.get("data", {}).get("val_max_samples", 8)))),
        _scalar_override("data.max_prompt_length", max_prompt_length),
        _scalar_override("data.max_response_length", max_response_length),
        _scalar_override("data.truncation", str(config.get("data", {}).get("truncation", "error"))),
        _scalar_override("data.return_raw_chat", True),
        _scalar_override("data.trust_remote_code", True),
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
        _list_override("+actor_rollout_ref.rollout.stop", [str(item) for item in rollout_cfg.get("stop_sequences", ["<answer>"])]),
        _scalar_override("+actor_rollout_ref.rollout.include_stop_str_in_output", bool(rollout_cfg.get("include_stop_str_in_output", True))),
        _scalar_override("custom_reward_function.path", str(reward_path)),
        _scalar_override("custom_reward_function.name", "compute_spad_search_policy_reward_details"),
        _scalar_override("+custom_reward_function.reward_kwargs.visible_top_m", reward_kwargs["visible_top_m"]),
        _scalar_override("+custom_reward_function.reward_kwargs.batch_workers", reward_kwargs["batch_workers"]),
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
    resource_plan: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    sub_cfg = spad_cfg["sub_stages"]["search_policy_rl"]
    runtime_dir = stage_dir / "runtime"
    trainer_cfg = dict(sub_cfg.get("trainer") or {})
    manager = SpadServiceManager(runtime_dir=runtime_dir / "services", verl_root=Path(str(trainer_cfg.get("verl_root") or project_root() / "verl")))
    service_outputs: dict[str, Any] = {}
    try:
        teacher_resource = resource_plan.get("services", {}).get("teacher_answerer", {})
        recall_resource = resource_plan.get("services", {}).get("recall", {})
        if dry_run:
            service_outputs = {"teacher": {"status": "planned"}, "recall": {"status": "planned"}}
            teacher_output = {"endpoint": teacher_resource.get("endpoint", "http://127.0.0.1:8067/v1/chat/completions"), "model": teacher_resource.get("served_model_name", "GLM-4.7-Flash")}
            recall_output = {"retrieval_url": recall_resource.get("retrieval_service_url", "http://127.0.0.1:8130/retrieve")}
        else:
            service_outputs["teacher"] = manager.start_teacher(
                teacher_cfg=spad_cfg["teacher_answerer"],
                resource_cfg=teacher_resource,
            )
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
            status = "planned"
            run_info = None
        else:
            log_path = runtime_dir / "verl_train.log"
            run_info = _run_shell_script(script_path, log_path, timeout_s=trainer_cfg.get("timeout_seconds"))
            if int(run_info["return_code"]) != 0:
                raise RuntimeError(f"SPAD Stage 1 VERL exited with code {run_info['return_code']}; log={log_path}\n{tail_text(log_path)}")
            checkpoint = str(_find_latest_checkpoint(Path(plan["output_dir"])))
            status = "completed"
        outputs = {
            "status": status,
            "backend": "verl",
            "init_actor_model": init_actor_model(config, spad_cfg),
            "actor_checkpoint": checkpoint,
            "verl_command_plan": str(runtime_dir / "verl_command_plan.json"),
            "verl_launch_script": str(script_path),
            "runtime_dir": str(runtime_dir),
            "rollout_jsonl": str(Path(plan["rollout_data_dir"])),
            "reward_jsonl": str(Path(plan["rollout_data_dir"])),
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
    resource_plan: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Run Stage 1. The first implementation provides a smoke backend."""

    sub_cfg = spad_cfg["sub_stages"]["search_policy_rl"]
    backend = str(sub_cfg.get("backend") or spad_cfg.get("default_backend") or "smoke")
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = stage_dir / f"actor_checkpoint_{backend}"
    rollout_jsonl = stage_dir / "rollout_smoke.jsonl"
    reward_jsonl = stage_dir / "reward_smoke.jsonl"
    if backend == "verl":
        return _run_verl_backend(
            config=config,
            spad_cfg=spad_cfg,
            stage_dir=stage_dir,
            resource_plan=resource_plan,
            dry_run=dry_run,
        )
    if dry_run:
        outputs = {
            "status": "planned",
            "backend": backend,
            "actor_checkpoint": str(checkpoint_dir),
            "rollout_jsonl": str(rollout_jsonl),
            "reward_jsonl": str(reward_jsonl),
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

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "README.txt").write_text(
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
        "actor_checkpoint": str(checkpoint_dir),
        "rollout_jsonl": str(rollout_jsonl),
        "reward_jsonl": str(reward_jsonl),
        "resource_plan": resource_plan,
        "warning": "smoke backend does not update actor parameters",
    }
    outputs["manifest"] = str(stage_dir / "manifest.json")
    write_sub_stage_manifest(outputs["manifest"], sub_stage="search_policy_rl", outputs=outputs)
    return outputs

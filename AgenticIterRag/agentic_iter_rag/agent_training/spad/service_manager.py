"""Service manager for SPAD-RAG training sub-stages."""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def project_root() -> Path:
    return repo_root() / "AgenticIterRag"


def quote(value: Any) -> str:
    return shlex.quote(str(value))


def as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def csv_ids(value: Any) -> str:
    return ",".join(str(item) for item in as_int_list(value))


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])


def write_script(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o750)
    return path


def _as_dict_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{index}] must be a mapping")
        out.append(dict(item))
    return out


def validate_replica_config(*, service_name: str, service_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate Stage 2 replica service config and reject the legacy scalar path."""

    if "replicas" not in service_cfg:
        legacy_keys = sorted(set(service_cfg).intersection({"gpu_ids", "port", "endpoint"}))
        if legacy_keys:
            raise ValueError(
                f"SPAD Stage 2 {service_name} must use replicas; legacy scalar keys are not supported: {legacy_keys}"
            )
        raise ValueError(f"SPAD Stage 2 {service_name}.replicas must be set")
    replicas = _as_dict_list(service_cfg.get("replicas"), field_name=f"{service_name}.replicas")
    ports: set[int] = set()
    names: set[str] = set()
    for index, replica in enumerate(replicas):
        if replica.get("port") is None:
            raise ValueError(f"{service_name}.replicas[{index}].port must be set")
        port = int(replica["port"])
        if port in ports:
            raise ValueError(f"{service_name}.replicas has duplicate port: {port}")
        ports.add(port)
        gpu_ids = as_int_list(replica.get("gpu_ids"))
        if not gpu_ids:
            raise ValueError(f"{service_name}.replicas[{index}].gpu_ids must not be empty")
        tp = int(replica.get("tensor_parallel_size", len(gpu_ids)))
        if tp != len(gpu_ids):
            raise ValueError(
                f"{service_name}.replicas[{index}] tensor_parallel_size={tp} must equal gpu_ids count={len(gpu_ids)}"
            )
        name = str(replica.get("name") or replica.get("container_name") or "")
        if name:
            if name in names:
                raise ValueError(f"{service_name}.replicas has duplicate name/container_name: {name}")
            names.add(name)
    return replicas


def get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def vllm_batch_args(merged: dict[str, Any], *, indent: str) -> list[str]:
    args: list[str] = []
    if merged.get("max_num_seqs") is not None:
        args.append(f"{indent}--max-num-seqs {quote(merged['max_num_seqs'])} \\")
    if merged.get("max_num_batched_tokens") is not None:
        args.append(f"{indent}--max-num-batched-tokens {quote(merged['max_num_batched_tokens'])} \\")
    if bool(merged.get("enable_prefix_caching", False)):
        args.append(f"{indent}--enable-prefix-caching \\")
    if bool(merged.get("enable_chunked_prefill", False)):
        args.append(f"{indent}--enable-chunked-prefill \\")
    return args


def base_runtime_exports(verl_root: Path) -> list[str]:
    root = repo_root()
    air_accel = root / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "00_air_accelerator.sh"
    compat_python = root / "src" / "env_manage" / "compatible_python.sh"
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {quote(root)}",
        f"source {quote(compat_python)}",
        f"source {quote(air_accel)}",
        f"export PYTHONPATH={quote(verl_root)}:{quote(project_root())}:${{PYTHONPATH:-}}",
        "export TOKENIZERS_PARALLELISM=false",
        "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1",
    ]


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[Any]
    log_path: Path
    script_path: Path

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self, timeout_s: float = 20.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                return
            time.sleep(0.5)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        self.process.wait(timeout=5)


class SpadServiceManager:
    """Manage external services used by SPAD sub-stages."""

    def __init__(self, *, runtime_dir: Path, verl_root: Path) -> None:
        self.runtime_dir = runtime_dir
        self.verl_root = verl_root
        self.processes: list[ManagedProcess] = []
        self.containers: list[str] = []
        self._lock = threading.Lock()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def start_process(self, *, name: str, script_path: Path, log_path: Path) -> ManagedProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(repo_root()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        managed = ManagedProcess(name=name, process=process, log_path=log_path, script_path=script_path)
        with self._lock:
            self.processes.append(managed)
        return managed

    def wait_for_http_json(self, *, url: str, processes: list[ManagedProcess], timeout_s: float) -> None:
        started = time.time()
        while time.time() - started < timeout_s:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"service exited before ready: {proc.name}\nlog={proc.log_path}\n{tail_text(proc.log_path)}")
            try:
                if isinstance(get_json(url), dict):
                    return
            except Exception:
                time.sleep(2.0)
        logs = "\n\n".join(f"--- {proc.name}: {proc.log_path}\n{tail_text(proc.log_path)}" for proc in processes)
        raise TimeoutError(f"timed out waiting for {url}\n{logs}")

    def wait_for_http_post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        processes: list[ManagedProcess],
        timeout_s: float,
        request_timeout_s: float = 15.0,
    ) -> None:
        """Wait until a POST endpoint answers with JSON.

        The recall proxy /health endpoint can be healthy even when all backend
        retrievers are gone. A small retrieve preflight prevents training from
        starting against an empty proxy.
        """

        started = time.time()
        last_error = ""
        while time.time() - started < timeout_s:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"service exited before ready: {proc.name}\nlog={proc.log_path}\n{tail_text(proc.log_path)}")
            try:
                if isinstance(post_json(url, payload, timeout=request_timeout_s), dict):
                    return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2.0)
        logs = "\n\n".join(f"--- {proc.name}: {proc.log_path}\n{tail_text(proc.log_path)}" for proc in processes)
        raise TimeoutError(f"timed out waiting for POST {url}; last_error={last_error}\n{logs}")

    def recall_existing_ready(self, *, retrieval_url: str, backend_urls: list[str], top_n: int) -> tuple[bool, dict[str, Any] | None]:
        """Return whether an existing recall proxy and all its backends are usable."""

        try:
            health = get_json(retrieval_url.rsplit("/", 1)[0] + "/health")
            if not isinstance(health, dict) or health.get("status") != "ok":
                return False, health if isinstance(health, dict) else None
            if int(health.get("backend_count") or 0) != len(backend_urls):
                return False, health
            for backend_url in backend_urls:
                backend_health = get_json(backend_url.rsplit("/", 1)[0] + "/gpu_status")
                if not isinstance(backend_health, dict):
                    return False, health
            payload = {"queries": ["who got the first nobel prize in physics?"], "topk": min(max(int(top_n), 1), 3)}
            if not isinstance(post_json(retrieval_url, payload, timeout=15.0), dict):
                return False, health
            return True, health
        except Exception:
            return False, None

    def kill_tcp_ports(self, ports: list[int]) -> None:
        """Clear stale auto-started services on the configured ports."""

        for port in ports:
            subprocess.run(
                ["fuser", "-k", f"{int(port)}/tcp"],
                cwd=str(repo_root()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        time.sleep(1.0)

    def container_running(self, container_name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            cwd=str(repo_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def docker_logs_tail(self, container_name: str, max_lines: int = 120) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(max_lines), container_name],
            cwd=str(repo_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return result.stdout

    def wait_for_container_http_json(self, *, url: str, container_name: str, log_path: Path, timeout_s: float) -> None:
        started = time.time()
        while time.time() - started < timeout_s:
            try:
                if isinstance(get_json(url), dict):
                    return
            except Exception:
                pass
            if not self.container_running(container_name):
                raise RuntimeError(
                    f"container exited before ready: {container_name}\n"
                    f"log={log_path}\n{tail_text(log_path)}\n{self.docker_logs_tail(container_name)}"
                )
            time.sleep(2.0)
        raise TimeoutError(
            f"timed out waiting for {url}\ncontainer={container_name}\n"
            f"log={log_path}\n{tail_text(log_path)}\n{self.docker_logs_tail(container_name)}"
        )

    def start_teacher_container(self, *, profile_name: str, merged: dict[str, Any], port: int, endpoint: str) -> dict[str, Any]:
        wait_seconds = float(merged.get("wait_seconds") or 900)
        model_name = str(merged["served_model_name"])
        if not bool(merged.get("auto_start", True)):
            return {"status": "external", "endpoint": endpoint, "model": model_name}

        try:
            if isinstance(get_json(f"http://127.0.0.1:{port}/v1/models"), dict):
                return {"status": "existing", "endpoint": endpoint, "model": model_name, "profile": profile_name}
        except Exception:
            pass

        container_name = str(merged.get("container_name") or f"spad_{profile_name}_{port}")
        image = str(merged["container_image"])
        gpu_ids = csv_ids(merged["gpu_ids"])
        log_path = self.runtime_dir / f"teacher_{profile_name}_container_port{port}.log"
        script = write_script(
            self.runtime_dir / f"start_teacher_{profile_name}_container_port{port}.sh",
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"docker rm -f {quote(container_name)} >/dev/null 2>&1 || true",
                f"docker image inspect {quote(image)} >/dev/null 2>&1 || docker pull {quote(image)}",
                "docker run -d \\",
                f"  --name {quote(container_name)} \\",
                "  --privileged --net=host --ipc=host \\",
                f"  -e ASCEND_RT_VISIBLE_DEVICES={quote(gpu_ids)} \\",
                "  -e HF_HUB_OFFLINE=1 \\",
                "  -e TRANSFORMERS_OFFLINE=1 \\",
                "  -e VLLM_DISABLE_FLASHINFER=1 \\",
                "  -e VLLM_USE_FLASHINFER_SAMPLER=0 \\",
                "  -e VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN} \\",
                "  -e VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0} \\",
                "  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \\",
                "  -v /data01:/data01 \\",
                "  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\",
                "  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \\",
                f"  {quote(image)} \\",
                "  bash -lc \"exec vllm serve " + quote(merged["model_path"]) + " \\",
                f"    --served-model-name {quote(model_name)} \\",
                "    --host 0.0.0.0 \\",
                f"    --port {quote(port)} \\",
                "    --trust-remote-code \\",
                f"    --tensor-parallel-size {quote(merged.get('tensor_parallel_size', 1))} \\",
                f"    --gpu-memory-utilization {quote(merged.get('gpu_memory_utilization', 0.9))} \\",
                f"    --max-model-len {quote(merged.get('max_model_len', 32000))} \\",
                f"    --kv-cache-dtype {quote(merged.get('kv_cache_dtype', 'auto'))} \\",
                *vllm_batch_args(merged, indent="    "),
                *( [f"    --moe-backend {quote(merged['moe_backend'])} \\"] if merged.get("moe_backend") else [] ),
                "    --dtype bfloat16 \\",
                *(["    --enforce-eager \\"] if bool(merged.get("enforce_eager", False)) else []),
                "    --disable-custom-all-reduce\"",
            ],
        )
        start_log = self.runtime_dir / f"teacher_{profile_name}_container_start_port{port}.log"
        with start_log.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(["bash", str(script)], cwd=str(repo_root()), text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"failed to start teacher container: {container_name}\nlog={start_log}\n{tail_text(start_log)}")

        log_file = log_path.open("w", encoding="utf-8")
        log_proc = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            cwd=str(repo_root()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        with self._lock:
            self.processes.append(
                ManagedProcess(
                    name=f"spad-teacher-container-logs-{container_name}",
                    process=log_proc,
                    log_path=log_path,
                    script_path=script,
                )
            )
            self.containers.append(container_name)
        self.wait_for_container_http_json(
            url=f"http://127.0.0.1:{port}/v1/models",
            container_name=container_name,
            log_path=log_path,
            timeout_s=wait_seconds,
        )
        return {
            "status": "started",
            "backend_type": "vllm_container",
            "endpoint": endpoint,
            "model": model_name,
            "profile": profile_name,
            "gpu_ids": as_int_list(merged["gpu_ids"]),
            "container_name": container_name,
            "log": str(log_path),
        }

    def start_teacher(self, *, teacher_cfg: dict[str, Any], resource_cfg: dict[str, Any]) -> dict[str, Any]:
        profile_name = str(resource_cfg.get("profile") or teacher_cfg["default_service_profile"])
        profile = dict(teacher_cfg["service_profiles"][profile_name])
        merged = {**profile, **resource_cfg}
        port = int(merged.get("port") or str(merged["endpoint"]).rsplit(":", 1)[-1].split("/", 1)[0])
        gpu_ids = csv_ids(merged["gpu_ids"])
        endpoint = str(merged.get("endpoint") or f"http://127.0.0.1:{port}/v1/chat/completions")
        wait_seconds = float(merged.get("wait_seconds") or 900)
        backend_type = str(merged.get("backend_type") or "vllm_single")
        if backend_type == "vllm_container":
            return self.start_teacher_container(
                profile_name=profile_name,
                merged=merged,
                port=port,
                endpoint=endpoint,
            )
        if backend_type != "vllm_single":
            raise ValueError(f"unsupported SPAD teacher backend_type: {backend_type!r}")
        if not bool(merged.get("auto_start", True)):
            return {"status": "external", "endpoint": endpoint, "model": merged["served_model_name"]}

        script = write_script(
            self.runtime_dir / f"start_teacher_{profile_name}_port{port}.sh",
            base_runtime_exports(self.verl_root)
            + [
                f"export ASCEND_RT_VISIBLE_DEVICES={quote(gpu_ids)}",
                f"export CUDA_VISIBLE_DEVICES={quote(gpu_ids)}",
                "export VLLM_DISABLE_FLASHINFER=1",
                "export VLLM_USE_FLASHINFER_SAMPLER=0",
                "export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}",
                "export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}",
                "exec \"$PY\" -m vllm.entrypoints.openai.api_server \\",
                "  --host 127.0.0.1 \\",
                f"  --port {quote(port)} \\",
                f"  --model {quote(merged['model_path'])} \\",
                f"  --served-model-name {quote(merged['served_model_name'])} \\",
                f"  --tensor-parallel-size {quote(merged.get('tensor_parallel_size', 1))} \\",
                f"  --max-model-len {quote(merged.get('max_model_len', 32000))} \\",
                f"  --gpu-memory-utilization {quote(merged.get('gpu_memory_utilization', 0.9))} \\",
                f"  --kv-cache-dtype {quote(merged.get('kv_cache_dtype', 'auto'))} \\",
                *vllm_batch_args(merged, indent="  "),
                *( [f"  --moe-backend {quote(merged['moe_backend'])} \\"] if merged.get("moe_backend") else [] ),
                "  --trust-remote-code \\",
                "  --dtype bfloat16 \\",
                "  --enforce-eager" + (" \\" if bool(merged.get("disable_custom_all_reduce", False)) else ""),
                *(["  --disable-custom-all-reduce"] if bool(merged.get("disable_custom_all_reduce", False)) else []),
            ],
        )
        proc = self.start_process(
            name="spad-teacher-vllm",
            script_path=script,
            log_path=self.runtime_dir / f"teacher_{profile_name}_port{port}.log",
        )
        self.wait_for_http_json(url=f"http://127.0.0.1:{port}/v1/models", processes=[proc], timeout_s=wait_seconds)
        return {
            "status": "started",
            "endpoint": endpoint,
            "model": str(merged["served_model_name"]),
            "profile": profile_name,
            "gpu_ids": as_int_list(merged["gpu_ids"]),
            "log": str(proc.log_path),
        }

    def start_teacher_replicas(self, *, teacher_cfg: dict[str, Any], resource_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Start multiple OpenAI-compatible teacher services for Stage 2 Phase B."""

        replicas = validate_replica_config(service_name="teacher_answerer", service_cfg=resource_cfg)
        common = dict(resource_cfg.get("common") or {})
        outputs: list[dict[str, Any]] = []
        def start_one(index: int, replica: dict[str, Any]) -> dict[str, Any]:
            merged = {
                key: value
                for key, value in resource_cfg.items()
                if key not in {"replicas", "common", "gpu_ids", "port", "endpoint"}
            }
            merged.update(common)
            merged.update(replica)
            merged.setdefault("profile", resource_cfg.get("profile") or teacher_cfg["default_service_profile"])
            merged.setdefault("served_model_name", f"{teacher_cfg['service_profiles'][merged['profile']]['served_model_name']}-{index}")
            if merged.get("endpoint") is None:
                merged["endpoint"] = f"http://127.0.0.1:{int(merged['port'])}/v1/chat/completions"
            if merged.get("name") and merged.get("container_name") is None:
                merged["container_name"] = str(merged["name"])
            return self.start_teacher(teacher_cfg=teacher_cfg, resource_cfg=merged)
        with ThreadPoolExecutor(max_workers=len(replicas), thread_name_prefix="spad-teacher-start") as executor:
            futures = [executor.submit(start_one, index, replica) for index, replica in enumerate(replicas)]
            for future in as_completed(futures):
                outputs.append(future.result())
        return outputs

    def start_actor_vllm(self, *, actor_cfg: dict[str, Any], model_path: str) -> dict[str, Any]:
        """Start an OpenAI-compatible vLLM service for Stage 2 actor refresh."""

        port = int(actor_cfg["port"])
        endpoint = str(actor_cfg.get("endpoint") or f"http://127.0.0.1:{port}/v1/chat/completions")
        model_name = str(actor_cfg.get("served_model_name") or "spad-refresh-actor")
        if not bool(actor_cfg.get("auto_start", True)):
            return {"status": "external", "endpoint": endpoint, "model": model_name, "model_path": model_path}

        try:
            if isinstance(get_json(f"http://127.0.0.1:{port}/v1/models"), dict):
                return {"status": "existing", "endpoint": endpoint, "model": model_name, "model_path": model_path}
        except Exception:
            pass

        gpu_ids = csv_ids(actor_cfg["gpu_ids"])
        wait_seconds = float(actor_cfg.get("wait_seconds") or 600)
        script = write_script(
            self.runtime_dir / f"start_actor_vllm_port{port}.sh",
            base_runtime_exports(self.verl_root)
            + [
                f"export ASCEND_RT_VISIBLE_DEVICES={quote(gpu_ids)}",
                f"export CUDA_VISIBLE_DEVICES={quote(gpu_ids)}",
                "export VLLM_DISABLE_FLASHINFER=1",
                "export VLLM_USE_FLASHINFER_SAMPLER=0",
                "export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}",
                "export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}",
                "export VLLM_ALLREDUCE_USE_SYMM_MEM=0",
                "args=(--host 127.0.0.1)",
                f"args+=(--port {quote(port)})",
                f"args+=(--model {quote(model_path)})",
                f"args+=(--served-model-name {quote(model_name)})",
                f"args+=(--tensor-parallel-size {quote(actor_cfg.get('tensor_parallel_size', len(as_int_list(actor_cfg['gpu_ids']))))})",
                f"args+=(--max-model-len {quote(actor_cfg.get('max_model_len', 16096))})",
                f"args+=(--gpu-memory-utilization {quote(actor_cfg.get('gpu_memory_utilization', 0.6))})",
                f"args+=(--kv-cache-dtype {quote(actor_cfg.get('kv_cache_dtype', 'auto'))})",
                "args+=(--trust-remote-code)",
                "args+=(--dtype bfloat16)",
                *( [f"args+=(--max-num-seqs {quote(actor_cfg['max_num_seqs'])})"] if actor_cfg.get("max_num_seqs") is not None else [] ),
                *(
                    [f"args+=(--max-num-batched-tokens {quote(actor_cfg['max_num_batched_tokens'])})"]
                    if actor_cfg.get("max_num_batched_tokens") is not None
                    else []
                ),
                *(["args+=(--enable-prefix-caching)"] if bool(actor_cfg.get("enable_prefix_caching", False)) else []),
                *(["args+=(--enable-chunked-prefill)"] if bool(actor_cfg.get("enable_chunked_prefill", False)) else []),
                *(["args+=(--enforce-eager)"] if bool(actor_cfg.get("enforce_eager", True)) else []),
                *(["args+=(--disable-custom-all-reduce)"] if bool(actor_cfg.get("disable_custom_all_reduce", False)) else []),
                "exec \"$PY\" -m vllm.entrypoints.openai.api_server \"${args[@]}\"",
            ],
        )
        proc = self.start_process(
            name="spad-refresh-actor-vllm",
            script_path=script,
            log_path=self.runtime_dir / f"actor_vllm_port{port}.log",
        )
        self.wait_for_http_json(url=f"http://127.0.0.1:{port}/v1/models", processes=[proc], timeout_s=wait_seconds)
        return {
            "status": "started",
            "endpoint": endpoint,
            "model": model_name,
            "model_path": model_path,
            "gpu_ids": as_int_list(actor_cfg["gpu_ids"]),
            "log": str(proc.log_path),
        }

    def start_actor_vllm_replicas(self, *, actor_cfg: dict[str, Any], model_path: str) -> list[dict[str, Any]]:
        """Start multiple actor vLLM services for Stage 2 Phase A."""

        replicas = validate_replica_config(service_name="actor_vllm", service_cfg=actor_cfg)
        common = dict(actor_cfg.get("common") or {})
        outputs: list[dict[str, Any]] = []
        def start_one(index: int, replica: dict[str, Any]) -> dict[str, Any]:
            merged = {
                key: value
                for key, value in actor_cfg.items()
                if key not in {"replicas", "common", "gpu_ids", "port", "endpoint"}
            }
            merged.update(common)
            merged.update(replica)
            merged.setdefault("served_model_name", f"spad-refresh-actor-{index}")
            if merged.get("endpoint") is None:
                merged["endpoint"] = f"http://127.0.0.1:{int(merged['port'])}/v1/chat/completions"
            return self.start_actor_vllm(actor_cfg=merged, model_path=model_path)
        with ThreadPoolExecutor(max_workers=len(replicas), thread_name_prefix="spad-actor-start") as executor:
            futures = [executor.submit(start_one, index, replica) for index, replica in enumerate(replicas)]
            for future in as_completed(futures):
                outputs.append(future.result())
        return outputs

    def start_recall(self, *, recall_cfg: dict[str, Any], final_top_n: int, recall_model: str) -> dict[str, Any]:
        port = int(recall_cfg["port"])
        backend_base_port = int(recall_cfg.get("backend_base_port") or (port + 1))
        retrieval_url = str(recall_cfg.get("retrieval_service_url") or f"http://127.0.0.1:{port}/retrieve")
        wait_seconds = float(recall_cfg.get("wait_seconds") or 360)
        backend_type = str(recall_cfg.get("backend_type") or "npu").lower()
        accelerator_cfg = recall_cfg.get("accelerator_backend") if isinstance(recall_cfg.get("accelerator_backend"), dict) else {}
        gpu_ids = as_int_list(accelerator_cfg.get("gpu_ids", recall_cfg.get("gpu_ids")))
        if backend_type not in {"npu", "cuda"}:
            raise ValueError(f"SPAD Stage 1 recall backend_type must be npu/cuda, got {backend_type!r}")
        if not gpu_ids:
            raise ValueError("SPAD recall accelerator gpu_ids must not be empty")
        instance_count = int(recall_cfg.get("instance_count") or len(gpu_ids))
        backend_urls = [f"http://127.0.0.1:{backend_base_port + index}/retrieve" for index in range(instance_count)]
        existing_ready, health = self.recall_existing_ready(retrieval_url=retrieval_url, backend_urls=backend_urls, top_n=final_top_n)
        if existing_ready:
            return {
                "status": "existing",
                "retrieval_url": retrieval_url,
                "backend_urls": backend_urls,
                "gpu_ids": gpu_ids[:instance_count],
                "health": health,
            }
        if not bool(recall_cfg.get("auto_start", True)):
            raise RuntimeError(
                "SPAD recall auto_start is disabled, but the configured service is not fully usable: "
                f"retrieval_url={retrieval_url}, backend_urls={backend_urls}"
            )
        self.kill_tcp_ports([port, *[backend_base_port + index for index in range(instance_count)]])
        launcher = repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "00_start_dense_retriever_server.sh"
        started: list[ManagedProcess] = []
        for index, gpu_id in enumerate(gpu_ids[:instance_count]):
            backend_port = backend_base_port + index
            backend_url = f"http://127.0.0.1:{backend_port}/retrieve"
            script = write_script(
                self.runtime_dir / f"start_recall_{backend_type}{gpu_id}_port{backend_port}.sh",
                base_runtime_exports(self.verl_root)
                + [
                    f"export PORT={quote(backend_port)}",
                    f"export RECALL_GPU_ID={quote(gpu_id)}",
                    f"export RETRIEVER_GPU_IDS={quote(gpu_id)}",
                    f"export RETRIEVER_MODEL={quote(recall_model)}",
                    f"export RECALL_FINAL_TOP_N={quote(final_top_n)}",
                    f"export QUERY_BATCH_SIZE={quote(accelerator_cfg.get('query_batch_size', 32))}",
                    f"export DOC_DTYPE={quote(accelerator_cfg.get('doc_dtype', 'float16'))}",
                    f"export DEVICE={quote(backend_type)}",
                    "export SKIP_RETRIEVAL_ASSET_VERIFY=1",
                    f"exec bash {quote(launcher)}",
                ],
            )
            started.append(
                self.start_process(
                    name=f"spad-recall-{backend_type}{gpu_id}",
                    script_path=script,
                    log_path=self.runtime_dir / f"recall_{backend_type}{gpu_id}_port{backend_port}.log",
                )
            )

        for proc, backend_url in zip(started, backend_urls, strict=True):
            self.wait_for_http_json(
                url=backend_url.rsplit("/", 1)[0] + "/gpu_status",
                processes=[proc],
                timeout_s=wait_seconds,
            )

        proxy_cfg = recall_cfg.get("proxy") if isinstance(recall_cfg.get("proxy"), dict) else {}
        proxy_script = write_script(
            self.runtime_dir / f"start_recall_proxy_port{port}.sh",
            base_runtime_exports(self.verl_root)
            + [
                "args=(--host 127.0.0.1)",
                f"args+=(--port {quote(port)})",
                f"args+=(--timeout {quote(proxy_cfg.get('timeout', 180))})",
                f"args+=(--strategy {quote(proxy_cfg.get('strategy', 'least_inflight'))})",
                f"args+=(--failure-cooldown-seconds {quote(proxy_cfg.get('failure_cooldown_seconds', 10))})",
                f"args+=(--latency-ewma-alpha {quote(proxy_cfg.get('latency_ewma_alpha', 0.2))})",
                f"args+=(--max-retries-per-request {quote(proxy_cfg.get('max_retries_per_request', len(backend_urls)))})",
                *[f"args+=(--backend {quote(url)})" for url in backend_urls],
                f"exec \"$PY\" {quote(repo_root() / 'src' / 'retrievers' / 'retrieval_load_balancing_proxy.py')} \"${{args[@]}}\"",
            ],
        )
        proxy_proc = self.start_process(
            name="spad-recall-proxy",
            script_path=proxy_script,
            log_path=self.runtime_dir / f"recall_proxy_port{port}.log",
        )
        started.append(proxy_proc)
        self.wait_for_http_json(url=retrieval_url.rsplit("/", 1)[0] + "/health", processes=started, timeout_s=wait_seconds)
        self.wait_for_http_post_json(
            url=retrieval_url,
            payload={"queries": ["who got the first nobel prize in physics?"], "topk": min(max(int(final_top_n), 1), 3)},
            processes=started,
            timeout_s=wait_seconds,
        )
        return {
            "status": "started",
            "retrieval_url": retrieval_url,
            "backend_urls": backend_urls,
            "gpu_ids": gpu_ids[:instance_count],
            "logs": [str(proc.log_path) for proc in started],
        }

    def stop_all(self) -> None:
        with self._lock:
            processes = list(self.processes)
            containers = list(self.containers)
        for proc in reversed(processes):
            proc.terminate()
        for container_name in reversed(containers):
            subprocess.run(["docker", "rm", "-f", container_name], cwd=str(repo_root()), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

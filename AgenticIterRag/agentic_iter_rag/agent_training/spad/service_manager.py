"""Service manager for SPAD-RAG training sub-stages."""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import time
import urllib.request
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


def get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
                *( [f"    --moe-backend {quote(merged['moe_backend'])} \\"] if merged.get("moe_backend") else [] ),
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
                "exec \"$PY\" -m vllm.entrypoints.openai.api_server \\",
                "  --host 127.0.0.1 \\",
                f"  --port {quote(port)} \\",
                f"  --model {quote(model_path)} \\",
                f"  --served-model-name {quote(model_name)} \\",
                f"  --tensor-parallel-size {quote(actor_cfg.get('tensor_parallel_size', len(as_int_list(actor_cfg['gpu_ids']))))} \\",
                f"  --max-model-len {quote(actor_cfg.get('max_model_len', 16096))} \\",
                f"  --gpu-memory-utilization {quote(actor_cfg.get('gpu_memory_utilization', 0.6))} \\",
                f"  --kv-cache-dtype {quote(actor_cfg.get('kv_cache_dtype', 'auto'))} \\",
                "  --trust-remote-code \\",
                "  --dtype bfloat16 \\",
                "  --enforce-eager" + (" \\" if bool(actor_cfg.get("disable_custom_all_reduce", False)) else ""),
                *(["  --disable-custom-all-reduce"] if bool(actor_cfg.get("disable_custom_all_reduce", False)) else []),
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
        try:
            health = get_json(retrieval_url.rsplit("/", 1)[0] + "/health")
            if isinstance(health, dict) and health.get("status") == "ok":
                return {
                    "status": "existing",
                    "retrieval_url": retrieval_url,
                    "backend_urls": [f"http://127.0.0.1:{backend_base_port + index}/retrieve" for index in range(instance_count)],
                    "gpu_ids": gpu_ids[:instance_count],
                    "health": health,
                }
        except Exception:
            pass
        launcher = repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "00_start_dense_retriever_server.sh"
        backend_urls: list[str] = []
        started: list[ManagedProcess] = []
        for index, gpu_id in enumerate(gpu_ids[:instance_count]):
            backend_port = backend_base_port + index
            backend_url = f"http://127.0.0.1:{backend_port}/retrieve"
            backend_urls.append(backend_url)
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
        return {
            "status": "started",
            "retrieval_url": retrieval_url,
            "backend_urls": backend_urls,
            "gpu_ids": gpu_ids[:instance_count],
            "logs": [str(proc.log_path) for proc in started],
        }

    def stop_all(self) -> None:
        for proc in reversed(self.processes):
            proc.terminate()
        for container_name in reversed(self.containers):
            subprocess.run(["docker", "rm", "-f", container_name], cwd=str(repo_root()), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

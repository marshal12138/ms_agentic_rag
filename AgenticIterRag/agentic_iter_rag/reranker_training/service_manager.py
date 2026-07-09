"""AIR LLM reranker 训练阶段的外部服务编排。

这个模块只负责真实训练前后的服务生命周期：

1. 启动 frozen agent vLLM，用于 reranker reward 的 continuation rollout。
2. 启动 retriever-only recall 服务，用于 continuation 后续 search。
3. 等待 HTTP endpoint 就绪，并在训练结束后清理本次启动的进程组。

业务语义仍然放在 trainer_entry 和 continuation_reward 中；这里不计算 reward，也不改训练数据。
"""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


def quote(value: Any) -> str:
    return shlex.quote(str(value))


def as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def write_script(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o750)
    return path


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def base_runtime_exports(repo_root: Path, project_root: Path, verl_root: Path) -> list[str]:
    """生成所有服务脚本共享的环境初始化片段。

    这里显式 source AIR 现有 Python/Ascend 兼容脚本，原因是 pipeline launcher 本身不 source 这些脚本。
    如果省略这一步，子进程可能看不到 torch_npu、vLLM-Ascend 或 repo-local overlay 包。
    """

    air_accel = repo_root / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "00_air_accelerator.sh"
    compat_python = repo_root / "src" / "env_manage" / "compatible_python.sh"
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {quote(repo_root)}",
        f"source {quote(compat_python)}",
        f"source {quote(air_accel)}",
        # VERL 必须优先使用完整的 CoAgenticRetriever/verl；AIR 自身 package 仍用于 custom reward。
        f"export PYTHONPATH={quote(str(verl_root))}:{quote(str(project_root))}:${{PYTHONPATH:-}}",
        "export TOKENIZERS_PARALLELISM=false",
        "export AIR_ACCELERATOR=${AIR_ACCELERATOR:-npu}",
    ]


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[Any]
    log_path: Path
    script_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

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


class TrainingServiceManager:
    """管理 reranker GRPO 训练需要的 retriever 和 frozen agent 服务。"""

    def __init__(
        self,
        *,
        repo_root: Path,
        project_root: Path,
        verl_root: Path,
        runtime_dir: Path,
        config: dict[str, Any],
    ) -> None:
        self.repo_root = repo_root
        self.project_root = project_root
        self.verl_root = verl_root
        self.runtime_dir = runtime_dir
        self.config = config
        self.processes: list[ManagedProcess] = []
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def start_process(self, *, name: str, script_path: Path, log_path: Path) -> ManagedProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(self.repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        managed = ManagedProcess(name=name, process=process, log_path=log_path, script_path=script_path)
        self.processes.append(managed)
        return managed

    def wait_for_recall(
        self,
        *,
        url: str,
        processes: list[ManagedProcess],
        timeout_s: float,
        query: str,
        topk: int,
    ) -> None:
        started = time.time()
        payload = {"queries": [query], "topk": min(max(int(topk), 1), 3), "return_scores": False}
        while time.time() - started < timeout_s:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"recall process exited before ready: {proc.name}\n"
                        f"log={proc.log_path}\n{tail_text(proc.log_path)}"
                    )
            try:
                data = post_json(url, payload, timeout=5.0)
                if isinstance(data.get("result"), list):
                    return
            except Exception:
                time.sleep(2.0)
        logs = "\n\n".join(f"--- {proc.name}: {proc.log_path}\n{tail_text(proc.log_path)}" for proc in processes)
        raise TimeoutError(f"timed out waiting for recall url={url}\n{logs}")

    def wait_for_recall_health(
        self,
        *,
        url: str,
        processes: list[ManagedProcess],
        timeout_s: float,
    ) -> None:
        """等待 retriever backend 完成模型和向量加载。

        默认关闭 retrieval query 预检时，backend 用 /gpu_status 判断向量已加载，
        proxy 用 /health 判断 HTTP 服务已启动。
        """

        started = time.time()
        while time.time() - started < timeout_s:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"recall process exited before health ready: {proc.name}\n"
                        f"log={proc.log_path}\n{tail_text(proc.log_path)}"
                    )
            try:
                data = get_json(url, timeout=5.0)
                if isinstance(data.get("doc_embeddings_shape"), list) or data.get("status") == "ok":
                    return
            except Exception:
                time.sleep(2.0)
        logs = "\n\n".join(f"--- {proc.name}: {proc.log_path}\n{tail_text(proc.log_path)}" for proc in processes)
        raise TimeoutError(f"timed out waiting for recall health url={url}\n{logs}")

    def wait_for_vllm(self, *, port: int, process: ManagedProcess, timeout_s: float) -> None:
        url = f"http://127.0.0.1:{int(port)}/v1/models"
        started = time.time()
        while time.time() - started < timeout_s:
            if process.poll() is not None:
                raise RuntimeError(
                    f"frozen agent vLLM exited before ready: {process.name}\n"
                    f"log={process.log_path}\n{tail_text(process.log_path)}"
                )
            try:
                data = get_json(url, timeout=5.0)
                if isinstance(data, dict):
                    return
            except Exception:
                time.sleep(5.0)
        raise TimeoutError(f"timed out waiting for frozen agent vLLM port={port}\n{tail_text(process.log_path)}")

    def wait_for_frozen_agent_proxy(
        self,
        *,
        port: int,
        process: ManagedProcess,
        timeout_s: float,
    ) -> None:
        """等待 frozen-agent proxy 就绪。

        proxy 就绪只代表代理 HTTP 服务启动成功；后端 vLLM 实例已经在启动 proxy 前逐个预检过。
        """

        url = f"http://127.0.0.1:{int(port)}/health"
        started = time.time()
        while time.time() - started < timeout_s:
            if process.poll() is not None:
                raise RuntimeError(
                    f"frozen agent proxy exited before ready: {process.name}\n"
                    f"log={process.log_path}\n{tail_text(process.log_path)}"
                )
            try:
                data = get_json(url, timeout=5.0)
                if data.get("status") == "ok":
                    return
            except Exception:
                time.sleep(2.0)
        raise TimeoutError(f"timed out waiting for frozen agent proxy port={port}\n{tail_text(process.log_path)}")

    def start_recall(self, recall_cfg: dict[str, Any]) -> dict[str, Any]:
        """启动 retriever-only recall 服务。

        continuation 后续 search 必须只走 retriever，所以这里直接启动 dense retriever HTTP 服务，
        不挂任何 reranker 或 agent 逻辑。
        """

        port = int(recall_cfg["port"])
        backend_base_port = int(recall_cfg.get("backend_base_port") or (port + 1))
        retrieval_url = str(recall_cfg.get("retrieval_service_url") or f"http://127.0.0.1:{port}/retrieve")
        wait_seconds = float(recall_cfg.get("wait_seconds") or 360)
        asset_precheck = as_bool(recall_cfg.get("asset_precheck"), default=False)
        query_preflight = as_bool(recall_cfg.get("query_preflight"), default=False)
        preflight_query = str(recall_cfg.get("preflight_query") or "who got the first nobel prize in physics?")
        final_top_n = int(self.config["reranker_training"]["branch_dataset"]["candidate_top_n"])
        recall_model = str(self.config["infer_runtime"]["models"]["recall_model_path"])
        backend_type = str(recall_cfg.get("backend_type") or ("npu" if recall_cfg.get("gpu_ids") else "cpu")).lower()
        proxy_cfg = recall_cfg.get("proxy") if isinstance(recall_cfg.get("proxy"), dict) else {}
        proxy_strategy = str(proxy_cfg.get("strategy") or recall_cfg.get("proxy_strategy") or "least_inflight")
        proxy_timeout = float(proxy_cfg.get("timeout") or recall_cfg.get("proxy_timeout") or 180)
        proxy_failure_cooldown = float(
            proxy_cfg.get("failure_cooldown_seconds") or recall_cfg.get("proxy_failure_cooldown_seconds") or 10
        )
        proxy_latency_alpha = float(
            proxy_cfg.get("latency_ewma_alpha") or recall_cfg.get("proxy_latency_ewma_alpha") or 0.2
        )

        started: list[ManagedProcess] = []
        backend_urls: list[str] = []
        backend_processes: list[ManagedProcess] = []

        if backend_type == "cpu":
            # CPU 模式不读取 gpu_ids；所有 CPU 专属参数只从 cpu_backend 生效。
            cpu_cfg = recall_cfg.get("cpu_backend") if isinstance(recall_cfg.get("cpu_backend"), dict) else {}
            instance_count = int(recall_cfg.get("instance_count") or cpu_cfg.get("instance_count") or 8)
            if instance_count < 1:
                raise ValueError("train_llm_reranker recall.instance_count must be >= 1 for backend_type=cpu")
            cpu_threads = int(cpu_cfg.get("cpu_threads_per_instance") or 8)
            query_batch_size = int(cpu_cfg.get("query_batch_size") or 8)
            doc_dtype = str(cpu_cfg.get("doc_dtype") or "float32")
            launcher = (
                self.repo_root
                / "scripts"
                / "agenticIterRag_v1"
                / "assets"
                / "infer_backend"
                / "00_start_cpu_dense_retriever_server.sh"
            )
            for index in range(instance_count):
                backend_port = backend_base_port + index
                backend_url = f"http://127.0.0.1:{backend_port}/retrieve"
                backend_urls.append(backend_url)
                skip_asset_verify = 0 if asset_precheck and index == 0 else 1
                script = write_script(
                    self.runtime_dir / f"start_recall_cpu{index}_port{backend_port}.sh",
                    base_runtime_exports(self.repo_root, self.project_root, self.verl_root)
                    + [
                        f"export PORT={quote(backend_port)}",
                        f"export RETRIEVER_MODEL={quote(recall_model)}",
                        f"export RECALL_FINAL_TOP_N={quote(final_top_n)}",
                        f"export QUERY_BATCH_SIZE={quote(query_batch_size)}",
                        f"export CPU_THREADS_PER_INSTANCE={quote(cpu_threads)}",
                        f"export DOC_DTYPE={quote(doc_dtype)}",
                        f"export SKIP_RETRIEVAL_ASSET_VERIFY={skip_asset_verify}",
                        f"exec bash {quote(launcher)}",
                    ],
                )
                proc = self.start_process(
                    name=f"recall-cpu{index}",
                    script_path=script,
                    log_path=self.runtime_dir / f"recall_cpu{index}_port{backend_port}.log",
                )
                started.append(proc)
                backend_processes.append(proc)
        elif backend_type in {"npu", "cuda"}:
            # accelerator 模式只读取 accelerator_backend；旧 gpu_ids 字段仅作为迁移兜底。
            accelerator_cfg = (
                recall_cfg.get("accelerator_backend") if isinstance(recall_cfg.get("accelerator_backend"), dict) else {}
            )
            gpu_ids = as_int_list(accelerator_cfg.get("gpu_ids", recall_cfg.get("gpu_ids")))
            if not gpu_ids:
                raise ValueError(
                    "train_llm_reranker recall.accelerator_backend.gpu_ids must not be empty when backend_type is npu/cuda"
                )
            instance_count = int(recall_cfg.get("instance_count") or accelerator_cfg.get("instance_count") or len(gpu_ids))
            if instance_count < 1:
                raise ValueError("train_llm_reranker recall.instance_count must be >= 1")
            if instance_count > len(gpu_ids):
                raise ValueError(
                    "train_llm_reranker recall.instance_count cannot exceed accelerator_backend.gpu_ids count "
                    f"({instance_count} > {len(gpu_ids)})"
                )
            query_batch_size = int(accelerator_cfg.get("query_batch_size") or 32)
            doc_dtype = str(accelerator_cfg.get("doc_dtype") or "float16")
            launcher = (
                self.repo_root
                / "scripts"
                / "agenticIterRag_v1"
                / "assets"
                / "infer_backend"
                / "00_start_dense_retriever_server.sh"
            )
            for index, gpu_id in enumerate(gpu_ids[:instance_count]):
                backend_port = backend_base_port + index
                backend_url = f"http://127.0.0.1:{backend_port}/retrieve"
                backend_urls.append(backend_url)
                skip_asset_verify = 0 if asset_precheck and index == 0 else 1
                script = write_script(
                    self.runtime_dir / f"start_recall_{backend_type}{gpu_id}_port{backend_port}.sh",
                    base_runtime_exports(self.repo_root, self.project_root, self.verl_root)
                    + [
                        f"export PORT={quote(backend_port)}",
                        f"export RECALL_GPU_ID={quote(gpu_id)}",
                        f"export RETRIEVER_GPU_IDS={quote(gpu_id)}",
                        f"export RETRIEVER_MODEL={quote(recall_model)}",
                        f"export RECALL_FINAL_TOP_N={quote(final_top_n)}",
                        f"export QUERY_BATCH_SIZE={quote(query_batch_size)}",
                        f"export DOC_DTYPE={quote(doc_dtype)}",
                        f"export SKIP_RETRIEVAL_ASSET_VERIFY={skip_asset_verify}",
                        f"export DEVICE={quote(backend_type)}",
                        f"exec bash {quote(launcher)}",
                    ],
                )
                proc = self.start_process(
                    name=f"recall-{backend_type}{gpu_id}",
                    script_path=script,
                    log_path=self.runtime_dir / f"recall_{backend_type}{gpu_id}_port{backend_port}.log",
                )
                started.append(proc)
                backend_processes.append(proc)
        else:
            raise ValueError(f"unsupported train_llm_reranker recall.backend_type={backend_type!r}")

        if not backend_urls:
            raise RuntimeError("no recall backend urls were created")

        # backend 先并行启动。默认只用 /gpu_status 做轻量 ready 检查；
        # 如果显式打开 query_preflight，则只对第一个 backend 发真实 retrieval 预检，避免 8 实例重复做重查询。
        if query_preflight:
            self.wait_for_recall(
                url=backend_urls[0],
                processes=[backend_processes[0]],
                timeout_s=wait_seconds,
                query=preflight_query,
                topk=final_top_n,
            )
            health_start = 1
        else:
            health_start = 0
        for proc, backend_url in zip(backend_processes[health_start:], backend_urls[health_start:], strict=True):
            health_url = backend_url.rsplit("/", 1)[0] + "/gpu_status"
            self.wait_for_recall_health(
                url=health_url,
                processes=[proc],
                timeout_s=wait_seconds,
            )

        proxy_max_retries = int(proxy_cfg.get("max_retries_per_request") or recall_cfg.get("proxy_max_retries_per_request") or len(backend_urls))
        proxy_script = write_script(
            self.runtime_dir / f"start_recall_proxy_port{port}.sh",
            base_runtime_exports(self.repo_root, self.project_root, self.verl_root)
            + [
                "args=(--host 127.0.0.1)",
                f"args+=(--port {quote(port)})",
                f"args+=(--timeout {quote(proxy_timeout)})",
                f"args+=(--strategy {quote(proxy_strategy)})",
                f"args+=(--failure-cooldown-seconds {quote(proxy_failure_cooldown)})",
                f"args+=(--latency-ewma-alpha {quote(proxy_latency_alpha)})",
                f"args+=(--max-retries-per-request {quote(proxy_max_retries)})",
                *[f"args+=(--backend {quote(url)})" for url in backend_urls],
                f"exec \"$PY\" {quote(self.repo_root / 'src' / 'retrievers' / 'retrieval_load_balancing_proxy.py')} \"${{args[@]}}\"",
            ],
        )
        proxy_proc = self.start_process(
            name="recall-proxy",
            script_path=proxy_script,
            log_path=self.runtime_dir / f"recall_proxy_port{port}.log",
        )
        started.append(proxy_proc)

        if query_preflight:
            self.wait_for_recall(
                url=retrieval_url,
                processes=started,
                timeout_s=wait_seconds,
                query=preflight_query,
                topk=final_top_n,
            )
        else:
            self.wait_for_recall_health(
                url=retrieval_url.rsplit("/", 1)[0] + "/health",
                processes=started,
                timeout_s=wait_seconds,
            )
        return {
            "retrieval_url": retrieval_url,
            "backend_urls": backend_urls,
            "backend_type": backend_type,
            "instance_count": len(backend_urls),
            "proxy_strategy": proxy_strategy,
            "processes": [proc.name for proc in started],
            "logs": [str(proc.log_path) for proc in started],
            "asset_precheck": asset_precheck,
            "query_preflight": query_preflight,
        }

    def _start_single_frozen_agent_instance(
        self,
        agent_cfg: dict[str, Any],
        *,
        agent_model: Path,
        name: str,
        wait_ready: bool = True,
    ) -> dict[str, Any]:
        """启动单个 frozen agent vLLM 实例。

        多实例服务池和旧单实例服务都复用这个函数，避免两套 vLLM 启动参数漂移。
        """

        gpu_ids = csv_ids(agent_cfg["gpu_ids"])
        port = int(agent_cfg["port"])
        tp_size = int(agent_cfg.get("tensor_parallel_size") or 1)
        served_model = str(agent_cfg.get("served_model_name") or "agentic-iter-rag-frozen-agent")
        continuation = self.config["reranker_training"]["continuation"]
        infer_budget = self.config.get("infer_budget", {})
        vllm_budget = infer_budget.get("vllm", {}) if isinstance(infer_budget.get("vllm"), dict) else {}
        max_model_len = int(vllm_budget.get("max_model_len") or continuation.get("max_prompt_length") or 12288)
        max_num_seqs = int(
            agent_cfg.get("max_num_seqs")
            or vllm_budget.get("max_num_seqs")
            or self.config["reranker_training"]["trainer"].get("agent_max_num_seqs")
            or 16
        )
        gpu_memory_utilization = float(agent_cfg.get("gpu_memory_utilization") or vllm_budget.get("gpu_memory_utilization") or 0.6)
        startup_timeout = float(vllm_budget.get("startup_timeout") or 1800)

        script = write_script(
            self.runtime_dir / f"start_{name}_port{port}.sh",
            base_runtime_exports(self.repo_root, self.project_root, self.verl_root)
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
                f"  --model {quote(agent_model)} \\",
                f"  --served-model-name {quote(served_model)} \\",
                f"  --tensor-parallel-size {quote(tp_size)} \\",
                f"  --max-model-len {quote(max_model_len)} \\",
                f"  --max-num-seqs {quote(max_num_seqs)} \\",
                f"  --gpu-memory-utilization {quote(gpu_memory_utilization)} \\",
                "  --trust-remote-code \\",
                "  --dtype bfloat16 \\",
                "  --enforce-eager",
            ],
        )
        proc = self.start_process(
            name=name,
            script_path=script,
            log_path=self.runtime_dir / f"{name}_port{port}.log",
        )
        if wait_ready:
            self.wait_for_vllm(port=port, process=proc, timeout_s=startup_timeout)
        output = {
            "base_url": f"http://127.0.0.1:{port}",
            "served_model": served_model,
            "model": str(agent_model),
            "gpu_ids": as_int_list(agent_cfg["gpu_ids"]),
            "tensor_parallel_size": tp_size,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "process": proc.name,
            "log": str(proc.log_path),
        }
        # 多实例模式需要先并行启动全部进程，再逐个等待 ready；这两个内部字段不能写入 manifest。
        if not wait_ready:
            output["_managed_process"] = proc
            output["_startup_timeout"] = startup_timeout
        return output

    def start_frozen_agent(self, agent_cfg: dict[str, Any], *, agent_model: Path) -> dict[str, Any]:
        """启动 frozen agent vLLM 服务。

        支持两种模式：
        1. 旧单实例模式：一个 vLLM 服务直接暴露给 continuation reward。
        2. stage2 服务池模式：多个 TP1 vLLM 实例并行启动，前面挂 least-inflight proxy。
        """

        backend_type = str(agent_cfg.get("backend_type") or "single").lower()
        if backend_type != "multi_instance_proxy":
            return self._start_single_frozen_agent_instance(
                agent_cfg,
                agent_model=agent_model,
                name="frozen-agent-vllm",
            )

        served_model = str(agent_cfg.get("served_model_name") or "agentic-iter-rag-frozen-agent")
        proxy_cfg = agent_cfg.get("proxy") if isinstance(agent_cfg.get("proxy"), dict) else {}
        proxy_port = int(proxy_cfg.get("port") or agent_cfg.get("port") or 8140)
        proxy_host = str(proxy_cfg.get("host") or "127.0.0.1")
        proxy_strategy = str(proxy_cfg.get("strategy") or "least_inflight")
        proxy_timeout = float(proxy_cfg.get("timeout") or 300)
        proxy_failure_cooldown = float(proxy_cfg.get("failure_cooldown_seconds") or 10)
        proxy_latency_alpha = float(proxy_cfg.get("latency_ewma_alpha") or 0.2)
        proxy_max_retries = int(proxy_cfg.get("max_retries_per_request") or 3)
        startup_timeout = float(proxy_cfg.get("startup_timeout") or 1800)

        instances_cfg = agent_cfg.get("instances")
        if not isinstance(instances_cfg, list) or not instances_cfg:
            raise ValueError("frozen_agent_vllm.backend_type=multi_instance_proxy requires non-empty instances")

        instance_outputs: list[dict[str, Any]] = []
        for index, raw_instance in enumerate(instances_cfg):
            if not isinstance(raw_instance, dict):
                raise TypeError("frozen_agent_vllm.instances items must be mappings")
            instance_cfg = dict(raw_instance)
            instance_cfg.setdefault("served_model_name", served_model)
            instance_cfg.setdefault("tensor_parallel_size", 1)
            if int(instance_cfg["tensor_parallel_size"]) != 1:
                raise ValueError("frozen_agent_vllm multi_instance_proxy currently requires tensor_parallel_size=1")
            name = str(instance_cfg.get("name") or f"frozen-agent-vllm-{index}")
            # 三个实例互不共享 NPU，按进程并行能力启动；这里顺序写脚本、启动进程，ready 检查独立进行。
            instance_outputs.append(
                self._start_single_frozen_agent_instance(
                    instance_cfg,
                    agent_model=agent_model,
                    name=name,
                    wait_ready=False,
                )
            )
        for item in instance_outputs:
            self.wait_for_vllm(
                port=int(str(item["base_url"]).rsplit(":", 1)[-1]),
                process=item["_managed_process"],
                timeout_s=float(item["_startup_timeout"]),
            )
            item.pop("_managed_process", None)
            item.pop("_startup_timeout", None)

        backend_urls = [str(item["base_url"]) for item in instance_outputs]
        proxy_script = write_script(
            self.runtime_dir / f"start_frozen_agent_proxy_port{proxy_port}.sh",
            base_runtime_exports(self.repo_root, self.project_root, self.verl_root)
            + [
                "args=(--host 127.0.0.1)",
                f"args+=(--port {quote(proxy_port)})",
                f"args+=(--timeout {quote(proxy_timeout)})",
                f"args+=(--strategy {quote(proxy_strategy)})",
                f"args+=(--failure-cooldown-seconds {quote(proxy_failure_cooldown)})",
                f"args+=(--latency-ewma-alpha {quote(proxy_latency_alpha)})",
                f"args+=(--max-retries-per-request {quote(proxy_max_retries)})",
                *[f"args+=(--backend {quote(url)})" for url in backend_urls],
                f"exec \"$PY\" {quote(self.project_root / 'agentic_iter_rag' / 'reranker_training' / 'frozen_agent_proxy.py')} \"${{args[@]}}\"",
            ],
        )
        proxy_proc = self.start_process(
            name="frozen-agent-proxy",
            script_path=proxy_script,
            log_path=self.runtime_dir / f"frozen_agent_proxy_port{proxy_port}.log",
        )
        self.wait_for_frozen_agent_proxy(port=proxy_port, process=proxy_proc, timeout_s=startup_timeout)
        return {
            "status": "multi_instance_proxy",
            "base_url": f"http://{proxy_host}:{proxy_port}",
            "served_model": served_model,
            "model": str(agent_model),
            "backend_urls": backend_urls,
            "instances": instance_outputs,
            "proxy": {
                "host": proxy_host,
                "port": proxy_port,
                "strategy": proxy_strategy,
                "timeout": proxy_timeout,
                "failure_cooldown_seconds": proxy_failure_cooldown,
                "latency_ewma_alpha": proxy_latency_alpha,
                "process": proxy_proc.name,
                "log": str(proxy_proc.log_path),
            },
        }

    def stop_all(self) -> None:
        """清理本次训练启动的全部服务。

        倒序停止可以先关 proxy/vLLM，再关底层 retriever backend，减少 shutdown 期间的无效请求。
        """

        for proc in reversed(self.processes):
            proc.terminate()

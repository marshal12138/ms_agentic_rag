#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PIPELINE_DIR}/../../.." && pwd)"
DEFAULT_CONFIG="${PROJECT_ROOT}/CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml"
DEFAULT_LAUNCHER="${PROJECT_ROOT}/CoAgenticRetriever/scripts/launch_llm_as_judge.sh"

CLIENT_PYTHON="${LLM_JUDGE_CLIENT_PYTHON:-/data04/envs/ms/ms_cosearch_official/bin/python}"
CONFIG_PATH="${LLM_JUDGE_CONFIG:-${DEFAULT_CONFIG}}"
LAUNCHER="${LLM_JUDGE_LAUNCHER:-${DEFAULT_LAUNCHER}}"

usage() {
  cat <<'EOF'
Usage:
  llm_judge_dpskv4f.sh start [--config PATH] [--wait|--no-wait] [--timeout SEC] [--dry-run]
  llm_judge_dpskv4f.sh status [--config PATH]
  llm_judge_dpskv4f.sh stop [--config PATH] [--pid-file PATH]
  llm_judge_dpskv4f.sh call --input REQUEST.json [--output RESULT.json] [client options]
  llm_judge_dpskv4f.sh batch --input REQUESTS.jsonl --output RESULTS.jsonl [client options]
  llm_judge_dpskv4f.sh render-example --output REQUEST.json

Client options:
  --config PATH
  --endpoint URL
  --model NAME
  --prompt-path PATH
  --prompt-version NAME
  --max-chunk-chars N
  --output-mode no_think|think_high|think_max
  --temperature FLOAT
  --max-tokens N
  --timeout SEC
  --retries N
  --concurrency N          batch only
  --limit N                batch only; 0 means no limit
  --dry-run                render request payloads without calling the service

Input records may be native AsyncLabelRequest objects, rollout tool-call records
with rank_top50_docs/ranked_chunk_list, or chunk-ranking examples with
passage_list_top50.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

cfg_value() {
  local config="$1"
  local key="$2"
  local default="$3"
  if [[ ! -f "${config}" ]]; then
    printf '%s\n' "${default}"
    return 0
  fi
  "${CLIENT_PYTHON}" - "${config}" "${key}" "${default}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    value = OmegaConf.select(cfg, key)
except Exception:
    value = None

if value is None:
    print(default)
else:
    print(value)
PY
}

models_url_from_config() {
  local config="$1"
  "${CLIENT_PYTHON}" - "${config}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

path = Path(sys.argv[1])
endpoint = ""
port = "8067"
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    endpoint = str(OmegaConf.select(cfg, "server.endpoint") or "")
    port = str(OmegaConf.select(cfg, "server.port") or port)
except Exception:
    pass

if endpoint:
    parts = urlsplit(endpoint)
    print(urlunsplit((parts.scheme or "http", parts.netloc or f"127.0.0.1:{port}", "/v1/models", "", "")))
else:
    print(f"http://127.0.0.1:{port}/v1/models")
PY
}

pid_file_from_config() {
  local config="$1"
  local log_dir
  local pid_name
  log_dir="$(cfg_value "${config}" logs.log_dir "")"
  pid_name="$(cfg_value "${config}" logs.pid_file "vllm_gpu06_07_8067.pid")"
  if [[ -z "${log_dir}" || "${log_dir}" == "null" ]]; then
    log_dir="${PROJECT_ROOT}/log/llm_judge"
  fi
  printf '%s/%s\n' "${log_dir}" "${pid_name}"
}

service_ready() {
  local config="$1"
  local url
  url="$(models_url_from_config "${config}")"
  "${CLIENT_PYTHON}" - "${url}" <<'PY' >/dev/null 2>&1
from __future__ import annotations

import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as resp:
    raise SystemExit(0 if resp.status < 500 else 1)
PY
}

wait_ready() {
  local config="$1"
  local timeout="$2"
  local url
  local start_ts
  local now_ts
  url="$(models_url_from_config "${config}")"
  start_ts="$(date +%s)"
  while true; do
    if service_ready "${config}"; then
      echo "llm judge ready: ${url}"
      return 0
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout )); then
      echo "ERROR: timed out waiting for llm judge: ${url}" >&2
      return 1
    fi
    sleep 5
  done
}

start_service() {
  local config="${CONFIG_PATH}"
  local wait=1
  local timeout=900
  local dry_run=0
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --config)
        config="$2"
        shift 2
        ;;
      --wait)
        wait=1
        shift
        ;;
      --no-wait)
        wait=0
        shift
        ;;
      --timeout)
        timeout="$2"
        shift 2
        ;;
      --dry-run)
        dry_run=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown start argument: $1"
        ;;
    esac
  done
  [[ -x "${LAUNCHER}" ]] || die "launcher not executable: ${LAUNCHER}"
  [[ -f "${config}" ]] || die "config not found: ${config}"
  if [[ "${dry_run}" == "1" ]]; then
    bash "${LAUNCHER}" --config "${config}" --dry-run
    return 0
  fi
  bash "${LAUNCHER}" --config "${config}"
  if [[ "${wait}" == "1" ]]; then
    wait_ready "${config}" "${timeout}"
  fi
}

status_service() {
  local config="${CONFIG_PATH}"
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --config)
        config="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown status argument: $1"
        ;;
    esac
  done
  local url
  local pid_file
  url="$(models_url_from_config "${config}")"
  pid_file="$(pid_file_from_config "${config}")"
  if service_ready "${config}"; then
    echo "ready: ${url}"
  else
    echo "not ready: ${url}"
  fi
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "pid live: ${pid} (${pid_file})"
    else
      echo "pid stale: ${pid} (${pid_file})"
    fi
  else
    echo "pid file missing: ${pid_file}"
  fi
}

stop_service() {
  local config="${CONFIG_PATH}"
  local pid_file=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --config)
        config="$2"
        shift 2
        ;;
      --pid-file)
        pid_file="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown stop argument: $1"
        ;;
    esac
  done
  if [[ -z "${pid_file}" ]]; then
    pid_file="$(pid_file_from_config "${config}")"
  fi
  [[ -f "${pid_file}" ]] || die "pid file not found: ${pid_file}"
  local pid
  pid="$(cat "${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || die "invalid pid in ${pid_file}: ${pid}"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    if kill -TERM "-${pid}" >/dev/null 2>&1; then
      echo "sent SIGTERM to process group=${pid}"
    else
      kill -TERM "${pid}"
      echo "sent SIGTERM to pid=${pid}"
    fi
  else
    echo "pid already stopped: ${pid}"
  fi
}

run_client() {
  local mode="$1"
  shift
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${CLIENT_PYTHON}" - "${mode}" --project-root "${PROJECT_ROOT}" --config "${CONFIG_PATH}" "$@" <<'PY'
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from CoAgenticRetriever.async_labeling.schemas import AsyncLabelRequest, CandidateChunk
from CoAgenticRetriever.async_labeling.stages.llm_judge_rank50 import LLMJudgeRank50Stage


def load_service_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    except Exception:
        try:
            import yaml

            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}


def json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path, limit: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append((line_no, json.loads(line)))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def candidate_from_mapping(obj: dict[str, Any], fallback_rank: int) -> CandidateChunk:
    doc_id = str(
        obj.get("doc_id")
        or obj.get("id")
        or obj.get("passage_id")
        or obj.get("pid")
        or f"doc_{fallback_rank:03d}"
    )
    text = str(obj.get("text") or obj.get("contents") or obj.get("passage") or obj.get("snippet") or "")
    return CandidateChunk(
        doc_id=doc_id,
        text=text,
        title=obj.get("title"),
        recall_rank=obj.get("recall_rank"),
        recall_score=obj.get("recall_score"),
        rank_rank=obj.get("rank_rank") or obj.get("rank") or fallback_rank,
        rank_score=obj.get("rank_score") or obj.get("score"),
        metadata=dict(obj.get("metadata") or {}),
    )


def unwrap_record(obj: dict[str, Any]) -> dict[str, Any]:
    for key in ("request", "async_label_request", "tool_call"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return obj


def request_from_record(obj: dict[str, Any], *, line_no: int, prompt_version: str) -> AsyncLabelRequest:
    obj = unwrap_record(obj)
    ranked_docs = (
        obj.get("ranked_chunk_list")
        or obj.get("rank_top50_docs")
        or obj.get("passage_list_top50")
        or obj.get("passages")
        or obj.get("candidates")
    )
    if not isinstance(ranked_docs, list):
        raise ValueError("missing ranked_chunk_list/rank_top50_docs/passage_list_top50 list")
    chunks = [candidate_from_mapping(item, idx) for idx, item in enumerate(ranked_docs, start=1)]
    request = AsyncLabelRequest(
        request_id=str(obj.get("request_id") or obj.get("example_id") or f"line_{line_no}"),
        created_global_step=int(obj.get("created_global_step") or obj.get("global_step") or 0),
        origin_query=str(obj.get("origin_query") or obj.get("question") or ""),
        sub_query=str(obj.get("sub_query") or obj.get("query") or obj.get("origin_query") or obj.get("question") or ""),
        trajectory_id=str(obj.get("trajectory_id") or obj.get("uid") or f"line_{line_no}"),
        tool_call_id=str(obj.get("tool_call_id") or f"line_{line_no}:search:0"),
        ranked_chunk_list=chunks,
        turn_idx=int(obj.get("turn_idx") or 0),
        trajectory_score=float(obj.get("trajectory_score") or obj.get("reward") or 0.0),
        score_type=str(obj.get("score_type") or "unknown"),
        trace_metadata=dict(obj.get("metadata") or {}),
        label_policy=str(obj.get("label_policy") or "llm_judge_rank50"),
        prompt_version=str(obj.get("prompt_version") or prompt_version),
    )
    request.validate_rank50()
    return request


def build_stage(args: argparse.Namespace) -> LLMJudgeRank50Stage:
    cfg = load_service_config(args.config)
    server_cfg = dict(cfg.get("server") or {})
    model_cfg = dict(cfg.get("model") or {})
    endpoint = args.endpoint or server_cfg.get("endpoint")
    if not endpoint:
        endpoint = f"http://127.0.0.1:{server_cfg.get('port', 8067)}/v1/chat/completions"
    model = args.model or model_cfg.get("served_model_name") or "DeepSeek-V4-Flash"
    stage_cfg = {
        "type": "llm_as_judge",
        "endpoint": endpoint,
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "request_timeout_seconds": args.timeout,
        "max_retries": args.retries,
        "prompt": {
            "path": args.prompt_path,
            "version": args.prompt_version,
            "max_chunk_chars": args.max_chunk_chars,
            "output_mode": args.output_mode,
        },
    }
    return LLMJudgeRank50Stage(stage_cfg, project_root=args.project_root)


def result_from_stage(request: AsyncLabelRequest, stage: LLMJudgeRank50Stage, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        messages = stage.prompt.render_messages(request)
        return {
            "request_id": request.request_id,
            "ok": True,
            "dry_run": True,
            "payload": {
                "model": stage.model,
                "messages": messages,
                "temperature": stage.temperature,
                "max_tokens": stage.max_tokens,
                "chat_template_kwargs": dict(stage.chat_template_kwargs),
            },
        }
    result = stage.score(request)
    if not result.ok:
        return {
            "request_id": request.request_id,
            "ok": False,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "latency_ms": result.latency_ms,
        }
    scores = [asdict(score) for score in result.scores]
    return {
        "request_id": request.request_id,
        "ok": True,
        "ranked_ids": [score["doc_id"] for score in scores],
        "scores": scores,
        "raw_response": result.raw_response,
        "usage": result.usage,
        "latency_ms": result.latency_ms,
    }


def add_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--prompt-path", default="CoAgenticRetriever/async_labeling/prompts/llm_judge_rank50_v1.md")
    parser.add_argument("--prompt-version", default="llm_judge_rank50_v1")
    parser.add_argument("--max-chunk-chars", type=int, default=512)
    parser.add_argument("--output-mode", default="no_think", choices=["no_think", "think_high", "think_max"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")


def cmd_call(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="llm_judge_dpskv4f.sh call")
    add_client_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    stage = build_stage(args)
    request = request_from_record(load_json(args.input), line_no=1, prompt_version=args.prompt_version)
    result = result_from_stage(request, stage, dry_run=args.dry_run)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result.get("ok") else 1


def cmd_batch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="llm_judge_dpskv4f.sh batch")
    add_client_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    rows = iter_jsonl(args.input, args.limit)
    stage = build_stage(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run_one(row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        line_no, obj = row
        started = time.perf_counter()
        try:
            request = request_from_record(obj, line_no=line_no, prompt_version=args.prompt_version)
            result = result_from_stage(request, stage, dry_run=args.dry_run)
            result["line_no"] = line_no
            return result
        except Exception as exc:
            return {
                "line_no": line_no,
                "request_id": str(obj.get("request_id") or obj.get("example_id") or f"line_{line_no}"),
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }

    results: list[dict[str, Any] | None] = [None] * len(rows)
    worker_count = max(1, int(args.concurrency))
    if worker_count == 1:
        for idx, row in enumerate(rows):
            results[idx] = run_one(row)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_idx = {pool.submit(run_one, row): idx for idx, row in enumerate(rows)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

    ok_count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for result in results:
            assert result is not None
            ok_count += int(bool(result.get("ok")))
            f.write(json_dump(result) + "\n")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "count": len(rows),
        "ok_count": ok_count,
        "error_count": len(rows) - ok_count,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok_count == len(rows) else 1


def cmd_render_example(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="llm_judge_dpskv4f.sh render-example")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    obj = {
        "request_id": "example_001",
        "created_global_step": 0,
        "origin_query": "new york times vs united states who won?",
        "sub_query": "New York Times Co. v. United States case outcome",
        "trajectory_id": "example_trajectory_001",
        "tool_call_id": "example_trajectory_001:search:0",
        "ranked_chunk_list": [
            {
                "doc_id": f"doc_{idx:03d}",
                "title": "Example passage",
                "text": (
                    "Example evidence snippet "
                    f"{idx}. Replace this text with a real retrieved passage before judging."
                ),
                "rank_rank": idx,
                "rank_score": round(1.0 - idx / 100.0, 4),
            }
            for idx in range(1, 51)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(args.output))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("missing client mode", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    argv = sys.argv[2:]
    if mode == "call":
        return cmd_call(argv)
    if mode == "batch":
        return cmd_batch(argv)
    if mode == "render-example":
        return cmd_render_example(argv)
    print(f"unknown client mode: {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
PY
}

main() {
  local command="${1:-help}"
  if [[ "$#" -gt 0 ]]; then
    shift
  fi
  case "${command}" in
    start)
      start_service "$@"
      ;;
    status)
      status_service "$@"
      ;;
    stop)
      stop_service "$@"
      ;;
    call)
      run_client call "$@"
      ;;
    batch)
      run_client batch "$@"
      ;;
    render-example)
      run_client render-example "$@"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      die "unknown command: ${command}"
      ;;
  esac
}

main "$@"

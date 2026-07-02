#!/usr/bin/env python3
"""Recall retriever runtime preflight helper for the v2 train launcher.

这个模块服务于 `scripts/coagenticRetriever_v2/01_train_launcher.sh` 的运行期
recall retriever 检查。它的边界和 `compile_config.py` 不同：

- `compile_config.py` 只做配置编译，不访问外部 HTTP 服务。
- 本模块只做 recall retriever 预检，不合并 YAML、不生成 Hydra 参数。

职责范围：

- 校验 `RECALL_TOP_K` / `TOP_M` 这类预检参数。
- 用轻量 HTTP 请求判断 retriever endpoint 是否已经可用。
- 执行 semantic preflight，确认 endpoint 返回的候选文档数量和可见文档约束符合训练假设。

明确不负责：

- 不启动 `00_start_dense_retriever_server.sh`。
- 不管理 `RECALL_SERVICE_PID`。
- 不注册 Bash trap。
- 不 kill/wait 后台进程。
- 不 tail 服务日志。

这些有进程生命周期副作用的步骤继续留在 Bash launcher 中。这样 Python 负责可测试的
检查逻辑，Bash 负责它已经拥有的后台进程控制。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence


REWARD_PREFLIGHT_TOP_M_LIMIT = 5


@dataclass(frozen=True)
class RecallPreflightConfig:
    """一次 recall retriever semantic preflight 所需的全部输入。

    字段名刻意贴近 launcher 中的环境变量含义，便于从 Bash 调用时逐项对应：

    - `top_n` 对应 `RECALL_TOP_K`，即 retriever 返回候选数量。
    - `top_m` 对应 `TOP_M`，即 agent/reward 当前可见的文档数量。
    - `expect_contains` 是可选语义断言，留空时只检查数量和返回结构。
    """

    url: str
    query: str
    top_n: int
    top_m: int
    expect_contains: str = ""
    timeout: float = 120.0


def validate_recall_args(*, top_n: int, top_m: int) -> None:
    """校验 recall preflight 的数量参数。

    这里保留旧 Bash launcher 的约束：`TOP_M` 是 agent-visible documents，不是
    `ranker.top_k`，当前 reward preflight 最多只支持 5 篇可见文档。
    """

    if top_n < 1:
        raise ValueError(f"--top-n must be a positive integer; got {top_n}")
    if top_m < 1:
        raise ValueError(f"--top-m must be a positive integer; got {top_m}")
    if top_m > top_n:
        raise ValueError(f"--top-m {top_m} exceeds --top-n {top_n}")
    if top_m > REWARD_PREFLIGHT_TOP_M_LIMIT:
        raise ValueError(
            "--top-m exceeds current reward preflight limit of 5 visible documents; "
            "use agent-visible TOP_M here, not ranker.top_k/RANK_TOP_K."
        )


def _post_recall_request(*, url: str, query: str, topk: int, return_scores: bool, timeout: float) -> dict[str, Any]:
    """向 recall retriever 发送标准 `/retrieve` 请求并解析 JSON 响应。

    这里不假设服务实现细节，只依赖训练工具链当前使用的通用 HTTP contract：
    request 包含 `queries/topk/return_scores`，response 顶层包含 `result`。
    """

    payload = json.dumps(
        {
            "queries": [query],
            "topk": topk,
            "return_scores": return_scores,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status >= 500:
            raise RuntimeError(f"recall retriever returned HTTP {response.status}")
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("recall retriever response is not a JSON object")
    if "result" not in data:
        raise RuntimeError("recall retriever response is missing top-level 'result'")
    return data


def check_http_ready(*, url: str, query: str, timeout: float = 5.0) -> bool:
    """检查 recall HTTP endpoint 是否具备最小可用性。

    这个函数用于 Bash 的等待循环：失败只表示“现在还没 ready”，不是最终 fatal。
    因此它返回 bool，不打印错误。
    """

    try:
        _post_recall_request(url=url, query=query, topk=1, return_scores=False, timeout=timeout)
    except (OSError, urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def _extract_documents(data: dict[str, Any]) -> list[dict[str, Any]]:
    """把 retriever 原始响应规整成预检需要的文档列表。

    兼容两种候选格式：

    - `{document: {...}, score: ...}`
    - 直接返回 document mapping
    """

    raw_candidates = (data.get("result") or [[]])[0]
    if not isinstance(raw_candidates, list):
        raise RuntimeError("recall retriever result[0] is not a list")

    documents: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue
        doc = item.get("document", item)
        if not isinstance(doc, dict):
            doc = {}
        score = item.get("score", doc.get("score", 0.0))
        documents.append(
            {
                "rank": idx,
                "id": str(doc.get("id", "")),
                "title": str(doc.get("title", "")),
                "contents": str(doc.get("contents") or doc.get("text") or doc.get("passage") or ""),
                "score": float(score or 0.0),
            }
        )
    return documents


def run_semantic_preflight(config: RecallPreflightConfig) -> dict[str, Any]:
    """执行 semantic preflight，并返回可打印的简要结果。

    semantic preflight 比 HTTP ready 更严格：它要求返回数量等于 `top_n`，并在配置了
    `expect_contains` 时检查 agent-visible top_m 文本中包含期望子串。
    """

    validate_recall_args(top_n=config.top_n, top_m=config.top_m)
    data = _post_recall_request(
        url=config.url,
        query=config.query,
        topk=config.top_n,
        return_scores=True,
        timeout=config.timeout,
    )
    documents = _extract_documents(data)
    if len(documents) != config.top_n:
        raise RuntimeError(f"expected {config.top_n} recall docs, got {len(documents)}")

    visible_text = "\n".join(f"{doc['title']}\n{doc['contents']}" for doc in documents[: config.top_m])
    if config.expect_contains and config.expect_contains.lower() not in visible_text.lower():
        raise RuntimeError(f"expected substring not found in recall top-{config.top_m}: {config.expect_contains}")

    preview = visible_text.splitlines()[0][:160] if visible_text.splitlines() else ""
    return {
        "url": config.url,
        "query": config.query,
        "top_n": config.top_n,
        "top_m": config.top_m,
        "num_recall_docs": len(documents),
        "preview": preview,
    }


def _add_common_query_args(parser: argparse.ArgumentParser) -> None:
    """给需要访问 endpoint 的子命令添加通用参数。"""

    parser.add_argument("--url", required=True)
    parser.add_argument("--query", required=True)


def build_parser() -> argparse.ArgumentParser:
    """构建 recall preflight CLI。

    使用子命令是为了让 Bash 明确表达当前要做哪类检查：

    - `validate`：只校验参数。
    - `http-ready`：用于等待循环的轻量 endpoint 探活。
    - `semantic`：用于启动训练前的严格语义预检。
    """

    parser = argparse.ArgumentParser(description="Recall retriever runtime preflight helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate recall preflight numeric arguments")
    validate.add_argument("--top-n", type=int, required=True)
    validate.add_argument("--top-m", type=int, required=True)

    http_ready = subparsers.add_parser("http-ready", help="check whether recall HTTP endpoint is ready")
    _add_common_query_args(http_ready)
    http_ready.add_argument("--timeout", type=float, default=5.0)

    semantic = subparsers.add_parser("semantic", help="run strict semantic recall preflight")
    semantic.add_argument("--project-root", default="")
    _add_common_query_args(semantic)
    semantic.add_argument("--top-n", type=int, required=True)
    semantic.add_argument("--top-m", type=int, required=True)
    semantic.add_argument("--expect-contains", default="")
    semantic.add_argument("--timeout", type=float, default=120.0)

    return parser


def _print_semantic_success(result: dict[str, Any]) -> None:
    """保持旧 standalone 检查脚本的人类可读输出格式。"""

    print("CoAgentic retrieval verification passed.")
    print(f"  url:      {result['url']}")
    print(f"  query:    {result['query']}")
    print(f"  top_n:    {result['top_n']}")
    print(f"  top_m:    {result['top_m']}")
    print(
        "  metrics:  "
        + json.dumps({"num_recall_docs": result["num_recall_docs"]}, ensure_ascii=False, sort_keys=True)
    )
    print("  preview:  " + str(result["preview"]))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口，返回 shell 可直接使用的 exit code。"""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            validate_recall_args(top_n=args.top_n, top_m=args.top_m)
            return 0
        if args.command == "http-ready":
            return 0 if check_http_ready(url=args.url, query=args.query, timeout=args.timeout) else 1
        if args.command == "semantic":
            result = run_semantic_preflight(
                RecallPreflightConfig(
                    url=args.url,
                    query=args.query,
                    top_n=args.top_n,
                    top_m=args.top_m,
                    expect_contains=args.expect_contains,
                    timeout=args.timeout,
                )
            )
            _print_semantic_success(result)
            return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"ERROR: unsupported recall preflight command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

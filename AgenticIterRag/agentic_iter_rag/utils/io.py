"""Small JSON/YAML/JSONL helpers used by AgenticIterRag v1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False, width=4096)


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=False)
        f.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"JSONL line {line_no} must be an object: {path}")
            yield obj


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=True, sort_keys=False)
            f.write("\n")
            count += 1
    return count


def write_example(path: str | Path, record: dict[str, Any] | None) -> None:
    """写出单条 example.json；空数据时也写明原因，方便验收排查。"""

    payload = record if record is not None else {"_empty": True, "reason": "no records"}
    write_json(path, payload)


def copy_file(src: str | Path, dst: str | Path) -> None:
    """复制审计文件；用于把 final config 或 source manifest 固化到数据集目录。"""

    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def stable_config_hash(data: Any, length: int = 8) -> str:
    """根据配置内容生成稳定短 hash，用于自动版本命名。"""

    import hashlib

    text = json.dumps(data, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    cur = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

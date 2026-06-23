#!/usr/bin/env python3
"""Convert partial YAML config files into Hydra dotlist overrides."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

_HYDRA_DICT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _format_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _format_string(value)
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def _format_hydra_value(value: Any) -> str:
    value = _to_plain(value)
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"YAML mapping keys must be strings, got {key!r}")
            if not _HYDRA_DICT_KEY_RE.match(key):
                raise ValueError(f"YAML mapping key is not safe for Hydra dict syntax: {key!r}")
            parts.append(f"{key}:{_format_hydra_value(child)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_format_hydra_value(item) for item in value) + "]"
    return _format_scalar(value)


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    value = _to_plain(value)
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"YAML mapping keys must be strings, got {key!r}")
            child_key = f"{prefix}.{key}" if prefix else key
            items.extend(_flatten(child_key, child))
        return items
    return [(prefix, value)]


def yaml_to_overrides(path: Path, prefix: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"override YAML not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"override YAML is not a file: {path}")

    config = OmegaConf.load(path)
    plain = _to_plain(config)
    if plain is None:
        return []
    if not isinstance(plain, dict):
        raise TypeError(f"override YAML must contain a mapping at top level: {path}")
    if "defaults" in plain:
        raise ValueError(
            f"{path} contains a Hydra defaults section. "
            "This tool only supports partial value overrides, not config composition."
        )

    overrides = []
    for key, value in _flatten("", plain):
        if not key:
            continue
        overrides.append(f"{prefix}{key}={_format_hydra_value(value)}")
    return overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more partial YAML files into Hydra dotlist overrides. "
            "Files are emitted in the provided order, so later files can override earlier files."
        )
    )
    parser.add_argument("yaml_files", nargs="+", type=Path, help="Partial YAML override files.")
    parser.add_argument(
        "--prefix",
        default="++",
        help="Prefix for each Hydra override key. Use '++' by default to add or override keys.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        for yaml_file in args.yaml_files:
            for override in yaml_to_overrides(yaml_file, args.prefix):
                print(override)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

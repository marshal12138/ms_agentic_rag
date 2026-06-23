#!/usr/bin/env python3
"""Clean checkpoint roots so only retained trainable model steps remain.

This utility is intentionally conservative:
- It only reasons about `global_step_*` directories at checkpoint root level.
- It treats a step as trainable-model-bearing when at least one configured role
  contains recognizable model files.
- When requested, it deletes old `global_step_*` directories wholesale instead
  of leaving behind `data.pt` or other residue after trainer-side retention.
- It can also strip known legacy root-level directories/files such as a stale
  top-level `ranker/` or `retriever/`.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
from pathlib import Path


DEFAULT_TRAINABLE_ROLES = ("actor", "reranker_actor_rollout", "ranker", "retriever")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean retained checkpoint layout.")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--trainable-roles",
        nargs="+",
        default=list(DEFAULT_TRAINABLE_ROLES),
        help=f"Trainable roles used to detect valid global_step dirs. Default: {' '.join(DEFAULT_TRAINABLE_ROLES)}",
    )
    parser.add_argument(
        "--keep-latest-global-steps",
        type=int,
        default=1,
        help="Keep at most the latest N valid global_step dirs. Use 0 to keep all valid steps.",
    )
    parser.add_argument(
        "--delete-old-global-steps",
        action="store_true",
        help="Delete whole global_step_* directories older than the retained set.",
    )
    parser.add_argument(
        "--delete-empty-global-steps",
        action="store_true",
        help="Delete global_step_* directories that contain no recognized trainable model.",
    )
    parser.add_argument(
        "--remove-root-dirs",
        nargs="*",
        default=[],
        help="Exact root-level directory names to remove, excluding global_step_* dirs.",
    )
    parser.add_argument(
        "--remove-root-globs",
        nargs="*",
        default=[],
        help="Root-level file/dir glob patterns to remove, excluding global_step_* dirs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[checkpoint-cleanup] {message}", flush=True)


def step_number(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def collect_global_step_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and step_number(path) >= 0],
        key=step_number,
    )


def has_fsdp_model_shards(role_dir: Path) -> bool:
    return any(role_dir.glob("model_world_size_*_rank_*.pt"))


def has_ranker_checkpoint(role_dir: Path) -> bool:
    rank_encoder_dir = role_dir / "rank_encoder"
    if not rank_encoder_dir.is_dir():
        return False
    if not (rank_encoder_dir / "config.json").exists():
        return False
    if (rank_encoder_dir / "model.safetensors").exists():
        return True
    if (rank_encoder_dir / "pytorch_model.bin").exists():
        return True
    if (rank_encoder_dir / "model.safetensors.index.json").exists():
        return True
    if any(rank_encoder_dir.glob("*.safetensors")):
        return True
    return False


def step_has_trainable_model(step_dir: Path, roles: list[str]) -> bool:
    for role in roles:
        role_dir = step_dir / role
        if not role_dir.is_dir():
            continue
        if role in {"actor", "reranker_actor_rollout", "critic"}:
            if has_fsdp_model_shards(role_dir):
                return True
            continue
        if role in {"ranker", "retriever"}:
            if has_ranker_checkpoint(role_dir):
                return True
            continue
        if any(role_dir.iterdir()):
            return True
    return False


def delete_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        log(f"would delete: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    log(f"deleted: {path}")


def clean_global_steps(root: Path, args: argparse.Namespace) -> None:
    step_dirs = collect_global_step_dirs(root)
    valid_steps = [step for step in step_dirs if step_has_trainable_model(step, args.trainable_roles)]
    log(
        "global_step_dirs="
        f"{len(step_dirs)} valid_trainable_steps={len(valid_steps)} keep_latest={args.keep_latest_global_steps}"
    )

    keep_set: set[Path] = set()
    if valid_steps:
        if args.keep_latest_global_steps == 0:
            keep_set = set(valid_steps)
        else:
            keep_set = set(valid_steps[-args.keep_latest_global_steps :])

    if args.delete_old_global_steps and keep_set:
        for step_dir in step_dirs:
            if step_dir not in keep_set:
                delete_path(step_dir, args.dry_run)
    elif args.delete_old_global_steps and not keep_set:
        log("skip deleting old global_step dirs because no valid retained step was detected")

    if args.delete_empty_global_steps:
        for step_dir in collect_global_step_dirs(root):
            if not step_has_trainable_model(step_dir, args.trainable_roles):
                delete_path(step_dir, args.dry_run)


def clean_root_artifacts(root: Path, args: argparse.Namespace) -> None:
    exact_dirs = set(args.remove_root_dirs or [])
    patterns = list(args.remove_root_globs or [])
    for path in root.iterdir():
        if step_number(path) >= 0:
            continue
        if path.name in exact_dirs:
            delete_path(path, args.dry_run)
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            delete_path(path, args.dry_run)


def main() -> int:
    args = parse_args()
    if args.keep_latest_global_steps < 0:
        raise ValueError("--keep-latest-global-steps must be >= 0")

    root = args.checkpoint_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root not found: {root}")

    log(f"checkpoint_root={root}")
    clean_global_steps(root, args)
    clean_root_artifacts(root, args)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

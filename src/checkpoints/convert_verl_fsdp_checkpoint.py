#!/usr/bin/env python3
"""Convert retained VERL/FSDP checkpoints to HuggingFace safetensors.

The converter keeps the original VERL checkpoint layout intact:

  global_step_N/
    actor/
      model_world_size_*_rank_*.pt
      extra_state_world_size_*_rank_*.pt
      fsdp_config.json
      huggingface/

It writes evaluation-friendly HuggingFace output under:

  global_step_N/hf_safetensors/<role>/

It also removes global_step_* directories that do not contain any model shard
for the requested roles, which commonly happens when only data.pt remains after
VERL checkpoint retention.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ROLES = ("actor", "reranker_actor_rollout")


@dataclass(frozen=True)
class RoleCheckpoint:
    global_step_dir: Path
    role: str
    role_dir: Path
    world_size: int
    model_shards: tuple[Path, ...]
    target_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep VERL/FSDP shards and export HuggingFace safetensors for valid global_step checkpoints."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Directory containing global_step_* checkpoint directories.",
    )
    source.add_argument(
        "--global-step-dir",
        type=Path,
        help="Single global_step_* directory to convert.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=list(DEFAULT_ROLES),
        help=f"Checkpoint roles to export. Default: {' '.join(DEFAULT_ROLES)}",
    )
    parser.add_argument(
        "--target-subdir",
        default="hf_safetensors",
        help="Subdirectory created inside each global_step_* directory for HF outputs.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root used to discover local VERL package. Defaults to walking upward from this script.",
    )
    parser.add_argument(
        "--verl-root",
        type=Path,
        default=None,
        help="Directory that contains the local VERL package, e.g. <project>/verl. Defaults to auto discovery.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run local VERL model_merger.",
    )
    parser.add_argument(
        "--delete-empty",
        action="store_true",
        help="Delete global_step_* directories that contain no requested role model shards.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=1,
        help="Convert at most the latest N valid global_step directories. Use 0 to convert all valid directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing HF target directory before exporting.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass --trust-remote-code to VERL merger. Enabled by default.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Do not pass --trust-remote-code to VERL merger.",
    )
    parser.add_argument(
        "--use-cpu-initialization",
        action="store_true",
        default=True,
        help="Use CPU initialization when constructing the HF model. Enabled by default.",
    )
    parser.add_argument(
        "--no-use-cpu-initialization",
        action="store_false",
        dest="use_cpu_initialization",
        help="Do not pass --use_cpu_initialization to VERL merger.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cleanups and conversions without changing files.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[convert-verl-fsdp] {message}", flush=True)


def step_number(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def discover_project_root(script_path: Path) -> Path:
    for parent in script_path.resolve().parents:
        if (parent / "src").is_dir() and (
            (parent / "CoAgenticRetriever").is_dir()
            or (parent / "CoSearch").is_dir()
            or (parent / "AgenticIterRag").is_dir()
        ):
            return parent
    raise RuntimeError(f"Cannot discover project root from script path: {script_path}")


def discover_verl_root(project_root: Path) -> Path:
    candidates = [
        project_root / "CoAgenticRetriever" / "verl",
        project_root / "AgenticIterRag" / "verl",
        project_root / "CoSearch" / "verl",
    ]
    for candidate in candidates:
        if (candidate / "verl" / "model_merger" / "__main__.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find a local VERL model_merger package. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def collect_global_step_dirs(args: argparse.Namespace) -> list[Path]:
    if args.global_step_dir is not None:
        path = args.global_step_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"global step directory not found: {path}")
        return [path]

    root = args.checkpoint_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    dirs = [path for path in root.iterdir() if path.is_dir() and step_number(path) >= 0]
    return sorted(dirs, key=step_number)


def read_world_size(role_dir: Path) -> int | None:
    config_path = role_dir / "fsdp_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    world_size = config.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Invalid world_size in {config_path}: {world_size!r}")
    return world_size


def role_checkpoint(global_step_dir: Path, role: str, target_subdir: str) -> RoleCheckpoint | None:
    role_dir = global_step_dir / role
    if not role_dir.is_dir():
        return None
    world_size = read_world_size(role_dir)
    if world_size is None:
        return None
    shards = tuple(role_dir.glob(f"model_world_size_{world_size}_rank_*.pt"))
    if len(shards) != world_size:
        return None
    if not (role_dir / "huggingface" / "config.json").exists():
        return None
    return RoleCheckpoint(
        global_step_dir=global_step_dir,
        role=role,
        role_dir=role_dir,
        world_size=world_size,
        model_shards=tuple(sorted(shards)),
        target_dir=global_step_dir / target_subdir / role,
    )


def collect_role_checkpoints(
    global_step_dirs: Iterable[Path], roles: Iterable[str], target_subdir: str
) -> tuple[list[RoleCheckpoint], list[Path]]:
    valid: list[RoleCheckpoint] = []
    empty_global_steps: list[Path] = []
    for global_step_dir in global_step_dirs:
        found_for_step = [
            checkpoint
            for role in roles
            for checkpoint in [role_checkpoint(global_step_dir, role, target_subdir)]
            if checkpoint is not None
        ]
        if found_for_step:
            valid.extend(found_for_step)
        else:
            empty_global_steps.append(global_step_dir)
    return valid, empty_global_steps


def filter_latest_steps(checkpoints: list[RoleCheckpoint], keep: int) -> list[RoleCheckpoint]:
    if keep == 0:
        return checkpoints
    valid_steps = sorted({checkpoint.global_step_dir for checkpoint in checkpoints}, key=step_number)
    kept_steps = set(valid_steps[-keep:])
    return [checkpoint for checkpoint in checkpoints if checkpoint.global_step_dir in kept_steps]


def target_has_hf_safetensors(target_dir: Path) -> bool:
    if not target_dir.is_dir():
        return False
    has_model_index = (target_dir / "model.safetensors.index.json").exists()
    has_single_model = (target_dir / "model.safetensors").exists()
    has_safetensor_shard = any(target_dir.glob("*.safetensors"))
    has_config = (target_dir / "config.json").exists()
    has_tokenizer = (target_dir / "tokenizer.json").exists() or (target_dir / "tokenizer_config.json").exists()
    return (has_model_index or has_single_model) and has_safetensor_shard and has_config and has_tokenizer


def build_env(project_root: Path, verl_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(project_root), str(verl_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_merger(
    checkpoint: RoleCheckpoint,
    args: argparse.Namespace,
    project_root: Path,
    verl_root: Path,
) -> None:
    if target_has_hf_safetensors(checkpoint.target_dir) and not args.overwrite:
        log(f"skip existing HF safetensors: {checkpoint.target_dir}")
        return

    if checkpoint.target_dir.exists() and args.overwrite:
        if args.dry_run:
            log(f"would remove existing target: {checkpoint.target_dir}")
        else:
            shutil.rmtree(checkpoint.target_dir)

    command = [
        args.python,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint.role_dir),
        "--target_dir",
        str(checkpoint.target_dir),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.use_cpu_initialization:
        command.append("--use_cpu_initialization")

    if args.dry_run:
        log("would run: " + " ".join(command))
        return

    checkpoint.target_dir.parent.mkdir(parents=True, exist_ok=True)
    log(
        f"exporting {checkpoint.global_step_dir.name}/{checkpoint.role} "
        f"world_size={checkpoint.world_size} -> {checkpoint.target_dir}"
    )
    subprocess.run(command, cwd=str(project_root), env=build_env(project_root, verl_root), check=True)
    if not target_has_hf_safetensors(checkpoint.target_dir):
        raise RuntimeError(f"VERL merger finished but no safetensors were found in {checkpoint.target_dir}")
    write_export_metadata(checkpoint, project_root, verl_root)


def write_export_metadata(checkpoint: RoleCheckpoint, project_root: Path, verl_root: Path) -> None:
    metadata = {
        "source_global_step_dir": str(checkpoint.global_step_dir),
        "source_role_dir": str(checkpoint.role_dir),
        "role": checkpoint.role,
        "world_size": checkpoint.world_size,
        "model_shards": [path.name for path in checkpoint.model_shards],
        "project_root": str(project_root),
        "verl_root": str(verl_root),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = checkpoint.target_dir / "verl_fsdp_export_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def delete_empty_dirs(empty_dirs: Iterable[Path], dry_run: bool) -> None:
    for path in empty_dirs:
        if dry_run:
            log(f"would delete empty/non-model checkpoint dir: {path}")
        else:
            log(f"deleting empty/non-model checkpoint dir: {path}")
            shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    if args.keep < 0:
        raise ValueError("--keep must be >= 0")

    script_path = Path(__file__)
    project_root = (args.project_root.resolve() if args.project_root else discover_project_root(script_path))
    verl_root = args.verl_root.resolve() if args.verl_root else discover_verl_root(project_root)
    if not (verl_root / "verl" / "model_merger" / "__main__.py").exists():
        raise FileNotFoundError(f"Invalid VERL root: {verl_root}")
    log(f"project_root={project_root}")
    log(f"verl_root={verl_root}")

    global_step_dirs = collect_global_step_dirs(args)
    checkpoints, empty_dirs = collect_role_checkpoints(global_step_dirs, args.roles, args.target_subdir)
    selected = filter_latest_steps(checkpoints, args.keep)

    log(f"global_step_dirs={len(global_step_dirs)} valid_role_checkpoints={len(checkpoints)} selected={len(selected)}")
    if args.delete_empty:
        delete_empty_dirs(empty_dirs, args.dry_run)
    elif empty_dirs:
        log(f"empty/non-model checkpoint dirs retained: {len(empty_dirs)}; pass --delete-empty to remove them")

    if not selected:
        log("no valid role checkpoints selected")
        return 0

    for checkpoint in selected:
        run_merger(checkpoint, args, project_root, verl_root)

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

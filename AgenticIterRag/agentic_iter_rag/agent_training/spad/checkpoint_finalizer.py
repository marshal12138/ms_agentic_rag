"""Shared validation and HF export for SPAD VERL actor checkpoints."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.service_manager import project_root, repo_root, tail_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hf_actor_checkpoint(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    weights = sorted(path.glob("model*.safetensors"))
    tokenizer_files = [path / "tokenizer.json", path / "tokenizer_config.json"]
    if not config_path.is_file() or not weights or not all(item.is_file() for item in tokenizer_files):
        raise ValueError(f"incomplete Hugging Face actor checkpoint: {path}")
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    return {
        "model_type": str(config.model_type),
        "config_sha256": _sha256(config_path),
        "weight_files": [str(item) for item in weights],
        "weight_sha256": {item.name: _sha256(item) for item in weights},
    }


def finalize_actor_checkpoint(
    raw_checkpoint: str | Path,
    *,
    hf_root: Path,
    log_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    raw_path = Path(raw_checkpoint)
    target_dir = hf_root / raw_path.name
    log_path = log_dir / "merge_actor_checkpoint.log"
    if dry_run:
        return {
            "status": "planned",
            "raw_actor_checkpoint": str(raw_path),
            "hf_actor_checkpoint": str(target_dir),
            "conversion_log": str(log_path),
        }
    if not raw_path.exists():
        raise ValueError(f"raw actor checkpoint does not exist: {raw_path}")
    if (raw_path / "config.json").is_file():
        validation = validate_hf_actor_checkpoint(raw_path)
        return {
            "status": "validated_existing_hf",
            "raw_actor_checkpoint": str(raw_path),
            "hf_actor_checkpoint": str(raw_path),
            "conversion_log": "",
            "validation": validation,
        }
    if target_dir.exists():
        validation = validate_hf_actor_checkpoint(target_dir)
        return {
            "status": "reused_valid_hf",
            "raw_actor_checkpoint": str(raw_path),
            "hf_actor_checkpoint": str(target_dir),
            "conversion_log": str(log_path),
            "validation": validation,
        }

    actor_dir = raw_path / "actor" if raw_path.name.startswith("global_step_") else raw_path
    if not (actor_dir / "fsdp_config.json").is_file():
        raise ValueError(f"unsupported VERL actor checkpoint layout: {raw_path}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid.uuid4().hex}"
    cmd = [
        str(repo_root() / ".venvs" / "ms_agt_rag_overlay" / "bin" / "python"),
        str(project_root() / "verl" / "scripts" / "legacy_model_merger.py"),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(temporary_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root() / 'verl'}:{project_root()}:{env.get('PYTHONPATH', '')}"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                cwd=str(repo_root()),
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"actor checkpoint merge failed; log={log_path}\n{tail_text(log_path)}")
        validation = validate_hf_actor_checkpoint(temporary_dir)
        temporary_dir.rename(target_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return {
        "status": "converted",
        "raw_actor_checkpoint": str(raw_path),
        "hf_actor_checkpoint": str(target_dir),
        "conversion_log": str(log_path),
        "validation": validation,
    }

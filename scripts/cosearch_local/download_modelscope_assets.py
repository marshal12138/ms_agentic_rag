#!/usr/bin/env python3
"""Download or link assets needed for the local CoSearch reproduction.

The environment cannot rely on HuggingFace. This script uses local assets first
and ModelScope downloads only when a target directory is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copied"
    os.symlink(src, dst, target_is_directory=src.is_dir())
    return "symlinked"


def modelscope_download(model_id: str, local_dir: Path) -> str:
    from modelscope import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(model_id, local_dir=str(local_dir))


def looks_like_hf_model_dir(path: Path) -> bool:
    if not path.exists():
        return False
    has_config = (path / "config.json").exists()
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "vocab.txt", "vocab.json"))
    has_weights = any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    return has_config and has_tokenizer and has_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("/data01/ms_wksp/agent_up_to_date/models/llm"))
    parser.add_argument(
        "--retriever-models-root",
        type=Path,
        default=Path("/data01/ms_wksp/agent_up_to_date/models/retriever"),
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("/data01/ms_wksp/agent_up_to_date/models"))
    parser.add_argument("--copy-local", action="store_true")
    parser.add_argument(
        "--local-qwen3",
        type=Path,
        default=Path("/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-0.6B"),
    )
    parser.add_argument("--qwen3-modelscope-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--e5-modelscope-id", default="AI-ModelScope/e5-base-v2")
    parser.add_argument("--download-e5", action="store_true")
    args = parser.parse_args()

    report = {}
    qwen_dst = args.models_root / "Qwen3-0.6B"
    if args.local_qwen3.exists():
        report["qwen3_0_6b"] = {
            "status": link_or_copy(args.local_qwen3, qwen_dst, args.copy_local),
            "path": str(qwen_dst),
            "source": str(args.local_qwen3),
        }
    else:
        path = modelscope_download(args.qwen3_modelscope_id, qwen_dst)
        report["qwen3_0_6b"] = {"status": "downloaded", "path": str(path), "source": args.qwen3_modelscope_id}

    if args.download_e5:
        e5_dst = args.retriever_models_root / "e5-base-v2"
        if looks_like_hf_model_dir(e5_dst):
            report["e5_base_v2"] = {"status": "exists", "path": str(e5_dst), "source": "local"}
        else:
            path = modelscope_download(args.e5_modelscope_id, e5_dst)
            report["e5_base_v2"] = {"status": "downloaded", "path": str(path), "source": args.e5_modelscope_id}
    else:
        report["e5_base_v2"] = {
            "status": "not_downloaded",
            "note": "Pass --download-e5 to download via ModelScope; local smoke scripts can use BM25 if E5 is absent.",
        }

    report_path = args.manifest_dir / "asset_manifest.cosearch.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

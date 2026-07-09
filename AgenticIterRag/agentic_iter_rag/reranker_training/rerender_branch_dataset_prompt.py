"""Re-render prompts in an existing AIR branch dataset.

This keeps the candidate pool, doc-id mapping, targets, and extra continuation
context unchanged. Only the model-visible prompt text and prompt version fields
are replaced. It is useful when a reward-bound or training diagnosis shows that
the prompt protocol, rather than the data slice, needs to change.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_iter_rag.llm_reranker.format import render_air_rerank_tags_prompt
from agentic_iter_rag.reranker_training.branch_dataset import write_dataset_readme, write_parquet
from agentic_iter_rag.utils.io import copy_file, iter_jsonl, read_json, stable_config_hash, write_example, write_json, write_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists and overwrite=false: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_version_dir(base_dir: Path, version: str, overwrite: bool) -> tuple[str, Path]:
    if overwrite or not (base_dir / version).exists():
        return version, base_dir / version
    for idx in range(1, 26):
        candidate = f"{version}_{chr(ord('a') + idx)}"
        if not (base_dir / candidate).exists():
            return candidate, base_dir / candidate
    raise FileExistsError(f"cannot find free version dir for {version}")


def rerender_row(row: dict[str, Any], *, prompt_version: str, max_doc_chars: int) -> dict[str, Any]:
    out = dict(row)
    extra = dict(out.get("extra_info") or {})
    docs = list(extra.get("candidate_docs") or [])
    visible_top_m = int(extra.get("visible_top_m") or 5)
    prompt, index_to_doc_id = render_air_rerank_tags_prompt(
        initial_query=str(extra.get("question") or ""),
        sub_query=str(extra.get("sub_query") or ""),
        docs=docs,
        top_m=visible_top_m,
        max_doc_chars=max_doc_chars,
    )
    out["prompt"] = prompt
    out["prompt_template_version"] = prompt_version
    extra["prompt_template_version"] = prompt_version
    extra["candidate_index_to_doc_id"] = index_to_doc_id
    out["extra_info"] = extra
    return out


def build_rerendered_dataset(
    *,
    source_manifest_path: Path,
    out_root: Path | None,
    out_version: str,
    prompt_version: str,
    max_doc_chars: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_manifest = read_json(source_manifest_path)
    if source_manifest.get("schema_version") != "air_reranker_branch_dataset_v1":
        raise ValueError("source manifest schema_version must be air_reranker_branch_dataset_v1")
    data_path = Path(str(source_manifest["dataset_jsonl"]))
    base_dir = out_root or source_manifest_path.parent.parent
    version, out_dir = resolve_version_dir(base_dir, out_version, overwrite)
    ensure_output_dir(out_dir, overwrite)

    rows = [
        rerender_row(row, prompt_version=prompt_version, max_doc_chars=max_doc_chars)
        for row in iter_jsonl(data_path)
    ]
    if not rows:
        raise ValueError(f"source dataset is empty: {data_path}")

    dataset_jsonl = out_dir / "dataset.jsonl"
    dataset_parquet = out_dir / "dataset.parquet"
    manifest_path = out_dir / "manifest.json"
    example_json = out_dir / "example.json"
    source_snapshot = out_dir / "source_branch_dataset.manifest.json"

    count = write_jsonl(dataset_jsonl, rows)
    parquet_written = write_parquet(dataset_parquet, rows)
    write_example(example_json, rows[0])
    copy_file(source_manifest_path, source_snapshot)
    readme = write_dataset_readme(out_dir)

    rerender_cfg = {
        "source": str(source_manifest_path),
        "prompt_template_version": prompt_version,
        "max_doc_chars": max_doc_chars,
    }
    manifest = dict(source_manifest)
    manifest.update(
        {
            "version": version,
            "version_dir": str(out_dir),
            "created_at": utc_now(),
            "source_branch_dataset_manifest": str(source_manifest_path),
            "source_branch_dataset_version": source_manifest.get("version"),
            "rerender_prompt": rerender_cfg,
            "prompt_template_version": prompt_version,
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_parquet": str(dataset_parquet) if parquet_written else None,
            "example_json": str(example_json),
            "readme": str(readme),
            "sample_count": count,
            "config_hash": stable_config_hash(rerender_cfg),
        }
    )
    write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "manifest": str(manifest_path),
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_parquet": str(dataset_parquet) if parquet_written else None,
        "version": version,
        "sample_count": count,
        "prompt_template_version": prompt_version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render prompts in an AIR branch dataset.")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--out-version", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--max-doc-chars", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_rerendered_dataset(
        source_manifest_path=args.source_manifest,
        out_root=args.out_root,
        out_version=args.out_version,
        prompt_version=args.prompt_version,
        max_doc_chars=args.max_doc_chars,
        overwrite=args.overwrite,
    )
    print(outputs)


if __name__ == "__main__":
    main()

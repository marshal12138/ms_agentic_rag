"""Entry point for the alpha-fusion validation pipeline.

Run from the repo root, e.g.:

    python -m AgenticDynamicRecallRag.method_validation_check.main \
        --llm-endpoint http://127.0.0.1:8067/v1/chat/completions \
        --llm-model DeepSeek-V4-Flash \
        --retriever-endpoint http://localhost:9011/retrieve \
        --num-samples 100

Writes a JSONL result file (one row per valid sample, original fields plus
generated_query and alpha_<v> position arrays) and a PNG report.
"""
import json
import sys
import time

from .config import parse_args
from .runner import Pipeline
from .visualize import build_report, summary_stats


def main(argv=None) -> int:
    config = parse_args(argv)

    print(
        f"[validate] dataset={config.dataset_path}\n"
        f"[validate] corpus={config.corpus_path}\n"
        f"[validate] llm={config.llm_model} @ {config.llm_endpoint}\n"
        f"[validate] retriever={config.retriever_endpoint}\n"
        f"[validate] alphas={[f'{a:g}' for a in config.alphas]} topk={config.topk} "
        f"target_samples={config.num_samples} workers={config.workers}",
        flush=True,
    )

    pipeline = Pipeline(config)

    started = time.time()

    def on_progress(collected: int, scanned: int) -> None:
        print(
            f"\r[validate] valid={collected}/{config.num_samples} scanned={scanned}",
            end="",
            flush=True,
        )

    rows = pipeline.run(on_progress=on_progress)
    print(flush=True)
    elapsed = time.time() - started

    with open(config.output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = summary_stats(rows, config)
    build_report(rows, config, config.report_path)

    print(
        f"[validate] collected {len(rows)} valid samples in {elapsed:.1f}s",
        flush=True,
    )
    print(f"[validate] results -> {config.output_path}", flush=True)
    print(f"[validate] report  -> {config.report_path}", flush=True)
    print("[validate] per-alpha summary (alpha = bm25 weight):", flush=True)
    for key, s in stats.items():
        mean_pos = s["mean_position"]
        mean_str = f"{mean_pos:.2f}" if mean_pos is not None else "n/a"
        print(
            f"  {key:>12}  hit_rate={s['hit_rate']:.3f}  "
            f"mean_pos={mean_str}  pushed_out={s['pushed_out']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

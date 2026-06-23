#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
from pathlib import Path


ROOT = Path("/data01/ms_wksp/agent_up_to_date")
MODEL = ROOT / "models/llm/DeepSeek-V4-Flash"
DATA = ROOT / "CoSearch_derevitives/data/llm_judge/chunk_ranking/examples/chunk_ranking_judge_examples_100.jsonl"


def main() -> None:
    print("model", MODEL, "exists", MODEL.exists(), "is_dir", MODEL.is_dir())
    if MODEL.is_dir():
        print("model_children", [p.name for p in list(MODEL.iterdir())[:40]])
    print("data", DATA, "exists", DATA.exists(), "is_file", DATA.is_file())
    if DATA.is_file():
        with DATA.open("r", encoding="utf-8") as f:
            for i, line in zip(range(2), f):
                obj = json.loads(line)
                print("row", i, "keys", list(obj.keys()))
                for key in ("origin_query", "sub_query", "passage_list_top50"):
                    value = obj.get(key)
                    size = len(value) if hasattr(value, "__len__") else None
                    print(key, type(value).__name__, size, str(value)[:800])
    print("vllm_bins", glob.glob("/data04/envs/ms/*/bin/vllm"))
    print("python_bins", glob.glob("/data04/envs/ms/*/bin/python"))


if __name__ == "__main__":
    main()

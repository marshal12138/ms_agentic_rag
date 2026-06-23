#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig


MODEL = "/data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash"


def search(root: Path, needles: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    if not root.exists():
        return hits
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(needle in text for needle in needles):
            hits.append(str(path))
            if len(hits) >= 50:
                break
    return hits


def main() -> None:
    import transformers

    print("transformers", transformers.__version__, transformers.__file__)
    site = Path(transformers.__file__).parents[1]
    print("search_transformers", search(site / "transformers", ("deepseek_v4", "DeepseekV4", "DeepSeekV4")))
    try:
        config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        print("autoconfig_ok", type(config), getattr(config, "model_type", None))
    except Exception as exc:
        print("autoconfig_error", type(exc).__name__, exc)
    try:
        import vllm
        print("vllm", vllm.__version__, vllm.__file__)
        print("search_vllm", search(site / "vllm", ("deepseek_v4", "DeepseekV4", "DeepSeekV4")))
    except Exception as exc:
        print("vllm_unavailable", type(exc).__name__, exc)


if __name__ == "__main__":
    main()

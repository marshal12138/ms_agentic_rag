import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewards.search_qa_f1_with_format_penalty import search_qa_f1_penalty_compute_score


def search_qa_f1_penalty_with_trace_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    format_penalty=-0.2,
    **kwargs,
):
    result = search_qa_f1_penalty_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_penalty=format_penalty,
        **kwargs,
    )

    result["tool_call_details"] = extra_info.get("tool_call_details", [])
    result["initial_query"] = extra_info.get("initial_query", "")
    result["answers"] = extra_info.get("answers", [])
    return result

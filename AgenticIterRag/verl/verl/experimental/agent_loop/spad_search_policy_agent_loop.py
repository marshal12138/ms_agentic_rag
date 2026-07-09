"""SPAD-RAG Stage 1 search-policy agent loop."""

from __future__ import annotations

import re

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.co_search_agent_loop import SearchR1FixedRerankerAgentLoop


@register("spad_search_policy_agent")
class SpadSearchPolicyAgentLoop(SearchR1FixedRerankerAgentLoop):
    """Search-only policy loop that treats ``<answer>`` as the stop action.

    Stage 1 optimizes when the actor decides to stop, not how it writes the final
    answer. Therefore generation is stopped at the opening answer tag and the
    teacher answerer computes reward from saved search evidence.
    """

    def detect_answer(self, response_ids: list[int]) -> bool:
        text = self.tokenizer.decode(response_ids)
        return re.search(r"<answer>\s*$|<answer>", text, re.DOTALL) is not None

"""SPAD-RAG Stage 1 search-policy agent loop."""

from __future__ import annotations

import os
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.co_search_agent_loop import SearchR1FixedRerankerAgentLoop


@register("spad_search_policy_agent")
class SpadSearchPolicyAgentLoop(SearchR1FixedRerankerAgentLoop):
    """Search-only policy loop that treats ``<answer>`` as the stop action.

    Stage 1 optimizes when the actor decides to stop, not how it writes the final
    answer. Therefore generation is stopped at the opening answer tag and the
    teacher answerer computes reward from saved search evidence.
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config=config, tokenizer=tokenizer, processor=processor, **kwargs)
        response_length = int(config.actor_rollout_ref.rollout.response_length)
        raw_turn_max_tokens = os.environ.get("COSEARCH_TURN_MAX_TOKENS", "").strip()
        if raw_turn_max_tokens:
            cls.turn_max_tokens = max(1, min(int(raw_turn_max_tokens), response_length))
        else:
            cls.turn_max_tokens = response_length

    async def _handle_generating_state(
        self, agent_data, sampling_params: dict[str, Any], ignore_termination: bool = False
    ):
        turn_sampling_params = dict(sampling_params)
        remaining_tokens = max(1, self.response_length - len(agent_data.response_mask))
        turn_sampling_params["max_tokens"] = max(1, min(self.turn_max_tokens, remaining_tokens))
        return await super()._handle_generating_state(
            agent_data,
            turn_sampling_params,
            ignore_termination=ignore_termination,
        )

    def detect_answer(self, response_ids: list[int]) -> bool:
        text = self.tokenizer.decode(response_ids)
        return re.search(r"<answer>\s*$|<answer>", text, re.DOTALL) is not None

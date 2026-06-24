"""Best/worst trajectory selection for async ranker labeling."""

from __future__ import annotations

from typing import Any

from ranker_strategies.schemas import RankerTrajectory, ToolCallContext
from ranker_strategies.trajectory_selector.top_f1 import TopF1TrajectorySelector


class BestAndWorstTrajectorySelector:
    """Select top-k and bottom-n valid trajectories by final F1/reward score."""

    def __init__(self, top_k: int = 1, bottom_n: int = 2, min_final_reward: float = 0.0):
        self.top_k = max(0, int(top_k))
        self.bottom_n = max(0, int(bottom_n))
        self.min_final_reward = float(min_final_reward)
        self._top_f1_parser = TopF1TrajectorySelector(
            max_selected_trajectories=1,
            min_final_reward=self.min_final_reward,
        )

    def select(self, fresh_trajectories: list[dict[str, Any] | RankerTrajectory]) -> list[ToolCallContext]:
        parsed = [self._top_f1_parser._parse_trajectory(item) for item in fresh_trajectories]
        valid = [
            trajectory
            for trajectory in parsed
            if trajectory is not None and trajectory.is_valid and trajectory.score >= self.min_final_reward
        ]
        if not valid or (self.top_k <= 0 and self.bottom_n <= 0):
            return []

        valid.sort(key=lambda item: item.score, reverse=True)
        selected = self._select_unique_trajectories(valid)

        contexts: list[ToolCallContext] = []
        for trajectory, role, role_rank in selected:
            for call_idx, tool_call in enumerate(trajectory.tool_calls):
                context = self._top_f1_parser._parse_context(trajectory, tool_call, call_idx)
                if context is None:
                    continue
                context.metadata = {
                    **dict(context.metadata or {}),
                    "trajectory_selection_strategy": "best_and_worst_f1",
                    "trajectory_selection_role": role,
                    "trajectory_selection_rank": role_rank,
                    "trajectory_selection_score": trajectory.score,
                }
                contexts.append(context)
        return contexts

    def _select_unique_trajectories(
        self,
        sorted_valid_desc: list[RankerTrajectory],
    ) -> list[tuple[RankerTrajectory, str, int]]:
        selected: list[tuple[RankerTrajectory, str, int]] = []
        seen_ids: set[str] = set()

        for role_rank, trajectory in enumerate(sorted_valid_desc[: self.top_k], start=1):
            if trajectory.trajectory_id in seen_ids:
                continue
            selected.append((trajectory, "best", role_rank))
            seen_ids.add(trajectory.trajectory_id)

        worst_candidates = list(reversed(sorted_valid_desc))[: self.bottom_n]
        for role_rank, trajectory in enumerate(worst_candidates, start=1):
            if trajectory.trajectory_id in seen_ids:
                continue
            selected.append((trajectory, "worst", role_rank))
            seen_ids.add(trajectory.trajectory_id)

        return selected

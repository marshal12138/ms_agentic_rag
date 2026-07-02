"""Select all valid trajectories for async ranker labeling."""

from __future__ import annotations

from typing import Any

from ranker_strategies.schemas import RankerTrajectory, ToolCallContext
from ranker_strategies.trajectory_selector.top_f1 import TrajectoryContextParser


class SelectAllTrajectorySelector:
    """Select every valid trajectory without score-based filtering."""

    def __init__(self):
        self._parser = TrajectoryContextParser()

    def select(self, fresh_trajectories: list[dict[str, Any] | RankerTrajectory]) -> list[ToolCallContext]:
        contexts: list[ToolCallContext] = []
        selected_rank = 0

        for item in fresh_trajectories:
            trajectory = self._parser._parse_trajectory(item)
            if trajectory is None or not trajectory.is_valid:
                continue

            selected_rank += 1
            for call_idx, tool_call in enumerate(trajectory.tool_calls):
                context = self._parser._parse_context(trajectory, tool_call, call_idx)
                if context is None:
                    continue
                context.metadata = {
                    **dict(context.metadata or {}),
                    "trajectory_selection_strategy": "select_all",
                    "trajectory_selection_role": "all",
                    "trajectory_selection_rank": selected_rank,
                    "trajectory_selection_score": trajectory.score,
                }
                contexts.append(context)

        return contexts

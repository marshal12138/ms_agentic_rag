"""Streaming UID-group reward scheduling tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import _chunk_complete_uid_groups, _object_vector


def make_rollout_batch(group_count: int, group_size: int) -> DataProto:
    size = group_count * group_size
    return DataProto(
        batch=TensorDict(
            {"input_ids": torch.arange(size, dtype=torch.long).reshape(size, 1)},
            batch_size=size,
        ),
        non_tensor_batch={
            "uid": np.asarray(
                [f"uid-{group_index}" for group_index in range(group_count) for _ in range(group_size)],
                dtype=object,
            )
        },
    )


class StreamGroupRewardTest(unittest.TestCase):
    def test_nested_reward_extras_are_always_one_dimensional(self) -> None:
        values = [[{"role": "system"}, {"role": "user"}] for _ in range(8)]
        array = _object_vector(values)
        self.assertEqual(array.shape, (8,))
        self.assertEqual(array[0][1]["role"], "user")

    def test_worker_chunks_keep_complete_uid_groups_and_order(self) -> None:
        chunks = _chunk_complete_uid_groups(
            make_rollout_batch(group_count=10, group_size=8),
            num_workers=3,
            expected_group_size=8,
        )

        self.assertEqual([len(chunk) for chunk in chunks], [32, 24, 24])
        flattened_uids = []
        for chunk in chunks:
            uids = chunk.non_tensor_batch["uid"].tolist()
            flattened_uids.extend(uids)
            for uid in set(uids):
                self.assertEqual(uids.count(uid), 8)
        self.assertEqual(
            flattened_uids,
            make_rollout_batch(group_count=10, group_size=8).non_tensor_batch["uid"].tolist(),
        )

    def test_incomplete_uid_group_is_rejected(self) -> None:
        batch = make_rollout_batch(group_count=2, group_size=8)
        batch = batch.select_idxs(list(range(15)))
        with self.assertRaisesRegex(ValueError, "exactly 8 rollouts per uid"):
            _chunk_complete_uid_groups(batch, num_workers=1, expected_group_size=8)


if __name__ == "__main__":
    unittest.main()

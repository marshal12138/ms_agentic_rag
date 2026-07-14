"""Batch reward manager rollout-extra propagation tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import _as_object_vector
from verl.workers.reward_manager.batch import BatchRewardManager


class TokenizerStub:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(value)) for value in token_ids)


class BatchRewardManagerTest(unittest.TestCase):
    def test_nested_reward_extras_remain_one_object_per_rollout(self) -> None:
        values = [
            [{"role": "system"}, {"role": "user", "content": str(index)}]
            for index in range(8)
        ]
        array = _as_object_vector(values)
        self.assertEqual(array.shape, (8,))
        self.assertEqual(array.dtype, np.dtype(object))
        self.assertEqual(array[3][1]["content"], "3")

    def test_flattened_agent_loop_fields_reach_batch_reward(self) -> None:
        batch_size = 8
        captured = {}

        def compute_score(**kwargs):
            captured.update(kwargs)
            return [0.0] * batch_size

        manager = BatchRewardManager(
            tokenizer=TokenizerStub(),
            num_examine=0,
            compute_score=compute_score,
            n_samples_per_prompt=8,
        )
        tensor_batch = TensorDict(
            {
                "prompts": torch.ones((batch_size, 2), dtype=torch.long),
                "responses": torch.ones((batch_size, 3), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
            },
            batch_size=batch_size,
        )
        tool_details = np.empty(batch_size, dtype=object)
        tool_details[:] = [
            [{"sub_query": f"query-{index}", "top_5_documents": [{"text": "evidence"}]}]
            for index in range(batch_size)
        ]
        non_tensor_batch = {
            "reward_model": np.asarray(
                [{"ground_truth": {"target": ["answer"]}} for _ in range(batch_size)],
                dtype=object,
            ),
            "data_source": np.asarray(["unit-test"] * batch_size, dtype=object),
            "extra_info": np.asarray([{"question": "q"} for _ in range(batch_size)], dtype=object),
            "uid": np.asarray(["same-uid"] * batch_size, dtype=object),
            "__num_turns__": np.asarray([3] * batch_size, dtype=np.int32),
            "tool_call_details": tool_details,
        }
        manager.verify(DataProto(batch=tensor_batch, non_tensor_batch=non_tensor_batch))

        extras = captured["extra_infos"]
        self.assertEqual(len(extras), batch_size)
        self.assertEqual(extras[0]["uid"], "same-uid")
        self.assertEqual(extras[0]["num_turns"], 3)
        self.assertEqual(extras[0]["tool_call_details"][0]["sub_query"], "query-0")


if __name__ == "__main__":
    unittest.main()

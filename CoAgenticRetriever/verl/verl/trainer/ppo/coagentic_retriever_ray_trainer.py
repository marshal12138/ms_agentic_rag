"""CoAgenticRetriever PPO helpers.

The trainable dense ranker is not an LLM agent. This module keeps the main
agent PPO step separate from older dual-agent training code.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import ray
import torch

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


def _to_float_array(values) -> np.ndarray:
    converted = []
    for value in np.array(values, dtype=object).reshape(-1):
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y"}:
                value = 1.0
            elif lowered in {"false", "no", "n"}:
                value = 0.0
        try:
            converted.append(float(value))
        except (TypeError, ValueError):
            converted.append(0.0)
    return np.array(converted, dtype=float)


def _to_python_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@ray.remote
def process_main_agent_ppo_step(
    batch: DataProto,
    actor_rollout_wg: RayWorkerGroup,
    ref_policy_wg: Optional[RayWorkerGroup],
    critic_wg: Optional[RayWorkerGroup],
    tokenizer,
    config,
    global_steps: int,
    use_reference_policy: bool,
    use_critic: bool,
    kl_ctrl_in_reward,
    ref_in_actor: bool,
) -> tuple[DataProto, dict, dict, dict]:
    metrics: dict[str, Any] = {}
    timing_raw: dict[str, float] = {}
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    with marked_timer("main_agent_reward", timing_raw, color="yellow"):
        assert "rm_scores" in batch.batch.keys(), "rm_scores should be obtained in rollout phase"
        assert not config.reward_model.launch_reward_fn_async, "Reward loop async mode is not supported here."
        reward_tensor = batch.batch["rm_scores"]
        reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
        reward_extra_infos_dict = (
            {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
        )

    rollout_corr_config = config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
    if bypass_recomputing_logprobs:
        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

        apply_rollout_correction(
            batch=batch,
            rollout_corr_config=rollout_corr_config,
            policy_loss_config=config.actor_rollout_ref.actor.policy_loss,
        )
    else:
        with marked_timer("main_agent_old_log_prob", timing_raw, color="blue"):
            old_log_prob = actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["main_agent_actor/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)
            if "rollout_log_probs" in batch.batch.keys():
                from verl.utils.debug.metrics import calculate_debug_metrics

                metrics.update(calculate_debug_metrics(batch))

    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

    if use_reference_policy:
        with marked_timer("main_agent_ref_log_prob", timing_raw, color="olive"):
            if not ref_in_actor:
                ref_log_prob = ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

    if use_critic:
        with marked_timer("main_agent_values", timing_raw, color="cyan"):
            values = critic_wg.compute_values(batch)
            batch = batch.union(values)

    with marked_timer("main_agent_adv", timing_raw, color="brown"):
        batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra_infos_dict.items()})

        if config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=kl_ctrl_in_reward,
                kl_penalty=config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        if (
            rollout_corr_config is not None
            and "rollout_log_probs" in batch.batch
            and not bypass_recomputing_logprobs
        ):
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
            metrics.update(is_metrics)

        batch = compute_advantage(
            batch,
            adv_estimator=config.algorithm.adv_estimator,
            gamma=config.algorithm.gamma,
            lam=config.algorithm.lam,
            num_repeat=config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=config.algorithm,
        )

    if use_critic:
        with marked_timer("main_agent_update_critic", timing_raw, color="pink"):
            critic_output = critic_wg.update_critic(batch)
        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

    if config.trainer.critic_warmup <= global_steps:
        with marked_timer("main_agent_update_actor", timing_raw, color="red"):
            rollout_config = config.actor_rollout_ref.rollout
            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
            batch.meta_info["temperature"] = rollout_config.temperature
            actor_output = actor_rollout_wg.update_actor(batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        for key, value in actor_output_metrics.items():
            if key.startswith("actor/"):
                metrics[key.replace("actor/", "main_agent_actor/", 1)] = value
            else:
                metrics[f"main_agent_{key}"] = value

    return batch, metrics, timing_raw, reward_extra_infos_dict


class CoAgenticRetrieverRayTrainer(RayPPOTrainer):
    """Ray PPO trainer base for CoAgenticRetriever main-agent training."""

    _ROLL_OUT_TRACE_MODES = {"partial", "full"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "kl_ctrl_in_reward"):
            self.kl_ctrl_in_reward = None

    def _rollout_trace_mode(self) -> str:
        mode = str(self.config.trainer.get("rollout_trace_mode", "full")).strip().lower()
        if mode not in self._ROLL_OUT_TRACE_MODES:
            raise ValueError(
                f"trainer.rollout_trace_mode must be one of {sorted(self._ROLL_OUT_TRACE_MODES)}, got {mode!r}"
            )
        return mode

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        trace_mode = self._rollout_trace_mode()
        reward_extra_infos_to_dump = dict(reward_extra_infos_dict or {})
        batch_size = len(batch)
        reward_extra_infos_to_dump["rollout_trace_mode"] = [trace_mode] * batch_size

        if "request_id" in batch.non_tensor_batch:
            request_ids = _to_python_list(batch.non_tensor_batch["request_id"])
            if len(request_ids) == batch_size:
                reward_extra_infos_to_dump["request_id"] = request_ids

        if trace_mode == "full":
            for key in ("tool_call_details", "messages", "initial_query", "answers"):
                if key not in batch.non_tensor_batch:
                    continue
                values = _to_python_list(batch.non_tensor_batch[key])
                if len(values) == batch_size:
                    reward_extra_infos_to_dump[key] = values

        super()._log_rollout_data(
            batch=batch,
            reward_extra_infos_dict=reward_extra_infos_to_dump,
            timing_raw=timing_raw,
            rollout_data_dir=rollout_data_dir,
        )

    @staticmethod
    def _compute_rollout_stats(batch: DataProto, reward_extra_infos_dict: dict, prefix: str = "main_agent") -> dict:
        stats: dict[str, Any] = {}

        if "valid" in reward_extra_infos_dict:
            valid_array = _to_float_array(reward_extra_infos_dict["valid"])
            total_count = len(valid_array)
            stats[f"{prefix}/valid_rate"] = float(np.sum(valid_array == 1) / total_count) if total_count else 0.0

        min_one_search_array = None
        if "min_one_search" in reward_extra_infos_dict:
            min_one_search_array = _to_float_array(reward_extra_infos_dict["min_one_search"])
            total_count = len(min_one_search_array)
            stats[f"{prefix}/min_one_search_rate"] = (
                float(np.sum(min_one_search_array == 1) / total_count) if total_count else 0.0
            )

        if "tool_call_count" in reward_extra_infos_dict:
            tool_call_count_array = _to_float_array(reward_extra_infos_dict["tool_call_count"])
            total_count = len(tool_call_count_array)
            stats[f"{prefix}/tool_call_rate"] = (
                float(np.sum(tool_call_count_array > 0) / total_count) if total_count else 0.0
            )
            stats[f"{prefix}/tool_call_count_mean"] = (
                float(np.mean(tool_call_count_array)) if total_count else 0.0
            )
        elif "has_search_tool_call" in reward_extra_infos_dict:
            has_search_array = _to_float_array(reward_extra_infos_dict["has_search_tool_call"])
            total_count = len(has_search_array)
            stats[f"{prefix}/tool_call_rate"] = (
                float(np.sum(has_search_array == 1) / total_count) if total_count else 0.0
            )

        if min_one_search_array is not None and "valid" in reward_extra_infos_dict:
            valid_array = _to_float_array(reward_extra_infos_dict["valid"])
            total_count = min(len(min_one_search_array), len(valid_array))
            if total_count:
                stats[f"{prefix}/min_one_search_valid_rate"] = float(
                    np.sum((min_one_search_array[:total_count] == 1) & (valid_array[:total_count] == 1)) / total_count
                )
            else:
                stats[f"{prefix}/min_one_search_valid_rate"] = 0.0

        if "invalid_direct_answer_before_search" in reward_extra_infos_dict:
            invalid_direct_array = _to_float_array(reward_extra_infos_dict["invalid_direct_answer_before_search"])
            total_count = len(invalid_direct_array)
            stats[f"{prefix}/direct_answer_before_search_rate"] = (
                float(np.sum(invalid_direct_array == 1) / total_count) if total_count else 0.0
            )

        if "token_level_scores" in batch.batch:
            scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
            stats[f"{prefix}/score_mean"] = float(np.mean(scores))

        if "f1" in reward_extra_infos_dict:
            stats[f"{prefix}/f1_mean"] = float(np.mean(np.array(reward_extra_infos_dict["f1"])))

        if "agent_reward" in reward_extra_infos_dict:
            stats[f"{prefix}/agent_reward_mean"] = float(np.mean(np.array(reward_extra_infos_dict["agent_reward"])))

        if "tool_reward" in reward_extra_infos_dict:
            stats[f"{prefix}/tool_reward_mean"] = float(np.mean(np.array(reward_extra_infos_dict["tool_reward"])))

        return stats
